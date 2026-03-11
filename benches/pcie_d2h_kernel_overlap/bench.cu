#include "bench_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::array<size_t, 3> kDefaultSizesMb = {128, 512, 1024};
constexpr int kThreadsPerBlock = 256;
constexpr int kCalibrationPassLimit = 8;

struct KernelStats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    unsigned long long loop_count = 0;
};

struct OverlapStats {
    bool success = false;
    std::string error;
    double avg_wall_ms = 0.0;
    double copy_stream_avg_ms = 0.0;
    double kernel_stream_avg_ms = 0.0;
    double wall_vs_solo_sum_ratio = 0.0;
    double wall_vs_solo_max_ratio = 0.0;
    double copy_gib_per_s = 0.0;
};

struct DirectionRow {
    bench::CopyStats copy_solo;
    KernelStats kernel_solo;
    OverlapStats overlap;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    DirectionRow h2d;
    DirectionRow d2h;
};

__global__ void low_memory_spin_kernel(float* output, unsigned long long loops) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t x = 0x12345678u ^ static_cast<std::uint32_t>(tid * 747796405u + 2891336453u);
    std::uint32_t y = 0x9E3779B9u + static_cast<std::uint32_t>(tid * 1664525u + 1013904223u);
    std::uint32_t z = 0xA5A5A5A5u ^ static_cast<std::uint32_t>(tid * 2246822519u + 3266489917u);
    for (unsigned long long i = 0; i < loops; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        y = y * 1664525u + 1013904223u;
        z ^= x + 0x9E3779B9u + (z << 6) + (z >> 2);
    }
    output[tid] = static_cast<float>((x ^ y ^ z) & 0x00FFFFFFu);
}

bool is_default_size_list(const std::vector<size_t>& sizes_mb) {
    const std::vector<size_t> shared_default_sizes = {8, 32, 128, 512, 1024};
    return sizes_mb == shared_default_sizes;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb <= 128) {
        return std::min(requested_iterations, 20);
    }
    if (size_mb <= 512) {
        return std::min(requested_iterations, 10);
    }
    return std::min(requested_iterations, 5);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb <= 512) {
        return std::min(requested_warmup, 3);
    }
    return std::min(requested_warmup, 2);
}

bool buffer_has_byte_value(const void* data, size_t bytes, unsigned char expected) {
    const auto* ptr = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        if (ptr[i] != expected) {
            return false;
        }
    }
    return true;
}

bench::CopyStats measure_async_copy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    bench::CopyStats stats;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cudaError_t status = cudaEventCreate(&start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(start) failed: ") + cudaGetErrorString(status);
        return stats;
    }
    status = cudaEventCreate(&stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(stop) failed: ") + cudaGetErrorString(status);
        cudaEventDestroy(start);
        return stats;
    }

    for (int i = 0; i < warmup; ++i) {
        status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaMemcpyAsync failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
    }

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        bench::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(copy start)");
        bench::check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
        bench::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(copy stop)");
        bench::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(copy stop)");

        float elapsed_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(copy)");
        total_ms += elapsed_ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    return stats;
}

KernelStats measure_kernel(
    float* output,
    int total_threads,
    unsigned long long loops,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    KernelStats stats;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cudaError_t status = cudaEventCreate(&start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(kernel start) failed: ") + cudaGetErrorString(status);
        return stats;
    }
    status = cudaEventCreate(&stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(kernel stop) failed: ") + cudaGetErrorString(status);
        cudaEventDestroy(start);
        return stats;
    }

    const int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int i = 0; i < warmup; ++i) {
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(output, loops);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup kernel launch failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
    }

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        bench::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(kernel start)");
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(output, loops);
        bench::check_cuda(cudaGetLastError(), "kernel launch");
        bench::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(kernel stop)");
        bench::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(kernel stop)");

        float elapsed_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(kernel)");
        total_ms += elapsed_ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.loop_count = loops;
    return stats;
}

KernelStats calibrate_kernel_to_target(
    float* output,
    int total_threads,
    double target_ms,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    unsigned long long loops = 1ull << 16;
    KernelStats stats;

    for (int pass = 0; pass < kCalibrationPassLimit; ++pass) {
        stats = measure_kernel(output, total_threads, loops, stream, warmup, iterations);
        if (!stats.success) {
            return stats;
        }
        if (stats.avg_ms > 0.0) {
            const double ratio = target_ms / stats.avg_ms;
            if (ratio >= 0.85 && ratio <= 1.15) {
                return stats;
            }

            double scaled = static_cast<double>(loops) * std::clamp(ratio, 0.25, 4.0);
            if (scaled < 1.0) {
                scaled = 1.0;
            }
            unsigned long long next_loops = static_cast<unsigned long long>(scaled);
            if (next_loops == loops) {
                next_loops += 1;
            }
            loops = next_loops;
        } else {
            loops *= 4;
        }
    }

    return stats;
}

OverlapStats measure_overlap_pair(
    void* copy_dst,
    const void* copy_src,
    size_t copy_bytes,
    cudaMemcpyKind kind,
    float* kernel_output,
    int total_threads,
    unsigned long long loop_count,
    cudaStream_t copy_stream,
    cudaStream_t kernel_stream,
    int warmup,
    int iterations,
    const bench::CopyStats& copy_solo,
    const KernelStats& kernel_solo) {
    OverlapStats stats;
    cudaEvent_t copy_start = nullptr;
    cudaEvent_t copy_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    auto destroy_events = [&]() {
        if (copy_start != nullptr) cudaEventDestroy(copy_start);
        if (copy_stop != nullptr) cudaEventDestroy(copy_stop);
        if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
        if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    };

    cudaError_t status = cudaEventCreate(&copy_start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(copy_start) failed: ") + cudaGetErrorString(status);
        return stats;
    }
    status = cudaEventCreate(&copy_stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(copy_stop) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }
    status = cudaEventCreate(&kernel_start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(kernel_start) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }
    status = cudaEventCreate(&kernel_stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(kernel_stop) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }

    const int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int i = 0; i < warmup; ++i) {
        bench::check_cuda(cudaMemcpyAsync(copy_dst, copy_src, copy_bytes, kind, copy_stream), "warmup copy");
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(kernel_output, loop_count);
        bench::check_cuda(cudaGetLastError(), "warmup kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "warmup cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "warmup cudaStreamSynchronize(kernel)");
    }

    double total_wall_ms = 0.0;
    float total_copy_stream_ms = 0.0f;
    float total_kernel_stream_ms = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        const auto wall_start = std::chrono::steady_clock::now();

        bench::check_cuda(cudaEventRecord(copy_start, copy_stream), "cudaEventRecord(copy_start)");
        bench::check_cuda(cudaMemcpyAsync(copy_dst, copy_src, copy_bytes, kind, copy_stream), "pair copy");
        bench::check_cuda(cudaEventRecord(copy_stop, copy_stream), "cudaEventRecord(copy_stop)");

        bench::check_cuda(cudaEventRecord(kernel_start, kernel_stream), "cudaEventRecord(kernel_start)");
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(kernel_output, loop_count);
        bench::check_cuda(cudaGetLastError(), "pair kernel launch");
        bench::check_cuda(cudaEventRecord(kernel_stop, kernel_stream), "cudaEventRecord(kernel_stop)");

        bench::check_cuda(cudaEventSynchronize(copy_stop), "cudaEventSynchronize(copy_stop)");
        bench::check_cuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize(kernel_stop)");
        const auto wall_end = std::chrono::steady_clock::now();

        float copy_stream_ms = 0.0f;
        float kernel_stream_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&copy_stream_ms, copy_start, copy_stop), "cudaEventElapsedTime(copy)");
        bench::check_cuda(cudaEventElapsedTime(&kernel_stream_ms, kernel_start, kernel_stop), "cudaEventElapsedTime(kernel)");

        total_copy_stream_ms += copy_stream_ms;
        total_kernel_stream_ms += kernel_stream_ms;
        total_wall_ms += std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    }

    destroy_events();

    stats.success = true;
    stats.avg_wall_ms = total_wall_ms / static_cast<double>(iterations);
    stats.copy_stream_avg_ms = total_copy_stream_ms / static_cast<double>(iterations);
    stats.kernel_stream_avg_ms = total_kernel_stream_ms / static_cast<double>(iterations);
    stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (copy_solo.avg_ms + kernel_solo.avg_ms);
    stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(copy_solo.avg_ms, kernel_solo.avg_ms);
    const double gib = static_cast<double>(copy_bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.copy_gib_per_s = gib / (stats.avg_wall_ms / 1000.0);
    return stats;
}

std::string render_copy_stats_json(const bench::CopyStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << bench::format_double(stats.avg_ms) << ","
        << "\"gib_per_s\":" << bench::format_double(stats.gib_per_s);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_kernel_stats_json(const KernelStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << bench::format_double(stats.avg_ms) << ","
        << "\"loop_count\":" << stats.loop_count;
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_overlap_stats_json(const OverlapStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_wall_ms\":" << bench::format_double(stats.avg_wall_ms) << ","
        << "\"copy_stream_avg_ms\":" << bench::format_double(stats.copy_stream_avg_ms) << ","
        << "\"kernel_stream_avg_ms\":" << bench::format_double(stats.kernel_stream_avg_ms) << ","
        << "\"copy_gib_per_s\":" << bench::format_double(stats.copy_gib_per_s) << ","
        << "\"wall_vs_solo_sum_ratio\":" << bench::format_double(stats.wall_vs_solo_sum_ratio) << ","
        << "\"wall_vs_solo_max_ratio\":" << bench::format_double(stats.wall_vs_solo_max_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_direction_json(const DirectionRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"copy_solo\":" << render_copy_stats_json(row.copy_solo) << ","
        << "\"kernel_solo\":" << render_kernel_stats_json(row.kernel_solo) << ","
        << "\"overlap\":" << render_overlap_stats_json(row.overlap)
        << "}";
    return oss.str();
}

std::string render_json(
    const bench::Options& options,
    int async_engine_count,
    int device_overlap,
    const std::vector<CaseRow>& rows,
    bool validation_passed) {
    double best_h2d_overlap_copy_gib_per_s = 0.0;
    double best_d2h_overlap_copy_gib_per_s = 0.0;
    double min_h2d_wall_vs_sum = 0.0;
    double min_d2h_wall_vs_sum = 0.0;
    double min_h2d_wall_vs_max = 0.0;
    double min_d2h_wall_vs_max = 0.0;
    bool first_h2d = true;
    bool first_d2h = true;

    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0) {
            cases << ",";
        }
        cases << "{"
              << "\"size_mb\":" << row.size_mb << ","
              << "\"iterations\":" << row.iterations << ","
              << "\"warmup\":" << row.warmup << ","
              << "\"h2d\":" << render_direction_json(row.h2d) << ","
              << "\"d2h\":" << render_direction_json(row.d2h)
              << "}";

        if (row.h2d.copy_solo.success && row.h2d.kernel_solo.success && row.h2d.overlap.success) {
            best_h2d_overlap_copy_gib_per_s = std::max(best_h2d_overlap_copy_gib_per_s, row.h2d.overlap.copy_gib_per_s);
            if (first_h2d) {
                min_h2d_wall_vs_sum = row.h2d.overlap.wall_vs_solo_sum_ratio;
                min_h2d_wall_vs_max = row.h2d.overlap.wall_vs_solo_max_ratio;
                first_h2d = false;
            } else {
                min_h2d_wall_vs_sum = std::min(min_h2d_wall_vs_sum, row.h2d.overlap.wall_vs_solo_sum_ratio);
                min_h2d_wall_vs_max = std::min(min_h2d_wall_vs_max, row.h2d.overlap.wall_vs_solo_max_ratio);
            }
        }

        if (row.d2h.copy_solo.success && row.d2h.kernel_solo.success && row.d2h.overlap.success) {
            best_d2h_overlap_copy_gib_per_s = std::max(best_d2h_overlap_copy_gib_per_s, row.d2h.overlap.copy_gib_per_s);
            if (first_d2h) {
                min_d2h_wall_vs_sum = row.d2h.overlap.wall_vs_solo_sum_ratio;
                min_d2h_wall_vs_max = row.d2h.overlap.wall_vs_solo_max_ratio;
                first_d2h = false;
            } else {
                min_d2h_wall_vs_sum = std::min(min_d2h_wall_vs_sum, row.d2h.overlap.wall_vs_solo_sum_ratio);
                min_d2h_wall_vs_max = std::min(min_d2h_wall_vs_max, row.d2h.overlap.wall_vs_solo_max_ratio);
            }
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.h2d.copy_solo.success && row.h2d.kernel_solo.success && row.h2d.overlap.success &&
               row.d2h.copy_solo.success && row.d2h.kernel_solo.success && row.d2h.overlap.success;
    });
    const std::string status = (all_ok && validation_passed) ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"min_h2d_wall_vs_solo_sum_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"copy_directions\":[\"H2D\",\"D2H\"],"
        << "\"memory_types\":[\"pinned\"],"
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event_and_wall_clock\","
        << "\"async_engine_count\":" << async_engine_count << ","
        << "\"device_overlap\":" << device_overlap << ","
        << "\"best_h2d_overlap_copy_gib_per_s\":" << bench::format_double(best_h2d_overlap_copy_gib_per_s) << ","
        << "\"best_d2h_overlap_copy_gib_per_s\":" << bench::format_double(best_d2h_overlap_copy_gib_per_s) << ","
        << "\"min_h2d_wall_vs_solo_sum_ratio\":" << bench::format_double(min_h2d_wall_vs_sum) << ","
        << "\"min_d2h_wall_vs_solo_sum_ratio\":" << bench::format_double(min_d2h_wall_vs_sum) << ","
        << "\"min_h2d_wall_vs_solo_max_ratio\":" << bench::format_double(min_h2d_wall_vs_max) << ","
        << "\"min_d2h_wall_vs_solo_max_ratio\":" << bench::format_double(min_d2h_wall_vs_max) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "}"
        << "}";
    return oss.str();
}

bool verify_direction_result(
    cudaMemcpyKind kind,
    const void* host_reference,
    void* host_buffer,
    void* device_buffer,
    void* verify_buffer,
    size_t bytes) {
    if (kind == cudaMemcpyDeviceToHost) {
        return buffer_has_byte_value(host_buffer, bytes, 0x5A);
    }
    bench::check_cuda(cudaMemcpy(verify_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost), "verify H2D overlap");
    return std::memcmp(verify_buffer, host_reference, bytes) == 0;
}

DirectionRow run_direction_case(
    cudaMemcpyKind kind,
    void* copy_dst,
    const void* copy_src,
    const void* host_reference,
    void* host_buffer,
    void* device_buffer,
    void* verify_buffer,
    size_t copy_bytes,
    float* kernel_output,
    int total_threads,
    cudaStream_t copy_stream,
    cudaStream_t kernel_stream,
    int warmup,
    int iterations,
    bool* validation_passed) {
    DirectionRow row;
    row.copy_solo = measure_async_copy(copy_dst, copy_src, copy_bytes, kind, copy_stream, warmup, iterations);
    if (!row.copy_solo.success) {
        *validation_passed = false;
        return row;
    }

    if (!verify_direction_result(kind, host_reference, host_buffer, device_buffer, verify_buffer, copy_bytes)) {
        *validation_passed = false;
    }

    row.kernel_solo = calibrate_kernel_to_target(kernel_output, total_threads, row.copy_solo.avg_ms, kernel_stream, warmup, iterations);
    if (!row.kernel_solo.success) {
        *validation_passed = false;
        return row;
    }

    std::vector<float> host_kernel_output(static_cast<size_t>(total_threads), 0.0f);
    bench::check_cuda(
        cudaMemcpy(
            host_kernel_output.data(),
            kernel_output,
            static_cast<size_t>(total_threads) * sizeof(float),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(kernel output verify)");
    bool any_non_zero = false;
    for (float value : host_kernel_output) {
        if (std::isfinite(value) && value != 0.0f) {
            any_non_zero = true;
            break;
        }
    }
    if (!any_non_zero) {
        *validation_passed = false;
    }

    bench::check_cuda(
        cudaMemset(kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
        "cudaMemset(kernel_output reset)");
    if (kind == cudaMemcpyDeviceToHost) {
        std::memset(host_buffer, 0x00, copy_bytes);
    } else {
        bench::check_cuda(cudaMemset(device_buffer, 0x00, copy_bytes), "cudaMemset(h2d destination reset)");
    }

    row.overlap = measure_overlap_pair(
        copy_dst,
        copy_src,
        copy_bytes,
        kind,
        kernel_output,
        total_threads,
        row.kernel_solo.loop_count,
        copy_stream,
        kernel_stream,
        warmup,
        iterations,
        row.copy_solo,
        row.kernel_solo);
    if (!row.overlap.success) {
        *validation_passed = false;
        return row;
    }

    if (!verify_direction_result(kind, host_reference, host_buffer, device_buffer, verify_buffer, copy_bytes)) {
        *validation_passed = false;
    }

    return row;
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options{};
    try {
        options = bench::parse_common_args(argc, argv);
        if (is_default_size_list(options.sizes_mb)) {
            options.sizes_mb.assign(kDefaultSizesMb.begin(), kDefaultSizesMb.end());
        }

        int device_count = 0;
        bench::check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options, "min_h2d_wall_vs_solo_sum_ratio"));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp props{};
        bench::check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");

        cudaStream_t copy_stream = nullptr;
        cudaStream_t kernel_stream = nullptr;
        bench::check_cuda(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(copy)");
        bench::check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(kernel)");

        const int total_threads = props.multiProcessorCount * 256;

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t copy_bytes = size_mb * 1024ull * 1024ull;
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            void* device_h2d_dst = nullptr;
            void* device_d2h_src = nullptr;
            void* host_h2d_src = nullptr;
            void* host_d2h_dst = nullptr;
            void* verify_buffer = nullptr;
            float* device_kernel_output = nullptr;

            auto fail_case = [&](const std::string& message) {
                row.h2d.copy_solo.error = message;
                row.h2d.kernel_solo.error = message;
                row.h2d.overlap.error = message;
                row.d2h.copy_solo.error = message;
                row.d2h.kernel_solo.error = message;
                row.d2h.overlap.error = message;
                validation_passed = false;
            };

            const cudaError_t h2d_dst_status = cudaMalloc(&device_h2d_dst, copy_bytes);
            if (h2d_dst_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_h2d_dst) failed: ") + cudaGetErrorString(h2d_dst_status));
                rows.push_back(row);
                continue;
            }

            const cudaError_t d2h_src_status = cudaMalloc(&device_d2h_src, copy_bytes);
            if (d2h_src_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_d2h_src) failed: ") + cudaGetErrorString(d2h_src_status));
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t host_h2d_status = cudaMallocHost(&host_h2d_src, copy_bytes);
            if (host_h2d_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(host_h2d_src) failed: ") + cudaGetErrorString(host_h2d_status));
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t host_d2h_status = cudaMallocHost(&host_d2h_dst, copy_bytes);
            if (host_d2h_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(host_d2h_dst) failed: ") + cudaGetErrorString(host_d2h_status));
                cudaFreeHost(host_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            verify_buffer = std::malloc(copy_bytes);
            if (verify_buffer == nullptr) {
                fail_case("verify buffer allocation failed");
                cudaFreeHost(host_d2h_dst);
                cudaFreeHost(host_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t kernel_output_status =
                cudaMalloc(&device_kernel_output, static_cast<size_t>(total_threads) * sizeof(float));
            if (kernel_output_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_kernel_output) failed: ") + cudaGetErrorString(kernel_output_status));
                std::free(verify_buffer);
                cudaFreeHost(host_d2h_dst);
                cudaFreeHost(host_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            std::memset(host_h2d_src, 0x3C, copy_bytes);
            std::memset(host_d2h_dst, 0x00, copy_bytes);
            bench::check_cuda(cudaMemset(device_h2d_dst, 0x00, copy_bytes), "cudaMemset(device_h2d_dst)");
            bench::check_cuda(cudaMemset(device_d2h_src, 0x5A, copy_bytes), "cudaMemset(device_d2h_src)");
            bench::check_cuda(
                cudaMemset(device_kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
                "cudaMemset(device_kernel_output)");

            row.h2d = run_direction_case(
                cudaMemcpyHostToDevice,
                device_h2d_dst,
                host_h2d_src,
                host_h2d_src,
                host_h2d_src,
                device_h2d_dst,
                verify_buffer,
                copy_bytes,
                device_kernel_output,
                total_threads,
                copy_stream,
                kernel_stream,
                row.warmup,
                row.iterations,
                &validation_passed);

            bench::check_cuda(
                cudaMemset(device_kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
                "cudaMemset(device_kernel_output before d2h)");

            row.d2h = run_direction_case(
                cudaMemcpyDeviceToHost,
                host_d2h_dst,
                device_d2h_src,
                nullptr,
                host_d2h_dst,
                device_d2h_src,
                verify_buffer,
                copy_bytes,
                device_kernel_output,
                total_threads,
                copy_stream,
                kernel_stream,
                row.warmup,
                row.iterations,
                &validation_passed);

            cudaFree(device_kernel_output);
            std::free(verify_buffer);
            cudaFreeHost(host_d2h_dst);
            cudaFreeHost(host_h2d_src);
            cudaFree(device_d2h_src);
            cudaFree(device_h2d_dst);
            rows.push_back(row);
        }

        cudaStreamDestroy(kernel_stream);
        cudaStreamDestroy(copy_stream);

        bench::emit_json(render_json(options, props.asyncEngineCount, props.deviceOverlap, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "min_h2d_wall_vs_solo_sum_ratio"));
        return 1;
    }
}
