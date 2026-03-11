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

struct WallStats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
};

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
    double wall_vs_solo_sum_ratio = 0.0;
    double wall_vs_solo_max_ratio = 0.0;
    double copy_gib_per_s = 0.0;
};

struct DirectionRow {
    WallStats copy_solo;
    KernelStats kernel_solo;
    OverlapStats overlap;
};

struct MemoryRow {
    DirectionRow h2d;
    DirectionRow d2h;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    MemoryRow pinned;
    MemoryRow pageable;
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

WallStats measure_copy_wall(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    WallStats stats;

    for (int i = 0; i < warmup; ++i) {
        const cudaError_t status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaMemcpyAsync failed: ") + cudaGetErrorString(status);
            return stats;
        }
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(sync_status);
            return stats;
        }
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        bench::check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
        bench::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(copy)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    return stats;
}

KernelStats measure_kernel_wall(
    float* output,
    int total_threads,
    unsigned long long loops,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    KernelStats stats;
    const int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int i = 0; i < warmup; ++i) {
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(output, loops);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            stats.error = std::string("Warmup kernel launch failed: ") + cudaGetErrorString(launch_status);
            return stats;
        }
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(sync_status);
            return stats;
        }
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(output, loops);
        bench::check_cuda(cudaGetLastError(), "kernel launch");
        bench::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(kernel)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

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
        stats = measure_kernel_wall(output, total_threads, loops, stream, warmup, iterations);
        if (!stats.success) {
            return stats;
        }
        if (stats.avg_ms <= 0.0) {
            loops *= 4;
            continue;
        }

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
    }

    return stats;
}

OverlapStats measure_overlap_wall(
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
    const WallStats& copy_solo,
    const KernelStats& kernel_solo) {
    OverlapStats stats;
    const int blocks = (total_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;

    for (int i = 0; i < warmup; ++i) {
        bench::check_cuda(cudaMemcpyAsync(copy_dst, copy_src, copy_bytes, kind, copy_stream), "warmup copy");
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(kernel_output, loop_count);
        bench::check_cuda(cudaGetLastError(), "warmup kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "warmup cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "warmup cudaStreamSynchronize(kernel)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        bench::check_cuda(cudaMemcpyAsync(copy_dst, copy_src, copy_bytes, kind, copy_stream), "pair copy");
        low_memory_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(kernel_output, loop_count);
        bench::check_cuda(cudaGetLastError(), "pair kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "cudaStreamSynchronize(kernel)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_wall_ms = total_ms / static_cast<double>(iterations);
    stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (copy_solo.avg_ms + kernel_solo.avg_ms);
    stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(copy_solo.avg_ms, kernel_solo.avg_ms);
    const double gib = static_cast<double>(copy_bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.copy_gib_per_s = gib / (stats.avg_wall_ms / 1000.0);
    return stats;
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
    row.copy_solo = measure_copy_wall(copy_dst, copy_src, copy_bytes, kind, copy_stream, warmup, iterations);
    if (!row.copy_solo.success) {
        *validation_passed = false;
        return row;
    }

    if (!verify_direction_result(kind, host_reference, host_buffer, device_buffer, verify_buffer, copy_bytes)) {
        *validation_passed = false;
    }

    row.kernel_solo = calibrate_kernel_to_target(
        kernel_output,
        total_threads,
        row.copy_solo.avg_ms,
        kernel_stream,
        warmup,
        iterations);
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

    row.overlap = measure_overlap_wall(
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

std::string render_wall_stats_json(const WallStats& stats) {
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
        << "\"copy_solo\":" << render_wall_stats_json(row.copy_solo) << ","
        << "\"kernel_solo\":" << render_kernel_stats_json(row.kernel_solo) << ","
        << "\"overlap\":" << render_overlap_stats_json(row.overlap)
        << "}";
    return oss.str();
}

std::string render_memory_json(const MemoryRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"h2d\":" << render_direction_json(row.h2d) << ","
        << "\"d2h\":" << render_direction_json(row.d2h)
        << "}";
    return oss.str();
}

bool direction_ok(const DirectionRow& row) {
    return row.copy_solo.success && row.kernel_solo.success && row.overlap.success;
}

std::string render_json(
    const bench::Options& options,
    int async_engine_count,
    int device_overlap,
    const std::vector<CaseRow>& rows,
    bool validation_passed) {
    double min_pinned_h2d = 0.0;
    double min_pinned_d2h = 0.0;
    double min_pageable_h2d = 0.0;
    double min_pageable_d2h = 0.0;
    double min_pinned = 0.0;
    double min_pageable = 0.0;
    bool have_pinned_h2d = false;
    bool have_pinned_d2h = false;
    bool have_pageable_h2d = false;
    bool have_pageable_d2h = false;

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
              << "\"pinned\":" << render_memory_json(row.pinned) << ","
              << "\"pageable\":" << render_memory_json(row.pageable)
              << "}";

        if (direction_ok(row.pinned.h2d)) {
            min_pinned_h2d = have_pinned_h2d
                ? std::min(min_pinned_h2d, row.pinned.h2d.overlap.wall_vs_solo_sum_ratio)
                : row.pinned.h2d.overlap.wall_vs_solo_sum_ratio;
            have_pinned_h2d = true;
        }
        if (direction_ok(row.pinned.d2h)) {
            min_pinned_d2h = have_pinned_d2h
                ? std::min(min_pinned_d2h, row.pinned.d2h.overlap.wall_vs_solo_sum_ratio)
                : row.pinned.d2h.overlap.wall_vs_solo_sum_ratio;
            have_pinned_d2h = true;
        }
        if (direction_ok(row.pageable.h2d)) {
            min_pageable_h2d = have_pageable_h2d
                ? std::min(min_pageable_h2d, row.pageable.h2d.overlap.wall_vs_solo_sum_ratio)
                : row.pageable.h2d.overlap.wall_vs_solo_sum_ratio;
            have_pageable_h2d = true;
        }
        if (direction_ok(row.pageable.d2h)) {
            min_pageable_d2h = have_pageable_d2h
                ? std::min(min_pageable_d2h, row.pageable.d2h.overlap.wall_vs_solo_sum_ratio)
                : row.pageable.d2h.overlap.wall_vs_solo_sum_ratio;
            have_pageable_d2h = true;
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return direction_ok(row.pinned.h2d) &&
               direction_ok(row.pinned.d2h) &&
               direction_ok(row.pageable.h2d) &&
               direction_ok(row.pageable.d2h);
    });

    if (have_pinned_h2d && have_pinned_d2h) {
        min_pinned = std::min(min_pinned_h2d, min_pinned_d2h);
    } else if (have_pinned_h2d) {
        min_pinned = min_pinned_h2d;
    } else if (have_pinned_d2h) {
        min_pinned = min_pinned_d2h;
    }

    if (have_pageable_h2d && have_pageable_d2h) {
        min_pageable = std::min(min_pageable_h2d, min_pageable_d2h);
    } else if (have_pageable_h2d) {
        min_pageable = min_pageable_h2d;
    } else if (have_pageable_d2h) {
        min_pageable = min_pageable_d2h;
    }

    double penalty = 0.0;
    if (min_pinned > 0.0) {
        penalty = min_pageable / min_pinned;
    }

    const std::string status = (all_ok && validation_passed) ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"min_pageable_wall_vs_solo_sum_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"copy_directions\":[\"H2D\",\"D2H\"],"
        << "\"memory_types\":[\"pinned\",\"pageable\"],"
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\","
        << "\"async_engine_count\":" << async_engine_count << ","
        << "\"device_overlap\":" << device_overlap << ","
        << "\"min_pinned_h2d_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pinned_h2d) << ","
        << "\"min_pinned_d2h_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pinned_d2h) << ","
        << "\"min_pageable_h2d_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pageable_h2d) << ","
        << "\"min_pageable_d2h_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pageable_d2h) << ","
        << "\"min_pinned_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pinned) << ","
        << "\"min_pageable_wall_vs_solo_sum_ratio\":" << bench::format_double(min_pageable) << ","
        << "\"pageable_to_pinned_overlap_penalty\":" << bench::format_double(penalty) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "}"
        << "}";
    return oss.str();
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
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options, "min_pageable_wall_vs_solo_sum_ratio"));
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
            void* pinned_h2d_src = nullptr;
            void* pinned_d2h_dst = nullptr;
            void* pageable_h2d_src = nullptr;
            void* pageable_d2h_dst = nullptr;
            void* verify_buffer = nullptr;
            float* device_kernel_output = nullptr;

            auto fail_case = [&](const std::string& message) {
                row.pinned.h2d.copy_solo.error = message;
                row.pinned.h2d.kernel_solo.error = message;
                row.pinned.h2d.overlap.error = message;
                row.pinned.d2h.copy_solo.error = message;
                row.pinned.d2h.kernel_solo.error = message;
                row.pinned.d2h.overlap.error = message;
                row.pageable.h2d.copy_solo.error = message;
                row.pageable.h2d.kernel_solo.error = message;
                row.pageable.h2d.overlap.error = message;
                row.pageable.d2h.copy_solo.error = message;
                row.pageable.d2h.kernel_solo.error = message;
                row.pageable.d2h.overlap.error = message;
                validation_passed = false;
            };

            const cudaError_t h2d_status = cudaMalloc(&device_h2d_dst, copy_bytes);
            if (h2d_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_h2d_dst) failed: ") + cudaGetErrorString(h2d_status));
                rows.push_back(row);
                continue;
            }

            const cudaError_t d2h_status = cudaMalloc(&device_d2h_src, copy_bytes);
            if (d2h_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_d2h_src) failed: ") + cudaGetErrorString(d2h_status));
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t pinned_h2d_status = cudaMallocHost(&pinned_h2d_src, copy_bytes);
            if (pinned_h2d_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(pinned_h2d_src) failed: ") + cudaGetErrorString(pinned_h2d_status));
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t pinned_d2h_status = cudaMallocHost(&pinned_d2h_dst, copy_bytes);
            if (pinned_d2h_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(pinned_d2h_dst) failed: ") + cudaGetErrorString(pinned_d2h_status));
                cudaFreeHost(pinned_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            pageable_h2d_src = std::malloc(copy_bytes);
            pageable_d2h_dst = std::malloc(copy_bytes);
            verify_buffer = std::malloc(copy_bytes);
            if (pageable_h2d_src == nullptr || pageable_d2h_dst == nullptr || verify_buffer == nullptr) {
                fail_case("pageable or verify buffer allocation failed");
                if (verify_buffer != nullptr) std::free(verify_buffer);
                if (pageable_d2h_dst != nullptr) std::free(pageable_d2h_dst);
                if (pageable_h2d_src != nullptr) std::free(pageable_h2d_src);
                cudaFreeHost(pinned_d2h_dst);
                cudaFreeHost(pinned_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t kernel_status =
                cudaMalloc(&device_kernel_output, static_cast<size_t>(total_threads) * sizeof(float));
            if (kernel_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_kernel_output) failed: ") + cudaGetErrorString(kernel_status));
                std::free(verify_buffer);
                std::free(pageable_d2h_dst);
                std::free(pageable_h2d_src);
                cudaFreeHost(pinned_d2h_dst);
                cudaFreeHost(pinned_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            std::memset(pinned_h2d_src, 0x3C, copy_bytes);
            std::memset(pinned_d2h_dst, 0x00, copy_bytes);
            std::memset(pageable_h2d_src, 0x3C, copy_bytes);
            std::memset(pageable_d2h_dst, 0x00, copy_bytes);
            bench::check_cuda(cudaMemset(device_h2d_dst, 0x00, copy_bytes), "cudaMemset(device_h2d_dst)");
            bench::check_cuda(cudaMemset(device_d2h_src, 0x5A, copy_bytes), "cudaMemset(device_d2h_src)");
            bench::check_cuda(
                cudaMemset(device_kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
                "cudaMemset(device_kernel_output)");

            row.pinned.h2d = run_direction_case(
                cudaMemcpyHostToDevice,
                device_h2d_dst,
                pinned_h2d_src,
                pinned_h2d_src,
                pinned_h2d_src,
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
                "cudaMemset(device_kernel_output before pinned d2h)");

            row.pinned.d2h = run_direction_case(
                cudaMemcpyDeviceToHost,
                pinned_d2h_dst,
                device_d2h_src,
                nullptr,
                pinned_d2h_dst,
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

            bench::check_cuda(cudaMemset(device_h2d_dst, 0x00, copy_bytes), "cudaMemset(device_h2d_dst before pageable)");
            bench::check_cuda(
                cudaMemset(device_kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
                "cudaMemset(device_kernel_output before pageable h2d)");

            row.pageable.h2d = run_direction_case(
                cudaMemcpyHostToDevice,
                device_h2d_dst,
                pageable_h2d_src,
                pageable_h2d_src,
                pageable_h2d_src,
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

            bench::check_cuda(cudaMemset(device_d2h_src, 0x5A, copy_bytes), "cudaMemset(device_d2h_src before pageable d2h)");
            bench::check_cuda(
                cudaMemset(device_kernel_output, 0x00, static_cast<size_t>(total_threads) * sizeof(float)),
                "cudaMemset(device_kernel_output before pageable d2h)");

            row.pageable.d2h = run_direction_case(
                cudaMemcpyDeviceToHost,
                pageable_d2h_dst,
                device_d2h_src,
                nullptr,
                pageable_d2h_dst,
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
            std::free(pageable_d2h_dst);
            std::free(pageable_h2d_src);
            cudaFreeHost(pinned_d2h_dst);
            cudaFreeHost(pinned_h2d_src);
            cudaFree(device_d2h_src);
            cudaFree(device_h2d_dst);
            rows.push_back(row);
        }

        cudaStreamDestroy(kernel_stream);
        cudaStreamDestroy(copy_stream);

        bench::emit_json(render_json(options, props.asyncEngineCount, props.deviceOverlap, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "min_pageable_wall_vs_solo_sum_ratio"));
        return 1;
    }
}
