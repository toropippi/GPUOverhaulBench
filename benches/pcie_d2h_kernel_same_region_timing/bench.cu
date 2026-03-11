#include "bench_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    WallStats copy_solo;
    KernelStats kernel_solo;
    OverlapStats overlap;
};

__global__ void same_region_spin_kernel(std::uint32_t* output, size_t words, unsigned long long loops) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= words) {
        return;
    }

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
    output[tid] = x ^ y ^ z;
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

WallStats measure_copy_wall(
    void* host_dst,
    const void* device_src,
    size_t bytes,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    WallStats stats;

    for (int i = 0; i < warmup; ++i) {
        const cudaError_t status = cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream);
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
        bench::check_cuda(cudaMemcpyAsync(host_dst, device_src, bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        bench::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(copy)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    return stats;
}

KernelStats measure_kernel_wall(
    std::uint32_t* device_buffer,
    size_t words,
    unsigned long long loops,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    KernelStats stats;
    const int blocks = static_cast<int>((words + kThreadsPerBlock - 1) / kThreadsPerBlock);

    for (int i = 0; i < warmup; ++i) {
        same_region_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_buffer, words, loops);
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
        same_region_spin_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_buffer, words, loops);
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
    std::uint32_t* device_buffer,
    size_t words,
    double target_ms,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    unsigned long long loops = 1ull << 16;
    KernelStats stats;

    for (int pass = 0; pass < kCalibrationPassLimit; ++pass) {
        stats = measure_kernel_wall(device_buffer, words, loops, stream, warmup, iterations);
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
    void* host_dst,
    std::uint32_t* device_buffer,
    size_t bytes,
    size_t words,
    unsigned long long loops,
    cudaStream_t copy_stream,
    cudaStream_t kernel_stream,
    int warmup,
    int iterations,
    const WallStats& copy_solo,
    const KernelStats& kernel_solo) {
    OverlapStats stats;
    const int blocks = static_cast<int>((words + kThreadsPerBlock - 1) / kThreadsPerBlock);

    for (int i = 0; i < warmup; ++i) {
        bench::check_cuda(
            cudaMemcpyAsync(host_dst, device_buffer, bytes, cudaMemcpyDeviceToHost, copy_stream),
            "warmup copy");
        same_region_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(device_buffer, words, loops);
        bench::check_cuda(cudaGetLastError(), "warmup kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "warmup cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "warmup cudaStreamSynchronize(kernel)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        bench::check_cuda(
            cudaMemcpyAsync(host_dst, device_buffer, bytes, cudaMemcpyDeviceToHost, copy_stream),
            "pair copy");
        same_region_spin_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(device_buffer, words, loops);
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
    return stats;
}

std::string render_wall_stats_json(const WallStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << bench::format_double(stats.avg_ms);
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
        << "\"wall_vs_solo_sum_ratio\":" << bench::format_double(stats.wall_vs_solo_sum_ratio) << ","
        << "\"wall_vs_solo_max_ratio\":" << bench::format_double(stats.wall_vs_solo_max_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(const bench::Options& options, const std::vector<CaseRow>& rows, bool validation_passed) {
    double best_overlap_wall_ms = 0.0;
    double min_wall_vs_sum = 0.0;
    double min_wall_vs_max = 0.0;
    unsigned long long representative_loop_count = 0;
    bool first_success = true;

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
              << "\"solo_copy\":" << render_wall_stats_json(row.copy_solo) << ","
              << "\"solo_kernel\":" << render_kernel_stats_json(row.kernel_solo) << ","
              << "\"overlap\":" << render_overlap_stats_json(row.overlap)
              << "}";

        if (row.copy_solo.success && row.kernel_solo.success && row.overlap.success) {
            if (first_success) {
                best_overlap_wall_ms = row.overlap.avg_wall_ms;
                min_wall_vs_sum = row.overlap.wall_vs_solo_sum_ratio;
                min_wall_vs_max = row.overlap.wall_vs_solo_max_ratio;
                representative_loop_count = row.kernel_solo.loop_count;
                first_success = false;
            } else {
                best_overlap_wall_ms = std::min(best_overlap_wall_ms, row.overlap.avg_wall_ms);
                if (row.overlap.wall_vs_solo_sum_ratio < min_wall_vs_sum) {
                    min_wall_vs_sum = row.overlap.wall_vs_solo_sum_ratio;
                    representative_loop_count = row.kernel_solo.loop_count;
                }
                min_wall_vs_max = std::min(min_wall_vs_max, row.overlap.wall_vs_solo_max_ratio);
            }
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.copy_solo.success && row.kernel_solo.success && row.overlap.success;
    });
    const std::string status = (all_ok && validation_passed) ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"min_wall_vs_solo_sum_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"copy_direction\":\"D2H\","
        << "\"shared_region\":true,"
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\","
        << "\"best_overlap_wall_ms\":" << bench::format_double(best_overlap_wall_ms) << ","
        << "\"min_wall_vs_solo_sum_ratio\":" << bench::format_double(min_wall_vs_sum) << ","
        << "\"min_wall_vs_solo_max_ratio\":" << bench::format_double(min_wall_vs_max) << ","
        << "\"representative_kernel_loop_count\":" << representative_loop_count << ","
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
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options, "min_wall_vs_solo_sum_ratio"));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaStream_t copy_stream = nullptr;
        cudaStream_t kernel_stream = nullptr;
        bench::check_cuda(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(copy)");
        bench::check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(kernel)");

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            const size_t words = bytes / sizeof(std::uint32_t);
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            void* device_buffer = nullptr;
            void* host_dst = nullptr;

            const cudaError_t device_status = cudaMalloc(&device_buffer, bytes);
            if (device_status != cudaSuccess) {
                row.copy_solo.error = std::string("cudaMalloc failed: ") + cudaGetErrorString(device_status);
                row.kernel_solo.error = row.copy_solo.error;
                row.overlap.error = row.copy_solo.error;
                validation_passed = false;
                rows.push_back(row);
                continue;
            }

            const cudaError_t host_status = cudaMallocHost(&host_dst, bytes);
            if (host_status != cudaSuccess) {
                const std::string message = std::string("cudaMallocHost failed: ") + cudaGetErrorString(host_status);
                row.copy_solo.error = message;
                row.kernel_solo.error = message;
                row.overlap.error = message;
                validation_passed = false;
                cudaFree(device_buffer);
                rows.push_back(row);
                continue;
            }

            std::memset(host_dst, 0x00, bytes);
            bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer)");

            row.copy_solo = measure_copy_wall(host_dst, device_buffer, bytes, copy_stream, row.warmup, row.iterations);
            if (!row.copy_solo.success) {
                validation_passed = false;
            }

            row.kernel_solo = calibrate_kernel_to_target(
                static_cast<std::uint32_t*>(device_buffer),
                words,
                row.copy_solo.avg_ms,
                kernel_stream,
                row.warmup,
                row.iterations);
            if (!row.kernel_solo.success) {
                validation_passed = false;
            }

            bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer reset)");
            std::memset(host_dst, 0x00, bytes);

            if (row.copy_solo.success && row.kernel_solo.success) {
                row.overlap = measure_overlap_wall(
                    host_dst,
                    static_cast<std::uint32_t*>(device_buffer),
                    bytes,
                    words,
                    row.kernel_solo.loop_count,
                    copy_stream,
                    kernel_stream,
                    row.warmup,
                    row.iterations,
                    row.copy_solo,
                    row.kernel_solo);
                if (!row.overlap.success) {
                    validation_passed = false;
                }
            }

            cudaFreeHost(host_dst);
            cudaFree(device_buffer);
            rows.push_back(row);
        }

        cudaStreamDestroy(kernel_stream);
        cudaStreamDestroy(copy_stream);

        bench::emit_json(render_json(options, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "min_wall_vs_solo_sum_ratio"));
        return 1;
    }
}
