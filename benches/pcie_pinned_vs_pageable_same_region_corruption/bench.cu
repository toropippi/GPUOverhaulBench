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
constexpr std::uint32_t kOldPatternWord = 0x5A5A5A5Au;

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

struct CorruptionStats {
    bool success = false;
    std::string error;
    double avg_wall_ms = 0.0;
    unsigned long long old_pattern_bytes = 0;
    unsigned long long new_pattern_bytes = 0;
    unsigned long long mixed_bytes = 0;
    unsigned long long other_bytes = 0;
    double old_pattern_ratio = 0.0;
    double new_pattern_ratio = 0.0;
    double mixed_ratio = 0.0;
    double other_ratio = 0.0;
};

struct MemoryRow {
    WallStats copy_solo;
    KernelStats kernel_solo;
    CorruptionStats corruption;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    MemoryRow pinned;
    MemoryRow pageable;
};

__host__ __device__ __forceinline__ std::uint32_t make_new_pattern(std::uint32_t index) {
    return 0xA5A5A5A5u ^ (index * 2654435761u);
}

__global__ void corruption_pattern_kernel(std::uint32_t* output, size_t words, unsigned long long loops) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= words) {
        return;
    }

    volatile std::uint32_t* volatile_output = output;
    const std::uint32_t value = make_new_pattern(static_cast<std::uint32_t>(tid));
    for (unsigned long long i = 0; i < loops; ++i) {
        volatile_output[tid] = value;
    }
}

bool is_default_size_list(const std::vector<size_t>& sizes_mb) {
    const std::vector<size_t> shared_default_sizes = {8, 32, 128, 512, 1024};
    return sizes_mb == shared_default_sizes;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb <= 128) {
        return std::min(requested_iterations, 12);
    }
    if (size_mb <= 512) {
        return std::min(requested_iterations, 6);
    }
    return std::min(requested_iterations, 4);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb <= 512) {
        return std::min(requested_warmup, 2);
    }
    return 1;
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
        corruption_pattern_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_buffer, words, loops);
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
        corruption_pattern_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(device_buffer, words, loops);
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
    unsigned long long loops = 1ull << 12;
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

CorruptionStats measure_corruption(
    void* host_dst,
    std::uint32_t* device_buffer,
    size_t bytes,
    size_t words,
    unsigned long long loops,
    cudaStream_t copy_stream,
    cudaStream_t kernel_stream,
    int warmup,
    int iterations) {
    CorruptionStats stats;
    const int blocks = static_cast<int>((words + kThreadsPerBlock - 1) / kThreadsPerBlock);

    for (int i = 0; i < warmup; ++i) {
        bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer warmup)");
        bench::check_cuda(cudaMemcpyAsync(host_dst, device_buffer, bytes, cudaMemcpyDeviceToHost, copy_stream), "warmup copy");
        corruption_pattern_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(device_buffer, words, loops);
        bench::check_cuda(cudaGetLastError(), "warmup kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "warmup cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "warmup cudaStreamSynchronize(kernel)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer reset)");
        std::memset(host_dst, 0x00, bytes);

        const auto start = std::chrono::steady_clock::now();
        bench::check_cuda(cudaMemcpyAsync(host_dst, device_buffer, bytes, cudaMemcpyDeviceToHost, copy_stream), "pair copy");
        corruption_pattern_kernel<<<blocks, kThreadsPerBlock, 0, kernel_stream>>>(device_buffer, words, loops);
        bench::check_cuda(cudaGetLastError(), "pair kernel launch");
        bench::check_cuda(cudaStreamSynchronize(copy_stream), "cudaStreamSynchronize(copy)");
        bench::check_cuda(cudaStreamSynchronize(kernel_stream), "cudaStreamSynchronize(kernel)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();

        const auto* host_words = static_cast<const std::uint32_t*>(host_dst);
        for (size_t word_index = 0; word_index < words; ++word_index) {
            const std::uint32_t observed = host_words[word_index];
            const std::uint32_t expected_new = make_new_pattern(static_cast<std::uint32_t>(word_index));

            if (observed == kOldPatternWord) {
                stats.old_pattern_bytes += sizeof(std::uint32_t);
                continue;
            }
            if (observed == expected_new) {
                stats.new_pattern_bytes += sizeof(std::uint32_t);
                continue;
            }

            bool seen_old = false;
            bool seen_new = false;
            bool unknown = false;
            for (int byte_index = 0; byte_index < 4; ++byte_index) {
                const std::uint8_t observed_byte = static_cast<std::uint8_t>((observed >> (byte_index * 8)) & 0xFFu);
                const std::uint8_t old_byte = static_cast<std::uint8_t>((kOldPatternWord >> (byte_index * 8)) & 0xFFu);
                const std::uint8_t new_byte = static_cast<std::uint8_t>((expected_new >> (byte_index * 8)) & 0xFFu);
                if (observed_byte == old_byte) {
                    seen_old = true;
                } else if (observed_byte == new_byte) {
                    seen_new = true;
                } else {
                    unknown = true;
                }
            }

            if (!unknown && seen_old && seen_new) {
                stats.mixed_bytes += sizeof(std::uint32_t);
            } else {
                stats.other_bytes += sizeof(std::uint32_t);
            }
        }
    }

    stats.success = true;
    stats.avg_wall_ms = total_ms / static_cast<double>(iterations);
    const double total_bytes = static_cast<double>(bytes) * static_cast<double>(iterations);
    stats.old_pattern_ratio = stats.old_pattern_bytes / total_bytes;
    stats.new_pattern_ratio = stats.new_pattern_bytes / total_bytes;
    stats.mixed_ratio = stats.mixed_bytes / total_bytes;
    stats.other_ratio = stats.other_bytes / total_bytes;
    return stats;
}

MemoryRow run_memory_case(
    void* host_dst,
    std::uint32_t* device_buffer,
    size_t bytes,
    size_t words,
    cudaStream_t copy_stream,
    cudaStream_t kernel_stream,
    int warmup,
    int iterations) {
    MemoryRow row;
    row.copy_solo = measure_copy_wall(host_dst, device_buffer, bytes, copy_stream, warmup, iterations);
    if (!row.copy_solo.success) {
        return row;
    }

    row.kernel_solo = calibrate_kernel_to_target(device_buffer, words, row.copy_solo.avg_ms, kernel_stream, warmup, iterations);
    if (!row.kernel_solo.success) {
        return row;
    }

    row.corruption = measure_corruption(
        host_dst,
        device_buffer,
        bytes,
        words,
        row.kernel_solo.loop_count,
        copy_stream,
        kernel_stream,
        warmup,
        iterations);
    return row;
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

std::string render_corruption_stats_json(const CorruptionStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_wall_ms\":" << bench::format_double(stats.avg_wall_ms) << ","
        << "\"old_pattern_bytes\":" << stats.old_pattern_bytes << ","
        << "\"new_pattern_bytes\":" << stats.new_pattern_bytes << ","
        << "\"mixed_bytes\":" << stats.mixed_bytes << ","
        << "\"other_bytes\":" << stats.other_bytes << ","
        << "\"old_pattern_ratio\":" << bench::format_double(stats.old_pattern_ratio) << ","
        << "\"new_pattern_ratio\":" << bench::format_double(stats.new_pattern_ratio) << ","
        << "\"mixed_ratio\":" << bench::format_double(stats.mixed_ratio) << ","
        << "\"other_ratio\":" << bench::format_double(stats.other_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_memory_row_json(const MemoryRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"solo_copy\":" << render_wall_stats_json(row.copy_solo) << ","
        << "\"solo_kernel\":" << render_kernel_stats_json(row.kernel_solo) << ","
        << "\"corruption\":" << render_corruption_stats_json(row.corruption)
        << "}";
    return oss.str();
}

bool memory_ok(const MemoryRow& row) {
    return row.copy_solo.success && row.kernel_solo.success && row.corruption.success;
}

bool row_has_corruption(const MemoryRow& row) {
    return row.corruption.mixed_ratio > 0.0 ||
           row.corruption.other_ratio > 0.0 ||
           (row.corruption.old_pattern_ratio > 0.0 && row.corruption.new_pattern_ratio > 0.0);
}

double corruption_score(const MemoryRow& row) {
    return row.corruption.mixed_ratio +
           row.corruption.other_ratio +
           std::min(row.corruption.old_pattern_ratio, row.corruption.new_pattern_ratio);
}

std::string render_json(const bench::Options& options, const std::vector<CaseRow>& rows) {
    bool observed_pinned_corruption = false;
    bool observed_pageable_corruption = false;
    double best_pinned_score = -1.0;
    double best_pageable_score = -1.0;
    size_t pinned_case_mb = 0;
    size_t pageable_case_mb = 0;
    double pinned_old_ratio = 0.0;
    double pinned_new_ratio = 0.0;
    double pinned_mixed_ratio = 0.0;
    double pinned_other_ratio = 0.0;
    double pageable_old_ratio = 0.0;
    double pageable_new_ratio = 0.0;
    double pageable_mixed_ratio = 0.0;
    double pageable_other_ratio = 0.0;

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
              << "\"pinned\":" << render_memory_row_json(row.pinned) << ","
              << "\"pageable\":" << render_memory_row_json(row.pageable)
              << "}";

        if (memory_ok(row.pinned) && row_has_corruption(row.pinned)) {
            observed_pinned_corruption = true;
            const double score = corruption_score(row.pinned);
            if (score > best_pinned_score) {
                best_pinned_score = score;
                pinned_case_mb = row.size_mb;
                pinned_old_ratio = row.pinned.corruption.old_pattern_ratio;
                pinned_new_ratio = row.pinned.corruption.new_pattern_ratio;
                pinned_mixed_ratio = row.pinned.corruption.mixed_ratio;
                pinned_other_ratio = row.pinned.corruption.other_ratio;
            }
        }

        if (memory_ok(row.pageable) && row_has_corruption(row.pageable)) {
            observed_pageable_corruption = true;
            const double score = corruption_score(row.pageable);
            if (score > best_pageable_score) {
                best_pageable_score = score;
                pageable_case_mb = row.size_mb;
                pageable_old_ratio = row.pageable.corruption.old_pattern_ratio;
                pageable_new_ratio = row.pageable.corruption.new_pattern_ratio;
                pageable_mixed_ratio = row.pageable.corruption.mixed_ratio;
                pageable_other_ratio = row.pageable.corruption.other_ratio;
            }
        }
    }
    cases << "]";

    if (pinned_case_mb == 0) {
        for (const auto& row : rows) {
            if (memory_ok(row.pinned)) {
                pinned_case_mb = row.size_mb;
                pinned_old_ratio = row.pinned.corruption.old_pattern_ratio;
                pinned_new_ratio = row.pinned.corruption.new_pattern_ratio;
                pinned_mixed_ratio = row.pinned.corruption.mixed_ratio;
                pinned_other_ratio = row.pinned.corruption.other_ratio;
                break;
            }
        }
    }

    if (pageable_case_mb == 0) {
        for (const auto& row : rows) {
            if (memory_ok(row.pageable)) {
                pageable_case_mb = row.size_mb;
                pageable_old_ratio = row.pageable.corruption.old_pattern_ratio;
                pageable_new_ratio = row.pageable.corruption.new_pattern_ratio;
                pageable_mixed_ratio = row.pageable.corruption.mixed_ratio;
                pageable_other_ratio = row.pageable.corruption.other_ratio;
                break;
            }
        }
    }

    const bool all_ran = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return memory_ok(row.pinned) && memory_ok(row.pageable);
    });
    const bool validation_passed = all_ran;
    const std::string status = validation_passed ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"pageable_representative_old_pattern_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"copy_direction\":\"D2H\","
        << "\"shared_region\":true,"
        << "\"memory_types\":[\"pinned\",\"pageable\"],"
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\","
        << "\"pinned_representative_case_mb\":" << pinned_case_mb << ","
        << "\"pinned_representative_old_pattern_ratio\":" << bench::format_double(pinned_old_ratio) << ","
        << "\"pinned_representative_new_pattern_ratio\":" << bench::format_double(pinned_new_ratio) << ","
        << "\"pinned_representative_mixed_ratio\":" << bench::format_double(pinned_mixed_ratio) << ","
        << "\"pinned_representative_other_ratio\":" << bench::format_double(pinned_other_ratio) << ","
        << "\"pageable_representative_case_mb\":" << pageable_case_mb << ","
        << "\"pageable_representative_old_pattern_ratio\":" << bench::format_double(pageable_old_ratio) << ","
        << "\"pageable_representative_new_pattern_ratio\":" << bench::format_double(pageable_new_ratio) << ","
        << "\"pageable_representative_mixed_ratio\":" << bench::format_double(pageable_mixed_ratio) << ","
        << "\"pageable_representative_other_ratio\":" << bench::format_double(pageable_other_ratio) << ","
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
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options, "pageable_representative_old_pattern_ratio"));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaStream_t copy_stream = nullptr;
        cudaStream_t kernel_stream = nullptr;
        bench::check_cuda(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(copy)");
        bench::check_cuda(cudaStreamCreateWithFlags(&kernel_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(kernel)");

        std::vector<CaseRow> rows;

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            const size_t words = bytes / sizeof(std::uint32_t);
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            void* device_buffer = nullptr;
            void* pinned_host_dst = nullptr;
            void* pageable_host_dst = nullptr;

            const cudaError_t device_status = cudaMalloc(&device_buffer, bytes);
            if (device_status != cudaSuccess) {
                const std::string message = std::string("cudaMalloc failed: ") + cudaGetErrorString(device_status);
                row.pinned.copy_solo.error = message;
                row.pinned.kernel_solo.error = message;
                row.pinned.corruption.error = message;
                row.pageable.copy_solo.error = message;
                row.pageable.kernel_solo.error = message;
                row.pageable.corruption.error = message;
                rows.push_back(row);
                continue;
            }

            const cudaError_t pinned_status = cudaMallocHost(&pinned_host_dst, bytes);
            if (pinned_status != cudaSuccess) {
                const std::string message = std::string("cudaMallocHost failed: ") + cudaGetErrorString(pinned_status);
                row.pinned.copy_solo.error = message;
                row.pinned.kernel_solo.error = message;
                row.pinned.corruption.error = message;
                row.pageable.copy_solo.error = message;
                row.pageable.kernel_solo.error = message;
                row.pageable.corruption.error = message;
                cudaFree(device_buffer);
                rows.push_back(row);
                continue;
            }

            pageable_host_dst = std::malloc(bytes);
            if (pageable_host_dst == nullptr) {
                const std::string message = "pageable host allocation failed";
                row.pinned.copy_solo.error = message;
                row.pinned.kernel_solo.error = message;
                row.pinned.corruption.error = message;
                row.pageable.copy_solo.error = message;
                row.pageable.kernel_solo.error = message;
                row.pageable.corruption.error = message;
                cudaFreeHost(pinned_host_dst);
                cudaFree(device_buffer);
                rows.push_back(row);
                continue;
            }

            std::memset(pinned_host_dst, 0x00, bytes);
            std::memset(pageable_host_dst, 0x00, bytes);
            bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer before pinned)");

            row.pinned = run_memory_case(
                pinned_host_dst,
                static_cast<std::uint32_t*>(device_buffer),
                bytes,
                words,
                copy_stream,
                kernel_stream,
                row.warmup,
                row.iterations);

            bench::check_cuda(cudaMemset(device_buffer, 0x5A, bytes), "cudaMemset(device_buffer before pageable)");
            row.pageable = run_memory_case(
                pageable_host_dst,
                static_cast<std::uint32_t*>(device_buffer),
                bytes,
                words,
                copy_stream,
                kernel_stream,
                row.warmup,
                row.iterations);

            std::free(pageable_host_dst);
            cudaFreeHost(pinned_host_dst);
            cudaFree(device_buffer);
            rows.push_back(row);
        }

        cudaStreamDestroy(kernel_stream);
        cudaStreamDestroy(copy_stream);

        bench::emit_json(render_json(options, rows));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "pageable_representative_old_pattern_ratio"));
        return 1;
    }
}
