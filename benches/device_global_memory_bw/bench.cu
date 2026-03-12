#include "bench_support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct OperationStats {
    bench::CopyStats copy;
    bench::CopyStats memset;
    bench::CopyStats read_only;
    bench::CopyStats write_only;
    bench::CopyStats read_write;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    OperationStats stats;
};

constexpr uint32_t kPatternSeed = 0xA5A5A5A5u;
constexpr uint32_t kWriteSeed = 0x13579BDFu;
constexpr uint32_t kReadWriteXor = 0x9E3779B9u;
constexpr unsigned char kMemsetValue = 0x3C;
constexpr int kThreadsPerBlock = 256;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

uint32_t pattern_value(uint32_t index) {
    return (index * 2654435761u) ^ kPatternSeed;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb <= 128) {
        return requested_iterations;
    }
    if (size_mb <= 512) {
        return std::min(requested_iterations, 25);
    }
    if (size_mb <= 1024) {
        return std::min(requested_iterations, 12);
    }
    return std::min(requested_iterations, 6);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb <= 512) {
        return requested_warmup;
    }
    if (size_mb <= 1024) {
        return std::min(requested_warmup, 3);
    }
    return std::min(requested_warmup, 2);
}

int choose_block_count(int sm_count, size_t word_count) {
    const size_t blocks_by_size = (word_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int target_blocks = std::max(1, sm_count * 8);
    return static_cast<int>(std::min<size_t>(blocks_by_size, static_cast<size_t>(target_blocks)));
}

std::vector<uint32_t> build_source_words(size_t max_words) {
    std::vector<uint32_t> words(max_words);
    for (size_t i = 0; i < max_words; ++i) {
        words[i] = pattern_value(static_cast<uint32_t>(i));
    }
    return words;
}

std::vector<uint64_t> build_prefix_sums(const std::vector<uint32_t>& words) {
    std::vector<uint64_t> prefix(words.size() + 1, 0);
    for (size_t i = 0; i < words.size(); ++i) {
        prefix[i + 1] = prefix[i] + static_cast<uint64_t>(words[i]);
    }
    return prefix;
}

template <typename LaunchFn>
bench::CopyStats measure_timed_operation(size_t logical_bytes, int warmup, int iterations, LaunchFn launch) {
    bench::CopyStats stats;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const cudaError_t start_status = cudaEventCreate(&start);
    if (start_status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(start) failed: ") + cudaGetErrorString(start_status);
        return stats;
    }

    const cudaError_t stop_status = cudaEventCreate(&stop);
    if (stop_status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(stop) failed: ") + cudaGetErrorString(stop_status);
        cudaEventDestroy(start);
        return stats;
    }

    try {
        for (int i = 0; i < warmup; ++i) {
            launch();
            bench::check_cuda(cudaDeviceSynchronize(), "warmup cudaDeviceSynchronize");
        }

        float total_ms = 0.0f;
        for (int i = 0; i < iterations; ++i) {
            bench::check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            launch();
            bench::check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            bench::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

            float elapsed_ms = 0.0f;
            bench::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
            total_ms += elapsed_ms;
        }

        stats.success = true;
        stats.avg_ms = total_ms / static_cast<double>(iterations);
        const double gib = static_cast<double>(logical_bytes) / kBytesPerGiB;
        stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return stats;
}

__global__ void read_only_kernel(const uint32_t* src, uint64_t* partial, size_t word_count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    uint64_t sum = 0;
    for (size_t i = global_tid; i < word_count; i += stride) {
        sum += static_cast<uint64_t>(src[i]);
    }
    partial[global_tid] = sum;
}

__global__ void write_only_kernel(uint32_t* dst, size_t word_count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = global_tid; i < word_count; i += stride) {
        dst[i] = static_cast<uint32_t>(i) ^ kWriteSeed;
    }
}

__global__ void read_write_kernel(const uint32_t* src, uint32_t* dst, size_t word_count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (size_t i = global_tid; i < word_count; i += stride) {
        dst[i] = src[i] ^ kReadWriteXor;
    }
}

uint64_t reduce_partials(const std::vector<uint64_t>& partials) {
    return std::accumulate(partials.begin(), partials.end(), uint64_t{0});
}

bool validate_copy(void* dst_words_device, const std::vector<uint32_t>& expected_words, size_t word_count) {
    std::vector<uint32_t> observed(word_count);
    bench::check_cuda(
        cudaMemcpy(observed.data(), dst_words_device, word_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "validate copy");
    return std::memcmp(observed.data(), expected_words.data(), word_count * sizeof(uint32_t)) == 0;
}

bool validate_memset(void* dst_device, size_t bytes) {
    std::vector<unsigned char> observed(bytes);
    bench::check_cuda(cudaMemcpy(observed.data(), dst_device, bytes, cudaMemcpyDeviceToHost), "validate memset");
    return std::all_of(observed.begin(), observed.end(), [](unsigned char v) { return v == kMemsetValue; });
}

bool validate_read_only(
    const std::vector<uint64_t>& prefix_sums,
    uint64_t* partial_device,
    size_t word_count,
    size_t partial_count) {
    std::vector<uint64_t> partials(partial_count);
    bench::check_cuda(
        cudaMemcpy(partials.data(), partial_device, partial_count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "validate read_only");
    const uint64_t expected = prefix_sums[word_count];
    return reduce_partials(partials) == expected;
}

bool validate_write_only(void* dst_words_device, size_t word_count) {
    std::vector<uint32_t> observed(word_count);
    bench::check_cuda(
        cudaMemcpy(observed.data(), dst_words_device, word_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "validate write_only");
    for (size_t i = 0; i < word_count; ++i) {
        if (observed[i] != (static_cast<uint32_t>(i) ^ kWriteSeed)) {
            return false;
        }
    }
    return true;
}

bool validate_read_write(void* dst_words_device, const std::vector<uint32_t>& source_words, size_t word_count) {
    std::vector<uint32_t> observed(word_count);
    bench::check_cuda(
        cudaMemcpy(observed.data(), dst_words_device, word_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "validate read_write");
    for (size_t i = 0; i < word_count; ++i) {
        if (observed[i] != (source_words[i] ^ kReadWriteXor)) {
            return false;
        }
    }
    return true;
}

std::string render_stat_json(const bench::CopyStats& stats) {
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

std::string render_json(const bench::Options& options, const std::vector<CaseRow>& rows, bool validation_passed) {
    double best_copy = 0.0;
    double best_memset = 0.0;
    double best_read_only = 0.0;
    double best_write_only = 0.0;
    double best_read_write = 0.0;

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
              << "\"copy\":" << render_stat_json(row.stats.copy) << ","
              << "\"memset\":" << render_stat_json(row.stats.memset) << ","
              << "\"read_only\":" << render_stat_json(row.stats.read_only) << ","
              << "\"write_only\":" << render_stat_json(row.stats.write_only) << ","
              << "\"read_write\":" << render_stat_json(row.stats.read_write)
              << "}";

        if (row.stats.copy.success) {
            best_copy = std::max(best_copy, row.stats.copy.gib_per_s);
        }
        if (row.stats.memset.success) {
            best_memset = std::max(best_memset, row.stats.memset.gib_per_s);
        }
        if (row.stats.read_only.success) {
            best_read_only = std::max(best_read_only, row.stats.read_only.gib_per_s);
        }
        if (row.stats.write_only.success) {
            best_write_only = std::max(best_write_only, row.stats.write_only.gib_per_s);
        }
        if (row.stats.read_write.success) {
            best_read_write = std::max(best_read_write, row.stats.read_write.gib_per_s);
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.stats.copy.success && row.stats.memset.success && row.stats.read_only.success &&
               row.stats.write_only.success && row.stats.read_write.success;
    });

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"best_read_only_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event\","
        << "\"adaptive_iteration_schedule\":true,"
        << "\"operations\":[\"copy\",\"memset\",\"read_only\",\"write_only\",\"read_write\"],"
        << "\"best_copy_gib_per_s\":" << bench::format_double(best_copy) << ","
        << "\"best_memset_gib_per_s\":" << bench::format_double(best_memset) << ","
        << "\"best_read_only_gib_per_s\":" << bench::format_double(best_read_only) << ","
        << "\"best_write_only_gib_per_s\":" << bench::format_double(best_write_only) << ","
        << "\"best_read_write_gib_per_s\":" << bench::format_double(best_read_write) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "},"
        << "\"notes\":["
        << bench::quote("copy is measured with cudaMemcpyDeviceToDevice.") << ","
        << bench::quote("memset is measured with cudaMemset.") << ","
        << bench::quote("read_only, write_only, and read_write are measured with simple global-memory kernels.")
        << "]"
        << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options{};
    try {
        options = bench::parse_common_args(argc, argv);

        int device_count = 0;
        bench::check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaDeviceProp prop{};
        bench::check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

        const size_t max_size_mb = *std::max_element(options.sizes_mb.begin(), options.sizes_mb.end());
        const size_t max_bytes = max_size_mb * 1024ull * 1024ull;
        const size_t max_words = max_bytes / sizeof(uint32_t);

        const auto source_words = build_source_words(max_words);
        const auto prefix_sums = build_prefix_sums(source_words);

        uint32_t* src_device = nullptr;
        uint32_t* dst_device = nullptr;
        bench::check_cuda(cudaMalloc(&src_device, max_words * sizeof(uint32_t)), "cudaMalloc(src_device)");
        bench::check_cuda(cudaMalloc(&dst_device, max_words * sizeof(uint32_t)), "cudaMalloc(dst_device)");
        bench::check_cuda(
            cudaMemcpy(src_device, source_words.data(), max_words * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "seed src_device");

        const int block_count = choose_block_count(prop.multiProcessorCount, max_words);
        const size_t partial_count = static_cast<size_t>(block_count) * kThreadsPerBlock;
        uint64_t* partial_device = nullptr;
        bench::check_cuda(cudaMalloc(&partial_device, partial_count * sizeof(uint64_t)), "cudaMalloc(partial_device)");

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            const size_t word_count = bytes / sizeof(uint32_t);

            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            row.stats.copy = measure_timed_operation(bytes * 2ull, row.warmup, row.iterations, [&]() {
                bench::check_cuda(cudaMemcpy(dst_device, src_device, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpyDeviceToDevice");
            });
            if (row.stats.copy.success && !validate_copy(dst_device, source_words, word_count)) {
                row.stats.copy.success = false;
                row.stats.copy.error = "copy validation failed";
                validation_passed = false;
            } else if (!row.stats.copy.success) {
                validation_passed = false;
            }

            row.stats.memset = measure_timed_operation(bytes, row.warmup, row.iterations, [&]() {
                bench::check_cuda(cudaMemset(dst_device, kMemsetValue, bytes), "cudaMemset");
            });
            if (row.stats.memset.success && !validate_memset(dst_device, bytes)) {
                row.stats.memset.success = false;
                row.stats.memset.error = "memset validation failed";
                validation_passed = false;
            } else if (!row.stats.memset.success) {
                validation_passed = false;
            }

            row.stats.read_only = measure_timed_operation(bytes, row.warmup, row.iterations, [&]() {
                read_only_kernel<<<block_count, kThreadsPerBlock>>>(src_device, partial_device, word_count);
                bench::check_cuda(cudaGetLastError(), "read_only_kernel launch");
            });
            if (row.stats.read_only.success && !validate_read_only(prefix_sums, partial_device, word_count, partial_count)) {
                row.stats.read_only.success = false;
                row.stats.read_only.error = "read_only validation failed";
                validation_passed = false;
            } else if (!row.stats.read_only.success) {
                validation_passed = false;
            }

            row.stats.write_only = measure_timed_operation(bytes, row.warmup, row.iterations, [&]() {
                write_only_kernel<<<block_count, kThreadsPerBlock>>>(dst_device, word_count);
                bench::check_cuda(cudaGetLastError(), "write_only_kernel launch");
            });
            if (row.stats.write_only.success && !validate_write_only(dst_device, word_count)) {
                row.stats.write_only.success = false;
                row.stats.write_only.error = "write_only validation failed";
                validation_passed = false;
            } else if (!row.stats.write_only.success) {
                validation_passed = false;
            }

            row.stats.read_write = measure_timed_operation(bytes * 2ull, row.warmup, row.iterations, [&]() {
                read_write_kernel<<<block_count, kThreadsPerBlock>>>(src_device, dst_device, word_count);
                bench::check_cuda(cudaGetLastError(), "read_write_kernel launch");
            });
            if (row.stats.read_write.success && !validate_read_write(dst_device, source_words, word_count)) {
                row.stats.read_write.success = false;
                row.stats.read_write.error = "read_write validation failed";
                validation_passed = false;
            } else if (!row.stats.read_write.success) {
                validation_passed = false;
            }

            rows.push_back(row);
        }

        cudaFree(partial_device);
        cudaFree(dst_device);
        cudaFree(src_device);

        bench::emit_json(render_json(options, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "best_read_only_gib_per_s"));
        return 1;
    }
}
