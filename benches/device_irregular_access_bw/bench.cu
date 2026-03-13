#include "bench_support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class PatternKind {
    StrideRead,
    StrideWrite,
    StrideReadWrite,
    Gather,
    Scatter,
    RandomBoth,
};

struct CaseRow {
    PatternKind pattern = PatternKind::StrideRead;
    size_t size_mb = 0;
    size_t stride_bytes = 0;
    size_t access_unit_bytes = 0;
    int iterations = 0;
    int warmup = 0;
    bench::CopyStats stats;
};

constexpr int kThreadsPerBlock = 256;
constexpr unsigned char kDstSentinel = 0xCD;
constexpr unsigned char kStrideWriteXor = 0x5A;
constexpr unsigned char kStrideReadWriteXor = 0xA7;
constexpr uint64_t kBytesPerGiB = 1024ull * 1024ull * 1024ull;
constexpr size_t kValidationChunkBytes = 16ull * 1024ull * 1024ull;
constexpr size_t kDefaultSizesMb[] = {8, 32, 128, 512, 1024};
constexpr size_t kExtendedSizesMb[] = {8, 32, 128, 512, 1024};

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t next = a % b;
        a = b;
        b = next;
    }
    return a;
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
    (void)size_mb;
    (void)requested_warmup;
    return 1;
}

std::vector<size_t> default_strides_bytes() {
    std::vector<size_t> strides;
    for (size_t stride = 1; stride <= 1024; stride <<= 1) {
        strides.push_back(stride);
    }
    return strides;
}

bool is_default_size_list(const std::vector<size_t>& sizes_mb) {
    return sizes_mb == std::vector<size_t>(std::begin(kDefaultSizesMb), std::end(kDefaultSizesMb));
}

int choose_block_count(int sm_count, size_t item_count) {
    const size_t blocks_by_size = (item_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const int target_blocks = std::max(1, sm_count * 8);
    return static_cast<int>(std::min<size_t>(blocks_by_size, static_cast<size_t>(target_blocks)));
}

const char* pattern_name(PatternKind pattern) {
    switch (pattern) {
        case PatternKind::StrideRead:
            return "stride_read";
        case PatternKind::StrideWrite:
            return "stride_write";
        case PatternKind::StrideReadWrite:
            return "stride_read_write";
        case PatternKind::Gather:
            return "gather";
        case PatternKind::Scatter:
            return "scatter";
        case PatternKind::RandomBoth:
            return "random_both";
    }
    return "unknown";
}

uint64_t splitmix64_next(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31u);
}

std::vector<uint32_t> build_random_permutation(size_t element_count) {
    std::vector<uint32_t> permutation(element_count);
    std::iota(permutation.begin(), permutation.end(), 0u);
    uint64_t rng_state = 0xA5A5A5A55A5A5A5Aull ^ static_cast<uint64_t>(element_count);
    for (size_t i = element_count - 1; i > 0; --i) {
        const size_t j = static_cast<size_t>(splitmix64_next(rng_state) % static_cast<uint64_t>(i + 1));
        std::swap(permutation[i], permutation[j]);
    }
    return permutation;
}

std::vector<uint32_t> build_inverse_permutation(const std::vector<uint32_t>& permutation) {
    std::vector<uint32_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        inverse[permutation[i]] = static_cast<uint32_t>(i);
    }
    return inverse;
}

unsigned char source_byte_value(uint64_t offset) {
    return static_cast<unsigned char>(((offset * 17ull) + 23ull) & 0xFFu);
}

uint32_t source_word_value(uint64_t word_index) {
    const uint64_t base = word_index * 4ull;
    return static_cast<uint32_t>(source_byte_value(base + 0)) |
           (static_cast<uint32_t>(source_byte_value(base + 1)) << 8u) |
           (static_cast<uint32_t>(source_byte_value(base + 2)) << 16u) |
           (static_cast<uint32_t>(source_byte_value(base + 3)) << 24u);
}

uint64_t expected_stride_read_sum(size_t size_bytes, size_t stride_bytes) {
    const uint64_t touched_count = static_cast<uint64_t>(size_bytes / stride_bytes);
    const uint64_t step = stride_bytes & 0xFFu;
    if (touched_count == 0) {
        return 0;
    }

    const uint64_t period = 256ull / gcd_u64(step == 0 ? 256ull : step, 256ull);
    uint64_t cycle_sum = 0;
    for (uint64_t i = 0; i < period; ++i) {
        cycle_sum += static_cast<uint64_t>(source_byte_value(i * stride_bytes));
    }

    const uint64_t full_cycles = touched_count / period;
    const uint64_t remainder = touched_count % period;
    uint64_t total = full_cycles * cycle_sum;
    for (uint64_t i = 0; i < remainder; ++i) {
        total += static_cast<uint64_t>(source_byte_value(i * stride_bytes));
    }
    return total;
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
        const double gib = static_cast<double>(logical_bytes) / static_cast<double>(kBytesPerGiB);
        stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return stats;
}

__global__ void init_source_bytes_kernel(unsigned char* dst, size_t count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < count; i += stride) {
        dst[i] = static_cast<unsigned char>(((i * 17ull) + 23ull) & 0xFFu);
    }
}

__global__ void stride_read_kernel(
    const unsigned char* src,
    uint64_t* partial,
    size_t touched_count,
    size_t stride_bytes) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint64_t sum = 0;
    for (size_t i = global_tid; i < touched_count; i += total_threads) {
        sum += static_cast<uint64_t>(src[i * stride_bytes]);
    }
    partial[global_tid] = sum;
}

__global__ void stride_write_kernel(unsigned char* dst, size_t touched_count, size_t stride_bytes) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < touched_count; i += total_threads) {
        const size_t offset = i * stride_bytes;
        dst[offset] = static_cast<unsigned char>(offset & 0xFFu) ^ kStrideWriteXor;
    }
}

__global__ void stride_read_write_kernel(
    const unsigned char* src,
    unsigned char* dst,
    size_t touched_count,
    size_t stride_bytes) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < touched_count; i += total_threads) {
        const size_t offset = i * stride_bytes;
        dst[offset] = src[offset] ^ kStrideReadWriteXor;
    }
}

__global__ void gather_kernel(
    const uint32_t* src,
    uint32_t* dst,
    const uint32_t* permutation,
    size_t element_count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < element_count; i += total_threads) {
        dst[i] = src[permutation[i]];
    }
}

__global__ void scatter_kernel(
    const uint32_t* src,
    uint32_t* dst,
    const uint32_t* permutation,
    size_t element_count) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < element_count; i += total_threads) {
        dst[permutation[i]] = src[i];
    }
}

__global__ void random_both_kernel(
    const uint32_t* src,
    uint32_t* dst,
    const uint32_t* permutation,
    size_t element_count,
    uint64_t mask) {
    const size_t global_tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total_threads = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = global_tid; i < element_count; i += total_threads) {
        const uint64_t src_index = permutation[i];
        const uint64_t dst_index = permutation[(i + 1) & mask];
        dst[dst_index] = src[src_index];
    }
}

bool validate_stride_read(uint64_t* partial_device, size_t active_partial_count, size_t size_bytes, size_t stride_bytes) {
    std::vector<uint64_t> partials(active_partial_count);
    bench::check_cuda(
        cudaMemcpy(partials.data(), partial_device, active_partial_count * sizeof(uint64_t), cudaMemcpyDeviceToHost),
        "validate stride_read");
    uint64_t observed = 0;
    for (uint64_t value : partials) {
        observed += value;
    }
    return observed == expected_stride_read_sum(size_bytes, stride_bytes);
}

bool validate_stride_written_bytes(
    const unsigned char* dst_device,
    size_t size_bytes,
    size_t stride_bytes,
    bool read_write) {
    std::vector<unsigned char> chunk(std::min<size_t>(kValidationChunkBytes, size_bytes));
    const size_t touched_limit = (size_bytes / stride_bytes) * stride_bytes;

    for (size_t offset = 0; offset < size_bytes; offset += chunk.size()) {
        const size_t bytes_this_chunk = std::min(chunk.size(), size_bytes - offset);
        bench::check_cuda(
            cudaMemcpy(chunk.data(), dst_device + offset, bytes_this_chunk, cudaMemcpyDeviceToHost),
            "validate stride write chunk");

        for (size_t i = 0; i < bytes_this_chunk; ++i) {
            const size_t global_offset = offset + i;
            const bool touched =
                (global_offset < touched_limit) && ((global_offset % stride_bytes) == 0);
            const unsigned char expected = touched
                ? (read_write
                    ? static_cast<unsigned char>(source_byte_value(global_offset) ^ kStrideReadWriteXor)
                    : static_cast<unsigned char>((global_offset & 0xFFu) ^ kStrideWriteXor))
                : kDstSentinel;
            if (chunk[i] != expected) {
                return false;
            }
        }
    }
    return true;
}

uint32_t expected_random_word(
    PatternKind pattern,
    uint64_t dst_index,
    const std::vector<uint32_t>& permutation,
    const std::vector<uint32_t>& inverse_permutation,
    uint64_t mask) {
    switch (pattern) {
        case PatternKind::Gather: {
            return source_word_value(permutation[dst_index]);
        }
        case PatternKind::Scatter: {
            return source_word_value(inverse_permutation[dst_index]);
        }
        case PatternKind::RandomBoth: {
            const uint64_t logical_index = inverse_permutation[dst_index];
            const uint64_t src_index = permutation[(logical_index - 1ull) & mask];
            return source_word_value(src_index);
        }
        default:
            return 0;
    }
}

bool validate_random_pattern(
    const uint32_t* dst_device,
    size_t element_count,
    PatternKind pattern,
    const std::vector<uint32_t>& permutation,
    const std::vector<uint32_t>& inverse_permutation) {
    const size_t chunk_words = std::max<size_t>(1, kValidationChunkBytes / sizeof(uint32_t));
    std::vector<uint32_t> chunk(chunk_words);
    const uint64_t mask = static_cast<uint64_t>(element_count - 1);

    for (size_t offset = 0; offset < element_count; offset += chunk_words) {
        const size_t words_this_chunk = std::min(chunk_words, element_count - offset);
        bench::check_cuda(
            cudaMemcpy(chunk.data(), dst_device + offset, words_this_chunk * sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "validate random chunk");

        for (size_t i = 0; i < words_this_chunk; ++i) {
            const uint64_t dst_index = static_cast<uint64_t>(offset + i);
            const uint32_t expected =
                expected_random_word(pattern, dst_index, permutation, inverse_permutation, mask);
            if (chunk[i] != expected) {
                return false;
            }
        }
    }
    return true;
}

std::string render_case_json(const CaseRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"pattern\":" << bench::quote(pattern_name(row.pattern)) << ","
        << "\"size_mb\":" << row.size_mb << ","
        << "\"access_unit_bytes\":" << row.access_unit_bytes << ","
        << "\"iterations\":" << row.iterations << ","
        << "\"warmup\":" << row.warmup << ","
        << "\"avg_ms\":" << bench::format_double(row.stats.avg_ms) << ","
        << "\"gib_per_s\":" << bench::format_double(row.stats.gib_per_s) << ","
        << "\"success\":" << (row.stats.success ? "true" : "false");
    if (row.stride_bytes != 0) {
        oss << ",\"stride_bytes\":" << row.stride_bytes;
    }
    if (!row.stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(row.stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(const bench::Options& options, const std::vector<CaseRow>& rows, bool validation_passed) {
    double best_stride_read = 0.0;
    double best_stride_write = 0.0;
    double best_stride_read_write = 0.0;
    double best_gather = 0.0;
    double best_scatter = 0.0;
    double best_random_both = 0.0;
    double random_both_1024 = 0.0;

    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) {
            cases << ",";
        }
        cases << render_case_json(rows[i]);

        if (!rows[i].stats.success) {
            continue;
        }
        switch (rows[i].pattern) {
            case PatternKind::StrideRead:
                best_stride_read = std::max(best_stride_read, rows[i].stats.gib_per_s);
                break;
            case PatternKind::StrideWrite:
                best_stride_write = std::max(best_stride_write, rows[i].stats.gib_per_s);
                break;
            case PatternKind::StrideReadWrite:
                best_stride_read_write = std::max(best_stride_read_write, rows[i].stats.gib_per_s);
                break;
            case PatternKind::Gather:
                best_gather = std::max(best_gather, rows[i].stats.gib_per_s);
                break;
            case PatternKind::Scatter:
                best_scatter = std::max(best_scatter, rows[i].stats.gib_per_s);
                break;
            case PatternKind::RandomBoth:
                best_random_both = std::max(best_random_both, rows[i].stats.gib_per_s);
                if (rows[i].size_mb == 1024) {
                    random_both_1024 = rows[i].stats.gib_per_s;
                }
                break;
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.stats.success;
    });

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"random_both_1024mb_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event\","
        << "\"adaptive_iteration_schedule\":true,"
        << "\"patterns\":[\"stride_read\",\"stride_write\",\"stride_read_write\",\"gather\",\"scatter\",\"random_both\"],"
        << "\"best_stride_read_gib_per_s\":" << bench::format_double(best_stride_read) << ","
        << "\"best_stride_write_gib_per_s\":" << bench::format_double(best_stride_write) << ","
        << "\"best_stride_read_write_gib_per_s\":" << bench::format_double(best_stride_read_write) << ","
        << "\"best_gather_gib_per_s\":" << bench::format_double(best_gather) << ","
        << "\"best_scatter_gib_per_s\":" << bench::format_double(best_scatter) << ","
        << "\"best_random_both_gib_per_s\":" << bench::format_double(best_random_both) << ","
        << "\"random_both_1024mb_gib_per_s\":" << bench::format_double(random_both_1024) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "},"
        << "\"notes\":["
        << bench::quote("stride uses byte-addressed accesses from 1 B to 1024 B.") << ","
        << bench::quote("gather, scatter, and random_both use explicit host-generated permutation indices over 4-byte elements with no repeated addresses.") << ","
        << bench::quote("results should be compared against device_global_memory_bw for the sequential baseline.")
        << "]"
        << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    bench::Options options{};
    try {
        options = bench::parse_common_args(argc, argv);
        options.warmup = 1;
        if (is_default_size_list(options.sizes_mb)) {
            options.sizes_mb.assign(std::begin(kExtendedSizesMb), std::end(kExtendedSizesMb));
        }

        int device_count = 0;
        bench::check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options, "random_both_1024mb_gib_per_s"));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaDeviceProp prop{};
        bench::check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

        const size_t max_size_mb = *std::max_element(options.sizes_mb.begin(), options.sizes_mb.end());
        const size_t max_size_bytes = max_size_mb * 1024ull * 1024ull;

        unsigned char* src_device = nullptr;
        unsigned char* dst_device = nullptr;
        bench::check_cuda(cudaMalloc(&src_device, max_size_bytes), "cudaMalloc(src_device)");
        bench::check_cuda(cudaMalloc(&dst_device, max_size_bytes), "cudaMalloc(dst_device)");

        const int init_blocks = choose_block_count(prop.multiProcessorCount, max_size_bytes);
        init_source_bytes_kernel<<<init_blocks, kThreadsPerBlock>>>(src_device, max_size_bytes);
        bench::check_cuda(cudaGetLastError(), "init_source_bytes_kernel launch");
        bench::check_cuda(cudaDeviceSynchronize(), "init_source_bytes_kernel sync");

        const int max_partial_blocks = choose_block_count(prop.multiProcessorCount, max_size_bytes);
        const size_t partial_count = static_cast<size_t>(max_partial_blocks) * kThreadsPerBlock;
        uint64_t* partial_device = nullptr;
        bench::check_cuda(cudaMalloc(&partial_device, partial_count * sizeof(uint64_t)), "cudaMalloc(partial_device)");

        const auto strides = default_strides_bytes();

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t size_bytes = size_mb * 1024ull * 1024ull;

            for (size_t stride_bytes : strides) {
                const size_t touched_count = size_bytes / stride_bytes;
                const int block_count = choose_block_count(prop.multiProcessorCount, std::max<size_t>(touched_count, 1));

                for (PatternKind pattern : {PatternKind::StrideRead, PatternKind::StrideWrite, PatternKind::StrideReadWrite}) {
                    CaseRow row;
                    row.pattern = pattern;
                    row.size_mb = size_mb;
                    row.stride_bytes = stride_bytes;
                    row.access_unit_bytes = 1;
                    row.iterations = effective_iterations(size_mb, options.iterations);
                    row.warmup = effective_warmup(size_mb, options.warmup);

                    try {
                        if (pattern != PatternKind::StrideRead) {
                            bench::check_cuda(cudaMemset(dst_device, kDstSentinel, size_bytes), "cudaMemset(dst sentinel)");
                        }

                        switch (pattern) {
                            case PatternKind::StrideRead:
                                row.stats = measure_timed_operation(touched_count, row.warmup, row.iterations, [&]() {
                                    stride_read_kernel<<<block_count, kThreadsPerBlock>>>(
                                        src_device, partial_device, touched_count, stride_bytes);
                                    bench::check_cuda(cudaGetLastError(), "stride_read_kernel launch");
                                });
                                if (row.stats.success && !validate_stride_read(
                                        partial_device,
                                        static_cast<size_t>(block_count) * kThreadsPerBlock,
                                        size_bytes,
                                        stride_bytes)) {
                                    row.stats.success = false;
                                    row.stats.error = "stride_read validation failed";
                                    validation_passed = false;
                                }
                                break;
                            case PatternKind::StrideWrite:
                                row.stats = measure_timed_operation(touched_count, row.warmup, row.iterations, [&]() {
                                    stride_write_kernel<<<block_count, kThreadsPerBlock>>>(
                                        dst_device, touched_count, stride_bytes);
                                    bench::check_cuda(cudaGetLastError(), "stride_write_kernel launch");
                                });
                                if (row.stats.success && !validate_stride_written_bytes(dst_device, size_bytes, stride_bytes, false)) {
                                    row.stats.success = false;
                                    row.stats.error = "stride_write validation failed";
                                    validation_passed = false;
                                }
                                break;
                            case PatternKind::StrideReadWrite:
                                row.stats = measure_timed_operation(touched_count * 2ull, row.warmup, row.iterations, [&]() {
                                    stride_read_write_kernel<<<block_count, kThreadsPerBlock>>>(
                                        src_device, dst_device, touched_count, stride_bytes);
                                    bench::check_cuda(cudaGetLastError(), "stride_read_write_kernel launch");
                                });
                                if (row.stats.success && !validate_stride_written_bytes(dst_device, size_bytes, stride_bytes, true)) {
                                    row.stats.success = false;
                                    row.stats.error = "stride_read_write validation failed";
                                    validation_passed = false;
                                }
                                break;
                            default:
                                break;
                        }
                    } catch (const std::exception& ex) {
                        row.stats.error = ex.what();
                        validation_passed = false;
                    }

                    if (!row.stats.success && row.stats.error.empty()) {
                        validation_passed = false;
                    }
                    rows.push_back(row);
                }
            }

            const size_t element_count = size_bytes / sizeof(uint32_t);
            const uint64_t mask = static_cast<uint64_t>(element_count - 1);
            const int block_count = choose_block_count(prop.multiProcessorCount, element_count);
            auto* src_words = reinterpret_cast<uint32_t*>(src_device);
            auto* dst_words = reinterpret_cast<uint32_t*>(dst_device);
            auto permutation = build_random_permutation(element_count);
            auto inverse_permutation = build_inverse_permutation(permutation);
            uint32_t* permutation_device = nullptr;
            bench::check_cuda(cudaMalloc(&permutation_device, size_bytes), "cudaMalloc(permutation_device)");
            bench::check_cuda(
                cudaMemcpy(permutation_device, permutation.data(), size_bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy(permutation_device)");

            for (PatternKind pattern : {PatternKind::Gather, PatternKind::Scatter, PatternKind::RandomBoth}) {
                CaseRow row;
                row.pattern = pattern;
                row.size_mb = size_mb;
                row.access_unit_bytes = sizeof(uint32_t);
                row.iterations = effective_iterations(size_mb, options.iterations);
                row.warmup = effective_warmup(size_mb, options.warmup);

                try {
                    bench::check_cuda(cudaMemset(dst_device, kDstSentinel, size_bytes), "cudaMemset(random dst)");

                    switch (pattern) {
                        case PatternKind::Gather:
                            row.stats = measure_timed_operation(size_bytes * 2ull, row.warmup, row.iterations, [&]() {
                                gather_kernel<<<block_count, kThreadsPerBlock>>>(
                                    src_words, dst_words, permutation_device, element_count);
                                bench::check_cuda(cudaGetLastError(), "gather_kernel launch");
                            });
                            break;
                        case PatternKind::Scatter:
                            row.stats = measure_timed_operation(size_bytes * 2ull, row.warmup, row.iterations, [&]() {
                                scatter_kernel<<<block_count, kThreadsPerBlock>>>(
                                    src_words, dst_words, permutation_device, element_count);
                                bench::check_cuda(cudaGetLastError(), "scatter_kernel launch");
                            });
                            break;
                        case PatternKind::RandomBoth:
                            row.stats = measure_timed_operation(size_bytes * 2ull, row.warmup, row.iterations, [&]() {
                                random_both_kernel<<<block_count, kThreadsPerBlock>>>(
                                    src_words, dst_words, permutation_device, element_count, mask);
                                bench::check_cuda(cudaGetLastError(), "random_both_kernel launch");
                            });
                            break;
                        default:
                            break;
                    }

                    if (row.stats.success &&
                        !validate_random_pattern(
                            dst_words, element_count, pattern, permutation, inverse_permutation)) {
                        row.stats.success = false;
                        row.stats.error = std::string(pattern_name(pattern)) + " validation failed";
                        validation_passed = false;
                    }
                } catch (const std::exception& ex) {
                    row.stats.error = ex.what();
                    validation_passed = false;
                }

                if (!row.stats.success && row.stats.error.empty()) {
                    validation_passed = false;
                }
                rows.push_back(row);
            }

            cudaFree(permutation_device);
        }

        cudaFree(partial_device);
        cudaFree(dst_device);
        cudaFree(src_device);

        bench::emit_json(render_json(options, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options, "random_both_1024mb_gib_per_s"));
        return 1;
    }
}
