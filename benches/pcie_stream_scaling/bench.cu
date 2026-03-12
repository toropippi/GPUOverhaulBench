#include "bench_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int iterations = 10;
    int warmup = 2;
    double target_batch_ms = 1.0;
    std::vector<size_t> sizes_bytes;
    std::vector<int> stream_counts = {1, 2, 4, 8, 16};
};

struct DirectionStats {
    bool success = false;
    std::string error;
    double avg_round_us = 0.0;
    double batch_avg_ms = 0.0;
    double aggregate_gib_per_s = 0.0;
    double scaling_vs_single_stream = 0.0;
};

struct CaseRow {
    size_t size_bytes = 0;
    int stream_count = 0;
    int iterations = 0;
    int warmup = 0;
    int inner_repeats = 0;
    DirectionStats h2d;
    DirectionStats d2h;
};

struct Buffers {
    std::vector<void*> host_src;
    std::vector<void*> host_dst;
    std::vector<void*> device;
    std::vector<cudaStream_t> streams;
};

struct CopyEndpoints {
    void* dst = nullptr;
    const void* src = nullptr;
};

constexpr double kReferenceBandwidthGiBPerS = 52.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kMaxInnerRepeats = 100000;

std::vector<size_t> default_sizes_bytes() {
    std::vector<size_t> sizes;
    for (size_t bytes = 4; bytes <= (1ull << 20); bytes <<= 1) {
        sizes.push_back(bytes);
    }
    return sizes;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::vector<size_t> parse_sizes_bytes(const std::string& text) {
    std::vector<size_t> sizes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed == 0) {
            throw std::runtime_error("Invalid size list: " + text);
        }
        sizes.push_back(static_cast<size_t>(parsed));
    }
    if (sizes.empty()) {
        throw std::runtime_error("No sizes provided");
    }
    return sizes;
}

std::vector<int> parse_stream_counts(const std::string& text) {
    std::vector<int> counts;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        const long parsed = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed <= 0) {
            throw std::runtime_error("Invalid stream count list: " + text);
        }
        counts.push_back(static_cast<int>(parsed));
    }
    if (counts.empty()) {
        throw std::runtime_error("No stream counts provided");
    }
    return counts;
}

Options parse_args(int argc, char** argv) {
    Options options;
    options.sizes_bytes = default_sizes_bytes();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto get_value = [&](const std::string& flag) -> std::string {
            if (starts_with(arg, flag + "=")) {
                return arg.substr(flag.size() + 1);
            }
            if (arg == flag && i + 1 < argc) {
                return argv[++i];
            }
            throw std::runtime_error("Missing value for " + flag);
        };

        if (starts_with(arg, "--iterations")) {
            options.iterations = std::stoi(get_value("--iterations"));
        } else if (starts_with(arg, "--warmup")) {
            options.warmup = std::stoi(get_value("--warmup"));
        } else if (starts_with(arg, "--target_batch_ms")) {
            options.target_batch_ms = std::stod(get_value("--target_batch_ms"));
        } else if (starts_with(arg, "--sizes_bytes")) {
            options.sizes_bytes = parse_sizes_bytes(get_value("--sizes_bytes"));
        } else if (starts_with(arg, "--stream_counts")) {
            options.stream_counts = parse_stream_counts(get_value("--stream_counts"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench.exe [--iterations N] [--warmup N] [--target_batch_ms X] [--sizes_bytes A,B,C] [--stream_counts A,B,C]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.iterations <= 0 || options.warmup <= 0) {
        throw std::runtime_error("iterations and warmup must be > 0");
    }
    if (!(options.target_batch_ms > 0.0)) {
        throw std::runtime_error("target_batch_ms must be > 0");
    }
    return options;
}

std::string sizes_bytes_to_json(const std::vector<size_t>& sizes_bytes) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes_bytes.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << sizes_bytes[i];
    }
    oss << "]";
    return oss.str();
}

std::string stream_counts_to_json(const std::vector<int>& counts) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < counts.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << counts[i];
    }
    oss << "]";
    return oss.str();
}

int estimate_inner_repeats(size_t bytes, int stream_count, double target_batch_ms) {
    const double per_round_ms = std::max(
        0.003,
        (static_cast<double>(bytes) * static_cast<double>(stream_count) / (kReferenceBandwidthGiBPerS * kBytesPerGiB)) * 1000.0);
    const int repeats = static_cast<int>(std::ceil(target_batch_ms / per_round_ms));
    return std::clamp(repeats, 1, kMaxInnerRepeats);
}

CopyEndpoints select_endpoints(Buffers& buffers, size_t index, cudaMemcpyKind kind) {
    if (kind == cudaMemcpyHostToDevice) {
        return CopyEndpoints{buffers.device[index], buffers.host_src[index]};
    }
    return CopyEndpoints{buffers.host_dst[index], buffers.device[index]};
}

void synchronize_all(const Buffers& buffers, const char* context) {
    for (auto stream : buffers.streams) {
        bench::check_cuda(cudaStreamSynchronize(stream), context);
    }
}

void run_copy_rounds(Buffers& buffers, size_t bytes, cudaMemcpyKind kind, int inner_repeats, const char* context) {
    for (int repeat = 0; repeat < inner_repeats; ++repeat) {
        for (size_t i = 0; i < buffers.streams.size(); ++i) {
            const auto endpoints = select_endpoints(buffers, i, kind);
            bench::check_cuda(cudaMemcpyAsync(endpoints.dst, endpoints.src, bytes, kind, buffers.streams[i]), context);
        }
    }
}

Buffers allocate_buffers(int stream_count, size_t bytes) {
    Buffers buffers;
    buffers.host_src.resize(stream_count, nullptr);
    buffers.host_dst.resize(stream_count, nullptr);
    buffers.device.resize(stream_count, nullptr);
    buffers.streams.resize(stream_count, nullptr);

    for (int i = 0; i < stream_count; ++i) {
        bench::check_cuda(cudaMallocHost(&buffers.host_src[i], bytes), "cudaMallocHost(host_src)");
        bench::check_cuda(cudaMallocHost(&buffers.host_dst[i], bytes), "cudaMallocHost(host_dst)");
        bench::check_cuda(cudaMalloc(&buffers.device[i], bytes), "cudaMalloc(device)");
        bench::check_cuda(cudaStreamCreate(&buffers.streams[i]), "cudaStreamCreate");

        std::memset(buffers.host_src[i], 0x10 + (i & 0x3F), bytes);
        std::memset(buffers.host_dst[i], 0x00, bytes);
    }
    return buffers;
}

void free_buffers(Buffers& buffers) {
    for (auto stream : buffers.streams) {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    for (auto ptr : buffers.device) {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
    for (auto ptr : buffers.host_dst) {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
        }
    }
    for (auto ptr : buffers.host_src) {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
        }
    }
}

DirectionStats measure_direction(
    Buffers& buffers,
    size_t bytes,
    cudaMemcpyKind kind,
    int warmup,
    int iterations,
    int inner_repeats) {
    DirectionStats stats;

    for (int iter = 0; iter < warmup; ++iter) {
        run_copy_rounds(buffers, bytes, kind, inner_repeats, "warmup cudaMemcpyAsync");
        synchronize_all(buffers, "warmup cudaStreamSynchronize");
    }

    double total_batch_ms = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        const auto wall_start = std::chrono::steady_clock::now();
        run_copy_rounds(buffers, bytes, kind, inner_repeats, "cudaMemcpyAsync");
        synchronize_all(buffers, "cudaStreamSynchronize");
        const auto wall_end = std::chrono::steady_clock::now();
        total_batch_ms += std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    }

    stats.success = true;
    stats.batch_avg_ms = total_batch_ms / static_cast<double>(iterations);
    stats.avg_round_us = (stats.batch_avg_ms * 1000.0) / static_cast<double>(inner_repeats);
    const double total_gib_per_round =
        (static_cast<double>(bytes) * static_cast<double>(buffers.streams.size())) / kBytesPerGiB;
    stats.aggregate_gib_per_s = total_gib_per_round / (stats.avg_round_us / 1000000.0);
    return stats;
}

bool validate_h2d(Buffers& buffers, size_t bytes) {
    std::vector<unsigned char> verify(bytes);
    for (size_t i = 0; i < buffers.streams.size(); ++i) {
        bench::check_cuda(cudaMemcpy(verify.data(), buffers.device[i], bytes, cudaMemcpyDeviceToHost), "validate H2D");
        if (std::memcmp(verify.data(), buffers.host_src[i], bytes) != 0) {
            return false;
        }
    }
    return true;
}

bool validate_d2h(Buffers& buffers, size_t bytes) {
    for (size_t i = 0; i < buffers.streams.size(); ++i) {
        if (std::memcmp(buffers.host_dst[i], buffers.host_src[i], bytes) != 0) {
            return false;
        }
    }
    return true;
}

void seed_device_from_host(Buffers& buffers, size_t bytes) {
    for (size_t i = 0; i < buffers.device.size(); ++i) {
        bench::check_cuda(
            cudaMemcpy(buffers.device[i], buffers.host_src[i], bytes, cudaMemcpyHostToDevice),
            "seed D2H");
    }
}

void mark_validation_failure(DirectionStats& stats, const char* message, bool& validation_passed) {
    if (stats.success) {
        stats.success = false;
        stats.error = message;
    }
    validation_passed = false;
}

CaseRow run_case(const Options& options, size_t size_bytes, int stream_count, bool& validation_passed) {
    CaseRow row;
    row.size_bytes = size_bytes;
    row.stream_count = stream_count;
    row.iterations = options.iterations;
    row.warmup = options.warmup;
    row.inner_repeats = estimate_inner_repeats(size_bytes, stream_count, options.target_batch_ms);

    Buffers buffers{};
    try {
        buffers = allocate_buffers(stream_count, size_bytes);

        row.h2d = measure_direction(
            buffers, size_bytes, cudaMemcpyHostToDevice, row.warmup, row.iterations, row.inner_repeats);
        if (!row.h2d.success || !validate_h2d(buffers, size_bytes)) {
            mark_validation_failure(row.h2d, "H2D validation failed", validation_passed);
        }

        seed_device_from_host(buffers, size_bytes);
        row.d2h = measure_direction(
            buffers, size_bytes, cudaMemcpyDeviceToHost, row.warmup, row.iterations, row.inner_repeats);
        if (!row.d2h.success || !validate_d2h(buffers, size_bytes)) {
            mark_validation_failure(row.d2h, "D2H validation failed", validation_passed);
        }
    } catch (const std::exception& ex) {
        row.h2d.error = ex.what();
        row.d2h.error = ex.what();
        validation_passed = false;
    }

    free_buffers(buffers);
    return row;
}

void apply_scaling_vs_single_stream(std::vector<CaseRow>& rows) {
    std::map<size_t, double> h2d_baseline;
    std::map<size_t, double> d2h_baseline;
    for (const auto& row : rows) {
        if (row.stream_count != 1) {
            continue;
        }
        if (row.h2d.success) {
            h2d_baseline[row.size_bytes] = row.h2d.aggregate_gib_per_s;
        }
        if (row.d2h.success) {
            d2h_baseline[row.size_bytes] = row.d2h.aggregate_gib_per_s;
        }
    }

    for (auto& row : rows) {
        const auto h2d_it = h2d_baseline.find(row.size_bytes);
        if (row.h2d.success && h2d_it != h2d_baseline.end()) {
            row.h2d.scaling_vs_single_stream = row.h2d.aggregate_gib_per_s / h2d_it->second;
        }

        const auto d2h_it = d2h_baseline.find(row.size_bytes);
        if (row.d2h.success && d2h_it != d2h_baseline.end()) {
            row.d2h.scaling_vs_single_stream = row.d2h.aggregate_gib_per_s / d2h_it->second;
        }
    }
}

std::string render_direction_json(const DirectionStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false");
    if (stats.success) {
        oss << ",\"avg_round_us\":" << bench::format_double(stats.avg_round_us)
            << ",\"batch_avg_ms\":" << bench::format_double(stats.batch_avg_ms)
            << ",\"aggregate_gib_per_s\":" << bench::format_double(stats.aggregate_gib_per_s)
            << ",\"scaling_vs_single_stream\":" << bench::format_double(stats.scaling_vs_single_stream);
    }
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(
    const Options& options,
    int async_engine_count,
    int device_overlap,
    const std::vector<CaseRow>& rows,
    bool validation_passed) {
    double best_h2d_aggregate = 0.0;
    double best_d2h_aggregate = 0.0;
    double best_h2d_scaling = 0.0;
    double best_d2h_scaling = 0.0;

    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0) {
            cases << ",";
        }
        cases << "{"
              << "\"size_bytes\":" << row.size_bytes << ","
              << "\"stream_count\":" << row.stream_count << ","
              << "\"iterations\":" << row.iterations << ","
              << "\"warmup\":" << row.warmup << ","
              << "\"inner_repeats\":" << row.inner_repeats << ","
              << "\"h2d\":" << render_direction_json(row.h2d) << ","
              << "\"d2h\":" << render_direction_json(row.d2h)
              << "}";

        if (row.h2d.success) {
            best_h2d_aggregate = std::max(best_h2d_aggregate, row.h2d.aggregate_gib_per_s);
            best_h2d_scaling = std::max(best_h2d_scaling, row.h2d.scaling_vs_single_stream);
        }
        if (row.d2h.success) {
            best_d2h_aggregate = std::max(best_d2h_aggregate, row.d2h.aggregate_gib_per_s);
            best_d2h_scaling = std::max(best_d2h_scaling, row.d2h.scaling_vs_single_stream);
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.h2d.success && row.d2h.success;
    });

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"best_h2d_scaling_factor\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"memory_type\":\"pinned\","
        << "\"directions\":[\"H2D\",\"D2H\"],"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"target_batch_ms\":" << bench::format_double(options.target_batch_ms) << ","
        << "\"sizes_bytes\":" << sizes_bytes_to_json(options.sizes_bytes) << ","
        << "\"stream_counts\":" << stream_counts_to_json(options.stream_counts)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock_batched\","
        << "\"async_engine_count\":" << async_engine_count << ","
        << "\"device_overlap\":" << device_overlap << ","
        << "\"best_h2d_aggregate_bandwidth_gib_per_s\":" << bench::format_double(best_h2d_aggregate) << ","
        << "\"best_d2h_aggregate_bandwidth_gib_per_s\":" << bench::format_double(best_d2h_aggregate) << ","
        << "\"best_h2d_scaling_factor\":" << bench::format_double(best_h2d_scaling) << ","
        << "\"best_d2h_scaling_factor\":" << bench::format_double(best_d2h_scaling) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "},"
        << "\"notes\":["
        << bench::quote("This bench measures small-transfer pinned-copy scaling versus stream count.") << ","
        << bench::quote("Tiny transfers are batched so avg_round_us represents one logical round of stream_count simultaneous copies.") << ","
        << bench::quote("Scaling factors are computed against the same-size single-stream baseline.")
        << "]"
        << "}";
    return oss.str();
}

std::string make_error_json(const Options& options, const std::string& message) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"best_h2d_scaling_factor\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"memory_type\":\"pinned\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"target_batch_ms\":" << bench::format_double(options.target_batch_ms) << ","
        << "\"sizes_bytes\":" << sizes_bytes_to_json(options.sizes_bytes) << ","
        << "\"stream_counts\":" << stream_counts_to_json(options.stream_counts)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock_batched\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << bench::quote(message) << "]"
        << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    try {
        options = parse_args(argc, argv);

        int device_count = 0;
        bench::check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            bench::emit_json(make_error_json(options, "No CUDA device found"));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        cudaDeviceProp prop{};
        bench::check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_bytes : options.sizes_bytes) {
            for (int stream_count : options.stream_counts) {
                rows.push_back(run_case(options, size_bytes, stream_count, validation_passed));
            }
        }

        apply_scaling_vs_single_stream(rows);

        bench::emit_json(render_json(options, prop.asyncEngineCount, prop.deviceOverlap, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(make_error_json(options, ex.what()));
        return 1;
    }
}
