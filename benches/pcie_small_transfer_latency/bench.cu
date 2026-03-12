#include "bench_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    int iterations = 20;
    int warmup = 3;
    double target_batch_ms = 10.0;
    std::vector<size_t> sizes_bytes;
};

struct DirectionStats {
    bool success = false;
    std::string error;
    double avg_copy_us = 0.0;
    double batch_avg_ms = 0.0;
    double gib_per_s = 0.0;
    double predicted_copy_us = 0.0;
    double residual_us = 0.0;
};

struct CaseRow {
    size_t size_bytes = 0;
    int iterations = 0;
    int warmup = 0;
    int inner_repeats = 0;
    DirectionStats h2d;
    DirectionStats d2h;
};

struct FitResult {
    bool success = false;
    double fixed_latency_us = 0.0;
    double slope_us_per_byte = 0.0;
    double fitted_bandwidth_bytes_per_us = 0.0;
    double fitted_bandwidth_gib_per_s = 0.0;
    double r2 = 0.0;
};

constexpr double kReferenceBandwidthGiBPerS = 52.0;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr int kMaxInnerRepeats = 100000;

std::vector<size_t> default_sizes_bytes() {
    std::vector<size_t> sizes;
    for (size_t bytes = 4; bytes <= (1ull << 30); bytes <<= 1) {
        sizes.push_back(bytes);
    }
    return sizes;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string quote(const std::string& value) {
    return bench::quote(value);
}

std::string format_double(double value, int precision = 6) {
    return bench::format_double(value, precision);
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench.exe [--iterations N] [--warmup N] [--target_batch_ms X] [--sizes_bytes A,B,C]\n";
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

int estimate_inner_repeats(size_t bytes, double target_batch_ms) {
    const double estimated_ms =
        std::max(0.003, (static_cast<double>(bytes) / (kReferenceBandwidthGiBPerS * kBytesPerGiB)) * 1000.0);
    const double estimated_repeats = std::ceil(target_batch_ms / estimated_ms);
    const int repeats = static_cast<int>(estimated_repeats);
    return std::clamp(repeats, 1, kMaxInnerRepeats);
}

DirectionStats measure_batched_copy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    int warmup,
    int iterations,
    int inner_repeats) {
    DirectionStats stats;
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
        for (int j = 0; j < inner_repeats; ++j) {
            status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
            if (status != cudaSuccess) {
                stats.error = std::string("Warmup cudaMemcpyAsync failed: ") + cudaGetErrorString(status);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return stats;
            }
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
    }

    double total_batch_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        bench::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        for (int j = 0; j < inner_repeats; ++j) {
            bench::check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
        }
        bench::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        bench::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        total_batch_ms += static_cast<double>(elapsed_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    stats.success = true;
    stats.batch_avg_ms = total_batch_ms / static_cast<double>(iterations);
    stats.avg_copy_us = (stats.batch_avg_ms * 1000.0) / static_cast<double>(inner_repeats);
    const double gib = static_cast<double>(bytes) / kBytesPerGiB;
    stats.gib_per_s = gib / (stats.avg_copy_us / 1000000.0);
    return stats;
}

bool validate_h2d(const void* host_source, void* device, void* host_verify, size_t bytes) {
    bench::check_cuda(cudaMemcpy(device, host_source, bytes, cudaMemcpyHostToDevice), "validate H2D copy");
    bench::check_cuda(cudaMemcpy(host_verify, device, bytes, cudaMemcpyDeviceToHost), "validate H2D readback");
    return std::memcmp(host_source, host_verify, bytes) == 0;
}

bool validate_d2h(const void* host_source, void* device, void* host_dest, size_t bytes) {
    bench::check_cuda(cudaMemcpy(device, host_source, bytes, cudaMemcpyHostToDevice), "seed D2H");
    bench::check_cuda(cudaMemcpy(host_dest, device, bytes, cudaMemcpyDeviceToHost), "validate D2H copy");
    return std::memcmp(host_source, host_dest, bytes) == 0;
}

FitResult fit_direction(std::vector<CaseRow>& rows, bool h2d) {
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(rows.size());
    y.reserve(rows.size());

    for (const auto& row : rows) {
        const auto& stats = h2d ? row.h2d : row.d2h;
        if (!stats.success) {
            continue;
        }
        x.push_back(static_cast<double>(row.size_bytes));
        y.push_back(stats.avg_copy_us);
    }

    FitResult fit;
    if (x.size() < 2) {
        return fit;
    }

    const double mean_x = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(x.size());
    const double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());

    double ss_xx = 0.0;
    double ss_xy = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i] - mean_x;
        const double dy = y[i] - mean_y;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
    }
    if (!(ss_xx > 0.0)) {
        return fit;
    }

    fit.slope_us_per_byte = ss_xy / ss_xx;
    fit.fixed_latency_us = mean_y - fit.slope_us_per_byte * mean_x;

    if (!(fit.slope_us_per_byte > 0.0) || !std::isfinite(fit.slope_us_per_byte) || !std::isfinite(fit.fixed_latency_us)) {
        return FitResult{};
    }

    fit.fitted_bandwidth_bytes_per_us = 1.0 / fit.slope_us_per_byte;
    fit.fitted_bandwidth_gib_per_s = fit.fitted_bandwidth_bytes_per_us * 1000000.0 / kBytesPerGiB;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double predicted = fit.fixed_latency_us + fit.slope_us_per_byte * x[i];
        const double residual = y[i] - predicted;
        ss_res += residual * residual;
        const double total = y[i] - mean_y;
        ss_tot += total * total;
    }
    fit.r2 = (ss_tot > 0.0) ? (1.0 - ss_res / ss_tot) : 1.0;
    fit.success = std::isfinite(fit.fitted_bandwidth_gib_per_s) && std::isfinite(fit.r2);

    if (!fit.success) {
        return FitResult{};
    }

    for (auto& row : rows) {
        auto& stats = h2d ? row.h2d : row.d2h;
        if (!stats.success) {
            continue;
        }
        stats.predicted_copy_us = fit.fixed_latency_us + fit.slope_us_per_byte * static_cast<double>(row.size_bytes);
        stats.residual_us = stats.avg_copy_us - stats.predicted_copy_us;
    }

    return fit;
}

std::string render_direction_json(const DirectionStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false");
    if (stats.success) {
        oss << ",\"avg_copy_us\":" << format_double(stats.avg_copy_us)
            << ",\"batch_avg_ms\":" << format_double(stats.batch_avg_ms)
            << ",\"gib_per_s\":" << format_double(stats.gib_per_s)
            << ",\"predicted_copy_us\":" << format_double(stats.predicted_copy_us)
            << ",\"residual_us\":" << format_double(stats.residual_us);
    }
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_fit_json(const FitResult& fit) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (fit.success ? "true" : "false");
    if (fit.success) {
        oss << ",\"fixed_latency_us\":" << format_double(fit.fixed_latency_us)
            << ",\"slope_us_per_byte\":" << format_double(fit.slope_us_per_byte, 12)
            << ",\"fitted_bandwidth_bytes_per_us\":" << format_double(fit.fitted_bandwidth_bytes_per_us)
            << ",\"fitted_bandwidth_gib_per_s\":" << format_double(fit.fitted_bandwidth_gib_per_s)
            << ",\"r2\":" << format_double(fit.r2);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(
    const Options& options,
    const std::vector<CaseRow>& rows,
    const FitResult& h2d_fit,
    const FitResult& d2h_fit,
    bool validation_passed) {
    bool all_ok = validation_passed && h2d_fit.success && d2h_fit.success;
    for (const auto& row : rows) {
        all_ok = all_ok && row.h2d.success && row.d2h.success;
    }

    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0) {
            cases << ",";
        }
        cases << "{"
              << "\"size_bytes\":" << row.size_bytes << ","
              << "\"iterations\":" << row.iterations << ","
              << "\"warmup\":" << row.warmup << ","
              << "\"inner_repeats\":" << row.inner_repeats << ","
              << "\"h2d\":" << render_direction_json(row.h2d) << ","
              << "\"d2h\":" << render_direction_json(row.d2h)
              << "}";
    }
    cases << "]";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(all_ok ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"h2d_model.fixed_latency_us\","
        << "\"unit\":\"us\","
        << "\"parameters\":{"
        << "\"memory_type\":\"pinned\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"target_batch_ms\":" << format_double(options.target_batch_ms) << ","
        << "\"sizes_bytes\":" << sizes_bytes_to_json(options.sizes_bytes)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event_batched\","
        << "\"memory_type\":\"pinned\","
        << "\"model_form\":\"avg_copy_us = fixed_latency_us + bytes / fitted_bandwidth_bytes_per_us\","
        << "\"h2d_model\":" << render_fit_json(h2d_fit) << ","
        << "\"d2h_model\":" << render_fit_json(d2h_fit) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "},"
        << "\"notes\":["
        << quote("This bench approximates pinned CUDA copy time with an affine model.") << ","
        << quote("The fitted fixed latency is an effective per-copy fixed latency under this measurement method.") << ","
        << quote("Do not label the fitted fixed latency as raw PCIe packet-level latency.") 
        << "]"
        << "}";
    return oss.str();
}

std::string make_error_json(const Options& options, const std::string& message) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"h2d_model.fixed_latency_us\","
        << "\"unit\":\"us\","
        << "\"parameters\":{"
        << "\"memory_type\":\"pinned\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"target_batch_ms\":" << format_double(options.target_batch_ms) << ","
        << "\"sizes_bytes\":" << sizes_bytes_to_json(options.sizes_bytes)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event_batched\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << quote(message) << "]"
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

        const size_t max_bytes = *std::max_element(options.sizes_bytes.begin(), options.sizes_bytes.end());

        void* host_source = nullptr;
        void* host_dest = nullptr;
        void* host_verify = nullptr;
        void* device = nullptr;
        cudaStream_t stream = nullptr;

        bench::check_cuda(cudaMallocHost(&host_source, max_bytes), "cudaMallocHost(host_source)");
        bench::check_cuda(cudaMallocHost(&host_dest, max_bytes), "cudaMallocHost(host_dest)");
        host_verify = std::malloc(max_bytes);
        if (host_verify == nullptr) {
            throw std::runtime_error("malloc(host_verify) failed");
        }
        bench::check_cuda(cudaMalloc(&device, max_bytes), "cudaMalloc(device)");
        bench::check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        auto* source_bytes = static_cast<unsigned char*>(host_source);
        auto* dest_bytes = static_cast<unsigned char*>(host_dest);
        auto* verify_bytes = static_cast<unsigned char*>(host_verify);
        for (size_t i = 0; i < max_bytes; ++i) {
            source_bytes[i] = static_cast<unsigned char>((i * 131u + 17u) & 0xFFu);
            dest_bytes[i] = 0;
            verify_bytes[i] = 0;
        }

        std::vector<CaseRow> rows;
        rows.reserve(options.sizes_bytes.size());
        bool validation_passed = true;

        for (size_t bytes : options.sizes_bytes) {
            CaseRow row;
            row.size_bytes = bytes;
            row.iterations = options.iterations;
            row.warmup = options.warmup;
            row.inner_repeats = estimate_inner_repeats(bytes, options.target_batch_ms);

            try {
                row.h2d = measure_batched_copy(
                    device, host_source, bytes, cudaMemcpyHostToDevice, stream, row.warmup, row.iterations, row.inner_repeats);
                if (!row.h2d.success) {
                    validation_passed = false;
                } else if (!validate_h2d(host_source, device, host_verify, bytes)) {
                    row.h2d.success = false;
                    row.h2d.error = "H2D validation memcmp failed";
                    validation_passed = false;
                }

                bench::check_cuda(cudaMemcpy(device, host_source, bytes, cudaMemcpyHostToDevice), "seed D2H");
                row.d2h = measure_batched_copy(
                    host_dest, device, bytes, cudaMemcpyDeviceToHost, stream, row.warmup, row.iterations, row.inner_repeats);
                if (!row.d2h.success) {
                    validation_passed = false;
                } else if (!validate_d2h(host_source, device, host_dest, bytes)) {
                    row.d2h.success = false;
                    row.d2h.error = "D2H validation memcmp failed";
                    validation_passed = false;
                }
            } catch (const std::exception& ex) {
                if (row.h2d.error.empty()) {
                    row.h2d.error = ex.what();
                }
                if (row.d2h.error.empty()) {
                    row.d2h.error = ex.what();
                }
                validation_passed = false;
            }

            rows.push_back(row);
        }

        const FitResult h2d_fit = fit_direction(rows, true);
        const FitResult d2h_fit = fit_direction(rows, false);

        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (device != nullptr) {
            cudaFree(device);
        }
        if (host_verify != nullptr) {
            std::free(host_verify);
        }
        if (host_dest != nullptr) {
            cudaFreeHost(host_dest);
        }
        if (host_source != nullptr) {
            cudaFreeHost(host_source);
        }

        bench::emit_json(render_json(options, rows, h2d_fit, d2h_fit, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(make_error_json(options, ex.what()));
        return 1;
    }
}
