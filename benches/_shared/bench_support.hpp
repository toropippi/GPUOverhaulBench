#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

struct Options {
    int iterations = 50;
    int warmup = 5;
    std::vector<size_t> sizes_mb = {8, 32, 128, 512, 1024};
};

struct CopyStats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
};

inline void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::ostringstream oss;
        oss << context << ": " << cudaGetErrorString(status);
        throw std::runtime_error(oss.str());
    }
}

inline bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

inline std::vector<size_t> parse_sizes_mb(const std::string& text) {
    std::vector<size_t> sizes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(item.c_str(), &end, 10);
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

inline Options parse_common_args(int argc, char** argv) {
    Options options;
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
        } else if (starts_with(arg, "--sizes_mb")) {
            options.sizes_mb = parse_sizes_mb(get_value("--sizes_mb"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench.exe [--iterations N] [--warmup N] [--sizes_mb A,B,C]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.iterations <= 0 || options.warmup <= 0) {
        throw std::runtime_error("iterations and warmup must be > 0");
    }
    return options;
}

inline std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (char c : value) {
        switch (c) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                oss << c;
                break;
        }
    }
    return oss.str();
}

inline std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

inline std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

inline std::string sizes_to_json(const std::vector<size_t>& sizes_mb) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes_mb.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << sizes_mb[i];
    }
    oss << "]";
    return oss.str();
}

inline CopyStats measure_memcpy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    int warmup,
    int iterations) {
    CopyStats stats;
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
        status = cudaMemcpy(dst, src, bytes, kind);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaMemcpy failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
    }

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
        check_cuda(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
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

inline std::string make_error_json(
    const std::string& status,
    const std::string& message,
    const Options& options,
    const std::string& primary_metric = "none") {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(status) << ","
        << "\"primary_metric\":" << quote(primary_metric) << ","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
}

inline void emit_json(const std::string& json) {
    std::cout << json << "\n";
}

}  // namespace bench
