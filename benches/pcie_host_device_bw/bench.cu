#include "bench_support.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct DirectionStats {
    bench::CopyStats pageable;
    bench::CopyStats pinned;
};

struct CaseRow {
    size_t size_mb = 0;
    DirectionStats h2d;
    DirectionStats d2h;
};

std::string render_json(const bench::Options& options, const std::vector<CaseRow>& rows, bool validation_passed) {
    double best_h2d_pinned = 0.0;
    double best_d2h_pinned = 0.0;
    double best_h2d_pageable = 0.0;
    double best_d2h_pageable = 0.0;
    double max_speedup = 0.0;

    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (i > 0) {
            cases << ",";
        }
        auto write_stats = [&](const DirectionStats& stats) {
            std::ostringstream oss;
            oss << "{"
                << "\"pageable\":{"
                << "\"success\":" << (stats.pageable.success ? "true" : "false") << ","
                << "\"avg_ms\":" << bench::format_double(stats.pageable.avg_ms) << ","
                << "\"gib_per_s\":" << bench::format_double(stats.pageable.gib_per_s);
            if (!stats.pageable.error.empty()) {
                oss << ",\"error\":" << bench::quote(stats.pageable.error);
            }
            oss << "},"
                << "\"pinned\":{"
                << "\"success\":" << (stats.pinned.success ? "true" : "false") << ","
                << "\"avg_ms\":" << bench::format_double(stats.pinned.avg_ms) << ","
                << "\"gib_per_s\":" << bench::format_double(stats.pinned.gib_per_s);
            if (!stats.pinned.error.empty()) {
                oss << ",\"error\":" << bench::quote(stats.pinned.error);
            }
            oss << "}";
            if (stats.pageable.success && stats.pinned.success) {
                const double speedup = stats.pinned.gib_per_s / stats.pageable.gib_per_s;
                oss << ",\"speedup\":" << bench::format_double(speedup);
            }
            oss << "}";
            return oss.str();
        };

        cases << "{"
              << "\"size_mb\":" << row.size_mb << ","
              << "\"h2d\":" << write_stats(row.h2d) << ","
              << "\"d2h\":" << write_stats(row.d2h)
              << "}";

        if (row.h2d.pageable.success && row.h2d.pinned.success) {
            best_h2d_pageable = std::max(best_h2d_pageable, row.h2d.pageable.gib_per_s);
            best_h2d_pinned = std::max(best_h2d_pinned, row.h2d.pinned.gib_per_s);
            max_speedup = std::max(max_speedup, row.h2d.pinned.gib_per_s / row.h2d.pageable.gib_per_s);
        }
        if (row.d2h.pageable.success && row.d2h.pinned.success) {
            best_d2h_pageable = std::max(best_d2h_pageable, row.d2h.pageable.gib_per_s);
            best_d2h_pinned = std::max(best_d2h_pinned, row.d2h.pinned.gib_per_s);
            max_speedup = std::max(max_speedup, row.d2h.pinned.gib_per_s / row.d2h.pageable.gib_per_s);
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.h2d.pageable.success && row.h2d.pinned.success && row.d2h.pageable.success &&
               row.d2h.pinned.success;
    });
    const std::string status = (all_ok && validation_passed) ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"best_h2d_pinned_bandwidth_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"directions\":[\"H2D\",\"D2H\"],"
        << "\"memory_types\":[\"pageable\",\"pinned\"],"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event\","
        << "\"best_h2d_pinned_bandwidth_gib_per_s\":" << bench::format_double(best_h2d_pinned) << ","
        << "\"best_h2d_pageable_bandwidth_gib_per_s\":" << bench::format_double(best_h2d_pageable) << ","
        << "\"best_d2h_pinned_bandwidth_gib_per_s\":" << bench::format_double(best_d2h_pinned) << ","
        << "\"best_d2h_pageable_bandwidth_gib_per_s\":" << bench::format_double(best_d2h_pageable) << ","
        << "\"max_speedup\":" << bench::format_double(max_speedup) << ","
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

        int device_count = 0;
        bench::check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            CaseRow row;
            row.size_mb = size_mb;

            void* source = std::malloc(bytes);
            void* pageable_dst = std::malloc(bytes);
            if (source == nullptr || pageable_dst == nullptr) {
                row.h2d.pageable.error = "host allocation failed";
                row.h2d.pinned.error = "host allocation failed";
                row.d2h.pageable.error = "host allocation failed";
                row.d2h.pinned.error = "host allocation failed";
                validation_passed = false;
                if (source != nullptr) {
                    std::free(source);
                }
                if (pageable_dst != nullptr) {
                    std::free(pageable_dst);
                }
                rows.push_back(row);
                continue;
            }

            unsigned char* source_bytes = static_cast<unsigned char*>(source);
            for (size_t i = 0; i < bytes; ++i) {
                source_bytes[i] = static_cast<unsigned char>((i * 13u + 7u) & 0xFFu);
            }

            void* pinned_src = nullptr;
            void* pinned_dst = nullptr;
            const cudaError_t pinned_src_status = cudaMallocHost(&pinned_src, bytes);
            const cudaError_t pinned_dst_status = cudaMallocHost(&pinned_dst, bytes);
            if (pinned_src_status == cudaSuccess) {
                std::memcpy(pinned_src, source, bytes);
            } else {
                const std::string message =
                    std::string("cudaMallocHost(src) failed: ") + cudaGetErrorString(pinned_src_status);
                row.h2d.pinned.error = message;
                validation_passed = false;
            }
            if (pinned_dst_status != cudaSuccess) {
                const std::string message =
                    std::string("cudaMallocHost(dst) failed: ") + cudaGetErrorString(pinned_dst_status);
                row.d2h.pinned.error = message;
                validation_passed = false;
            }

            void* device = nullptr;
            const cudaError_t device_status = cudaMalloc(&device, bytes);
            if (device_status != cudaSuccess) {
                const std::string message =
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(device_status);
                row.h2d.pageable.error = message;
                row.h2d.pinned.error = message;
                row.d2h.pageable.error = message;
                row.d2h.pinned.error = message;
                validation_passed = false;
            } else {
                row.h2d.pageable = bench::measure_memcpy(
                    device, source, bytes, cudaMemcpyHostToDevice, options.warmup, options.iterations);
                bench::check_cuda(cudaMemcpy(pageable_dst, device, bytes, cudaMemcpyDeviceToHost), "verify H2D pageable");
                if (std::memcmp(pageable_dst, source, bytes) != 0) {
                    validation_passed = false;
                }

                if (pinned_src != nullptr) {
                    row.h2d.pinned = bench::measure_memcpy(
                        device, pinned_src, bytes, cudaMemcpyHostToDevice, options.warmup, options.iterations);
                    bench::check_cuda(cudaMemcpy(pageable_dst, device, bytes, cudaMemcpyDeviceToHost), "verify H2D pinned");
                    if (std::memcmp(pageable_dst, pinned_src, bytes) != 0) {
                        validation_passed = false;
                    }
                }

                bench::check_cuda(cudaMemcpy(device, source, bytes, cudaMemcpyHostToDevice), "seed D2H pageable");
                row.d2h.pageable = bench::measure_memcpy(
                    pageable_dst, device, bytes, cudaMemcpyDeviceToHost, options.warmup, options.iterations);
                if (std::memcmp(pageable_dst, source, bytes) != 0) {
                    validation_passed = false;
                }

                if (pinned_dst != nullptr) {
                    bench::check_cuda(cudaMemcpy(device, source, bytes, cudaMemcpyHostToDevice), "seed D2H pinned");
                    row.d2h.pinned = bench::measure_memcpy(
                        pinned_dst, device, bytes, cudaMemcpyDeviceToHost, options.warmup, options.iterations);
                    if (std::memcmp(pinned_dst, source, bytes) != 0) {
                        validation_passed = false;
                    }
                }
            }

            if (device != nullptr) {
                cudaFree(device);
            }
            if (pinned_src != nullptr) {
                cudaFreeHost(pinned_src);
            }
            if (pinned_dst != nullptr) {
                cudaFreeHost(pinned_dst);
            }
            std::free(source);
            std::free(pageable_dst);
            rows.push_back(row);
        }

        bench::emit_json(render_json(options, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options));
        return 1;
    }
}
