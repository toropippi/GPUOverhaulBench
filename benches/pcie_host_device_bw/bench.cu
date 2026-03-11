#include "bench_support.hpp"

#include <algorithm>
#include <array>
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
    int iterations = 0;
    int warmup = 0;
    DirectionStats h2d;
    DirectionStats d2h;
};

constexpr std::array<size_t, 8> kExtendedSizesMb = {8, 32, 128, 512, 1024, 2048, 4096, 8192};

bool is_default_size_list(const std::vector<size_t>& sizes_mb) {
    const std::vector<size_t> default_sizes = {8, 32, 128, 512, 1024};
    return sizes_mb == default_sizes;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb <= 128) {
        return requested_iterations;
    }
    if (size_mb <= 512) {
        return std::min(requested_iterations, 25);
    }
    if (size_mb <= 1024) {
        return std::min(requested_iterations, 15);
    }
    if (size_mb <= 2048) {
        return std::min(requested_iterations, 8);
    }
    if (size_mb <= 4096) {
        return std::min(requested_iterations, 4);
    }
    return std::min(requested_iterations, 2);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb <= 1024) {
        return requested_warmup;
    }
    if (size_mb <= 4096) {
        return std::min(requested_warmup, 2);
    }
    return 1;
}

double gbps_decimal_to_gibps(double gbps_decimal) {
    return gbps_decimal * 1000000000.0 / (1024.0 * 1024.0 * 1024.0);
}

std::string render_stats_json(const DirectionStats& stats) {
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
        oss << ",\"speedup\":" << bench::format_double(stats.pinned.gib_per_s / stats.pageable.gib_per_s);
    }
    oss << "}";
    return oss.str();
}

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
        cases << "{"
              << "\"size_mb\":" << row.size_mb << ","
              << "\"iterations\":" << row.iterations << ","
              << "\"warmup\":" << row.warmup << ","
              << "\"h2d\":" << render_stats_json(row.h2d) << ","
              << "\"d2h\":" << render_stats_json(row.d2h)
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

    const double theoretical_gib_per_s = gbps_decimal_to_gibps(63.015);
    const double h2d_utilization = best_h2d_pinned / theoretical_gib_per_s;
    const double d2h_utilization = best_d2h_pinned / theoretical_gib_per_s;

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
        << "\"adaptive_iteration_schedule\":true,"
        << "\"best_h2d_pinned_bandwidth_gib_per_s\":" << bench::format_double(best_h2d_pinned) << ","
        << "\"best_h2d_pageable_bandwidth_gib_per_s\":" << bench::format_double(best_h2d_pageable) << ","
        << "\"best_d2h_pinned_bandwidth_gib_per_s\":" << bench::format_double(best_d2h_pinned) << ","
        << "\"best_d2h_pageable_bandwidth_gib_per_s\":" << bench::format_double(best_d2h_pageable) << ","
        << "\"theoretical_one_way_pcie_5_x16_gib_per_s\":" << bench::format_double(theoretical_gib_per_s) << ","
        << "\"h2d_pinned_utilization_of_theoretical\":" << bench::format_double(h2d_utilization) << ","
        << "\"d2h_pinned_utilization_of_theoretical\":" << bench::format_double(d2h_utilization) << ","
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
        if (is_default_size_list(options.sizes_mb)) {
            options.sizes_mb.assign(kExtendedSizesMb.begin(), kExtendedSizesMb.end());
        }

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
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

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
                rows.push_back(row);
                continue;
            }

            void* pageable_source = std::malloc(bytes);
            if (pageable_source == nullptr) {
                const std::string message = "pageable source allocation failed";
                row.h2d.pageable.error = message;
                row.h2d.pinned.error = message;
                row.d2h.pageable.error = message;
                row.d2h.pinned.error = message;
                validation_passed = false;
                cudaFree(device);
                rows.push_back(row);
                continue;
            }
            std::memset(pageable_source, 0x5A, bytes);

            void* verify_buffer = std::malloc(bytes);
            if (verify_buffer == nullptr) {
                const std::string message = "verify buffer allocation failed";
                row.h2d.pageable.error = message;
                row.h2d.pinned.error = message;
                row.d2h.pageable.error = message;
                row.d2h.pinned.error = message;
                validation_passed = false;
                std::free(pageable_source);
                cudaFree(device);
                rows.push_back(row);
                continue;
            }

            row.h2d.pageable = bench::measure_memcpy(
                device, pageable_source, bytes, cudaMemcpyHostToDevice, row.warmup, row.iterations);
            bench::check_cuda(cudaMemcpy(verify_buffer, device, bytes, cudaMemcpyDeviceToHost), "verify H2D pageable");
            if (std::memcmp(verify_buffer, pageable_source, bytes) != 0) {
                validation_passed = false;
            }
            std::free(verify_buffer);

            void* pinned_source = nullptr;
            const cudaError_t pinned_source_status = cudaMallocHost(&pinned_source, bytes);
            if (pinned_source_status == cudaSuccess) {
                std::memset(pinned_source, 0x5A, bytes);
                verify_buffer = std::malloc(bytes);
                if (verify_buffer == nullptr) {
                    row.h2d.pinned.error = "verify buffer allocation failed";
                    validation_passed = false;
                } else {
                    row.h2d.pinned = bench::measure_memcpy(
                        device, pinned_source, bytes, cudaMemcpyHostToDevice, row.warmup, row.iterations);
                    bench::check_cuda(
                        cudaMemcpy(verify_buffer, device, bytes, cudaMemcpyDeviceToHost), "verify H2D pinned");
                    if (std::memcmp(verify_buffer, pinned_source, bytes) != 0) {
                        validation_passed = false;
                    }
                    std::free(verify_buffer);
                }
            } else {
                row.h2d.pinned.error =
                    std::string("cudaMallocHost(src) failed: ") + cudaGetErrorString(pinned_source_status);
                validation_passed = false;
            }

            if (pinned_source != nullptr) {
                bench::check_cuda(cudaMemcpy(device, pinned_source, bytes, cudaMemcpyHostToDevice), "seed D2H");

                void* pageable_dest = std::malloc(bytes);
                if (pageable_dest == nullptr) {
                    row.d2h.pageable.error = "pageable destination allocation failed";
                    validation_passed = false;
                } else {
                    row.d2h.pageable = bench::measure_memcpy(
                        pageable_dest, device, bytes, cudaMemcpyDeviceToHost, row.warmup, row.iterations);
                    if (std::memcmp(pageable_dest, pinned_source, bytes) != 0) {
                        validation_passed = false;
                    }
                    std::free(pageable_dest);
                }

                void* pinned_dest = nullptr;
                const cudaError_t pinned_dest_status = cudaMallocHost(&pinned_dest, bytes);
                if (pinned_dest_status != cudaSuccess) {
                    row.d2h.pinned.error =
                        std::string("cudaMallocHost(dst) failed: ") + cudaGetErrorString(pinned_dest_status);
                    validation_passed = false;
                } else {
                    bench::check_cuda(cudaMemcpy(device, pinned_source, bytes, cudaMemcpyHostToDevice), "seed D2H pinned");
                    row.d2h.pinned = bench::measure_memcpy(
                        pinned_dest, device, bytes, cudaMemcpyDeviceToHost, row.warmup, row.iterations);
                    if (std::memcmp(pinned_dest, pinned_source, bytes) != 0) {
                        validation_passed = false;
                    }
                    cudaFreeHost(pinned_dest);
                }

                cudaFreeHost(pinned_source);
            } else {
                row.d2h.pageable.error = "skipped because pinned source allocation failed";
                row.d2h.pinned.error = "skipped because pinned source allocation failed";
            }

            std::free(pageable_source);
            cudaFree(device);
            rows.push_back(row);
        }

        bench::emit_json(render_json(options, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options));
        return 1;
    }
}
