#include "bench_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct BidirStats {
    bool success = false;
    std::string error;
    double avg_wall_ms = 0.0;
    double h2d_stream_avg_ms = 0.0;
    double d2h_stream_avg_ms = 0.0;
    double combined_gib_per_s = 0.0;
    double wall_vs_solo_sum_ratio = 0.0;
    double wall_vs_solo_max_ratio = 0.0;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    bench::CopyStats h2d_solo;
    bench::CopyStats d2h_solo;
    BidirStats bidir;
};

constexpr std::array<size_t, 5> kDefaultSizesMb = {128, 512, 1024, 2048, 4096};

bool is_default_size_list(const std::vector<size_t>& sizes_mb) {
    const std::vector<size_t> shared_default_sizes = {8, 32, 128, 512, 1024};
    return sizes_mb == shared_default_sizes;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb <= 128) {
        return std::min(requested_iterations, 30);
    }
    if (size_mb <= 512) {
        return std::min(requested_iterations, 15);
    }
    if (size_mb <= 1024) {
        return std::min(requested_iterations, 6);
    }
    if (size_mb <= 2048) {
        return std::min(requested_iterations, 4);
    }
    return std::min(requested_iterations, 2);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb <= 512) {
        return std::min(requested_warmup, 3);
    }
    if (size_mb <= 2048) {
        return std::min(requested_warmup, 2);
    }
    return 1;
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

bench::CopyStats measure_async_copy(
    void* dst,
    const void* src,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream,
    int warmup,
    int iterations) {
    bench::CopyStats stats;
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
        status = cudaMemcpyAsync(dst, src, bytes, kind, stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaMemcpyAsync failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            stats.error = std::string("Warmup cudaStreamSynchronize failed: ") + cudaGetErrorString(status);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return stats;
        }
    }

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        bench::check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
        bench::check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
        bench::check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
        bench::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
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

BidirStats measure_bidirectional_pair(
    void* device_h2d_dst,
    const void* host_h2d_src,
    void* host_d2h_dst,
    const void* device_d2h_src,
    size_t bytes,
    cudaStream_t h2d_stream,
    cudaStream_t d2h_stream,
    int warmup,
    int iterations,
    const bench::CopyStats& h2d_solo,
    const bench::CopyStats& d2h_solo) {
    BidirStats stats;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    auto destroy_events = [&]() {
        if (h2d_start != nullptr) {
            cudaEventDestroy(h2d_start);
        }
        if (h2d_stop != nullptr) {
            cudaEventDestroy(h2d_stop);
        }
        if (d2h_start != nullptr) {
            cudaEventDestroy(d2h_start);
        }
        if (d2h_stop != nullptr) {
            cudaEventDestroy(d2h_stop);
        }
    };

    cudaError_t status = cudaEventCreate(&h2d_start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(h2d_start) failed: ") + cudaGetErrorString(status);
        return stats;
    }
    status = cudaEventCreate(&h2d_stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(h2d_stop) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }
    status = cudaEventCreate(&d2h_start);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(d2h_start) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }
    status = cudaEventCreate(&d2h_stop);
    if (status != cudaSuccess) {
        stats.error = std::string("cudaEventCreate(d2h_stop) failed: ") + cudaGetErrorString(status);
        destroy_events();
        return stats;
    }

    for (int i = 0; i < warmup; ++i) {
        bench::check_cuda(
            cudaMemcpyAsync(device_h2d_dst, host_h2d_src, bytes, cudaMemcpyHostToDevice, h2d_stream),
            "warmup H2D cudaMemcpyAsync");
        bench::check_cuda(
            cudaMemcpyAsync(host_d2h_dst, device_d2h_src, bytes, cudaMemcpyDeviceToHost, d2h_stream),
            "warmup D2H cudaMemcpyAsync");
        bench::check_cuda(cudaStreamSynchronize(h2d_stream), "warmup cudaStreamSynchronize(h2d_stream)");
        bench::check_cuda(cudaStreamSynchronize(d2h_stream), "warmup cudaStreamSynchronize(d2h_stream)");
    }

    double total_wall_ms = 0.0;
    float total_h2d_stream_ms = 0.0f;
    float total_d2h_stream_ms = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        const auto wall_start = std::chrono::steady_clock::now();
        bench::check_cuda(cudaEventRecord(h2d_start, h2d_stream), "cudaEventRecord(h2d_start)");
        bench::check_cuda(
            cudaMemcpyAsync(device_h2d_dst, host_h2d_src, bytes, cudaMemcpyHostToDevice, h2d_stream),
            "pair H2D cudaMemcpyAsync");
        bench::check_cuda(cudaEventRecord(h2d_stop, h2d_stream), "cudaEventRecord(h2d_stop)");

        bench::check_cuda(cudaEventRecord(d2h_start, d2h_stream), "cudaEventRecord(d2h_start)");
        bench::check_cuda(
            cudaMemcpyAsync(host_d2h_dst, device_d2h_src, bytes, cudaMemcpyDeviceToHost, d2h_stream),
            "pair D2H cudaMemcpyAsync");
        bench::check_cuda(cudaEventRecord(d2h_stop, d2h_stream), "cudaEventRecord(d2h_stop)");

        bench::check_cuda(cudaEventSynchronize(h2d_stop), "cudaEventSynchronize(h2d_stop)");
        bench::check_cuda(cudaEventSynchronize(d2h_stop), "cudaEventSynchronize(d2h_stop)");
        const auto wall_end = std::chrono::steady_clock::now();

        float h2d_stream_ms = 0.0f;
        float d2h_stream_ms = 0.0f;
        bench::check_cuda(cudaEventElapsedTime(&h2d_stream_ms, h2d_start, h2d_stop), "cudaEventElapsedTime(h2d)");
        bench::check_cuda(cudaEventElapsedTime(&d2h_stream_ms, d2h_start, d2h_stop), "cudaEventElapsedTime(d2h)");

        total_h2d_stream_ms += h2d_stream_ms;
        total_d2h_stream_ms += d2h_stream_ms;
        total_wall_ms +=
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    }

    destroy_events();

    stats.success = true;
    stats.avg_wall_ms = total_wall_ms / static_cast<double>(iterations);
    stats.h2d_stream_avg_ms = total_h2d_stream_ms / static_cast<double>(iterations);
    stats.d2h_stream_avg_ms = total_d2h_stream_ms / static_cast<double>(iterations);
    const double total_gib = 2.0 * static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.combined_gib_per_s = total_gib / (stats.avg_wall_ms / 1000.0);
    stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (h2d_solo.avg_ms + d2h_solo.avg_ms);
    stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(h2d_solo.avg_ms, d2h_solo.avg_ms);
    return stats;
}

std::string render_copy_stats_json(const bench::CopyStats& stats) {
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

std::string render_bidir_stats_json(const BidirStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_wall_ms\":" << bench::format_double(stats.avg_wall_ms) << ","
        << "\"h2d_stream_avg_ms\":" << bench::format_double(stats.h2d_stream_avg_ms) << ","
        << "\"d2h_stream_avg_ms\":" << bench::format_double(stats.d2h_stream_avg_ms) << ","
        << "\"combined_gib_per_s\":" << bench::format_double(stats.combined_gib_per_s) << ","
        << "\"wall_vs_solo_sum_ratio\":" << bench::format_double(stats.wall_vs_solo_sum_ratio) << ","
        << "\"wall_vs_solo_max_ratio\":" << bench::format_double(stats.wall_vs_solo_max_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << bench::quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(
    const bench::Options& options,
    int async_engine_count,
    int device_overlap,
    const std::vector<CaseRow>& rows,
    bool validation_passed) {
    double best_combined = 0.0;
    double min_wall_vs_sum = 0.0;
    double min_wall_vs_max = 0.0;
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
              << "\"h2d_solo\":" << render_copy_stats_json(row.h2d_solo) << ","
              << "\"d2h_solo\":" << render_copy_stats_json(row.d2h_solo) << ","
              << "\"bidir\":" << render_bidir_stats_json(row.bidir)
              << "}";

        if (row.h2d_solo.success && row.d2h_solo.success && row.bidir.success) {
            best_combined = std::max(best_combined, row.bidir.combined_gib_per_s);
            if (first_success) {
                min_wall_vs_sum = row.bidir.wall_vs_solo_sum_ratio;
                min_wall_vs_max = row.bidir.wall_vs_solo_max_ratio;
                first_success = false;
            } else {
                min_wall_vs_sum = std::min(min_wall_vs_sum, row.bidir.wall_vs_solo_sum_ratio);
                min_wall_vs_max = std::min(min_wall_vs_max, row.bidir.wall_vs_solo_max_ratio);
            }
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return row.h2d_solo.success && row.d2h_solo.success && row.bidir.success;
    });
    const std::string status = (all_ok && validation_passed) ? "ok" : "invalid";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << bench::quote(status) << ","
        << "\"primary_metric\":\"best_combined_bidirectional_bandwidth_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"memory_types\":[\"pinned\"],"
        << "\"directions\":[\"H2D\",\"D2H\"],"
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << bench::sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cuda_event_and_wall_clock\","
        << "\"async_engine_count\":" << async_engine_count << ","
        << "\"device_overlap\":" << device_overlap << ","
        << "\"best_combined_bidirectional_bandwidth_gib_per_s\":" << bench::format_double(best_combined) << ","
        << "\"min_wall_vs_solo_sum_ratio\":" << bench::format_double(min_wall_vs_sum) << ","
        << "\"min_wall_vs_solo_max_ratio\":" << bench::format_double(min_wall_vs_max) << ","
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
            bench::emit_json(bench::make_error_json("unsupported", "No CUDA device found", options));
            return 0;
        }

        bench::check_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp props{};
        bench::check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");

        cudaStream_t h2d_stream = nullptr;
        cudaStream_t d2h_stream = nullptr;
        bench::check_cuda(cudaStreamCreateWithFlags(&h2d_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(h2d)");
        bench::check_cuda(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags(d2h)");

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            void* device_h2d_dst = nullptr;
            void* device_d2h_src = nullptr;
            void* host_h2d_src = nullptr;
            void* host_d2h_dst = nullptr;
            void* verify_buffer = nullptr;

            auto fail_case = [&](const std::string& message) {
                row.h2d_solo.error = message;
                row.d2h_solo.error = message;
                row.bidir.error = message;
                validation_passed = false;
            };

            const cudaError_t device_h2d_status = cudaMalloc(&device_h2d_dst, bytes);
            if (device_h2d_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_h2d_dst) failed: ") + cudaGetErrorString(device_h2d_status));
                rows.push_back(row);
                continue;
            }

            const cudaError_t device_d2h_status = cudaMalloc(&device_d2h_src, bytes);
            if (device_d2h_status != cudaSuccess) {
                fail_case(std::string("cudaMalloc(device_d2h_src) failed: ") + cudaGetErrorString(device_d2h_status));
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t host_h2d_status = cudaMallocHost(&host_h2d_src, bytes);
            if (host_h2d_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(host_h2d_src) failed: ") + cudaGetErrorString(host_h2d_status));
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            const cudaError_t host_d2h_status = cudaMallocHost(&host_d2h_dst, bytes);
            if (host_d2h_status != cudaSuccess) {
                fail_case(std::string("cudaMallocHost(host_d2h_dst) failed: ") + cudaGetErrorString(host_d2h_status));
                cudaFreeHost(host_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            verify_buffer = std::malloc(bytes);
            if (verify_buffer == nullptr) {
                fail_case("verify buffer allocation failed");
                cudaFreeHost(host_d2h_dst);
                cudaFreeHost(host_h2d_src);
                cudaFree(device_d2h_src);
                cudaFree(device_h2d_dst);
                rows.push_back(row);
                continue;
            }

            std::memset(host_h2d_src, 0x3C, bytes);
            std::memset(host_d2h_dst, 0x00, bytes);
            bench::check_cuda(cudaMemset(device_d2h_src, 0xA5, bytes), "cudaMemset(device_d2h_src)");
            bench::check_cuda(cudaMemset(device_h2d_dst, 0x00, bytes), "cudaMemset(device_h2d_dst)");

            row.h2d_solo = measure_async_copy(
                device_h2d_dst,
                host_h2d_src,
                bytes,
                cudaMemcpyHostToDevice,
                h2d_stream,
                row.warmup,
                row.iterations);
            if (!row.h2d_solo.success) {
                validation_passed = false;
            } else {
                bench::check_cuda(
                    cudaMemcpy(verify_buffer, device_h2d_dst, bytes, cudaMemcpyDeviceToHost),
                    "verify H2D solo");
                if (std::memcmp(verify_buffer, host_h2d_src, bytes) != 0) {
                    validation_passed = false;
                }
            }

            row.d2h_solo = measure_async_copy(
                host_d2h_dst,
                device_d2h_src,
                bytes,
                cudaMemcpyDeviceToHost,
                d2h_stream,
                row.warmup,
                row.iterations);
            if (!row.d2h_solo.success) {
                validation_passed = false;
            } else if (!buffer_has_byte_value(host_d2h_dst, bytes, 0xA5)) {
                validation_passed = false;
            }

            if (row.h2d_solo.success && row.d2h_solo.success) {
                std::memset(host_d2h_dst, 0x00, bytes);
                bench::check_cuda(cudaMemset(device_h2d_dst, 0x00, bytes), "cudaMemset(device_h2d_dst reset)");
                row.bidir = measure_bidirectional_pair(
                    device_h2d_dst,
                    host_h2d_src,
                    host_d2h_dst,
                    device_d2h_src,
                    bytes,
                    h2d_stream,
                    d2h_stream,
                    row.warmup,
                    row.iterations,
                    row.h2d_solo,
                    row.d2h_solo);

                if (!row.bidir.success) {
                    validation_passed = false;
                } else {
                    if (!buffer_has_byte_value(host_d2h_dst, bytes, 0xA5)) {
                        validation_passed = false;
                    }
                    bench::check_cuda(
                        cudaMemcpy(verify_buffer, device_h2d_dst, bytes, cudaMemcpyDeviceToHost),
                        "verify bidir H2D");
                    if (std::memcmp(verify_buffer, host_h2d_src, bytes) != 0) {
                        validation_passed = false;
                    }
                }
            }

            std::free(verify_buffer);
            cudaFreeHost(host_d2h_dst);
            cudaFreeHost(host_h2d_src);
            cudaFree(device_d2h_src);
            cudaFree(device_h2d_dst);
            rows.push_back(row);
        }

        cudaStreamDestroy(d2h_stream);
        cudaStreamDestroy(h2d_stream);

        bench::emit_json(render_json(options, props.asyncEngineCount, props.deviceOverlap, rows, validation_passed));
        return 0;
    } catch (const std::exception& ex) {
        bench::emit_json(bench::make_error_json("failed", ex.what(), options));
        return 1;
    }
}
