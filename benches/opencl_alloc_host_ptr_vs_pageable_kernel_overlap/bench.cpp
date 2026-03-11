#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<size_t, 3> kDefaultSizesMb = {128, 512, 1024};
constexpr size_t kThreadsPerBlock = 256;
constexpr int kCalibrationPassLimit = 8;

const char* kKernelSource = R"CLC(
__kernel void low_memory_spin(__global float* output, ulong loops) {
    const uint tid = get_global_id(0);
    uint x = 0x12345678u ^ (tid * 747796405u + 2891336453u);
    uint y = 0x9E3779B9u + (tid * 1664525u + 1013904223u);
    uint z = 0xA5A5A5A5u ^ (tid * 2246822519u + 3266489917u);
    for (ulong i = 0; i < loops; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        y = y * 1664525u + 1013904223u;
        z ^= x + 0x9E3779B9u + (z << 6) + (z >> 2);
    }
    output[tid] = (float)((x ^ y ^ z) & 0x00FFFFFFu);
}
)CLC";

struct Options {
    int iterations = 50;
    int warmup = 5;
    std::vector<size_t> sizes_mb = {8, 32, 128, 512, 1024};
};

struct WallStats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
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
    double copy_gib_per_s = 0.0;
};

struct DirectionRow {
    WallStats copy_solo;
    KernelStats kernel_solo;
    OverlapStats overlap;
};

struct MemoryRow {
    DirectionRow h2d;
    DirectionRow d2h;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    MemoryRow alloc_host_ptr;
    MemoryRow pageable;
};

struct OpenClRuntime {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue transfer_queue = nullptr;
    cl_command_queue kernel_queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    size_t total_threads = 0;
    std::string platform_name;
    std::string device_name;
    std::string device_version;
};

std::string json_escape(const std::string& value) {
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

std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string sizes_to_json(const std::vector<size_t>& sizes_mb) {
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

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::vector<size_t> parse_sizes_mb(const std::string& text) {
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

Options parse_common_args(int argc, char** argv) {
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

void emit_json(const std::string& json) {
    std::cout << json << "\n";
}

std::string make_error_json(
    const std::string& status,
    const std::string& message,
    const Options& options,
    const std::string& primary_metric) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(status) << ","
        << "\"primary_metric\":" << quote(primary_metric) << ","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"api\":\"opencl\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
}

void check_cl(cl_int status, const char* context) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << context << ": OpenCL error " << status;
        throw std::runtime_error(oss.str());
    }
}

std::string get_info_string(
    cl_platform_id platform,
    cl_device_id device,
    cl_platform_info platform_param,
    cl_device_info device_param) {
    size_t size = 0;
    if (platform != nullptr) {
        check_cl(clGetPlatformInfo(platform, platform_param, 0, nullptr, &size), "clGetPlatformInfo(size)");
        std::string value(size, '\0');
        check_cl(clGetPlatformInfo(platform, platform_param, size, value.data(), nullptr), "clGetPlatformInfo(value)");
        if (!value.empty() && value.back() == '\0') {
            value.pop_back();
        }
        return value;
    }

    check_cl(clGetDeviceInfo(device, device_param, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::string value(size, '\0');
    check_cl(clGetDeviceInfo(device, device_param, size, value.data(), nullptr), "clGetDeviceInfo(value)");
    if (!value.empty() && value.back() == '\0') {
        value.pop_back();
    }
    return value;
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

bool buffer_has_byte_value(const unsigned char* data, size_t bytes, unsigned char expected) {
    for (size_t i = 0; i < bytes; ++i) {
        if (data[i] != expected) {
            return false;
        }
    }
    return true;
}

OpenClRuntime create_runtime() {
    OpenClRuntime runtime;

    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    if (platform_count == 0) {
        throw std::runtime_error("No OpenCL platform found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (status != CL_SUCCESS || device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
        runtime.platform = platform;
        runtime.device = devices[0];
        break;
    }
    if (runtime.device == nullptr) {
        throw std::runtime_error("No GPU OpenCL device found");
    }

    runtime.platform_name = get_info_string(runtime.platform, nullptr, CL_PLATFORM_NAME, 0);
    runtime.device_name = get_info_string(nullptr, runtime.device, 0, CL_DEVICE_NAME);
    runtime.device_version = get_info_string(nullptr, runtime.device, 0, CL_DEVICE_VERSION);

    cl_int status = CL_SUCCESS;
    runtime.context = clCreateContext(nullptr, 1, &runtime.device, nullptr, nullptr, &status);
    check_cl(status, "clCreateContext");
    runtime.transfer_queue = clCreateCommandQueue(runtime.context, runtime.device, CL_QUEUE_PROFILING_ENABLE, &status);
    check_cl(status, "clCreateCommandQueue(transfer)");
    runtime.kernel_queue = clCreateCommandQueue(runtime.context, runtime.device, CL_QUEUE_PROFILING_ENABLE, &status);
    check_cl(status, "clCreateCommandQueue(kernel)");

    const char* source = kKernelSource;
    const size_t source_length = std::strlen(source);
    runtime.program = clCreateProgramWithSource(runtime.context, 1, &source, &source_length, &status);
    check_cl(status, "clCreateProgramWithSource");
    status = clBuildProgram(runtime.program, 1, &runtime.device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(runtime.program, runtime.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(runtime.program, runtime.device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed: " + build_log);
    }
    runtime.kernel = clCreateKernel(runtime.program, "low_memory_spin", &status);
    check_cl(status, "clCreateKernel");

    cl_uint compute_units = 0;
    check_cl(clGetDeviceInfo(runtime.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr), "clGetDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS)");
    runtime.total_threads = static_cast<size_t>(compute_units) * kThreadsPerBlock;
    if (runtime.total_threads == 0) {
        throw std::runtime_error("OpenCL device reported zero compute units");
    }

    return runtime;
}

void destroy_runtime(OpenClRuntime& runtime) {
    if (runtime.kernel != nullptr) {
        clReleaseKernel(runtime.kernel);
    }
    if (runtime.program != nullptr) {
        clReleaseProgram(runtime.program);
    }
    if (runtime.kernel_queue != nullptr) {
        clReleaseCommandQueue(runtime.kernel_queue);
    }
    if (runtime.transfer_queue != nullptr) {
        clReleaseCommandQueue(runtime.transfer_queue);
    }
    if (runtime.context != nullptr) {
        clReleaseContext(runtime.context);
    }
}

void fill_buffer(cl_command_queue queue, cl_mem buffer, unsigned char value, size_t bytes) {
    check_cl(clEnqueueFillBuffer(queue, buffer, &value, sizeof(value), 0, bytes, 0, nullptr, nullptr), "clEnqueueFillBuffer");
    check_cl(clFinish(queue), "clFinish(fill)");
}

WallStats measure_copy_wall(
    cl_command_queue queue,
    cl_mem device_transfer_buffer,
    void* host_ptr,
    size_t bytes,
    bool is_h2d,
    int warmup,
    int iterations) {
    WallStats stats;
    for (int i = 0; i < warmup; ++i) {
        cl_int status = is_h2d
            ? clEnqueueWriteBuffer(queue, device_transfer_buffer, CL_FALSE, 0, bytes, host_ptr, 0, nullptr, nullptr)
            : clEnqueueReadBuffer(queue, device_transfer_buffer, CL_FALSE, 0, bytes, host_ptr, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) {
            stats.error = "Warmup copy enqueue failed";
            return stats;
        }
        status = clFinish(queue);
        if (status != CL_SUCCESS) {
            stats.error = "Warmup clFinish(copy) failed";
            return stats;
        }
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (is_h2d) {
            check_cl(clEnqueueWriteBuffer(queue, device_transfer_buffer, CL_FALSE, 0, bytes, host_ptr, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        } else {
            check_cl(clEnqueueReadBuffer(queue, device_transfer_buffer, CL_FALSE, 0, bytes, host_ptr, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        }
        check_cl(clFinish(queue), "clFinish(copy)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    return stats;
}

KernelStats measure_kernel_wall(
    OpenClRuntime& runtime,
    cl_mem kernel_output_buffer,
    unsigned long long loops,
    int warmup,
    int iterations) {
    KernelStats stats;
    const cl_ulong loop_value = static_cast<cl_ulong>(loops);
    check_cl(clSetKernelArg(runtime.kernel, 0, sizeof(cl_mem), &kernel_output_buffer), "clSetKernelArg(output)");
    check_cl(clSetKernelArg(runtime.kernel, 1, sizeof(cl_ulong), &loop_value), "clSetKernelArg(loops)");

    const size_t global_size = runtime.total_threads;
    const size_t local_size = kThreadsPerBlock;
    for (int i = 0; i < warmup; ++i) {
        check_cl(clEnqueueNDRangeKernel(runtime.kernel_queue, runtime.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr), "clEnqueueNDRangeKernel(warmup)");
        check_cl(clFinish(runtime.kernel_queue), "clFinish(kernel warmup)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        check_cl(clEnqueueNDRangeKernel(runtime.kernel_queue, runtime.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        check_cl(clFinish(runtime.kernel_queue), "clFinish(kernel)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.loop_count = loops;
    return stats;
}

KernelStats calibrate_kernel_to_target(
    OpenClRuntime& runtime,
    cl_mem kernel_output_buffer,
    double target_ms,
    int warmup,
    int iterations) {
    unsigned long long loops = 1ull << 16;
    KernelStats stats;
    for (int pass = 0; pass < kCalibrationPassLimit; ++pass) {
        stats = measure_kernel_wall(runtime, kernel_output_buffer, loops, warmup, iterations);
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
    OpenClRuntime& runtime,
    cl_mem device_transfer_buffer,
    void* host_ptr,
    size_t copy_bytes,
    bool is_h2d,
    cl_mem kernel_output_buffer,
    unsigned long long loops,
    int warmup,
    int iterations,
    const WallStats& copy_solo,
    const KernelStats& kernel_solo) {
    OverlapStats stats;
    const cl_ulong loop_value = static_cast<cl_ulong>(loops);
    check_cl(clSetKernelArg(runtime.kernel, 0, sizeof(cl_mem), &kernel_output_buffer), "clSetKernelArg(output overlap)");
    check_cl(clSetKernelArg(runtime.kernel, 1, sizeof(cl_ulong), &loop_value), "clSetKernelArg(loops overlap)");

    const size_t global_size = runtime.total_threads;
    const size_t local_size = kThreadsPerBlock;
    for (int i = 0; i < warmup; ++i) {
        if (is_h2d) {
            check_cl(clEnqueueWriteBuffer(runtime.transfer_queue, device_transfer_buffer, CL_FALSE, 0, copy_bytes, host_ptr, 0, nullptr, nullptr), "warmup clEnqueueWriteBuffer");
        } else {
            check_cl(clEnqueueReadBuffer(runtime.transfer_queue, device_transfer_buffer, CL_FALSE, 0, copy_bytes, host_ptr, 0, nullptr, nullptr), "warmup clEnqueueReadBuffer");
        }
        check_cl(clEnqueueNDRangeKernel(runtime.kernel_queue, runtime.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr), "warmup clEnqueueNDRangeKernel");
        check_cl(clFinish(runtime.transfer_queue), "clFinish(transfer warmup)");
        check_cl(clFinish(runtime.kernel_queue), "clFinish(kernel warmup)");
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (is_h2d) {
            check_cl(clEnqueueWriteBuffer(runtime.transfer_queue, device_transfer_buffer, CL_FALSE, 0, copy_bytes, host_ptr, 0, nullptr, nullptr), "clEnqueueWriteBuffer");
        } else {
            check_cl(clEnqueueReadBuffer(runtime.transfer_queue, device_transfer_buffer, CL_FALSE, 0, copy_bytes, host_ptr, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        }
        check_cl(clEnqueueNDRangeKernel(runtime.kernel_queue, runtime.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr), "clEnqueueNDRangeKernel overlap");
        check_cl(clFinish(runtime.transfer_queue), "clFinish(transfer)");
        check_cl(clFinish(runtime.kernel_queue), "clFinish(kernel)");
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_wall_ms = total_ms / static_cast<double>(iterations);
    stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (copy_solo.avg_ms + kernel_solo.avg_ms);
    stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(copy_solo.avg_ms, kernel_solo.avg_ms);
    const double gib = static_cast<double>(copy_bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.copy_gib_per_s = gib / (stats.avg_wall_ms / 1000.0);
    return stats;
}

bool verify_kernel_output(OpenClRuntime& runtime, cl_mem kernel_output_buffer) {
    std::vector<float> host_output(runtime.total_threads, 0.0f);
    check_cl(clEnqueueReadBuffer(runtime.transfer_queue, kernel_output_buffer, CL_TRUE, 0, host_output.size() * sizeof(float), host_output.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(kernel output)");
    for (float value : host_output) {
        if (std::isfinite(value) && value != 0.0f) {
            return true;
        }
    }
    return false;
}

bool verify_direction_result(
    OpenClRuntime& runtime,
    cl_mem device_transfer_buffer,
    const unsigned char* host_reference,
    unsigned char* host_buffer,
    size_t bytes,
    bool is_h2d,
    std::vector<unsigned char>& verify_buffer) {
    if (!is_h2d) {
        return buffer_has_byte_value(host_buffer, bytes, 0x5A);
    }

    check_cl(clEnqueueReadBuffer(runtime.transfer_queue, device_transfer_buffer, CL_TRUE, 0, bytes, verify_buffer.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(verify H2D)");
    return std::memcmp(verify_buffer.data(), host_reference, bytes) == 0;
}

DirectionRow run_direction_case(
    OpenClRuntime& runtime,
    cl_mem device_transfer_buffer,
    unsigned char* host_buffer,
    const unsigned char* host_reference,
    cl_mem kernel_output_buffer,
    size_t copy_bytes,
    bool is_h2d,
    int warmup,
    int iterations,
    bool* validation_passed,
    std::vector<unsigned char>& verify_buffer) {
    DirectionRow row;

    row.copy_solo = measure_copy_wall(runtime.transfer_queue, device_transfer_buffer, host_buffer, copy_bytes, is_h2d, warmup, iterations);
    if (!row.copy_solo.success) {
        *validation_passed = false;
        return row;
    }

    if (!verify_direction_result(runtime, device_transfer_buffer, host_reference, host_buffer, copy_bytes, is_h2d, verify_buffer)) {
        *validation_passed = false;
    }

    fill_buffer(runtime.kernel_queue, kernel_output_buffer, 0x00, runtime.total_threads * sizeof(float));
    row.kernel_solo = calibrate_kernel_to_target(runtime, kernel_output_buffer, row.copy_solo.avg_ms, warmup, iterations);
    if (!row.kernel_solo.success) {
        *validation_passed = false;
        return row;
    }
    if (!verify_kernel_output(runtime, kernel_output_buffer)) {
        *validation_passed = false;
    }

    fill_buffer(runtime.kernel_queue, kernel_output_buffer, 0x00, runtime.total_threads * sizeof(float));
    if (is_h2d) {
        fill_buffer(runtime.transfer_queue, device_transfer_buffer, 0x00, copy_bytes);
    } else {
        std::memset(host_buffer, 0x00, copy_bytes);
    }

    row.overlap = measure_overlap_wall(
        runtime,
        device_transfer_buffer,
        host_buffer,
        copy_bytes,
        is_h2d,
        kernel_output_buffer,
        row.kernel_solo.loop_count,
        warmup,
        iterations,
        row.copy_solo,
        row.kernel_solo);
    if (!row.overlap.success) {
        *validation_passed = false;
        return row;
    }

    if (!verify_direction_result(runtime, device_transfer_buffer, host_reference, host_buffer, copy_bytes, is_h2d, verify_buffer)) {
        *validation_passed = false;
    }
    return row;
}

std::string render_wall_stats_json(const WallStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << format_double(stats.avg_ms) << ","
        << "\"gib_per_s\":" << format_double(stats.gib_per_s);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_kernel_stats_json(const KernelStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << format_double(stats.avg_ms) << ","
        << "\"loop_count\":" << stats.loop_count;
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_overlap_stats_json(const OverlapStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_wall_ms\":" << format_double(stats.avg_wall_ms) << ","
        << "\"copy_gib_per_s\":" << format_double(stats.copy_gib_per_s) << ","
        << "\"wall_vs_solo_sum_ratio\":" << format_double(stats.wall_vs_solo_sum_ratio) << ","
        << "\"wall_vs_solo_max_ratio\":" << format_double(stats.wall_vs_solo_max_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_direction_json(const DirectionRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"copy_solo\":" << render_wall_stats_json(row.copy_solo) << ","
        << "\"kernel_solo\":" << render_kernel_stats_json(row.kernel_solo) << ","
        << "\"overlap\":" << render_overlap_stats_json(row.overlap)
        << "}";
    return oss.str();
}

std::string render_memory_json(const MemoryRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"h2d\":" << render_direction_json(row.h2d) << ","
        << "\"d2h\":" << render_direction_json(row.d2h)
        << "}";
    return oss.str();
}

bool direction_ok(const DirectionRow& row) {
    return row.copy_solo.success && row.kernel_solo.success && row.overlap.success;
}

std::string render_json(const Options& options, const OpenClRuntime& runtime, const std::vector<CaseRow>& rows, bool validation_passed) {
    double min_alloc_host_ptr_h2d = 0.0;
    double min_alloc_host_ptr_d2h = 0.0;
    double min_pageable_h2d = 0.0;
    double min_pageable_d2h = 0.0;
    bool have_alloc_h2d = false;
    bool have_alloc_d2h = false;
    bool have_pageable_h2d = false;
    bool have_pageable_d2h = false;

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
              << "\"alloc_host_ptr\":" << render_memory_json(row.alloc_host_ptr) << ","
              << "\"pageable\":" << render_memory_json(row.pageable)
              << "}";

        if (direction_ok(row.alloc_host_ptr.h2d)) {
            min_alloc_host_ptr_h2d = have_alloc_h2d ? std::min(min_alloc_host_ptr_h2d, row.alloc_host_ptr.h2d.overlap.wall_vs_solo_sum_ratio) : row.alloc_host_ptr.h2d.overlap.wall_vs_solo_sum_ratio;
            have_alloc_h2d = true;
        }
        if (direction_ok(row.alloc_host_ptr.d2h)) {
            min_alloc_host_ptr_d2h = have_alloc_d2h ? std::min(min_alloc_host_ptr_d2h, row.alloc_host_ptr.d2h.overlap.wall_vs_solo_sum_ratio) : row.alloc_host_ptr.d2h.overlap.wall_vs_solo_sum_ratio;
            have_alloc_d2h = true;
        }
        if (direction_ok(row.pageable.h2d)) {
            min_pageable_h2d = have_pageable_h2d ? std::min(min_pageable_h2d, row.pageable.h2d.overlap.wall_vs_solo_sum_ratio) : row.pageable.h2d.overlap.wall_vs_solo_sum_ratio;
            have_pageable_h2d = true;
        }
        if (direction_ok(row.pageable.d2h)) {
            min_pageable_d2h = have_pageable_d2h ? std::min(min_pageable_d2h, row.pageable.d2h.overlap.wall_vs_solo_sum_ratio) : row.pageable.d2h.overlap.wall_vs_solo_sum_ratio;
            have_pageable_d2h = true;
        }
    }
    cases << "]";

    const double min_alloc_host_ptr = std::min(min_alloc_host_ptr_h2d, min_alloc_host_ptr_d2h);
    const double min_pageable = std::min(min_pageable_h2d, min_pageable_d2h);
    const double penalty = min_alloc_host_ptr > 0.0 ? min_pageable / min_alloc_host_ptr : 0.0;
    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return direction_ok(row.alloc_host_ptr.h2d) &&
               direction_ok(row.alloc_host_ptr.d2h) &&
               direction_ok(row.pageable.h2d) &&
               direction_ok(row.pageable.d2h);
    });

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"min_pageable_wall_vs_solo_sum_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"api\":\"opencl\","
        << "\"copy_directions\":[\"H2D\",\"D2H\"],"
        << "\"memory_types\":[\"CL_MEM_ALLOC_HOST_PTR\",\"pageable\"],"
        << "\"pinned_like_mode\":\"CL_MEM_ALLOC_HOST_PTR\","
        << "\"enqueue_mode\":\"non_blocking\","
        << "\"stream_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\","
        << "\"platform_name\":" << quote(runtime.platform_name) << ","
        << "\"device_name\":" << quote(runtime.device_name) << ","
        << "\"device_version\":" << quote(runtime.device_version) << ","
        << "\"min_alloc_host_ptr_h2d_wall_vs_solo_sum_ratio\":" << format_double(min_alloc_host_ptr_h2d) << ","
        << "\"min_alloc_host_ptr_d2h_wall_vs_solo_sum_ratio\":" << format_double(min_alloc_host_ptr_d2h) << ","
        << "\"min_pageable_h2d_wall_vs_solo_sum_ratio\":" << format_double(min_pageable_h2d) << ","
        << "\"min_pageable_d2h_wall_vs_solo_sum_ratio\":" << format_double(min_pageable_d2h) << ","
        << "\"min_alloc_host_ptr_wall_vs_solo_sum_ratio\":" << format_double(min_alloc_host_ptr) << ","
        << "\"min_pageable_wall_vs_solo_sum_ratio\":" << format_double(min_pageable) << ","
        << "\"pageable_to_alloc_host_ptr_overlap_penalty\":" << format_double(penalty) << ","
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
    Options options{};
    OpenClRuntime runtime{};
    try {
        options = parse_common_args(argc, argv);
        if (is_default_size_list(options.sizes_mb)) {
            options.sizes_mb.assign(kDefaultSizesMb.begin(), kDefaultSizesMb.end());
        }

        runtime = create_runtime();
        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t copy_bytes = size_mb * 1024ull * 1024ull;
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            cl_int status = CL_SUCCESS;
            cl_mem device_h2d_dst = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, copy_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(device_h2d_dst)");
            cl_mem device_d2h_src = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, copy_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(device_d2h_src)");
            cl_mem kernel_output = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, runtime.total_threads * sizeof(float), nullptr, &status);
            check_cl(status, "clCreateBuffer(kernel_output)");
            cl_mem alloc_h2d_host_buffer = clCreateBuffer(runtime.context, CL_MEM_ALLOC_HOST_PTR, copy_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(alloc_h2d_host_buffer)");
            cl_mem alloc_d2h_host_buffer = clCreateBuffer(runtime.context, CL_MEM_ALLOC_HOST_PTR, copy_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(alloc_d2h_host_buffer)");

            unsigned char* alloc_h2d_ptr = static_cast<unsigned char*>(
                clEnqueueMapBuffer(runtime.transfer_queue, alloc_h2d_host_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, copy_bytes, 0, nullptr, nullptr, &status));
            check_cl(status, "clEnqueueMapBuffer(alloc_h2d)");
            unsigned char* alloc_d2h_ptr = static_cast<unsigned char*>(
                clEnqueueMapBuffer(runtime.transfer_queue, alloc_d2h_host_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, copy_bytes, 0, nullptr, nullptr, &status));
            check_cl(status, "clEnqueueMapBuffer(alloc_d2h)");

            std::vector<unsigned char> pageable_h2d(copy_bytes, 0x3C);
            std::vector<unsigned char> pageable_d2h(copy_bytes, 0x00);
            std::vector<unsigned char> verify_buffer(copy_bytes, 0x00);

            std::memset(alloc_h2d_ptr, 0x3C, copy_bytes);
            std::memset(alloc_d2h_ptr, 0x00, copy_bytes);
            fill_buffer(runtime.transfer_queue, device_h2d_dst, 0x00, copy_bytes);
            fill_buffer(runtime.transfer_queue, device_d2h_src, 0x5A, copy_bytes);
            fill_buffer(runtime.kernel_queue, kernel_output, 0x00, runtime.total_threads * sizeof(float));

            row.alloc_host_ptr.h2d = run_direction_case(runtime, device_h2d_dst, alloc_h2d_ptr, alloc_h2d_ptr, kernel_output, copy_bytes, true, row.warmup, row.iterations, &validation_passed, verify_buffer);
            fill_buffer(runtime.kernel_queue, kernel_output, 0x00, runtime.total_threads * sizeof(float));
            row.alloc_host_ptr.d2h = run_direction_case(runtime, device_d2h_src, alloc_d2h_ptr, nullptr, kernel_output, copy_bytes, false, row.warmup, row.iterations, &validation_passed, verify_buffer);

            fill_buffer(runtime.transfer_queue, device_h2d_dst, 0x00, copy_bytes);
            fill_buffer(runtime.transfer_queue, device_d2h_src, 0x5A, copy_bytes);
            fill_buffer(runtime.kernel_queue, kernel_output, 0x00, runtime.total_threads * sizeof(float));

            row.pageable.h2d = run_direction_case(runtime, device_h2d_dst, pageable_h2d.data(), pageable_h2d.data(), kernel_output, copy_bytes, true, row.warmup, row.iterations, &validation_passed, verify_buffer);
            fill_buffer(runtime.kernel_queue, kernel_output, 0x00, runtime.total_threads * sizeof(float));
            row.pageable.d2h = run_direction_case(runtime, device_d2h_src, pageable_d2h.data(), nullptr, kernel_output, copy_bytes, false, row.warmup, row.iterations, &validation_passed, verify_buffer);

            check_cl(clEnqueueUnmapMemObject(runtime.transfer_queue, alloc_h2d_host_buffer, alloc_h2d_ptr, 0, nullptr, nullptr), "clEnqueueUnmapMemObject(alloc_h2d)");
            check_cl(clEnqueueUnmapMemObject(runtime.transfer_queue, alloc_d2h_host_buffer, alloc_d2h_ptr, 0, nullptr, nullptr), "clEnqueueUnmapMemObject(alloc_d2h)");
            check_cl(clFinish(runtime.transfer_queue), "clFinish(unmap)");

            clReleaseMemObject(alloc_d2h_host_buffer);
            clReleaseMemObject(alloc_h2d_host_buffer);
            clReleaseMemObject(kernel_output);
            clReleaseMemObject(device_d2h_src);
            clReleaseMemObject(device_h2d_dst);
            rows.push_back(row);
        }

        emit_json(render_json(options, runtime, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json("failed", ex.what(), options, "min_pageable_wall_vs_solo_sum_ratio"));
        destroy_runtime(runtime);
        return 1;
    }
}
