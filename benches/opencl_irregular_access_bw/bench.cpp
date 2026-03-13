#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class PatternKind { Gather, Scatter, RandomBoth };
enum class DeviceType { Gpu, Cpu };

struct Options {
    int iterations = 50;
    int medium_iterations = 25;
    int large_iterations = 12;
    int warmup = 1;
    std::vector<size_t> sizes_mb = {8, 32, 128, 512, 1024};
    std::string platform_substr;
    std::string device_substr;
    DeviceType device_type = DeviceType::Gpu;
};

struct Stats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
};

struct CaseRow {
    PatternKind pattern = PatternKind::Gather;
    size_t size_mb = 0;
    size_t access_unit_bytes = 4;
    int iterations = 0;
    int warmup = 1;
    Stats stats;
};

struct DeviceSelection {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    std::string platform_name;
    std::string device_name;
    std::string device_vendor;
    std::string device_version;
};

struct OpenClRuntime {
    DeviceSelection selection;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel init_source_kernel = nullptr;
    cl_kernel gather_kernel = nullptr;
    cl_kernel scatter_kernel = nullptr;
    cl_kernel random_both_kernel = nullptr;
};

constexpr int kThreadsPerBlock = 256;
constexpr uint64_t kBytesPerGiB = 1024ull * 1024ull * 1024ull;
constexpr size_t kValidationChunkBytes = 16ull * 1024ull * 1024ull;
constexpr uint32_t kDstSentinelWord = 0xCDCDCDCDu;

const char* kKernelSource = R"CLC(
uint source_word_value(ulong word_index) {
    const ulong base = word_index * 4ul;
    const uchar b0 = (uchar)(((base + 0ul) * 17ul + 23ul) & 255ul);
    const uchar b1 = (uchar)(((base + 1ul) * 17ul + 23ul) & 255ul);
    const uchar b2 = (uchar)(((base + 2ul) * 17ul + 23ul) & 255ul);
    const uchar b3 = (uchar)(((base + 3ul) * 17ul + 23ul) & 255ul);
    return (uint)b0 | ((uint)b1 << 8) | ((uint)b2 << 16) | ((uint)b3 << 24);
}

__kernel void init_source_words(__global uint* dst, ulong element_count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong i = gid; i < element_count; i += stride) {
        dst[i] = source_word_value(i);
    }
}

__kernel void gather_kernel(__global const uint* src, __global uint* dst, __global const uint* perm, ulong count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong i = gid; i < count; i += stride) {
        dst[i] = src[perm[i]];
    }
}

__kernel void scatter_kernel(__global const uint* src, __global uint* dst, __global const uint* perm, ulong count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    for (ulong i = gid; i < count; i += stride) {
        dst[perm[i]] = src[i];
    }
}

__kernel void random_both_kernel(__global const uint* src, __global uint* dst, __global const uint* perm, ulong count) {
    const ulong gid = (ulong)get_global_id(0);
    const ulong stride = (ulong)get_global_size(0);
    const ulong mask = count - 1ul;
    for (ulong i = gid; i < count; i += stride) {
        dst[perm[(i + 1ul) & mask]] = src[perm[i]];
    }
}
)CLC";

std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (char c : value) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << c; break;
        }
    }
    return oss.str();
}

std::string quote(const std::string& value) { return "\"" + json_escape(value) + "\""; }

std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string sizes_to_json(const std::vector<size_t>& sizes_mb) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes_mb.size(); ++i) {
        if (i > 0) oss << ",";
        oss << sizes_mb[i];
    }
    oss << "]";
    return oss.str();
}

bool starts_with(const std::string& value, const std::string& prefix) { return value.rfind(prefix, 0) == 0; }

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool icontains(const std::string& haystack, const std::string& needle) {
    return needle.empty() || to_lower(haystack).find(to_lower(needle)) != std::string::npos;
}

std::vector<size_t> parse_sizes_mb(const std::string& text) {
    std::vector<size_t> sizes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed == 0) {
            throw std::runtime_error("Invalid size list: " + text);
        }
        sizes.push_back(static_cast<size_t>(parsed));
    }
    if (sizes.empty()) throw std::runtime_error("No sizes provided");
    return sizes;
}

DeviceType parse_device_type(const std::string& text) {
    const std::string lowered = to_lower(text);
    if (lowered == "gpu") return DeviceType::Gpu;
    if (lowered == "cpu") return DeviceType::Cpu;
    throw std::runtime_error("device_type must be gpu or cpu");
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto get_value = [&](const std::string& flag) -> std::string {
            if (starts_with(arg, flag + "=")) return arg.substr(flag.size() + 1);
            if (arg == flag && i + 1 < argc) return argv[++i];
            throw std::runtime_error("Missing value for " + flag);
        };
        if (starts_with(arg, "--iterations")) options.iterations = std::stoi(get_value("--iterations"));
        else if (starts_with(arg, "--medium_iterations")) options.medium_iterations = std::stoi(get_value("--medium_iterations"));
        else if (starts_with(arg, "--large_iterations")) options.large_iterations = std::stoi(get_value("--large_iterations"));
        else if (starts_with(arg, "--warmup")) options.warmup = std::stoi(get_value("--warmup"));
        else if (starts_with(arg, "--sizes_mb")) options.sizes_mb = parse_sizes_mb(get_value("--sizes_mb"));
        else if (starts_with(arg, "--platform_substr")) options.platform_substr = get_value("--platform_substr");
        else if (starts_with(arg, "--device_substr")) options.device_substr = get_value("--device_substr");
        else if (starts_with(arg, "--device_type")) options.device_type = parse_device_type(get_value("--device_type"));
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench.exe [--device_type gpu|cpu] [--platform_substr TEXT] [--device_substr TEXT] "
                         "[--iterations N] [--medium_iterations N] [--large_iterations N] [--warmup N] [--sizes_mb A,B,C]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (options.iterations <= 0 || options.medium_iterations <= 0 || options.large_iterations <= 0 || options.warmup <= 0) {
        throw std::runtime_error("iterations, medium_iterations, large_iterations, and warmup must be > 0");
    }
    return options;
}

void emit_json(const std::string& json) { std::cout << json << "\n"; }

void check_cl(cl_int status, const char* context) {
    if (status != CL_SUCCESS) {
        std::ostringstream oss;
        oss << context << ": OpenCL error " << status;
        throw std::runtime_error(oss.str());
    }
}

std::string get_platform_info_string(cl_platform_id platform, cl_platform_info param) {
    size_t size = 0;
    check_cl(clGetPlatformInfo(platform, param, 0, nullptr, &size), "clGetPlatformInfo(size)");
    std::string value(size, '\0');
    check_cl(clGetPlatformInfo(platform, param, size, value.data(), nullptr), "clGetPlatformInfo(value)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

std::string get_device_info_string(cl_device_id device, cl_device_info param) {
    size_t size = 0;
    check_cl(clGetDeviceInfo(device, param, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::string value(size, '\0');
    check_cl(clGetDeviceInfo(device, param, size, value.data(), nullptr), "clGetDeviceInfo(value)");
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

DeviceSelection select_device(const Options& options) {
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    if (platform_count == 0) throw std::runtime_error("No OpenCL platform found");

    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs(list)");
    const cl_device_type desired_type = options.device_type == DeviceType::Gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    for (cl_platform_id platform : platforms) {
        const std::string platform_name = get_platform_info_string(platform, CL_PLATFORM_NAME);
        if (!icontains(platform_name, options.platform_substr)) continue;

        cl_uint device_count = 0;
        const cl_int status = clGetDeviceIDs(platform, desired_type, 0, nullptr, &device_count);
        if (status != CL_SUCCESS || device_count == 0) continue;

        std::vector<cl_device_id> devices(device_count);
        check_cl(clGetDeviceIDs(platform, desired_type, device_count, devices.data(), nullptr), "clGetDeviceIDs(list)");
        for (cl_device_id device : devices) {
            const std::string device_name = get_device_info_string(device, CL_DEVICE_NAME);
            const std::string device_vendor = get_device_info_string(device, CL_DEVICE_VENDOR);
            if (!icontains(device_name + " " + device_vendor, options.device_substr)) continue;

            DeviceSelection selection;
            selection.platform = platform;
            selection.device = device;
            selection.platform_name = platform_name;
            selection.device_name = device_name;
            selection.device_vendor = device_vendor;
            selection.device_version = get_device_info_string(device, CL_DEVICE_VERSION);
            return selection;
        }
    }

    std::ostringstream oss;
    oss << "No matching OpenCL device found for device_type="
        << (options.device_type == DeviceType::Gpu ? "gpu" : "cpu")
        << ", platform_substr=" << options.platform_substr
        << ", device_substr=" << options.device_substr;
    throw std::runtime_error(oss.str());
}

OpenClRuntime create_runtime(const DeviceSelection& selection) {
    OpenClRuntime runtime;
    runtime.selection = selection;

    cl_int status = CL_SUCCESS;
    runtime.context = clCreateContext(nullptr, 1, &runtime.selection.device, nullptr, nullptr, &status);
    check_cl(status, "clCreateContext");
    runtime.queue = clCreateCommandQueue(runtime.context, runtime.selection.device, CL_QUEUE_PROFILING_ENABLE, &status);
    check_cl(status, "clCreateCommandQueue");

    const char* source = kKernelSource;
    const size_t source_length = std::strlen(source);
    runtime.program = clCreateProgramWithSource(runtime.context, 1, &source, &source_length, &status);
    check_cl(status, "clCreateProgramWithSource");
    status = clBuildProgram(runtime.program, 1, &runtime.selection.device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(runtime.program, runtime.selection.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(runtime.program, runtime.selection.device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed: " + build_log);
    }

    runtime.init_source_kernel = clCreateKernel(runtime.program, "init_source_words", &status);
    check_cl(status, "clCreateKernel(init_source_words)");
    runtime.gather_kernel = clCreateKernel(runtime.program, "gather_kernel", &status);
    check_cl(status, "clCreateKernel(gather_kernel)");
    runtime.scatter_kernel = clCreateKernel(runtime.program, "scatter_kernel", &status);
    check_cl(status, "clCreateKernel(scatter_kernel)");
    runtime.random_both_kernel = clCreateKernel(runtime.program, "random_both_kernel", &status);
    check_cl(status, "clCreateKernel(random_both_kernel)");
    return runtime;
}

void destroy_runtime(OpenClRuntime& runtime) {
    if (runtime.random_both_kernel != nullptr) clReleaseKernel(runtime.random_both_kernel);
    if (runtime.scatter_kernel != nullptr) clReleaseKernel(runtime.scatter_kernel);
    if (runtime.gather_kernel != nullptr) clReleaseKernel(runtime.gather_kernel);
    if (runtime.init_source_kernel != nullptr) clReleaseKernel(runtime.init_source_kernel);
    if (runtime.program != nullptr) clReleaseProgram(runtime.program);
    if (runtime.queue != nullptr) clReleaseCommandQueue(runtime.queue);
    if (runtime.context != nullptr) clReleaseContext(runtime.context);
}

size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

uint32_t source_word_value(uint64_t word_index) {
    const uint64_t base = word_index * 4ull;
    const auto source_byte = [](uint64_t offset) -> uint32_t {
        return static_cast<uint32_t>(((offset * 17ull) + 23ull) & 0xFFu);
    };
    return source_byte(base + 0) |
           (source_byte(base + 1) << 8u) |
           (source_byte(base + 2) << 16u) |
           (source_byte(base + 3) << 24u);
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
    uint64_t rng_state = 0xC001D00D5A5A5A5Aull ^ static_cast<uint64_t>(element_count);
    for (size_t i = element_count - 1; i > 0; --i) {
        const size_t j = static_cast<size_t>(splitmix64_next(rng_state) % static_cast<uint64_t>(i + 1));
        std::swap(permutation[i], permutation[j]);
    }
    return permutation;
}

std::vector<uint32_t> build_inverse_permutation(const std::vector<uint32_t>& permutation) {
    std::vector<uint32_t> inverse(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) inverse[permutation[i]] = static_cast<uint32_t>(i);
    return inverse;
}

int effective_iterations(const Options& options, size_t size_mb) {
    if (size_mb <= 128) return options.iterations;
    if (size_mb <= 512) return options.medium_iterations;
    return options.large_iterations;
}

void fill_buffer_words(cl_command_queue queue, cl_mem buffer, uint32_t value, size_t bytes) {
    check_cl(clEnqueueFillBuffer(queue, buffer, &value, sizeof(value), 0, bytes, 0, nullptr, nullptr), "clEnqueueFillBuffer");
    check_cl(clFinish(queue), "clFinish(fill)");
}

void init_source_words(OpenClRuntime& runtime, cl_mem src_buffer, size_t element_count) {
    const size_t global_size = round_up(element_count, static_cast<size_t>(kThreadsPerBlock));
    const cl_ulong count_arg = static_cast<cl_ulong>(element_count);
    check_cl(clSetKernelArg(runtime.init_source_kernel, 0, sizeof(cl_mem), &src_buffer), "clSetKernelArg(init,src)");
    check_cl(clSetKernelArg(runtime.init_source_kernel, 1, sizeof(count_arg), &count_arg), "clSetKernelArg(init,count)");
    check_cl(clEnqueueNDRangeKernel(runtime.queue, runtime.init_source_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel(init)");
    check_cl(clFinish(runtime.queue), "clFinish(init)");
}

template <typename PrepareFn>
Stats measure_kernel(OpenClRuntime& runtime, cl_kernel kernel, size_t global_size, size_t logical_bytes, int warmup, int iterations, PrepareFn prepare) {
    Stats stats;
    try {
        for (int i = 0; i < warmup; ++i) {
            prepare();
            cl_event event = nullptr;
            check_cl(clEnqueueNDRangeKernel(runtime.queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, &event), "clEnqueueNDRangeKernel(warmup)");
            check_cl(clWaitForEvents(1, &event), "clWaitForEvents(warmup)");
            clReleaseEvent(event);
        }

        double total_ms = 0.0;
        for (int i = 0; i < iterations; ++i) {
            prepare();
            cl_event event = nullptr;
            check_cl(clEnqueueNDRangeKernel(runtime.queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, &event), "clEnqueueNDRangeKernel");
            check_cl(clWaitForEvents(1, &event), "clWaitForEvents");
            cl_ulong start = 0;
            cl_ulong end = 0;
            check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr), "clGetEventProfilingInfo(start)");
            check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr), "clGetEventProfilingInfo(end)");
            clReleaseEvent(event);
            total_ms += static_cast<double>(end - start) / 1'000'000.0;
        }

        stats.success = true;
        stats.avg_ms = total_ms / static_cast<double>(iterations);
        const double gib = static_cast<double>(logical_bytes) / static_cast<double>(kBytesPerGiB);
        stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }
    return stats;
}

bool validate_random_pattern(cl_command_queue queue, cl_mem dst_buffer, size_t element_count, PatternKind pattern, const std::vector<uint32_t>& permutation, const std::vector<uint32_t>& inverse_permutation) {
    const size_t chunk_words = std::max<size_t>(1, kValidationChunkBytes / sizeof(uint32_t));
    std::vector<uint32_t> chunk(chunk_words);
    const uint64_t mask = static_cast<uint64_t>(element_count - 1);
    for (size_t offset = 0; offset < element_count; offset += chunk_words) {
        const size_t words_this_chunk = std::min(chunk_words, element_count - offset);
        check_cl(clEnqueueReadBuffer(queue, dst_buffer, CL_TRUE, offset * sizeof(uint32_t), words_this_chunk * sizeof(uint32_t), chunk.data(), 0, nullptr, nullptr), "clEnqueueReadBuffer(validate)");
        for (size_t i = 0; i < words_this_chunk; ++i) {
            const uint64_t dst_index = static_cast<uint64_t>(offset + i);
            uint32_t expected = 0;
            switch (pattern) {
                case PatternKind::Gather:
                    expected = source_word_value(permutation[dst_index]);
                    break;
                case PatternKind::Scatter:
                    expected = source_word_value(inverse_permutation[dst_index]);
                    break;
                case PatternKind::RandomBoth: {
                    const uint64_t logical_index = inverse_permutation[dst_index];
                    expected = source_word_value(permutation[(logical_index + mask) & mask]);
                    break;
                }
            }
            if (chunk[i] != expected) return false;
        }
    }
    return true;
}

const char* pattern_name(PatternKind pattern) {
    switch (pattern) {
        case PatternKind::Gather: return "gather";
        case PatternKind::Scatter: return "scatter";
        case PatternKind::RandomBoth: return "random_both";
    }
    return "unknown";
}

std::string make_error_json(const std::string& status, const std::string& message, const Options& options) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(status) << ","
        << "\"primary_metric\":\"random_both_1024mb_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"api\":\"opencl\","
        << "\"device_type\":" << quote(options.device_type == DeviceType::Gpu ? "gpu" : "cpu") << ","
        << "\"platform_substr\":" << quote(options.platform_substr) << ","
        << "\"device_substr\":" << quote(options.device_substr) << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"opencl_event_profiling\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
}

std::string render_case_json(const CaseRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"pattern\":" << quote(pattern_name(row.pattern)) << ","
        << "\"size_mb\":" << row.size_mb << ","
        << "\"access_unit_bytes\":" << row.access_unit_bytes << ","
        << "\"iterations\":" << row.iterations << ","
        << "\"warmup\":" << row.warmup << ","
        << "\"avg_ms\":" << format_double(row.stats.avg_ms) << ","
        << "\"gib_per_s\":" << format_double(row.stats.gib_per_s) << ","
        << "\"success\":" << (row.stats.success ? "true" : "false");
    if (!row.stats.error.empty()) oss << ",\"error\":" << quote(row.stats.error);
    oss << "}";
    return oss.str();
}

std::string render_json(const Options& options, const DeviceSelection& selection, const std::vector<CaseRow>& rows, bool validation_passed) {
    double best_gather = 0.0;
    double best_scatter = 0.0;
    double best_random_both = 0.0;
    double random_both_1024 = 0.0;
    std::ostringstream cases;
    cases << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) cases << ",";
        cases << render_case_json(rows[i]);
        if (!rows[i].stats.success) continue;
        switch (rows[i].pattern) {
            case PatternKind::Gather:
                best_gather = std::max(best_gather, rows[i].stats.gib_per_s);
                break;
            case PatternKind::Scatter:
                best_scatter = std::max(best_scatter, rows[i].stats.gib_per_s);
                break;
            case PatternKind::RandomBoth:
                best_random_both = std::max(best_random_both, rows[i].stats.gib_per_s);
                if (rows[i].size_mb == 1024) random_both_1024 = rows[i].stats.gib_per_s;
                break;
        }
    }
    cases << "]";

    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) { return row.stats.success; });
    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"random_both_1024mb_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"api\":\"opencl\","
        << "\"device_type\":" << quote(options.device_type == DeviceType::Gpu ? "gpu" : "cpu") << ","
        << "\"platform_substr\":" << quote(options.platform_substr) << ","
        << "\"device_substr\":" << quote(options.device_substr) << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"opencl_event_profiling\","
        << "\"device\":{"
        << "\"platform_name\":" << quote(selection.platform_name) << ","
        << "\"device_name\":" << quote(selection.device_name) << ","
        << "\"device_vendor\":" << quote(selection.device_vendor) << ","
        << "\"device_version\":" << quote(selection.device_version)
        << "},"
        << "\"patterns\":[\"gather\",\"scatter\",\"random_both\"],"
        << "\"best_gather_gib_per_s\":" << format_double(best_gather) << ","
        << "\"best_scatter_gib_per_s\":" << format_double(best_scatter) << ","
        << "\"best_random_both_gib_per_s\":" << format_double(best_random_both) << ","
        << "\"random_both_1024mb_gib_per_s\":" << format_double(random_both_1024) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "},"
        << "\"notes\":["
        << quote("Permutation generation and upload happen outside the timed kernel interval.") << ","
        << quote("gather, scatter, and random_both share the same explicit permutation for each size.")
        << "]"
        << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    OpenClRuntime runtime{};
    try {
        options = parse_args(argc, argv);
        runtime = create_runtime(select_device(options));

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        for (size_t size_mb : options.sizes_mb) {
            const size_t size_bytes = size_mb * 1024ull * 1024ull;
            const size_t element_count = size_bytes / sizeof(uint32_t);
            const size_t global_size = round_up(element_count, static_cast<size_t>(kThreadsPerBlock));

            std::vector<uint32_t> permutation = build_random_permutation(element_count);
            std::vector<uint32_t> inverse_permutation = build_inverse_permutation(permutation);

            cl_int status = CL_SUCCESS;
            cl_mem src_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, size_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(src)");
            cl_mem dst_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE, size_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(dst)");
            cl_mem perm_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY, size_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(perm)");

            try {
                check_cl(clEnqueueWriteBuffer(runtime.queue, perm_buffer, CL_TRUE, 0, size_bytes, permutation.data(), 0, nullptr, nullptr), "clEnqueueWriteBuffer(perm)");
                init_source_words(runtime, src_buffer, element_count);

                for (PatternKind pattern : {PatternKind::Gather, PatternKind::Scatter, PatternKind::RandomBoth}) {
                    CaseRow row;
                    row.pattern = pattern;
                    row.size_mb = size_mb;
                    row.iterations = effective_iterations(options, size_mb);
                    row.warmup = options.warmup;

                    auto prepare = [&]() { fill_buffer_words(runtime.queue, dst_buffer, kDstSentinelWord, size_bytes); };
                    const cl_ulong count_arg = static_cast<cl_ulong>(element_count);

                    try {
                        if (pattern == PatternKind::Gather) {
                            check_cl(clSetKernelArg(runtime.gather_kernel, 0, sizeof(cl_mem), &src_buffer), "clSetKernelArg(gather,src)");
                            check_cl(clSetKernelArg(runtime.gather_kernel, 1, sizeof(cl_mem), &dst_buffer), "clSetKernelArg(gather,dst)");
                            check_cl(clSetKernelArg(runtime.gather_kernel, 2, sizeof(cl_mem), &perm_buffer), "clSetKernelArg(gather,perm)");
                            check_cl(clSetKernelArg(runtime.gather_kernel, 3, sizeof(count_arg), &count_arg), "clSetKernelArg(gather,count)");
                            row.stats = measure_kernel(runtime, runtime.gather_kernel, global_size, size_bytes * 2ull, row.warmup, row.iterations, prepare);
                        } else if (pattern == PatternKind::Scatter) {
                            check_cl(clSetKernelArg(runtime.scatter_kernel, 0, sizeof(cl_mem), &src_buffer), "clSetKernelArg(scatter,src)");
                            check_cl(clSetKernelArg(runtime.scatter_kernel, 1, sizeof(cl_mem), &dst_buffer), "clSetKernelArg(scatter,dst)");
                            check_cl(clSetKernelArg(runtime.scatter_kernel, 2, sizeof(cl_mem), &perm_buffer), "clSetKernelArg(scatter,perm)");
                            check_cl(clSetKernelArg(runtime.scatter_kernel, 3, sizeof(count_arg), &count_arg), "clSetKernelArg(scatter,count)");
                            row.stats = measure_kernel(runtime, runtime.scatter_kernel, global_size, size_bytes * 2ull, row.warmup, row.iterations, prepare);
                        } else {
                            check_cl(clSetKernelArg(runtime.random_both_kernel, 0, sizeof(cl_mem), &src_buffer), "clSetKernelArg(random_both,src)");
                            check_cl(clSetKernelArg(runtime.random_both_kernel, 1, sizeof(cl_mem), &dst_buffer), "clSetKernelArg(random_both,dst)");
                            check_cl(clSetKernelArg(runtime.random_both_kernel, 2, sizeof(cl_mem), &perm_buffer), "clSetKernelArg(random_both,perm)");
                            check_cl(clSetKernelArg(runtime.random_both_kernel, 3, sizeof(count_arg), &count_arg), "clSetKernelArg(random_both,count)");
                            row.stats = measure_kernel(runtime, runtime.random_both_kernel, global_size, size_bytes * 2ull, row.warmup, row.iterations, prepare);
                        }

                        if (row.stats.success && !validate_random_pattern(runtime.queue, dst_buffer, element_count, pattern, permutation, inverse_permutation)) {
                            row.stats.success = false;
                            row.stats.error = std::string(pattern_name(pattern)) + " validation failed";
                            validation_passed = false;
                        }
                    } catch (const std::exception& ex) {
                        row.stats.error = ex.what();
                        validation_passed = false;
                    }
                    if (!row.stats.success && row.stats.error.empty()) validation_passed = false;
                    rows.push_back(row);
                }
            } catch (...) {
                clReleaseMemObject(perm_buffer);
                clReleaseMemObject(dst_buffer);
                clReleaseMemObject(src_buffer);
                throw;
            }

            clReleaseMemObject(perm_buffer);
            clReleaseMemObject(dst_buffer);
            clReleaseMemObject(src_buffer);
        }

        emit_json(render_json(options, runtime.selection, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        destroy_runtime(runtime);
        emit_json(make_error_json("failed", ex.what(), options));
        return 1;
    }
}
