#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::array<size_t, 4> kDefaultSizesMb = {128, 256, 512, 1024};
constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

struct Options {
    int iterations = 12;
    int warmup = 2;
    std::vector<size_t> sizes_mb = {128, 256, 512, 1024};
};

struct ArchitectureInfo {
    bool used_architecture1 = false;
    UINT node_index = 0;
    bool uma = false;
    bool cache_coherent_uma = false;
    bool tile_based_renderer = false;
};

struct HeapPropertiesInfo {
    D3D12_HEAP_TYPE source_heap_type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES props{};
};

struct Runtime {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ArchitectureInfo architecture{};
    HeapPropertiesInfo upload_heap{};
    std::string adapter_name;
    std::string driver_version;
};

struct ResourceSet {
    ComPtr<ID3D12Resource> upload_buffer;
    uint8_t* upload_ptr = nullptr;
    std::vector<uint8_t> host_source;
    std::vector<uint8_t> host_dest;
};

struct CaseResult {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    double host_memset_ms = 0.0;
    double host_memset_gib_per_s = 0.0;
    double upload_memset_ms = 0.0;
    double upload_memset_gib_per_s = 0.0;
    double host_pattern64_ms = 0.0;
    double host_pattern64_gib_per_s = 0.0;
    double upload_pattern64_ms = 0.0;
    double upload_pattern64_gib_per_s = 0.0;
    double shadow_to_shadow_memcpy_ms = 0.0;
    double shadow_to_shadow_memcpy_gib_per_s = 0.0;
    double shadow_to_upload_memcpy_ms = 0.0;
    double shadow_to_upload_memcpy_gib_per_s = 0.0;
};

void check_hr(HRESULT hr, const char* context) {
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << context << ": HRESULT 0x" << std::hex << std::uppercase
            << static_cast<unsigned long>(hr);
        throw std::runtime_error(oss.str());
    }
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

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

std::string quote(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

std::string format_double(double value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
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
        const auto parsed = std::strtoull(item.c_str(), &end, 10);
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

std::string narrow_from_wide(const wchar_t* wide) {
    if (wide == nullptr) {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return {};
    }
    std::string out(static_cast<size_t>(size) - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, out.data(), size, nullptr, nullptr);
    return out;
}

std::string format_driver_version(const LARGE_INTEGER& version) {
    const uint64_t raw = static_cast<uint64_t>(version.QuadPart);
    std::ostringstream oss;
    oss << HIWORD(raw) << "."
        << LOWORD(raw) << "."
        << HIWORD(raw >> 32) << "."
        << LOWORD(raw >> 32);
    return oss.str();
}

std::string heap_type_name(D3D12_HEAP_TYPE type) {
    switch (type) {
        case D3D12_HEAP_TYPE_DEFAULT: return "default";
        case D3D12_HEAP_TYPE_UPLOAD: return "upload";
        case D3D12_HEAP_TYPE_READBACK: return "readback";
        case D3D12_HEAP_TYPE_CUSTOM: return "custom";
        default: return "unknown";
    }
}

std::string cpu_page_property_name(D3D12_CPU_PAGE_PROPERTY prop) {
    switch (prop) {
        case D3D12_CPU_PAGE_PROPERTY_UNKNOWN: return "unknown";
        case D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE: return "not_available";
        case D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE: return "write_combine";
        case D3D12_CPU_PAGE_PROPERTY_WRITE_BACK: return "write_back";
        default: return "unknown_value";
    }
}

std::string memory_pool_name(D3D12_MEMORY_POOL pool) {
    switch (pool) {
        case D3D12_MEMORY_POOL_UNKNOWN: return "unknown";
        case D3D12_MEMORY_POOL_L0: return "l0";
        case D3D12_MEMORY_POOL_L1: return "l1";
        default: return "unknown_value";
    }
}

Options parse_args(int argc, char** argv) {
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

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap_type, UINT64 bytes) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = heap_type;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> resource;
    check_hr(
        device->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&resource)),
        "CreateCommittedResource");
    return resource;
}

ArchitectureInfo query_architecture_info(ID3D12Device* device) {
    ArchitectureInfo info;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch1{};
    arch1.NodeIndex = 0;
    HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch1, sizeof(arch1));
    if (SUCCEEDED(hr)) {
        info.used_architecture1 = true;
        info.node_index = arch1.NodeIndex;
        info.uma = arch1.UMA == TRUE;
        info.cache_coherent_uma = arch1.CacheCoherentUMA == TRUE;
        info.tile_based_renderer = arch1.TileBasedRenderer == TRUE;
        return info;
    }

    D3D12_FEATURE_DATA_ARCHITECTURE arch{};
    arch.NodeIndex = 0;
    check_hr(
        device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)),
        "CheckFeatureSupport(ARCHITECTURE)");
    info.node_index = arch.NodeIndex;
    info.uma = arch.UMA == TRUE;
    info.cache_coherent_uma = arch.CacheCoherentUMA == TRUE;
    info.tile_based_renderer = arch.TileBasedRenderer == TRUE;
    return info;
}

HeapPropertiesInfo query_heap_properties(ID3D12Device* device, D3D12_HEAP_TYPE heap_type) {
    HeapPropertiesInfo info;
    info.source_heap_type = heap_type;
    info.props = device->GetCustomHeapProperties(0, heap_type);
    return info;
}

Runtime create_runtime() {
    Runtime runtime;
    check_hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&runtime.factory)), "CreateDXGIFactory2");

    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        if (runtime.factory->EnumAdapterByGpuPreference(
                index,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc{};
        check_hr(candidate->GetDesc1(&desc), "IDXGIAdapter1::GetDesc1");
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
            runtime.adapter = candidate;
            runtime.adapter_name = narrow_from_wide(desc.Description);
            LARGE_INTEGER driver{};
            if (SUCCEEDED(candidate->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver))) {
                runtime.driver_version = format_driver_version(driver);
            }
            break;
        }
    }

    if (!runtime.adapter) {
        throw std::runtime_error("No hardware D3D12 adapter found");
    }

    check_hr(
        D3D12CreateDevice(runtime.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&runtime.device)),
        "D3D12CreateDevice");

    runtime.architecture = query_architecture_info(runtime.device.Get());
    runtime.upload_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_UPLOAD);
    return runtime;
}

void destroy_runtime(Runtime&) {}

ResourceSet create_resources(Runtime& runtime, size_t bytes) {
    ResourceSet resources;
    resources.upload_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_UPLOAD, static_cast<UINT64>(bytes));
    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "UploadBuffer::Map");
    resources.host_source.resize(bytes);
    resources.host_dest.resize(bytes);

    uint64_t state = 0x123456789ABCDEF0ull;
    for (size_t offset = 0; offset + sizeof(uint64_t) <= bytes; offset += sizeof(uint64_t)) {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t value = state ^ (state >> 29);
        std::memcpy(resources.host_source.data() + offset, &value, sizeof(uint64_t));
    }
    if ((bytes % sizeof(uint64_t)) != 0u) {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t value = state ^ (state >> 29);
        std::memcpy(
            resources.host_source.data() + (bytes - (bytes % sizeof(uint64_t))),
            &value,
            bytes % sizeof(uint64_t));
    }
    return resources;
}

void destroy_resources(ResourceSet& resources) {
    if (resources.upload_buffer && resources.upload_ptr) {
        resources.upload_buffer->Unmap(0, nullptr);
        resources.upload_ptr = nullptr;
    }
}

void write_pattern64(uint8_t* ptr, size_t bytes) {
    uint64_t state = 0xCAFEBABE12345678ull;
    size_t offset = 0;
    while (offset + sizeof(uint64_t) <= bytes) {
        state += 0x9E3779B97F4A7C15ull;
        const uint64_t value = state ^ (state >> 31);
        std::memcpy(ptr + offset, &value, sizeof(uint64_t));
        offset += sizeof(uint64_t);
    }
    if (offset < bytes) {
        state += 0x9E3779B97F4A7C15ull;
        const uint64_t value = state ^ (state >> 31);
        std::memcpy(ptr + offset, &value, bytes - offset);
    }
}

template <typename Fn>
double measure_average_ms(Fn&& fn, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - begin).count();
    }
    return total_ms / static_cast<double>(iterations);
}

double gib_per_s_for_bytes(double bytes, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return (bytes / kBytesPerGiB) / (ms / 1000.0);
}

CaseResult measure_case(Runtime& runtime, size_t size_mb, int warmup, int iterations) {
    const size_t bytes = size_mb * kBytesPerMiB;
    ResourceSet resources = create_resources(runtime, bytes);
    CaseResult result;
    result.size_mb = size_mb;
    result.iterations = iterations;
    result.warmup = warmup;

    result.host_memset_ms = measure_average_ms(
        [&]() {
            std::memset(resources.host_dest.data(), 0, bytes);
        },
        warmup,
        iterations);
    result.host_memset_gib_per_s = gib_per_s_for_bytes(static_cast<double>(bytes), result.host_memset_ms);

    result.upload_memset_ms = measure_average_ms(
        [&]() {
            std::memset(resources.upload_ptr, 0, bytes);
        },
        warmup,
        iterations);
    result.upload_memset_gib_per_s = gib_per_s_for_bytes(static_cast<double>(bytes), result.upload_memset_ms);

    result.host_pattern64_ms = measure_average_ms(
        [&]() {
            write_pattern64(resources.host_dest.data(), bytes);
        },
        warmup,
        iterations);
    result.host_pattern64_gib_per_s = gib_per_s_for_bytes(static_cast<double>(bytes), result.host_pattern64_ms);

    result.upload_pattern64_ms = measure_average_ms(
        [&]() {
            write_pattern64(resources.upload_ptr, bytes);
        },
        warmup,
        iterations);
    result.upload_pattern64_gib_per_s = gib_per_s_for_bytes(static_cast<double>(bytes), result.upload_pattern64_ms);

    result.shadow_to_shadow_memcpy_ms = measure_average_ms(
        [&]() {
            std::memcpy(resources.host_dest.data(), resources.host_source.data(), bytes);
        },
        warmup,
        iterations);
    result.shadow_to_shadow_memcpy_gib_per_s =
        gib_per_s_for_bytes(static_cast<double>(bytes), result.shadow_to_shadow_memcpy_ms);

    result.shadow_to_upload_memcpy_ms = measure_average_ms(
        [&]() {
            std::memcpy(resources.upload_ptr, resources.host_source.data(), bytes);
        },
        warmup,
        iterations);
    result.shadow_to_upload_memcpy_gib_per_s =
        gib_per_s_for_bytes(static_cast<double>(bytes), result.shadow_to_upload_memcpy_ms);

    destroy_resources(resources);
    return result;
}

std::string render_architecture_json(const ArchitectureInfo& info) {
    std::ostringstream oss;
    oss << "{"
        << "\"feature_query_used\":" << quote(info.used_architecture1 ? "architecture1" : "architecture") << ","
        << "\"node_index\":" << info.node_index << ","
        << "\"uma\":" << (info.uma ? "true" : "false") << ","
        << "\"cache_coherent_uma\":" << (info.cache_coherent_uma ? "true" : "false") << ","
        << "\"tile_based_renderer\":" << (info.tile_based_renderer ? "true" : "false")
        << "}";
    return oss.str();
}

std::string render_heap_properties_json(const HeapPropertiesInfo& info) {
    std::ostringstream oss;
    oss << "{"
        << "\"source_heap_type\":" << quote(heap_type_name(info.source_heap_type)) << ","
        << "\"Type\":" << static_cast<int>(info.props.Type) << ","
        << "\"TypeName\":" << quote(heap_type_name(info.props.Type)) << ","
        << "\"CPUPageProperty\":" << static_cast<int>(info.props.CPUPageProperty) << ","
        << "\"CPUPagePropertyName\":" << quote(cpu_page_property_name(info.props.CPUPageProperty)) << ","
        << "\"MemoryPoolPreference\":" << static_cast<int>(info.props.MemoryPoolPreference) << ","
        << "\"MemoryPoolPreferenceName\":" << quote(memory_pool_name(info.props.MemoryPoolPreference)) << ","
        << "\"CreationNodeMask\":" << info.props.CreationNodeMask << ","
        << "\"VisibleNodeMask\":" << info.props.VisibleNodeMask
        << "}";
    return oss.str();
}

std::string render_case_json(const CaseResult& result) {
    std::ostringstream oss;
    oss << "{"
        << "\"size_mb\":" << result.size_mb << ","
        << "\"iterations\":" << result.iterations << ","
        << "\"warmup\":" << result.warmup << ","
        << "\"host_memset_ms\":" << format_double(result.host_memset_ms) << ","
        << "\"host_memset_gib_per_s\":" << format_double(result.host_memset_gib_per_s) << ","
        << "\"upload_memset_ms\":" << format_double(result.upload_memset_ms) << ","
        << "\"upload_memset_gib_per_s\":" << format_double(result.upload_memset_gib_per_s) << ","
        << "\"host_pattern64_ms\":" << format_double(result.host_pattern64_ms) << ","
        << "\"host_pattern64_gib_per_s\":" << format_double(result.host_pattern64_gib_per_s) << ","
        << "\"upload_pattern64_ms\":" << format_double(result.upload_pattern64_ms) << ","
        << "\"upload_pattern64_gib_per_s\":" << format_double(result.upload_pattern64_gib_per_s) << ","
        << "\"shadow_to_shadow_memcpy_ms\":" << format_double(result.shadow_to_shadow_memcpy_ms) << ","
        << "\"shadow_to_shadow_memcpy_gib_per_s\":" << format_double(result.shadow_to_shadow_memcpy_gib_per_s) << ","
        << "\"shadow_to_upload_memcpy_ms\":" << format_double(result.shadow_to_upload_memcpy_ms) << ","
        << "\"shadow_to_upload_memcpy_gib_per_s\":" << format_double(result.shadow_to_upload_memcpy_gib_per_s)
        << "}";
    return oss.str();
}

std::string render_result_json(const Options& options, const Runtime& runtime, const std::vector<CaseResult>& cases) {
    double best_upload_memcpy = 0.0;
    double best_upload_memset = 0.0;
    double best_upload_pattern = 0.0;
    double avg_upload_memcpy_ratio = 0.0;

    std::ostringstream cases_json;
    cases_json << "[";
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i > 0) {
            cases_json << ",";
        }
        cases_json << render_case_json(cases[i]);
        best_upload_memcpy = std::max(best_upload_memcpy, cases[i].shadow_to_upload_memcpy_gib_per_s);
        best_upload_memset = std::max(best_upload_memset, cases[i].upload_memset_gib_per_s);
        best_upload_pattern = std::max(best_upload_pattern, cases[i].upload_pattern64_gib_per_s);
        if (cases[i].shadow_to_shadow_memcpy_gib_per_s > 0.0) {
            avg_upload_memcpy_ratio +=
                cases[i].shadow_to_upload_memcpy_gib_per_s / cases[i].shadow_to_shadow_memcpy_gib_per_s;
        }
    }
    cases_json << "]";
    if (!cases.empty()) {
        avg_upload_memcpy_ratio /= static_cast<double>(cases.size());
    }

    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"ok\","
        << "\"primary_metric\":\"best_upload_memcpy_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"context\":{"
        << "\"architecture\":" << render_architecture_json(runtime.architecture) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap)
        << "},"
        << "\"parameters\":{"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":[";
    for (size_t i = 0; i < options.sizes_mb.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << options.sizes_mb[i];
    }
    oss << "]"
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"cpu_host_write\","
        << "\"cases\":" << cases_json.str() << ","
        << "\"aggregate\":{"
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"driver_version\":" << quote(runtime.driver_version) << ","
        << "\"best_upload_memcpy_gib_per_s\":" << format_double(best_upload_memcpy) << ","
        << "\"best_upload_memset_gib_per_s\":" << format_double(best_upload_memset) << ","
        << "\"best_upload_pattern64_gib_per_s\":" << format_double(best_upload_pattern) << ","
        << "\"average_upload_memcpy_over_shadow_memcpy_ratio\":" << format_double(avg_upload_memcpy_ratio) << ","
        << "\"architecture\":" << render_architecture_json(runtime.architecture) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap)
        << "}"
        << "},"
        << "\"validation\":{\"passed\":true},"
        << "\"notes\":["
        << quote("Host-side write probe only; no GPU copy is recorded here.") << ","
        << quote("Compare upload_memcpy against shadow_to_shadow_memcpy to see whether mapped UPLOAD writes look PCIe-like or system-memory-like.") << ","
        << quote("Compare pattern64 against memcpy to separate store bandwidth from data-generation cost.")
        << "]"
        << "}";
    return oss.str();
}

std::string make_error_json(const Options& options, const std::string& message) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"best_upload_memcpy_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup
        << "},"
        << "\"measurement\":{\"timing_backend\":\"cpu_host_write\"},"
        << "\"validation\":{\"passed\":false},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
}

void emit_json(const std::string& json) {
    std::cout << json << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    Runtime runtime;
    try {
        options = parse_args(argc, argv);
        runtime = create_runtime();
        std::vector<CaseResult> cases;
        cases.reserve(options.sizes_mb.size());
        for (size_t size_mb : options.sizes_mb) {
            cases.push_back(measure_case(runtime, size_mb, options.warmup, options.iterations));
        }
        emit_json(render_result_json(options, runtime, cases));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json(options, ex.what()));
        destroy_runtime(runtime);
        return 1;
    }
}
