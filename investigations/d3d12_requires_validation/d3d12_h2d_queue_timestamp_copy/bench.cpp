#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

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

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::array<size_t, 3> kDefaultSizesMb = {128, 512, 1024};
constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

struct Options {
    int iterations = 10;
    int warmup = 2;
    std::vector<size_t> sizes_mb = {kDefaultSizesMb.begin(), kDefaultSizesMb.end()};
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
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    D3D12_COMMAND_LIST_TYPE queue_type = D3D12_COMMAND_LIST_TYPE_COPY;
    D3D12_QUERY_HEAP_TYPE query_heap_type = D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP;
    bool copy_queue_timestamp_supported = false;
    UINT64 fence_value = 0;
    UINT64 timestamp_frequency_hz = 0;
    HANDLE fence_event = nullptr;
    std::string adapter_name;
    std::string driver_version;
    ArchitectureInfo architecture{};
    HeapPropertiesInfo upload_heap{};
    HeapPropertiesInfo default_heap{};
    HeapPropertiesInfo readback_heap{};
    bool raw_pcie_h2d_eligible = false;
    std::string metric_label;
    std::string path_interpretation;
};

struct Resources {
    ComPtr<ID3D12Resource> upload_buffer;
    ComPtr<ID3D12Resource> default_buffer;
    ComPtr<ID3D12Resource> readback_buffer;
    ComPtr<ID3D12QueryHeap> query_heap;
    ComPtr<ID3D12Resource> query_readback_buffer;
    uint8_t* upload_ptr = nullptr;
    uint8_t* readback_ptr = nullptr;
    uint64_t* query_readback_ptr = nullptr;
};

struct CaseResult {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    std::string queue_type_used;
    bool copy_queue_timestamp_supported = false;
    double gpu_copy_ms_avg = 0.0;
    double gpu_copy_gib_per_s = 0.0;
    double cpu_wall_ms_avg = 0.0;
    double cpu_wall_gib_per_s = 0.0;
    bool validation_passed = false;
    std::string validation_method;
    UINT64 timestamp_frequency_hz = 0;
    bool accepted = false;
    bool raw_pcie_h2d_eligible = false;
    std::string metric_label;
    std::string path_interpretation;
    std::string error;
};

void check_hr(HRESULT hr, const char* context) {
    if (FAILED(hr)) {
        std::ostringstream oss;
        oss << context << ": HRESULT 0x" << std::hex << std::uppercase
            << static_cast<unsigned long>(hr);
        throw std::runtime_error(oss.str());
    }
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

bool is_finite(double value) {
    return std::isfinite(value) != 0;
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

std::string narrow_from_wide(const wchar_t* wide) {
    if (wide == nullptr) {
        return {};
    }
    int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
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

std::string queue_type_name(D3D12_COMMAND_LIST_TYPE type) {
    switch (type) {
        case D3D12_COMMAND_LIST_TYPE_DIRECT:
            return "direct";
        case D3D12_COMMAND_LIST_TYPE_COPY:
            return "copy";
        case D3D12_COMMAND_LIST_TYPE_COMPUTE:
            return "compute";
        default:
            return "unknown";
    }
}

std::string heap_type_name(D3D12_HEAP_TYPE type) {
    switch (type) {
        case D3D12_HEAP_TYPE_DEFAULT:
            return "default";
        case D3D12_HEAP_TYPE_UPLOAD:
            return "upload";
        case D3D12_HEAP_TYPE_READBACK:
            return "readback";
        case D3D12_HEAP_TYPE_CUSTOM:
            return "custom";
        default:
            return "unknown";
    }
}

std::string cpu_page_property_name(D3D12_CPU_PAGE_PROPERTY prop) {
    switch (prop) {
        case D3D12_CPU_PAGE_PROPERTY_UNKNOWN:
            return "unknown";
        case D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE:
            return "not_available";
        case D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE:
            return "write_combine";
        case D3D12_CPU_PAGE_PROPERTY_WRITE_BACK:
            return "write_back";
        default:
            return "unknown_value";
    }
}

std::string memory_pool_name(D3D12_MEMORY_POOL pool) {
    switch (pool) {
        case D3D12_MEMORY_POOL_UNKNOWN:
            return "unknown";
        case D3D12_MEMORY_POOL_L0:
            return "l0";
        case D3D12_MEMORY_POOL_L1:
            return "l1";
        default:
            return "unknown_value";
    }
}

std::string bool_json(bool value) {
    return value ? "true" : "false";
}

ComPtr<ID3D12Resource> create_buffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heap_type,
    UINT64 bytes,
    D3D12_RESOURCE_STATES initial_state) {
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
            initial_state,
            nullptr,
            IID_PPV_ARGS(&resource)),
        "CreateCommittedResource");
    return resource;
}

void transition_buffer(
    ID3D12GraphicsCommandList* list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before,
    D3D12_RESOURCE_STATES after) {
    if (before == after) {
        return;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list->ResourceBarrier(1, &barrier);
}

void wait_for_fence(Runtime& runtime, UINT64 value) {
    if (runtime.fence->GetCompletedValue() >= value) {
        return;
    }
    check_hr(runtime.fence->SetEventOnCompletion(value, runtime.fence_event), "SetEventOnCompletion");
    if (WaitForSingleObject(runtime.fence_event, INFINITE) != WAIT_OBJECT_0) {
        throw std::runtime_error("WaitForSingleObject failed");
    }
}

void execute_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list};
    runtime.queue->ExecuteCommandLists(1, lists);
    runtime.fence_value += 1;
    check_hr(runtime.queue->Signal(runtime.fence.Get(), runtime.fence_value), "CommandQueue::Signal");
    wait_for_fence(runtime, runtime.fence_value);
}

ArchitectureInfo query_architecture_info(ID3D12Device* device) {
    ArchitectureInfo info;
    D3D12_FEATURE_DATA_ARCHITECTURE1 arch1{};
    arch1.NodeIndex = 0;
    HRESULT hr = device->CheckFeatureSupport(
        D3D12_FEATURE_ARCHITECTURE1,
        &arch1,
        sizeof(arch1));
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

bool is_raw_pcie_h2d_eligible(
    const ArchitectureInfo& architecture,
    const HeapPropertiesInfo& upload_heap,
    const HeapPropertiesInfo& default_heap) {
    if (architecture.uma) {
        return false;
    }
    return upload_heap.props.MemoryPoolPreference == D3D12_MEMORY_POOL_L0 &&
           default_heap.props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1 &&
           upload_heap.props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE &&
           default_heap.props.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
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

    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
    check_hr(
        runtime.device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS3,
            &options3,
            sizeof(options3)),
        "CheckFeatureSupport(D3D12_OPTIONS3)");
    runtime.copy_queue_timestamp_supported = options3.CopyQueueTimestampQueriesSupported == TRUE;

    runtime.queue_type = runtime.copy_queue_timestamp_supported
        ? D3D12_COMMAND_LIST_TYPE_COPY
        : D3D12_COMMAND_LIST_TYPE_DIRECT;
    runtime.query_heap_type = runtime.copy_queue_timestamp_supported
        ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP
        : D3D12_QUERY_HEAP_TYPE_TIMESTAMP;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = runtime.queue_type;
    check_hr(runtime.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&runtime.queue)), "CreateCommandQueue");
    check_hr(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.fence)), "CreateFence");

    runtime.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (runtime.fence_event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }

    check_hr(runtime.queue->GetTimestampFrequency(&runtime.timestamp_frequency_hz), "GetTimestampFrequency");
    if (runtime.timestamp_frequency_hz == 0) {
        throw std::runtime_error("Queue timestamp frequency is zero");
    }

    runtime.upload_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_UPLOAD);
    runtime.default_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_DEFAULT);
    runtime.readback_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_READBACK);
    runtime.raw_pcie_h2d_eligible = is_raw_pcie_h2d_eligible(
        runtime.architecture,
        runtime.upload_heap,
        runtime.default_heap);

    if (runtime.raw_pcie_h2d_eligible) {
        runtime.metric_label = "discrete_l0_to_l1_h2d_like_copy_bandwidth";
        runtime.path_interpretation =
            "copy path is consistent with system-memory(L0) to video-memory(L1) upload-to-default copy";
    } else {
        runtime.metric_label = "d3d12_upload_to_default_copy_bandwidth";
        runtime.path_interpretation =
            "timed copy succeeded, but memory architecture does not justify labeling this raw PCIe H2D";
    }

    std::cerr
        << "queue_type_used=" << queue_type_name(runtime.queue_type)
        << " copy_queue_timestamp_supported=" << (runtime.copy_queue_timestamp_supported ? "true" : "false")
        << " timestamp_frequency_hz=" << runtime.timestamp_frequency_hz
        << " uma=" << (runtime.architecture.uma ? "true" : "false")
        << " cache_coherent_uma=" << (runtime.architecture.cache_coherent_uma ? "true" : "false")
        << " raw_pcie_h2d_eligible=" << (runtime.raw_pcie_h2d_eligible ? "true" : "false")
        << "\n";

    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.fence_event != nullptr) {
        CloseHandle(runtime.fence_event);
        runtime.fence_event = nullptr;
    }
}

Resources create_resources(Runtime& runtime, size_t bytes) {
    Resources resources;
    resources.upload_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        static_cast<UINT64>(bytes),
        D3D12_RESOURCE_STATE_GENERIC_READ);
    resources.default_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        static_cast<UINT64>(bytes),
        D3D12_RESOURCE_STATE_COMMON);
    resources.readback_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        static_cast<UINT64>(bytes),
        D3D12_RESOURCE_STATE_COPY_DEST);
    resources.query_readback_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        sizeof(uint64_t) * 2,
        D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_QUERY_HEAP_DESC query_desc{};
    query_desc.Count = 2;
    query_desc.Type = runtime.query_heap_type;
    check_hr(runtime.device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&resources.query_heap)), "CreateQueryHeap");

    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "Map(upload)");
    check_hr(resources.readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.readback_ptr)), "Map(readback)");
    check_hr(
        resources.query_readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.query_readback_ptr)),
        "Map(query_readback)");
    return resources;
}

void destroy_resources(Resources& resources) {
    if (resources.query_readback_buffer) {
        resources.query_readback_buffer->Unmap(0, nullptr);
    }
    if (resources.readback_buffer) {
        resources.readback_buffer->Unmap(0, nullptr);
    }
    if (resources.upload_buffer) {
        resources.upload_buffer->Unmap(0, nullptr);
    }
}

void fill_upload_pattern(uint8_t* ptr, size_t bytes, uint64_t seed) {
    uint64_t state = seed;
    size_t offset = 0;
    while (offset + sizeof(uint64_t) <= bytes) {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        std::memcpy(ptr + offset, &z, sizeof(uint64_t));
        offset += sizeof(uint64_t);
    }
    if (offset < bytes) {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        std::memcpy(ptr + offset, &z, bytes - offset);
    }
}

void clear_readback(uint8_t* ptr, size_t bytes) {
    std::memset(ptr, 0, bytes);
}

void record_timed_copy(ID3D12GraphicsCommandList* list, Resources& resources, size_t bytes) {
    list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    transition_buffer(
        list,
        resources.default_buffer.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    transition_buffer(
        list,
        resources.default_buffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COMMON);
    list->ResolveQueryData(
        resources.query_heap.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        2,
        resources.query_readback_buffer.Get(),
        0);
}

void record_validation_copy(ID3D12GraphicsCommandList* list, Resources& resources, size_t bytes) {
    transition_buffer(
        list,
        resources.default_buffer.Get(),
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    transition_buffer(
        list,
        resources.default_buffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
    transition_buffer(
        list,
        resources.default_buffer.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COMMON);
}

bool technical_success_for_case(
    const Runtime& runtime,
    const CaseResult& result) {
    return runtime.copy_queue_timestamp_supported &&
           runtime.queue_type == D3D12_COMMAND_LIST_TYPE_COPY &&
           result.gpu_copy_ms_avg > 0.0 &&
           result.validation_passed &&
           is_finite(result.gpu_copy_ms_avg) &&
           is_finite(result.gpu_copy_gib_per_s) &&
           is_finite(result.cpu_wall_ms_avg) &&
           is_finite(result.cpu_wall_gib_per_s);
}

CaseResult measure_case(Runtime& runtime, size_t size_mb, int warmup, int iterations) {
    const size_t bytes = size_mb * kBytesPerMiB;
    Resources resources = create_resources(runtime, bytes);

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(
        runtime.device->CreateCommandAllocator(runtime.queue_type, IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator");
    check_hr(
        runtime.device->CreateCommandList(0, runtime.queue_type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
        "CreateCommandList");
    check_hr(list->Close(), "Close(initial)");

    CaseResult result;
    result.size_mb = size_mb;
    result.iterations = iterations;
    result.warmup = warmup;
    result.queue_type_used = queue_type_name(runtime.queue_type);
    result.copy_queue_timestamp_supported = runtime.copy_queue_timestamp_supported;
    result.validation_method = "default_to_readback_full_copy_memcmp";
    result.timestamp_frequency_hz = runtime.timestamp_frequency_hz;
    result.raw_pcie_h2d_eligible = runtime.raw_pcie_h2d_eligible;
    result.metric_label = runtime.metric_label;
    result.path_interpretation = runtime.path_interpretation;

    const uint64_t seed = 0x1234567800000000ull ^ static_cast<uint64_t>(size_mb);
    double total_gpu_ms = 0.0;
    double total_cpu_ms = 0.0;

    try {
        for (int i = 0; i < warmup; ++i) {
            fill_upload_pattern(resources.upload_ptr, bytes, seed + static_cast<uint64_t>(i));
            resources.query_readback_ptr[0] = 0;
            resources.query_readback_ptr[1] = 0;
            check_hr(allocator->Reset(), "Allocator::Reset(warmup)");
            check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset(warmup)");
            record_timed_copy(list.Get(), resources, bytes);
            execute_and_wait(runtime, list.Get());
        }

        for (int i = 0; i < iterations; ++i) {
            fill_upload_pattern(resources.upload_ptr, bytes, seed + 0x1000ull + static_cast<uint64_t>(i));
            resources.query_readback_ptr[0] = 0;
            resources.query_readback_ptr[1] = 0;
            check_hr(allocator->Reset(), "Allocator::Reset(measure)");
            check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset(measure)");
            record_timed_copy(list.Get(), resources, bytes);

            const auto cpu_start = std::chrono::steady_clock::now();
            execute_and_wait(runtime, list.Get());
            const auto cpu_end = std::chrono::steady_clock::now();

            const uint64_t start_tick = resources.query_readback_ptr[0];
            const uint64_t end_tick = resources.query_readback_ptr[1];
            if (end_tick <= start_tick) {
                throw std::runtime_error("Invalid timestamp delta");
            }
            const double gpu_ms =
                static_cast<double>(end_tick - start_tick) * 1000.0 /
                static_cast<double>(runtime.timestamp_frequency_hz);
            if (!(gpu_ms > 0.0) || !is_finite(gpu_ms)) {
                throw std::runtime_error("Invalid gpu_copy_ms");
            }

            total_gpu_ms += gpu_ms;
            total_cpu_ms += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        }

        result.gpu_copy_ms_avg = total_gpu_ms / static_cast<double>(iterations);
        result.cpu_wall_ms_avg = total_cpu_ms / static_cast<double>(iterations);
        const double gib = static_cast<double>(bytes) / kBytesPerGiB;
        result.gpu_copy_gib_per_s = gib / (result.gpu_copy_ms_avg / 1000.0);
        result.cpu_wall_gib_per_s = gib / (result.cpu_wall_ms_avg / 1000.0);

        clear_readback(resources.readback_ptr, bytes);
        fill_upload_pattern(resources.upload_ptr, bytes, seed + 0xABC000ull);
        check_hr(allocator->Reset(), "Allocator::Reset(validate)");
        check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset(validate)");
        record_validation_copy(list.Get(), resources, bytes);
        execute_and_wait(runtime, list.Get());
        result.validation_passed = std::memcmp(resources.upload_ptr, resources.readback_ptr, bytes) == 0;

        result.accepted = technical_success_for_case(runtime, result);
    } catch (const std::exception& ex) {
        result.error = ex.what();
        result.accepted = false;
    }

    destroy_resources(resources);
    return result;
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

std::string render_architecture_json(const ArchitectureInfo& info) {
    std::ostringstream oss;
    oss << "{"
        << "\"feature_query_used\":" << quote(info.used_architecture1 ? "architecture1" : "architecture") << ","
        << "\"node_index\":" << info.node_index << ","
        << "\"uma\":" << bool_json(info.uma) << ","
        << "\"cache_coherent_uma\":" << bool_json(info.cache_coherent_uma) << ","
        << "\"tile_based_renderer\":" << bool_json(info.tile_based_renderer)
        << "}";
    return oss.str();
}

std::string render_case_json(const CaseResult& result) {
    std::ostringstream oss;
    oss << "{"
        << "\"size_mb\":" << result.size_mb << ","
        << "\"iterations\":" << result.iterations << ","
        << "\"warmup\":" << result.warmup << ","
        << "\"queue_type_used\":" << quote(result.queue_type_used) << ","
        << "\"copy_queue_timestamp_supported\":" << bool_json(result.copy_queue_timestamp_supported) << ","
        << "\"gpu_copy_ms_avg\":" << format_double(result.gpu_copy_ms_avg) << ","
        << "\"gpu_copy_gib_per_s\":" << format_double(result.gpu_copy_gib_per_s) << ","
        << "\"cpu_wall_ms_avg\":" << format_double(result.cpu_wall_ms_avg) << ","
        << "\"cpu_wall_gib_per_s\":" << format_double(result.cpu_wall_gib_per_s) << ","
        << "\"validation_passed\":" << bool_json(result.validation_passed) << ","
        << "\"validation_method\":" << quote(result.validation_method) << ","
        << "\"timestamp_frequency_hz\":" << result.timestamp_frequency_hz << ","
        << "\"accepted\":" << bool_json(result.accepted) << ","
        << "\"raw_pcie_h2d_eligible\":" << bool_json(result.raw_pcie_h2d_eligible) << ","
        << "\"metric_label\":" << quote(result.metric_label) << ","
        << "\"path_interpretation\":" << quote(result.path_interpretation);
    if (!result.error.empty()) {
        oss << ",\"error\":" << quote(result.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(const Options& options, const Runtime& runtime, const std::vector<CaseResult>& cases) {
    bool validation_passed = !cases.empty();
    bool all_accepted = !cases.empty();
    double best_gpu_copy_gib_per_s = 0.0;
    double ratio_sum = 0.0;
    int ratio_count = 0;

    std::ostringstream cases_json;
    cases_json << "[";
    for (size_t i = 0; i < cases.size(); ++i) {
        const CaseResult& row = cases[i];
        if (i > 0) {
            cases_json << ",";
        }
        cases_json << render_case_json(row);
        validation_passed = validation_passed && row.validation_passed;
        all_accepted = all_accepted && row.accepted;
        if (row.accepted) {
            best_gpu_copy_gib_per_s = std::max(best_gpu_copy_gib_per_s, row.gpu_copy_gib_per_s);
        }
        if (row.gpu_copy_ms_avg > 0.0 && is_finite(row.cpu_wall_ms_avg / row.gpu_copy_ms_avg)) {
            ratio_sum += row.cpu_wall_ms_avg / row.gpu_copy_ms_avg;
            ratio_count += 1;
        }
    }
    cases_json << "]";

    const double average_cpu_over_gpu_ratio =
        ratio_count > 0 ? ratio_sum / static_cast<double>(ratio_count) : 0.0;
    const std::string status = all_accepted ? "ok" : "failed";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(status) << ","
        << "\"primary_metric\":\"best_gpu_copy_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"context\":{"
        << "\"architecture\":" << render_architecture_json(runtime.architecture) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap) << ","
        << "\"default_heap_properties\":" << render_heap_properties_json(runtime.default_heap) << ","
        << "\"readback_heap_properties\":" << render_heap_properties_json(runtime.readback_heap)
        << "},"
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"copy_direction\":\"upload_to_default\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb) << ","
        << "\"validation_method\":\"default_to_readback_full_copy_memcmp\""
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"queue_timestamp\","
        << "\"queue_type_used\":" << quote(queue_type_name(runtime.queue_type)) << ","
        << "\"copy_queue_timestamp_supported\":" << bool_json(runtime.copy_queue_timestamp_supported) << ","
        << "\"timestamp_frequency_hz\":" << runtime.timestamp_frequency_hz << ","
        << "\"cases\":" << cases_json.str() << ","
        << "\"aggregate\":{"
        << "\"best_gpu_copy_gib_per_s\":" << format_double(best_gpu_copy_gib_per_s) << ","
        << "\"average_cpu_over_gpu_ratio\":" << format_double(average_cpu_over_gpu_ratio) << ","
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"driver_runtime_context\":{"
        << "\"driver_version\":" << quote(runtime.driver_version) << ","
        << "\"runtime\":\"d3d12\","
        << "\"toolchain\":\"msvc_d3d12\","
        << "\"queue_type_used\":" << quote(queue_type_name(runtime.queue_type))
        << "},"
        << "\"uma\":" << bool_json(runtime.architecture.uma) << ","
        << "\"cache_coherent_uma\":" << bool_json(runtime.architecture.cache_coherent_uma) << ","
        << "\"tile_based_renderer\":" << bool_json(runtime.architecture.tile_based_renderer) << ","
        << "\"node_index\":" << runtime.architecture.node_index << ","
        << "\"raw_pcie_h2d_eligible\":" << bool_json(runtime.raw_pcie_h2d_eligible) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap) << ","
        << "\"default_heap_properties\":" << render_heap_properties_json(runtime.default_heap) << ","
        << "\"readback_heap_properties\":" << render_heap_properties_json(runtime.readback_heap) << ","
        << "\"metric_label\":" << quote(runtime.metric_label) << ","
        << "\"path_interpretation\":" << quote(runtime.path_interpretation)
        << "}"
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << bool_json(validation_passed)
        << "},"
        << "\"notes\":["
        << quote("DEFAULT/UPLOAD/READBACK are abstracted heap types.") << ","
        << quote("Use GetCustomHeapProperties and architecture flags to interpret the measured path.") << ","
        << quote("If UMA is true, this bench must not be labeled raw PCIe H2D.")
        << "]"
        << "}";
    return oss.str();
}

std::string make_error_json(const Options& options, const std::string& message) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"best_gpu_copy_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"copy_direction\":\"upload_to_default\","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"queue_timestamp\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":["
        << quote("DEFAULT/UPLOAD/READBACK are abstracted heap types.") << ","
        << quote("Use GetCustomHeapProperties and architecture flags to interpret the measured path.") << ","
        << quote("If UMA is true, this bench must not be labeled raw PCIe H2D.") << ","
        << quote(message)
        << "]"
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

        emit_json(render_json(options, runtime, cases));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json(options, ex.what()));
        destroy_runtime(runtime);
        return 1;
    }
}
