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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint64_t kBytesPerMiB = 1024ull * 1024ull;
constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kTheoryGiBPerS = 58.687292;

enum class Mode {
    H2DLike,
    D2HLike,
    DefaultToDefault,
    UploadToReadback,
};

enum class QueueSelection {
    Copy,
    Direct,
};

enum class ReuseMode {
    SameResources,
    RotateDstOffsets,
    RotateResourcePairs,
    RecreateEveryIter,
};

struct Options {
    Mode mode = Mode::H2DLike;
    QueueSelection queue = QueueSelection::Copy;
    ReuseMode reuse = ReuseMode::SameResources;
    size_t size_mb = 512;
    int iterations = 12;
    int warmup = 2;
    int rotation_depth = 1;
    bool vary_upload_seed = true;
    bool validate_each_iter = true;
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
    HANDLE fence_event = nullptr;
    UINT64 fence_value = 0;
    UINT64 timestamp_frequency_hz = 0;
    bool copy_queue_timestamp_supported = false;
    ArchitectureInfo architecture{};
    HeapPropertiesInfo upload_heap{};
    HeapPropertiesInfo default_heap{};
    HeapPropertiesInfo readback_heap{};
    std::string adapter_name;
    std::string driver_version;
};

struct PairResources {
    ComPtr<ID3D12Resource> source_default_buffer;
    ComPtr<ID3D12Resource> dest_default_buffer;
    ComPtr<ID3D12Resource> dest_readback_buffer;
    uint8_t* dest_readback_ptr = nullptr;
    D3D12_RESOURCE_STATES source_default_state = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES dest_default_state = D3D12_RESOURCE_STATE_COMMON;
};

struct RunResources {
    ComPtr<ID3D12Resource> upload_buffer;
    ComPtr<ID3D12Resource> validation_readback_buffer;
    ComPtr<ID3D12QueryHeap> query_heap;
    ComPtr<ID3D12Resource> query_readback_buffer;
    uint8_t* upload_ptr = nullptr;
    uint8_t* validation_readback_ptr = nullptr;
    uint64_t* query_readback_ptr = nullptr;
    std::vector<uint8_t> expected_bytes;
    std::vector<PairResources> pairs;
};

struct CommandContext {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
};

struct IterationRecord {
    int iteration = 0;
    double gpu_copy_ms = 0.0;
    double gpu_copy_gib_per_s = 0.0;
    double cpu_wall_ms = 0.0;
    double cpu_submit_to_validate_ms = 0.0;
    double cpu_fill_to_validate_ms = 0.0;
    double cpu_submit_to_validate_gib_per_s = 0.0;
    double cpu_fill_to_validate_gib_per_s = 0.0;
    uint64_t start_tick = 0;
    uint64_t end_tick = 0;
    int resource_pair_index = 0;
    uint64_t dst_offset_bytes = 0;
    uint64_t seed_id = 0;
    bool validation_passed = false;
};

struct AggregateStats {
    double gpu_copy_ms_avg = 0.0;
    double gpu_copy_ms_p50 = 0.0;
    double gpu_copy_ms_min = 0.0;
    double gpu_copy_ms_max = 0.0;
    double gpu_copy_gib_per_s_avg = 0.0;
    double gpu_copy_gib_per_s_p50 = 0.0;
    double gpu_copy_gib_per_s_min = 0.0;
    double gpu_copy_gib_per_s_max = 0.0;
    double cpu_wall_ms_avg = 0.0;
    double cpu_submit_to_validate_ms_avg = 0.0;
    double cpu_submit_to_validate_ms_p50 = 0.0;
    double cpu_submit_to_validate_ms_min = 0.0;
    double cpu_submit_to_validate_ms_max = 0.0;
    double cpu_submit_to_validate_gib_per_s_avg = 0.0;
    double cpu_submit_to_validate_gib_per_s_p50 = 0.0;
    double cpu_submit_to_validate_gib_per_s_min = 0.0;
    double cpu_submit_to_validate_gib_per_s_max = 0.0;
    double cpu_fill_to_validate_ms_avg = 0.0;
    double cpu_fill_to_validate_ms_p50 = 0.0;
    double cpu_fill_to_validate_ms_min = 0.0;
    double cpu_fill_to_validate_ms_max = 0.0;
    double cpu_fill_to_validate_gib_per_s_avg = 0.0;
    double cpu_fill_to_validate_gib_per_s_p50 = 0.0;
    double cpu_fill_to_validate_gib_per_s_min = 0.0;
    double cpu_fill_to_validate_gib_per_s_max = 0.0;
    double first_iter_gpu_ms = 0.0;
    double steady_state_gpu_ms_avg = 0.0;
    bool submit_to_validate_above_theory = false;
    bool fill_to_validate_above_theory = false;
    double timestamp_vs_completion_ratio = 0.0;
    bool validation_passed = false;
    int validation_failures = 0;
    bool accepted = false;
};

struct MeasurementResult {
    std::string status = "failed";
    std::string error;
    std::string notes;
    AggregateStats aggregate{};
    std::vector<IterationRecord> per_iteration;
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

bool is_finite(double value) {
    return std::isfinite(value) != 0;
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

std::string mode_name(Mode mode) {
    switch (mode) {
        case Mode::H2DLike:
            return "h2d_like";
        case Mode::D2HLike:
            return "d2h_like";
        case Mode::DefaultToDefault:
            return "default_to_default";
        case Mode::UploadToReadback:
            return "upload_to_readback";
        default:
            return "unknown";
    }
}

std::string queue_name(QueueSelection queue) {
    switch (queue) {
        case QueueSelection::Copy:
            return "copy";
        case QueueSelection::Direct:
            return "direct";
        default:
            return "unknown";
    }
}

std::string reuse_name(ReuseMode reuse) {
    switch (reuse) {
        case ReuseMode::SameResources:
            return "same_resources";
        case ReuseMode::RotateDstOffsets:
            return "rotate_dst_offsets";
        case ReuseMode::RotateResourcePairs:
            return "rotate_resource_pairs";
        case ReuseMode::RecreateEveryIter:
            return "recreate_every_iter";
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

Mode parse_mode(const std::string& value) {
    if (value == "h2d_like") {
        return Mode::H2DLike;
    }
    if (value == "d2h_like") {
        return Mode::D2HLike;
    }
    if (value == "default_to_default") {
        return Mode::DefaultToDefault;
    }
    if (value == "upload_to_readback") {
        return Mode::UploadToReadback;
    }
    throw std::runtime_error("Invalid mode: " + value);
}

QueueSelection parse_queue(const std::string& value) {
    if (value == "copy") {
        return QueueSelection::Copy;
    }
    if (value == "direct") {
        return QueueSelection::Direct;
    }
    throw std::runtime_error("Invalid queue: " + value);
}

ReuseMode parse_reuse(const std::string& value) {
    if (value == "same_resources") {
        return ReuseMode::SameResources;
    }
    if (value == "rotate_dst_offsets") {
        return ReuseMode::RotateDstOffsets;
    }
    if (value == "rotate_resource_pairs") {
        return ReuseMode::RotateResourcePairs;
    }
    if (value == "recreate_every_iter") {
        return ReuseMode::RecreateEveryIter;
    }
    throw std::runtime_error("Invalid reuse: " + value);
}

bool parse_bool_01(const std::string& value, const std::string& flag) {
    if (value == "0") {
        return false;
    }
    if (value == "1") {
        return true;
    }
    throw std::runtime_error("Invalid boolean for " + flag + ": " + value);
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

        if (starts_with(arg, "--mode")) {
            options.mode = parse_mode(get_value("--mode"));
        } else if (starts_with(arg, "--queue")) {
            options.queue = parse_queue(get_value("--queue"));
        } else if (starts_with(arg, "--size_mb")) {
            options.size_mb = static_cast<size_t>(std::stoull(get_value("--size_mb")));
        } else if (starts_with(arg, "--iterations")) {
            options.iterations = std::stoi(get_value("--iterations"));
        } else if (starts_with(arg, "--warmup")) {
            options.warmup = std::stoi(get_value("--warmup"));
        } else if (starts_with(arg, "--reuse")) {
            options.reuse = parse_reuse(get_value("--reuse"));
        } else if (starts_with(arg, "--rotation_depth")) {
            options.rotation_depth = std::stoi(get_value("--rotation_depth"));
        } else if (starts_with(arg, "--vary_upload_seed")) {
            options.vary_upload_seed = parse_bool_01(get_value("--vary_upload_seed"), "--vary_upload_seed");
        } else if (starts_with(arg, "--validate_each_iter")) {
            options.validate_each_iter = parse_bool_01(get_value("--validate_each_iter"), "--validate_each_iter");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bench.exe"
                << " [--mode h2d_like|d2h_like|default_to_default|upload_to_readback]"
                << " [--queue copy|direct]"
                << " [--size_mb N]"
                << " [--iterations N]"
                << " [--warmup N]"
                << " [--reuse same_resources|rotate_dst_offsets|rotate_resource_pairs|recreate_every_iter]"
                << " [--rotation_depth N]"
                << " [--vary_upload_seed 0|1]"
                << " [--validate_each_iter 0|1]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.size_mb == 0 || options.iterations <= 0 || options.warmup <= 0 || options.rotation_depth <= 0) {
        throw std::runtime_error("size_mb, iterations, warmup, and rotation_depth must be > 0");
    }
    if (options.reuse != ReuseMode::RotateDstOffsets && options.reuse != ReuseMode::RotateResourcePairs) {
        options.rotation_depth = 1;
    }
    return options;
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
    if (before == after || resource == nullptr) {
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

Runtime create_runtime(const Options& options) {
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
    runtime.default_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_DEFAULT);
    runtime.readback_heap = query_heap_properties(runtime.device.Get(), D3D12_HEAP_TYPE_READBACK);

    D3D12_FEATURE_DATA_D3D12_OPTIONS3 options3{};
    check_hr(
        runtime.device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS3,
            &options3,
            sizeof(options3)),
        "CheckFeatureSupport(D3D12_OPTIONS3)");
    runtime.copy_queue_timestamp_supported = options3.CopyQueueTimestampQueriesSupported == TRUE;

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = options.queue == QueueSelection::Copy
        ? D3D12_COMMAND_LIST_TYPE_COPY
        : D3D12_COMMAND_LIST_TYPE_DIRECT;
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
    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.fence_event != nullptr) {
        CloseHandle(runtime.fence_event);
        runtime.fence_event = nullptr;
    }
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

void execute_command_list(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list};
    runtime.queue->ExecuteCommandLists(1, lists);
}

UINT64 signal_queue(Runtime& runtime) {
    runtime.fence_value += 1;
    check_hr(runtime.queue->Signal(runtime.fence.Get(), runtime.fence_value), "CommandQueue::Signal");
    return runtime.fence_value;
}

void execute_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    execute_command_list(runtime, list);
    wait_for_fence(runtime, signal_queue(runtime));
}

CommandContext create_command_context(Runtime& runtime, const Options& options) {
    CommandContext context;
    D3D12_COMMAND_LIST_TYPE list_type = options.queue == QueueSelection::Copy
        ? D3D12_COMMAND_LIST_TYPE_COPY
        : D3D12_COMMAND_LIST_TYPE_DIRECT;
    check_hr(runtime.device->CreateCommandAllocator(list_type, IID_PPV_ARGS(&context.allocator)), "CreateCommandAllocator");
    check_hr(
        runtime.device->CreateCommandList(0, list_type, context.allocator.Get(), nullptr, IID_PPV_ARGS(&context.list)),
        "CreateCommandList");
    check_hr(context.list->Close(), "Close(initial)");
    return context;
}

void reset_command_context(CommandContext& context) {
    check_hr(context.allocator->Reset(), "CommandAllocator::Reset");
    check_hr(context.list->Reset(context.allocator.Get(), nullptr), "CommandList::Reset");
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

void clear_bytes(uint8_t* ptr, size_t bytes) {
    std::memset(ptr, 0, bytes);
}

D3D12_QUERY_HEAP_TYPE query_heap_type_for_options(const Options& options) {
    return options.queue == QueueSelection::Copy
        ? D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP
        : D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
}

bool mode_uses_source_default(Mode mode) {
    return mode == Mode::D2HLike || mode == Mode::DefaultToDefault;
}

bool mode_uses_dest_default(Mode mode) {
    return mode == Mode::H2DLike || mode == Mode::DefaultToDefault;
}

bool mode_uses_dest_readback(Mode mode) {
    return mode == Mode::D2HLike || mode == Mode::UploadToReadback;
}

uint64_t destination_allocation_bytes(size_t copy_bytes, const Options& options) {
    if (options.reuse == ReuseMode::RotateDstOffsets) {
        return static_cast<uint64_t>(copy_bytes) * static_cast<uint64_t>(options.rotation_depth);
    }
    return static_cast<uint64_t>(copy_bytes);
}

PairResources create_pair_resources(Runtime& runtime, const Options& options, size_t copy_bytes) {
    PairResources pair;
    const uint64_t dst_bytes = destination_allocation_bytes(copy_bytes, options);
    if (mode_uses_source_default(options.mode)) {
        pair.source_default_buffer = create_buffer(
            runtime.device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            static_cast<UINT64>(copy_bytes),
            D3D12_RESOURCE_STATE_COMMON);
        pair.source_default_state = D3D12_RESOURCE_STATE_COMMON;
    }
    if (mode_uses_dest_default(options.mode)) {
        pair.dest_default_buffer = create_buffer(
            runtime.device.Get(),
            D3D12_HEAP_TYPE_DEFAULT,
            dst_bytes,
            D3D12_RESOURCE_STATE_COMMON);
        pair.dest_default_state = D3D12_RESOURCE_STATE_COMMON;
    }
    if (mode_uses_dest_readback(options.mode)) {
        pair.dest_readback_buffer = create_buffer(
            runtime.device.Get(),
            D3D12_HEAP_TYPE_READBACK,
            dst_bytes,
            D3D12_RESOURCE_STATE_COPY_DEST);
        check_hr(
            pair.dest_readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&pair.dest_readback_ptr)),
            "Map(dest_readback)");
    }
    return pair;
}

RunResources create_run_resources(Runtime& runtime, const Options& options, size_t copy_bytes) {
    RunResources resources;
    resources.upload_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        static_cast<UINT64>(copy_bytes),
        D3D12_RESOURCE_STATE_GENERIC_READ);
    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "Map(upload)");

    resources.validation_readback_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        static_cast<UINT64>(copy_bytes),
        D3D12_RESOURCE_STATE_COPY_DEST);
    check_hr(
        resources.validation_readback_buffer->Map(
            0,
            nullptr,
            reinterpret_cast<void**>(&resources.validation_readback_ptr)),
        "Map(validation_readback)");

    D3D12_QUERY_HEAP_DESC query_desc{};
    query_desc.Count = 2;
    query_desc.Type = query_heap_type_for_options(options);
    check_hr(runtime.device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&resources.query_heap)), "CreateQueryHeap");

    resources.query_readback_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        sizeof(uint64_t) * 2,
        D3D12_RESOURCE_STATE_COPY_DEST);
    check_hr(
        resources.query_readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.query_readback_ptr)),
        "Map(query_readback)");
    resources.expected_bytes.resize(copy_bytes);

    const int pair_count = options.reuse == ReuseMode::RotateResourcePairs ? options.rotation_depth : 1;
    resources.pairs.reserve(static_cast<size_t>(pair_count));
    for (int i = 0; i < pair_count; ++i) {
        resources.pairs.push_back(create_pair_resources(runtime, options, copy_bytes));
    }
    return resources;
}

void destroy_pair_resources(PairResources& pair) {
    if (pair.dest_readback_buffer) {
        pair.dest_readback_buffer->Unmap(0, nullptr);
        pair.dest_readback_ptr = nullptr;
    }
}

void destroy_run_resources(RunResources& resources) {
    for (auto& pair : resources.pairs) {
        destroy_pair_resources(pair);
    }
    if (resources.query_readback_buffer) {
        resources.query_readback_buffer->Unmap(0, nullptr);
        resources.query_readback_ptr = nullptr;
    }
    if (resources.validation_readback_buffer) {
        resources.validation_readback_buffer->Unmap(0, nullptr);
        resources.validation_readback_ptr = nullptr;
    }
    if (resources.upload_buffer) {
        resources.upload_buffer->Unmap(0, nullptr);
        resources.upload_ptr = nullptr;
    }
}

void prepare_source_default(
    Runtime& runtime,
    CommandContext& context,
    RunResources& resources,
    PairResources& pair,
    size_t copy_bytes) {
    if (!pair.source_default_buffer) {
        return;
    }
    reset_command_context(context);
    transition_buffer(
        context.list.Get(),
        pair.source_default_buffer.Get(),
        pair.source_default_state,
        D3D12_RESOURCE_STATE_COPY_DEST);
    pair.source_default_state = D3D12_RESOURCE_STATE_COPY_DEST;
    context.list->CopyBufferRegion(pair.source_default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, copy_bytes);
    transition_buffer(
        context.list.Get(),
        pair.source_default_buffer.Get(),
        pair.source_default_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    pair.source_default_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    execute_and_wait(runtime, context.list.Get());
}

void record_validation_copy(
    CommandContext& context,
    RunResources& resources,
    PairResources& pair,
    size_t copy_bytes,
    uint64_t dst_offset_bytes) {
    transition_buffer(
        context.list.Get(),
        pair.dest_default_buffer.Get(),
        pair.dest_default_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    pair.dest_default_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    context.list->CopyBufferRegion(
        resources.validation_readback_buffer.Get(),
        0,
        pair.dest_default_buffer.Get(),
        dst_offset_bytes,
        copy_bytes);
    transition_buffer(
        context.list.Get(),
        pair.dest_default_buffer.Get(),
        pair.dest_default_state,
        D3D12_RESOURCE_STATE_COMMON);
    pair.dest_default_state = D3D12_RESOURCE_STATE_COMMON;
}

bool compare_copy_result(
    RunResources& resources,
    PairResources& pair,
    const Options& options,
    size_t copy_bytes,
    uint64_t dst_offset_bytes) {
    if (options.mode == Mode::H2DLike || options.mode == Mode::DefaultToDefault) {
        return std::memcmp(resources.expected_bytes.data(), resources.validation_readback_ptr, copy_bytes) == 0;
    }

    if (pair.dest_readback_ptr == nullptr) {
        return false;
    }
    return std::memcmp(
        resources.expected_bytes.data(),
        pair.dest_readback_ptr + dst_offset_bytes,
        copy_bytes) == 0;
}

bool validate_copy_result(
    Runtime& runtime,
    CommandContext& context,
    RunResources& resources,
    PairResources& pair,
    const Options& options,
    size_t copy_bytes,
    uint64_t dst_offset_bytes) {
    if (options.mode == Mode::H2DLike || options.mode == Mode::DefaultToDefault) {
        clear_bytes(resources.validation_readback_ptr, copy_bytes);
        reset_command_context(context);
        record_validation_copy(context, resources, pair, copy_bytes, dst_offset_bytes);
        execute_and_wait(runtime, context.list.Get());
    }
    return compare_copy_result(resources, pair, options, copy_bytes, dst_offset_bytes);
}

void record_timed_copy(
    CommandContext& context,
    RunResources& resources,
    PairResources& pair,
    const Options& options,
    size_t copy_bytes,
    uint64_t dst_offset_bytes) {
    context.list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);

    if (options.mode == Mode::H2DLike) {
        transition_buffer(
            context.list.Get(),
            pair.dest_default_buffer.Get(),
            pair.dest_default_state,
            D3D12_RESOURCE_STATE_COPY_DEST);
        pair.dest_default_state = D3D12_RESOURCE_STATE_COPY_DEST;
        context.list->CopyBufferRegion(
            pair.dest_default_buffer.Get(),
            dst_offset_bytes,
            resources.upload_buffer.Get(),
            0,
            copy_bytes);
    } else if (options.mode == Mode::D2HLike) {
        transition_buffer(
            context.list.Get(),
            pair.dest_readback_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_DEST);
        context.list->CopyBufferRegion(
            pair.dest_readback_buffer.Get(),
            dst_offset_bytes,
            pair.source_default_buffer.Get(),
            0,
            copy_bytes);
    } else if (options.mode == Mode::DefaultToDefault) {
        transition_buffer(
            context.list.Get(),
            pair.dest_default_buffer.Get(),
            pair.dest_default_state,
            D3D12_RESOURCE_STATE_COPY_DEST);
        pair.dest_default_state = D3D12_RESOURCE_STATE_COPY_DEST;
        context.list->CopyBufferRegion(
            pair.dest_default_buffer.Get(),
            dst_offset_bytes,
            pair.source_default_buffer.Get(),
            0,
            copy_bytes);
    } else {
        transition_buffer(
            context.list.Get(),
            pair.dest_readback_buffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_DEST);
        context.list->CopyBufferRegion(
            pair.dest_readback_buffer.Get(),
            dst_offset_bytes,
            resources.upload_buffer.Get(),
            0,
            copy_bytes);
    }

    context.list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);

    if (options.mode == Mode::H2DLike || options.mode == Mode::DefaultToDefault) {
        transition_buffer(
            context.list.Get(),
            pair.dest_default_buffer.Get(),
            pair.dest_default_state,
            D3D12_RESOURCE_STATE_COMMON);
        pair.dest_default_state = D3D12_RESOURCE_STATE_COMMON;
    }

    context.list->ResolveQueryData(
        resources.query_heap.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        0,
        2,
        resources.query_readback_buffer.Get(),
        0);
}

uint64_t iteration_seed(const Options& options, int iteration_index) {
    return options.vary_upload_seed
        ? static_cast<uint64_t>(0xABC000ull + iteration_index)
        : 0xABC000ull;
}

int pair_index_for_iteration(const Options& options, int iteration_index) {
    if (options.reuse == ReuseMode::RotateResourcePairs) {
        return iteration_index % options.rotation_depth;
    }
    if (options.reuse == ReuseMode::RecreateEveryIter) {
        return iteration_index;
    }
    return 0;
}

uint64_t dst_offset_for_iteration(const Options& options, size_t copy_bytes, int iteration_index) {
    if (options.reuse == ReuseMode::RotateDstOffsets) {
        return static_cast<uint64_t>(iteration_index % options.rotation_depth) * static_cast<uint64_t>(copy_bytes);
    }
    return 0;
}

double p50_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() % 2u) == 1u) {
        return values[mid];
    }
    return (values[mid - 1] + values[mid]) * 0.5;
}

AggregateStats compute_aggregate(const std::vector<IterationRecord>& records) {
    AggregateStats aggregate;
    if (records.empty()) {
        return aggregate;
    }

    std::vector<double> gpu_ms;
    std::vector<double> gpu_gib;
    std::vector<double> submit_ms;
    std::vector<double> submit_gib;
    std::vector<double> fill_ms;
    std::vector<double> fill_gib;
    double cpu_ms_sum = 0.0;
    bool validation_passed = true;
    int validation_failures = 0;
    for (const auto& record : records) {
        gpu_ms.push_back(record.gpu_copy_ms);
        gpu_gib.push_back(record.gpu_copy_gib_per_s);
        cpu_ms_sum += record.cpu_wall_ms;
        if (record.cpu_submit_to_validate_ms > 0.0 && is_finite(record.cpu_submit_to_validate_ms)) {
            submit_ms.push_back(record.cpu_submit_to_validate_ms);
        }
        if (record.cpu_submit_to_validate_gib_per_s > 0.0 && is_finite(record.cpu_submit_to_validate_gib_per_s)) {
            submit_gib.push_back(record.cpu_submit_to_validate_gib_per_s);
        }
        if (record.cpu_fill_to_validate_ms > 0.0 && is_finite(record.cpu_fill_to_validate_ms)) {
            fill_ms.push_back(record.cpu_fill_to_validate_ms);
        }
        if (record.cpu_fill_to_validate_gib_per_s > 0.0 && is_finite(record.cpu_fill_to_validate_gib_per_s)) {
            fill_gib.push_back(record.cpu_fill_to_validate_gib_per_s);
        }
        validation_passed = validation_passed && record.validation_passed;
        if (!record.validation_passed) {
            validation_failures += 1;
        }
    }

    aggregate.gpu_copy_ms_avg = std::accumulate(gpu_ms.begin(), gpu_ms.end(), 0.0) / static_cast<double>(gpu_ms.size());
    aggregate.gpu_copy_ms_p50 = p50_of(gpu_ms);
    aggregate.gpu_copy_ms_min = *std::min_element(gpu_ms.begin(), gpu_ms.end());
    aggregate.gpu_copy_ms_max = *std::max_element(gpu_ms.begin(), gpu_ms.end());
    aggregate.gpu_copy_gib_per_s_avg = std::accumulate(gpu_gib.begin(), gpu_gib.end(), 0.0) / static_cast<double>(gpu_gib.size());
    aggregate.gpu_copy_gib_per_s_p50 = p50_of(gpu_gib);
    aggregate.gpu_copy_gib_per_s_min = *std::min_element(gpu_gib.begin(), gpu_gib.end());
    aggregate.gpu_copy_gib_per_s_max = *std::max_element(gpu_gib.begin(), gpu_gib.end());
    aggregate.cpu_wall_ms_avg = cpu_ms_sum / static_cast<double>(records.size());
    if (!submit_ms.empty()) {
        aggregate.cpu_submit_to_validate_ms_avg =
            std::accumulate(submit_ms.begin(), submit_ms.end(), 0.0) / static_cast<double>(submit_ms.size());
        aggregate.cpu_submit_to_validate_ms_p50 = p50_of(submit_ms);
        aggregate.cpu_submit_to_validate_ms_min = *std::min_element(submit_ms.begin(), submit_ms.end());
        aggregate.cpu_submit_to_validate_ms_max = *std::max_element(submit_ms.begin(), submit_ms.end());
    }
    if (!submit_gib.empty()) {
        aggregate.cpu_submit_to_validate_gib_per_s_avg =
            std::accumulate(submit_gib.begin(), submit_gib.end(), 0.0) / static_cast<double>(submit_gib.size());
        aggregate.cpu_submit_to_validate_gib_per_s_p50 = p50_of(submit_gib);
        aggregate.cpu_submit_to_validate_gib_per_s_min = *std::min_element(submit_gib.begin(), submit_gib.end());
        aggregate.cpu_submit_to_validate_gib_per_s_max = *std::max_element(submit_gib.begin(), submit_gib.end());
    }
    if (!fill_ms.empty()) {
        aggregate.cpu_fill_to_validate_ms_avg =
            std::accumulate(fill_ms.begin(), fill_ms.end(), 0.0) / static_cast<double>(fill_ms.size());
        aggregate.cpu_fill_to_validate_ms_p50 = p50_of(fill_ms);
        aggregate.cpu_fill_to_validate_ms_min = *std::min_element(fill_ms.begin(), fill_ms.end());
        aggregate.cpu_fill_to_validate_ms_max = *std::max_element(fill_ms.begin(), fill_ms.end());
    }
    if (!fill_gib.empty()) {
        aggregate.cpu_fill_to_validate_gib_per_s_avg =
            std::accumulate(fill_gib.begin(), fill_gib.end(), 0.0) / static_cast<double>(fill_gib.size());
        aggregate.cpu_fill_to_validate_gib_per_s_p50 = p50_of(fill_gib);
        aggregate.cpu_fill_to_validate_gib_per_s_min = *std::min_element(fill_gib.begin(), fill_gib.end());
        aggregate.cpu_fill_to_validate_gib_per_s_max = *std::max_element(fill_gib.begin(), fill_gib.end());
    }
    aggregate.first_iter_gpu_ms = records.front().gpu_copy_ms;
    if (records.size() > 1u) {
        double steady = 0.0;
        for (size_t i = 1; i < records.size(); ++i) {
            steady += records[i].gpu_copy_ms;
        }
        aggregate.steady_state_gpu_ms_avg = steady / static_cast<double>(records.size() - 1u);
    } else {
        aggregate.steady_state_gpu_ms_avg = records.front().gpu_copy_ms;
    }
    aggregate.submit_to_validate_above_theory =
        aggregate.cpu_submit_to_validate_gib_per_s_avg > kTheoryGiBPerS;
    aggregate.fill_to_validate_above_theory =
        aggregate.cpu_fill_to_validate_gib_per_s_avg > kTheoryGiBPerS;
    if (aggregate.cpu_submit_to_validate_gib_per_s_avg > 0.0 &&
        is_finite(aggregate.cpu_submit_to_validate_gib_per_s_avg)) {
        aggregate.timestamp_vs_completion_ratio =
            aggregate.gpu_copy_gib_per_s_avg / aggregate.cpu_submit_to_validate_gib_per_s_avg;
    }
    aggregate.validation_passed = validation_passed;
    aggregate.validation_failures = validation_failures;
    aggregate.accepted =
        validation_passed &&
        aggregate.gpu_copy_ms_avg > 0.0 &&
        is_finite(aggregate.gpu_copy_ms_avg) &&
        is_finite(aggregate.gpu_copy_gib_per_s_avg);
    return aggregate;
}

MeasurementResult measure_once(Runtime& runtime, const Options& options) {
    MeasurementResult result;
    if (options.queue == QueueSelection::Copy && !runtime.copy_queue_timestamp_supported) {
        result.status = "unsupported";
        result.notes = "COPY queue timestamp queries are unsupported on this adapter.";
        return result;
    }

    const size_t copy_bytes = options.size_mb * kBytesPerMiB;
    CommandContext measure_context = create_command_context(runtime, options);
    CommandContext aux_context = create_command_context(runtime, options);
    RunResources persistent_resources;
    bool use_persistent_resources = options.reuse != ReuseMode::RecreateEveryIter;
    if (use_persistent_resources) {
        persistent_resources = create_run_resources(runtime, options, copy_bytes);
    }

    auto run_iteration = [&](int logical_iteration, bool collect_metrics, std::vector<IterationRecord>* out_records) {
        RunResources local_resources;
        RunResources* resources_ptr = &persistent_resources;
        if (!use_persistent_resources) {
            local_resources = create_run_resources(runtime, options, copy_bytes);
            resources_ptr = &local_resources;
        }
        RunResources& resources = *resources_ptr;

        const uint64_t seed_id = iteration_seed(options, logical_iteration);
        const auto fill_start = std::chrono::steady_clock::now();
        fill_upload_pattern(resources.upload_ptr, copy_bytes, seed_id);
        fill_upload_pattern(resources.expected_bytes.data(), copy_bytes, seed_id);

        const int pair_index = use_persistent_resources
            ? pair_index_for_iteration(options, logical_iteration)
            : logical_iteration;
        PairResources& pair = resources.pairs[use_persistent_resources ? static_cast<size_t>(pair_index) : 0u];
        const uint64_t dst_offset = dst_offset_for_iteration(options, copy_bytes, logical_iteration);

        if (mode_uses_source_default(options.mode)) {
            prepare_source_default(runtime, aux_context, resources, pair, copy_bytes);
        }

        resources.query_readback_ptr[0] = 0;
        resources.query_readback_ptr[1] = 0;
        reset_command_context(measure_context);
        record_timed_copy(measure_context, resources, pair, options, copy_bytes, dst_offset);
        const bool should_validate_now =
            options.validate_each_iter || !use_persistent_resources || (!collect_metrics && logical_iteration == options.warmup - 1);
        const auto submit_start = std::chrono::steady_clock::now();
        execute_command_list(runtime, measure_context.list.Get());
        if (should_validate_now && mode_uses_dest_default(options.mode)) {
            clear_bytes(resources.validation_readback_ptr, copy_bytes);
            reset_command_context(aux_context);
            record_validation_copy(aux_context, resources, pair, copy_bytes, dst_offset);
            execute_command_list(runtime, aux_context.list.Get());
        }
        wait_for_fence(runtime, signal_queue(runtime));
        const auto gpu_work_end = std::chrono::steady_clock::now();

        const uint64_t start_tick = resources.query_readback_ptr[0];
        const uint64_t end_tick = resources.query_readback_ptr[1];
        if (collect_metrics && end_tick <= start_tick) {
            throw std::runtime_error("Invalid timestamp delta");
        }

        bool validation_passed = true;
        if (should_validate_now) {
            validation_passed = compare_copy_result(resources, pair, options, copy_bytes, dst_offset);
        }
        const auto validation_end = std::chrono::steady_clock::now();

        if (collect_metrics) {
            IterationRecord record;
            record.iteration = static_cast<int>(out_records->size());
            record.start_tick = start_tick;
            record.end_tick = end_tick;
            record.cpu_wall_ms = std::chrono::duration<double, std::milli>(gpu_work_end - submit_start).count();
            record.gpu_copy_ms =
                static_cast<double>(end_tick - start_tick) * 1000.0 / static_cast<double>(runtime.timestamp_frequency_hz);
            record.gpu_copy_gib_per_s = (static_cast<double>(copy_bytes) / kBytesPerGiB) / (record.gpu_copy_ms / 1000.0);
            if (should_validate_now) {
                record.cpu_submit_to_validate_ms =
                    std::chrono::duration<double, std::milli>(validation_end - submit_start).count();
                record.cpu_fill_to_validate_ms =
                    std::chrono::duration<double, std::milli>(validation_end - fill_start).count();
                record.cpu_submit_to_validate_gib_per_s =
                    (static_cast<double>(copy_bytes) / kBytesPerGiB) / (record.cpu_submit_to_validate_ms / 1000.0);
                record.cpu_fill_to_validate_gib_per_s =
                    (static_cast<double>(copy_bytes) / kBytesPerGiB) / (record.cpu_fill_to_validate_ms / 1000.0);
            }
            record.resource_pair_index = use_persistent_resources ? pair_index : logical_iteration;
            record.dst_offset_bytes = dst_offset;
            record.seed_id = seed_id;
            record.validation_passed = should_validate_now ? validation_passed : true;
            out_records->push_back(record);
        }

        if (!use_persistent_resources) {
            destroy_run_resources(local_resources);
        }
    };

    try {
        for (int i = 0; i < options.warmup; ++i) {
            run_iteration(i, false, nullptr);
        }
        for (int i = 0; i < options.iterations; ++i) {
            run_iteration(i + options.warmup, true, &result.per_iteration);
        }
        if (!options.validate_each_iter && use_persistent_resources && !result.per_iteration.empty()) {
            const auto& last = result.per_iteration.back();
            PairResources& pair = persistent_resources.pairs[
                options.reuse == ReuseMode::RotateResourcePairs
                    ? static_cast<size_t>(last.resource_pair_index % options.rotation_depth)
                    : 0u];
            const bool final_validation = validate_copy_result(
                runtime,
                aux_context,
                persistent_resources,
                pair,
                options,
                copy_bytes,
                last.dst_offset_bytes);
            for (auto& record : result.per_iteration) {
                record.validation_passed = final_validation;
            }
        }
        result.aggregate = compute_aggregate(result.per_iteration);
        result.status = result.aggregate.validation_passed ? "ok" : "invalid";
    } catch (const std::exception& ex) {
        result.status = "failed";
        result.error = ex.what();
    }

    if (use_persistent_resources) {
        destroy_run_resources(persistent_resources);
    }
    return result;
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
        << "\"uma\":" << (info.uma ? "true" : "false") << ","
        << "\"cache_coherent_uma\":" << (info.cache_coherent_uma ? "true" : "false") << ","
        << "\"tile_based_renderer\":" << (info.tile_based_renderer ? "true" : "false")
        << "}";
    return oss.str();
}

std::string render_iteration_json(const IterationRecord& record) {
    std::ostringstream oss;
    oss << "{"
        << "\"iteration\":" << record.iteration << ","
        << "\"gpu_copy_ms\":" << format_double(record.gpu_copy_ms) << ","
        << "\"gpu_copy_gib_per_s\":" << format_double(record.gpu_copy_gib_per_s) << ","
        << "\"cpu_wall_ms\":" << format_double(record.cpu_wall_ms) << ","
        << "\"cpu_submit_to_validate_ms\":" << format_double(record.cpu_submit_to_validate_ms) << ","
        << "\"cpu_fill_to_validate_ms\":" << format_double(record.cpu_fill_to_validate_ms) << ","
        << "\"cpu_submit_to_validate_gib_per_s\":" << format_double(record.cpu_submit_to_validate_gib_per_s) << ","
        << "\"cpu_fill_to_validate_gib_per_s\":" << format_double(record.cpu_fill_to_validate_gib_per_s) << ","
        << "\"start_tick\":" << record.start_tick << ","
        << "\"end_tick\":" << record.end_tick << ","
        << "\"resource_pair_index\":" << record.resource_pair_index << ","
        << "\"dst_offset_bytes\":" << record.dst_offset_bytes << ","
        << "\"seed_id\":" << record.seed_id << ","
        << "\"validation_passed\":" << (record.validation_passed ? "true" : "false")
        << "}";
    return oss.str();
}

std::string render_result_json(const Options& options, const Runtime& runtime, const MeasurementResult& measurement) {
    std::ostringstream per_iteration;
    per_iteration << "[";
    for (size_t i = 0; i < measurement.per_iteration.size(); ++i) {
        if (i > 0) {
            per_iteration << ",";
        }
        per_iteration << render_iteration_json(measurement.per_iteration[i]);
    }
    per_iteration << "]";

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(measurement.status) << ","
        << "\"primary_metric\":\"gpu_copy_gib_per_s_avg\","
        << "\"unit\":\"GiB/s\","
        << "\"context\":{"
        << "\"architecture\":" << render_architecture_json(runtime.architecture) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap) << ","
        << "\"default_heap_properties\":" << render_heap_properties_json(runtime.default_heap) << ","
        << "\"readback_heap_properties\":" << render_heap_properties_json(runtime.readback_heap)
        << "},"
        << "\"parameters\":{"
        << "\"mode\":" << quote(mode_name(options.mode)) << ","
        << "\"queue\":" << quote(queue_name(options.queue)) << ","
        << "\"size_mb\":" << options.size_mb << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"reuse\":" << quote(reuse_name(options.reuse)) << ","
        << "\"rotation_depth\":" << options.rotation_depth << ","
        << "\"vary_upload_seed\":" << (options.vary_upload_seed ? "true" : "false") << ","
        << "\"validate_each_iter\":" << (options.validate_each_iter ? "true" : "false")
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"queue_timestamp\","
        << "\"queue_type_used\":" << quote(queue_name(options.queue)) << ","
        << "\"copy_queue_timestamp_supported\":" << (runtime.copy_queue_timestamp_supported ? "true" : "false") << ","
        << "\"timestamp_frequency_hz\":" << runtime.timestamp_frequency_hz << ","
        << "\"per_iteration\":" << per_iteration.str() << ","
        << "\"aggregate\":{"
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"driver_version\":" << quote(runtime.driver_version) << ","
        << "\"gpu_copy_ms_avg\":" << format_double(measurement.aggregate.gpu_copy_ms_avg) << ","
        << "\"gpu_copy_ms_p50\":" << format_double(measurement.aggregate.gpu_copy_ms_p50) << ","
        << "\"gpu_copy_ms_min\":" << format_double(measurement.aggregate.gpu_copy_ms_min) << ","
        << "\"gpu_copy_ms_max\":" << format_double(measurement.aggregate.gpu_copy_ms_max) << ","
        << "\"gpu_copy_gib_per_s_avg\":" << format_double(measurement.aggregate.gpu_copy_gib_per_s_avg) << ","
        << "\"gpu_copy_gib_per_s_p50\":" << format_double(measurement.aggregate.gpu_copy_gib_per_s_p50) << ","
        << "\"gpu_copy_gib_per_s_min\":" << format_double(measurement.aggregate.gpu_copy_gib_per_s_min) << ","
        << "\"gpu_copy_gib_per_s_max\":" << format_double(measurement.aggregate.gpu_copy_gib_per_s_max) << ","
        << "\"cpu_wall_ms_avg\":" << format_double(measurement.aggregate.cpu_wall_ms_avg) << ","
        << "\"cpu_submit_to_validate_ms_avg\":" << format_double(measurement.aggregate.cpu_submit_to_validate_ms_avg) << ","
        << "\"cpu_submit_to_validate_ms_p50\":" << format_double(measurement.aggregate.cpu_submit_to_validate_ms_p50) << ","
        << "\"cpu_submit_to_validate_ms_min\":" << format_double(measurement.aggregate.cpu_submit_to_validate_ms_min) << ","
        << "\"cpu_submit_to_validate_ms_max\":" << format_double(measurement.aggregate.cpu_submit_to_validate_ms_max) << ","
        << "\"cpu_submit_to_validate_gib_per_s_avg\":" << format_double(measurement.aggregate.cpu_submit_to_validate_gib_per_s_avg) << ","
        << "\"cpu_submit_to_validate_gib_per_s_p50\":" << format_double(measurement.aggregate.cpu_submit_to_validate_gib_per_s_p50) << ","
        << "\"cpu_submit_to_validate_gib_per_s_min\":" << format_double(measurement.aggregate.cpu_submit_to_validate_gib_per_s_min) << ","
        << "\"cpu_submit_to_validate_gib_per_s_max\":" << format_double(measurement.aggregate.cpu_submit_to_validate_gib_per_s_max) << ","
        << "\"cpu_fill_to_validate_ms_avg\":" << format_double(measurement.aggregate.cpu_fill_to_validate_ms_avg) << ","
        << "\"cpu_fill_to_validate_ms_p50\":" << format_double(measurement.aggregate.cpu_fill_to_validate_ms_p50) << ","
        << "\"cpu_fill_to_validate_ms_min\":" << format_double(measurement.aggregate.cpu_fill_to_validate_ms_min) << ","
        << "\"cpu_fill_to_validate_ms_max\":" << format_double(measurement.aggregate.cpu_fill_to_validate_ms_max) << ","
        << "\"cpu_fill_to_validate_gib_per_s_avg\":" << format_double(measurement.aggregate.cpu_fill_to_validate_gib_per_s_avg) << ","
        << "\"cpu_fill_to_validate_gib_per_s_p50\":" << format_double(measurement.aggregate.cpu_fill_to_validate_gib_per_s_p50) << ","
        << "\"cpu_fill_to_validate_gib_per_s_min\":" << format_double(measurement.aggregate.cpu_fill_to_validate_gib_per_s_min) << ","
        << "\"cpu_fill_to_validate_gib_per_s_max\":" << format_double(measurement.aggregate.cpu_fill_to_validate_gib_per_s_max) << ","
        << "\"first_iter_gpu_ms\":" << format_double(measurement.aggregate.first_iter_gpu_ms) << ","
        << "\"steady_state_gpu_ms_avg\":" << format_double(measurement.aggregate.steady_state_gpu_ms_avg) << ","
        << "\"submit_to_validate_above_theory\":" << (measurement.aggregate.submit_to_validate_above_theory ? "true" : "false") << ","
        << "\"fill_to_validate_above_theory\":" << (measurement.aggregate.fill_to_validate_above_theory ? "true" : "false") << ","
        << "\"timestamp_vs_completion_ratio\":" << format_double(measurement.aggregate.timestamp_vs_completion_ratio) << ","
        << "\"validation_passed\":" << (measurement.aggregate.validation_passed ? "true" : "false") << ","
        << "\"validation_failures\":" << measurement.aggregate.validation_failures << ","
        << "\"accepted\":" << (measurement.aggregate.accepted ? "true" : "false") << ","
        << "\"architecture\":" << render_architecture_json(runtime.architecture) << ","
        << "\"upload_heap_properties\":" << render_heap_properties_json(runtime.upload_heap) << ","
        << "\"default_heap_properties\":" << render_heap_properties_json(runtime.default_heap) << ","
        << "\"readback_heap_properties\":" << render_heap_properties_json(runtime.readback_heap)
        << "}"
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (measurement.aggregate.validation_passed ? "true" : "false")
        << "}";
    if (!measurement.error.empty() || !measurement.notes.empty()) {
        oss << ",\"notes\":[";
        bool wrote = false;
        if (!measurement.error.empty()) {
            oss << quote(measurement.error);
            wrote = true;
        }
        if (!measurement.notes.empty()) {
            if (wrote) {
                oss << ",";
            }
            oss << quote(measurement.notes);
        }
        oss << "]";
    }
    oss << "}";
    return oss.str();
}

std::string make_error_json(const Options& options, const std::string& message) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"gpu_copy_gib_per_s_avg\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"mode\":" << quote(mode_name(options.mode)) << ","
        << "\"queue\":" << quote(queue_name(options.queue)) << ","
        << "\"size_mb\":" << options.size_mb << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"reuse\":" << quote(reuse_name(options.reuse)) << ","
        << "\"rotation_depth\":" << options.rotation_depth << ","
        << "\"vary_upload_seed\":" << (options.vary_upload_seed ? "true" : "false") << ","
        << "\"validate_each_iter\":" << (options.validate_each_iter ? "true" : "false")
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"queue_timestamp\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
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
        runtime = create_runtime(options);
        const MeasurementResult measurement = measure_once(runtime, options);
        emit_json(render_result_json(options, runtime, measurement));
        destroy_runtime(runtime);
        return measurement.status == "ok" ? 0 : (measurement.status == "unsupported" ? 2 : 1);
    } catch (const std::exception& ex) {
        emit_json(make_error_json(options, ex.what()));
        destroy_runtime(runtime);
        return 1;
    }
}
