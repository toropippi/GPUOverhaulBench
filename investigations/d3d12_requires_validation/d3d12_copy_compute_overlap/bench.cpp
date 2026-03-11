#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <d3d12.h>
#include <d3dcompiler.h>
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
constexpr UINT kThreadsPerGroup = 256;
constexpr UINT kDispatchGroups = 1024;
constexpr int kCalibrationPassLimit = 8;

const char* kShaderSource = R"HLSL(
RWStructuredBuffer<uint> output : register(u0);
cbuffer Params : register(b0) {
    uint loops;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint tid = dispatchThreadId.x;
    uint x = 0x12345678u ^ (tid * 747796405u + 2891336453u);
    uint y = 0x9E3779B9u + (tid * 1664525u + 1013904223u);
    uint z = 0xA5A5A5A5u ^ (tid * 2246822519u + 3266489917u);
    [loop]
    for (uint i = 0; i < loops; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        y = y * 1664525u + 1013904223u;
        z ^= x + 0x9E3779B9u + (z << 6) + (z >> 2);
    }
    output[tid] = x ^ y ^ z;
}
)HLSL";

struct Options {
    int iterations = 50;
    int warmup = 5;
    std::vector<size_t> sizes_mb = {128, 512, 1024};
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
    uint32_t loop_count = 0;
};

struct OverlapStats {
    bool success = false;
    std::string error;
    double avg_wall_ms = 0.0;
    double copy_gib_per_s = 0.0;
    double wall_vs_solo_sum_ratio = 0.0;
    double wall_vs_solo_max_ratio = 0.0;
};

struct DirectionRow {
    WallStats copy_solo;
    KernelStats kernel_solo;
    OverlapStats overlap;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    DirectionRow h2d_like;
    DirectionRow d2h_like;
};

struct Runtime {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> copy_queue;
    ComPtr<ID3D12CommandQueue> compute_queue;
    ComPtr<ID3D12Fence> copy_fence;
    ComPtr<ID3D12Fence> compute_fence;
    UINT64 copy_fence_value = 0;
    UINT64 compute_fence_value = 0;
    HANDLE copy_event = nullptr;
    HANDLE compute_event = nullptr;
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> pipeline_state;
    UINT total_threads = kThreadsPerGroup * kDispatchGroups;
    std::string adapter_name;
};

struct Resources {
    ComPtr<ID3D12Resource> upload_buffer;
    ComPtr<ID3D12Resource> default_buffer;
    ComPtr<ID3D12Resource> readback_buffer;
    ComPtr<ID3D12Resource> kernel_output_buffer;
    ComPtr<ID3D12Resource> kernel_output_readback;
    uint8_t* upload_ptr = nullptr;
    uint8_t* readback_ptr = nullptr;
    uint32_t* kernel_output_ptr = nullptr;
};

void check_hr(HRESULT hr, const char* context);
std::string json_escape(const std::string& value);
std::string quote(const std::string& value);
std::string format_double(double value, int precision = 6);
std::string sizes_to_json(const std::vector<size_t>& sizes_mb);
bool starts_with(const std::string& value, const std::string& prefix);
std::vector<size_t> parse_sizes_mb(const std::string& text);
Options parse_common_args(int argc, char** argv);
int effective_iterations(size_t size_mb, int requested_iterations);
int effective_warmup(size_t size_mb, int requested_warmup);
std::string narrow_from_wide(const wchar_t* wide);
void emit_json(const std::string& json);
std::string make_error_json(const std::string& status, const std::string& message, const Options& options, const std::string& primary_metric);
Runtime create_runtime();
void destroy_runtime(Runtime& runtime);
ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap_type, UINT64 bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initial_state);
void wait_for_fence(ID3D12Fence* fence, UINT64 value, HANDLE event_handle);
UINT64 signal_queue(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64* fence_value);
ComPtr<ID3D12GraphicsCommandList> create_command_list(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator** allocator_out);
void reset_list(ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list, ID3D12PipelineState* pipeline = nullptr);
void transition_buffer(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource);
Resources create_resources(Runtime& runtime, size_t copy_bytes);
void destroy_resources(Resources& resources);
void execute_copy_list_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list);
void execute_compute_list_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list);
void fill_host_buffer(uint8_t* ptr, size_t bytes, uint8_t value);
void initialize_default_from_upload(Runtime& runtime, Resources& resources, size_t bytes, uint8_t fill_value, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
WallStats measure_copy_wall(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, int warmup, int iterations, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
KernelStats measure_kernel_wall(Runtime& runtime, Resources& resources, uint32_t loops, int warmup, int iterations, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
KernelStats calibrate_kernel_to_target(Runtime& runtime, Resources& resources, double target_ms, int warmup, int iterations, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
OverlapStats measure_overlap_wall(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, uint32_t loops, int warmup, int iterations, const WallStats& copy_solo, const KernelStats& kernel_solo, ID3D12CommandAllocator* copy_allocator, ID3D12GraphicsCommandList* copy_list, ID3D12CommandAllocator* compute_allocator, ID3D12GraphicsCommandList* compute_list);
bool validate_h2d_copy(Runtime& runtime, Resources& resources, size_t bytes, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
bool validate_d2h_copy(Runtime& runtime, Resources& resources, size_t bytes, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
bool validate_kernel_output(Runtime& runtime, Resources& resources, uint32_t loops, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
DirectionRow run_direction_case(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, int warmup, int iterations, ID3D12CommandAllocator* copy_allocator, ID3D12GraphicsCommandList* copy_list, ID3D12CommandAllocator* compute_allocator, ID3D12GraphicsCommandList* compute_list);
std::string render_wall_stats_json(const WallStats& stats);
std::string render_kernel_stats_json(const KernelStats& stats);
std::string render_overlap_stats_json(const OverlapStats& stats);
std::string render_direction_json(const DirectionRow& row);
bool direction_ok(const DirectionRow& row);
std::string render_json(const Options& options, const Runtime& runtime, const std::vector<CaseRow>& rows, bool validation_passed);

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

std::string format_double(double value, int precision) {
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
    if (options.sizes_mb.empty()) {
        options.sizes_mb.assign(kDefaultSizesMb.begin(), kDefaultSizesMb.end());
    }
    return options;
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
        << "\"api\":\"d3d12\","
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

Runtime create_runtime() {
    Runtime runtime;
    UINT factory_flags = 0;
#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> debug_controller;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_controller)))) {
            debug_controller->EnableDebugLayer();
            factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
        }
    }
#endif

    check_hr(CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&runtime.factory)), "CreateDXGIFactory2");

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
            break;
        }
    }
    if (!runtime.adapter) {
        throw std::runtime_error("No hardware D3D12 adapter found");
    }

    check_hr(D3D12CreateDevice(runtime.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&runtime.device)), "D3D12CreateDevice");

    D3D12_COMMAND_QUEUE_DESC copy_desc{};
    copy_desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
    check_hr(runtime.device->CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&runtime.copy_queue)), "CreateCommandQueue(copy)");

    D3D12_COMMAND_QUEUE_DESC compute_desc{};
    compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    check_hr(runtime.device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&runtime.compute_queue)), "CreateCommandQueue(compute)");

    check_hr(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.copy_fence)), "CreateFence(copy)");
    check_hr(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.compute_fence)), "CreateFence(compute)");

    runtime.copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    runtime.compute_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (runtime.copy_event == nullptr || runtime.compute_event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }

    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT shader_hr = D3DCompile(
        kShaderSource,
        std::strlen(kShaderSource),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shader_blob,
        &error_blob);
    if (FAILED(shader_hr)) {
        std::string message = "D3DCompile failed";
        if (error_blob && error_blob->GetBufferSize() > 0) {
            message += ": ";
            message.append(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize());
        }
        throw std::runtime_error(message);
    }

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].Descriptor.RegisterSpace = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Constants.ShaderRegister = 0;
    params[1].Constants.RegisterSpace = 0;
    params[1].Constants.Num32BitValues = 1;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 2;
    root_desc.pParameters = params;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serialized_root;
    ComPtr<ID3DBlob> root_error;
    check_hr(
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized_root, &root_error),
        "D3D12SerializeRootSignature");
    check_hr(
        runtime.device->CreateRootSignature(
            0,
            serialized_root->GetBufferPointer(),
            serialized_root->GetBufferSize(),
            IID_PPV_ARGS(&runtime.root_signature)),
        "CreateRootSignature");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = runtime.root_signature.Get();
    pso_desc.CS.pShaderBytecode = shader_blob->GetBufferPointer();
    pso_desc.CS.BytecodeLength = shader_blob->GetBufferSize();
    check_hr(runtime.device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&runtime.pipeline_state)), "CreateComputePipelineState");

    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.copy_event != nullptr) {
        CloseHandle(runtime.copy_event);
        runtime.copy_event = nullptr;
    }
    if (runtime.compute_event != nullptr) {
        CloseHandle(runtime.compute_event);
        runtime.compute_event = nullptr;
    }
}

ComPtr<ID3D12Resource> create_buffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heap_type,
    UINT64 bytes,
    D3D12_RESOURCE_FLAGS flags,
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
    desc.Flags = flags;

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

void wait_for_fence(ID3D12Fence* fence, UINT64 value, HANDLE event_handle) {
    if (fence->GetCompletedValue() >= value) {
        return;
    }
    check_hr(fence->SetEventOnCompletion(value, event_handle), "SetEventOnCompletion");
    if (WaitForSingleObject(event_handle, INFINITE) != WAIT_OBJECT_0) {
        throw std::runtime_error("WaitForSingleObject failed");
    }
}

UINT64 signal_queue(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64* fence_value) {
    *fence_value += 1;
    check_hr(queue->Signal(fence, *fence_value), "CommandQueue::Signal");
    return *fence_value;
}

ComPtr<ID3D12GraphicsCommandList> create_command_list(
    ID3D12Device* device,
    D3D12_COMMAND_LIST_TYPE type,
    ID3D12CommandAllocator** allocator_out) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    check_hr(list->Close(), "CommandList::Close(initial)");
    *allocator_out = allocator.Detach();
    return list;
}

void reset_list(ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list, ID3D12PipelineState* pipeline) {
    check_hr(allocator->Reset(), "CommandAllocator::Reset");
    check_hr(list->Reset(allocator, pipeline), "CommandList::Reset");
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

void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    list->ResourceBarrier(1, &barrier);
}

Resources create_resources(Runtime& runtime, size_t copy_bytes) {
    Resources resources;
    resources.upload_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_UPLOAD,
        static_cast<UINT64>(copy_bytes),
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    resources.default_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        static_cast<UINT64>(copy_bytes),
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON);
    resources.readback_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        static_cast<UINT64>(copy_bytes),
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST);
    resources.kernel_output_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        static_cast<UINT64>(runtime.total_threads) * sizeof(uint32_t),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    resources.kernel_output_readback = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_READBACK,
        static_cast<UINT64>(runtime.total_threads) * sizeof(uint32_t),
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST);

    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "Map(upload)");
    check_hr(resources.readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.readback_ptr)), "Map(readback)");
    check_hr(resources.kernel_output_readback->Map(0, nullptr, reinterpret_cast<void**>(&resources.kernel_output_ptr)), "Map(kernel readback)");
    return resources;
}

void destroy_resources(Resources& resources) {
    if (resources.kernel_output_readback) {
        resources.kernel_output_readback->Unmap(0, nullptr);
    }
    if (resources.readback_buffer) {
        resources.readback_buffer->Unmap(0, nullptr);
    }
    if (resources.upload_buffer) {
        resources.upload_buffer->Unmap(0, nullptr);
    }
}

void execute_copy_list_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "Copy list close");
    ID3D12CommandList* lists[] = {list};
    runtime.copy_queue->ExecuteCommandLists(1, lists);
    UINT64 fence_value = signal_queue(runtime.copy_queue.Get(), runtime.copy_fence.Get(), &runtime.copy_fence_value);
    wait_for_fence(runtime.copy_fence.Get(), fence_value, runtime.copy_event);
}

void execute_compute_list_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "Compute list close");
    ID3D12CommandList* lists[] = {list};
    runtime.compute_queue->ExecuteCommandLists(1, lists);
    UINT64 fence_value = signal_queue(runtime.compute_queue.Get(), runtime.compute_fence.Get(), &runtime.compute_fence_value);
    wait_for_fence(runtime.compute_fence.Get(), fence_value, runtime.compute_event);
}

void fill_host_buffer(uint8_t* ptr, size_t bytes, uint8_t value) {
    std::memset(ptr, value, bytes);
}

void initialize_default_from_upload(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    uint8_t fill_value,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    fill_host_buffer(resources.upload_ptr, bytes, fill_value);
    reset_list(allocator, list, nullptr);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    execute_copy_list_and_wait(runtime, list);
}

WallStats measure_copy_wall(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    int warmup,
    int iterations,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    WallStats stats;
    auto record = [&]() {
        if (is_h2d_like) {
            transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
            transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        } else {
            transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
            transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
    };

    for (int i = 0; i < warmup; ++i) {
        reset_list(allocator, list, nullptr);
        record();
        execute_copy_list_and_wait(runtime, list);
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        reset_list(allocator, list, nullptr);
        record();
        const auto start = std::chrono::steady_clock::now();
        execute_copy_list_and_wait(runtime, list);
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
    Runtime& runtime,
    Resources& resources,
    uint32_t loops,
    int warmup,
    int iterations,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    KernelStats stats;
    auto record = [&]() {
        list->SetComputeRootSignature(runtime.root_signature.Get());
        list->SetPipelineState(runtime.pipeline_state.Get());
        list->SetComputeRootUnorderedAccessView(0, resources.kernel_output_buffer->GetGPUVirtualAddress());
        list->SetComputeRoot32BitConstant(1, loops, 0);
        list->Dispatch(kDispatchGroups, 1, 1);
        uav_barrier(list, resources.kernel_output_buffer.Get());
    };

    for (int i = 0; i < warmup; ++i) {
        reset_list(allocator, list, runtime.pipeline_state.Get());
        record();
        execute_compute_list_and_wait(runtime, list);
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        reset_list(allocator, list, runtime.pipeline_state.Get());
        record();
        const auto start = std::chrono::steady_clock::now();
        execute_compute_list_and_wait(runtime, list);
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.loop_count = loops;
    return stats;
}

KernelStats calibrate_kernel_to_target(
    Runtime& runtime,
    Resources& resources,
    double target_ms,
    int warmup,
    int iterations,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    uint32_t loops = 1u << 16;
    KernelStats stats;
    for (int pass = 0; pass < kCalibrationPassLimit; ++pass) {
        stats = measure_kernel_wall(runtime, resources, loops, warmup, iterations, allocator, list);
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
        uint32_t next_loops = static_cast<uint32_t>(scaled);
        if (next_loops == loops) {
            next_loops += 1;
        }
        loops = next_loops;
    }
    return stats;
}

OverlapStats measure_overlap_wall(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    uint32_t loops,
    int warmup,
    int iterations,
    const WallStats& copy_solo,
    const KernelStats& kernel_solo,
    ID3D12CommandAllocator* copy_allocator,
    ID3D12GraphicsCommandList* copy_list,
    ID3D12CommandAllocator* compute_allocator,
    ID3D12GraphicsCommandList* compute_list) {
    OverlapStats stats;
    auto record_copy = [&]() {
        if (is_h2d_like) {
            transition_buffer(copy_list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            copy_list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
            transition_buffer(copy_list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        } else {
            transition_buffer(copy_list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            copy_list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
            transition_buffer(copy_list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
    };
    auto record_compute = [&]() {
        compute_list->SetComputeRootSignature(runtime.root_signature.Get());
        compute_list->SetPipelineState(runtime.pipeline_state.Get());
        compute_list->SetComputeRootUnorderedAccessView(0, resources.kernel_output_buffer->GetGPUVirtualAddress());
        compute_list->SetComputeRoot32BitConstant(1, loops, 0);
        compute_list->Dispatch(kDispatchGroups, 1, 1);
        uav_barrier(compute_list, resources.kernel_output_buffer.Get());
    };

    for (int i = 0; i < warmup; ++i) {
        reset_list(copy_allocator, copy_list, nullptr);
        reset_list(compute_allocator, compute_list, runtime.pipeline_state.Get());
        record_copy();
        record_compute();
        check_hr(copy_list->Close(), "Overlap warmup copy close");
        check_hr(compute_list->Close(), "Overlap warmup compute close");
        ID3D12CommandList* copy_lists[] = {copy_list};
        ID3D12CommandList* compute_lists[] = {compute_list};
        runtime.copy_queue->ExecuteCommandLists(1, copy_lists);
        runtime.compute_queue->ExecuteCommandLists(1, compute_lists);
        UINT64 copy_fence_value = signal_queue(runtime.copy_queue.Get(), runtime.copy_fence.Get(), &runtime.copy_fence_value);
        UINT64 compute_fence_value = signal_queue(runtime.compute_queue.Get(), runtime.compute_fence.Get(), &runtime.compute_fence_value);
        wait_for_fence(runtime.copy_fence.Get(), copy_fence_value, runtime.copy_event);
        wait_for_fence(runtime.compute_fence.Get(), compute_fence_value, runtime.compute_event);
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        reset_list(copy_allocator, copy_list, nullptr);
        reset_list(compute_allocator, compute_list, runtime.pipeline_state.Get());
        record_copy();
        record_compute();
        check_hr(copy_list->Close(), "Overlap copy close");
        check_hr(compute_list->Close(), "Overlap compute close");

        ID3D12CommandList* copy_lists[] = {copy_list};
        ID3D12CommandList* compute_lists[] = {compute_list};
        const auto start = std::chrono::steady_clock::now();
        runtime.copy_queue->ExecuteCommandLists(1, copy_lists);
        runtime.compute_queue->ExecuteCommandLists(1, compute_lists);
        UINT64 copy_fence_value = signal_queue(runtime.copy_queue.Get(), runtime.copy_fence.Get(), &runtime.copy_fence_value);
        UINT64 compute_fence_value = signal_queue(runtime.compute_queue.Get(), runtime.compute_fence.Get(), &runtime.compute_fence_value);
        wait_for_fence(runtime.copy_fence.Get(), copy_fence_value, runtime.copy_event);
        wait_for_fence(runtime.compute_fence.Get(), compute_fence_value, runtime.compute_event);
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    stats.success = true;
    stats.avg_wall_ms = total_ms / static_cast<double>(iterations);
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    stats.copy_gib_per_s = gib / (stats.avg_wall_ms / 1000.0);
    stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (copy_solo.avg_ms + kernel_solo.avg_ms);
    stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(copy_solo.avg_ms, kernel_solo.avg_ms);
    return stats;
}

bool validate_h2d_copy(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    fill_host_buffer(resources.upload_ptr, bytes, 0x3C);
    std::memset(resources.readback_ptr, 0x00, bytes);
    reset_list(allocator, list, nullptr);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    execute_copy_list_and_wait(runtime, list);
    for (size_t i = 0; i < bytes; ++i) {
        if (resources.readback_ptr[i] != 0x3C) {
            return false;
        }
    }
    return true;
}

bool validate_d2h_copy(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    initialize_default_from_upload(runtime, resources, bytes, 0x5A, allocator, list);
    std::memset(resources.readback_ptr, 0x00, bytes);
    reset_list(allocator, list, nullptr);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
    transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    execute_copy_list_and_wait(runtime, list);
    for (size_t i = 0; i < bytes; ++i) {
        if (resources.readback_ptr[i] != 0x5A) {
            return false;
        }
    }
    return true;
}

bool validate_kernel_output(
    Runtime& runtime,
    Resources& resources,
    uint32_t loops,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    std::memset(resources.kernel_output_ptr, 0x00, static_cast<size_t>(runtime.total_threads) * sizeof(uint32_t));
    reset_list(allocator, list, runtime.pipeline_state.Get());
    list->SetComputeRootSignature(runtime.root_signature.Get());
    list->SetPipelineState(runtime.pipeline_state.Get());
    list->SetComputeRootUnorderedAccessView(0, resources.kernel_output_buffer->GetGPUVirtualAddress());
    list->SetComputeRoot32BitConstant(1, loops, 0);
    list->Dispatch(kDispatchGroups, 1, 1);
    uav_barrier(list, resources.kernel_output_buffer.Get());
    transition_buffer(list, resources.kernel_output_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(
        resources.kernel_output_readback.Get(),
        0,
        resources.kernel_output_buffer.Get(),
        0,
        static_cast<UINT64>(runtime.total_threads) * sizeof(uint32_t));
    transition_buffer(list, resources.kernel_output_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    execute_compute_list_and_wait(runtime, list);
    for (UINT i = 0; i < runtime.total_threads; ++i) {
        if (resources.kernel_output_ptr[i] != 0u) {
            return true;
        }
    }
    return false;
}

DirectionRow run_direction_case(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    int warmup,
    int iterations,
    ID3D12CommandAllocator* copy_allocator,
    ID3D12GraphicsCommandList* copy_list,
    ID3D12CommandAllocator* compute_allocator,
    ID3D12GraphicsCommandList* compute_list) {
    DirectionRow row;
    row.copy_solo = measure_copy_wall(runtime, resources, bytes, is_h2d_like, warmup, iterations, copy_allocator, copy_list);
    if (!row.copy_solo.success) {
        return row;
    }
    row.kernel_solo = calibrate_kernel_to_target(runtime, resources, row.copy_solo.avg_ms, warmup, iterations, compute_allocator, compute_list);
    if (!row.kernel_solo.success) {
        return row;
    }
    row.overlap = measure_overlap_wall(
        runtime,
        resources,
        bytes,
        is_h2d_like,
        row.kernel_solo.loop_count,
        warmup,
        iterations,
        row.copy_solo,
        row.kernel_solo,
        copy_allocator,
        copy_list,
        compute_allocator,
        compute_list);
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

bool direction_ok(const DirectionRow& row) {
    return row.copy_solo.success && row.kernel_solo.success && row.overlap.success;
}

std::string render_json(const Options& options, const Runtime& runtime, const std::vector<CaseRow>& rows, bool validation_passed) {
    double min_h2d = 0.0;
    double min_d2h = 0.0;
    bool have_h2d = false;
    bool have_d2h = false;

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
              << "\"h2d_like\":" << render_direction_json(row.h2d_like) << ","
              << "\"d2h_like\":" << render_direction_json(row.d2h_like)
              << "}";

        if (direction_ok(row.h2d_like)) {
            min_h2d = have_h2d ? std::min(min_h2d, row.h2d_like.overlap.wall_vs_solo_sum_ratio) : row.h2d_like.overlap.wall_vs_solo_sum_ratio;
            have_h2d = true;
        }
        if (direction_ok(row.d2h_like)) {
            min_d2h = have_d2h ? std::min(min_d2h, row.d2h_like.overlap.wall_vs_solo_sum_ratio) : row.d2h_like.overlap.wall_vs_solo_sum_ratio;
            have_d2h = true;
        }
    }
    cases << "]";

    const double min_all = std::min(min_h2d, min_d2h);
    const bool all_ok = std::all_of(rows.begin(), rows.end(), [](const CaseRow& row) {
        return direction_ok(row.h2d_like) && direction_ok(row.d2h_like);
    });

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote((all_ok && validation_passed) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"min_wall_vs_solo_sum_ratio\","
        << "\"unit\":\"ratio\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"copy_directions\":[\"H2D-like\",\"D2H-like\"],"
        << "\"queue_count\":2,"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"wall_clock\","
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"min_h2d_wall_vs_solo_sum_ratio\":" << format_double(min_h2d) << ","
        << "\"min_d2h_wall_vs_solo_sum_ratio\":" << format_double(min_d2h) << ","
        << "\"min_wall_vs_solo_sum_ratio\":" << format_double(min_all) << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "}"
        << "}";
    return oss.str();
}

// IMPLEMENTATION_MARKER

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    Runtime runtime{};
    try {
        options = parse_common_args(argc, argv);
        runtime = create_runtime();

        std::vector<CaseRow> rows;
        bool validation_passed = true;

        ComPtr<ID3D12CommandAllocator> copy_allocator;
        ComPtr<ID3D12GraphicsCommandList> copy_list = create_command_list(
            runtime.device.Get(), D3D12_COMMAND_LIST_TYPE_COPY, copy_allocator.GetAddressOf());
        ComPtr<ID3D12CommandAllocator> compute_allocator;
        ComPtr<ID3D12GraphicsCommandList> compute_list = create_command_list(
            runtime.device.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE, compute_allocator.GetAddressOf());

        for (size_t size_mb : options.sizes_mb) {
            const size_t copy_bytes = size_mb * 1024ull * 1024ull;
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);

            Resources resources = create_resources(runtime, copy_bytes);

            fill_host_buffer(resources.upload_ptr, copy_bytes, 0x3C);
            row.h2d_like = run_direction_case(
                runtime,
                resources,
                copy_bytes,
                true,
                row.warmup,
                row.iterations,
                copy_allocator.Get(),
                copy_list.Get(),
                compute_allocator.Get(),
                compute_list.Get());

            initialize_default_from_upload(runtime, resources, copy_bytes, 0x5A, copy_allocator.Get(), copy_list.Get());
            row.d2h_like = run_direction_case(
                runtime,
                resources,
                copy_bytes,
                false,
                row.warmup,
                row.iterations,
                copy_allocator.Get(),
                copy_list.Get(),
                compute_allocator.Get(),
                compute_list.Get());

            if (!validate_h2d_copy(runtime, resources, copy_bytes, copy_allocator.Get(), copy_list.Get())) {
                validation_passed = false;
            }
            if (!validate_d2h_copy(runtime, resources, copy_bytes, copy_allocator.Get(), copy_list.Get())) {
                validation_passed = false;
            }
            uint32_t validation_loops = std::max(row.h2d_like.kernel_solo.loop_count, row.d2h_like.kernel_solo.loop_count);
            if (!validate_kernel_output(runtime, resources, validation_loops, compute_allocator.Get(), compute_list.Get())) {
                validation_passed = false;
            }

            destroy_resources(resources);
            rows.push_back(row);
        }

        emit_json(render_json(options, runtime, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json("failed", ex.what(), options, "min_wall_vs_solo_sum_ratio"));
        destroy_runtime(runtime);
        return 1;
    }
}
