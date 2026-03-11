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
constexpr double kCudaRefH2DGiBps = 53.310166;
constexpr double kTheoryGiBps = 58.687292;
constexpr double kTolerance = 0.30;
constexpr uint32_t kThreadsPerGroup = 256;
constexpr uint32_t kSeed = 0x31415926u;

const char* kShaderSource = R"HLSL(
ByteAddressBuffer source_buffer : register(t0);
RWByteAddressBuffer result_buffer : register(u0);

cbuffer Params : register(b0) {
    uint word_count;
    uint seed;
    uint total_threads;
    uint pad0;
};

groupshared uint shared_hash[256];

uint mix_word(uint value, uint index, uint local_seed) {
    uint mixed = value ^ (index * 0x9E3779B9u) ^ local_seed;
    mixed ^= mixed >> 16;
    mixed *= 0x7FEB352Du;
    mixed ^= mixed >> 15;
    mixed *= 0x846CA68Bu;
    mixed ^= mixed >> 16;
    return mixed;
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID, uint3 dispatch_thread_id : SV_DispatchThreadID) {
    uint local = 0u;
    uint stride = max(total_threads, 1u);
    [loop]
    for (uint index = dispatch_thread_id.x; index < word_count; index += stride) {
        uint value = source_buffer.Load(index * 4u);
        local ^= mix_word(value, index, seed);
    }

    shared_hash[group_thread_id.x] = local;
    GroupMemoryBarrierWithGroupSync();

    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (group_thread_id.x < offset) {
            shared_hash[group_thread_id.x] ^= shared_hash[group_thread_id.x + offset];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (group_thread_id.x == 0u) {
        result_buffer.Store(group_id.x * 4u, shared_hash[0]);
    }
}
)HLSL";

struct Options {
    int iterations = 20;
    int warmup = 3;
    std::vector<size_t> sizes_mb = {128, 512, 1024};
};

struct MeasureStats {
    bool success = false;
    bool validation_passed = false;
    bool within_cuda_reference = false;
    bool within_theoretical_limit = false;
    bool accepted = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
    uint32_t expected_value = 0;
    uint32_t observed_value = 0;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    double fill_shadow_ms = 0.0;
    double fill_shadow_gib_per_s = 0.0;
    double expected_hash_ms = 0.0;
    double expected_hash_gib_per_s = 0.0;
    double validation_path_overhead_ms = 0.0;
    double validation_path_overhead_ratio = 0.0;
    MeasureStats upload_write_to_consume;
    MeasureStats h2d_only;
    MeasureStats end_to_end;
};

struct Runtime {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> direct_queue;
    ComPtr<ID3D12Fence> fence;
    UINT64 fence_value = 0;
    HANDLE fence_event = nullptr;
    ComPtr<ID3D12RootSignature> root_signature;
    ComPtr<ID3D12PipelineState> pipeline_state;
    std::string adapter_name;
};

struct Resources {
    ComPtr<ID3D12Resource> upload_buffer;
    ComPtr<ID3D12Resource> default_buffer;
    ComPtr<ID3D12Resource> partial_buffer;
    ComPtr<ID3D12Resource> result_buffer;
    ComPtr<ID3D12Resource> readback_buffer;
    uint8_t* upload_ptr = nullptr;
    uint32_t* readback_ptr = nullptr;
    uint32_t partial_count = 0;
    std::vector<uint8_t> shadow_bytes;
};

void check_hr(HRESULT hr, const char* context);
std::string json_escape(const std::string& value);
std::string quote(const std::string& value);
std::string format_double(double value, int precision = 6);
bool starts_with(const std::string& value, const std::string& prefix);
std::vector<size_t> parse_sizes_mb(const std::string& text);
Options parse_common_args(int argc, char** argv);
int effective_iterations(size_t size_mb, int requested_iterations);
int effective_warmup(size_t size_mb, int requested_warmup);
std::string narrow_from_wide(const wchar_t* wide);
void emit_json(const std::string& json);
std::string make_error_json(const std::string& message, const Options& options);
Runtime create_runtime();
void destroy_runtime(Runtime& runtime);
ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap_type, UINT64 bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initial_state);
ComPtr<ID3D12GraphicsCommandList> create_direct_list(ID3D12Device* device, ID3D12CommandAllocator** allocator_out);
void reset_list(ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list, ID3D12PipelineState* pipeline = nullptr);
void transition_buffer(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource);
void wait_for_fence(ID3D12Fence* fence, UINT64 value, HANDLE event_handle);
UINT64 signal_queue(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64* fence_value);
void execute_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list);
Resources create_resources(Runtime& runtime, size_t bytes);
void destroy_resources(Resources& resources);
void fill_upload_pattern(uint8_t* ptr, size_t bytes);
uint32_t load_word_le(const uint8_t* ptr, uint32_t index);
uint32_t mix_word(uint32_t value, uint32_t index);
uint32_t compute_expected_accumulator(const uint8_t* ptr, size_t bytes);
uint32_t dispatch_group_count(uint32_t word_count);
MeasureStats measure_h2d_only(Runtime& runtime, Resources& resources, size_t bytes, int warmup, int iterations, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
MeasureStats measure_upload_write_to_consume(Runtime& runtime, Resources& resources, size_t bytes, int warmup, int iterations, uint32_t expected_value, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
MeasureStats measure_end_to_end(Runtime& runtime, Resources& resources, size_t bytes, int warmup, int iterations, uint32_t expected_value, ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list);
std::string render_stats_json(const MeasureStats& stats);
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

std::string format_double(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
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

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb >= 1024) {
        return std::max(8, requested_iterations / 2);
    }
    return requested_iterations;
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb >= 1024) {
        return std::max(1, requested_warmup / 2);
    }
    return requested_warmup;
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

void emit_json(const std::string& json) {
    std::cout << json << "\n";
}

std::string make_error_json(const std::string& message, const Options& options) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"best_end_to_end_bandwidth_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"sizes_mb\":[";
    for (size_t i = 0; i < options.sizes_mb.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << options.sizes_mb[i];
    }
    oss << "],"
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup
        << "},"
        << "\"measurement\":{\"timing_backend\":\"cpu_wall_end_to_end\"},"
        << "\"validation\":{\"passed\":false},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
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
            break;
        }
    }
    if (!runtime.adapter) {
        throw std::runtime_error("No hardware D3D12 adapter found");
    }

    check_hr(D3D12CreateDevice(runtime.adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&runtime.device)), "D3D12CreateDevice");

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check_hr(runtime.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&runtime.direct_queue)), "CreateCommandQueue(direct)");
    check_hr(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.fence)), "CreateFence");
    runtime.fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (runtime.fence_event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }

    ComPtr<ID3DBlob> shader_blob;
    ComPtr<ID3DBlob> error_blob;
    HRESULT hr = D3DCompile(
        kShaderSource,
        std::strlen(kShaderSource),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        0,
        0,
        &shader_blob,
        &error_blob);
    if (FAILED(hr)) {
        std::string error = "D3DCompile failed";
        if (error_blob) {
            error.assign(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize());
        }
        throw std::runtime_error(error);
    }

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].Descriptor.RegisterSpace = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].Descriptor.ShaderRegister = 0;
    params[1].Descriptor.RegisterSpace = 0;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.RegisterSpace = 0;
    params[2].Constants.Num32BitValues = 4;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> root_blob;
    ComPtr<ID3DBlob> root_error;
    check_hr(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &root_error), "D3D12SerializeRootSignature");
    check_hr(runtime.device->CreateRootSignature(0, root_blob->GetBufferPointer(), root_blob->GetBufferSize(), IID_PPV_ARGS(&runtime.root_signature)), "CreateRootSignature");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = runtime.root_signature.Get();
    pso_desc.CS = {shader_blob->GetBufferPointer(), shader_blob->GetBufferSize()};
    check_hr(runtime.device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&runtime.pipeline_state)), "CreateComputePipelineState");

    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.fence_event != nullptr) {
        CloseHandle(runtime.fence_event);
        runtime.fence_event = nullptr;
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
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;

    ComPtr<ID3D12Resource> resource;
    check_hr(device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        initial_state,
        nullptr,
        IID_PPV_ARGS(&resource)), "CreateCommittedResource");
    return resource;
}

ComPtr<ID3D12GraphicsCommandList> create_direct_list(ID3D12Device* device, ID3D12CommandAllocator** allocator_out) {
    check_hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator_out)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, *allocator_out, nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    check_hr(list->Close(), "InitialClose");
    return list;
}

void reset_list(ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList* list, ID3D12PipelineState* pipeline) {
    check_hr(allocator->Reset(), "CommandAllocator::Reset");
    check_hr(list->Reset(allocator, pipeline), "CommandList::Reset");
}

void transition_buffer(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
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

void wait_for_fence(ID3D12Fence* fence, UINT64 value, HANDLE event_handle) {
    if (fence->GetCompletedValue() >= value) {
        return;
    }
    check_hr(fence->SetEventOnCompletion(value, event_handle), "Fence::SetEventOnCompletion");
    WaitForSingleObject(event_handle, INFINITE);
}

UINT64 signal_queue(ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64* fence_value) {
    *fence_value += 1;
    check_hr(queue->Signal(fence, *fence_value), "CommandQueue::Signal");
    return *fence_value;
}

void execute_and_wait(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list};
    runtime.direct_queue->ExecuteCommandLists(1, lists);
    const UINT64 value = signal_queue(runtime.direct_queue.Get(), runtime.fence.Get(), &runtime.fence_value);
    wait_for_fence(runtime.fence.Get(), value, runtime.fence_event);
}

Resources create_resources(Runtime& runtime, size_t bytes) {
    Resources resources;
    const uint32_t word_count = static_cast<uint32_t>(bytes / 4ull);
    resources.partial_count = dispatch_group_count(word_count);
    resources.upload_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_UPLOAD, static_cast<UINT64>(bytes), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    resources.default_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_DEFAULT, static_cast<UINT64>(bytes), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    resources.partial_buffer = create_buffer(
        runtime.device.Get(),
        D3D12_HEAP_TYPE_DEFAULT,
        static_cast<UINT64>(resources.partial_count) * sizeof(uint32_t),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON);
    resources.result_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_DEFAULT, 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    resources.readback_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_READBACK, 4, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    resources.shadow_bytes.resize(bytes);

    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "UploadBuffer::Map");
    check_hr(resources.readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.readback_ptr)), "ReadbackBuffer::Map");
    return resources;
}

void destroy_resources(Resources& resources) {
    if (resources.readback_buffer && resources.readback_ptr) {
        resources.readback_buffer->Unmap(0, nullptr);
        resources.readback_ptr = nullptr;
    }
    if (resources.upload_buffer && resources.upload_ptr) {
        resources.upload_buffer->Unmap(0, nullptr);
        resources.upload_ptr = nullptr;
    }
}

void fill_upload_pattern(uint8_t* ptr, size_t bytes) {
    uint32_t state = 0x1234ABCDu;
    for (size_t i = 0; i < bytes; ++i) {
        state = state * 1664525u + 1013904223u;
        ptr[i] = static_cast<uint8_t>((state >> 16) & 0xFFu);
    }
}

uint32_t load_word_le(const uint8_t* ptr, uint32_t index) {
    const size_t offset = static_cast<size_t>(index) * 4ull;
    return static_cast<uint32_t>(ptr[offset]) |
           (static_cast<uint32_t>(ptr[offset + 1]) << 8) |
           (static_cast<uint32_t>(ptr[offset + 2]) << 16) |
           (static_cast<uint32_t>(ptr[offset + 3]) << 24);
}

uint32_t mix_word(uint32_t value, uint32_t index) {
    uint32_t mixed = value ^ (index * 0x9E3779B9u) ^ kSeed;
    mixed ^= mixed >> 16;
    mixed *= 0x7FEB352Du;
    mixed ^= mixed >> 15;
    mixed *= 0x846CA68Bu;
    mixed ^= mixed >> 16;
    return mixed;
}

uint32_t compute_expected_accumulator(const uint8_t* ptr, size_t bytes) {
    const uint32_t word_count = static_cast<uint32_t>(bytes / 4ull);
    const uint32_t partial_count = dispatch_group_count(word_count);
    const uint32_t total_threads = partial_count * kThreadsPerGroup;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(ptr);

    std::vector<uint32_t> partials(static_cast<size_t>(partial_count), 0u);
    for (uint32_t group = 0; group < partial_count; ++group) {
        uint32_t acc = 0u;
        for (uint32_t lane = 0; lane < kThreadsPerGroup; ++lane) {
            for (uint32_t index = group * kThreadsPerGroup + lane; index < word_count; index += total_threads) {
                acc ^= mix_word(words[index], index);
            }
        }
        partials[group] = acc;
    }

    uint32_t final_acc = 0u;
    for (uint32_t index = 0; index < partial_count; ++index) {
        final_acc ^= mix_word(partials[index], index);
    }
    return final_acc;
}

uint32_t dispatch_group_count(uint32_t word_count) {
    if (word_count == 0u) {
        return 1u;
    }
    const uint32_t groups = (word_count + kThreadsPerGroup - 1u) / kThreadsPerGroup;
    return std::min(groups, 65535u);
}

MeasureStats measure_h2d_only(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    int warmup,
    int iterations,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    MeasureStats stats;
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

    auto record_once = [&]() -> double {
        reset_list(allocator, list, nullptr);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        const auto begin = std::chrono::steady_clock::now();
        execute_and_wait(runtime, list);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    for (int i = 0; i < warmup; ++i) {
        (void)record_once();
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        total_ms += record_once();
    }

    stats.success = true;
    stats.validation_passed = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    return stats;
}

MeasureStats measure_upload_write_to_consume(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    int warmup,
    int iterations,
    uint32_t expected,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    MeasureStats stats;
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t word_count = static_cast<uint32_t>(bytes / 4ull);
    const uint32_t partial_count = resources.partial_count;
    const uint32_t total_threads = partial_count * kThreadsPerGroup;

    auto record_once = [&](uint32_t* observed_value_out) -> double {
        if (resources.readback_ptr != nullptr) {
            *resources.readback_ptr = 0u;
        }

        const auto begin = std::chrono::steady_clock::now();
        std::memcpy(resources.upload_ptr, resources.shadow_bytes.data(), bytes);

        reset_list(allocator, list, runtime.pipeline_state.Get());
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(runtime.root_signature.Get());
        list->SetPipelineState(runtime.pipeline_state.Get());
        list->SetComputeRootShaderResourceView(0, resources.default_buffer->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, resources.partial_buffer->GetGPUVirtualAddress());
        const UINT constants[4] = {word_count, kSeed, total_threads, 0};
        list->SetComputeRoot32BitConstants(2, 4, constants, 0);
        list->Dispatch(partial_count, 1, 1);
        uav_barrier(list, resources.partial_buffer.Get());
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootShaderResourceView(0, resources.partial_buffer->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, resources.result_buffer->GetGPUVirtualAddress());
        const UINT reduce_constants[4] = {partial_count, kSeed, kThreadsPerGroup, 0};
        list->SetComputeRoot32BitConstants(2, 4, reduce_constants, 0);
        list->Dispatch(1, 1, 1);
        uav_barrier(list, resources.result_buffer.Get());
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.result_buffer.Get(), 0, 4);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        execute_and_wait(runtime, list);
        const auto end = std::chrono::steady_clock::now();

        if (observed_value_out != nullptr) {
            *observed_value_out = resources.readback_ptr != nullptr ? *resources.readback_ptr : 0u;
        }
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    for (int i = 0; i < warmup; ++i) {
        uint32_t ignored = 0;
        (void)record_once(&ignored);
    }

    double total_ms = 0.0;
    uint32_t observed_last = 0;
    bool all_match = true;
    for (int i = 0; i < iterations; ++i) {
        uint32_t observed = 0;
        total_ms += record_once(&observed);
        observed_last = observed;
        all_match = all_match && (observed == expected);
    }

    stats.success = true;
    stats.validation_passed = all_match;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    stats.expected_value = expected;
    stats.observed_value = observed_last;
    stats.within_cuda_reference = std::fabs(stats.gib_per_s - kCudaRefH2DGiBps) <= (kCudaRefH2DGiBps * kTolerance);
    stats.within_theoretical_limit = stats.gib_per_s <= kTheoryGiBps;
    stats.accepted = stats.validation_passed && stats.within_cuda_reference && stats.within_theoretical_limit;
    return stats;
}

MeasureStats measure_end_to_end(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    int warmup,
    int iterations,
    uint32_t expected,
    ID3D12CommandAllocator* allocator,
    ID3D12GraphicsCommandList* list) {
    MeasureStats stats;
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    const uint32_t word_count = static_cast<uint32_t>(bytes / 4ull);
    const uint32_t partial_count = resources.partial_count;
    const uint32_t total_threads = partial_count * kThreadsPerGroup;

    auto record_once = [&](uint32_t* observed_value_out) -> double {
        if (resources.readback_ptr != nullptr) {
            *resources.readback_ptr = 0u;
        }

        reset_list(allocator, list, runtime.pipeline_state.Get());
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootSignature(runtime.root_signature.Get());
        list->SetPipelineState(runtime.pipeline_state.Get());
        list->SetComputeRootShaderResourceView(0, resources.default_buffer->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, resources.partial_buffer->GetGPUVirtualAddress());
        const UINT constants[4] = {word_count, kSeed, total_threads, 0};
        list->SetComputeRoot32BitConstants(2, 4, constants, 0);
        list->Dispatch(partial_count, 1, 1);
        uav_barrier(list, resources.partial_buffer.Get());
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetComputeRootShaderResourceView(0, resources.partial_buffer->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, resources.result_buffer->GetGPUVirtualAddress());
        const UINT reduce_constants[4] = {partial_count, kSeed, kThreadsPerGroup, 0};
        list->SetComputeRoot32BitConstants(2, 4, reduce_constants, 0);
        list->Dispatch(1, 1, 1);
        uav_barrier(list, resources.result_buffer.Get());
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.result_buffer.Get(), 0, 4);
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition_buffer(list, resources.partial_buffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        transition_buffer(list, resources.result_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);

        const auto begin = std::chrono::steady_clock::now();
        execute_and_wait(runtime, list);
        const auto end = std::chrono::steady_clock::now();

        if (observed_value_out != nullptr) {
            *observed_value_out = resources.readback_ptr != nullptr ? *resources.readback_ptr : 0u;
        }
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    for (int i = 0; i < warmup; ++i) {
        uint32_t ignored = 0;
        (void)record_once(&ignored);
    }

    double total_ms = 0.0;
    uint32_t observed_last = 0;
    bool all_match = true;
    for (int i = 0; i < iterations; ++i) {
        uint32_t observed = 0;
        total_ms += record_once(&observed);
        observed_last = observed;
        all_match = all_match && (observed == expected);
    }

    stats.success = true;
    stats.validation_passed = all_match;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    stats.expected_value = expected;
    stats.observed_value = observed_last;
    stats.within_cuda_reference = std::fabs(stats.gib_per_s - kCudaRefH2DGiBps) <= (kCudaRefH2DGiBps * kTolerance);
    stats.within_theoretical_limit = stats.gib_per_s <= kTheoryGiBps;
    stats.accepted = stats.validation_passed && stats.within_cuda_reference && stats.within_theoretical_limit;
    return stats;
}

std::string render_stats_json(const MeasureStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"accepted\":" << (stats.accepted ? "true" : "false") << ","
        << "\"avg_ms\":" << format_double(stats.avg_ms) << ","
        << "\"expected_value\":" << stats.expected_value << ","
        << "\"gib_per_s\":" << format_double(stats.gib_per_s) << ","
        << "\"observed_value\":" << stats.observed_value << ","
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"validation_passed\":" << (stats.validation_passed ? "true" : "false") << ","
        << "\"within_cuda_reference\":" << (stats.within_cuda_reference ? "true" : "false") << ","
        << "\"within_theoretical_limit\":" << (stats.within_theoretical_limit ? "true" : "false");
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_json(const Options& options, const Runtime& runtime, const std::vector<CaseRow>& rows, bool validation_passed) {
    size_t accepted_case_count = 0;
    double best_end_to_end = 0.0;
    double best_upload_write_to_consume = 0.0;
    double overhead_sum = 0.0;
    size_t overhead_count = 0;

    for (const auto& row : rows) {
        if (row.end_to_end.accepted) {
            ++accepted_case_count;
        }
        best_end_to_end = std::max(best_end_to_end, row.end_to_end.gib_per_s);
        best_upload_write_to_consume = std::max(best_upload_write_to_consume, row.upload_write_to_consume.gib_per_s);
        if (row.h2d_only.success && row.end_to_end.success && row.h2d_only.avg_ms > 0.0) {
            overhead_sum += row.end_to_end.avg_ms / row.h2d_only.avg_ms;
            ++overhead_count;
        }
    }

    const bool overall_ok = validation_passed && accepted_case_count > 0;
    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote(overall_ok ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"best_end_to_end_bandwidth_gib_per_s\","
        << "\"unit\":\"GiB/s\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"copy_direction\":\"H2D-like\","
        << "\"cuda_reference_h2d_gib_per_s\":" << format_double(kCudaRefH2DGiBps) << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"sizes_mb\":[";
    for (size_t i = 0; i < options.sizes_mb.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << options.sizes_mb[i];
    }
    oss << "],"
        << "\"validation_method\":\"gpu_full_buffer_hash_4b_readback\","
        << "\"hash_threads_per_group\":" << kThreadsPerGroup << ","
        << "\"warmup\":" << options.warmup
        << "},"
        << "\"measurement\":{"
        << "\"accepted_case_count\":" << accepted_case_count << ","
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"average_end_to_end_overhead_ratio\":" << format_double(overhead_count == 0 ? 0.0 : (overhead_sum / static_cast<double>(overhead_count))) << ","
        << "\"best_end_to_end_bandwidth_gib_per_s\":" << format_double(best_end_to_end) << ","
        << "\"best_upload_write_to_consume_gib_per_s\":" << format_double(best_upload_write_to_consume) << ","
        << "\"cases\":[";

    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        const auto& row = rows[i];
        const double overhead_ratio = row.h2d_only.avg_ms > 0.0 ? (row.end_to_end.avg_ms / row.h2d_only.avg_ms) : 0.0;
        oss << "{"
            << "\"end_to_end\":" << render_stats_json(row.end_to_end) << ","
            << "\"end_to_end_overhead_ratio\":" << format_double(overhead_ratio) << ","
            << "\"fill_shadow_ms\":" << format_double(row.fill_shadow_ms) << ","
            << "\"fill_shadow_gib_per_s\":" << format_double(row.fill_shadow_gib_per_s) << ","
            << "\"expected_hash_ms\":" << format_double(row.expected_hash_ms) << ","
            << "\"expected_hash_gib_per_s\":" << format_double(row.expected_hash_gib_per_s) << ","
            << "\"h2d_only\":" << render_stats_json(row.h2d_only) << ","
            << "\"iterations\":" << row.iterations << ","
            << "\"size_mb\":" << row.size_mb << ","
            << "\"upload_write_to_consume\":" << render_stats_json(row.upload_write_to_consume) << ","
            << "\"validation_path_overhead_ms\":" << format_double(row.validation_path_overhead_ms) << ","
            << "\"validation_path_overhead_ratio\":" << format_double(row.validation_path_overhead_ratio) << ","
            << "\"warmup\":" << row.warmup
            << "}";
    }

    oss << "],"
        << "\"theoretical_one_way_pcie_5_x16_gib_per_s\":" << format_double(kTheoryGiBps) << ","
        << "\"timing_backend\":\"cpu_wall_end_to_end\""
        << "},"
        << "\"validation\":{\"passed\":" << (validation_passed ? "true" : "false") << "}"
        << "}";
    return oss.str();
}

// -- IMPLEMENTATION MARKER --

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_common_args(argc, argv);
        Runtime runtime = create_runtime();
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list = create_direct_list(runtime.device.Get(), &allocator);

        bool validation_passed = true;
        std::vector<CaseRow> rows;
        rows.reserve(options.sizes_mb.size());

        for (size_t size_mb : options.sizes_mb) {
            const size_t bytes = size_mb * 1024ull * 1024ull;
            Resources resources = create_resources(runtime, bytes);

            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = effective_iterations(size_mb, options.iterations);
            row.warmup = effective_warmup(size_mb, options.warmup);
            const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);

            const auto fill_begin = std::chrono::steady_clock::now();
            fill_upload_pattern(resources.upload_ptr, bytes);
            fill_upload_pattern(resources.shadow_bytes.data(), bytes);
            const auto fill_end = std::chrono::steady_clock::now();
            row.fill_shadow_ms = std::chrono::duration<double, std::milli>(fill_end - fill_begin).count();
            row.fill_shadow_gib_per_s = row.fill_shadow_ms > 0.0 ? (gib * 2.0) / (row.fill_shadow_ms / 1000.0) : 0.0;

            const auto expected_begin = std::chrono::steady_clock::now();
            const uint32_t expected_value = compute_expected_accumulator(resources.shadow_bytes.data(), bytes);
            const auto expected_end = std::chrono::steady_clock::now();
            row.expected_hash_ms = std::chrono::duration<double, std::milli>(expected_end - expected_begin).count();
            row.expected_hash_gib_per_s = row.expected_hash_ms > 0.0 ? gib / (row.expected_hash_ms / 1000.0) : 0.0;

            row.upload_write_to_consume = measure_upload_write_to_consume(
                runtime,
                resources,
                bytes,
                row.warmup,
                row.iterations,
                expected_value,
                allocator.Get(),
                list.Get());
            row.h2d_only = measure_h2d_only(runtime, resources, bytes, row.warmup, row.iterations, allocator.Get(), list.Get());
            row.end_to_end = measure_end_to_end(
                runtime,
                resources,
                bytes,
                row.warmup,
                row.iterations,
                expected_value,
                allocator.Get(),
                list.Get());
            row.end_to_end.expected_value = expected_value;
            row.validation_path_overhead_ms = std::max(0.0, row.end_to_end.avg_ms - row.h2d_only.avg_ms);
            row.validation_path_overhead_ratio = row.h2d_only.avg_ms > 0.0
                ? row.end_to_end.avg_ms / row.h2d_only.avg_ms
                : 0.0;
            validation_passed =
                validation_passed &&
                row.upload_write_to_consume.validation_passed &&
                row.end_to_end.validation_passed;
            rows.push_back(row);

            destroy_resources(resources);
        }

        emit_json(render_json(options, runtime, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        try {
            emit_json(make_error_json(ex.what(), parse_common_args(argc, argv)));
        } catch (...) {
            emit_json(make_error_json(ex.what(), Options{}));
        }
        return 0;
    }
}
