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
constexpr double kCudaRefH2DGiBps = 53.310166;
constexpr double kCudaRefD2HGiBps = 53.012081;
constexpr double kTheoryGiBps = 58.687292;
constexpr double kTolerance = 0.30;

struct Options {
    int iterations = 20;
    int warmup = 3;
    std::vector<size_t> sizes_mb = {128, 512, 1024};
};

struct MethodStats {
    bool success = false;
    std::string error;
    double avg_ms = 0.0;
    double gib_per_s = 0.0;
    bool within_cuda_reference = false;
    bool within_theoretical_limit = false;
    bool accepted = false;
};

struct DirectionRow {
    MethodStats cpu_wall_single;
    MethodStats cpu_wall_batch;
    MethodStats gpu_timestamp_batch;
};

struct CaseRow {
    size_t size_mb = 0;
    int iterations = 0;
    int warmup = 0;
    int batch_repeats = 0;
    DirectionRow h2d_like;
    DirectionRow d2h_like;
};

struct Runtime {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> copy_queue;
    ComPtr<ID3D12Fence> copy_fence;
    UINT64 copy_fence_value = 0;
    HANDLE copy_event = nullptr;
    UINT64 timestamp_frequency = 0;
    std::string adapter_name;
};

struct Resources {
    ComPtr<ID3D12Resource> upload_buffer;
    ComPtr<ID3D12Resource> default_buffer;
    ComPtr<ID3D12Resource> readback_buffer;
    ComPtr<ID3D12QueryHeap> query_heap;
    ComPtr<ID3D12Resource> query_readback;
    uint8_t* upload_ptr = nullptr;
    uint8_t* readback_ptr = nullptr;
    uint64_t* query_ptr = nullptr;
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

Runtime create_runtime() {
    Runtime runtime;
    check_hr(CreateDXGIFactory2(0, IID_PPV_ARGS(&runtime.factory)), "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        if (runtime.factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) {
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
    check_hr(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&runtime.copy_fence)), "CreateFence(copy)");
    runtime.copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (runtime.copy_event == nullptr) {
        throw std::runtime_error("CreateEventW failed");
    }
    check_hr(runtime.copy_queue->GetTimestampFrequency(&runtime.timestamp_frequency), "GetTimestampFrequency");
    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.copy_event != nullptr) {
        CloseHandle(runtime.copy_event);
        runtime.copy_event = nullptr;
    }
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

Resources create_resources(Runtime& runtime, size_t copy_bytes) {
    Resources resources;
    resources.upload_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_UPLOAD, static_cast<UINT64>(copy_bytes), D3D12_RESOURCE_STATE_GENERIC_READ);
    resources.default_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_DEFAULT, static_cast<UINT64>(copy_bytes), D3D12_RESOURCE_STATE_COMMON);
    resources.readback_buffer = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_READBACK, static_cast<UINT64>(copy_bytes), D3D12_RESOURCE_STATE_COPY_DEST);
    resources.query_readback = create_buffer(runtime.device.Get(), D3D12_HEAP_TYPE_READBACK, sizeof(uint64_t) * 2, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_QUERY_HEAP_DESC query_desc{};
    query_desc.Count = 2;
    query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    check_hr(runtime.device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&resources.query_heap)), "CreateQueryHeap");

    check_hr(resources.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.upload_ptr)), "Map(upload)");
    check_hr(resources.readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&resources.readback_ptr)), "Map(readback)");
    check_hr(resources.query_readback->Map(0, nullptr, reinterpret_cast<void**>(&resources.query_ptr)), "Map(query_readback)");
    return resources;
}

void destroy_resources(Resources& resources) {
    if (resources.query_readback) {
        resources.query_readback->Unmap(0, nullptr);
    }
    if (resources.readback_buffer) {
        resources.readback_buffer->Unmap(0, nullptr);
    }
    if (resources.upload_buffer) {
        resources.upload_buffer->Unmap(0, nullptr);
    }
}

void wait_for_fence(Runtime& runtime, UINT64 value) {
    if (runtime.copy_fence->GetCompletedValue() >= value) {
        return;
    }
    check_hr(runtime.copy_fence->SetEventOnCompletion(value, runtime.copy_event), "SetEventOnCompletion");
    if (WaitForSingleObject(runtime.copy_event, INFINITE) != WAIT_OBJECT_0) {
        throw std::runtime_error("WaitForSingleObject failed");
    }
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

void fill_host(uint8_t* ptr, size_t bytes, uint8_t value) {
    std::memset(ptr, value, bytes);
}

int batch_repeats_for_size(size_t size_mb) {
    if (size_mb <= 128) {
        return 16;
    }
    if (size_mb <= 512) {
        return 4;
    }
    return 2;
}

void record_copy_batch(
    ID3D12GraphicsCommandList* list,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    int repeats) {
    if (is_h2d_like) {
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        for (int i = 0; i < repeats; ++i) {
            list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
        }
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    } else {
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (int i = 0; i < repeats; ++i) {
            list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
        }
        transition_buffer(list, resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
}

UINT64 execute_copy(Runtime& runtime, ID3D12GraphicsCommandList* list) {
    check_hr(list->Close(), "CommandList::Close");
    ID3D12CommandList* lists[] = {list};
    runtime.copy_queue->ExecuteCommandLists(1, lists);
    runtime.copy_fence_value += 1;
    check_hr(runtime.copy_queue->Signal(runtime.copy_fence.Get(), runtime.copy_fence_value), "Signal(copy)");
    wait_for_fence(runtime, runtime.copy_fence_value);
    return runtime.copy_fence_value;
}

MethodStats finalize_stats(double total_ms, size_t bytes, int repeats, int iterations, double cuda_ref) {
    MethodStats stats;
    stats.success = true;
    stats.avg_ms = total_ms / static_cast<double>(iterations);
    const double gib_total = static_cast<double>(bytes) * static_cast<double>(repeats) / (1024.0 * 1024.0 * 1024.0);
    stats.gib_per_s = gib_total / (stats.avg_ms / 1000.0);
    stats.within_cuda_reference = std::fabs((stats.gib_per_s - cuda_ref) / cuda_ref) <= kTolerance;
    stats.within_theoretical_limit = stats.gib_per_s <= kTheoryGiBps;
    stats.accepted = stats.within_cuda_reference && stats.within_theoretical_limit;
    return stats;
}

MethodStats measure_cpu_wall(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    int repeats,
    int warmup,
    int iterations,
    double cuda_ref) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    check_hr(list->Close(), "Close(initial)");

    for (int i = 0; i < warmup; ++i) {
        check_hr(allocator->Reset(), "Allocator::Reset");
        check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset");
        record_copy_batch(list.Get(), resources, bytes, is_h2d_like, repeats);
        execute_copy(runtime, list.Get());
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        check_hr(allocator->Reset(), "Allocator::Reset");
        check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset");
        record_copy_batch(list.Get(), resources, bytes, is_h2d_like, repeats);
        const auto start = std::chrono::steady_clock::now();
        execute_copy(runtime, list.Get());
        const auto end = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return finalize_stats(total_ms, bytes, repeats, iterations, cuda_ref);
}

MethodStats measure_gpu_timestamp_batch(
    Runtime& runtime,
    Resources& resources,
    size_t bytes,
    bool is_h2d_like,
    int repeats,
    int warmup,
    int iterations,
    double cuda_ref) {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    check_hr(list->Close(), "Close(initial)");

    auto record = [&]() {
        list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        record_copy_batch(list.Get(), resources, bytes, is_h2d_like, repeats);
        list->EndQuery(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list->ResolveQueryData(resources.query_heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, resources.query_readback.Get(), 0);
    };

    for (int i = 0; i < warmup; ++i) {
        check_hr(allocator->Reset(), "Allocator::Reset");
        check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset");
        record();
        execute_copy(runtime, list.Get());
    }

    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        resources.query_ptr[0] = 0;
        resources.query_ptr[1] = 0;
        check_hr(allocator->Reset(), "Allocator::Reset");
        check_hr(list->Reset(allocator.Get(), nullptr), "List::Reset");
        record();
        execute_copy(runtime, list.Get());
        const uint64_t begin = resources.query_ptr[0];
        const uint64_t end = resources.query_ptr[1];
        if (end <= begin || runtime.timestamp_frequency == 0) {
            MethodStats failed;
            failed.error = "Invalid timestamp query result";
            return failed;
        }
        total_ms += static_cast<double>(end - begin) * 1000.0 / static_cast<double>(runtime.timestamp_frequency);
    }
    return finalize_stats(total_ms, bytes, repeats, iterations, cuda_ref);
}

void initialize_default(Runtime& runtime, Resources& resources, size_t bytes, uint8_t fill_value) {
    fill_host(resources.upload_ptr, bytes, fill_value);
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    execute_copy(runtime, list.Get());
}

bool validate_h2d(Runtime& runtime, Resources& resources, size_t bytes) {
    fill_host(resources.upload_ptr, bytes, 0x3C);
    std::memset(resources.readback_ptr, 0x00, bytes);
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(resources.default_buffer.Get(), 0, resources.upload_buffer.Get(), 0, bytes);
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    execute_copy(runtime, list.Get());
    for (size_t i = 0; i < bytes; ++i) {
        if (resources.readback_ptr[i] != 0x3C) {
            return false;
        }
    }
    return true;
}

bool validate_d2h(Runtime& runtime, Resources& resources, size_t bytes) {
    initialize_default(runtime, resources, bytes, 0x5A);
    std::memset(resources.readback_ptr, 0x00, bytes);
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check_hr(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, IID_PPV_ARGS(&allocator)), "CreateCommandAllocator");
    check_hr(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "CreateCommandList");
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(resources.readback_buffer.Get(), 0, resources.default_buffer.Get(), 0, bytes);
    transition_buffer(list.Get(), resources.default_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    execute_copy(runtime, list.Get());
    for (size_t i = 0; i < bytes; ++i) {
        if (resources.readback_ptr[i] != 0x5A) {
            return false;
        }
    }
    return true;
}

std::string render_method_json(const MethodStats& stats) {
    std::ostringstream oss;
    oss << "{"
        << "\"success\":" << (stats.success ? "true" : "false") << ","
        << "\"avg_ms\":" << format_double(stats.avg_ms) << ","
        << "\"gib_per_s\":" << format_double(stats.gib_per_s) << ","
        << "\"within_cuda_reference\":" << (stats.within_cuda_reference ? "true" : "false") << ","
        << "\"within_theoretical_limit\":" << (stats.within_theoretical_limit ? "true" : "false") << ","
        << "\"accepted\":" << (stats.accepted ? "true" : "false");
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_direction_json(const DirectionRow& row) {
    std::ostringstream oss;
    oss << "{"
        << "\"cpu_wall_single\":" << render_method_json(row.cpu_wall_single) << ","
        << "\"cpu_wall_batch\":" << render_method_json(row.cpu_wall_batch) << ","
        << "\"gpu_timestamp_batch\":" << render_method_json(row.gpu_timestamp_batch)
        << "}";
    return oss.str();
}

const MethodStats* choose_accepted_method(const DirectionRow& row) {
    const MethodStats* best = nullptr;
    for (const MethodStats* stats : {&row.cpu_wall_single, &row.cpu_wall_batch, &row.gpu_timestamp_batch}) {
        if (!stats->accepted) {
            continue;
        }
        if (best == nullptr || stats->gib_per_s > best->gib_per_s) {
            best = stats;
        }
    }
    return best;
}

std::string choose_method_name(const DirectionRow& row) {
    const MethodStats* best = choose_accepted_method(row);
    if (best == nullptr) {
        return "";
    }
    if (best == &row.cpu_wall_single) {
        return "cpu_wall_single";
    }
    if (best == &row.cpu_wall_batch) {
        return "cpu_wall_batch";
    }
    return "gpu_timestamp_batch";
}

std::string render_json(const Options& options, const Runtime& runtime, const std::vector<CaseRow>& rows, bool validation_passed) {
    std::string accepted_h2d_method;
    std::string accepted_d2h_method;
    double accepted_h2d_gib = 0.0;
    double accepted_d2h_gib = 0.0;

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
              << "\"batch_repeats\":" << row.batch_repeats << ","
              << "\"h2d_like\":" << render_direction_json(row.h2d_like) << ","
              << "\"d2h_like\":" << render_direction_json(row.d2h_like)
              << "}";

        const MethodStats* h2d = choose_accepted_method(row.h2d_like);
        if (h2d != nullptr && h2d->gib_per_s > accepted_h2d_gib) {
            accepted_h2d_gib = h2d->gib_per_s;
            accepted_h2d_method = choose_method_name(row.h2d_like);
        }
        const MethodStats* d2h = choose_accepted_method(row.d2h_like);
        if (d2h != nullptr && d2h->gib_per_s > accepted_d2h_gib) {
            accepted_d2h_gib = d2h->gib_per_s;
            accepted_d2h_method = choose_method_name(row.d2h_like);
        }
    }
    cases << "]";

    int validated_direction_count = 0;
    if (!accepted_h2d_method.empty()) {
        validated_direction_count += 1;
    }
    if (!accepted_d2h_method.empty()) {
        validated_direction_count += 1;
    }

    std::ostringstream oss;
    oss << "{"
        << "\"status\":" << quote((validation_passed && validated_direction_count == 2) ? "ok" : "invalid") << ","
        << "\"primary_metric\":\"validated_direction_count\","
        << "\"unit\":\"count\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"copy_directions\":[\"H2D-like\",\"D2H-like\"],"
        << "\"methods\":[\"cpu_wall_single\",\"cpu_wall_batch\",\"gpu_timestamp_batch\"],"
        << "\"cuda_reference_h2d_gib_per_s\":" << format_double(kCudaRefH2DGiBps) << ","
        << "\"cuda_reference_d2h_gib_per_s\":" << format_double(kCudaRefD2HGiBps) << ","
        << "\"theoretical_one_way_pcie_5_x16_gib_per_s\":" << format_double(kTheoryGiBps) << ","
        << "\"reference_tolerance_ratio\":" << format_double(kTolerance) << ","
        << "\"iterations\":" << options.iterations << ","
        << "\"warmup\":" << options.warmup << ","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"mixed_cpu_wall_and_gpu_timestamp\","
        << "\"adapter_name\":" << quote(runtime.adapter_name) << ","
        << "\"accepted_h2d_method\":" << quote(accepted_h2d_method) << ","
        << "\"accepted_d2h_method\":" << quote(accepted_d2h_method) << ","
        << "\"validated_direction_count\":" << validated_direction_count << ","
        << "\"cases\":" << cases.str()
        << "},"
        << "\"validation\":{"
        << "\"passed\":" << (validation_passed ? "true" : "false")
        << "}"
        << "}";
    return oss.str();
}

std::string make_error_json(const std::string& message, const Options& options) {
    std::ostringstream oss;
    oss << "{"
        << "\"status\":\"failed\","
        << "\"primary_metric\":\"validated_direction_count\","
        << "\"unit\":\"count\","
        << "\"parameters\":{"
        << "\"api\":\"d3d12\","
        << "\"sizes_mb\":" << sizes_to_json(options.sizes_mb)
        << "},"
        << "\"measurement\":{"
        << "\"timing_backend\":\"mixed_cpu_wall_and_gpu_timestamp\""
        << "},"
        << "\"validation\":{"
        << "\"passed\":false"
        << "},"
        << "\"notes\":[" << quote(message) << "]"
        << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options{};
    Runtime runtime{};
    try {
        options = parse_common_args(argc, argv);
        runtime = create_runtime();

        std::vector<CaseRow> rows;
        bool validation_passed = true;
        auto capture_method = [](auto&& fn) {
            MethodStats stats;
            try {
                stats = fn();
            } catch (const std::exception& ex) {
                stats.success = false;
                stats.error = ex.what();
            }
            return stats;
        };

        for (size_t size_mb : options.sizes_mb) {
            const size_t copy_bytes = size_mb * 1024ull * 1024ull;
            Resources resources = create_resources(runtime, copy_bytes);
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = options.iterations;
            row.warmup = options.warmup;
            row.batch_repeats = batch_repeats_for_size(size_mb);

            fill_host(resources.upload_ptr, copy_bytes, 0x3C);
            row.h2d_like.cpu_wall_single = capture_method([&] {
                return measure_cpu_wall(runtime, resources, copy_bytes, true, 1, options.warmup, options.iterations, kCudaRefH2DGiBps);
            });
            row.h2d_like.cpu_wall_batch = capture_method([&] {
                return measure_cpu_wall(runtime, resources, copy_bytes, true, row.batch_repeats, options.warmup, options.iterations, kCudaRefH2DGiBps);
            });
            row.h2d_like.gpu_timestamp_batch = capture_method([&] {
                return measure_gpu_timestamp_batch(runtime, resources, copy_bytes, true, row.batch_repeats, options.warmup, options.iterations, kCudaRefH2DGiBps);
            });

            initialize_default(runtime, resources, copy_bytes, 0x5A);
            row.d2h_like.cpu_wall_single = capture_method([&] {
                return measure_cpu_wall(runtime, resources, copy_bytes, false, 1, options.warmup, options.iterations, kCudaRefD2HGiBps);
            });
            row.d2h_like.cpu_wall_batch = capture_method([&] {
                return measure_cpu_wall(runtime, resources, copy_bytes, false, row.batch_repeats, options.warmup, options.iterations, kCudaRefD2HGiBps);
            });
            row.d2h_like.gpu_timestamp_batch = capture_method([&] {
                return measure_gpu_timestamp_batch(runtime, resources, copy_bytes, false, row.batch_repeats, options.warmup, options.iterations, kCudaRefD2HGiBps);
            });

            if (!validate_h2d(runtime, resources, copy_bytes)) {
                validation_passed = false;
            }
            if (!validate_d2h(runtime, resources, copy_bytes)) {
                validation_passed = false;
            }

            destroy_resources(resources);
            rows.push_back(row);
        }

        emit_json(render_json(options, runtime, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json(ex.what(), options));
        destroy_runtime(runtime);
        return 1;
    }
}
