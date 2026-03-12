#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<size_t, 3> kDefaultSizesMb = {128, 512, 1024};
constexpr uint32_t kThreadsPerGroup = 256;
constexpr uint32_t kDispatchGroups = 1024;
constexpr uint32_t kTotalThreads = kThreadsPerGroup * kDispatchGroups;
constexpr int kCalibrationPassLimit = 8;

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

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

struct Resources {
    Buffer staging_upload;
    Buffer device_buffer;
    Buffer readback_buffer;
    Buffer kernel_output;
    Buffer kernel_output_readback;
    Buffer uniform_buffer;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct UniformParams {
    uint32_t loops = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};

struct Runtime {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t transfer_family = 0;
    uint32_t compute_family = 0;
    uint32_t transfer_queue_index = 0;
    uint32_t compute_queue_index = 0;
    VkCommandPool transfer_pool = VK_NULL_HANDLE;
    VkCommandPool compute_pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    std::string adapter_name;
};

void check_vk(VkResult result, const char* context) {
    if (result != VK_SUCCESS) {
        std::ostringstream oss;
        oss << context << ": VkResult " << static_cast<int>(result);
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
            if (arg == flag) {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + flag);
                }
                return argv[++i];
            }
            if (starts_with(arg, flag + "=")) {
                return arg.substr(flag.size() + 1);
            }
            return {};
        };
        if (auto value = get_value("--iterations"); !value.empty()) {
            options.iterations = std::max(1, std::atoi(value.c_str()));
            continue;
        }
        if (auto value = get_value("--warmup"); !value.empty()) {
            options.warmup = std::max(0, std::atoi(value.c_str()));
            continue;
        }
        if (auto value = get_value("--sizes_mb"); !value.empty()) {
            options.sizes_mb = parse_sizes_mb(value);
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench.exe [--iterations N] [--warmup N] [--sizes_mb a,b,c]\n";
            std::exit(0);
        }
        throw std::runtime_error("Unknown argument: " + arg);
    }
    return options;
}

int effective_iterations(size_t size_mb, int requested_iterations) {
    if (size_mb >= 1024) {
        return std::max(3, requested_iterations / 10);
    }
    if (size_mb >= 512) {
        return std::max(5, requested_iterations / 5);
    }
    return std::max(5, requested_iterations / 2);
}

int effective_warmup(size_t size_mb, int requested_warmup) {
    if (size_mb >= 1024) {
        return std::min(2, requested_warmup);
    }
    return std::min(3, requested_warmup);
}

void emit_json(const std::string& json) {
    std::cout << json << std::endl;
}

std::string make_error_json(const std::string& status, const std::string& message, const Options& options, const std::string& primary_metric) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"status\":" << quote(status) << ",";
    oss << "\"primary_metric\":" << quote(primary_metric) << ",";
    oss << "\"unit\":\"ratio\",";
    oss << "\"parameters\":{\"api\":\"vulkan\",\"sizes_mb\":[";
    for (size_t i = 0; i < options.sizes_mb.size(); ++i) {
        if (i > 0) oss << ",";
        oss << options.sizes_mb[i];
    }
    oss << "],\"iterations\":" << options.iterations << ",\"warmup\":" << options.warmup << "},";
    oss << "\"measurement\":{\"timing_backend\":\"wall_clock\",\"error\":" << quote(message) << "},";
    oss << "\"validation\":{\"passed\":false},";
    oss << "\"notes\":[" << quote(message) << "]}";
    return oss.str();
}

void fill_pattern(uint8_t* ptr, size_t bytes, uint8_t seed) {
    uint32_t x = 0x12345678u ^ seed;
    for (size_t i = 0; i < bytes; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        ptr[i] = static_cast<uint8_t>(x & 0xFFu);
    }
}

uint32_t compute_kernel_word(uint32_t tid, uint32_t loops) {
    uint32_t x = 0x12345678u ^ (tid * 747796405u + 2891336453u);
    uint32_t y = 0x9E3779B9u + (tid * 1664525u + 1013904223u);
    uint32_t z = 0xA5A5A5A5u ^ (tid * 2246822519u + 3266489917u);
    for (uint32_t i = 0; i < loops; ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        y = y * 1664525u + 1013904223u;
        z ^= x + 0x9E3779B9u + (z << 6) + (z >> 2);
    }
    return x ^ y ^ z;
}

std::vector<uint32_t> read_spirv_words(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V file size: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("Failed to read SPIR-V file: " + path);
    }
    return words;
}

uint32_t find_memory_type(const Runtime& runtime, uint32_t type_bits, VkMemoryPropertyFlags required) {
    for (uint32_t i = 0; i < runtime.memory_properties.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 && (runtime.memory_properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("No matching Vulkan memory type");
}

Runtime create_runtime() {
    Runtime runtime;
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "vulkan_transfer_compute_overlap";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "none";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;
    check_vk(vkCreateInstance(&instance_info, nullptr, &runtime.instance), "vkCreateInstance");

    uint32_t device_count = 0;
    check_vk(vkEnumeratePhysicalDevices(runtime.instance, &device_count, nullptr), "vkEnumeratePhysicalDevices count");
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check_vk(vkEnumeratePhysicalDevices(runtime.instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");

    VkPhysicalDeviceProperties selected_props = {};
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            runtime.physical_device = candidate;
            selected_props = props;
            break;
        }
    }
    if (runtime.physical_device == VK_NULL_HANDLE) {
        runtime.physical_device = devices[0];
        vkGetPhysicalDeviceProperties(runtime.physical_device, &selected_props);
    }
    runtime.adapter_name = selected_props.deviceName;
    vkGetPhysicalDeviceMemoryProperties(runtime.physical_device, &runtime.memory_properties);

    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device, &family_count, families.data());

    auto find_family = [&](VkQueueFlags required, VkQueueFlags excluded) -> uint32_t {
        for (uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & required) == required && (families[i].queueFlags & excluded) == 0) {
                return i;
            }
        }
        for (uint32_t i = 0; i < family_count; ++i) {
            if ((families[i].queueFlags & required) == required) {
                return i;
            }
        }
        return UINT32_MAX;
    };

    runtime.compute_family = find_family(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT);
    if (runtime.compute_family == UINT32_MAX) {
        throw std::runtime_error("No compute-capable Vulkan queue family");
    }
    runtime.transfer_family = find_family(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
    if (runtime.transfer_family == UINT32_MAX) {
        runtime.transfer_family = find_family(VK_QUEUE_TRANSFER_BIT, 0);
    }
    if (runtime.transfer_family == UINT32_MAX) {
        throw std::runtime_error("No transfer-capable Vulkan queue family");
    }

    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    const float priorities[2] = {1.0f, 1.0f};
    if (runtime.transfer_family == runtime.compute_family) {
        if (families[runtime.transfer_family].queueCount < 2) {
            throw std::runtime_error("Vulkan device does not expose two queues for transfer+compute overlap");
        }
        VkDeviceQueueCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = runtime.transfer_family;
        info.queueCount = 2;
        info.pQueuePriorities = priorities;
        queue_infos.push_back(info);
        runtime.transfer_queue_index = 0;
        runtime.compute_queue_index = 1;
    } else {
        VkDeviceQueueCreateInfo transfer_info = {};
        transfer_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transfer_info.queueFamilyIndex = runtime.transfer_family;
        transfer_info.queueCount = 1;
        transfer_info.pQueuePriorities = priorities;
        queue_infos.push_back(transfer_info);

        VkDeviceQueueCreateInfo compute_info = {};
        compute_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        compute_info.queueFamilyIndex = runtime.compute_family;
        compute_info.queueCount = 1;
        compute_info.pQueuePriorities = priorities;
        queue_infos.push_back(compute_info);
    }

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    device_info.pQueueCreateInfos = queue_infos.data();
    check_vk(vkCreateDevice(runtime.physical_device, &device_info, nullptr, &runtime.device), "vkCreateDevice");

    vkGetDeviceQueue(runtime.device, runtime.transfer_family, runtime.transfer_queue_index, &runtime.transfer_queue);
    vkGetDeviceQueue(runtime.device, runtime.compute_family, runtime.compute_queue_index, &runtime.compute_queue);

    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = runtime.transfer_family;
    check_vk(vkCreateCommandPool(runtime.device, &pool_info, nullptr, &runtime.transfer_pool), "vkCreateCommandPool transfer");
    pool_info.queueFamilyIndex = runtime.compute_family;
    check_vk(vkCreateCommandPool(runtime.device, &pool_info, nullptr, &runtime.compute_pool), "vkCreateCommandPool compute");

    std::vector<uint32_t> spirv = read_spirv_words("compute.spv");
    VkShaderModuleCreateInfo shader_info = {};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(runtime.device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

    const std::array<VkDescriptorSetLayoutBinding, 2> bindings = {{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }};
    VkDescriptorSetLayoutCreateInfo set_layout_info = {};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    set_layout_info.pBindings = bindings.data();
    check_vk(vkCreateDescriptorSetLayout(runtime.device, &set_layout_info, nullptr, &runtime.descriptor_set_layout), "vkCreateDescriptorSetLayout");

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &runtime.descriptor_set_layout;
    check_vk(vkCreatePipelineLayout(runtime.device, &layout_info, nullptr, &runtime.pipeline_layout), "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = runtime.pipeline_layout;
    check_vk(vkCreateComputePipelines(runtime.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &runtime.pipeline), "vkCreateComputePipelines");
    vkDestroyShaderModule(runtime.device, shader_module, nullptr);
    return runtime;
}

void destroy_runtime(Runtime& runtime) {
    if (runtime.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(runtime.device);
        if (runtime.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(runtime.device, runtime.pipeline, nullptr);
        if (runtime.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(runtime.device, runtime.pipeline_layout, nullptr);
        if (runtime.descriptor_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(runtime.device, runtime.descriptor_set_layout, nullptr);
        if (runtime.transfer_pool != VK_NULL_HANDLE) vkDestroyCommandPool(runtime.device, runtime.transfer_pool, nullptr);
        if (runtime.compute_pool != VK_NULL_HANDLE) vkDestroyCommandPool(runtime.device, runtime.compute_pool, nullptr);
        vkDestroyDevice(runtime.device, nullptr);
    }
    if (runtime.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(runtime.instance, nullptr);
    }
}

Buffer create_buffer(Runtime& runtime, VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    Buffer buffer;
    buffer.size = bytes;

    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(runtime.device, &info, nullptr, &buffer.buffer), "vkCreateBuffer");

    VkMemoryRequirements req = {};
    vkGetBufferMemoryRequirements(runtime.device, buffer.buffer, &req);

    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = find_memory_type(runtime, req.memoryTypeBits, properties);
    check_vk(vkAllocateMemory(runtime.device, &alloc, nullptr, &buffer.memory), "vkAllocateMemory");
    check_vk(vkBindBufferMemory(runtime.device, buffer.buffer, buffer.memory, 0), "vkBindBufferMemory");

    if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check_vk(vkMapMemory(runtime.device, buffer.memory, 0, bytes, 0, &buffer.mapped), "vkMapMemory");
    }
    return buffer;
}

void destroy_buffer(Runtime& runtime, Buffer& buffer) {
    if (buffer.mapped) {
        vkUnmapMemory(runtime.device, buffer.memory);
        buffer.mapped = nullptr;
    }
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(runtime.device, buffer.buffer, nullptr);
        buffer.buffer = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(runtime.device, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
}

Resources create_resources(Runtime& runtime, size_t copy_bytes) {
    Resources resources;
    const VkDeviceSize copy_size = static_cast<VkDeviceSize>(copy_bytes);
    resources.staging_upload = create_buffer(runtime, copy_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    resources.device_buffer = create_buffer(runtime, copy_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    resources.readback_buffer = create_buffer(runtime, copy_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    resources.kernel_output = create_buffer(runtime, kTotalThreads * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    resources.kernel_output_readback = create_buffer(runtime, kTotalThreads * sizeof(uint32_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    resources.uniform_buffer = create_buffer(runtime, sizeof(UniformParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDescriptorPoolSize pool_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    check_vk(vkCreateDescriptorPool(runtime.device, &pool_info, nullptr, &resources.descriptor_pool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = resources.descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &runtime.descriptor_set_layout;
    check_vk(vkAllocateDescriptorSets(runtime.device, &set_info, &resources.descriptor_set), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo storage_info = {resources.kernel_output.buffer, 0, resources.kernel_output.size};
    VkDescriptorBufferInfo uniform_info = {resources.uniform_buffer.buffer, 0, resources.uniform_buffer.size};
    std::array<VkWriteDescriptorSet, 2> writes = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = resources.descriptor_set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &storage_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = resources.descriptor_set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &uniform_info;
    vkUpdateDescriptorSets(runtime.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return resources;
}

void destroy_resources(Runtime& runtime, Resources& resources) {
    if (resources.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        resources.descriptor_pool = VK_NULL_HANDLE;
    }
    destroy_buffer(runtime, resources.uniform_buffer);
    destroy_buffer(runtime, resources.kernel_output_readback);
    destroy_buffer(runtime, resources.kernel_output);
    destroy_buffer(runtime, resources.readback_buffer);
    destroy_buffer(runtime, resources.device_buffer);
    destroy_buffer(runtime, resources.staging_upload);
}

VkCommandBuffer allocate_command_buffer(Runtime& runtime, VkCommandPool pool) {
    VkCommandBufferAllocateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(runtime.device, &info, &cmd), "vkAllocateCommandBuffers");
    return cmd;
}

VkFence create_fence(Runtime& runtime) {
    VkFenceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    check_vk(vkCreateFence(runtime.device, &info, nullptr, &fence), "vkCreateFence");
    return fence;
}

void begin_command_buffer(VkCommandBuffer cmd) {
    check_vk(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(cmd, &info), "vkBeginCommandBuffer");
}

void end_command_buffer(VkCommandBuffer cmd) {
    check_vk(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
}

void submit_and_wait(Runtime& runtime, VkQueue queue, VkCommandBuffer cmd, VkFence fence) {
    check_vk(vkResetFences(runtime.device, 1, &fence), "vkResetFences");
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    check_vk(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
    check_vk(vkWaitForFences(runtime.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

void update_uniform(Resources& resources, uint32_t loops) {
    UniformParams params = {};
    params.loops = loops;
    std::memcpy(resources.uniform_buffer.mapped, &params, sizeof(params));
}

void record_kernel_dispatch(Runtime& runtime, Resources& resources, VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, resources.kernel_output.buffer, 0, resources.kernel_output.size, 0);

    VkBufferMemoryBarrier pre_barrier = {};
    pre_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pre_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre_barrier.buffer = resources.kernel_output.buffer;
    pre_barrier.offset = 0;
    pre_barrier.size = resources.kernel_output.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &pre_barrier, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, runtime.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, runtime.pipeline_layout, 0, 1, &resources.descriptor_set, 0, nullptr);
    vkCmdDispatch(cmd, kDispatchGroups, 1, 1);
}
WallStats measure_copy_wall(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, int warmup, int iterations, VkCommandBuffer cmd, VkFence fence) {
    WallStats stats;
    try {
        for (int i = 0; i < warmup; ++i) {
            begin_command_buffer(cmd);
            VkBuffer src = is_h2d_like ? resources.staging_upload.buffer : resources.device_buffer.buffer;
            VkBuffer dst = is_h2d_like ? resources.device_buffer.buffer : resources.readback_buffer.buffer;
            VkBufferCopy region = {};
            region.size = static_cast<VkDeviceSize>(bytes);
            vkCmdCopyBuffer(cmd, src, dst, 1, &region);
            end_command_buffer(cmd);
            submit_and_wait(runtime, runtime.transfer_queue, cmd, fence);
        }
        double total_ms = 0.0;
        for (int i = 0; i < iterations; ++i) {
            begin_command_buffer(cmd);
            VkBuffer src = is_h2d_like ? resources.staging_upload.buffer : resources.device_buffer.buffer;
            VkBuffer dst = is_h2d_like ? resources.device_buffer.buffer : resources.readback_buffer.buffer;
            VkBufferCopy region = {};
            region.size = static_cast<VkDeviceSize>(bytes);
            vkCmdCopyBuffer(cmd, src, dst, 1, &region);
            end_command_buffer(cmd);
            const auto start = std::chrono::steady_clock::now();
            submit_and_wait(runtime, runtime.transfer_queue, cmd, fence);
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        stats.success = true;
        stats.avg_ms = total_ms / static_cast<double>(iterations);
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        stats.gib_per_s = gib / (stats.avg_ms / 1000.0);
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }
    return stats;
}

KernelStats measure_kernel_wall(Runtime& runtime, Resources& resources, uint32_t loops, int warmup, int iterations, VkCommandBuffer cmd, VkFence fence) {
    KernelStats stats;
    try {
        update_uniform(resources, loops);
        for (int i = 0; i < warmup; ++i) {
            begin_command_buffer(cmd);
            record_kernel_dispatch(runtime, resources, cmd);
            end_command_buffer(cmd);
            submit_and_wait(runtime, runtime.compute_queue, cmd, fence);
        }
        double total_ms = 0.0;
        for (int i = 0; i < iterations; ++i) {
            begin_command_buffer(cmd);
            record_kernel_dispatch(runtime, resources, cmd);
            end_command_buffer(cmd);
            const auto start = std::chrono::steady_clock::now();
            submit_and_wait(runtime, runtime.compute_queue, cmd, fence);
            const auto end = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        stats.success = true;
        stats.avg_ms = total_ms / static_cast<double>(iterations);
        stats.loop_count = loops;
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }
    return stats;
}

KernelStats calibrate_kernel_to_target(Runtime& runtime, Resources& resources, double target_ms, int warmup, int iterations, VkCommandBuffer cmd, VkFence fence) {
    uint32_t loops = 1u << 14;
    KernelStats stats;
    for (int pass = 0; pass < kCalibrationPassLimit; ++pass) {
        stats = measure_kernel_wall(runtime, resources, loops, warmup, iterations, cmd, fence);
        if (!stats.success) {
            return stats;
        }
        if (stats.avg_ms <= 0.0) {
            loops *= 4;
            continue;
        }
        if (target_ms <= 0.0) {
            return stats;
        }
        const double ratio = target_ms / stats.avg_ms;
        if (ratio >= 0.8 && ratio <= 1.25) {
            return stats;
        }
        const double next = std::clamp(static_cast<double>(loops) * ratio, 256.0, static_cast<double>(1u << 31));
        loops = std::max<uint32_t>(1u, static_cast<uint32_t>(next));
    }
    return stats;
}

OverlapStats measure_overlap_wall(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, uint32_t loops, int warmup, int iterations, const WallStats& copy_solo, const KernelStats& kernel_solo, VkCommandBuffer transfer_cmd, VkFence transfer_fence, VkCommandBuffer compute_cmd, VkFence compute_fence) {
    OverlapStats stats;
    try {
        update_uniform(resources, loops);
        auto submit_pair = [&](bool timed, double* total_ms) {
            begin_command_buffer(transfer_cmd);
            VkBuffer src = is_h2d_like ? resources.staging_upload.buffer : resources.device_buffer.buffer;
            VkBuffer dst = is_h2d_like ? resources.device_buffer.buffer : resources.readback_buffer.buffer;
            VkBufferCopy region = {};
            region.size = static_cast<VkDeviceSize>(bytes);
            vkCmdCopyBuffer(transfer_cmd, src, dst, 1, &region);
            end_command_buffer(transfer_cmd);

            begin_command_buffer(compute_cmd);
            record_kernel_dispatch(runtime, resources, compute_cmd);
            end_command_buffer(compute_cmd);

            check_vk(vkResetFences(runtime.device, 1, &transfer_fence), "vkResetFences transfer");
            check_vk(vkResetFences(runtime.device, 1, &compute_fence), "vkResetFences compute");
            VkSubmitInfo transfer_submit = {};
            transfer_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            transfer_submit.commandBufferCount = 1;
            transfer_submit.pCommandBuffers = &transfer_cmd;
            VkSubmitInfo compute_submit = {};
            compute_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            compute_submit.commandBufferCount = 1;
            compute_submit.pCommandBuffers = &compute_cmd;

            const auto start = std::chrono::steady_clock::now();
            check_vk(vkQueueSubmit(runtime.transfer_queue, 1, &transfer_submit, transfer_fence), "vkQueueSubmit transfer");
            check_vk(vkQueueSubmit(runtime.compute_queue, 1, &compute_submit, compute_fence), "vkQueueSubmit compute");
            check_vk(vkWaitForFences(runtime.device, 1, &transfer_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences transfer");
            check_vk(vkWaitForFences(runtime.device, 1, &compute_fence, VK_TRUE, UINT64_MAX), "vkWaitForFences compute");
            if (timed) {
                const auto end = std::chrono::steady_clock::now();
                *total_ms += std::chrono::duration<double, std::milli>(end - start).count();
            }
        };

        for (int i = 0; i < warmup; ++i) {
            submit_pair(false, nullptr);
        }
        double total_ms = 0.0;
        for (int i = 0; i < iterations; ++i) {
            submit_pair(true, &total_ms);
        }
        stats.success = true;
        stats.avg_wall_ms = total_ms / static_cast<double>(iterations);
        const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        stats.copy_gib_per_s = gib / (stats.avg_wall_ms / 1000.0);
        stats.wall_vs_solo_sum_ratio = stats.avg_wall_ms / (copy_solo.avg_ms + kernel_solo.avg_ms);
        stats.wall_vs_solo_max_ratio = stats.avg_wall_ms / std::max(copy_solo.avg_ms, kernel_solo.avg_ms);
    } catch (const std::exception& ex) {
        stats.error = ex.what();
    }
    return stats;
}

bool validate_h2d_copy(Runtime& runtime, Resources& resources, size_t bytes, VkCommandBuffer cmd, VkFence fence) {
    std::vector<uint8_t> expected(bytes);
    fill_pattern(static_cast<uint8_t*>(resources.staging_upload.mapped), bytes, 0x5A);
    std::memcpy(expected.data(), resources.staging_upload.mapped, bytes);
    begin_command_buffer(cmd);
    VkBufferCopy region = {};
    region.size = static_cast<VkDeviceSize>(bytes);
    vkCmdCopyBuffer(cmd, resources.staging_upload.buffer, resources.device_buffer.buffer, 1, &region);
    vkCmdCopyBuffer(cmd, resources.device_buffer.buffer, resources.readback_buffer.buffer, 1, &region);
    end_command_buffer(cmd);
    submit_and_wait(runtime, runtime.transfer_queue, cmd, fence);
    return std::memcmp(resources.readback_buffer.mapped, expected.data(), bytes) == 0;
}

bool initialize_device_from_staging(Runtime& runtime, Resources& resources, size_t bytes, uint8_t seed, VkCommandBuffer cmd, VkFence fence) {
    fill_pattern(static_cast<uint8_t*>(resources.staging_upload.mapped), bytes, seed);
    begin_command_buffer(cmd);
    VkBufferCopy region = {};
    region.size = static_cast<VkDeviceSize>(bytes);
    vkCmdCopyBuffer(cmd, resources.staging_upload.buffer, resources.device_buffer.buffer, 1, &region);
    end_command_buffer(cmd);
    submit_and_wait(runtime, runtime.transfer_queue, cmd, fence);
    return true;
}

bool validate_d2h_copy(Runtime& runtime, Resources& resources, size_t bytes, VkCommandBuffer cmd, VkFence fence) {
    std::vector<uint8_t> expected(bytes);
    fill_pattern(static_cast<uint8_t*>(resources.staging_upload.mapped), bytes, 0xA5);
    std::memcpy(expected.data(), resources.staging_upload.mapped, bytes);
    begin_command_buffer(cmd);
    VkBufferCopy region = {};
    region.size = static_cast<VkDeviceSize>(bytes);
    vkCmdCopyBuffer(cmd, resources.staging_upload.buffer, resources.device_buffer.buffer, 1, &region);
    vkCmdCopyBuffer(cmd, resources.device_buffer.buffer, resources.readback_buffer.buffer, 1, &region);
    end_command_buffer(cmd);
    submit_and_wait(runtime, runtime.transfer_queue, cmd, fence);
    return std::memcmp(resources.readback_buffer.mapped, expected.data(), bytes) == 0;
}

bool validate_kernel_output(Runtime& runtime, Resources& resources, uint32_t loops, VkCommandBuffer cmd, VkFence fence) {
    update_uniform(resources, loops);
    begin_command_buffer(cmd);
    record_kernel_dispatch(runtime, resources, cmd);
    VkBufferMemoryBarrier post_barrier = {};
    post_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    post_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post_barrier.buffer = resources.kernel_output.buffer;
    post_barrier.offset = 0;
    post_barrier.size = resources.kernel_output.size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &post_barrier, 0, nullptr);
    VkBufferCopy region = {};
    region.size = resources.kernel_output.size;
    vkCmdCopyBuffer(cmd, resources.kernel_output.buffer, resources.kernel_output_readback.buffer, 1, &region);
    end_command_buffer(cmd);
    submit_and_wait(runtime, runtime.compute_queue, cmd, fence);

    static const std::array<uint32_t, 16> kProbeIndices = {
        0u, 1u, 2u, 3u,
        17u, 257u, 1023u, 4095u,
        8191u, 16384u, 32768u, 65535u,
        99991u, 131071u, 196607u, 262143u
    };
    const uint32_t* observed = static_cast<const uint32_t*>(resources.kernel_output_readback.mapped);
    for (uint32_t tid : kProbeIndices) {
        if (observed[tid] != compute_kernel_word(tid, loops)) {
            return false;
        }
    }
    return true;
}

DirectionRow run_direction_case(Runtime& runtime, Resources& resources, size_t bytes, bool is_h2d_like, int warmup, int iterations, VkCommandBuffer transfer_cmd, VkFence transfer_fence, VkCommandBuffer compute_cmd, VkFence compute_fence) {
    DirectionRow row;
    if (is_h2d_like) {
        fill_pattern(static_cast<uint8_t*>(resources.staging_upload.mapped), bytes, 0x13);
    } else {
        initialize_device_from_staging(runtime, resources, bytes, 0x31, transfer_cmd, transfer_fence);
    }
    row.copy_solo = measure_copy_wall(runtime, resources, bytes, is_h2d_like, warmup, iterations, transfer_cmd, transfer_fence);
    if (!row.copy_solo.success) {
        return row;
    }
    row.kernel_solo = calibrate_kernel_to_target(runtime, resources, row.copy_solo.avg_ms, warmup, iterations, compute_cmd, compute_fence);
    if (!row.kernel_solo.success) {
        return row;
    }
    row.overlap = measure_overlap_wall(runtime, resources, bytes, is_h2d_like, row.kernel_solo.loop_count, warmup, iterations, row.copy_solo, row.kernel_solo, transfer_cmd, transfer_fence, compute_cmd, compute_fence);
    return row;
}
std::string render_wall_stats_json(const WallStats& stats) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"success\":" << (stats.success ? "true" : "false") << ",";
    oss << "\"avg_ms\":" << format_double(stats.avg_ms) << ",";
    oss << "\"gib_per_s\":" << format_double(stats.gib_per_s);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_kernel_stats_json(const KernelStats& stats) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"success\":" << (stats.success ? "true" : "false") << ",";
    oss << "\"avg_ms\":" << format_double(stats.avg_ms) << ",";
    oss << "\"loop_count\":" << stats.loop_count;
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_overlap_stats_json(const OverlapStats& stats) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"success\":" << (stats.success ? "true" : "false") << ",";
    oss << "\"avg_wall_ms\":" << format_double(stats.avg_wall_ms) << ",";
    oss << "\"copy_gib_per_s\":" << format_double(stats.copy_gib_per_s) << ",";
    oss << "\"wall_vs_solo_sum_ratio\":" << format_double(stats.wall_vs_solo_sum_ratio) << ",";
    oss << "\"wall_vs_solo_max_ratio\":" << format_double(stats.wall_vs_solo_max_ratio);
    if (!stats.error.empty()) {
        oss << ",\"error\":" << quote(stats.error);
    }
    oss << "}";
    return oss.str();
}

std::string render_direction_json(const DirectionRow& row) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"copy_solo\":" << render_wall_stats_json(row.copy_solo) << ",";
    oss << "\"kernel_solo\":" << render_kernel_stats_json(row.kernel_solo) << ",";
    oss << "\"overlap\":" << render_overlap_stats_json(row.overlap);
    oss << "}";
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
        const CaseRow& row = rows[i];
        if (i > 0) {
            cases << ",";
        }
        cases << "{";
        cases << "\"size_mb\":" << row.size_mb << ",";
        cases << "\"iterations\":" << row.iterations << ",";
        cases << "\"warmup\":" << row.warmup << ",";
        cases << "\"h2d_like\":" << render_direction_json(row.h2d_like) << ",";
        cases << "\"d2h_like\":" << render_direction_json(row.d2h_like);
        cases << "}";
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

    const bool ok = validation_passed && have_h2d && have_d2h;
    const double min_all = (have_h2d && have_d2h) ? std::min(min_h2d, min_d2h) : 0.0;

    std::ostringstream oss;
    oss << "{";
    oss << "\"status\":" << quote(ok ? "ok" : "failed") << ",";
    oss << "\"primary_metric\":\"min_wall_vs_solo_sum_ratio\",";
    oss << "\"unit\":\"ratio\",";
    oss << "\"parameters\":{";
    oss << "\"api\":\"vulkan\",";
    oss << "\"copy_directions\":[\"H2D-like\",\"D2H-like\"],";
    oss << "\"iterations\":" << options.iterations << ",";
    oss << "\"warmup\":" << options.warmup << ",";
    oss << "\"sizes_mb\":[";
    for (size_t i = 0; i < options.sizes_mb.size(); ++i) {
        if (i > 0) oss << ",";
        oss << options.sizes_mb[i];
    }
    oss << "],";
    oss << "\"queue_count\":2},";
    oss << "\"measurement\":{";
    oss << "\"timing_backend\":\"wall_clock\",";
    oss << "\"adapter_name\":" << quote(runtime.adapter_name) << ",";
    oss << "\"cases\":" << cases.str() << ",";
    oss << "\"min_h2d_wall_vs_solo_sum_ratio\":" << format_double(have_h2d ? min_h2d : 0.0) << ",";
    oss << "\"min_d2h_wall_vs_solo_sum_ratio\":" << format_double(have_d2h ? min_d2h : 0.0) << ",";
    oss << "\"min_wall_vs_solo_sum_ratio\":" << format_double(min_all);
    oss << "},";
    oss << "\"validation\":{\"passed\":" << (validation_passed ? "true" : "false") << "}";
    oss << "}";
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    Runtime runtime;
    try {
        options = parse_common_args(argc, argv);
        runtime = create_runtime();

        VkCommandBuffer transfer_cmd = allocate_command_buffer(runtime, runtime.transfer_pool);
        VkCommandBuffer compute_cmd = allocate_command_buffer(runtime, runtime.compute_pool);
        VkFence transfer_fence = create_fence(runtime);
        VkFence compute_fence = create_fence(runtime);

        std::vector<CaseRow> rows;
        bool validation_passed = true;
        for (size_t size_mb : options.sizes_mb) {
            const int iterations = effective_iterations(size_mb, options.iterations);
            const int warmup = effective_warmup(size_mb, options.warmup);
            const size_t bytes = size_mb * 1024ull * 1024ull;

            Resources resources = create_resources(runtime, bytes);
            CaseRow row;
            row.size_mb = size_mb;
            row.iterations = iterations;
            row.warmup = warmup;

            if (!validate_h2d_copy(runtime, resources, bytes, transfer_cmd, transfer_fence)) {
                validation_passed = false;
            }
            if (!validate_d2h_copy(runtime, resources, bytes, transfer_cmd, transfer_fence)) {
                validation_passed = false;
            }

            row.h2d_like = run_direction_case(runtime, resources, bytes, true, warmup, iterations, transfer_cmd, transfer_fence, compute_cmd, compute_fence);
            row.d2h_like = run_direction_case(runtime, resources, bytes, false, warmup, iterations, transfer_cmd, transfer_fence, compute_cmd, compute_fence);

            const uint32_t validation_loops = std::max(row.h2d_like.kernel_solo.loop_count, row.d2h_like.kernel_solo.loop_count);
            if (validation_loops == 0 || !validate_kernel_output(runtime, resources, validation_loops, compute_cmd, compute_fence)) {
                validation_passed = false;
            }

            destroy_resources(runtime, resources);
            rows.push_back(row);
        }

        vkDestroyFence(runtime.device, transfer_fence, nullptr);
        vkDestroyFence(runtime.device, compute_fence, nullptr);
        emit_json(render_json(options, runtime, rows, validation_passed));
        destroy_runtime(runtime);
        return 0;
    } catch (const std::exception& ex) {
        emit_json(make_error_json("failed", ex.what(), options, "min_wall_vs_solo_sum_ratio"));
        destroy_runtime(runtime);
        return 1;
    }
}

