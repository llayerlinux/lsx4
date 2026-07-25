// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef __ANDROID__

#define VK_USE_PLATFORM_ANDROID_KHR 1

#include "executor/ps5_desktop/vulkan_presenter.h"

#include <android/log.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace Lsx4::Ps5Desktop {
namespace {

struct Presenter {
    ANativeWindow* window{};
    VkInstance instance{};
    VkSurfaceKHR surface{};
    VkPhysicalDevice physical_device{};
    VkDevice device{};
    std::uint32_t queue_family{std::numeric_limits<std::uint32_t>::max()};
    VkQueue queue{};
    VkSwapchainKHR swapchain{};
    VkFormat swapchain_format{VK_FORMAT_UNDEFINED};
    VkExtent2D swapchain_extent{};
    std::uint32_t configured_width{};
    std::uint32_t configured_height{};
    std::vector<VkImage> swapchain_images;
    VkCommandPool command_pool{};
    VkCommandBuffer command_buffer{};
    VkSemaphore acquired{};
    VkSemaphore rendered{};
    VkFence fence{};
    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    void* staging_map{};
    VkDeviceSize staging_capacity{};
    VkImage source_image{};
    VkDeviceMemory source_memory{};
    VkExtent2D source_extent{};
    bool source_initialized{};
    bool permanently_failed{};
};

std::mutex g_mutex;
Presenter g_presenter;

void LogFailure(const char* const operation, const VkResult result) {
    __android_log_print(
        ANDROID_LOG_WARN, "LSX4-PS5",
        "vulkan presenter failed operation=%s result=%d",
        operation, static_cast<int>(result));
}

std::uint32_t FindMemoryType(const Presenter& presenter,
                             const std::uint32_t type_bits,
                             const VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(
        presenter.physical_device, &properties);
    for (std::uint32_t index = 0;
         index < properties.memoryTypeCount; ++index) {
        if ((type_bits & (1u << index)) != 0u &&
            (properties.memoryTypes[index].propertyFlags & required) ==
                required) {
            return index;
        }
    }
    return std::numeric_limits<std::uint32_t>::max();
}

void DestroySource(Presenter& presenter) {
    if (presenter.device != VK_NULL_HANDLE) {
        if (presenter.source_image != VK_NULL_HANDLE) {
            vkDestroyImage(
                presenter.device, presenter.source_image, nullptr);
        }
        if (presenter.source_memory != VK_NULL_HANDLE) {
            vkFreeMemory(
                presenter.device, presenter.source_memory, nullptr);
        }
    }
    presenter.source_image = VK_NULL_HANDLE;
    presenter.source_memory = VK_NULL_HANDLE;
    presenter.source_extent = {};
    presenter.source_initialized = false;
}

void DestroyStaging(Presenter& presenter) {
    if (presenter.device != VK_NULL_HANDLE) {
        if (presenter.staging_map != nullptr &&
            presenter.staging_memory != VK_NULL_HANDLE) {
            vkUnmapMemory(
                presenter.device, presenter.staging_memory);
        }
        if (presenter.staging != VK_NULL_HANDLE) {
            vkDestroyBuffer(
                presenter.device, presenter.staging, nullptr);
        }
        if (presenter.staging_memory != VK_NULL_HANDLE) {
            vkFreeMemory(
                presenter.device, presenter.staging_memory, nullptr);
        }
    }
    presenter.staging = VK_NULL_HANDLE;
    presenter.staging_memory = VK_NULL_HANDLE;
    presenter.staging_map = nullptr;
    presenter.staging_capacity = 0;
}

void DestroyPresenter(Presenter& presenter) {
    if (presenter.device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(presenter.device);
    }
    DestroySource(presenter);
    DestroyStaging(presenter);
    if (presenter.device != VK_NULL_HANDLE) {
        if (presenter.fence != VK_NULL_HANDLE) {
            vkDestroyFence(presenter.device, presenter.fence, nullptr);
        }
        if (presenter.rendered != VK_NULL_HANDLE) {
            vkDestroySemaphore(
                presenter.device, presenter.rendered, nullptr);
        }
        if (presenter.acquired != VK_NULL_HANDLE) {
            vkDestroySemaphore(
                presenter.device, presenter.acquired, nullptr);
        }
        if (presenter.command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(
                presenter.device, presenter.command_pool, nullptr);
        }
        if (presenter.swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(
                presenter.device, presenter.swapchain, nullptr);
        }
        vkDestroyDevice(presenter.device, nullptr);
    }
    if (presenter.surface != VK_NULL_HANDLE &&
        presenter.instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(
            presenter.instance, presenter.surface, nullptr);
    }
    if (presenter.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(presenter.instance, nullptr);
    }
    presenter = {};
}

bool CreateDeviceAndSurface(Presenter& presenter,
                            ANativeWindow* const window) {
    constexpr std::array InstanceExtensions{
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "LSX4 PS5 isolated presenter",
        1,
        "LSX4",
        1,
        VK_API_VERSION_1_1};
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        static_cast<std::uint32_t>(InstanceExtensions.size()),
        InstanceExtensions.data()};
    auto result = vkCreateInstance(
        &instance_info, nullptr, &presenter.instance);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateInstance", result);
        return false;
    }
    const VkAndroidSurfaceCreateInfoKHR surface_info{
        VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
        nullptr,
        0,
        window};
    result = vkCreateAndroidSurfaceKHR(
        presenter.instance, &surface_info, nullptr,
        &presenter.surface);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateAndroidSurfaceKHR", result);
        return false;
    }
    std::uint32_t physical_count{};
    result = vkEnumeratePhysicalDevices(
        presenter.instance, &physical_count, nullptr);
    if (result != VK_SUCCESS || physical_count == 0) {
        LogFailure("vkEnumeratePhysicalDevices", result);
        return false;
    }
    std::vector<VkPhysicalDevice> devices(physical_count);
    result = vkEnumeratePhysicalDevices(
        presenter.instance, &physical_count, devices.data());
    if (result != VK_SUCCESS) {
        LogFailure("vkEnumeratePhysicalDevices(list)", result);
        return false;
    }
    for (const auto device : devices) {
        std::uint32_t family_count{};
        vkGetPhysicalDeviceQueueFamilyProperties(
            device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            device, &family_count, families.data());
        for (std::uint32_t family = 0;
             family < family_count; ++family) {
            VkBool32 present_support{};
            (void)vkGetPhysicalDeviceSurfaceSupportKHR(
                device, family, presenter.surface,
                &present_support);
            if ((families[family].queueFlags &
                 VK_QUEUE_GRAPHICS_BIT) != 0u &&
                present_support == VK_TRUE) {
                presenter.physical_device = device;
                presenter.queue_family = family;
                break;
            }
        }
        if (presenter.physical_device != VK_NULL_HANDLE) {
            break;
        }
    }
    if (presenter.physical_device == VK_NULL_HANDLE) {
        return false;
    }
    constexpr float QueuePriority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        presenter.queue_family,
        1,
        &QueuePriority};
    constexpr std::array DeviceExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        nullptr,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(DeviceExtensions.size()),
        DeviceExtensions.data(),
        nullptr};
    result = vkCreateDevice(
        presenter.physical_device, &device_info, nullptr,
        &presenter.device);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateDevice", result);
        return false;
    }
    vkGetDeviceQueue(
        presenter.device, presenter.queue_family, 0,
        &presenter.queue);
    presenter.window = window;
    return true;
}

bool CreateSwapchain(Presenter& presenter) {
    VkSurfaceCapabilitiesKHR capabilities{};
    auto result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        presenter.physical_device, presenter.surface,
        &capabilities);
    if (result != VK_SUCCESS) {
        LogFailure("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", result);
        return false;
    }
    if ((capabilities.supportedUsageFlags &
         VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0u) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4-PS5",
            "vulkan surface lacks transfer-dst usage");
        return false;
    }
    std::uint32_t format_count{};
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(
        presenter.physical_device, presenter.surface,
        &format_count, nullptr);
    if (result != VK_SUCCESS || format_count == 0) {
        LogFailure("vkGetPhysicalDeviceSurfaceFormatsKHR", result);
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(
        presenter.physical_device, presenter.surface,
        &format_count, formats.data());
    if (result != VK_SUCCESS) {
        LogFailure("vkGetPhysicalDeviceSurfaceFormatsKHR(list)", result);
        return false;
    }
    auto selected = formats.front();
    for (const auto& format : formats) {
        if ((format.format == VK_FORMAT_R8G8B8A8_UNORM ||
             format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            selected = format;
            break;
        }
    }
    std::uint32_t present_mode_count{};
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        presenter.physical_device, presenter.surface,
        &present_mode_count, nullptr);
    if (result != VK_SUCCESS || present_mode_count == 0) {
        LogFailure("vkGetPhysicalDeviceSurfacePresentModesKHR",
                   result);
        return false;
    }
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(
        presenter.physical_device, presenter.surface,
        &present_mode_count, present_modes.data());
    if (result != VK_SUCCESS) {
        LogFailure("vkGetPhysicalDeviceSurfacePresentModesKHR(list)",
                   result);
        return false;
    }
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto mode : present_modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR &&
            present_mode == VK_PRESENT_MODE_FIFO_KHR) {
            present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        }
    }
    auto extent = capabilities.currentExtent;
    if (extent.width == std::numeric_limits<std::uint32_t>::max()) {
        extent.width = std::clamp(
            static_cast<std::uint32_t>(
                std::max(ANativeWindow_getWidth(presenter.window), 1)),
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        extent.height = std::clamp(
            static_cast<std::uint32_t>(
                std::max(ANativeWindow_getHeight(presenter.window), 1)),
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
    }
    if (extent.width == 0 || extent.height == 0) {
        return false;
    }
    const auto image_count = std::clamp(
        capabilities.minImageCount + 1u,
        capabilities.minImageCount,
        capabilities.maxImageCount == 0
            ? std::numeric_limits<std::uint32_t>::max()
            : capabilities.maxImageCount);
    VkCompositeAlphaFlagBitsKHR composite =
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (const auto candidate : {
             VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
             VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
             VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
        if ((capabilities.supportedCompositeAlpha & candidate) != 0u) {
            composite = candidate;
            break;
        }
    }
    const auto pre_transform =
        (capabilities.supportedTransforms &
         VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) != 0u
        ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        : capabilities.currentTransform;
    const VkSwapchainCreateInfoKHR info{
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        nullptr,
        0,
        presenter.surface,
        image_count,
        selected.format,
        selected.colorSpace,
        extent,
        1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        pre_transform,
        composite,
        present_mode,
        VK_TRUE,
        VK_NULL_HANDLE};
    result = vkCreateSwapchainKHR(
        presenter.device, &info, nullptr,
        &presenter.swapchain);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSwapchainKHR", result);
        return false;
    }
    std::uint32_t swapchain_count{};
    result = vkGetSwapchainImagesKHR(
        presenter.device, presenter.swapchain,
        &swapchain_count, nullptr);
    if (result != VK_SUCCESS || swapchain_count == 0) {
        LogFailure("vkGetSwapchainImagesKHR", result);
        return false;
    }
    presenter.swapchain_images.resize(swapchain_count);
    result = vkGetSwapchainImagesKHR(
        presenter.device, presenter.swapchain,
        &swapchain_count, presenter.swapchain_images.data());
    if (result != VK_SUCCESS) {
        LogFailure("vkGetSwapchainImagesKHR(list)", result);
        return false;
    }
    presenter.swapchain_format = selected.format;
    presenter.swapchain_extent = extent;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-PS5",
        "vulkan surface current_transform=0x%x pre_transform=0x%x "
        "supported=0x%x",
        capabilities.currentTransform, pre_transform,
        capabilities.supportedTransforms);
    return true;
}

bool CreateCommandsAndSync(Presenter& presenter) {
    const VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        presenter.queue_family};
    auto result = vkCreateCommandPool(
        presenter.device, &pool_info, nullptr,
        &presenter.command_pool);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateCommandPool", result);
        return false;
    }
    const VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        presenter.command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1};
    result = vkAllocateCommandBuffers(
        presenter.device, &command_info,
        &presenter.command_buffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkAllocateCommandBuffers", result);
        return false;
    }
    const VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    result = vkCreateSemaphore(
        presenter.device, &semaphore_info, nullptr,
        &presenter.acquired);
    if (result == VK_SUCCESS) {
        result = vkCreateSemaphore(
            presenter.device, &semaphore_info, nullptr,
            &presenter.rendered);
    }
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSemaphore", result);
        return false;
    }
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        nullptr,
        VK_FENCE_CREATE_SIGNALED_BIT};
    result = vkCreateFence(
        presenter.device, &fence_info, nullptr,
        &presenter.fence);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateFence", result);
        return false;
    }
    return true;
}

bool EnsureStaging(Presenter& presenter,
                   const VkDeviceSize capacity) {
    if (presenter.staging_capacity >= capacity &&
        presenter.staging_map != nullptr) {
        return true;
    }
    (void)vkDeviceWaitIdle(presenter.device);
    DestroyStaging(presenter);
    const VkBufferCreateInfo buffer_info{
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        capacity,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_SHARING_MODE_EXCLUSIVE};
    auto result = vkCreateBuffer(
        presenter.device, &buffer_info, nullptr,
        &presenter.staging);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateBuffer", result);
        return false;
    }
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(
        presenter.device, presenter.staging, &requirements);
    const auto memory_type = FindMemoryType(
        presenter, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const VkMemoryAllocateInfo allocation{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        memory_type};
    result = vkAllocateMemory(
        presenter.device, &allocation, nullptr,
        &presenter.staging_memory);
    if (result == VK_SUCCESS) {
        result = vkBindBufferMemory(
            presenter.device, presenter.staging,
            presenter.staging_memory, 0);
    }
    if (result == VK_SUCCESS) {
        result = vkMapMemory(
            presenter.device, presenter.staging_memory,
            0, capacity, 0, &presenter.staging_map);
    }
    if (result != VK_SUCCESS) {
        LogFailure("staging memory", result);
        return false;
    }
    presenter.staging_capacity = capacity;
    return true;
}

bool EnsureSourceImage(Presenter& presenter,
                       const std::uint32_t width,
                       const std::uint32_t height) {
    if (presenter.source_image != VK_NULL_HANDLE &&
        presenter.source_extent.width == width &&
        presenter.source_extent.height == height) {
        return true;
    }
    (void)vkDeviceWaitIdle(presenter.device);
    DestroySource(presenter);
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
        0,
        VK_IMAGE_TYPE_2D,
        VK_FORMAT_R8G8B8A8_UNORM,
        {width, height, 1},
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED};
    auto result = vkCreateImage(
        presenter.device, &image_info, nullptr,
        &presenter.source_image);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateImage", result);
        return false;
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(
        presenter.device, presenter.source_image,
        &requirements);
    const auto memory_type = FindMemoryType(
        presenter, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const VkMemoryAllocateInfo allocation{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        memory_type};
    result = vkAllocateMemory(
        presenter.device, &allocation, nullptr,
        &presenter.source_memory);
    if (result == VK_SUCCESS) {
        result = vkBindImageMemory(
            presenter.device, presenter.source_image,
            presenter.source_memory, 0);
    }
    if (result != VK_SUCCESS) {
        LogFailure("source image memory", result);
        return false;
    }
    presenter.source_extent = {width, height};
    presenter.source_initialized = false;
    return true;
}

bool EnsureSwapchain(Presenter& presenter,
                     const std::uint32_t width,
                     const std::uint32_t height) {
    if (presenter.swapchain == VK_NULL_HANDLE ||
        (presenter.swapchain_extent.width == width &&
         presenter.swapchain_extent.height == height)) {
        return presenter.swapchain != VK_NULL_HANDLE;
    }
    if (presenter.device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(presenter.device);
    }
    if (presenter.swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(presenter.device,
                              presenter.swapchain, nullptr);
        presenter.swapchain = VK_NULL_HANDLE;
    }
    presenter.swapchain_images.clear();
    return CreateSwapchain(presenter);
}

bool Initialize(Presenter& presenter, ANativeWindow* const window) {
    if (!CreateDeviceAndSurface(presenter, window) ||
        !CreateSwapchain(presenter) ||
        !CreateCommandsAndSync(presenter)) {
        return false;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        presenter.physical_device, &properties);
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-PS5",
        "vulkan presenter ready gpu=%s swapchain=%ux%u format=%d images=%zu",
        properties.deviceName,
        presenter.swapchain_extent.width,
        presenter.swapchain_extent.height,
        static_cast<int>(presenter.swapchain_format),
        presenter.swapchain_images.size());
    return true;
}

void ImageBarrier(VkCommandBuffer command,
                  VkImage image,
                  VkImageLayout old_layout,
                  VkImageLayout new_layout,
                  VkAccessFlags source_access,
                  VkAccessFlags destination_access,
                  VkPipelineStageFlags source_stage,
                  VkPipelineStageFlags destination_stage) {
    const VkImageMemoryBarrier barrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        source_access,
        destination_access,
        old_layout,
        new_layout,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(
        command, source_stage, destination_stage, 0,
        0, nullptr, 0, nullptr, 1, &barrier);
}

bool SubmitFrame(Presenter& presenter,
                 const std::uint8_t* const rgba,
                 const std::size_t byte_count,
                 const std::uint32_t width,
                 const std::uint32_t height) {
    if (presenter.source_initialized &&
        presenter.device != VK_NULL_HANDLE &&
        presenter.fence != VK_NULL_HANDLE) {
        const auto fence_status = vkGetFenceStatus(
            presenter.device, presenter.fence);
        if (fence_status == VK_NOT_READY) {
            // Keep input and CPU threads moving when the GPU is already
            // processing the previous frame. The stale contents are still
            // on screen; presenting this frame would block and worsen
            // interactivity.
            return true;
        }
        if (fence_status != VK_SUCCESS) {
            LogFailure("vkGetFenceStatus", fence_status);
            return false;
        }
    }
    if (!EnsureStaging(presenter, byte_count) ||
        !EnsureSwapchain(presenter, width, height)) {
        return false;
    }
    VkResult result{};
    std::uint32_t image_index{};
    result = vkAcquireNextImageKHR(
        presenter.device, presenter.swapchain, 0u,
        presenter.acquired, VK_NULL_HANDLE, &image_index);
    if (result == VK_TIMEOUT || result == VK_NOT_READY) {
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LogFailure("vkAcquireNextImageKHR", result);
        return false;
    }
    std::memcpy(presenter.staging_map, rgba, byte_count);
    (void)vkResetFences(
        presenter.device, 1, &presenter.fence);
    (void)vkResetCommandBuffer(
        presenter.command_buffer, 0);
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    result = vkBeginCommandBuffer(
        presenter.command_buffer, &begin);
    if (result != VK_SUCCESS) {
        LogFailure("vkBeginCommandBuffer", result);
        return false;
    }
    const bool direct_copy =
        width == presenter.swapchain_extent.width &&
        height == presenter.swapchain_extent.height;
    if (!direct_copy &&
        !EnsureSourceImage(presenter, width, height)) {
        return false;
    }
    const VkBufferImageCopy copy_to_surface{
        0,
        0,
        0,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0},
        {width, height, 1}};
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (direct_copy) {
        vkCmdCopyBufferToImage(
            presenter.command_buffer, presenter.staging,
            presenter.swapchain_images[image_index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copy_to_surface);
    } else {
        ImageBarrier(
            presenter.command_buffer, presenter.source_image,
            presenter.source_initialized
                ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            presenter.source_initialized
                ? VK_ACCESS_TRANSFER_READ_BIT : 0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            presenter.source_initialized
                ? VK_PIPELINE_STAGE_TRANSFER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdCopyBufferToImage(
            presenter.command_buffer, presenter.staging,
            presenter.source_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copy_to_surface);
        ImageBarrier(
            presenter.command_buffer, presenter.source_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkImageCopy image_copy{
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0},
            {static_cast<std::uint32_t>(width),
             static_cast<std::uint32_t>(height),
             1u}};
        vkCmdCopyImage(
            presenter.command_buffer, presenter.source_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            presenter.swapchain_images[image_index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &image_copy);
        const VkImageBlit blit{
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {{0, 0, 0},
             {static_cast<std::int32_t>(width),
              static_cast<std::int32_t>(height), 1}},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {{0, 0, 0},
             {static_cast<std::int32_t>(
                  presenter.swapchain_extent.width),
              static_cast<std::int32_t>(
                  presenter.swapchain_extent.height), 1}}};
        vkCmdBlitImage(
            presenter.command_buffer, presenter.source_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            presenter.swapchain_images[image_index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);
    }
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    result = vkEndCommandBuffer(presenter.command_buffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkEndCommandBuffer", result);
        return false;
    }
    constexpr VkPipelineStageFlags WaitStage =
        VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        1,
        &presenter.acquired,
        &WaitStage,
        1,
        &presenter.command_buffer,
        1,
        &presenter.rendered};
    result = vkQueueSubmit(
        presenter.queue, 1, &submit, presenter.fence);
    if (result != VK_SUCCESS) {
        LogFailure("vkQueueSubmit", result);
        return false;
    }
    presenter.source_initialized = true;
    const VkPresentInfoKHR present{
        VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        nullptr,
        1,
        &presenter.rendered,
        1,
        &presenter.swapchain,
        &image_index,
        nullptr};
    result = vkQueuePresentKHR(presenter.queue, &present);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LogFailure("vkQueuePresentKHR", result);
        return false;
    }
    return true;
}

}  // namespace

bool PresentVulkanFrame(ANativeWindow* const window,
                        const std::uint8_t* const rgba,
                        const std::size_t byte_count,
                        const std::uint32_t width,
                        const std::uint32_t height,
                        const std::uint64_t frame_number) {
    if (window == nullptr || rgba == nullptr ||
        width == 0 || height == 0 ||
        byte_count !=
            static_cast<std::size_t>(width) * height * 4u) {
        return false;
    }
    const std::lock_guard lock{g_mutex};
    if (g_presenter.window != nullptr &&
        g_presenter.window != window) {
        DestroyPresenter(g_presenter);
    }
    if (g_presenter.configured_width != width ||
        g_presenter.configured_height != height) {
        (void)ANativeWindow_setBuffersGeometry(
            window,
            static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(height),
            WINDOW_FORMAT_RGBA_8888);
        g_presenter.configured_width = width;
        g_presenter.configured_height = height;
    }
    if (g_presenter.permanently_failed) {
        return false;
    }
    if (g_presenter.device == VK_NULL_HANDLE &&
        !Initialize(g_presenter, window)) {
        DestroyPresenter(g_presenter);
        g_presenter.window = window;
        g_presenter.permanently_failed = true;
        return false;
    }
    if (!SubmitFrame(
            g_presenter, rgba, byte_count, width, height)) {
        DestroyPresenter(g_presenter);
        g_presenter.window = window;
        return false;
    }
    if (frame_number <= 8u || frame_number % 120u == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "vulkan present count=%llu source=%ux%u swapchain=%ux%u",
            static_cast<unsigned long long>(frame_number),
            width, height,
            g_presenter.swapchain_extent.width,
            g_presenter.swapchain_extent.height);
    }
    return true;
}

void ResetVulkanPresenter() {
    const std::lock_guard lock{g_mutex};
    DestroyPresenter(g_presenter);
}

}  // namespace Lsx4::Ps5Desktop

#endif
