// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef __ANDROID__

#define VK_USE_PLATFORM_ANDROID_KHR 1

#include "executor/ps5_desktop/vulkan_presenter.h"
#include "executor/ps5_desktop/gen5_compute_recompiler.h"
#include "executor/ps5_desktop/vulkan_guest_shaders.h"

#include <android/log.h>
#include <android/native_window.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Lsx4::Ps5Desktop {
namespace {

inline constexpr std::uint32_t VulkanFullscreenTriangleShader[] =
#include "executor/ps5_desktop/vulkan_fullscreen_triangle.spv.inc"
;

struct GuestTextureResource {
    std::uint64_t key{};
    std::uint64_t signature{};
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkDescriptorSet descriptor{};
    std::uint32_t width{};
    std::uint32_t height{};
    bool initialized{};
    bool repeat_texture{};
    bool nearest_texture{};
    std::uint64_t last_use{};
};

struct GuestTargetResource {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkFramebuffer framebuffer{};
    VkDescriptorSet descriptor{};
    VkDescriptorSet nearest_descriptor{};
    VkExtent2D extent{};
    bool initialized{};
    std::uint64_t last_batch_id{};
    std::uint64_t last_use{};
};

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
    VkRenderPass guest_render_pass{};
    VkImage guest_target_image{};
    VkDeviceMemory guest_target_memory{};
    VkImageView guest_target_view{};
    VkFramebuffer guest_target_framebuffer{};
    VkExtent2D guest_target_extent{};
    bool guest_target_initialized{};
    VkDescriptorSetLayout guest_descriptor_layout{};
    VkDescriptorPool guest_descriptor_pool{};
    VkPipelineLayout guest_pipeline_layout{};
    VkPipeline guest_pipeline{};
    VkPipeline guest_premultiplied_pipeline{};
    VkPipeline guest_destination_source_alpha_pipeline{};
    VkPipeline guest_destination_inverse_source_alpha_pipeline{};
    VkPipeline guest_additive_pipeline{};
    VkPipeline guest_wave_pipeline{};
    VkPipeline guest_opaque_pipeline{};
    VkShaderModule guest_vertex_shader{};
    VkShaderModule guest_fragment_shader{};
    VkShaderModule guest_wave_fragment_shader{};
    VkSampler guest_sampler{};
    VkSampler guest_repeat_sampler{};
    VkSampler guest_nearest_sampler{};
    VkSampler guest_nearest_repeat_sampler{};
    std::unordered_map<std::uint64_t, GuestTextureResource>
        guest_textures;
    std::unordered_map<std::uint64_t, GuestTargetResource>
        guest_targets;
    std::uint64_t guest_frame_serial{};
    std::uint64_t last_guest_batch_id{};
    std::uint64_t last_guest_target_key{};
    VkImage source_image{};
    VkDeviceMemory source_memory{};
    VkExtent2D source_extent{};
    bool source_initialized{};
    bool submission_in_flight{};
    bool permanently_failed{};
};

std::mutex g_mutex;
Presenter g_presenter;
std::atomic<std::uint64_t> g_present_count{};

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

void DestroyGuestSwapchainResources(Presenter& presenter) {
    if (presenter.device == VK_NULL_HANDLE) {
        return;
    }
    for (const auto& [key, target] : presenter.guest_targets) {
        (void)key;
        if (target.framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(
                presenter.device, target.framebuffer, nullptr);
        }
        if (target.view != VK_NULL_HANDLE) {
            vkDestroyImageView(
                presenter.device, target.view, nullptr);
        }
        if (target.image != VK_NULL_HANDLE) {
            vkDestroyImage(
                presenter.device, target.image, nullptr);
        }
        if (target.memory != VK_NULL_HANDLE) {
            vkFreeMemory(
                presenter.device, target.memory, nullptr);
        }
    }
    presenter.guest_targets.clear();
    if (presenter.guest_target_framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(
            presenter.device, presenter.guest_target_framebuffer,
            nullptr);
        presenter.guest_target_framebuffer = VK_NULL_HANDLE;
    }
    if (presenter.guest_target_view != VK_NULL_HANDLE) {
        vkDestroyImageView(
            presenter.device, presenter.guest_target_view, nullptr);
        presenter.guest_target_view = VK_NULL_HANDLE;
    }
    if (presenter.guest_target_image != VK_NULL_HANDLE) {
        vkDestroyImage(
            presenter.device, presenter.guest_target_image, nullptr);
        presenter.guest_target_image = VK_NULL_HANDLE;
    }
    if (presenter.guest_target_memory != VK_NULL_HANDLE) {
        vkFreeMemory(
            presenter.device, presenter.guest_target_memory, nullptr);
        presenter.guest_target_memory = VK_NULL_HANDLE;
    }
    presenter.guest_target_extent = {};
    presenter.guest_target_initialized = false;
    presenter.last_guest_batch_id = 0;
    presenter.last_guest_target_key = 0;
    if (presenter.guest_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device, presenter.guest_pipeline, nullptr);
        presenter.guest_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_premultiplied_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device,
            presenter.guest_premultiplied_pipeline, nullptr);
        presenter.guest_premultiplied_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_destination_source_alpha_pipeline !=
        VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device,
            presenter.guest_destination_source_alpha_pipeline, nullptr);
        presenter.guest_destination_source_alpha_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_destination_inverse_source_alpha_pipeline !=
        VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device,
            presenter.guest_destination_inverse_source_alpha_pipeline,
            nullptr);
        presenter.guest_destination_inverse_source_alpha_pipeline =
            VK_NULL_HANDLE;
    }
    if (presenter.guest_additive_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device, presenter.guest_additive_pipeline,
            nullptr);
        presenter.guest_additive_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_wave_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device, presenter.guest_wave_pipeline, nullptr);
        presenter.guest_wave_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_opaque_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            presenter.device, presenter.guest_opaque_pipeline, nullptr);
        presenter.guest_opaque_pipeline = VK_NULL_HANDLE;
    }
    if (presenter.guest_render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(
            presenter.device, presenter.guest_render_pass, nullptr);
        presenter.guest_render_pass = VK_NULL_HANDLE;
    }
}

void DestroyGuestResources(Presenter& presenter) {
    if (presenter.device == VK_NULL_HANDLE) {
        return;
    }
    DestroyGuestSwapchainResources(presenter);
    for (const auto& [key, texture] : presenter.guest_textures) {
        (void)key;
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(presenter.device, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
            vkDestroyImage(presenter.device, texture.image, nullptr);
        }
        if (texture.memory != VK_NULL_HANDLE) {
            vkFreeMemory(presenter.device, texture.memory, nullptr);
        }
    }
    presenter.guest_textures.clear();
    if (presenter.guest_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            presenter.device, presenter.guest_sampler, nullptr);
        presenter.guest_sampler = VK_NULL_HANDLE;
    }
    if (presenter.guest_repeat_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            presenter.device, presenter.guest_repeat_sampler, nullptr);
        presenter.guest_repeat_sampler = VK_NULL_HANDLE;
    }
    if (presenter.guest_nearest_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            presenter.device, presenter.guest_nearest_sampler, nullptr);
        presenter.guest_nearest_sampler = VK_NULL_HANDLE;
    }
    if (presenter.guest_nearest_repeat_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            presenter.device, presenter.guest_nearest_repeat_sampler,
            nullptr);
        presenter.guest_nearest_repeat_sampler = VK_NULL_HANDLE;
    }
    if (presenter.guest_descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            presenter.device, presenter.guest_descriptor_pool, nullptr);
        presenter.guest_descriptor_pool = VK_NULL_HANDLE;
    }
    if (presenter.guest_pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            presenter.device, presenter.guest_pipeline_layout, nullptr);
        presenter.guest_pipeline_layout = VK_NULL_HANDLE;
    }
    if (presenter.guest_descriptor_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            presenter.device, presenter.guest_descriptor_layout, nullptr);
        presenter.guest_descriptor_layout = VK_NULL_HANDLE;
    }
    if (presenter.guest_vertex_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            presenter.device, presenter.guest_vertex_shader, nullptr);
        presenter.guest_vertex_shader = VK_NULL_HANDLE;
    }
    if (presenter.guest_fragment_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            presenter.device, presenter.guest_fragment_shader, nullptr);
        presenter.guest_fragment_shader = VK_NULL_HANDLE;
    }
    if (presenter.guest_wave_fragment_shader != VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            presenter.device, presenter.guest_wave_fragment_shader,
            nullptr);
        presenter.guest_wave_fragment_shader = VK_NULL_HANDLE;
    }
}

void DestroyPresenter(Presenter& presenter) {
    if (presenter.device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(presenter.device);
    }
    DestroySource(presenter);
    DestroyStaging(presenter);
    DestroyGuestResources(presenter);
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
    VkPhysicalDeviceFeatures supported_features{};
    vkGetPhysicalDeviceFeatures(
        presenter.physical_device, &supported_features);
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderStorageImageReadWithoutFormat =
        supported_features.shaderStorageImageReadWithoutFormat;
    enabled_features.shaderStorageImageWriteWithoutFormat =
        supported_features.shaderStorageImageWriteWithoutFormat;
    enabled_features.shaderInt64 =
        supported_features.shaderInt64;
    enabled_features.shaderFloat64 =
        supported_features.shaderFloat64;
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
        &enabled_features};
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
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            selected = format;
            break;
        }
    }
    if (selected.format != VK_FORMAT_R8G8B8A8_UNORM) {
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                selected = format;
                break;
            }
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
    VkImageUsageFlags image_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((capabilities.supportedUsageFlags &
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0u) {
        image_usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
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
        image_usage,
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
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
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
    DestroyGuestSwapchainResources(presenter);
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

VkDeviceSize AlignGuestOffset(const VkDeviceSize value,
                              const VkDeviceSize alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

bool CreateGuestFixedResources(Presenter& presenter) {
    if (presenter.guest_descriptor_layout != VK_NULL_HANDLE) {
        return true;
    }
    const VkDescriptorSetLayoutBinding binding{
        0,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        1,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        nullptr};
    const VkDescriptorSetLayoutCreateInfo descriptor_layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &binding};
    auto result = vkCreateDescriptorSetLayout(
        presenter.device, &descriptor_layout_info, nullptr,
        &presenter.guest_descriptor_layout);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateDescriptorSetLayout(guest)", result);
        return false;
    }
    const VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024};
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        0,
        1024,
        1,
        &pool_size};
    result = vkCreateDescriptorPool(
        presenter.device, &pool_info, nullptr,
        &presenter.guest_descriptor_pool);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateDescriptorPool(guest)", result);
        return false;
    }
    const VkPushConstantRange push_range{
        VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(float) * 16u};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &presenter.guest_descriptor_layout,
        1,
        &push_range};
    result = vkCreatePipelineLayout(
        presenter.device, &pipeline_layout_info, nullptr,
        &presenter.guest_pipeline_layout);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreatePipelineLayout(guest)", result);
        return false;
    }
    const VkShaderModuleCreateInfo vertex_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        sizeof(VulkanGuestVertexShader),
        VulkanGuestVertexShader};
    const VkShaderModuleCreateInfo fragment_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        sizeof(VulkanGuestFragmentShader),
        VulkanGuestFragmentShader};
    const VkShaderModuleCreateInfo wave_fragment_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        sizeof(VulkanGuestWaveFragmentShader),
        VulkanGuestWaveFragmentShader};
    result = vkCreateShaderModule(
        presenter.device, &vertex_info, nullptr,
        &presenter.guest_vertex_shader);
    if (result == VK_SUCCESS) {
        result = vkCreateShaderModule(
            presenter.device, &fragment_info, nullptr,
            &presenter.guest_fragment_shader);
    }
    if (result == VK_SUCCESS) {
        result = vkCreateShaderModule(
            presenter.device, &wave_fragment_info, nullptr,
            &presenter.guest_wave_fragment_shader);
    }
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateShaderModule(guest)", result);
        return false;
    }
    const VkSamplerCreateInfo sampler_info{
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        nullptr,
        0,
        VK_FILTER_LINEAR,
        VK_FILTER_LINEAR,
        VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        0.0f,
        VK_FALSE,
        1.0f,
        VK_FALSE,
        VK_COMPARE_OP_NEVER,
        0.0f,
        0.0f,
        VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        VK_FALSE};
    result = vkCreateSampler(
        presenter.device, &sampler_info, nullptr,
        &presenter.guest_sampler);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSampler(guest)", result);
        return false;
    }
    auto repeat_sampler_info = sampler_info;
    repeat_sampler_info.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    repeat_sampler_info.addressModeV =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    result = vkCreateSampler(
        presenter.device, &repeat_sampler_info, nullptr,
        &presenter.guest_repeat_sampler);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSampler(guest repeat)", result);
        return false;
    }
    auto nearest_sampler_info = sampler_info;
    nearest_sampler_info.magFilter = VK_FILTER_NEAREST;
    nearest_sampler_info.minFilter = VK_FILTER_NEAREST;
    result = vkCreateSampler(
        presenter.device, &nearest_sampler_info, nullptr,
        &presenter.guest_nearest_sampler);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSampler(guest nearest)", result);
        return false;
    }
    nearest_sampler_info.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    nearest_sampler_info.addressModeV =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;
    result = vkCreateSampler(
        presenter.device, &nearest_sampler_info, nullptr,
        &presenter.guest_nearest_repeat_sampler);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateSampler(guest nearest repeat)", result);
        return false;
    }
    return true;
}

bool CreateGuestSwapchainResources(Presenter& presenter) {
    if (presenter.guest_pipeline != VK_NULL_HANDLE &&
        presenter.guest_target_framebuffer != VK_NULL_HANDLE &&
        presenter.guest_target_extent.width ==
            presenter.swapchain_extent.width &&
        presenter.guest_target_extent.height ==
            presenter.swapchain_extent.height) {
        return true;
    }
    if (!CreateGuestFixedResources(presenter)) {
        return false;
    }
    const VkAttachmentDescription attachment{
        0,
        presenter.swapchain_format,
        VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_LOAD,
        VK_ATTACHMENT_STORE_OP_STORE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference color_reference{
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkSubpassDescription subpass{
        0,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        0,
        nullptr,
        1,
        &color_reference,
        nullptr,
        nullptr,
        0,
        nullptr};
    const VkSubpassDependency dependency{
        VK_SUBPASS_EXTERNAL,
        0,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        0};
    const VkRenderPassCreateInfo render_pass_info{
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        nullptr,
        0,
        1,
        &attachment,
        1,
        &subpass,
        1,
        &dependency};
    auto result = vkCreateRenderPass(
        presenter.device, &render_pass_info, nullptr,
        &presenter.guest_render_pass);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateRenderPass(guest)", result);
        return false;
    }

    const std::array stages{
        VkPipelineShaderStageCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            presenter.guest_vertex_shader,
            "main",
            nullptr},
        VkPipelineShaderStageCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            presenter.guest_fragment_shader,
            "main",
            nullptr}};
    const VkVertexInputBindingDescription vertex_binding{
        0, sizeof(VulkanGuestVertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array vertex_attributes{
        VkVertexInputAttributeDescription{
            0, 0, VK_FORMAT_R32G32_SFLOAT,
            static_cast<std::uint32_t>(
                offsetof(VulkanGuestVertex, x))},
        VkVertexInputAttributeDescription{
            1, 0, VK_FORMAT_R32G32_SFLOAT,
            static_cast<std::uint32_t>(
                offsetof(VulkanGuestVertex, u))},
        VkVertexInputAttributeDescription{
            2, 0, VK_FORMAT_R8G8B8A8_UNORM,
            static_cast<std::uint32_t>(
                offsetof(VulkanGuestVertex, color))}};
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        nullptr,
        0,
        1,
        &vertex_binding,
        static_cast<std::uint32_t>(vertex_attributes.size()),
        vertex_attributes.data()};
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_FALSE};
    const VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        nullptr,
        0,
        1,
        nullptr,
        1,
        nullptr};
    const VkPipelineRasterizationStateCreateInfo rasterization{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_FALSE,
        VK_FALSE,
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        VK_FRONT_FACE_COUNTER_CLOCKWISE,
        VK_FALSE,
        0.0f,
        0.0f,
        0.0f,
        1.0f};
    const VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_SAMPLE_COUNT_1_BIT,
        VK_FALSE,
        0.0f,
        nullptr,
        VK_FALSE,
        VK_FALSE};
    const VkPipelineColorBlendAttachmentState blend_attachment{
        VK_TRUE,
        VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        VK_BLEND_FACTOR_SRC_ALPHA,
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VK_BLEND_OP_ADD,
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        nullptr,
        0,
        VK_FALSE,
        VK_LOGIC_OP_COPY,
        1,
        &blend_attachment,
        {0.0f, 0.0f, 0.0f, 0.0f}};
    constexpr std::array dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        nullptr,
        0,
        static_cast<std::uint32_t>(dynamic_states.size()),
        dynamic_states.data()};
    const VkGraphicsPipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        static_cast<std::uint32_t>(stages.size()),
        stages.data(),
        &vertex_input,
        &assembly,
        nullptr,
        &viewport_state,
        &rasterization,
        &multisample,
        nullptr,
        &blend,
        &dynamic,
        presenter.guest_pipeline_layout,
        presenter.guest_render_pass,
        0,
        VK_NULL_HANDLE,
        -1};
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &pipeline_info, nullptr, &presenter.guest_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateGraphicsPipelines(guest)", result);
        return false;
    }
    auto premultiplied_attachment = blend_attachment;
    premultiplied_attachment.srcColorBlendFactor =
        VK_BLEND_FACTOR_ONE;
    auto premultiplied_blend = blend;
    premultiplied_blend.pAttachments =
        &premultiplied_attachment;
    auto premultiplied_pipeline_info = pipeline_info;
    premultiplied_pipeline_info.pColorBlendState =
        &premultiplied_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &premultiplied_pipeline_info, nullptr,
        &presenter.guest_premultiplied_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure(
            "vkCreateGraphicsPipelines(guest premultiplied)",
            result);
        return false;
    }
    auto destination_source_alpha_attachment = blend_attachment;
    destination_source_alpha_attachment.srcColorBlendFactor =
        VK_BLEND_FACTOR_ZERO;
    destination_source_alpha_attachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_SRC_ALPHA;
    destination_source_alpha_attachment.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ZERO;
    destination_source_alpha_attachment.dstAlphaBlendFactor =
        VK_BLEND_FACTOR_SRC_ALPHA;
    auto destination_source_alpha_blend = blend;
    destination_source_alpha_blend.pAttachments =
        &destination_source_alpha_attachment;
    auto destination_source_alpha_pipeline_info = pipeline_info;
    destination_source_alpha_pipeline_info.pColorBlendState =
        &destination_source_alpha_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &destination_source_alpha_pipeline_info, nullptr,
        &presenter.guest_destination_source_alpha_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure(
            "vkCreateGraphicsPipelines(guest destination source alpha)",
            result);
        return false;
    }
    auto destination_inverse_source_alpha_attachment = blend_attachment;
    destination_inverse_source_alpha_attachment.srcColorBlendFactor =
        VK_BLEND_FACTOR_ZERO;
    destination_inverse_source_alpha_attachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    destination_inverse_source_alpha_attachment.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ZERO;
    destination_inverse_source_alpha_attachment.dstAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    auto destination_inverse_source_alpha_blend = blend;
    destination_inverse_source_alpha_blend.pAttachments =
        &destination_inverse_source_alpha_attachment;
    auto destination_inverse_source_alpha_pipeline_info = pipeline_info;
    destination_inverse_source_alpha_pipeline_info.pColorBlendState =
        &destination_inverse_source_alpha_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &destination_inverse_source_alpha_pipeline_info, nullptr,
        &presenter.guest_destination_inverse_source_alpha_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure(
            "vkCreateGraphicsPipelines("
            "guest destination inverse source alpha)",
            result);
        return false;
    }
    auto additive_attachment = blend_attachment;
    additive_attachment.srcColorBlendFactor =
        VK_BLEND_FACTOR_ONE;
    additive_attachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE;
    additive_attachment.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE;
    additive_attachment.dstAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE;
    auto additive_blend = blend;
    additive_blend.pAttachments = &additive_attachment;
    auto additive_pipeline_info = pipeline_info;
    additive_pipeline_info.pColorBlendState = &additive_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &additive_pipeline_info, nullptr,
        &presenter.guest_additive_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure(
            "vkCreateGraphicsPipelines(guest additive)",
            result);
        return false;
    }
    auto wave_stages = stages;
    wave_stages[1].module =
        presenter.guest_wave_fragment_shader;
    auto wave_pipeline_info = pipeline_info;
    wave_pipeline_info.pStages = wave_stages.data();
    wave_pipeline_info.pColorBlendState =
        &premultiplied_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &wave_pipeline_info, nullptr,
        &presenter.guest_wave_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure(
            "vkCreateGraphicsPipelines(guest wave)", result);
        return false;
    }
    auto opaque_attachment = blend_attachment;
    opaque_attachment.blendEnable = VK_FALSE;
    auto opaque_blend = blend;
    opaque_blend.pAttachments = &opaque_attachment;
    auto opaque_pipeline_info = pipeline_info;
    opaque_pipeline_info.pColorBlendState = &opaque_blend;
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &opaque_pipeline_info, nullptr,
        &presenter.guest_opaque_pipeline);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateGraphicsPipelines(guest opaque)", result);
        return false;
    }
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
        0,
        VK_IMAGE_TYPE_2D,
        presenter.swapchain_format,
        {presenter.swapchain_extent.width,
         presenter.swapchain_extent.height, 1},
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED};
    result = vkCreateImage(
        presenter.device, &image_info, nullptr,
        &presenter.guest_target_image);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateImage(guest target)", result);
        return false;
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(
        presenter.device, presenter.guest_target_image,
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
        &presenter.guest_target_memory);
    if (result == VK_SUCCESS) {
        result = vkBindImageMemory(
            presenter.device, presenter.guest_target_image,
            presenter.guest_target_memory, 0);
    }
    if (result != VK_SUCCESS) {
        LogFailure("guest target memory", result);
        return false;
    }
    const VkImageViewCreateInfo view_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,
        0,
        presenter.guest_target_image,
        VK_IMAGE_VIEW_TYPE_2D,
        presenter.swapchain_format,
        {VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    result = vkCreateImageView(
        presenter.device, &view_info, nullptr,
        &presenter.guest_target_view);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateImageView(guest target)", result);
        return false;
    }
    const VkFramebufferCreateInfo framebuffer_info{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        nullptr,
        0,
        presenter.guest_render_pass,
        1,
        &presenter.guest_target_view,
        presenter.swapchain_extent.width,
        presenter.swapchain_extent.height,
        1};
    result = vkCreateFramebuffer(
        presenter.device, &framebuffer_info, nullptr,
        &presenter.guest_target_framebuffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateFramebuffer(guest target)", result);
        return false;
    }
    presenter.guest_target_extent = presenter.swapchain_extent;
    presenter.guest_target_initialized = false;
    presenter.last_guest_batch_id = 0;
    presenter.last_guest_target_key = 0;
    return true;
}

GuestTargetResource* EnsureGuestTarget(
    Presenter& presenter,
    const std::uint64_t key) {
    const auto target_key = key != 0u ? key : UINT64_MAX;
    auto [iterator, inserted] =
        presenter.guest_targets.try_emplace(target_key);
    auto& target = iterator->second;
    if (!inserted && target.image != VK_NULL_HANDLE &&
        target.extent.width == presenter.swapchain_extent.width &&
        target.extent.height == presenter.swapchain_extent.height) {
        target.last_use = presenter.guest_frame_serial;
        return &target;
    }
    if (!inserted) {
        if (target.framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(
                presenter.device, target.framebuffer, nullptr);
        }
        if (target.view != VK_NULL_HANDLE) {
            vkDestroyImageView(
                presenter.device, target.view, nullptr);
        }
        if (target.image != VK_NULL_HANDLE) {
            vkDestroyImage(
                presenter.device, target.image, nullptr);
        }
        if (target.memory != VK_NULL_HANDLE) {
            vkFreeMemory(
                presenter.device, target.memory, nullptr);
        }
        target = {};
    }
    const VkImageCreateInfo image_info{
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
        0,
        VK_IMAGE_TYPE_2D,
        presenter.swapchain_format,
        {presenter.swapchain_extent.width,
         presenter.swapchain_extent.height, 1},
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED};
    auto result = vkCreateImage(
        presenter.device, &image_info, nullptr, &target.image);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateImage(guest keyed target)", result);
        return nullptr;
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(
        presenter.device, target.image, &requirements);
    const auto memory_type = FindMemoryType(
        presenter, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
        return nullptr;
    }
    const VkMemoryAllocateInfo allocation{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        requirements.size,
        memory_type};
    result = vkAllocateMemory(
        presenter.device, &allocation, nullptr, &target.memory);
    if (result == VK_SUCCESS) {
        result = vkBindImageMemory(
            presenter.device, target.image, target.memory, 0);
    }
    if (result != VK_SUCCESS) {
        LogFailure("guest keyed target memory", result);
        return nullptr;
    }
    const VkImageViewCreateInfo view_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        nullptr,
        0,
        target.image,
        VK_IMAGE_VIEW_TYPE_2D,
        presenter.swapchain_format,
        {VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY,
         VK_COMPONENT_SWIZZLE_IDENTITY},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    result = vkCreateImageView(
        presenter.device, &view_info, nullptr, &target.view);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateImageView(guest keyed target)", result);
        return nullptr;
    }
    const VkFramebufferCreateInfo framebuffer_info{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        nullptr,
        0,
        presenter.guest_render_pass,
        1,
        &target.view,
        presenter.swapchain_extent.width,
        presenter.swapchain_extent.height,
        1};
    result = vkCreateFramebuffer(
        presenter.device, &framebuffer_info, nullptr,
        &target.framebuffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkCreateFramebuffer(guest keyed target)", result);
        return nullptr;
    }
    const std::array descriptor_layouts{
        presenter.guest_descriptor_layout,
        presenter.guest_descriptor_layout};
    const VkDescriptorSetAllocateInfo descriptor_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr,
        presenter.guest_descriptor_pool,
        static_cast<std::uint32_t>(descriptor_layouts.size()),
        descriptor_layouts.data()};
    std::array<VkDescriptorSet, 2> descriptors{};
    result = vkAllocateDescriptorSets(
        presenter.device, &descriptor_info, descriptors.data());
    if (result != VK_SUCCESS) {
        LogFailure("vkAllocateDescriptorSets(guest target)", result);
        return nullptr;
    }
    target.descriptor = descriptors[0];
    target.nearest_descriptor = descriptors[1];
    const std::array image_descriptors{
        VkDescriptorImageInfo{
            presenter.guest_sampler,
            target.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkDescriptorImageInfo{
            presenter.guest_nearest_sampler,
            target.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    const std::array writes{
        VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            target.descriptor,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            &image_descriptors[0],
            nullptr,
            nullptr},
        VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            target.nearest_descriptor,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            &image_descriptors[1],
            nullptr,
            nullptr}};
    vkUpdateDescriptorSets(
        presenter.device,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);
    target.extent = presenter.swapchain_extent;
    target.initialized = false;
    target.last_use = presenter.guest_frame_serial;
    return &target;
}

GuestTextureResource* EnsureGuestTexture(
    Presenter& presenter,
    const VulkanGuestDraw& draw,
    bool& needs_upload) {
    needs_upload = false;
    if (draw.texture_rgba == nullptr ||
        draw.texture_width == 0 || draw.texture_height == 0) {
        return nullptr;
    }
    auto [iterator, inserted] = presenter.guest_textures.try_emplace(
        draw.texture_key);
    auto& texture = iterator->second;
    if (!inserted &&
        (texture.width != draw.texture_width ||
         texture.height != draw.texture_height)) {
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(presenter.device, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE) {
            vkDestroyImage(presenter.device, texture.image, nullptr);
        }
        if (texture.memory != VK_NULL_HANDLE) {
            vkFreeMemory(presenter.device, texture.memory, nullptr);
        }
        const auto descriptor = texture.descriptor;
        texture = {};
        texture.descriptor = descriptor;
    }
    if (texture.image == VK_NULL_HANDLE) {
        const VkImageCreateInfo image_info{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr,
            0,
            VK_IMAGE_TYPE_2D,
            VK_FORMAT_R8G8B8A8_UNORM,
            {draw.texture_width, draw.texture_height, 1},
            1,
            1,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_SHARING_MODE_EXCLUSIVE,
            0,
            nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED};
        auto result = vkCreateImage(
            presenter.device, &image_info, nullptr, &texture.image);
        if (result != VK_SUCCESS) {
            LogFailure("vkCreateImage(guest texture)", result);
            return nullptr;
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(
            presenter.device, texture.image, &requirements);
        const auto memory_type = FindMemoryType(
            presenter, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memory_type == std::numeric_limits<std::uint32_t>::max()) {
            return nullptr;
        }
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr,
            requirements.size,
            memory_type};
        result = vkAllocateMemory(
            presenter.device, &allocation, nullptr, &texture.memory);
        if (result == VK_SUCCESS) {
            result = vkBindImageMemory(
                presenter.device, texture.image, texture.memory, 0);
        }
        if (result != VK_SUCCESS) {
            LogFailure("guest texture memory", result);
            return nullptr;
        }
        const VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr,
            0,
            texture.image,
            VK_IMAGE_VIEW_TYPE_2D,
            VK_FORMAT_R8G8B8A8_UNORM,
            {VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        result = vkCreateImageView(
            presenter.device, &view_info, nullptr, &texture.view);
        if (result != VK_SUCCESS) {
            LogFailure("vkCreateImageView(guest texture)", result);
            return nullptr;
        }
        if (texture.descriptor == VK_NULL_HANDLE) {
            const VkDescriptorSetAllocateInfo descriptor_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                nullptr,
                presenter.guest_descriptor_pool,
                1,
                &presenter.guest_descriptor_layout};
            result = vkAllocateDescriptorSets(
                presenter.device, &descriptor_info,
                &texture.descriptor);
            if (result != VK_SUCCESS) {
                LogFailure("vkAllocateDescriptorSets(guest)", result);
                return nullptr;
            }
        }
        const VkDescriptorImageInfo image_descriptor{
            draw.nearest_texture
                ? draw.repeat_texture
                    ? presenter.guest_nearest_repeat_sampler
                    : presenter.guest_nearest_sampler
                : draw.repeat_texture
                    ? presenter.guest_repeat_sampler
                    : presenter.guest_sampler,
            texture.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            texture.descriptor,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            &image_descriptor,
            nullptr,
            nullptr};
        vkUpdateDescriptorSets(
            presenter.device, 1, &write, 0, nullptr);
        texture.key = draw.texture_key;
        texture.width = draw.texture_width;
        texture.height = draw.texture_height;
        texture.repeat_texture = draw.repeat_texture;
        texture.nearest_texture = draw.nearest_texture;
        texture.initialized = false;
    } else if (texture.repeat_texture != draw.repeat_texture ||
               texture.nearest_texture != draw.nearest_texture) {
        const VkDescriptorImageInfo image_descriptor{
            draw.nearest_texture
                ? draw.repeat_texture
                    ? presenter.guest_nearest_repeat_sampler
                    : presenter.guest_nearest_sampler
                : draw.repeat_texture
                    ? presenter.guest_repeat_sampler
                    : presenter.guest_sampler,
            texture.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkWriteDescriptorSet write{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            texture.descriptor,
            0,
            0,
            1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            &image_descriptor,
            nullptr,
            nullptr};
        vkUpdateDescriptorSets(
            presenter.device, 1, &write, 0, nullptr);
        texture.repeat_texture = draw.repeat_texture;
        texture.nearest_texture = draw.nearest_texture;
    }
    needs_upload =
        texture.signature != draw.texture_signature ||
        !texture.initialized;
    texture.signature = draw.texture_signature;
    texture.last_use = presenter.guest_frame_serial;
    return &texture;
}

struct GuestDrawUpload {
    const VulkanGuestDraw* draw{};
    GuestTextureResource* texture{};
    GuestTargetResource* source_target{};
    VkDeviceSize vertex_offset{};
    VkDeviceSize index_offset{};
    VkDeviceSize texture_offset{};
    bool upload_texture{};
};

struct GuestPassUpload {
    const VulkanGuestDraw* draw{};
    GuestTargetResource* target{};
    GuestTargetResource* source_target{};
    GuestTextureResource* texture{};
    VkDeviceSize vertex_offset{};
    VkDeviceSize index_offset{};
    VkDeviceSize texture_offset{};
    std::size_t pass_index{};
    bool upload_texture{};
};

bool SubmitGuestPasses(Presenter& presenter,
                       const VulkanGuestFrame& frame) {
    if (presenter.submission_in_flight) {
        const auto fence_status = vkGetFenceStatus(
            presenter.device, presenter.fence);
        if (fence_status == VK_NOT_READY) {
            return true;
        }
        if (fence_status != VK_SUCCESS) {
            LogFailure("vkGetFenceStatus(guest passes)", fence_status);
            return false;
        }
        presenter.submission_in_flight = false;
    }
    if (!EnsureSwapchain(presenter, frame.width, frame.height) ||
        !CreateGuestSwapchainResources(presenter)) {
        return false;
    }
    ++presenter.guest_frame_serial;
    auto* final_target =
        EnsureGuestTarget(presenter, frame.target_key);
    if (final_target == nullptr) {
        return false;
    }
    std::vector<GuestTargetResource*> pass_targets(
        frame.passes.size(), nullptr);
    for (std::size_t pass_index = 0;
         pass_index < frame.passes.size(); ++pass_index) {
        pass_targets[pass_index] = EnsureGuestTarget(
            presenter, frame.passes[pass_index].target_key);
        if (pass_targets[pass_index] == nullptr) {
            return false;
        }
    }
    std::vector<GuestPassUpload> uploads;
    VkDeviceSize staging_size{};
    for (std::size_t pass_index = 0;
         pass_index < frame.passes.size(); ++pass_index) {
        const auto& pass = frame.passes[pass_index];
        for (const auto& draw : pass.draws) {
            if (draw.vertices.empty() || draw.indices.empty()) {
                continue;
            }
            GuestPassUpload upload{};
            upload.draw = &draw;
            upload.target = pass_targets[pass_index];
            upload.pass_index = pass_index;
            upload.vertex_offset =
                AlignGuestOffset(staging_size, 16u);
            staging_size = upload.vertex_offset +
                draw.vertices.size() * sizeof(VulkanGuestVertex);
            upload.index_offset =
                AlignGuestOffset(staging_size, 4u);
            staging_size = upload.index_offset +
                draw.indices.size() * sizeof(std::uint32_t);
            const auto source =
                presenter.guest_targets.find(draw.texture_address);
            if (source != presenter.guest_targets.end() &&
                &source->second != upload.target &&
                (draw.require_target_source ||
                 source->second.initialized)) {
                upload.source_target = &source->second;
            } else if (draw.require_target_source) {
                continue;
            } else {
                bool needs_upload{};
                upload.texture =
                    EnsureGuestTexture(presenter, draw, needs_upload);
                if (upload.texture == nullptr) {
                    continue;
                }
                upload.upload_texture = needs_upload;
                if (needs_upload) {
                    upload.texture_offset =
                        AlignGuestOffset(staging_size, 4u);
                    staging_size = upload.texture_offset +
                        draw.texture_rgba->size();
                }
            }
            uploads.push_back(upload);
        }
    }
    if (!EnsureStaging(
            presenter, std::max<VkDeviceSize>(staging_size, 4u))) {
        return false;
    }
    auto* const staging =
        static_cast<std::uint8_t*>(presenter.staging_map);
    for (const auto& upload : uploads) {
        std::memcpy(
            staging + upload.vertex_offset,
            upload.draw->vertices.data(),
            upload.draw->vertices.size() *
                sizeof(VulkanGuestVertex));
        std::memcpy(
            staging + upload.index_offset,
            upload.draw->indices.data(),
            upload.draw->indices.size() *
                sizeof(std::uint32_t));
        if (upload.upload_texture) {
            std::memcpy(
                staging + upload.texture_offset,
                upload.draw->texture_rgba->data(),
                upload.draw->texture_rgba->size());
        }
    }

    std::uint32_t image_index{};
    auto result = vkAcquireNextImageKHR(
        presenter.device, presenter.swapchain, 0u,
        presenter.acquired, VK_NULL_HANDLE, &image_index);
    if (result == VK_TIMEOUT || result == VK_NOT_READY) {
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LogFailure("vkAcquireNextImageKHR(guest passes)", result);
        return false;
    }
    (void)vkResetFences(presenter.device, 1, &presenter.fence);
    (void)vkResetCommandBuffer(presenter.command_buffer, 0);
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    result = vkBeginCommandBuffer(presenter.command_buffer, &begin);
    if (result != VK_SUCCESS) {
        LogFailure("vkBeginCommandBuffer(guest passes)", result);
        return false;
    }
    for (const auto& upload : uploads) {
        if (!upload.upload_texture) {
            continue;
        }
        ImageBarrier(
            presenter.command_buffer, upload.texture->image,
            upload.texture->initialized
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            upload.texture->initialized
                ? VK_ACCESS_SHADER_READ_BIT : 0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            upload.texture->initialized
                ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            upload.texture_offset, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0},
            {upload.draw->texture_width,
             upload.draw->texture_height, 1}};
        vkCmdCopyBufferToImage(
            presenter.command_buffer, presenter.staging,
            upload.texture->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        ImageBarrier(
            presenter.command_buffer, upload.texture->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        upload.texture->initialized = true;
    }

    std::vector<GuestTargetResource*> targets_seen_this_frame;
    for (std::size_t pass_index = 0;
         pass_index < frame.passes.size(); ++pass_index) {
        const auto& pass = frame.passes[pass_index];
        auto* const target = pass_targets[pass_index];
        const bool first_target_pass =
            std::ranges::find(
                targets_seen_this_frame, target) ==
            targets_seen_this_frame.end();
        if (first_target_pass) {
            targets_seen_this_frame.push_back(target);
        }
        const bool clear_target =
            pass.clear_target || !target->initialized ||
            first_target_pass;
        if (clear_target) {
            ImageBarrier(
                presenter.command_buffer, target->image,
                target->initialized
                    ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                target->initialized
                    ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                target->initialized
                    ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                    : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);
            const auto rgba =
                pass.clear_target
                    ? pass.clear_rgba
                    : UINT32_C(0x00000000);
            const VkClearColorValue clear{{
                static_cast<float>(rgba & 0xffu) / 255.0f,
                static_cast<float>((rgba >> 8u) & 0xffu) / 255.0f,
                static_cast<float>((rgba >> 16u) & 0xffu) / 255.0f,
                static_cast<float>((rgba >> 24u) & 0xffu) / 255.0f}};
            const VkImageSubresourceRange range{
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(
                presenter.command_buffer, target->image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &clear, 1, &range);
            ImageBarrier(
                presenter.command_buffer, target->image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        }
        std::vector<GuestTargetResource*> sampled_targets;
        for (const auto& upload : uploads) {
            if (upload.pass_index != pass_index ||
                upload.source_target == nullptr ||
                std::ranges::find(
                    sampled_targets, upload.source_target) !=
                    sampled_targets.end()) {
                continue;
            }
            sampled_targets.push_back(upload.source_target);
            ImageBarrier(
                presenter.command_buffer,
                upload.source_target->image,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }
        const VkRenderPassBeginInfo render_begin{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr, presenter.guest_render_pass,
            target->framebuffer, {{0, 0}, target->extent},
            0, nullptr};
        vkCmdBeginRenderPass(
            presenter.command_buffer, &render_begin,
            VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{
            0.0f, 0.0f,
            static_cast<float>(target->extent.width),
            static_cast<float>(target->extent.height),
            0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, target->extent};
        vkCmdSetViewport(
            presenter.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(
            presenter.command_buffer, 0, 1, &scissor);
        const std::array<float, 4> transform{
            2.0f / static_cast<float>(frame.width),
            2.0f / static_cast<float>(frame.height),
            -1.0f, -1.0f};
        vkCmdPushConstants(
            presenter.command_buffer,
            presenter.guest_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT, 0,
            sizeof(transform), transform.data());
        for (const auto& upload : uploads) {
            if (upload.pass_index != pass_index) {
                continue;
            }
            vkCmdBindPipeline(
                presenter.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                upload.draw->opaque
                    ? presenter.guest_opaque_pipeline
                    : upload.draw->destination_source_alpha
                    ? presenter.guest_destination_source_alpha_pipeline
                    : upload.draw->destination_inverse_source_alpha
                    ? presenter
                          .guest_destination_inverse_source_alpha_pipeline
                    : upload.draw->additive
                    ? presenter.guest_additive_pipeline
                    : upload.draw->wave_effect
                    ? presenter.guest_wave_pipeline
                    : upload.draw->premultiplied_alpha
                    ? presenter.guest_premultiplied_pipeline
                    : presenter.guest_pipeline);
            if (upload.draw->wave_effect) {
                vkCmdPushConstants(
                    presenter.command_buffer,
                    presenter.guest_pipeline_layout,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(float) * 4u,
                    sizeof(upload.draw->wave_parameters),
                    upload.draw->wave_parameters.data());
            }
            const auto descriptor =
                upload.source_target != nullptr
                    ? upload.draw->nearest_texture
                        ? upload.source_target->nearest_descriptor
                        : upload.source_target->descriptor
                    : upload.texture->descriptor;
            vkCmdBindDescriptorSets(
                presenter.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                presenter.guest_pipeline_layout,
                0, 1, &descriptor, 0, nullptr);
            vkCmdBindVertexBuffers(
                presenter.command_buffer, 0, 1,
                &presenter.staging, &upload.vertex_offset);
            vkCmdBindIndexBuffer(
                presenter.command_buffer, presenter.staging,
                upload.index_offset, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(
                presenter.command_buffer,
                static_cast<std::uint32_t>(
                    upload.draw->indices.size()),
                std::max(upload.draw->instance_count, 1u),
                0, 0, 0);
        }
        vkCmdEndRenderPass(presenter.command_buffer);
        for (auto* const sampled_target : sampled_targets) {
            ImageBarrier(
                presenter.command_buffer, sampled_target->image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        }
        target->initialized = true;
        target->last_use = presenter.guest_frame_serial;
    }
    if (!final_target->initialized) {
        return false;
    }
    ImageBarrier(
        presenter.command_buffer, final_target->image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkImageBlit blit{
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0},
         {static_cast<std::int32_t>(final_target->extent.width),
          static_cast<std::int32_t>(final_target->extent.height), 1}},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0},
         {static_cast<std::int32_t>(
              presenter.swapchain_extent.width),
          static_cast<std::int32_t>(
              presenter.swapchain_extent.height), 1}}};
    vkCmdBlitImage(
        presenter.command_buffer, final_target->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_NEAREST);
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    ImageBarrier(
        presenter.command_buffer, final_target->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    result = vkEndCommandBuffer(presenter.command_buffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkEndCommandBuffer(guest passes)", result);
        return false;
    }
    constexpr VkPipelineStageFlags WaitStage =
        VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
        1, &presenter.acquired, &WaitStage,
        1, &presenter.command_buffer,
        1, &presenter.rendered};
    result = vkQueueSubmit(
        presenter.queue, 1, &submit, presenter.fence);
    if (result != VK_SUCCESS) {
        LogFailure("vkQueueSubmit(guest passes)", result);
        return false;
    }
    presenter.submission_in_flight = true;
    presenter.last_guest_batch_id = frame.batch_id;
    presenter.last_guest_target_key = frame.target_key;
    const VkPresentInfoKHR present{
        VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr,
        1, &presenter.rendered, 1, &presenter.swapchain,
        &image_index, nullptr};
    result = vkQueuePresentKHR(presenter.queue, &present);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LogFailure("vkQueuePresentKHR(guest passes)", result);
        return false;
    }
    g_present_count.fetch_add(1u, std::memory_order_relaxed);
    return true;
}

bool SubmitGuestFrame(Presenter& presenter,
                      const VulkanGuestFrame& frame) {
    if (!frame.passes.empty()) {
        return SubmitGuestPasses(presenter, frame);
    }
    if (presenter.submission_in_flight) {
        const auto fence_status = vkGetFenceStatus(
            presenter.device, presenter.fence);
        if (fence_status == VK_NOT_READY) {
            return true;
        }
        if (fence_status != VK_SUCCESS) {
            LogFailure("vkGetFenceStatus(guest)", fence_status);
            return false;
        }
        presenter.submission_in_flight = false;
    }
    if (!EnsureSwapchain(presenter, frame.width, frame.height) ||
        !CreateGuestSwapchainResources(presenter)) {
        return false;
    }
    ++presenter.guest_frame_serial;
    auto* const target =
        EnsureGuestTarget(presenter, frame.target_key);
    if (target == nullptr) {
        return false;
    }
    const bool render_batch =
        frame.batch_id == 0 ||
        frame.batch_id != presenter.last_guest_batch_id;
    const bool legacy_single_pass =
        std::ranges::none_of(
            frame.draws,
            [](const VulkanGuestDraw& draw) {
                return draw.extended_blend_contract;
            });
    const bool preserve_target =
        legacy_single_pass
        ? render_batch &&
            target->initialized &&
            frame.preserve_target &&
            frame.base_batch_id != 0u &&
            target->last_batch_id == frame.base_batch_id &&
            frame.first_new_draw <= frame.draws.size()
        : render_batch &&
            target->initialized &&
            !frame.clear_target &&
            ((frame.preserve_target &&
              frame.base_batch_id != 0u &&
              target->last_batch_id == frame.base_batch_id &&
              frame.first_new_draw <= frame.draws.size()) ||
             (!frame.preserve_target && !frame.clear_target));
    if (!legacy_single_pass &&
        render_batch && frame.preserve_target &&
        !frame.clear_target && !preserve_target) {
        // This is a delta against a render-target version that is no longer
        // resident. Replaying its unbounded draw history produces a stale
        // fullscreen effect; wait for the next explicit clear/full frame.
        return true;
    }
    const auto first_draw =
        preserve_target ? frame.first_new_draw : 0u;
    std::vector<GuestDrawUpload> uploads;
    if (render_batch) {
        uploads.reserve(frame.draws.size());
    }
    VkDeviceSize staging_size{};
    if (render_batch) {
        for (auto draw_index = first_draw;
             draw_index < frame.draws.size(); ++draw_index) {
            const auto& draw = frame.draws[draw_index];
            if (draw.vertices.empty() || draw.indices.empty()) {
                continue;
            }
            GuestDrawUpload upload{};
            upload.draw = &draw;
            upload.vertex_offset =
                AlignGuestOffset(staging_size, 16u);
            staging_size = upload.vertex_offset +
                draw.vertices.size() * sizeof(VulkanGuestVertex);
            upload.index_offset =
                AlignGuestOffset(staging_size, 4u);
            staging_size = upload.index_offset +
                draw.indices.size() * sizeof(std::uint32_t);
            const auto source =
                presenter.guest_targets.find(draw.texture_address);
            if (source != presenter.guest_targets.end() &&
                source->second.initialized &&
                &source->second != target) {
                upload.source_target = &source->second;
            } else {
                if (draw.require_target_source) {
                    continue;
                }
                bool needs_upload{};
                upload.texture =
                    EnsureGuestTexture(presenter, draw, needs_upload);
                if (upload.texture == nullptr) {
                    continue;
                }
                upload.upload_texture = needs_upload;
                if (needs_upload) {
                    upload.texture_offset =
                        AlignGuestOffset(staging_size, 4u);
                    staging_size = upload.texture_offset +
                        draw.texture_rgba->size();
                }
            }
            uploads.push_back(upload);
        }
    }
    if (render_batch && uploads.empty()) {
        return false;
    }
    if (!legacy_single_pass &&
        render_batch && !target->initialized &&
        !frame.clear_target &&
        !uploads.empty() &&
        std::ranges::none_of(
            uploads,
            [](const GuestDrawUpload& upload) {
                return upload.draw != nullptr &&
                    upload.draw->opaque;
            })) {
        // A render target is persistent until an explicit clear. A
        // blend-only command buffer without its base is an incomplete
        // continuation (typically bloom/UI), not a presentable frame.
        return true;
    }
    if (render_batch &&
        !EnsureStaging(presenter, std::max<VkDeviceSize>(
            staging_size, 4u))) {
        return false;
    }
    if (render_batch) {
        auto* const staging =
            static_cast<std::uint8_t*>(presenter.staging_map);
        for (const auto& upload : uploads) {
            std::memcpy(
                staging + upload.vertex_offset,
                upload.draw->vertices.data(),
                upload.draw->vertices.size() *
                    sizeof(VulkanGuestVertex));
            std::memcpy(
                staging + upload.index_offset,
                upload.draw->indices.data(),
                upload.draw->indices.size() *
                    sizeof(std::uint32_t));
            if (upload.upload_texture) {
                std::memcpy(
                    staging + upload.texture_offset,
                    upload.draw->texture_rgba->data(),
                    upload.draw->texture_rgba->size());
            }
        }
    }

    std::uint32_t image_index{};
    auto result = vkAcquireNextImageKHR(
        presenter.device, presenter.swapchain, 0u,
        presenter.acquired, VK_NULL_HANDLE, &image_index);
    if (result == VK_TIMEOUT || result == VK_NOT_READY) {
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        LogFailure("vkAcquireNextImageKHR(guest)", result);
        return false;
    }
    (void)vkResetFences(presenter.device, 1, &presenter.fence);
    (void)vkResetCommandBuffer(presenter.command_buffer, 0);
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    result = vkBeginCommandBuffer(presenter.command_buffer, &begin);
    if (result != VK_SUCCESS) {
        LogFailure("vkBeginCommandBuffer(guest)", result);
        return false;
    }
    for (const auto& upload : uploads) {
        if (!upload.upload_texture) {
            continue;
        }
        ImageBarrier(
            presenter.command_buffer,
            upload.texture->image,
            upload.texture->initialized
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            upload.texture->initialized
                ? VK_ACCESS_SHADER_READ_BIT : 0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            upload.texture->initialized
                ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            upload.texture_offset,
            0,
            0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            {0, 0, 0},
            {upload.draw->texture_width,
             upload.draw->texture_height, 1}};
        vkCmdCopyBufferToImage(
            presenter.command_buffer,
            presenter.staging,
            upload.texture->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy);
        ImageBarrier(
            presenter.command_buffer,
            upload.texture->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        upload.texture->initialized = true;
    }
    std::vector<GuestTargetResource*> sampled_targets;
    for (const auto& upload : uploads) {
        if (upload.source_target == nullptr ||
            std::ranges::find(
                sampled_targets, upload.source_target) !=
                sampled_targets.end()) {
            continue;
        }
        sampled_targets.push_back(upload.source_target);
        ImageBarrier(
            presenter.command_buffer,
            upload.source_target->image,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    if (render_batch && !preserve_target) {
        ImageBarrier(
            presenter.command_buffer,
            target->image,
            target->initialized
                ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            target->initialized
                ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            target->initialized
                ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearColorValue clear{{
            static_cast<float>(frame.clear_rgba & 0xffu) / 255.0f,
            static_cast<float>(
                (frame.clear_rgba >> 8u) & 0xffu) / 255.0f,
            static_cast<float>(
                (frame.clear_rgba >> 16u) & 0xffu) / 255.0f,
            static_cast<float>(
                (frame.clear_rgba >> 24u) & 0xffu) / 255.0f}};
        const VkImageSubresourceRange range{
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(
            presenter.command_buffer,
            target->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &clear, 1, &range);
        ImageBarrier(
            presenter.command_buffer,
            target->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    if (render_batch) {
        const VkRenderPassBeginInfo render_begin{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            nullptr,
            presenter.guest_render_pass,
            target->framebuffer,
            {{0, 0}, target->extent},
            0,
            nullptr};
        vkCmdBeginRenderPass(
            presenter.command_buffer, &render_begin,
            VK_SUBPASS_CONTENTS_INLINE);
        const VkViewport viewport{
            0.0f,
            0.0f,
            static_cast<float>(target->extent.width),
            static_cast<float>(target->extent.height),
            0.0f,
            1.0f};
        const VkRect2D scissor{
            {0, 0}, target->extent};
        vkCmdSetViewport(
            presenter.command_buffer, 0, 1, &viewport);
        vkCmdSetScissor(
            presenter.command_buffer, 0, 1, &scissor);
        const std::array<float, 4> transform{
            2.0f / static_cast<float>(frame.width),
            2.0f / static_cast<float>(frame.height),
            -1.0f,
            -1.0f};
        vkCmdPushConstants(
            presenter.command_buffer,
            presenter.guest_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(transform),
            transform.data());
        for (const auto& upload : uploads) {
            vkCmdBindPipeline(
                presenter.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                !upload.draw->extended_blend_contract
                    ? presenter.guest_pipeline
                    : upload.draw->opaque
                    ? presenter.guest_opaque_pipeline
                    : upload.draw->destination_source_alpha
                    ? presenter.guest_destination_source_alpha_pipeline
                    : upload.draw->destination_inverse_source_alpha
                    ? presenter
                          .guest_destination_inverse_source_alpha_pipeline
                    : upload.draw->additive
                    ? presenter.guest_additive_pipeline
                    : upload.draw->wave_effect
                    ? presenter.guest_wave_pipeline
                    : upload.draw->premultiplied_alpha
                    ? presenter.guest_premultiplied_pipeline
                    : presenter.guest_pipeline);
            if (upload.draw->wave_effect) {
                vkCmdPushConstants(
                    presenter.command_buffer,
                    presenter.guest_pipeline_layout,
                    VK_SHADER_STAGE_FRAGMENT_BIT,
                    sizeof(float) * 4u,
                    sizeof(upload.draw->wave_parameters),
                    upload.draw->wave_parameters.data());
            }
            const auto descriptor =
                upload.source_target != nullptr
                ? upload.draw->nearest_texture
                    ? upload.source_target->nearest_descriptor
                    : upload.source_target->descriptor
                : upload.texture->descriptor;
            vkCmdBindDescriptorSets(
                presenter.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                presenter.guest_pipeline_layout,
                0,
                1,
                &descriptor,
                0,
                nullptr);
            vkCmdBindVertexBuffers(
                presenter.command_buffer, 0, 1,
                &presenter.staging, &upload.vertex_offset);
            vkCmdBindIndexBuffer(
                presenter.command_buffer,
                presenter.staging,
                upload.index_offset,
                VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(
                presenter.command_buffer,
                static_cast<std::uint32_t>(
                    upload.draw->indices.size()),
                std::max(upload.draw->instance_count, 1u),
                0,
                0,
                0);
        }
        vkCmdEndRenderPass(presenter.command_buffer);
    }
    for (auto* const sampled_target : sampled_targets) {
        ImageBarrier(
            presenter.command_buffer,
            sampled_target->image,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    }
    ImageBarrier(
        presenter.command_buffer,
        target->image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        render_batch ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0,
        VK_ACCESS_TRANSFER_READ_BIT,
        render_batch
            ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkImageBlit blit{
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0},
         {static_cast<std::int32_t>(
              target->extent.width),
          static_cast<std::int32_t>(
              target->extent.height), 1}},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {{0, 0, 0},
         {static_cast<std::int32_t>(
              presenter.swapchain_extent.width),
          static_cast<std::int32_t>(
              presenter.swapchain_extent.height), 1}}};
    vkCmdBlitImage(
        presenter.command_buffer,
        target->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blit, VK_FILTER_NEAREST);
    ImageBarrier(
        presenter.command_buffer,
        presenter.swapchain_images[image_index],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    ImageBarrier(
        presenter.command_buffer,
        target->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    result = vkEndCommandBuffer(presenter.command_buffer);
    if (result != VK_SUCCESS) {
        LogFailure("vkEndCommandBuffer(guest)", result);
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
        LogFailure("vkQueueSubmit(guest)", result);
        return false;
    }
    presenter.submission_in_flight = true;
    target->initialized = true;
    if (render_batch) {
        target->last_batch_id = frame.batch_id;
        presenter.last_guest_batch_id = frame.batch_id;
        presenter.last_guest_target_key = frame.target_key;
    }
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
        LogFailure("vkQueuePresentKHR(guest)", result);
        return false;
    }
    g_present_count.fetch_add(1u, std::memory_order_relaxed);
    return true;
}

bool SubmitFrame(Presenter& presenter,
                 const std::uint8_t* const rgba,
                 const std::size_t byte_count,
                 const std::uint32_t width,
                 const std::uint32_t height) {
    if (presenter.submission_in_flight &&
        presenter.device != VK_NULL_HANDLE &&
        presenter.fence != VK_NULL_HANDLE) {
        const auto fence_status = vkGetFenceStatus(
            presenter.device, presenter.fence);
        if (fence_status == VK_NOT_READY) {
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
        height == presenter.swapchain_extent.height &&
        presenter.swapchain_format == VK_FORMAT_R8G8B8A8_UNORM;
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
    presenter.submission_in_flight = true;
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
    g_present_count.fetch_add(1u, std::memory_order_relaxed);
    return true;
}

}

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

bool PresentVulkanGuestFrame(
    ANativeWindow* const window,
    const VulkanGuestFrame& frame,
    const std::uint64_t frame_number) {
    if (window == nullptr || frame.width == 0 ||
        frame.height == 0 || frame.draws.empty()) {
        return false;
    }
    const std::lock_guard lock{g_mutex};
    if (g_presenter.window != nullptr &&
        g_presenter.window != window) {
        DestroyPresenter(g_presenter);
    }
    if (g_presenter.configured_width != frame.width ||
        g_presenter.configured_height != frame.height) {
        (void)ANativeWindow_setBuffersGeometry(
            window,
            static_cast<std::int32_t>(frame.width),
            static_cast<std::int32_t>(frame.height),
            WINDOW_FORMAT_RGBA_8888);
        g_presenter.configured_width = frame.width;
        g_presenter.configured_height = frame.height;
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
    if (!SubmitGuestFrame(g_presenter, frame)) {
        DestroyPresenter(g_presenter);
        g_presenter.window = window;
        return false;
    }
    if (frame_number <= 8u || frame_number % 120u == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "vulkan guest present count=%llu draws=%zu "
            "source=%ux%u swapchain=%ux%u textures=%zu",
            static_cast<unsigned long long>(frame_number),
            frame.draws.size(),
            frame.width,
            frame.height,
            g_presenter.swapchain_extent.width,
            g_presenter.swapchain_extent.height,
            g_presenter.guest_textures.size());
    }
    return true;
}

bool EnsureVulkanPresenter(ANativeWindow* const window) {
    if (window == nullptr) {
        return false;
    }
    const std::lock_guard lock{g_mutex};
    if (g_presenter.window != nullptr &&
        g_presenter.window != window) {
        DestroyPresenter(g_presenter);
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
    return true;
}

bool ExecuteVulkanGen5ComputeBuffers(
    const std::span<const std::uint32_t> spirv,
    const Gen5ComputeProgramInfo& program,
    const std::array<std::uint32_t, 3>& groups,
    const VulkanGen5ReadBytes read_bytes,
    const VulkanGen5WriteBytes write_bytes,
    void* const memory_context) {
    if (spirv.empty() || program.buffers.empty() ||
        !program.images.empty() || !program.samplers.empty() ||
        !program.addresses.empty() || read_bytes == nullptr ||
        write_bytes == nullptr || groups[0] == 0 ||
        groups[1] == 0 || groups[2] == 0) {
        return false;
    }
    for (const auto& binding : program.descriptors) {
        // Buffers, FlattenedSrt and UserData are storage-buffer bindings.
        if (binding.kind != 0u && binding.kind != 16u &&
            binding.kind != 17u) {
            return false;
        }
    }
    const std::lock_guard lock{g_mutex};
    auto& presenter = g_presenter;
    if (presenter.device == VK_NULL_HANDLE ||
        presenter.queue == VK_NULL_HANDLE ||
        presenter.command_pool == VK_NULL_HANDLE ||
        presenter.permanently_failed) {
        return false;
    }

    struct HostBuffer {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        void* map{};
        VkDeviceSize size{};
        std::uint64_t guest_address{};
        bool written{};
    };
    std::vector<HostBuffer> buffers;
    VkShaderModule module{};
    VkDescriptorSetLayout set_layout{};
    VkPipelineLayout pipeline_layout{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptor_pool{};
    VkCommandBuffer command_buffer{};
    VkFence fence{};
    const auto cleanup = [&] {
        if (presenter.device == VK_NULL_HANDLE) {
            return;
        }
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(presenter.device, fence, nullptr);
        }
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(
                presenter.device, presenter.command_pool, 1,
                &command_buffer);
        }
        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(
                presenter.device, descriptor_pool, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(presenter.device, pipeline, nullptr);
        }
        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                presenter.device, pipeline_layout, nullptr);
        }
        if (set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                presenter.device, set_layout, nullptr);
        }
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(presenter.device, module, nullptr);
        }
        for (auto& buffer : buffers) {
            if (buffer.map != nullptr) {
                vkUnmapMemory(presenter.device, buffer.memory);
            }
            if (buffer.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(
                    presenter.device, buffer.buffer, nullptr);
            }
            if (buffer.memory != VK_NULL_HANDLE) {
                vkFreeMemory(
                    presenter.device, buffer.memory, nullptr);
            }
        }
    };
    const auto fail = [&](const char* const operation,
                          const VkResult result) {
        LogFailure(operation, result);
        cleanup();
        return false;
    };
    const auto create_buffer =
        [&](const std::uint64_t guest_address,
            const VkDeviceSize requested_size,
            const bool read, const bool written,
            const void* const initial_data) {
            if (requested_size == 0 ||
                requested_size > UINT64_C(0x10000000)) {
                return false;
            }
            HostBuffer host{};
            host.size = std::max<VkDeviceSize>(requested_size, 16u);
            host.guest_address = guest_address;
            host.written = written;
            const VkBufferCreateInfo info{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                nullptr, 0, host.size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
            auto result = vkCreateBuffer(
                presenter.device, &info, nullptr, &host.buffer);
            if (result != VK_SUCCESS) {
                return false;
            }
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(
                presenter.device, host.buffer, &requirements);
            const auto memory_type = FindMemoryType(
                presenter, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (memory_type ==
                std::numeric_limits<std::uint32_t>::max()) {
                vkDestroyBuffer(
                    presenter.device, host.buffer, nullptr);
                return false;
            }
            const VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                nullptr, requirements.size, memory_type};
            result = vkAllocateMemory(
                presenter.device, &allocation, nullptr,
                &host.memory);
            if (result == VK_SUCCESS) {
                result = vkBindBufferMemory(
                    presenter.device, host.buffer,
                    host.memory, 0);
            }
            if (result == VK_SUCCESS) {
                result = vkMapMemory(
                    presenter.device, host.memory, 0,
                    host.size, 0, &host.map);
            }
            if (result != VK_SUCCESS || host.map == nullptr) {
                if (host.memory != VK_NULL_HANDLE) {
                    vkFreeMemory(
                        presenter.device, host.memory, nullptr);
                }
                vkDestroyBuffer(
                    presenter.device, host.buffer, nullptr);
                return false;
            }
            std::memset(
                host.map, 0,
                static_cast<std::size_t>(host.size));
            if (initial_data != nullptr) {
                std::memcpy(
                    host.map, initial_data,
                    static_cast<std::size_t>(requested_size));
            } else if (read &&
                       !read_bytes(
                           memory_context, guest_address, host.map,
                           static_cast<std::size_t>(requested_size))) {
                vkUnmapMemory(
                    presenter.device, host.memory);
                vkFreeMemory(
                    presenter.device, host.memory, nullptr);
                vkDestroyBuffer(
                    presenter.device, host.buffer, nullptr);
                return false;
            }
            buffers.push_back(host);
            return true;
        };

    buffers.reserve(
        program.buffers.size() + program.descriptors.size());
    std::uint64_t total_guest_bytes{};
    for (const auto& buffer : program.buffers) {
        if (buffer.byte_count >
                UINT64_C(0x10000000) - total_guest_bytes) {
            cleanup();
            return false;
        }
        total_guest_bytes += buffer.byte_count;
        if (buffer.guest_address == 0 ||
            !create_buffer(
                buffer.guest_address, buffer.byte_count,
                buffer.read, buffer.written, nullptr)) {
            cleanup();
            return false;
        }
    }
    std::size_t flattened_index = SIZE_MAX;
    std::size_t user_data_index = SIZE_MAX;
    for (const auto& binding : program.descriptors) {
        if (binding.kind == 16u) {
            flattened_index = buffers.size();
            const auto size = std::max<std::size_t>(
                program.flattened_srt.size() *
                    sizeof(std::uint32_t),
                sizeof(std::uint32_t));
            std::vector<std::uint32_t> data(
                (size + sizeof(std::uint32_t) - 1u) /
                    sizeof(std::uint32_t));
            std::copy(
                program.flattened_srt.begin(),
                program.flattened_srt.end(), data.begin());
            if (!create_buffer(
                    0, size, false, false, data.data())) {
                cleanup();
                return false;
            }
        } else if (binding.kind == 17u) {
            user_data_index = buffers.size();
            const auto size = std::max<std::size_t>(
                program.packed_user_data.size() *
                    sizeof(std::uint32_t),
                sizeof(std::uint32_t));
            std::vector<std::uint32_t> data(
                (size + sizeof(std::uint32_t) - 1u) /
                    sizeof(std::uint32_t));
            std::copy(
                program.packed_user_data.begin(),
                program.packed_user_data.end(), data.begin());
            if (!create_buffer(
                    0, size, false, false, data.data())) {
                cleanup();
                return false;
            }
        }
    }

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    layout_bindings.reserve(program.descriptors.size());
    std::uint32_t descriptor_count{};
    for (const auto& binding : program.descriptors) {
        const auto count =
            binding.kind == 0u
            ? static_cast<std::uint32_t>(
                  binding.resources.size())
            : 1u;
        if (count == 0) {
            continue;
        }
        layout_bindings.push_back({
            binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        descriptor_count += count;
    }
    const VkDescriptorSetLayoutCreateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0,
        static_cast<std::uint32_t>(layout_bindings.size()),
        layout_bindings.data()};
    auto result = vkCreateDescriptorSetLayout(
        presenter.device, &set_info, nullptr, &set_layout);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreateDescriptorSetLayout", result);
    }
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = program.push_constant_offset;
    push_range.size = program.push_constant_size;
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &set_layout,
        push_range.size == 0 ? 0u : 1u,
        push_range.size == 0 ? nullptr : &push_range};
    result = vkCreatePipelineLayout(
        presenter.device, &pipeline_layout_info, nullptr,
        &pipeline_layout);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreatePipelineLayout", result);
    }
    const VkShaderModuleCreateInfo module_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, spirv.size_bytes(), spirv.data()};
    result = vkCreateShaderModule(
        presenter.device, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreateShaderModule", result);
    }
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module,
        "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout, VK_NULL_HANDLE, 0};
    result = vkCreateComputePipelines(
        presenter.device, VK_NULL_HANDLE, 1, &pipeline_info,
        nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreateComputePipelines", result);
    }
    const VkDescriptorPoolSize pool_size{
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        std::max(descriptor_count, 1u)};
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1, 1, &pool_size};
    result = vkCreateDescriptorPool(
        presenter.device, &pool_info, nullptr,
        &descriptor_pool);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreateDescriptorPool", result);
    }
    const VkDescriptorSetAllocateInfo allocation_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool, 1, &set_layout};
    VkDescriptorSet descriptor_set{};
    result = vkAllocateDescriptorSets(
        presenter.device, &allocation_info, &descriptor_set);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkAllocateDescriptorSets", result);
    }
    std::vector<std::vector<VkDescriptorBufferInfo>>
        binding_infos;
    std::vector<VkWriteDescriptorSet> writes;
    binding_infos.reserve(program.descriptors.size());
    writes.reserve(program.descriptors.size());
    for (const auto& binding : program.descriptors) {
        binding_infos.emplace_back();
        auto& infos = binding_infos.back();
        if (binding.kind == 0u) {
            infos.reserve(binding.resources.size());
            for (const auto resource : binding.resources) {
                if (resource >= program.buffers.size()) {
                    cleanup();
                    return false;
                }
                const auto& host = buffers[resource];
                infos.push_back({
                    host.buffer, 0, host.size});
            }
        } else {
            const auto index =
                binding.kind == 16u
                ? flattened_index : user_data_index;
            if (index >= buffers.size()) {
                cleanup();
                return false;
            }
            const auto& host = buffers[index];
            infos.push_back({host.buffer, 0, host.size});
        }
        if (!infos.empty()) {
            writes.push_back({
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr, descriptor_set, binding.binding, 0,
                static_cast<std::uint32_t>(infos.size()),
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr, infos.data(), nullptr});
        }
    }
    vkUpdateDescriptorSets(
        presenter.device,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    const VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, presenter.command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    result = vkAllocateCommandBuffers(
        presenter.device, &command_info, &command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkAllocateCommandBuffers", result);
    }
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    result = vkCreateFence(
        presenter.device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkCreateFence", result);
    }
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr};
    result = vkBeginCommandBuffer(command_buffer, &begin);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkBeginCommandBuffer", result);
    }
    vkCmdBindPipeline(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline);
    vkCmdBindDescriptorSets(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout, program.descriptor_set, 1,
        &descriptor_set, 0, nullptr);
    if (program.push_constant_size != 0) {
        if (program.packed_user_data.size() *
                sizeof(std::uint32_t) !=
            program.push_constant_size) {
            cleanup();
            return false;
        }
        vkCmdPushConstants(
            command_buffer, pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            program.push_constant_offset,
            program.push_constant_size,
            program.packed_user_data.data());
    }
    vkCmdDispatch(
        command_buffer, groups[0], groups[1], groups[2]);
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkEndCommandBuffer", result);
    }
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr, 0, nullptr, nullptr, 1,
        &command_buffer, 0, nullptr};
    result = vkQueueSubmit(
        presenter.queue, 1, &submit, fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 vkQueueSubmit", result);
    }
    result = vkWaitForFences(
        presenter.device, 1, &fence, VK_TRUE,
        UINT64_C(5000000000));
    if (result != VK_SUCCESS) {
        return fail("gen5 vkWaitForFences", result);
    }
    bool copied = true;
    for (std::size_t index = 0;
         index < program.buffers.size(); ++index) {
        const auto& guest = program.buffers[index];
        const auto& host = buffers[index];
        if (guest.written &&
            !write_bytes(
                memory_context, guest.guest_address, host.map,
                static_cast<std::size_t>(guest.byte_count))) {
            copied = false;
        }
    }
    cleanup();
    return copied;
}

struct Gen5GuestFormat {
    VkFormat vulkan{VK_FORMAT_UNDEFINED};
    std::uint32_t element_bytes{};
    std::uint32_t block_width{1u};
    std::uint32_t block_height{1u};
};

Gen5GuestFormat GetGen5GuestFormat(
    const std::uint32_t format,
    const bool force_uint = false) {
    if (force_uint) {
        return {VK_FORMAT_R32_UINT, 4u};
    }
    switch (format) {
    case 1u: return {VK_FORMAT_R8_UNORM, 1u};
    case 5u: return {VK_FORMAT_R8_UINT, 1u};
    case 7u: return {VK_FORMAT_R16_UNORM, 2u};
    case 11u: return {VK_FORMAT_R16_UINT, 2u};
    case 13u: return {VK_FORMAT_R16_SFLOAT, 2u};
    case 14u: return {VK_FORMAT_R8G8_UNORM, 2u};
    case 18u: return {VK_FORMAT_R8G8_UINT, 2u};
    case 20u: return {VK_FORMAT_R32_UINT, 4u};
    case 21u: return {VK_FORMAT_R32_SINT, 4u};
    case 22u: return {VK_FORMAT_R32_SFLOAT, 4u};
    case 23u: return {VK_FORMAT_R16G16_UNORM, 4u};
    case 27u: return {VK_FORMAT_R16G16_UINT, 4u};
    case 29u: return {VK_FORMAT_R16G16_SFLOAT, 4u};
    case 36u: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4u};
    case 56u: return {VK_FORMAT_R8G8B8A8_UNORM, 4u};
    case 60u: return {VK_FORMAT_R8G8B8A8_UINT, 4u};
    case 62u: return {VK_FORMAT_R32G32_UINT, 8u};
    case 64u: return {VK_FORMAT_R32G32_SFLOAT, 8u};
    case 65u: return {VK_FORMAT_R16G16B16A16_UNORM, 8u};
    case 69u: return {VK_FORMAT_R16G16B16A16_UINT, 8u};
    case 71u: return {VK_FORMAT_R16G16B16A16_SFLOAT, 8u};
    case 75u: return {VK_FORMAT_R32G32B32A32_UINT, 16u};
    case 77u: return {VK_FORMAT_R32G32B32A32_SFLOAT, 16u};
    case 169u: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8u, 4u, 4u};
    case 170u: return {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, 8u, 4u, 4u};
    case 171u: return {VK_FORMAT_BC2_UNORM_BLOCK, 16u, 4u, 4u};
    case 172u: return {VK_FORMAT_BC2_SRGB_BLOCK, 16u, 4u, 4u};
    case 173u: return {VK_FORMAT_BC3_UNORM_BLOCK, 16u, 4u, 4u};
    case 174u: return {VK_FORMAT_BC3_SRGB_BLOCK, 16u, 4u, 4u};
    case 175u: return {VK_FORMAT_BC4_UNORM_BLOCK, 8u, 4u, 4u};
    case 176u: return {VK_FORMAT_BC4_SNORM_BLOCK, 8u, 4u, 4u};
    case 177u: return {VK_FORMAT_BC5_UNORM_BLOCK, 16u, 4u, 4u};
    case 178u: return {VK_FORMAT_BC5_SNORM_BLOCK, 16u, 4u, 4u};
    case 179u: return {VK_FORMAT_BC6H_UFLOAT_BLOCK, 16u, 4u, 4u};
    case 180u: return {VK_FORMAT_BC6H_SFLOAT_BLOCK, 16u, 4u, 4u};
    case 181u: return {VK_FORMAT_BC7_UNORM_BLOCK, 16u, 4u, 4u};
    case 182u: return {VK_FORMAT_BC7_SRGB_BLOCK, 16u, 4u, 4u};
    default: return {};
    }
}

Gen5GuestFormat GetGen5RenderTargetFormat(
    const std::uint32_t layout,
    const std::uint32_t type,
    const std::uint32_t order) {
    if (layout == 9u && type == 0u) {
        return {
            order == 1u
                ? VK_FORMAT_A2R10G10B10_UNORM_PACK32
                : VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            4u};
    }
    if (layout == 10u) {
        if (type == 0u) {
            return {
                order == 1u
                    ? VK_FORMAT_B8G8R8A8_UNORM
                    : VK_FORMAT_R8G8B8A8_UNORM,
                4u};
        }
        if (type == 1u && order == 0u) {
            return {VK_FORMAT_R8G8B8A8_SNORM, 4u};
        }
        if (type == 4u && order == 0u) {
            return {VK_FORMAT_R8G8B8A8_UINT, 4u};
        }
        if (type == 5u && order == 0u) {
            return {VK_FORMAT_R8G8B8A8_SINT, 4u};
        }
        if ((type == 6u || type == 9u) && order <= 1u) {
            return {
                order == 1u
                    ? VK_FORMAT_B8G8R8A8_SRGB
                    : VK_FORMAT_R8G8B8A8_SRGB,
                4u};
        }
    }
    if (layout == 12u && order <= 2u) {
        if (type == 0u) {
            return {VK_FORMAT_R16G16B16A16_UNORM, 8u};
        }
        if (type == 4u) {
            return {VK_FORMAT_R16G16B16A16_UINT, 8u};
        }
        if (type == 7u) {
            return {VK_FORMAT_R16G16B16A16_SFLOAT, 8u};
        }
    }
    if (layout == 6u && type == 7u && order == 0u) {
        return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4u};
    }
    if (layout == 4u && type == 7u && order == 0u) {
        return {VK_FORMAT_R32_SFLOAT, 4u};
    }
    return {};
}

bool GetGen5SwizzlePattern(
    const std::uint32_t mode,
    const std::uint32_t element_bytes,
    std::array<std::uint32_t, 16>& x_masks,
    std::array<std::uint32_t, 16>& y_masks,
    std::uint32_t& address_bits,
    std::uint32_t& block_bytes) {
    x_masks.fill(0u);
    y_masks.fill(0u);
    const auto x = [&](const std::uint32_t bit) {
        return 1u << bit;
    };
    const auto set = [&](const std::uint32_t index,
                         const std::uint32_t xm,
                         const std::uint32_t ym) {
        x_masks[index] = xm;
        y_masks[index] = ym;
    };
    std::uint32_t bpp_log{};
    for (auto bytes = element_bytes;
         bytes > 1u; bytes >>= 1u) {
        if ((bytes & 1u) != 0u) {
            return false;
        }
        ++bpp_log;
    }
    if ((1u << bpp_log) != element_bytes ||
        bpp_log > 4u) {
        return false;
    }
    if (mode == 5u) {
        address_bits = 12u;
        block_bytes = 4096u;
        constexpr std::array<std::array<std::uint8_t, 12>, 5>
            axis{{
                {1,1,1,1,2,2,2,2,2,1,2,1},
                {0,1,1,1,2,2,2,1,2,1,2,1},
                {0,0,1,1,2,2,2,1,2,1,2,1},
                {0,0,0,1,2,2,1,1,2,1,2,1},
                {0,0,0,0,2,2,1,1,2,1,2,1}}};
        constexpr std::array<std::array<std::uint8_t, 12>, 5>
            bit{{
                {0,1,2,3,0,1,2,3,4,4,5,5},
                {0,0,1,2,0,1,2,3,3,4,4,5},
                {0,0,0,1,0,1,2,2,3,3,4,4},
                {0,0,0,0,0,1,1,2,2,3,3,4},
                {0,0,0,0,0,1,0,1,2,2,3,3}}};
        for (std::uint32_t index = 0; index < 12u; ++index) {
            set(index,
                axis[bpp_log][index] == 1
                    ? x(bit[bpp_log][index]) : 0u,
                axis[bpp_log][index] == 2
                    ? x(bit[bpp_log][index]) : 0u);
        }
        return true;
    }
    if (mode != 9u && mode != 24u && mode != 27u) {
        return false;
    }
    address_bits = 16u;
    block_bytes = 65536u;
    if (mode == 9u) {
        constexpr std::array<std::array<std::uint8_t, 16>, 5>
            axis{{
                {1,1,1,1,2,2,2,2,2,1,2,1,2,1,2,1},
                {0,1,1,1,2,2,2,1,2,1,2,1,2,1,2,1},
                {0,0,1,1,2,2,2,1,2,1,2,1,2,1,2,1},
                {0,0,0,1,2,2,1,1,2,1,2,1,2,1,2,1},
                {0,0,0,0,2,2,1,1,2,1,2,1,2,1,2,1}}};
        constexpr std::array<std::array<std::uint8_t, 16>, 5>
            bit{{
                {0,1,2,3,0,1,2,3,4,4,5,5,6,6,7,7},
                {0,0,1,2,0,1,2,3,3,4,4,5,5,6,6,7},
                {0,0,0,1,0,1,2,2,3,3,4,4,5,5,6,6},
                {0,0,0,0,0,1,1,2,2,3,3,4,4,5,5,6},
                {0,0,0,0,0,1,0,1,2,2,3,3,4,4,5,5}}};
        for (std::uint32_t index = 0; index < 16u; ++index) {
            set(index,
                axis[bpp_log][index] == 1
                    ? x(bit[bpp_log][index]) : 0u,
                axis[bpp_log][index] == 2
                    ? x(bit[bpp_log][index]) : 0u);
        }
        return true;
    }
    // Oberon RB+ 64 KiB Z_X/R_X exact single-sample equations.
    constexpr std::array<std::array<std::uint8_t, 16>, 5>
        render_axis{{
            {1,1,1,1,2,2,2,2,3,3,3,3,1,2,3,3},
            {0,1,1,1,2,2,2,1,3,3,3,3,2,1,3,3},
            {0,0,1,1,2,2,1,2,3,3,3,3,1,2,3,3},
            {0,0,0,1,2,1,1,2,3,3,3,3,2,1,3,3},
            {0,0,0,0,1,2,1,2,3,3,3,3,1,2,3,3}}};
    constexpr std::array<std::array<std::uint8_t, 16>, 5>
        render_xbit{{
            {0,1,2,3,0,0,0,0,7,4,6,5,6,0,7,8},
            {0,0,1,2,0,0,0,3,7,4,6,5,0,6,7,8},
            {0,0,0,1,0,0,2,0,7,4,6,5,3,0,6,7},
            {0,0,0,0,0,1,2,0,7,4,6,5,0,3,7,6},
            {0,0,0,0,0,0,1,0,7,4,6,5,2,0,6,3}}};
    constexpr std::array<std::array<std::uint8_t, 16>, 5>
        render_ybit{{
            {0,0,0,0,0,1,2,3,4,4,5,6,0,6,8,7},
            {0,0,0,0,0,1,2,0,4,4,5,6,3,0,7,6},
            {0,0,0,0,0,1,0,2,4,4,5,6,0,3,7,6},
            {0,0,0,0,0,0,0,1,4,4,5,6,2,0,3,6},
            {0,0,0,0,0,0,0,1,4,4,5,6,0,2,3,6}}};
    for (std::uint32_t index = 0; index < 16u; ++index) {
        auto axis = render_axis[bpp_log][index];
        auto xb = render_xbit[bpp_log][index];
        auto yb = render_ybit[bpp_log][index];
        if (mode == 24u && index < 8u) {
            // Z_X interleaves x/y in the low byte.
            if (index >= bpp_log) {
                const auto relative = index - bpp_log;
                axis = (relative & 1u) == 0u ? 1u : 2u;
                const auto coordinate_bit = relative / 2u;
                xb = coordinate_bit;
                yb = coordinate_bit;
            }
        }
        if (axis == 1u) {
            set(index, x(xb), 0u);
        } else if (axis == 2u) {
            set(index, 0u, x(yb));
        } else if (axis == 3u) {
            // The RB+ high bits combine one X and one/two Y bits.
            std::uint32_t ym = x(yb);
            if (index == 8u) {
                ym |= x(7u);
            }
            set(index, x(xb), ym);
        }
    }
    return true;
}

bool TransformGen5GuestImage(
    const bool to_linear,
    const std::span<const std::uint8_t> source,
    const std::span<std::uint8_t> destination,
    const std::uint32_t mode,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t element_bytes) {
    const auto linear_size =
        static_cast<std::uint64_t>(width) * height *
        element_bytes;
    if (linear_size > (to_linear
            ? destination.size() : source.size())) {
        return false;
    }
    if (mode == 0u) {
        const auto row_bytes =
            static_cast<std::uint64_t>(width) *
            element_bytes;
        const auto pitch = (row_bytes + 255u) & ~UINT64_C(255);
        if (pitch * height >
            (to_linear ? source.size() : destination.size())) {
            return false;
        }
        for (std::uint32_t row = 0; row < height; ++row) {
            const auto linear_offset =
                static_cast<std::size_t>(row * row_bytes);
            const auto guest_offset =
                static_cast<std::size_t>(row * pitch);
            if (to_linear) {
                std::memcpy(
                    destination.data() + linear_offset,
                    source.data() + guest_offset,
                    static_cast<std::size_t>(row_bytes));
            } else {
                std::memcpy(
                    destination.data() + guest_offset,
                    source.data() + linear_offset,
                    static_cast<std::size_t>(row_bytes));
            }
        }
        return true;
    }
    std::array<std::uint32_t, 16> x_masks{};
    std::array<std::uint32_t, 16> y_masks{};
    std::uint32_t address_bits{};
    std::uint32_t block_bytes{};
    if (!GetGen5SwizzlePattern(
            mode, element_bytes, x_masks, y_masks,
            address_bits, block_bytes)) {
        return false;
    }
    const auto block_elements = block_bytes / element_bytes;
    const auto total_bits =
        static_cast<std::uint32_t>(
            __builtin_ctz(block_elements));
    const auto width_bits = (total_bits + 1u) / 2u;
    const auto height_bits = total_bits - width_bits;
    const auto block_width = 1u << width_bits;
    const auto block_height = 1u << height_bits;
    const auto blocks_wide =
        (static_cast<std::uint64_t>(width) +
         block_width - 1u) / block_width;
    const auto blocks_high =
        (static_cast<std::uint64_t>(height) +
         block_height - 1u) / block_height;
    const auto guest_size =
        blocks_wide * blocks_high * block_bytes;
    if (guest_size >
        (to_linear ? source.size() : destination.size())) {
        return false;
    }
    const auto axis_term =
        [&](const std::uint32_t coordinate,
            const auto& masks) {
            std::uint32_t offset{};
            for (std::uint32_t bit = 0;
                 bit < address_bits; ++bit) {
                offset |=
                    (static_cast<std::uint32_t>(
                         __builtin_popcount(
                             coordinate & masks[bit])) &
                     1u) << bit;
            }
            return offset;
        };
    std::vector<std::uint32_t> x_terms(width);
    for (std::uint32_t x_coord = 0;
         x_coord < width; ++x_coord) {
        x_terms[x_coord] =
            axis_term(x_coord, x_masks);
    }
    for (std::uint32_t y_coord = 0;
         y_coord < height; ++y_coord) {
        const auto y_term = axis_term(y_coord, y_masks);
        const auto block_y = y_coord / block_height;
        for (std::uint32_t x_coord = 0;
             x_coord < width; ++x_coord) {
            const auto block_x = x_coord / block_width;
            const auto block_index =
                static_cast<std::uint64_t>(block_y) *
                    blocks_wide + block_x;
            const auto guest_offset =
                block_index * block_bytes +
                (x_terms[x_coord] ^ y_term);
            const auto linear_offset =
                (static_cast<std::uint64_t>(y_coord) *
                     width + x_coord) *
                element_bytes;
            if (to_linear) {
                std::memcpy(
                    destination.data() + linear_offset,
                    source.data() + guest_offset,
                    element_bytes);
            } else {
                std::memcpy(
                    destination.data() + guest_offset,
                    source.data() + linear_offset,
                    element_bytes);
            }
        }
    }
    return true;
}

bool ExecuteVulkanGen5ComputeImages(
    const std::span<const std::uint32_t> spirv,
    const Gen5ComputeProgramInfo& program,
    const std::array<std::uint32_t, 3>& groups,
    const VulkanGen5ReadBytes read_bytes,
    const VulkanGen5WriteBytes write_bytes,
    void* const memory_context) {
    if (spirv.empty() || program.images.empty() ||
        read_bytes == nullptr || write_bytes == nullptr ||
        groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return false;
    }
    for (const auto& image : program.images) {
        const auto null_image =
            image.type == 0u && image.guest_address == 0u;
        if ((!null_image &&
             image.type != 9u && image.type != 10u &&
             image.type != 11u && image.type != 13u) ||
            image.depth == 0u || image.depth > 256u ||
            image.width == 0 || image.height == 0 ||
            image.width > 8192u || image.height > 8192u) {
            return false;
        }
    }
    for (const auto& binding : program.descriptors) {
        if (!(binding.kind == 0u ||
              (binding.kind >= 1u && binding.kind <= 13u) ||
              binding.kind == 15u || binding.kind == 16u ||
              binding.kind == 17u)) {
            return false;
        }
    }
    const std::lock_guard lock{g_mutex};
    auto& presenter = g_presenter;
    if (presenter.device == VK_NULL_HANDLE ||
        presenter.queue == VK_NULL_HANDLE ||
        presenter.command_pool == VK_NULL_HANDLE ||
        presenter.permanently_failed) {
        return false;
    }

    struct NativeImage {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        std::uint32_t element_bytes{};
        std::uint32_t element_width{};
        std::uint32_t element_height{};
        std::size_t staging_offset{};
        std::size_t linear_size{};
        std::size_t guest_size{};
        std::size_t linear_slice_size{};
        std::size_t guest_slice_size{};
    };
    struct NativeBuffer {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        void* map{};
        VkDeviceSize size{};
    };
    std::vector<NativeImage> images(program.images.size());
    std::vector<VkSampler> samplers;
    std::vector<NativeBuffer> guest_buffers;
    std::vector<NativeBuffer> address_buffers;
    NativeBuffer staging{};
    NativeBuffer flattened{};
    NativeBuffer user_data{};
    VkShaderModule module{};
    VkDescriptorSetLayout set_layout{};
    VkPipelineLayout pipeline_layout{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptor_pool{};
    VkCommandBuffer command_buffer{};
    VkFence fence{};
    const auto destroy_buffer = [&](NativeBuffer& buffer) {
        if (buffer.map != nullptr) {
            vkUnmapMemory(presenter.device, buffer.memory);
        }
        if (buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(
                presenter.device, buffer.buffer, nullptr);
        }
        if (buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(
                presenter.device, buffer.memory, nullptr);
        }
        buffer = {};
    };
    const auto cleanup = [&] {
        if (presenter.device == VK_NULL_HANDLE) {
            return;
        }
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(presenter.device, fence, nullptr);
        }
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(
                presenter.device, presenter.command_pool, 1,
                &command_buffer);
        }
        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(
                presenter.device, descriptor_pool, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(presenter.device, pipeline, nullptr);
        }
        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                presenter.device, pipeline_layout, nullptr);
        }
        if (set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(
                presenter.device, set_layout, nullptr);
        }
        if (module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(presenter.device, module, nullptr);
        }
        for (const auto sampler : samplers) {
            vkDestroySampler(presenter.device, sampler, nullptr);
        }
        for (auto& image : images) {
            if (image.view != VK_NULL_HANDLE) {
                vkDestroyImageView(
                    presenter.device, image.view, nullptr);
            }
            if (image.image != VK_NULL_HANDLE) {
                vkDestroyImage(
                    presenter.device, image.image, nullptr);
            }
            if (image.memory != VK_NULL_HANDLE) {
                vkFreeMemory(
                    presenter.device, image.memory, nullptr);
            }
        }
        for (auto& buffer : guest_buffers) {
            destroy_buffer(buffer);
        }
        for (auto& buffer : address_buffers) {
            destroy_buffer(buffer);
        }
        destroy_buffer(staging);
        destroy_buffer(flattened);
        destroy_buffer(user_data);
    };
    const auto fail = [&](const char* const operation,
                          const VkResult result) {
        LogFailure(operation, result);
        cleanup();
        return false;
    };
    const auto create_host_buffer =
        [&](NativeBuffer& buffer, const VkDeviceSize size,
            const VkBufferUsageFlags usage,
            const void* const initial_data) {
            buffer.size = std::max<VkDeviceSize>(size, 4u);
            const VkBufferCreateInfo info{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                nullptr, 0, buffer.size, usage,
                VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
            auto result = vkCreateBuffer(
                presenter.device, &info, nullptr,
                &buffer.buffer);
            if (result != VK_SUCCESS) {
                return false;
            }
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(
                presenter.device, buffer.buffer, &requirements);
            const auto type = FindMemoryType(
                presenter, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (type ==
                std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            const VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                nullptr, requirements.size, type};
            result = vkAllocateMemory(
                presenter.device, &allocation, nullptr,
                &buffer.memory);
            if (result == VK_SUCCESS) {
                result = vkBindBufferMemory(
                    presenter.device, buffer.buffer,
                    buffer.memory, 0);
            }
            if (result == VK_SUCCESS) {
                result = vkMapMemory(
                    presenter.device, buffer.memory, 0,
                    buffer.size, 0, &buffer.map);
            }
            if (result != VK_SUCCESS || buffer.map == nullptr) {
                return false;
            }
            std::memset(
                buffer.map, 0,
                static_cast<std::size_t>(buffer.size));
            if (initial_data != nullptr && size != 0) {
                std::memcpy(
                    buffer.map, initial_data,
                    static_cast<std::size_t>(size));
            }
            return true;
        };
    const auto guest_image_size =
        [&](const Gen5ImageBinding& image,
            const std::uint32_t bytes,
            const std::uint32_t width,
            const std::uint32_t height) -> std::uint64_t {
            if (image.tile_mode == 0u) {
                const auto pitch =
                    (static_cast<std::uint64_t>(
                         width) * bytes + 255u) &
                    ~UINT64_C(255);
                return pitch * height;
            }
            std::array<std::uint32_t, 16> xm{};
            std::array<std::uint32_t, 16> ym{};
            std::uint32_t bits{};
            std::uint32_t block_bytes{};
            if (!GetGen5SwizzlePattern(
                    image.tile_mode, bytes, xm, ym,
                    bits, block_bytes)) {
                return 0u;
            }
            const auto elements = block_bytes / bytes;
            const auto total_bits =
                static_cast<std::uint32_t>(
                    __builtin_ctz(elements));
            const auto block_width =
                1u << ((total_bits + 1u) / 2u);
            const auto block_height =
                1u << (total_bits / 2u);
            return
                ((static_cast<std::uint64_t>(width) +
                  block_width - 1u) / block_width) *
                ((static_cast<std::uint64_t>(height) +
                  block_height - 1u) / block_height) *
                block_bytes;
        };

    guest_buffers.resize(program.buffers.size());
    for (std::size_t index = 0;
         index < program.buffers.size(); ++index) {
        const auto& guest = program.buffers[index];
        if (guest.guest_address == 0 ||
            guest.byte_count == 0 ||
            guest.byte_count > UINT32_C(0x10000000) ||
            !create_host_buffer(
                guest_buffers[index], guest.byte_count,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                nullptr) ||
            (guest.read &&
             !read_bytes(
                 memory_context, guest.guest_address,
                 guest_buffers[index].map,
                 static_cast<std::size_t>(
                     guest.byte_count)))) {
            cleanup();
            return false;
        }
    }
    address_buffers.resize(program.addresses.size());
    for (std::size_t index = 0;
         index < program.addresses.size(); ++index) {
        const auto& guest = program.addresses[index];
        constexpr std::size_t FlatWindow = 0x10000u;
        if (guest.binding_base == 0 ||
            guest.written ||
            !create_host_buffer(
                address_buffers[index], FlatWindow,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                nullptr)) {
            cleanup();
            return false;
        }
        if (!read_bytes(
                memory_context, guest.binding_base,
                address_buffers[index].map, FlatWindow)) {
            bool read_any{};
            for (std::size_t size = 0x8000u;
                 size >= 0x100u; size >>= 1u) {
                if (read_bytes(
                        memory_context, guest.binding_base,
                        address_buffers[index].map, size)) {
                    read_any = true;
                    break;
                }
            }
            if (!read_any) {
                cleanup();
                return false;
            }
        }
    }

    std::size_t staging_size{};
    for (std::size_t index = 0;
         index < program.images.size(); ++index) {
        const auto& guest = program.images[index];
        const auto null_image =
            guest.type == 0u && guest.guest_address == 0u;
        bool force_uint{};
        for (const auto& binding : program.descriptors) {
            if (binding.kind < 10u || binding.kind > 12u) {
                continue;
            }
            if (std::ranges::find(
                    binding.resources, index) !=
                binding.resources.end()) {
                force_uint = true;
                break;
            }
        }
        const auto format = null_image
            ? Gen5GuestFormat{
                  force_uint ? VK_FORMAT_R32_UINT
                             : VK_FORMAT_R8G8B8A8_UNORM,
                  4u}
            : GetGen5GuestFormat(
                  guest.format, force_uint);
        if (format.vulkan == VK_FORMAT_UNDEFINED ||
            format.element_bytes == 0u) {
            cleanup();
            return false;
        }
        auto& native = images[index];
        native.format = format.vulkan;
        native.element_bytes = format.element_bytes;
        native.element_width =
            (guest.width + format.block_width - 1u) /
            format.block_width;
        native.element_height =
            (guest.height + format.block_height - 1u) /
            format.block_height;
        native.linear_slice_size =
            static_cast<std::size_t>(native.element_width) *
            native.element_height * format.element_bytes;
        native.guest_slice_size = null_image
            ? native.linear_slice_size
            : static_cast<std::size_t>(
                  guest_image_size(
                      guest, format.element_bytes,
                      native.element_width,
                      native.element_height));
        native.linear_size =
            native.linear_slice_size * guest.depth;
        native.guest_size =
            native.guest_slice_size * guest.depth;
        if (native.guest_size == 0u ||
            native.guest_size > UINT32_C(0x10000000) ||
            native.linear_size > UINT32_C(0x10000000)) {
            cleanup();
            return false;
        }
        staging_size =
            (staging_size + 255u) &
            ~static_cast<std::size_t>(255u);
        native.staging_offset = staging_size;
        if (native.linear_size >
            SIZE_MAX - staging_size) {
            cleanup();
            return false;
        }
        staging_size += native.linear_size;
    }
    if (staging_size == 0u ||
        staging_size > UINT32_C(0x20000000) ||
        !create_host_buffer(
            staging, staging_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            nullptr)) {
        cleanup();
        return false;
    }
    for (std::size_t index = 0;
         index < program.images.size(); ++index) {
        const auto& guest = program.images[index];
        auto& native = images[index];
        const auto null_image =
            guest.type == 0u && guest.guest_address == 0u;
        std::vector<std::uint8_t> tiled(native.guest_size);
        if (!null_image && !read_bytes(
                memory_context, guest.guest_address,
                tiled.data(), tiled.size())) {
            cleanup();
            return false;
        }
        for (std::uint32_t slice = 0;
             slice < guest.depth; ++slice) {
            const auto tiled_slice =
                std::span<const std::uint8_t>{
                    tiled.data() +
                        slice * native.guest_slice_size,
                    native.guest_slice_size};
            auto linear_slice =
                std::span<std::uint8_t>{
                    static_cast<std::uint8_t*>(
                        staging.map) +
                        native.staging_offset +
                        slice * native.linear_slice_size,
                    native.linear_slice_size};
            if (!null_image && !TransformGen5GuestImage(
                    true, tiled_slice, linear_slice,
                    guest.tile_mode, native.element_width,
                    native.element_height,
                    native.element_bytes)) {
                cleanup();
                return false;
            }
        }
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(
            presenter.physical_device, native.format,
            &properties);
        auto required_features =
            VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (guest.read) {
            required_features |=
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        }
        if (guest.written || guest.atomic) {
            required_features |=
                VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        }
        if ((properties.optimalTilingFeatures &
             required_features) != required_features) {
            cleanup();
            return false;
        }
        VkImageUsageFlags usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (guest.read) {
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (guest.written || guest.atomic) {
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        const VkImageCreateInfo image_info{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            nullptr, 0,
            guest.type == 10u
                ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
            native.format,
            {guest.width, guest.height,
             guest.type == 10u ? guest.depth : 1u},
            1u,
            (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u,
            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            usage,
            VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
            VK_IMAGE_LAYOUT_UNDEFINED};
        auto result = vkCreateImage(
            presenter.device, &image_info, nullptr,
            &native.image);
        if (result != VK_SUCCESS) {
            return fail("gen5 image vkCreateImage", result);
        }
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(
            presenter.device, native.image, &requirements);
        const auto type = FindMemoryType(
            presenter, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type ==
            std::numeric_limits<std::uint32_t>::max()) {
            cleanup();
            return false;
        }
        const VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            nullptr, requirements.size, type};
        result = vkAllocateMemory(
            presenter.device, &allocation, nullptr,
            &native.memory);
        if (result == VK_SUCCESS) {
            result = vkBindImageMemory(
                presenter.device, native.image,
                native.memory, 0);
        }
        if (result != VK_SUCCESS) {
            return fail("gen5 image memory", result);
        }
        const VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            nullptr, 0, native.image,
            guest.type == 10u
                ? VK_IMAGE_VIEW_TYPE_3D
                : ((guest.type == 11u || guest.type == 13u)
                    ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                    : VK_IMAGE_VIEW_TYPE_2D),
            native.format,
            {VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY,
             VK_COMPONENT_SWIZZLE_IDENTITY},
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0,
             (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u}};
        result = vkCreateImageView(
            presenter.device, &view_info, nullptr,
            &native.view);
        if (result != VK_SUCCESS) {
            return fail("gen5 image vkCreateImageView", result);
        }
    }

    samplers.reserve(program.samplers.size());
    for (const auto& guest : program.samplers) {
        const auto word0 = guest.descriptor[0];
        const auto word2 = guest.descriptor[2];
        const auto address_mode =
            [](const std::uint32_t value) {
                if (value == 0u) {
                    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
                }
                if (value == 1u) {
                    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                }
                return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            };
        const VkSamplerCreateInfo sampler_info{
            VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            nullptr, 0,
            ((word2 >> 20u) & 3u) == 0u
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            ((word2 >> 22u) & 3u) == 0u
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            VK_SAMPLER_MIPMAP_MODE_NEAREST,
            address_mode(word0 & 7u),
            address_mode((word0 >> 3u) & 7u),
            address_mode((word0 >> 6u) & 7u),
            0.0f, VK_FALSE, 1.0f, VK_FALSE,
            VK_COMPARE_OP_ALWAYS, 0.0f, 16.0f,
            VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            VK_FALSE};
        VkSampler sampler{};
        const auto result = vkCreateSampler(
            presenter.device, &sampler_info, nullptr,
            &sampler);
        if (result != VK_SUCCESS) {
            return fail("gen5 image vkCreateSampler", result);
        }
        samplers.push_back(sampler);
    }
    bool needs_flattened{};
    bool needs_user_data{};
    for (const auto& binding : program.descriptors) {
        needs_flattened |= binding.kind == 16u;
        needs_user_data |= binding.kind == 17u;
    }
    if (needs_flattened) {
        const auto bytes =
            program.flattened_srt.size() *
            sizeof(std::uint32_t);
        if (!create_host_buffer(
                flattened, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                program.flattened_srt.empty()
                    ? nullptr
                    : program.flattened_srt.data())) {
            cleanup();
            return false;
        }
    }
    if (needs_user_data) {
        const auto bytes =
            program.packed_user_data.size() *
            sizeof(std::uint32_t);
        if (!create_host_buffer(
                user_data, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                program.packed_user_data.empty()
                    ? nullptr
                    : program.packed_user_data.data())) {
            cleanup();
            return false;
        }
    }

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    std::array<std::uint32_t, 4> pool_counts{};
    for (const auto& binding : program.descriptors) {
        VkDescriptorType type{};
        std::uint32_t count =
            static_cast<std::uint32_t>(
                binding.resources.size());
        if (binding.kind == 0u || binding.kind == 15u) {
            type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pool_counts[3] += count;
        } else if (binding.kind >= 1u && binding.kind <= 6u) {
            type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            pool_counts[0] += count;
        } else if (binding.kind >= 7u &&
                   binding.kind <= 12u) {
            type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            pool_counts[1] += count;
        } else if (binding.kind == 13u) {
            type = VK_DESCRIPTOR_TYPE_SAMPLER;
            pool_counts[2] += count;
        } else {
            type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            count = 1u;
            pool_counts[3] += 1u;
        }
        if (count != 0u) {
            layout_bindings.push_back({
                binding.binding, type, count,
                VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }
    }
    const VkDescriptorSetLayoutCreateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        nullptr, 0,
        static_cast<std::uint32_t>(layout_bindings.size()),
        layout_bindings.data()};
    auto result = vkCreateDescriptorSetLayout(
        presenter.device, &set_info, nullptr, &set_layout);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkCreateDescriptorSetLayout", result);
    }
    VkPushConstantRange push_range{
        VK_SHADER_STAGE_COMPUTE_BIT,
        program.push_constant_offset,
        program.push_constant_size};
    const VkPipelineLayoutCreateInfo pipeline_layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr, 0, 1, &set_layout,
        push_range.size == 0u ? 0u : 1u,
        push_range.size == 0u ? nullptr : &push_range};
    result = vkCreatePipelineLayout(
        presenter.device, &pipeline_layout_info, nullptr,
        &pipeline_layout);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkCreatePipelineLayout", result);
    }
    const VkShaderModuleCreateInfo module_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr, 0, spirv.size_bytes(), spirv.data()};
    result = vkCreateShaderModule(
        presenter.device, &module_info, nullptr, &module);
    if (result != VK_SUCCESS) {
        return fail("gen5 image vkCreateShaderModule", result);
    }
    const VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
        module, "main", nullptr};
    const VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr, 0, stage, pipeline_layout,
        VK_NULL_HANDLE, 0};
    result = vkCreateComputePipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkCreateComputePipelines", result);
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    constexpr std::array PoolTypes{
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    for (std::size_t index = 0;
         index < pool_counts.size(); ++index) {
        if (pool_counts[index] != 0u) {
            pool_sizes.push_back(
                {PoolTypes[index], pool_counts[index]});
        }
    }
    const VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr, 0, 1,
        static_cast<std::uint32_t>(pool_sizes.size()),
        pool_sizes.data()};
    result = vkCreateDescriptorPool(
        presenter.device, &pool_info, nullptr,
        &descriptor_pool);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkCreateDescriptorPool", result);
    }
    const VkDescriptorSetAllocateInfo allocation_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr, descriptor_pool, 1, &set_layout};
    VkDescriptorSet descriptor_set{};
    result = vkAllocateDescriptorSets(
        presenter.device, &allocation_info,
        &descriptor_set);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkAllocateDescriptorSets", result);
    }
    std::vector<std::vector<VkDescriptorImageInfo>>
        descriptor_image_infos;
    std::vector<VkDescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<VkWriteDescriptorSet> writes;
    descriptor_image_infos.reserve(program.descriptors.size());
    descriptor_buffer_infos.reserve(
        program.buffers.size() + program.addresses.size() +
        program.descriptors.size());
    writes.reserve(program.descriptors.size());
    for (const auto& binding : program.descriptors) {
        if (binding.kind == 0u || binding.kind == 15u ||
            binding.kind == 16u || binding.kind == 17u) {
            const auto first =
                descriptor_buffer_infos.size();
            if (binding.kind == 0u) {
                for (const auto resource : binding.resources) {
                    if (resource >= guest_buffers.size()) {
                        cleanup();
                        return false;
                    }
                    const auto& buffer =
                        guest_buffers[resource];
                    descriptor_buffer_infos.push_back(
                        {buffer.buffer, 0, buffer.size});
                }
            } else if (binding.kind == 15u) {
                for (const auto resource : binding.resources) {
                    if (resource >= address_buffers.size()) {
                        cleanup();
                        return false;
                    }
                    const auto& buffer =
                        address_buffers[resource];
                    descriptor_buffer_infos.push_back(
                        {buffer.buffer, 0, buffer.size});
                }
            } else {
                const auto& buffer =
                    binding.kind == 16u
                    ? flattened : user_data;
                descriptor_buffer_infos.push_back(
                    {buffer.buffer, 0, buffer.size});
            }
            const auto count = static_cast<std::uint32_t>(
                descriptor_buffer_infos.size() - first);
            if (count == 0u) {
                continue;
            }
            writes.push_back({
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr, descriptor_set, binding.binding, 0,
                count,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                descriptor_buffer_infos.data() + first,
                nullptr});
            continue;
        }
        descriptor_image_infos.emplace_back();
        auto& infos = descriptor_image_infos.back();
        infos.reserve(binding.resources.size());
        if (binding.kind == 13u) {
            for (const auto resource : binding.resources) {
                if (resource >= samplers.size()) {
                    cleanup();
                    return false;
                }
                infos.push_back({
                    samplers[resource], VK_NULL_HANDLE,
                    VK_IMAGE_LAYOUT_UNDEFINED});
            }
        } else {
            for (const auto resource : binding.resources) {
                if (resource >= images.size()) {
                    cleanup();
                    return false;
                }
                infos.push_back({
                    VK_NULL_HANDLE, images[resource].view,
                    VK_IMAGE_LAYOUT_GENERAL});
            }
        }
        if (!infos.empty()) {
            const auto type =
                binding.kind <= 6u
                ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                : (binding.kind <= 12u
                    ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    : VK_DESCRIPTOR_TYPE_SAMPLER);
            writes.push_back({
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr, descriptor_set, binding.binding, 0,
                static_cast<std::uint32_t>(infos.size()), type,
                infos.data(), nullptr, nullptr});
        }
    }
    vkUpdateDescriptorSets(
        presenter.device,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    const VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, presenter.command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    result = vkAllocateCommandBuffers(
        presenter.device, &command_info, &command_buffer);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkAllocateCommandBuffers", result);
    }
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    result = vkCreateFence(
        presenter.device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 image vkCreateFence", result);
    }
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr};
    result = vkBeginCommandBuffer(command_buffer, &begin);
    if (result != VK_SUCCESS) {
        return fail(
            "gen5 image vkBeginCommandBuffer", result);
    }
    for (std::size_t index = 0;
         index < images.size(); ++index) {
        const auto& guest = program.images[index];
        const auto& native = images[index];
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            native.staging_offset, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
             (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u},
            {0, 0, 0},
            {guest.width, guest.height,
             guest.type == 10u ? guest.depth : 1u}};
        vkCmdCopyBufferToImage(
            command_buffer, staging.buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 image upload end", result);
    }
    const VkSubmitInfo upload_submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
        0, nullptr, nullptr, 1, &command_buffer,
        0, nullptr};
    result = vkQueueSubmit(
        presenter.queue, 1, &upload_submit, fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 image upload submit", result);
    }
    result = vkWaitForFences(
        presenter.device, 1, &fence, VK_TRUE,
        UINT64_C(10000000000));
    if (result != VK_SUCCESS) {
        return fail("gen5 image upload wait", result);
    }
    result = vkResetFences(presenter.device, 1, &fence);
    if (result == VK_SUCCESS) {
        result = vkResetCommandBuffer(command_buffer, 0);
    }
    if (result == VK_SUCCESS) {
        result = vkBeginCommandBuffer(command_buffer, &begin);
    }
    if (result != VK_SUCCESS) {
        return fail("gen5 image dispatch begin", result);
    }
    if (program.push_constant_size != 0u &&
        program.packed_user_data.size() *
                sizeof(std::uint32_t) !=
            program.push_constant_size) {
        cleanup();
        return false;
    }
    const std::uint64_t total_groups =
        static_cast<std::uint64_t>(groups[0]) *
        groups[1] * groups[2];
    const bool split_dispatch =
        spirv.size() >= 20000u && total_groups > 4096u;
    std::uint32_t rows_per_dispatch = groups[1];
    std::uint32_t layers_per_dispatch = groups[2];
    if (split_dispatch) {
        const std::uint64_t plane_groups =
            static_cast<std::uint64_t>(groups[0]) *
            groups[1];
        if (plane_groups <= 4096u) {
            layers_per_dispatch = std::max(
                1u, std::min(
                        groups[2],
                        static_cast<std::uint32_t>(
                            4096u / plane_groups)));
        } else {
            rows_per_dispatch = std::max(
                1u, std::min(
                        groups[1],
                        4096u /
                            std::max(1u, groups[0])));
            layers_per_dispatch = 1u;
        }
    }
    if (split_dispatch) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-GEN5",
            "split image dispatch spirv_words=%zu "
            "groups=%u/%u/%u rows=%u layers=%u",
            spirv.size(), groups[0], groups[1], groups[2],
            rows_per_dispatch, layers_per_dispatch);
    }
    std::uint32_t base_z = 0u;
    bool dispatch_complete = false;
    do {
        std::uint32_t base_y = 0u;
        do {
        vkCmdBindPipeline(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline);
        vkCmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout, program.descriptor_set, 1,
            &descriptor_set, 0, nullptr);
        if (program.push_constant_size != 0u) {
            vkCmdPushConstants(
                command_buffer, pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                program.push_constant_offset,
                program.push_constant_size,
                program.packed_user_data.data());
        }
        const std::uint32_t row_count =
            std::min(rows_per_dispatch, groups[1] - base_y);
        const std::uint32_t layer_count =
            std::min(
                layers_per_dispatch, groups[2] - base_z);
        if (split_dispatch) {
            vkCmdDispatchBase(
                command_buffer, 0u, base_y, base_z,
                groups[0], row_count, layer_count);
        } else {
            vkCmdDispatch(
                command_buffer, groups[0], groups[1],
                groups[2]);
        }
        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            return fail("gen5 image dispatch end", result);
        }
        const VkSubmitInfo dispatch_submit{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
            0, nullptr, nullptr, 1, &command_buffer,
            0, nullptr};
        result = vkQueueSubmit(
            presenter.queue, 1, &dispatch_submit, fence);
        if (result != VK_SUCCESS) {
            return fail("gen5 image dispatch submit", result);
        }
        result = vkWaitForFences(
            presenter.device, 1, &fence, VK_TRUE,
            UINT64_C(10000000000));
        if (result != VK_SUCCESS) {
            return fail("gen5 image dispatch wait", result);
        }
        base_y += row_count;
        dispatch_complete =
            base_y >= groups[1] &&
            base_z + layer_count >= groups[2];
        result = vkResetFences(presenter.device, 1, &fence);
        if (result == VK_SUCCESS) {
            result =
                vkResetCommandBuffer(command_buffer, 0);
        }
        if (result == VK_SUCCESS) {
            result =
                vkBeginCommandBuffer(command_buffer, &begin);
        }
        if (result != VK_SUCCESS) {
            return fail(
                !dispatch_complete
                    ? "gen5 image next dispatch begin"
                    : "gen5 image readback begin",
                result);
        }
        } while (base_y < groups[1]);
        base_z += std::min(
            layers_per_dispatch, groups[2] - base_z);
    } while (base_z < groups[2]);
    for (std::size_t index = 0;
         index < images.size(); ++index) {
        if (!program.images[index].written) {
            continue;
        }
        const auto& guest = program.images[index];
        const auto& native = images[index];
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            native.staging_offset, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
             (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u},
            {0, 0, 0},
            {guest.width, guest.height,
             guest.type == 10u ? guest.depth : 1u}};
        vkCmdCopyImageToBuffer(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            staging.buffer, 1, &copy);
        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            return fail("gen5 image output end", result);
        }
        const VkSubmitInfo output_submit{
            VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
            0, nullptr, nullptr, 1, &command_buffer,
            0, nullptr};
        result = vkQueueSubmit(
            presenter.queue, 1, &output_submit, fence);
        if (result != VK_SUCCESS) {
            return fail("gen5 image output submit", result);
        }
        result = vkWaitForFences(
            presenter.device, 1, &fence, VK_TRUE,
            UINT64_C(10000000000));
        if (result != VK_SUCCESS) {
            return fail("gen5 image output wait", result);
        }
        result = vkResetFences(presenter.device, 1, &fence);
        if (result == VK_SUCCESS) {
            result = vkResetCommandBuffer(command_buffer, 0);
        }
        if (result == VK_SUCCESS) {
            result = vkBeginCommandBuffer(
                command_buffer, &begin);
        }
        if (result != VK_SUCCESS) {
            return fail("gen5 image next output begin", result);
        }
    }
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 image readback finish", result);
    }
    bool copied = true;
    for (std::size_t index = 0;
         index < program.buffers.size(); ++index) {
        const auto& guest = program.buffers[index];
        if (guest.written &&
            !write_bytes(
                memory_context, guest.guest_address,
                guest_buffers[index].map,
                static_cast<std::size_t>(
                    guest.byte_count))) {
            copied = false;
        }
    }
    for (std::size_t index = 0;
         index < images.size(); ++index) {
        const auto& guest = program.images[index];
        const auto& native = images[index];
        if (!guest.written) {
            continue;
        }
        {
            const auto* const linear =
                static_cast<const std::uint8_t*>(staging.map) +
                native.staging_offset;
            const auto stride =
                std::max<std::size_t>(
                    1u, native.linear_size / 4096u);
            std::size_t sampled{};
            std::size_t nonzero{};
            std::uint8_t maximum{};
            for (std::size_t offset = 0;
                 offset < native.linear_size;
                 offset += stride) {
                const auto byte = linear[offset];
                ++sampled;
                nonzero += byte != 0u ? 1u : 0u;
                maximum = std::max(maximum, byte);
            }
            static std::atomic<std::uint32_t> output_logs{};
            const auto output_index = output_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (output_index < 128u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-GEN5",
                    "compute output n=%u addr=0x%llx "
                    "fmt=%u size=%ux%ux%u bytes=%zu "
                    "nonzero=%zu/%zu max=%u",
                    output_index + 1u,
                    static_cast<unsigned long long>(
                        guest.guest_address),
                    guest.format, guest.width, guest.height,
                    guest.depth, native.linear_size,
                    nonzero, sampled,
                    static_cast<unsigned>(maximum));
            }
        }
        std::vector<std::uint8_t> tiled(native.guest_size);
        bool transformed = true;
        for (std::uint32_t slice = 0;
             slice < guest.depth; ++slice) {
            const auto linear_slice =
                std::span<const std::uint8_t>{
                    static_cast<const std::uint8_t*>(
                        staging.map) +
                        native.staging_offset +
                        slice * native.linear_slice_size,
                    native.linear_slice_size};
            auto tiled_slice = std::span<std::uint8_t>{
                tiled.data() +
                    slice * native.guest_slice_size,
                native.guest_slice_size};
            transformed &=
                TransformGen5GuestImage(
                    false, linear_slice, tiled_slice,
                    guest.tile_mode, native.element_width,
                    native.element_height,
                    native.element_bytes);
        }
        if (!transformed || !write_bytes(
                memory_context, guest.guest_address,
                tiled.data(), tiled.size())) {
            copied = false;
        }
    }
    cleanup();
    return copied;
}

bool ExecuteVulkanGen5Graphics(
    const std::span<const std::uint32_t> vertex_spirv,
    const Gen5ComputeProgramInfo& vertex_program,
    const std::span<const std::uint32_t> pixel_spirv,
    const Gen5ComputeProgramInfo& pixel_program,
    const VulkanGen5GraphicsDraw& draw,
    const VulkanGen5ReadBytes read_bytes,
    const VulkanGen5WriteBytes write_bytes,
    void* const memory_context) {
    const auto missing_vertex_parameters =
        pixel_program.parameter_input_mask &
        ~vertex_program.parameter_output_mask;
    const auto use_fullscreen_triangle =
        missing_vertex_parameters == 1u &&
        draw.vertex_count == 3u && !draw.indexed &&
        vertex_program.buffers.empty() &&
        vertex_program.images.empty() &&
        vertex_program.samplers.empty() &&
        vertex_program.addresses.empty() &&
        vertex_program.descriptors.empty();
    const std::span<const std::uint32_t> effective_vertex_spirv =
        use_fullscreen_triangle
        ? std::span<const std::uint32_t>{
              VulkanFullscreenTriangleShader}
        : vertex_spirv;
    const auto reject = [&](const char* const reason) {
        static std::atomic<std::uint32_t> reject_logs{};
        const auto index =
            reject_logs.fetch_add(1u, std::memory_order_relaxed);
        if (index < 192u) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4-PS5-GFX",
                "graphics reject n=%u reason=%s target=0x%llx "
                "%ux%u images=%zu samplers=%zu addresses=%zu "
                "vin=%08x vout=%08x pin=%08x",
                index + 1u, reason,
                static_cast<unsigned long long>(
                    draw.target_address),
                draw.target_width, draw.target_height,
                pixel_program.images.size(),
                pixel_program.samplers.size(),
                pixel_program.addresses.size(),
                vertex_program.parameter_input_mask,
                vertex_program.parameter_output_mask,
                pixel_program.parameter_input_mask);
        }
        return false;
    };
    const auto storage_output = std::ranges::find_if(
        pixel_program.images,
        [](const Gen5ImageBinding& image) {
            return image.written || image.atomic;
        });
    const auto has_color_target =
        draw.target_address != 0u &&
        draw.target_width != 0u &&
        draw.target_height != 0u;
    const auto render_width = has_color_target
        ? draw.target_width
        : storage_output != pixel_program.images.end()
        ? storage_output->width : 0u;
    const auto render_height = has_color_target
        ? draw.target_height
        : storage_output != pixel_program.images.end()
        ? storage_output->height : 0u;
    if (vertex_spirv.empty() || pixel_spirv.empty() ||
        render_width == 0u || render_height == 0u ||
        render_width > 8192u || render_height > 8192u ||
        draw.vertex_count == 0u ||
        read_bytes == nullptr || write_bytes == nullptr ||
        !vertex_program.images.empty() ||
        !vertex_program.samplers.empty() ||
        !vertex_program.addresses.empty() ||
        (missing_vertex_parameters != 0u &&
         !use_fullscreen_triangle)) {
        return reject("contract");
    }
    for (const auto& binding : vertex_program.descriptors) {
        if (binding.kind != 0u &&
            binding.kind != 16u && binding.kind != 17u) {
            return reject("vertex-descriptor");
        }
    }
    for (const auto& binding : pixel_program.descriptors) {
        if (!(binding.kind == 0u ||
              (binding.kind >= 1u && binding.kind <= 13u) ||
              binding.kind == 15u ||
              binding.kind == 16u || binding.kind == 17u)) {
            return reject("pixel-descriptor");
        }
    }
    const std::lock_guard lock{g_mutex};
    auto& presenter = g_presenter;
    if (presenter.device == VK_NULL_HANDLE ||
        presenter.queue == VK_NULL_HANDLE ||
        presenter.command_pool == VK_NULL_HANDLE ||
        presenter.permanently_failed) {
        return reject("presenter");
    }

    struct NativeBuffer {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        void* map{};
        VkDeviceSize size{};
    };
    struct NativeImage {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        std::uint32_t element_bytes{};
        std::uint32_t element_width{};
        std::uint32_t element_height{};
        std::size_t staging_offset{};
        std::size_t linear_size{};
        std::size_t guest_size{};
    };
    std::vector<NativeImage> images(pixel_program.images.size());
    std::vector<VkSampler> samplers;
    std::vector<NativeBuffer> vertex_buffers(
        vertex_program.buffers.size());
    std::vector<NativeBuffer> pixel_buffers(
        pixel_program.buffers.size());
    NativeImage target{};
    NativeBuffer staging{};
    NativeBuffer vertex_flattened{};
    NativeBuffer vertex_user_data{};
    NativeBuffer pixel_flattened{};
    NativeBuffer pixel_user_data{};
    std::vector<NativeBuffer> pixel_addresses(
        pixel_program.addresses.size());
    NativeBuffer index_buffer{};
    VkShaderModule vertex_module{};
    VkShaderModule pixel_module{};
    std::array<VkDescriptorSetLayout, 2> set_layouts{};
    VkPipelineLayout pipeline_layout{};
    VkRenderPass render_pass{};
    VkFramebuffer framebuffer{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptor_pool{};
    VkDescriptorSet vertex_set{};
    VkDescriptorSet pixel_set{};
    VkCommandBuffer command_buffer{};
    VkFence fence{};

    const auto destroy_buffer = [&](NativeBuffer& buffer) {
        if (buffer.map != nullptr) {
            vkUnmapMemory(presenter.device, buffer.memory);
        }
        if (buffer.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(presenter.device, buffer.buffer, nullptr);
        }
        if (buffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(presenter.device, buffer.memory, nullptr);
        }
        buffer = {};
    };
    const auto destroy_image = [&](NativeImage& image) {
        if (image.view != VK_NULL_HANDLE) {
            vkDestroyImageView(presenter.device, image.view, nullptr);
        }
        if (image.image != VK_NULL_HANDLE) {
            vkDestroyImage(presenter.device, image.image, nullptr);
        }
        if (image.memory != VK_NULL_HANDLE) {
            vkFreeMemory(presenter.device, image.memory, nullptr);
        }
        image = {};
    };
    const auto cleanup = [&] {
        if (fence != VK_NULL_HANDLE) {
            vkDestroyFence(presenter.device, fence, nullptr);
        }
        if (command_buffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(
                presenter.device, presenter.command_pool, 1,
                &command_buffer);
        }
        if (descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(
                presenter.device, descriptor_pool, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(presenter.device, pipeline, nullptr);
        }
        if (framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(presenter.device, framebuffer, nullptr);
        }
        if (render_pass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(presenter.device, render_pass, nullptr);
        }
        if (pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                presenter.device, pipeline_layout, nullptr);
        }
        for (const auto layout : set_layouts) {
            if (layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(
                    presenter.device, layout, nullptr);
            }
        }
        if (vertex_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(
                presenter.device, vertex_module, nullptr);
        }
        if (pixel_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(
                presenter.device, pixel_module, nullptr);
        }
        for (const auto sampler : samplers) {
            vkDestroySampler(presenter.device, sampler, nullptr);
        }
        for (auto& image : images) {
            destroy_image(image);
        }
        for (auto& buffer : vertex_buffers) {
            destroy_buffer(buffer);
        }
        for (auto& buffer : pixel_buffers) {
            destroy_buffer(buffer);
        }
        destroy_image(target);
        destroy_buffer(staging);
        destroy_buffer(vertex_flattened);
        destroy_buffer(vertex_user_data);
        destroy_buffer(pixel_flattened);
        destroy_buffer(pixel_user_data);
        for (auto& buffer : pixel_addresses) {
            destroy_buffer(buffer);
        }
        destroy_buffer(index_buffer);
    };
    const auto fail = [&](const char* operation, const VkResult result) {
        LogFailure(operation, result);
        cleanup();
        return false;
    };
    const auto create_buffer =
        [&](NativeBuffer& buffer, VkDeviceSize size,
            VkBufferUsageFlags usage, const void* initial_data) {
            buffer.size = std::max<VkDeviceSize>(size, 4u);
            const VkBufferCreateInfo info{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                buffer.size, usage, VK_SHARING_MODE_EXCLUSIVE,
                0, nullptr};
            auto result = vkCreateBuffer(
                presenter.device, &info, nullptr, &buffer.buffer);
            if (result != VK_SUCCESS) {
                return false;
            }
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(
                presenter.device, buffer.buffer, &requirements);
            const auto type = FindMemoryType(
                presenter, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (type == std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            const VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                requirements.size, type};
            result = vkAllocateMemory(
                presenter.device, &allocation, nullptr, &buffer.memory);
            if (result == VK_SUCCESS) {
                result = vkBindBufferMemory(
                    presenter.device, buffer.buffer, buffer.memory, 0);
            }
            if (result == VK_SUCCESS) {
                result = vkMapMemory(
                    presenter.device, buffer.memory, 0,
                    buffer.size, 0, &buffer.map);
            }
            if (result != VK_SUCCESS || buffer.map == nullptr) {
                return false;
            }
            std::memset(
                buffer.map, 0, static_cast<std::size_t>(buffer.size));
            if (initial_data != nullptr && size != 0u) {
                std::memcpy(
                    buffer.map, initial_data,
                    static_cast<std::size_t>(size));
            }
            return true;
        };
    const auto create_guest_buffers =
        [&](const Gen5ComputeProgramInfo& program,
            std::vector<NativeBuffer>& buffers) {
            for (std::size_t index = 0;
                 index < program.buffers.size(); ++index) {
                const auto& guest = program.buffers[index];
                if (guest.guest_address == 0u ||
                    guest.byte_count == 0u ||
                    guest.byte_count > UINT32_C(0x10000000) ||
                    !create_buffer(
                        buffers[index], guest.byte_count,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        nullptr) ||
                    (guest.read &&
                     !read_bytes(
                         memory_context, guest.guest_address,
                         buffers[index].map,
                         static_cast<std::size_t>(
                             guest.byte_count)))) {
                    return false;
                }
            }
            return true;
        };
    if (!create_guest_buffers(vertex_program, vertex_buffers) ||
        !create_guest_buffers(pixel_program, pixel_buffers)) {
        cleanup();
        return reject("guest-buffer");
    }
    const auto buffer_backed =
        !vertex_program.buffers.empty() ||
        !pixel_program.buffers.empty();
    std::uint32_t buffer_log_index{UINT32_MAX};
    if (buffer_backed) {
        static std::atomic<std::uint32_t> buffer_pass_logs{};
        buffer_log_index = buffer_pass_logs.fetch_add(
            1u, std::memory_order_relaxed);
        if (buffer_log_index < 32u) {
            const auto log_buffers =
                [&](const char* const stage,
                    const Gen5ComputeProgramInfo& program,
                    const std::vector<NativeBuffer>& buffers) {
                    for (std::size_t index = 0;
                         index < program.buffers.size(); ++index) {
                        const auto& guest = program.buffers[index];
                        const auto* const bytes =
                            static_cast<const std::uint8_t*>(
                                buffers[index].map);
                        const auto size = static_cast<std::size_t>(
                            guest.byte_count);
                        const auto stride =
                            std::max<std::size_t>(size / 512u, 1u);
                        std::size_t sampled{};
                        std::size_t nonzero{};
                        for (std::size_t offset = 0;
                             offset < size; offset += stride) {
                            nonzero += bytes[offset] != 0u ? 1u : 0u;
                            ++sampled;
                        }
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4-PS5-GFX",
                            "buffer pass=%u stage=%s i=%zu "
                            "addr=0x%llx bytes=%llu rw=%u/%u/%u "
                            "nonzero=%zu/%zu",
                            buffer_log_index + 1u, stage, index,
                            static_cast<unsigned long long>(
                                guest.guest_address),
                            static_cast<unsigned long long>(
                                guest.byte_count),
                            guest.read ? 1u : 0u,
                            guest.written ? 1u : 0u,
                            guest.atomic ? 1u : 0u,
                            nonzero, sampled);
                    }
                };
            log_buffers("vs", vertex_program, vertex_buffers);
            log_buffers("ps", pixel_program, pixel_buffers);
        }
    }
    const auto guest_image_size =
        [&](const std::uint32_t width, const std::uint32_t height,
            const std::uint32_t tile_mode,
            const std::uint32_t bytes) -> std::size_t {
            if (tile_mode == 0u) {
                const auto pitch =
                    (static_cast<std::uint64_t>(width) * bytes + 255u) &
                    ~UINT64_C(255);
                return static_cast<std::size_t>(pitch * height);
            }
            std::array<std::uint32_t, 16> xm{};
            std::array<std::uint32_t, 16> ym{};
            std::uint32_t bits{};
            std::uint32_t block_bytes{};
            if (!GetGen5SwizzlePattern(
                    tile_mode, bytes, xm, ym, bits, block_bytes)) {
                return 0u;
            }
            const auto elements = block_bytes / bytes;
            const auto total_bits =
                static_cast<std::uint32_t>(__builtin_ctz(elements));
            const auto block_width = 1u << ((total_bits + 1u) / 2u);
            const auto block_height = 1u << (total_bits / 2u);
            return static_cast<std::size_t>(
                ((static_cast<std::uint64_t>(width) +
                  block_width - 1u) / block_width) *
                ((static_cast<std::uint64_t>(height) +
                  block_height - 1u) / block_height) * block_bytes);
        };

    std::size_t staging_size{};
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& guest = pixel_program.images[index];
        const auto null_image =
            guest.type == 0u && guest.guest_address == 0u;
        if ((!null_image &&
             guest.type != 9u && guest.type != 10u &&
             guest.type != 11u &&
             guest.type != 13u) ||
            guest.width == 0u || guest.height == 0u ||
            guest.width > 8192u || guest.height > 8192u ||
            guest.depth == 0u || guest.depth > 256u) {
            cleanup();
            return reject("image-contract");
        }
        bool force_uint{};
        for (const auto& binding : pixel_program.descriptors) {
            if (binding.kind >= 10u && binding.kind <= 12u &&
                std::ranges::find(
                    binding.resources,
                    static_cast<std::uint32_t>(index)) !=
                    binding.resources.end()) {
                force_uint = true;
            }
        }
        const auto format = null_image
            ? Gen5GuestFormat{
                  force_uint ? VK_FORMAT_R32_UINT
                             : VK_FORMAT_R8G8B8A8_UNORM,
                  4u}
            : GetGen5GuestFormat(
                  guest.format, force_uint);
        if (format.vulkan == VK_FORMAT_UNDEFINED ||
            format.element_bytes == 0u) {
            cleanup();
            return reject("image-format");
        }
        auto& native = images[index];
        native.format = format.vulkan;
        native.element_bytes = format.element_bytes;
        native.element_width =
            (guest.width + format.block_width - 1u) /
            format.block_width;
        native.element_height =
            (guest.height + format.block_height - 1u) /
            format.block_height;
        native.linear_size =
            static_cast<std::size_t>(native.element_width) *
            native.element_height * guest.depth *
            format.element_bytes;
        native.guest_size = null_image
            ? native.linear_size
            : guest_image_size(
                native.element_width, native.element_height,
                guest.tile_mode,
                format.element_bytes) * guest.depth;
        if (native.guest_size == 0u ||
            native.guest_size > UINT32_C(0x20000000) ||
            native.linear_size > UINT32_C(0x20000000)) {
            cleanup();
            return reject("image-size");
        }
        staging_size = (staging_size + 255u) & ~std::size_t{255u};
        native.staging_offset = staging_size;
        staging_size += native.linear_size;
    }
    auto target_format = has_color_target
        ? GetGen5RenderTargetFormat(
              draw.target_format, draw.target_number_type,
              draw.target_channel_order)
        : Gen5GuestFormat{
              VK_FORMAT_R8G8B8A8_UNORM, 4u};
    if (target_format.vulkan == VK_FORMAT_UNDEFINED ||
        target_format.element_bytes == 0u) {
        target_format = {
            VK_FORMAT_R8G8B8A8_UNORM, 4u};
    }
    target.format = target_format.vulkan;
    target.element_bytes = target_format.element_bytes;
    target.linear_size =
        static_cast<std::size_t>(render_width) *
        render_height * target.element_bytes;
    target.guest_size = guest_image_size(
        render_width, render_height,
        has_color_target ? draw.target_tile_mode : 0u,
        target.element_bytes);
    if (target.guest_size == 0u ||
        target.guest_size > UINT32_C(0x20000000)) {
        cleanup();
        return reject("target-size");
    }
    staging_size = (staging_size + 255u) & ~std::size_t{255u};
    target.staging_offset = staging_size;
    staging_size += target.linear_size;
    if (!create_buffer(
            staging, staging_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            nullptr)) {
        cleanup();
        return reject("staging-buffer");
    }

    const auto upload_guest_image =
        [&](const Gen5ImageBinding& guest, NativeImage& native) {
            if (guest.guest_address == 0u && guest.type == 0u) {
                return true;
            }
            std::vector<std::uint8_t> tiled(native.guest_size);
            if (!read_bytes(
                    memory_context, guest.guest_address,
                    tiled.data(), tiled.size())) {
                return false;
            }
            const auto guest_slice = native.guest_size / guest.depth;
            const auto linear_slice =
                native.linear_size / guest.depth;
            for (std::uint32_t slice = 0;
                 slice < guest.depth; ++slice) {
                if (!TransformGen5GuestImage(
                        true,
                        std::span<const std::uint8_t>{
                            tiled.data() + slice * guest_slice,
                            guest_slice},
                        std::span<std::uint8_t>{
                            static_cast<std::uint8_t*>(staging.map) +
                                native.staging_offset +
                                slice * linear_slice,
                            linear_slice},
                        guest.tile_mode, native.element_width,
                        native.element_height,
                        native.element_bytes)) {
                    return false;
                }
            }
            return true;
        };
    for (std::size_t index = 0; index < images.size(); ++index) {
        if (!upload_guest_image(
                pixel_program.images[index], images[index])) {
            cleanup();
            return reject("image-upload");
        }
        static std::atomic<std::uint32_t> image_input_logs{};
        const auto log_index =
            image_input_logs.fetch_add(1u, std::memory_order_relaxed);
        if (log_index < 128u) {
            const auto& guest = pixel_program.images[index];
            const auto& native = images[index];
            const auto* const bytes =
                static_cast<const std::uint8_t*>(staging.map) +
                native.staging_offset;
            const auto stride =
                std::max<std::size_t>(
                    native.linear_size / 1024u, 1u);
            std::size_t nonzero{};
            std::size_t sampled{};
            for (std::size_t offset = 0;
                 offset < native.linear_size; offset += stride) {
                nonzero += bytes[offset] != 0u ? 1u : 0u;
                ++sampled;
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-GFX",
                "image input n=%u addr=0x%llx %ux%u "
                "fmt=%u tile=%u nonzero=%zu/%zu",
                log_index + 1u,
                static_cast<unsigned long long>(
                    guest.guest_address),
                guest.width, guest.height, guest.format,
                guest.tile_mode, nonzero, sampled);
        }
    }
    {
        std::vector<std::uint8_t> tiled(target.guest_size);
        if (has_color_target && read_bytes(
                memory_context, draw.target_address,
                tiled.data(), tiled.size())) {
            (void)TransformGen5GuestImage(
                true, tiled,
                std::span<std::uint8_t>{
                    static_cast<std::uint8_t*>(staging.map) +
                        target.staging_offset,
                    target.linear_size},
                draw.target_tile_mode, render_width,
                render_height, target.element_bytes);
        }
    }

    const auto create_image =
        [&](NativeImage& native, std::uint32_t width,
            std::uint32_t height, std::uint32_t depth,
            std::uint32_t layers,
            VkImageUsageFlags usage) {
            const VkImageCreateInfo image_info{
                VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0,
                depth > 1u ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                native.format,
                {width, height, depth}, 1u, layers,
                VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr,
                VK_IMAGE_LAYOUT_UNDEFINED};
            auto result = vkCreateImage(
                presenter.device, &image_info, nullptr, &native.image);
            if (result != VK_SUCCESS) {
                return false;
            }
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(
                presenter.device, native.image, &requirements);
            const auto type = FindMemoryType(
                presenter, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type == std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }
            const VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr,
                requirements.size, type};
            result = vkAllocateMemory(
                presenter.device, &allocation, nullptr, &native.memory);
            if (result == VK_SUCCESS) {
                result = vkBindImageMemory(
                    presenter.device, native.image, native.memory, 0);
            }
            if (result != VK_SUCCESS) {
                return false;
            }
            const VkImageViewCreateInfo view_info{
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0,
                native.image,
                depth > 1u
                    ? VK_IMAGE_VIEW_TYPE_3D
                    : (layers > 1u
                        ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                        : VK_IMAGE_VIEW_TYPE_2D),
                native.format,
                {VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY,
                 VK_COMPONENT_SWIZZLE_IDENTITY},
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
            return vkCreateImageView(
                       presenter.device, &view_info, nullptr,
                       &native.view) == VK_SUCCESS;
        };
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& guest = pixel_program.images[index];
        VkImageUsageFlags usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (guest.read) {
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (guest.written || guest.atomic) {
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if (!create_image(
                images[index], guest.width, guest.height,
                guest.type == 10u ? guest.depth : 1u,
                (guest.type == 11u || guest.type == 13u)
                    ? guest.depth : 1u,
                usage)) {
            cleanup();
            return reject("image-create");
        }
    }
    if (!create_image(
            target, render_width, render_height, 1u, 1u,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        cleanup();
        return reject("target-create");
    }

    samplers.reserve(pixel_program.samplers.size());
    for (const auto& guest : pixel_program.samplers) {
        const auto word0 = guest.descriptor[0];
        const auto word2 = guest.descriptor[2];
        const auto address_mode = [](std::uint32_t value) {
            return value == 0u ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                : value == 1u
                ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
                : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        };
        const VkSamplerCreateInfo info{
            VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, nullptr, 0,
            ((word2 >> 20u) & 3u) == 0u
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            ((word2 >> 22u) & 3u) == 0u
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
            VK_SAMPLER_MIPMAP_MODE_NEAREST,
            address_mode(word0 & 7u),
            address_mode((word0 >> 3u) & 7u),
            address_mode((word0 >> 6u) & 7u),
            0.0f, VK_FALSE, 1.0f, VK_FALSE,
            VK_COMPARE_OP_ALWAYS, 0.0f, 16.0f,
            VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
        VkSampler sampler{};
        if (vkCreateSampler(
                presenter.device, &info, nullptr, &sampler) !=
            VK_SUCCESS) {
            cleanup();
            return reject("sampler-create");
        }
        samplers.push_back(sampler);
    }

    const auto create_program_buffers =
        [&](const Gen5ComputeProgramInfo& program,
            NativeBuffer& flattened, NativeBuffer& user_data) {
            for (const auto& binding : program.descriptors) {
                if (binding.kind == 16u &&
                    flattened.buffer == VK_NULL_HANDLE &&
                    !create_buffer(
                        flattened,
                        program.flattened_srt.size() *
                            sizeof(std::uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        program.flattened_srt.data())) {
                    return false;
                }
                if (binding.kind == 17u &&
                    user_data.buffer == VK_NULL_HANDLE &&
                    !create_buffer(
                        user_data,
                        program.packed_user_data.size() *
                            sizeof(std::uint32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        program.packed_user_data.data())) {
                    return false;
                }
            }
            return true;
        };
    if (!create_program_buffers(
            vertex_program, vertex_flattened, vertex_user_data) ||
        !create_program_buffers(
            pixel_program, pixel_flattened, pixel_user_data)) {
        cleanup();
        return reject("program-buffer");
    }
    for (std::size_t index = 0;
         index < pixel_program.addresses.size(); ++index) {
        const auto& guest = pixel_program.addresses[index];
        constexpr std::size_t FlatWindow = 0x10000u;
        if (guest.binding_base == 0u || guest.written ||
            !create_buffer(
                pixel_addresses[index], FlatWindow,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr)) {
            cleanup();
            return reject("address-buffer");
        }
        if (!read_bytes(
                memory_context, guest.binding_base,
                pixel_addresses[index].map, FlatWindow)) {
            bool read_any{};
            for (std::size_t size = 0x8000u;
                 size >= 0x100u; size >>= 1u) {
                if (read_bytes(
                        memory_context, guest.binding_base,
                        pixel_addresses[index].map, size)) {
                    read_any = true;
                    break;
                }
            }
            if (!read_any) {
                cleanup();
                return reject("address-read");
            }
        }
    }
    if (draw.indexed) {
        const auto index_bytes =
            draw.index_size == 4u ? 4u : 2u;
        const auto size =
            static_cast<std::size_t>(draw.vertex_count) * index_bytes;
        if (draw.index_address == 0u ||
            !create_buffer(
                index_buffer, size,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT, nullptr) ||
            !read_bytes(
                memory_context, draw.index_address,
                index_buffer.map, size)) {
            cleanup();
            return reject("index-buffer");
        }
    }

    const auto make_layout_bindings =
        [](const Gen5ComputeProgramInfo& program,
           VkShaderStageFlags stage_flags,
           std::array<std::uint32_t, 4>& counts) {
            std::vector<VkDescriptorSetLayoutBinding> result;
            for (const auto& binding : program.descriptors) {
                VkDescriptorType type{};
                auto count = static_cast<std::uint32_t>(
                    binding.resources.size());
                if (binding.kind >= 1u && binding.kind <= 6u) {
                    type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    counts[0] += count;
                } else if (binding.kind >= 7u &&
                           binding.kind <= 12u) {
                    type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    counts[1] += count;
                } else if (binding.kind == 13u) {
                    type = VK_DESCRIPTOR_TYPE_SAMPLER;
                    counts[2] += count;
                } else if (binding.kind == 0u ||
                           binding.kind == 15u) {
                    type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    counts[3] += count;
                } else {
                    type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    count = 1u;
                    counts[3] += 1u;
                }
                if (count != 0u) {
                    result.push_back({
                        binding.binding, type, count,
                        stage_flags, nullptr});
                }
            }
            return result;
        };
    std::array<std::uint32_t, 4> pool_counts{};
    const auto vertex_bindings = make_layout_bindings(
        vertex_program, VK_SHADER_STAGE_VERTEX_BIT, pool_counts);
    const auto pixel_bindings = make_layout_bindings(
        pixel_program, VK_SHADER_STAGE_FRAGMENT_BIT, pool_counts);
    const std::array binding_lists{
        std::span{vertex_bindings}, std::span{pixel_bindings}};
    for (std::size_t set = 0; set < set_layouts.size(); ++set) {
        const VkDescriptorSetLayoutCreateInfo info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            nullptr, 0,
            static_cast<std::uint32_t>(binding_lists[set].size()),
            binding_lists[set].data()};
        const auto result = vkCreateDescriptorSetLayout(
            presenter.device, &info, nullptr, &set_layouts[set]);
        if (result != VK_SUCCESS) {
            return fail("gen5 graphics descriptor layout", result);
        }
    }
    std::vector<VkPushConstantRange> push_ranges;
    if (vertex_program.push_constant_size != 0u) {
        push_ranges.push_back({
            VK_SHADER_STAGE_VERTEX_BIT,
            vertex_program.push_constant_offset,
            vertex_program.push_constant_size});
    }
    if (pixel_program.push_constant_size != 0u) {
        push_ranges.push_back({
            VK_SHADER_STAGE_FRAGMENT_BIT,
            pixel_program.push_constant_offset,
            pixel_program.push_constant_size});
    }
    const VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        static_cast<std::uint32_t>(set_layouts.size()),
        set_layouts.data(),
        static_cast<std::uint32_t>(push_ranges.size()),
        push_ranges.data()};
    auto result = vkCreatePipelineLayout(
        presenter.device, &layout_info, nullptr, &pipeline_layout);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics pipeline layout", result);
    }

    const VkAttachmentDescription attachment{
        0, target.format, VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference attachment_ref{
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkSubpassDescription subpass{
        0, VK_PIPELINE_BIND_POINT_GRAPHICS,
        0, nullptr, 1, &attachment_ref,
        nullptr, nullptr, 0, nullptr};
    const VkRenderPassCreateInfo render_pass_info{
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0,
        1, &attachment, 1, &subpass, 0, nullptr};
    result = vkCreateRenderPass(
        presenter.device, &render_pass_info, nullptr, &render_pass);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics render pass", result);
    }
    const VkFramebufferCreateInfo framebuffer_info{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0,
        render_pass, 1, &target.view,
        render_width, render_height, 1};
    result = vkCreateFramebuffer(
        presenter.device, &framebuffer_info, nullptr, &framebuffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics framebuffer", result);
    }
    const auto create_module =
        [&](std::span<const std::uint32_t> code,
            VkShaderModule& module) {
            const VkShaderModuleCreateInfo info{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                nullptr, 0, code.size_bytes(), code.data()};
            return vkCreateShaderModule(
                presenter.device, &info, nullptr, &module);
        };
    result = create_module(effective_vertex_spirv, vertex_module);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics vertex module", result);
    }
    result = create_module(pixel_spirv, pixel_module);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics pixel module", result);
    }
    const std::array stages{
        VkPipelineShaderStageCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,
            vertex_module, "main", nullptr},
        VkPipelineShaderStageCreateInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT,
            pixel_module, "main", nullptr}};
    const VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        nullptr, 0, 0, nullptr, 0, nullptr};
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    if (draw.primitive_type == 6u) {
        topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    } else if (draw.primitive_type == 5u) {
        topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    } else if (draw.primitive_type == 2u) {
        topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    }
    const VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        nullptr, 0, topology, VK_FALSE};
    const VkViewport viewport{
        0.0f, 0.0f,
        static_cast<float>(render_width),
        static_cast<float>(render_height),
        0.0f, 1.0f};
    const VkRect2D scissor{
        {0, 0}, {render_width, render_height}};
    const VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        nullptr, 0, 1, &viewport, 1, &scissor};
    const VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        nullptr, 0, VK_FALSE, VK_FALSE,
        VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
        VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE,
        0.0f, 0.0f, 0.0f, 1.0f};
    const VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        nullptr, 0, VK_SAMPLE_COUNT_1_BIT, VK_FALSE, 1.0f,
        nullptr, VK_FALSE, VK_FALSE};
    const auto blend_factor = [](const std::uint32_t factor) {
        switch (factor) {
        case 0x00u: return VK_BLEND_FACTOR_ZERO;
        case 0x01u: return VK_BLEND_FACTOR_ONE;
        case 0x02u: return VK_BLEND_FACTOR_SRC_COLOR;
        case 0x03u: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x04u: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x05u: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x06u: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x07u: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x08u: return VK_BLEND_FACTOR_DST_COLOR;
        case 0x09u: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x0au: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x0du: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x0eu: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x0fu: return VK_BLEND_FACTOR_SRC1_COLOR;
        case 0x10u: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
        case 0x11u: return VK_BLEND_FACTOR_SRC1_ALPHA;
        case 0x12u: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x13u: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x14u: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ZERO;
        }
    };
    const auto blend_op = [](const std::uint32_t operation) {
        switch (operation) {
        case 1u: return VK_BLEND_OP_SUBTRACT;
        case 2u: return VK_BLEND_OP_MIN;
        case 3u: return VK_BLEND_OP_MAX;
        case 4u: return VK_BLEND_OP_REVERSE_SUBTRACT;
        default: return VK_BLEND_OP_ADD;
        }
    };
    const auto blend_control = draw.blend_control;
    const auto separate_alpha =
        (blend_control & UINT32_C(0x20000000)) != 0u;
    VkColorComponentFlags color_write_mask{};
    if ((draw.color_write_mask & 1u) != 0u) {
        color_write_mask |= VK_COLOR_COMPONENT_R_BIT;
    }
    if ((draw.color_write_mask & 2u) != 0u) {
        color_write_mask |= VK_COLOR_COMPONENT_G_BIT;
    }
    if ((draw.color_write_mask & 4u) != 0u) {
        color_write_mask |= VK_COLOR_COMPONENT_B_BIT;
    }
    if ((draw.color_write_mask & 8u) != 0u) {
        color_write_mask |= VK_COLOR_COMPONENT_A_BIT;
    }
    const auto source_color =
        blend_factor(blend_control & 0x1fu);
    const auto destination_color =
        blend_factor((blend_control >> 8u) & 0x1fu);
    const auto color_operation =
        blend_op((blend_control >> 5u) & 0x7u);
    const VkPipelineColorBlendAttachmentState blend_attachment{
        (blend_control & UINT32_C(0x40000000)) != 0u
            ? VK_TRUE : VK_FALSE,
        source_color, destination_color, color_operation,
        separate_alpha
            ? blend_factor((blend_control >> 16u) & 0x1fu)
            : source_color,
        separate_alpha
            ? blend_factor((blend_control >> 24u) & 0x1fu)
            : destination_color,
        separate_alpha
            ? blend_op((blend_control >> 21u) & 0x7u)
            : color_operation,
        color_write_mask};
    const VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY,
        1, &blend_attachment, {0, 0, 0, 0}};
    const VkGraphicsPipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        nullptr, 0,
        static_cast<std::uint32_t>(stages.size()), stages.data(),
        &vertex_input, &assembly, nullptr, &viewport_state,
        &raster, &multisample, nullptr, &blend, nullptr,
        pipeline_layout, render_pass, 0, VK_NULL_HANDLE, -1};
    result = vkCreateGraphicsPipelines(
        presenter.device, VK_NULL_HANDLE, 1,
        &pipeline_info, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics pipeline", result);
    }

    std::vector<VkDescriptorPoolSize> pool_sizes;
    constexpr std::array PoolTypes{
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    for (std::size_t index = 0; index < pool_counts.size(); ++index) {
        if (pool_counts[index] != 0u) {
            pool_sizes.push_back({
                PoolTypes[index], pool_counts[index]});
        }
    }
    if (!vertex_bindings.empty() || !pixel_bindings.empty()) {
        const auto descriptor_set_count =
            static_cast<std::uint32_t>(
                !vertex_bindings.empty()) +
            static_cast<std::uint32_t>(
                !pixel_bindings.empty());
        const VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            nullptr, 0, descriptor_set_count,
            static_cast<std::uint32_t>(pool_sizes.size()),
            pool_sizes.data()};
        result = vkCreateDescriptorPool(
            presenter.device, &pool_info, nullptr, &descriptor_pool);
        if (result != VK_SUCCESS) {
            return fail("gen5 graphics descriptor pool", result);
        }
    }
    const auto allocate_set =
        [&](const std::size_t layout_index,
            VkDescriptorSet& set) {
            const VkDescriptorSetAllocateInfo allocate_info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                nullptr, descriptor_pool, 1,
                &set_layouts[layout_index]};
            return vkAllocateDescriptorSets(
                presenter.device, &allocate_info, &set);
        };
    if (!vertex_bindings.empty() &&
        (result = allocate_set(0u, vertex_set)) != VK_SUCCESS) {
        return fail("gen5 graphics vertex descriptor set", result);
    }
    if (!pixel_bindings.empty() &&
        (result = allocate_set(1u, pixel_set)) != VK_SUCCESS) {
        return fail("gen5 graphics pixel descriptor set", result);
    }
    const auto update_stage_set =
        [&](const Gen5ComputeProgramInfo& program,
            const std::vector<NativeBuffer>& guest_buffers,
            const std::span<const NativeBuffer> addresses,
            const NativeBuffer& flattened,
            const NativeBuffer& user_data,
            const std::span<const NativeImage> stage_images,
            const std::span<const VkSampler> stage_samplers,
            const VkDescriptorSet set) {
            std::size_t resource_count{};
            for (const auto& binding : program.descriptors) {
                resource_count += std::max<std::size_t>(
                    binding.resources.size(), 1u);
            }
            std::vector<std::vector<VkDescriptorImageInfo>>
                image_infos;
            std::vector<VkDescriptorBufferInfo> buffer_infos;
            std::vector<VkWriteDescriptorSet> writes;
            image_infos.reserve(program.descriptors.size());
            buffer_infos.reserve(resource_count);
            writes.reserve(program.descriptors.size());
            for (const auto& binding : program.descriptors) {
                if (binding.kind == 0u || binding.kind == 15u ||
                    binding.kind == 16u || binding.kind == 17u) {
                    const auto first = buffer_infos.size();
                    if (binding.kind == 0u) {
                        for (const auto resource : binding.resources) {
                            if (resource >= guest_buffers.size()) {
                                return false;
                            }
                            const auto& buffer =
                                guest_buffers[resource];
                            buffer_infos.push_back({
                                buffer.buffer, 0, buffer.size});
                        }
                    } else if (binding.kind == 15u) {
                        for (const auto resource : binding.resources) {
                            if (resource >= addresses.size()) {
                                return false;
                            }
                            const auto& buffer = addresses[resource];
                            buffer_infos.push_back({
                                buffer.buffer, 0, buffer.size});
                        }
                    } else {
                        const auto& buffer =
                            binding.kind == 16u
                            ? flattened : user_data;
                        if (buffer.buffer == VK_NULL_HANDLE) {
                            return false;
                        }
                        buffer_infos.push_back({
                            buffer.buffer, 0, buffer.size});
                    }
                    const auto count =
                        static_cast<std::uint32_t>(
                            buffer_infos.size() - first);
                    if (count != 0u) {
                        writes.push_back({
                            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            nullptr, set, binding.binding, 0, count,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
                            buffer_infos.data() + first, nullptr});
                    }
                    continue;
                }
                image_infos.emplace_back();
                auto& infos = image_infos.back();
                infos.reserve(binding.resources.size());
                for (const auto resource : binding.resources) {
                    if (binding.kind == 13u) {
                        if (resource >= stage_samplers.size()) {
                            return false;
                        }
                        infos.push_back({
                            stage_samplers[resource], VK_NULL_HANDLE,
                            VK_IMAGE_LAYOUT_UNDEFINED});
                    } else {
                        if (resource >= stage_images.size()) {
                            return false;
                        }
                        infos.push_back({
                            VK_NULL_HANDLE,
                            stage_images[resource].view,
                            VK_IMAGE_LAYOUT_GENERAL});
                    }
                }
                if (!infos.empty()) {
                    const auto type =
                        binding.kind <= 6u
                        ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                        : binding.kind <= 12u
                        ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                        : VK_DESCRIPTOR_TYPE_SAMPLER;
                    writes.push_back({
                        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        nullptr, set, binding.binding, 0,
                        static_cast<std::uint32_t>(infos.size()),
                        type, infos.data(), nullptr, nullptr});
                }
            }
            vkUpdateDescriptorSets(
                presenter.device,
                static_cast<std::uint32_t>(writes.size()),
                writes.data(), 0, nullptr);
            return true;
        };
    if (vertex_set != VK_NULL_HANDLE &&
        !update_stage_set(
            vertex_program, vertex_buffers, {},
            vertex_flattened, vertex_user_data, {}, {},
            vertex_set)) {
        cleanup();
        return reject("vertex-descriptor-update");
    }
    if (pixel_set != VK_NULL_HANDLE &&
        !update_stage_set(
            pixel_program, pixel_buffers, pixel_addresses,
            pixel_flattened, pixel_user_data, images, samplers,
            pixel_set)) {
        cleanup();
        return reject("pixel-descriptor-update");
    }

    const VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr, presenter.command_pool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    result = vkAllocateCommandBuffers(
        presenter.device, &command_info, &command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics command buffer", result);
    }
    const VkFenceCreateInfo fence_info{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
    result = vkCreateFence(
        presenter.device, &fence_info, nullptr, &fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics fence", result);
    }
    const VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    result = vkBeginCommandBuffer(command_buffer, &begin);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics begin", result);
    }
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& guest = pixel_program.images[index];
        auto& native = images[index];
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            native.staging_offset, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
             (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u},
            {0, 0, 0},
            {guest.width, guest.height,
             guest.type == 10u ? guest.depth : 1u}};
        vkCmdCopyBufferToImage(
            command_buffer, staging.buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    }
    ImageBarrier(
        command_buffer, target.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkBufferImageCopy target_copy{
        target.staging_offset, 0, 0,
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        {0, 0, 0},
        {render_width, render_height, 1}};
    vkCmdCopyBufferToImage(
        command_buffer, staging.buffer, target.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &target_copy);
    ImageBarrier(
        command_buffer, target.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    const VkRenderPassBeginInfo render_begin{
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr,
        render_pass, framebuffer,
        {{0, 0}, {render_width, render_height}},
        0, nullptr};
    vkCmdBeginRenderPass(
        command_buffer, &render_begin,
        VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(
        command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (vertex_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &vertex_set, 0, nullptr);
    }
    if (pixel_set != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(
            command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 1, 1, &pixel_set, 0, nullptr);
    }
    if (vertex_program.push_constant_size != 0u) {
        vkCmdPushConstants(
            command_buffer, pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT,
            vertex_program.push_constant_offset,
            vertex_program.push_constant_size,
            vertex_program.packed_user_data.data());
    }
    if (pixel_program.push_constant_size != 0u) {
        vkCmdPushConstants(
            command_buffer, pipeline_layout,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            pixel_program.push_constant_offset,
            pixel_program.push_constant_size,
            pixel_program.packed_user_data.data());
    }
    if (draw.indexed) {
        vkCmdBindIndexBuffer(
            command_buffer, index_buffer.buffer, 0,
            draw.index_size == 4u
                ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(
            command_buffer, draw.vertex_count,
            std::max(draw.instance_count, 1u), 0, 0, 0);
    } else {
        vkCmdDraw(
            command_buffer, draw.vertex_count,
            std::max(draw.instance_count, 1u), 0, 0);
    }
    vkCmdEndRenderPass(command_buffer);
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& guest = pixel_program.images[index];
        auto& native = images[index];
        if (!guest.written && !guest.atomic) {
            continue;
        }
        ImageBarrier(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkBufferImageCopy copy{
            native.staging_offset, 0, 0,
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
             (guest.type == 11u || guest.type == 13u)
                ? guest.depth : 1u},
            {0, 0, 0},
            {guest.width, guest.height,
             guest.type == 10u ? guest.depth : 1u}};
        vkCmdCopyImageToBuffer(
            command_buffer, native.image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            staging.buffer, 1, &copy);
    }
    ImageBarrier(
        command_buffer, target.image,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdCopyImageToBuffer(
        command_buffer, target.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        staging.buffer, 1, &target_copy);
    const VkMemoryBarrier host_barrier{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier,
        0, nullptr, 0, nullptr);
    result = vkEndCommandBuffer(command_buffer);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics end", result);
    }
    const VkSubmitInfo submit{
        VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr,
        0, nullptr, nullptr, 1, &command_buffer,
        0, nullptr};
    result = vkQueueSubmit(presenter.queue, 1, &submit, fence);
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics submit", result);
    }
    result = vkWaitForFences(
        presenter.device, 1, &fence, VK_TRUE,
        UINT64_C(10000000000));
    if (result != VK_SUCCESS) {
        return fail("gen5 graphics wait", result);
    }
    std::vector<std::uint8_t> tiled(target.guest_size);
    auto copied = true;
    const auto write_guest_buffers =
        [&](const Gen5ComputeProgramInfo& program,
            const std::vector<NativeBuffer>& buffers) {
            auto wrote = true;
            for (std::size_t index = 0;
                 index < program.buffers.size(); ++index) {
                const auto& guest = program.buffers[index];
                if (!guest.written && !guest.atomic) {
                    continue;
                }
                wrote &= write_bytes(
                    memory_context, guest.guest_address,
                    buffers[index].map,
                    static_cast<std::size_t>(guest.byte_count));
            }
            return wrote;
        };
    copied &= write_guest_buffers(vertex_program, vertex_buffers);
    copied &= write_guest_buffers(pixel_program, pixel_buffers);
    if (has_color_target) {
        {
            const auto* const linear =
                static_cast<const std::uint8_t*>(staging.map) +
                target.staging_offset;
            const auto stride =
                std::max<std::size_t>(
                    1u, target.linear_size / 4096u);
            std::size_t sampled{};
            std::size_t nonzero{};
            std::uint8_t maximum{};
            for (std::size_t offset = 0;
                 offset < target.linear_size;
                 offset += stride) {
                const auto byte = linear[offset];
                ++sampled;
                nonzero += byte != 0u ? 1u : 0u;
                maximum = std::max(maximum, byte);
            }
            static std::atomic<std::uint32_t> target_logs{};
            const auto target_index = target_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (target_index < 128u ||
                (buffer_backed && buffer_log_index < 32u)) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-GFX",
                    "target output n=%u buffer_pass=%u addr=0x%llx "
                    "layout=%u type=%u order=%u bytes=%zu "
                    "nonzero=%zu/%zu max=%u",
                    target_index + 1u,
                    buffer_backed ? buffer_log_index + 1u : 0u,
                    static_cast<unsigned long long>(
                        draw.target_address),
                    draw.target_format, draw.target_number_type,
                    draw.target_channel_order,
                    target.linear_size, nonzero, sampled,
                    static_cast<unsigned>(maximum));
            }
        }
        const auto transformed = TransformGen5GuestImage(
            false,
            std::span<const std::uint8_t>{
                static_cast<const std::uint8_t*>(staging.map) +
                    target.staging_offset,
                target.linear_size},
            tiled, draw.target_tile_mode, render_width,
            render_height, target.element_bytes);
        copied = transformed &&
            write_bytes(
                memory_context, draw.target_address,
                tiled.data(), tiled.size());
    }
    for (std::size_t index = 0; index < images.size(); ++index) {
        const auto& guest = pixel_program.images[index];
        const auto& native = images[index];
        if (!guest.written && !guest.atomic) {
            continue;
        }
        std::vector<std::uint8_t> guest_tiled(native.guest_size);
        const auto guest_slice = native.guest_size / guest.depth;
        const auto linear_slice = native.linear_size / guest.depth;
        auto image_transformed = true;
        for (std::uint32_t slice = 0;
             slice < guest.depth; ++slice) {
            image_transformed &= TransformGen5GuestImage(
                false,
                std::span<const std::uint8_t>{
                    static_cast<const std::uint8_t*>(staging.map) +
                        native.staging_offset +
                        slice * linear_slice,
                    linear_slice},
                std::span<std::uint8_t>{
                    guest_tiled.data() + slice * guest_slice,
                    guest_slice},
                guest.tile_mode, native.element_width,
                native.element_height,
                native.element_bytes);
        }
        copied &= image_transformed &&
            write_bytes(
                memory_context, guest.guest_address,
                guest_tiled.data(), guest_tiled.size());
        if (buffer_log_index < 32u) {
            const auto* const linear =
                static_cast<const std::uint8_t*>(staging.map) +
                native.staging_offset;
            const auto stride = std::max<std::size_t>(
                native.linear_size / 512u, 1u);
            std::size_t sampled{};
            std::size_t nonzero{};
            std::uint8_t maximum{};
            for (std::size_t offset = 0;
                 offset < native.linear_size; offset += stride) {
                const auto byte = linear[offset];
                nonzero += byte != 0u ? 1u : 0u;
                maximum = std::max(maximum, byte);
                ++sampled;
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-GFX",
                "storage output buffer_pass=%u i=%zu "
                "addr=0x%llx %ux%ux%u fmt=%u tile=%u "
                "nonzero=%zu/%zu max=%u",
                buffer_log_index + 1u, index,
                static_cast<unsigned long long>(
                    guest.guest_address),
                guest.width, guest.height, guest.depth,
                guest.format, guest.tile_mode,
                nonzero, sampled,
                static_cast<unsigned>(maximum));
        }
    }
    cleanup();
    return copied;
}

std::uint64_t GetVulkanPresentCount() {
    return g_present_count.load(std::memory_order_relaxed);
}

void ResetVulkanPresenter() {
    const std::lock_guard lock{g_mutex};
    DestroyPresenter(g_presenter);
    g_present_count.store(0u, std::memory_order_relaxed);
}

}

#endif
