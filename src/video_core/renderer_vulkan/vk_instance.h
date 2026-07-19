// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <span>
#include <unordered_map>

#include "video_core/renderer_vulkan/vk_platform.h"

#define TRACY_VK_USE_SYMBOL_TABLE
#include <tracy/TracyVulkan.hpp>

namespace Frontend {
class WindowSDL;
}

VK_DEFINE_HANDLE(VmaAllocator)

namespace Vulkan {

// Variant A probe: real Store OrbShdr PS -> production GCN->SPIR-V recompiler -> vk::ShaderModule on
// a headless device (no WSI). Defined in vk_instance.cpp (bundled-Vulkan TU). Appends markers to
// marker_path. Gated by the run-gcn-spirv-smoke sentinel.
void ExecutorRunGcnSpirvSmoke(const char* marker_path);
// Layer3PipelinePrep: real OrbShdr -> PM4 SetShReg -> regs.ps_program/vs_program -> AmdGpu::GetParams
// -> BinaryInfo + Shader::Info, no draw/Vulkan/GetProgram. Defined in vk_instance.cpp.
void ExecutorRunLayer3PipelinePrepSmoke(const char* marker_path);
// Layer3PipelineShaderCompile: PipelinePrep + RuntimeInfo-from-regs + production TranslateProgram/
// EmitSPIRV on the real PS (device vkCreateShaderModule opt-in). Defined in vk_instance.cpp.
void ExecutorRunLayer3PipelineShaderCompileSmoke(const char* marker_path);
// Layer3GnmcapReloc: rebase a staged shader into a 40-bit-representable untagged VA (.gnmcap reloc) +
// run the full pipeline shader chain off it. Closes the device 40-bit address gap. In vk_instance.cpp.
void ExecutorRunLayer3GnmcapRelocSmoke(const char* marker_path);
// Layer3VsCompileClassify: safely classify a real VS (vs_18) -- precheck (REQUIRES_RESOURCE_STATE,
// translate SKIPPED_SAFE) or opt-in try-translate. No state guessing / no crash. In vk_instance.cpp.
void ExecutorRunVsCompileClassifySmoke(const char* marker_path);
// Layer3GnmcapReplayDryRun: capture a parsed frame to .gnmcap, then read it back, apply reloc, re-parse,
// and verify state-equivalence (num_indices + shader hash). Proves writer output is replayable. In vk_instance.cpp.
void ExecutorRunGnmcapReplayDryRunSmoke(const char* marker_path);
// Real .gnmcap replay with captured memory ranges: load live-submit-N.gnmcap/replay-*.gnmcap next to
// marker_path, relocate memory, submit through Liverpool/rasterizer, and present replay RT.
void ExecutorRunGnmReplayWithMemorySmoke(const char* marker_path);

class Instance {
public:
    // create_headless_device=true also selects a physical device and creates a logical device WITHOUT a
    // window/surface/swapchain (no WSI). Used to drive the production shader_recompiler (GCN->SPIR-V) on
    // a real shader headlessly -- e.g. on the emulator where WSI swapchain creation reboots SwiftShader.
    explicit Instance(bool validation = false, bool crash_diagnostic = false,
                      bool create_headless_device = false);
    explicit Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                      bool enable_validation = false, bool enable_crash_diagnostic = false);
    ~Instance();

    /// Returns a formatted string for the driver version
    std::string GetDriverVersionName();

    /// Gets a compatibility format if the format is not supported.
    [[nodiscard]] vk::Format GetSupportedFormat(vk::Format format,
                                                vk::FormatFeatureFlags2 flags) const;

    /// Returns the Vulkan instance
    vk::Instance GetInstance() const {
        return *instance;
    }

    /// Returns the current physical device
    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }

    /// Returns the Vulkan device
    vk::Device GetDevice() const {
        return *device;
    }

    /// Returns the VMA allocator handle
    VmaAllocator GetAllocator() const {
        return allocator;
    }

    /// Returns a list of the available physical devices
    std::span<const vk::PhysicalDevice> GetPhysicalDevices() const {
        return physical_devices;
    }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const {
        return queue_family_index;
    }

    u32 GetPresentQueueFamilyIndex() const {
        return queue_family_index;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    TracyVkCtx GetProfilerContext() const {
        return profiler_context;
    }

    /// Returns true if anisotropic filtering is supported
    bool IsAnisotropicFilteringSupported() const {
        return features.samplerAnisotropy;
    }

    /// Returns true if depth bounds testing is supported
    bool IsDepthBoundsSupported() const {
        return features.depthBounds;
    }

    /// Returns true if 16-bit floats are supported in shaders
    bool IsShaderFloat16Supported() const {
        return vk12_features.shaderFloat16;
    }

    /// Returns true if 64-bit floats are supported in shaders
    bool IsShaderFloat64Supported() const {
        return features.shaderFloat64;
    }

    /// Returns true if 64-bit ints are supported in shaders
    bool IsShaderInt64Supported() const {
        return features.shaderInt64;
    }

    /// Returns true if 16-bit ints are supported in shaders
    bool IsShaderInt16Supported() const {
        return features.shaderInt16;
    }

    /// Returns true if 8-bit ints are supported in shaders
    bool IsShaderInt8Supported() const {
        return vk12_features.shaderInt8;
    }

    /// Returns true if VK_KHR_maintenance8 is supported
    bool IsMaintenance8Supported() const {
        return maintenance_8;
    }

    /// Returns true if VK_EXT_attachment_feedback_loop_layout is supported
    bool IsAttachmentFeedbackLoopLayoutSupported() const {
        return attachment_feedback_loop;
    }

    /// Returns true when VK_EXT_custom_border_color is supported
    bool IsCustomBorderColorSupported() const {
        return custom_border_color;
    }

    /// Returns true when VK_EXT_shader_stencil_export is supported
    bool IsShaderStencilExportSupported() const {
        return shader_stencil_export;
    }

    /// Returns true when VK_EXT_depth_clip_control is supported
    bool IsDepthClipControlSupported() const {
        return depth_clip_control;
    }

    /// Returns true when VK_EXT_depth_clip_enable is supported
    bool IsDepthClipEnableSupported() const {
        return depth_clip_enable;
    }

    /// Returns true when VK_EXT_depth_range_unrestricted is supported
    bool IsDepthRangeUnrestrictedSupported() const {
        return depth_range_unrestricted;
    }

    /// Returns true when VK_EXT_extended_dynamic_state3 is supported.
    bool IsExtendedDynamicState3Supported() const {
        return dynamic_state_3;
    }

    /// Returns true when the extendedDynamicState3ColorWriteMask feature of
    /// VK_EXT_extended_dynamic_state3 is supported.
    bool IsDynamicColorWriteMaskSupported() const {
        return dynamic_state_3 && dynamic_state_3_features.extendedDynamicState3ColorWriteMask;
    }

    /// Returns true when a coarse per-pipeline fragment rate is enabled. This keeps guest render
    /// target sizes and coordinates exact while reducing fragment shader invocations on mobile.
    bool IsCoarseFragmentShadingSupported() const {
        return fragment_shading_rate && coarse_fragment_size.width > 1 &&
               coarse_fragment_size.height > 1;
    }

    vk::Extent2D GetCoarseFragmentSize() const {
        return coarse_fragment_size;
    }

    /// Returns true when VK_EXT_vertex_input_dynamic_state is supported.
    bool IsVertexInputDynamicState() const {
        return vertex_input_dynamic_state;
    }

    /// Returns true when the robustBufferAccess2 feature of VK_EXT_robustness2 is supported.
    bool IsRobustBufferAccess2Supported() const {
        return robustness2 && robustness2_features.robustBufferAccess2;
    }

    /// Returns true when the robustImageAccess2 feature of VK_EXT_robustness2 is supported.
    bool IsRobustImageAccess2Supported() const {
        return robustness2 && robustness2_features.robustImageAccess2;
    }

    /// Returns true when the nullDescriptor feature of VK_EXT_robustness2 is supported.
    bool IsNullDescriptorSupported() const {
        return robustness2 && robustness2_features.nullDescriptor;
    }

    /// Returns true when VK_EXT_device_fault is enabled and fault reports can be queried.
    bool IsDeviceFaultSupported() const {
        return device_fault && device_fault_features.deviceFault;
    }

    /// Emits the driver fault description, GPU addresses and vendor data after device loss.
    void DumpDeviceFaultInfo(const char* where, u64 failing_tick) const noexcept;

    /// Returns true when VK_KHR_fragment_shader_barycentric is supported.
    bool IsFragmentShaderBarycentricSupported() const {
        return fragment_shader_barycentric;
    }

    /// Returns true when VK_AMD_shader_explicit_vertex_parameter is supported.
    bool IsAmdShaderExplicitVertexParameterSupported() const {
        return amd_shader_explicit_vertex_parameter;
    }

    /// Returns true when VK_EXT_primitive_topology_list_restart is supported for regular lists.
    bool IsListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyListRestart;
    }

    /// Returns true when VK_EXT_primitive_topology_list_restart is supported for patch lists.
    bool IsPatchListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyPatchListRestart;
    }

    /// Returns true when VK_EXT_legacy_vertex_attributes is supported.
    bool IsLegacyVertexAttributesSupported() const {
        return legacy_vertex_attributes;
    }

    /// Returns true when VK_EXT_provoking_vertex is supported.
    bool IsProvokingVertexSupported() const {
        return provoking_vertex;
    }

    /// Returns true when VK_AMD_shader_image_load_store_lod is supported.
    bool IsImageLoadStoreLodSupported() const {
        return image_load_store_lod;
    }

    /// Returns true when VK_AMD_gcn_shader is supported.
    bool IsAmdGcnShaderSupported() const {
        return amd_gcn_shader;
    }

    /// Returns true when VK_AMD_shader_trinary_minmax is supported.
    bool IsAmdShaderTrinaryMinMaxSupported() const {
        return amd_shader_trinary_minmax;
    }

    /// Returns true when the shaderBufferFloat32AtomicMinMax feature of
    /// VK_EXT_shader_atomic_float2 is supported.
    bool IsShaderAtomicFloatBuffer32MinMaxSupported() const {
        return shader_atomic_float2 &&
               shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax;
    }

    /// Returns true when the shaderImageFloat32AtomicMinMax feature of
    /// VK_EXT_shader_atomic_float2 is supported.
    bool IsShaderAtomicFloatImage32MinMaxSupported() const {
        return shader_atomic_float2 && shader_atomic_float2_features.shaderImageFloat32AtomicMinMax;
    }

    /// Returns true if 64-bit integer atomic operations can be used on buffers
    bool IsBufferInt64AtomicsSupported() const {
        return vk12_features.shaderBufferInt64Atomics;
    }

    /// Returns true if 64-bit integer atomic operations can be used on shared memory
    bool IsSharedInt64AtomicsSupported() const {
        return vk12_features.shaderSharedInt64Atomics;
    }

    /// Returns true if the subgroup size can be set to match guest subgroup size
    bool IsSubgroupSize64Supported() const {
        return vk13_features.subgroupSizeControl && vk13_props.maxSubgroupSize >= 64;
    }

    /// Returns true when VK_KHR_workgroup_memory_explicit_layout is supported.
    bool IsWorkgroupMemoryExplicitLayoutSupported() const {
        return workgroup_memory_explicit_layout &&
               workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess;
    }

    /// Returns true if VK_NV_framebuffer_mixed_samples or
    /// VK_AMD_mixed_attachment_samples is supported
    bool IsMixedDepthSamplesSupported() const {
        return nv_framebuffer_mixed_samples || amd_mixed_attachment_samples;
    }

    /// Returns true if VK_AMD_mixed_attachment_samples is supported
    bool IsMixedAnySamplesSupported() const {
        return amd_mixed_attachment_samples;
    }

    /// Returns true when geometry shaders are supported by the device
    bool IsGeometryStageSupported() const {
        return features.geometryShader;
    }

    /// Returns true when tessellation is supported by the device
    bool IsTessellationSupported() const {
        return features.tessellationShader;
    }

    /// Returns true when tessellation isolines are supported by the device
    bool IsTessellationIsolinesSupported() const {
        return !portability_subset || portability_features.tessellationIsolines;
    }

    /// Returns true when tessellation point mode is supported by the device
    bool IsTessellationPointModeSupported() const {
        return !portability_subset || portability_features.tessellationPointMode;
    }

    /// Returns the vendor ID of the physical device
    u32 GetVendorID() const {
        return properties.vendorID;
    }

    /// Returns the device ID of the physical device
    u32 GetDeviceID() const {
        return properties.deviceID;
    }

    /// Returns the driver ID.
    vk::DriverId GetDriverID() const {
        return driver_id;
    }

    /// Returns the current driver version provided in Vulkan-formatted version numbers.
    u32 GetDriverVersion() const {
        return properties.driverVersion;
    }

    /// Returns the current Vulkan API version provided in Vulkan-formatted version numbers.
    u32 ApiVersion() const {
        return properties.apiVersion;
    }

    /// Returns the vendor name reported from Vulkan.
    std::string_view GetVendorName() const {
        return vendor_name;
    }

    /// Returns the list of available extensions.
    std::span<const std::string> GetAvailableExtensions() const {
        return available_extensions;
    }

    /// Returns the device name.
    std::string_view GetModelName() const {
        return properties.deviceName;
    }

    /// Returns if the device is an integrated GPU.
    bool IsIntegrated() const {
        return properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    }

    /// Returns the pipeline cache unique identifier
    const auto GetPipelineCacheUUID() const {
        return properties.pipelineCacheUUID;
    }

    /// Returns the minimum required alignment for uniforms
    vk::DeviceSize UniformMinAlignment() const {
        return properties.limits.minUniformBufferOffsetAlignment;
    }

    ///  Returns the maximum size of uniform buffers.
    vk::DeviceSize UniformMaxSize() const {
        return properties.limits.maxUniformBufferRange;
    }

    /// Returns the minimum required alignment for storage buffers
    vk::DeviceSize StorageMinAlignment() const {
        return properties.limits.minStorageBufferOffsetAlignment;
    }

    /// Returns the minimum alignemt required for accessing host-mapped device memory
    vk::DeviceSize NonCoherentAtomSize() const {
        return properties.limits.nonCoherentAtomSize;
    }

    /// Returns the subgroup size of the selected physical device.
    u32 SubgroupSize() const {
        return vk11_props.subgroupSize;
    }

    /// Returns the maximum size of compute shared memory.
    u32 MaxComputeSharedMemorySize() const {
        return properties.limits.maxComputeSharedMemorySize;
    }

    /// Returns the maximum sampler LOD bias.
    float MaxSamplerLodBias() const {
        return properties.limits.maxSamplerLodBias;
    }

    /// Returns the maximum sampler anisotropy.
    float MaxSamplerAnisotropy() const {
        return properties.limits.maxSamplerAnisotropy;
    }

    /// Returns the maximum number of push descriptors.
    u32 MaxPushDescriptors() const {
        return push_descriptor_props.maxPushDescriptors;
    }

    /// Returns the vulkan 1.2 physical device properties.
    const vk::PhysicalDeviceVulkan12Properties& GetVk12Properties() const noexcept {
        return vk12_props;
    }

    /// Returns the memory properties of the physical device.
    const vk::PhysicalDeviceMemoryProperties& GetMemoryProperties() const noexcept {
        return memory_properties;
    }

    /// Returns true if shaders can declare the ClipDistance attribute
    bool IsShaderClipDistanceSupported() const {
        return features.shaderClipDistance;
    }

    /// Returns the maximim viewport width.
    u32 GetMaxViewportWidth() const {
        return properties.limits.maxViewportDimensions[0];
    }

    /// Returns the maximum viewport height.
    u32 GetMaxViewportHeight() const {
        return properties.limits.maxViewportDimensions[1];
    }

    /// Returns the maximum render area width.
    u32 GetMaxFramebufferWidth() const {
        return properties.limits.maxFramebufferWidth;
    }

    /// Returns the maximum render area height.
    u32 GetMaxFramebufferHeight() const {
        return properties.limits.maxFramebufferHeight;
    }

    /// Returns the maximum number of samplers that can be allocated at once.
    u32 GetMaxSamplerAllocationCount() const {
        if (driver_id == vk::DriverId::eMesaHoneykrisp) {
            // KosmicKrisp currently exposes a higher limit than it can reliably allocate.
            return 1024;
        }
        return properties.limits.maxSamplerAllocationCount;
    }

    /// Returns the sample count flags supported by color buffers.
    vk::SampleCountFlags GetColorSampleCounts() const {
        return properties.limits.framebufferColorSampleCounts;
    }

    /// Returns the sample count flags supported by depth buffer.
    vk::SampleCountFlags GetDepthSampleCounts() const {
        return properties.limits.framebufferDepthSampleCounts &
               properties.limits.framebufferStencilSampleCounts;
    }

    /// Returns true if logic ops are supported by the device.
    bool IsLogicOpSupported() const {
        return features.logicOp;
    }

    /// Returns whether VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT is supported for compressed
    /// images on this physical device.
    bool IsBlockTexelViewSupported() const {
        return supports_block_texel_view;
    }

    /// Returns whether VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT is supported on 3D images.
    bool Is2dViewOf3dSupported() const {
        return image_2d_view_of_3d && image_2d_view_of_3d_features.image2DViewOf3D &&
               image_2d_view_of_3d_features.sampler2DViewOf3D;
    }

    /// Returns whether the device can report memory usage.
    bool CanReportMemoryUsage() const {
        return supports_memory_budget;
    }

    /// Returns the amount of memory used.
    [[nodiscard]] u64 GetDeviceMemoryUsage() const;

    /// Returns the total memory budget available to the device.
    [[nodiscard]] u64 GetTotalMemoryBudget() const {
        return total_memory_budget;
    }

    /// Determines if a format is supported for a set of feature flags.
    [[nodiscard]] bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags2 flags) const;

private:
    /// Creates the logical device opportunistically enabling extensions
    bool CreateDevice();

    /// Creates the VMA allocator handle
    void CreateAllocator();

    /// Collects various information from the device.
    void CollectDeviceParameters();
    void CollectPhysicalMemoryInfo();
    void CollectImageFormatInfo();
    void CollectToolingInfo() const;

    /// Gets the supported feature flags for a format.
    [[nodiscard]] vk::FormatFeatureFlags2 GetFormatFeatureFlags(vk::Format format) const;

private:
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceMemoryProperties memory_properties;
    vk::PhysicalDeviceVulkan11Properties vk11_props;
    vk::PhysicalDeviceVulkan12Properties vk12_props;
    vk::PhysicalDeviceVulkan13Properties vk13_props;
    vk::PhysicalDevicePushDescriptorPropertiesKHR push_descriptor_props;
    vk::PhysicalDeviceFeatures features;
    vk::PhysicalDeviceVulkan12Features vk12_features;
    vk::PhysicalDeviceVulkan13Features vk13_features;
    vk::PhysicalDevicePortabilitySubsetFeaturesKHR portability_features;
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_state_3_features;
    vk::PhysicalDeviceFragmentShadingRateFeaturesKHR fragment_shading_rate_features;
    vk::PhysicalDeviceRobustness2FeaturesEXT robustness2_features;
    vk::PhysicalDeviceFaultFeaturesEXT device_fault_features;
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT shader_atomic_float2_features;
    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR
        workgroup_memory_explicit_layout_features;
    vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT image_2d_view_of_3d_features;
    vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT list_restart_features;
    vk::DriverIdKHR driver_id;
    vk::UniqueDebugUtilsMessengerEXT debug_callback{};
    std::string vendor_name;
    VmaAllocator allocator{};
    vk::Queue present_queue;
    vk::Queue graphics_queue;
    std::vector<vk::PhysicalDevice> physical_devices;
    std::vector<std::string> available_extensions;
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    TracyVkCtx profiler_context{};
    u32 queue_family_index{0};
    vk::Extent2D coarse_fragment_size{1, 1};
    bool custom_border_color{};
    bool fragment_shader_barycentric{};
    bool amd_shader_explicit_vertex_parameter{};
    bool depth_clip_control{};
    bool depth_clip_enable{};
    bool dynamic_state_3{};
    bool fragment_shading_rate{};
    bool depth_range_unrestricted{};
    bool vertex_input_dynamic_state{};
    bool robustness2{};
    bool device_fault{};
    bool list_restart{};
    bool legacy_vertex_attributes{};
    bool provoking_vertex{};
    bool shader_stencil_export{};
    bool image_load_store_lod{};
    bool amd_gcn_shader{};
    bool amd_shader_trinary_minmax{};
    bool nv_framebuffer_mixed_samples{};
    bool amd_mixed_attachment_samples{};
    bool shader_atomic_float{};
    bool shader_atomic_float2{};
    bool workgroup_memory_explicit_layout{};
    bool image_2d_view_of_3d{};
    bool portability_subset{};
    bool maintenance_8{};
    bool attachment_feedback_loop{};
    bool supports_memory_budget{};
    bool supports_block_texel_view{};
    mutable std::atomic_flag device_fault_dumped = ATOMIC_FLAG_INIT;
    // Both texture and buffer GC run at every guest submit. Share one throttled budget sample
    // instead of issuing two physical-device driver queries per submit.
    mutable std::atomic<u64> memory_usage_cache{};
    mutable std::atomic<u64> memory_usage_sample_time_ns{};
    mutable std::atomic_flag memory_usage_query_in_flight = ATOMIC_FLAG_INIT;
    u64 total_memory_budget{};
    std::vector<size_t> valid_heaps;
};

} // namespace Vulkan
