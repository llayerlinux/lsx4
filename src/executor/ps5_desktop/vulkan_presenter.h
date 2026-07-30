// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#ifdef __ANDROID__
struct ANativeWindow;

namespace Lsx4::Ps5Desktop {

struct Gen5ComputeProgramInfo;

struct VulkanAsyncPipelineStats {
    std::uint64_t compute_misses{};
    std::uint64_t cache_hits{};
    std::uint64_t fast_completed{};
    std::uint64_t optimized_queued{};
    std::uint64_t optimized_completed{};
    std::uint64_t published{};
    std::uint64_t optimized_replacements{};
    std::uint64_t build_failures{};
    std::uint64_t queue_drops{};
    std::uint64_t disk_cache_loads{};
    std::uint64_t disk_cache_writes{};
    std::uint64_t negative_cache_hits{};
};

struct VulkanDriverOptimizationStats {
    std::uint64_t enabled{};
    std::uint64_t pipeline_bind_attempts{};
    std::uint64_t pipeline_binds_elided{};
    std::uint64_t descriptor_bind_attempts{};
    std::uint64_t descriptor_binds_elided{};
    std::uint64_t push_constant_attempts{};
    std::uint64_t push_constants_elided{};
    std::uint64_t dynamic_state_commits{};
    std::uint64_t dynamic_state_full_replays{};
    std::uint64_t barriers_elided{};
    std::uint64_t render_scope_reuses{};
    std::uint64_t barrier_requests{};
    std::uint64_t barrier_batches{};
    std::uint64_t barrier_regions{};
    std::uint64_t barrier_regions_merged{};
    std::uint64_t transfer_requests{};
    std::uint64_t transfer_flushes{};
    std::uint64_t transfer_regions{};
    std::uint64_t transfer_copy_calls{};
    std::uint64_t queue_submits{};
    std::uint64_t queue_submit_logical_packets{};
    std::uint64_t queue_submit_command_buffers{};
    std::uint64_t queue_submit_calls_saved{};
    std::uint64_t queue_submit_forced_boundaries{};
    std::uint64_t vertex_bind_attempts{};
    std::uint64_t vertex_binds_elided{};
    std::uint64_t index_bind_attempts{};
    std::uint64_t index_binds_elided{};
};

struct VulkanMobileGpuStats {
    std::uint64_t effective_scale_percent{100};
    std::uint64_t max_gpu_backlog{};
    std::uint64_t coarse_rate_draws{};
    std::uint64_t fsr_frames{};
    std::uint64_t exact_surface_reuses{};
    std::uint64_t exact_surface_reuse_bytes{};
    std::uint64_t residency_gc_evictions{};
    std::uint64_t physical_drs_available{};
    std::uint64_t physical_scaled_draws{};
    std::uint64_t physical_scale_fallbacks{};
    std::uint64_t physical_resolve_blits{};
    std::uint64_t physical_pixels_saved{};
    std::uint64_t source_smaller_than_output_frames{};
    std::uint64_t scale_down_events{};
    std::uint64_t scale_up_events{};
};

struct VulkanReadbackBatchStats {
    std::uint64_t snapshots{};
    std::uint64_t ranges{};
    std::uint64_t copy_commands{};
    std::uint64_t copy_regions{};
    std::uint64_t retirement_waits{};
    std::uint64_t fallbacks{};
};

struct VulkanGen5GraphicsDraw {
    std::uint64_t target_address{};
    std::uint64_t index_address{};
    std::uint32_t target_width{};
    std::uint32_t target_height{};
    std::uint32_t target_tile_mode{};
    std::uint32_t target_format{};
    std::uint32_t target_number_type{};
    std::uint32_t target_channel_order{};
    std::uint32_t blend_control{};
    std::uint32_t color_write_mask{0xfu};
    std::uint32_t primitive_type{};
    std::uint32_t vertex_count{};
    std::uint32_t instance_count{1};
    std::uint32_t index_size{};
    bool indexed{};
};

using VulkanGen5ReadBytes =
    bool (*)(void*, std::uint64_t, void*, std::size_t);
using VulkanGen5WriteBytes =
    bool (*)(void*, std::uint64_t, const void*, std::size_t);

struct VulkanGuestVertex {
    float x{};
    float y{};
    float u{};
    float v{};
    std::uint32_t color{UINT32_C(0xffffffff)};
};

struct VulkanGuestDraw {
    std::uint64_t texture_key{};
    std::uint64_t texture_address{};
    std::uint64_t texture_signature{};
    std::uint32_t texture_width{};
    std::uint32_t texture_height{};
    std::uint32_t instance_count{1};
    bool require_target_source{};
    bool extended_blend_contract{};
    bool opaque{};
    bool premultiplied_alpha{};
    bool destination_source_alpha{};
    bool destination_inverse_source_alpha{};
    bool additive{};
    bool wave_effect{};
    bool repeat_texture{};
    bool nearest_texture{};
    std::array<float, 12> wave_parameters{
        0.00078125001164153218f,
        0.0013888889225199819f,
        0.016666000708937645f,
        0.10000000149011612f,
        0.30000001192092896f,
        0.30000001192092896f,
        3.0f, 4.0f, 7.0f, 20.0f, 0.0f,
        0.30000001192092896f};
    std::shared_ptr<const std::vector<std::uint8_t>> texture_rgba;
    std::vector<VulkanGuestVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct VulkanGuestPass {
    std::uint64_t target_key{};
    std::uint32_t clear_rgba{UINT32_C(0xff000000)};
    bool clear_target{};
    std::vector<VulkanGuestDraw> draws;
};

struct VulkanGuestFrame {
    std::uint64_t batch_id{};
    std::uint64_t base_batch_id{};
    std::uint64_t target_key{};
    std::size_t first_new_draw{};
    bool preserve_target{};
    bool clear_target{};
    std::uint32_t clear_rgba{UINT32_C(0xff000000)};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<VulkanGuestDraw> draws;
    std::vector<VulkanGuestPass> passes;
};

bool PresentVulkanFrame(ANativeWindow* window,
                        const std::uint8_t* rgba,
                        std::size_t byte_count,
                        std::uint32_t width,
                        std::uint32_t height,
                        std::uint64_t frame_number);

bool PresentVulkanGuestFrame(ANativeWindow* window,
                             const VulkanGuestFrame& frame,
                             std::uint64_t frame_number);

bool EnsureVulkanPresenter(ANativeWindow* window);

bool ExecuteVulkanGen5ComputeBuffers(
    std::span<const std::uint32_t> spirv,
    const Gen5ComputeProgramInfo& program,
    const std::array<std::uint32_t, 3>& groups,
    VulkanGen5ReadBytes read_bytes,
    VulkanGen5WriteBytes write_bytes,
    void* memory_context);

bool ExecuteVulkanGen5ComputeImages(
    std::span<const std::uint32_t> spirv,
    const Gen5ComputeProgramInfo& program,
    const std::array<std::uint32_t, 3>& groups,
    VulkanGen5ReadBytes read_bytes,
    VulkanGen5WriteBytes write_bytes,
    void* memory_context);

bool ExecuteVulkanGen5Graphics(
    std::span<const std::uint32_t> vertex_spirv,
    const Gen5ComputeProgramInfo& vertex_program,
    std::span<const std::uint32_t> pixel_spirv,
    const Gen5ComputeProgramInfo& pixel_program,
    const VulkanGen5GraphicsDraw& draw,
    VulkanGen5ReadBytes read_bytes,
    VulkanGen5WriteBytes write_bytes,
    void* memory_context);

std::uint64_t GetVulkanPresentCount();

void ConfigureVulkanPipelineCacheIdentity(
    std::string_view root_directory,
    std::string_view title_id);
void SetVulkanDriverOptimizationEnabled(bool enabled);
void SetVulkanReadbackBatchingEnabled(bool enabled);
void SetVulkanAsyncPipelineEnabled(bool enabled);
void SetVulkanMobileGpuEnabled(bool enabled);
void SetVulkanMaxAnisotropy(std::uint32_t max_anisotropy);
bool IsVulkanDriverOptimizationEnabled();
bool IsVulkanReadbackBatchingEnabled();
bool IsVulkanAsyncPipelineEnabled();
bool IsVulkanMobileGpuEnabled();
std::uint32_t GetVulkanEffectiveAnisotropy();
VulkanDriverOptimizationStats GetVulkanDriverOptimizationStats();
VulkanAsyncPipelineStats GetVulkanAsyncPipelineStats();
VulkanMobileGpuStats GetVulkanMobileGpuStats();
VulkanReadbackBatchStats GetVulkanReadbackBatchStats();
void ResetVulkanDriverOptimizationStats();
void ResetVulkanAsyncPipelineStats();
void ResetVulkanMobileGpuStats();
void ResetVulkanReadbackBatchStats();

void ResetVulkanPresenter();

}
#endif
