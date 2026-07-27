// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <span>
#include <vector>

#ifdef __ANDROID__
struct ANativeWindow;

namespace Lsx4::Ps5Desktop {

struct Gen5ComputeProgramInfo;

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

void ResetVulkanPresenter();

}
#endif
