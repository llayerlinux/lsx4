#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Lsx4::Ps5Desktop {

using Gen5ReadMemory = bool (*)(void*, std::uint64_t, std::uint32_t*);

enum class Gen5ShaderStage : std::uint32_t {
    Compute,
    Vertex,
    Pixel,
};

struct Gen5ComputeCompileRequest {
    std::uint64_t shader_address{};
    std::span<const std::uint32_t> code;
    std::array<std::uint32_t, 64> user_data{};
    std::array<std::uint32_t, 3> local_size{1u, 1u, 1u};
    std::array<bool, 3> compute_group_id{};
    std::array<std::uint32_t, 3> dispatch_threads{};
    std::uint32_t compute_thread_ids_num{};
    std::uint32_t compute_workgroup_register{};
    bool compute_tg_size_en{};
    bool dispatch_thread_dimensions{};
    std::array<std::uint32_t, 32> pixel_interpolator_settings{};
    std::array<std::uint8_t, 8> pixel_target_output_mode{};
    std::array<std::uint8_t, 8> pixel_target_export_mapping{
        0xe4u, 0xe4u, 0xe4u, 0xe4u,
        0xe4u, 0xe4u, 0xe4u, 0xe4u};
    std::uint32_t pixel_input_count{};
    std::uint32_t pixel_input_enable{};
    std::uint32_t pixel_input_address{};
    std::uint32_t pixel_mrt_output_mask{};
    bool pixel_kill_enable{};
    bool pixel_depth_export_enable{};
    bool pixel_sample_mask_export_enable{};
    bool pixel_early_z{};
    bool pixel_execute_on_noop{};
    std::uint32_t wave_size{64u};
    std::uint32_t user_data_base{};
    std::uint32_t user_data_count{64u};
    std::uint32_t descriptor_set{};
    Gen5ShaderStage stage{Gen5ShaderStage::Compute};
    Gen5ReadMemory read_memory{};
    void* read_memory_context{};
};

struct Gen5BufferBinding {
    std::uint64_t guest_address{};
    std::uint64_t byte_count{};
    std::array<std::uint32_t, 8> descriptor{};
    std::uint32_t descriptor_dword_count{};
    std::uint32_t user_data_dword{UINT32_MAX};
    bool read{};
    bool written{};
    bool atomic{};
};

struct Gen5ImageBinding {
    std::uint64_t guest_address{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{1};
    std::uint32_t type{};
    std::uint32_t format{};
    std::uint32_t tile_mode{};
    std::array<std::uint32_t, 8> descriptor{};
    std::uint32_t descriptor_dword_count{};
    std::uint32_t resource_kind{};
    bool read{};
    bool written{};
    bool atomic{};
};

struct Gen5SamplerBinding {
    std::array<std::uint32_t, 8> descriptor{};
    std::uint32_t descriptor_dword_count{};
};

struct Gen5AddressBinding {
    std::uint64_t guest_base{};
    std::uint64_t binding_base{};
    std::int32_t min_offset{};
    bool read{};
    bool written{};
    bool atomic{};
};

struct Gen5DescriptorBinding {
    std::uint32_t kind{};
    std::uint32_t binding{};
    std::vector<std::uint32_t> resources;
};

struct Gen5ComputeProgramInfo {
    std::vector<Gen5BufferBinding> buffers;
    std::vector<Gen5ImageBinding> images;
    std::vector<Gen5SamplerBinding> samplers;
    std::vector<Gen5AddressBinding> addresses;
    std::vector<Gen5DescriptorBinding> descriptors;
    std::vector<std::uint32_t> flattened_srt;
    std::vector<std::uint32_t> packed_user_data;
    std::vector<std::uint32_t> packed_user_data_indices;
    std::uint32_t descriptor_set{};
    std::uint32_t push_constant_offset{};
    std::uint32_t push_constant_size{};
    std::uint32_t parameter_input_mask{};
    std::uint32_t parameter_output_mask{};
    std::uint32_t mrt_output_mask{};
    std::size_t address_count{};
    bool uses_fragment_coordinates{};
    bool simple_buffer_copy{};
    std::uint32_t copy_source_user_data{UINT32_MAX};
    std::uint32_t copy_destination_user_data{UINT32_MAX};
    bool simple_image_copy{};
    std::uint32_t copy_source_image{UINT32_MAX};
    std::uint32_t copy_destination_image{UINT32_MAX};
};

bool TryCompileGen5Compute(
    const Gen5ComputeCompileRequest& request,
    std::vector<std::uint32_t>& spirv,
    Gen5ComputeProgramInfo* program_info,
    std::string& error);

} // namespace Lsx4::Ps5Desktop
