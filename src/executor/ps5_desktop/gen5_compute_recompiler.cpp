#include "executor/ps5_desktop/gen5_compute_recompiler.h"

#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <algorithm>
#include <bit>

namespace Lsx4::Ps5Desktop {

bool TryCompileGen5Compute(
    const Gen5ComputeCompileRequest& request,
    std::vector<std::uint32_t>& spirv,
    Gen5ComputeProgramInfo* const program_info,
    std::string& error) {
    using namespace Libs::Graphics;

    ShaderComputeInputInfo input{};
    input.threads_num[0] = request.local_size[0];
    input.threads_num[1] = request.local_size[1];
    input.threads_num[2] = request.local_size[2];
    input.group_id[0] = request.compute_group_id[0];
    input.group_id[1] = request.compute_group_id[1];
    input.group_id[2] = request.compute_group_id[2];
    input.thread_ids_num = request.compute_thread_ids_num;
    input.workgroup_register =
        request.compute_workgroup_register;
    input.tg_size_en = request.compute_tg_size_en;
    input.dispatch_thread_dimensions =
        request.dispatch_thread_dimensions;
    input.dispatch_threads_num[0] =
        request.dispatch_threads[0];
    input.dispatch_threads_num[1] =
        request.dispatch_threads[1];
    input.dispatch_threads_num[2] =
        request.dispatch_threads[2];
    input.wave_size = request.wave_size;
    ShaderVertexInputInfo vertex_input{};
    ShaderPixelInputInfo pixel_input{};
    pixel_input.input_num =
        std::min<std::uint32_t>(request.pixel_input_count, 32u);
    std::copy_n(
        request.pixel_interpolator_settings.begin(),
        pixel_input.input_num,
        pixel_input.interpolator_settings);
    const auto active_pixel_inputs =
        request.pixel_input_enable & request.pixel_input_address;
    pixel_input.ps_pos_x =
        (active_pixel_inputs & UINT32_C(0x00000100)) != 0u;
    pixel_input.ps_pos_y =
        (active_pixel_inputs & UINT32_C(0x00000200)) != 0u;
    pixel_input.ps_pos_xy =
        pixel_input.ps_pos_x && pixel_input.ps_pos_y;
    pixel_input.ps_pos_z =
        (active_pixel_inputs & UINT32_C(0x00000400)) != 0u;
    pixel_input.ps_pos_w =
        (active_pixel_inputs & UINT32_C(0x00000800)) != 0u;
    pixel_input.ps_front_face =
        (active_pixel_inputs & UINT32_C(0x00001000)) != 0u;
    pixel_input.ps_sample_shading =
        (active_pixel_inputs & UINT32_C(0x00000011)) != 0u;
    pixel_input.ps_no_perspective =
        (active_pixel_inputs & UINT32_C(0x00000020)) != 0u;
    pixel_input.ps_pixel_kill_enable =
        request.pixel_kill_enable;
    pixel_input.ps_depth_export_enable =
        request.pixel_depth_export_enable;
    pixel_input.ps_sample_mask_export_enable =
        request.pixel_sample_mask_export_enable;
    pixel_input.ps_early_z = request.pixel_early_z;
    pixel_input.ps_execute_on_noop =
        request.pixel_execute_on_noop;
    pixel_input.mrt_output_mask =
        request.pixel_mrt_output_mask;
    for (std::size_t index = 0;
         index < request.pixel_target_output_mode.size(); ++index) {
        pixel_input.target_output_mode[index] =
            request.pixel_target_output_mode[index];
        pixel_input.target_export_mapping[index].packed =
            request.pixel_target_export_mapping[index];
    }
    const auto system_inputs =
        request.pixel_input_address & UINT32_C(0x000000ff);
    pixel_input.ps_system_input_base =
        std::popcount(system_inputs & UINT32_C(0x00000001)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000002)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000004)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000008)) * 3u +
        std::popcount(system_inputs & UINT32_C(0x00000010)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000020)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000040)) * 2u +
        std::popcount(system_inputs & UINT32_C(0x00000080));

    ShaderRecompiler::CompileOptions options{};
    options.stage =
        request.stage == Gen5ShaderStage::Vertex
        ? ShaderType::Vertex
        : request.stage == Gen5ShaderStage::Pixel
            ? ShaderType::Pixel
            : ShaderType::Compute;
    options.lane_mask_mode = ShaderLaneMaskMode::PerInvocation;
    options.wave_size = request.wave_size;
    options.user_data_base = request.user_data_base;
    options.user_data_count = std::min<std::uint32_t>(
        request.user_data_count,
        static_cast<std::uint32_t>(request.user_data.size()));
    options.descriptor_set = request.descriptor_set;
    options.shader_hash = request.shader_address;
    options.shader_base = request.shader_address;
    options.dump_ir = false;
    options.user_data = request.user_data.data();
    options.read_memory = request.read_memory;
    options.read_memory_data = request.read_memory_context;
    options.compute_input_info =
        request.stage == Gen5ShaderStage::Compute ? &input : nullptr;
    options.vertex_input_info =
        request.stage == Gen5ShaderStage::Vertex
        ? &vertex_input : nullptr;
    options.pixel_input_info =
        request.stage == Gen5ShaderStage::Pixel
        ? &pixel_input : nullptr;

    ShaderRecompiler::CompileResult result{};
    if (!ShaderRecompiler::TryRecompile(
            request.code, options, result, &error)) {
        return false;
    }
    spirv = std::move(result.spirv);
    if (program_info != nullptr) {
        *program_info = {};
        program_info->descriptor_set =
            result.program.bindings.descriptor_set;
        program_info->push_constant_offset =
            result.program.bindings.push_constant_offset;
        program_info->push_constant_size =
            result.program.bindings.push_constant_size;
        for (const auto& input_binding :
             result.program.info.inputs) {
            if (input_binding.kind ==
                    ShaderRecompiler::IR::StageInputKind::Parameter &&
                input_binding.location < 32u) {
                const auto mapped_location =
                    request.stage == Gen5ShaderStage::Pixel &&
                            input_binding.location <
                                request.pixel_input_count
                    ? request.pixel_interpolator_settings[
                          input_binding.location] & 0x3fu
                    : input_binding.location;
                if (mapped_location >= 32u) {
                    continue;
                }
                program_info->parameter_input_mask |=
                    1u << mapped_location;
            } else if (
                input_binding.kind ==
                ShaderRecompiler::IR::StageInputKind::FragCoord) {
                program_info->uses_fragment_coordinates = true;
            }
        }
        for (const auto& output_binding :
             result.program.info.outputs) {
            if (output_binding.kind ==
                    ShaderRecompiler::IR::StageOutputKind::Parameter &&
                output_binding.location < 32u) {
                program_info->parameter_output_mask |=
                    1u << output_binding.location;
            } else if (
                output_binding.kind ==
                    ShaderRecompiler::IR::StageOutputKind::Mrt &&
                output_binding.index < 32u) {
                program_info->mrt_output_mask |=
                    1u << output_binding.index;
            }
        }
        program_info->flattened_srt =
            result.resources.flattened_srt;
        for (const auto reg :
             result.program.bindings.user_data_registers) {
            const auto index =
                reg - result.program.user_data_base;
            program_info->packed_user_data_indices.push_back(
                index);
            program_info->packed_user_data.push_back(
                index < result.resources.user_data.size()
                    ? result.resources.user_data[index] : 0u);
        }
        for (const auto& descriptor :
             result.program.bindings.descriptors) {
            Gen5DescriptorBinding binding{};
            binding.kind =
                static_cast<std::uint32_t>(descriptor.kind);
            binding.binding = descriptor.binding;
            binding.resources = descriptor.resources;
            program_info->descriptors.push_back(
                std::move(binding));
        }
        program_info->address_count =
            result.program.info.addresses.size();
        for (std::size_t index = 0;
             index < result.program.info.buffers.size(); ++index) {
            const auto& descriptor = result.resources.buffers[index];
            const auto& resource = result.program.info.buffers[index];
            Gen5BufferBinding binding{};
            binding.descriptor = descriptor.dwords;
            binding.descriptor_dword_count =
                descriptor.dword_count;
            binding.guest_address =
                static_cast<std::uint64_t>(descriptor.dwords[0]) |
                (static_cast<std::uint64_t>(
                     descriptor.dwords[1] & UINT32_C(0xffff)) << 32u);
            const auto stride =
                (descriptor.dwords[1] >> 16u) & UINT32_C(0x3fff);
            binding.byte_count =
                static_cast<std::uint64_t>(descriptor.dwords[2]) *
                std::max(stride, 1u);
            binding.byte_count = std::max<std::uint64_t>(
                binding.byte_count, resource.max_byte_extent);
            binding.read = resource.read;
            binding.written = resource.written;
            binding.atomic = resource.atomic;
            for (std::uint32_t offset = 0;
                 offset + descriptor.dword_count <=
                     request.user_data.size();
                 ++offset) {
                if (std::equal(
                        descriptor.dwords.begin(),
                        descriptor.dwords.begin() +
                            descriptor.dword_count,
                        request.user_data.begin() + offset)) {
                    binding.user_data_dword = offset;
                    break;
                }
            }
            program_info->buffers.push_back(binding);
        }
        for (std::size_t index = 0;
             index < result.program.info.images.size(); ++index) {
            const auto& descriptor = result.resources.images[index];
            const auto& resource = result.program.info.images[index];
            Gen5ImageBinding binding{};
            binding.descriptor = descriptor.dwords;
            binding.descriptor_dword_count =
                descriptor.dword_count;
            binding.guest_address =
                ((static_cast<std::uint64_t>(descriptor.dwords[0]) |
                  (static_cast<std::uint64_t>(
                       descriptor.dwords[1] & UINT32_C(0xff)) << 32u))
                 << 8u);
            binding.width =
                (((descriptor.dwords[1] >> 30u) & 3u) |
                 ((descriptor.dwords[2] & UINT32_C(0x3fff)) << 2u)) +
                1u;
            binding.height =
                ((descriptor.dwords[2] >> 14u) &
                 UINT32_C(0x3fff)) + 1u;
            binding.depth =
                (descriptor.dwords[4] & UINT32_C(0x1fff)) + 1u;
            binding.type =
                (descriptor.dwords[3] >> 28u) & UINT32_C(0xf);
            binding.format =
                (descriptor.dwords[1] >> 20u) & UINT32_C(0x1ff);
            binding.tile_mode =
                (descriptor.dwords[3] >> 20u) & UINT32_C(0x1f);
            binding.read = resource.read;
            binding.written = resource.written;
            binding.atomic = resource.atomic;
            binding.resource_kind =
                static_cast<std::uint32_t>(resource.kind);
            program_info->images.push_back(binding);
        }
        for (const auto& descriptor :
             result.resources.samplers) {
            Gen5SamplerBinding binding{};
            binding.descriptor = descriptor.dwords;
            binding.descriptor_dword_count =
                descriptor.dword_count;
            program_info->samplers.push_back(binding);
        }
        for (std::size_t index = 0;
             index < result.program.info.addresses.size();
             ++index) {
            const auto& resource =
                result.program.info.addresses[index];
            const auto& snapshot =
                result.resources.addresses[index];
            program_info->addresses.push_back({
                .guest_base = snapshot.guest_base,
                .binding_base = snapshot.binding_base,
                .min_offset = resource.min_offset,
                .read = resource.read,
                .written = resource.written,
                .atomic = resource.atomic});
        }
        if (program_info->buffers.size() == 2u &&
            program_info->images.empty() &&
            program_info->address_count == 0u) {
            const Gen5BufferBinding* source{};
            const Gen5BufferBinding* destination{};
            for (const auto& buffer : program_info->buffers) {
                if (buffer.read && !buffer.written &&
                    !buffer.atomic) {
                    source = &buffer;
                } else if (buffer.written && !buffer.atomic) {
                    destination = &buffer;
                }
            }
            if (source != nullptr && destination != nullptr &&
                source != destination &&
                source->user_data_dword != UINT32_MAX &&
                destination->user_data_dword != UINT32_MAX) {
                program_info->simple_buffer_copy = true;
                program_info->copy_source_user_data =
                    source->user_data_dword;
                program_info->copy_destination_user_data =
                    destination->user_data_dword;
            }
        }
        if (program_info->buffers.empty() &&
            program_info->images.size() == 2u &&
            program_info->address_count == 0u) {
            const Gen5ImageBinding* source{};
            const Gen5ImageBinding* destination{};
            std::uint32_t source_index{};
            std::uint32_t destination_index{};
            for (std::uint32_t index = 0;
                 index < program_info->images.size(); ++index) {
                const auto& image = program_info->images[index];
                if (image.read && !image.written && !image.atomic) {
                    source = &image;
                    source_index = index;
                } else if (image.written && !image.atomic) {
                    destination = &image;
                    destination_index = index;
                }
            }
            if (source != nullptr && destination != nullptr &&
                source != destination &&
                source->guest_address != 0 &&
                destination->guest_address != 0 &&
                source->guest_address != destination->guest_address &&
                source->width == destination->width &&
                source->height == destination->height &&
                source->format == destination->format &&
                source->tile_mode == destination->tile_mode) {
                program_info->simple_image_copy = true;
                program_info->copy_source_image = source_index;
                program_info->copy_destination_image =
                    destination_index;
            }
        }
    }
    return !spirv.empty();
}

} // namespace Lsx4::Ps5Desktop
