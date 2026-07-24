// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace Executor::Jit {
struct LsxDecodedOp;
struct LsxMachineImage;
}

namespace Lsx4::Translation {

struct GuestRegisterSnapshot {
    std::array<std::uint64_t, 16> integer{};
    std::uint64_t condition{};
    std::uint64_t fs_origin{};
    std::uint64_t gs_origin{};
};

struct DispatchFailureEvent {
    std::string_view reason{};
    std::string_view encoding{};
    std::uint64_t region_pc{};
    std::uint64_t predecessor_pc{};
    std::uint64_t predecessor_outcome{};
    std::uint64_t predecessor_state_pc{};
    std::uint64_t predecessor_successor{};
    std::uint32_t region_bytes{};
    std::uint32_t region_traits{};
    std::uint32_t predecessor_bytes{};
    GuestRegisterSnapshot machine{};
};

struct GuestFaultEvent {
    std::string_view operation{};
    std::uint64_t address{};
    std::uint64_t dispatch_pc{};
    std::uint32_t byte_count{};
};

struct ExitFrontierEvent {
    std::string_view encoding{};
    std::string_view region_summary{};
    std::uint64_t retired_regions{};
    std::uint64_t region_pc{};
    std::uint64_t outcome{};
    std::uint64_t completion_value{};
    std::uint32_t region_bytes{};
    std::uint32_t region_traits{};
    bool completion_pending{};
    GuestRegisterSnapshot machine{};
};

struct TranslationProbeResult {
    std::uint64_t translated_return{};
    std::uint64_t bridge_successor{};
    std::uint64_t machine_accumulator{};
    std::uint64_t replay_last_pc{};
    std::uint64_t replay_accumulator{};
    std::uint64_t smc_initial_result{};
    std::uint64_t smc_mutated_result{};
    bool bridge_compiled{};
    bool replay_passed{};
};

struct DifferentialMismatchEvent {
    std::string_view region_summary{};
    std::uint64_t region_pc{};
    std::uint64_t native_successor{};
    std::uint64_t semantic_successor{};
    std::uint64_t native_outcome{};
    std::uint64_t semantic_outcome{};
    GuestRegisterSnapshot native_machine{};
    GuestRegisterSnapshot semantic_machine{};
};

struct HleBoundaryEvent {
    std::string_view mode{};
    std::uint64_t sequence{};
    std::uint64_t thunk{};
    std::uint64_t host_function{};
    std::uint64_t continuation{};
    std::uint64_t stack_input{};
    std::uint64_t stack_output{};
    std::array<std::uint64_t, 6> arguments{};
    std::array<std::uint64_t, 6> preserved{};
};

struct NativeTierRejectionEvent {
    std::string_view operation{};
    std::string_view detail{};
    std::string_view instruction{};
    std::uint64_t sequence{};
    std::uint64_t region_pc{};
    std::size_t instruction_count{};
    std::uint32_t family{};
    std::uint32_t declared_operands{};
    std::uint32_t visible_operands{};
    std::uint32_t decoded_operands{};
    std::uint32_t operand_kind{};
    std::uint32_t operand_width{};
    bool region_eligible{};
    bool instruction_eligible{};
    bool decode_failed{};
};

[[nodiscard]] GuestRegisterSnapshot CaptureGuestRegisters(
    const Executor::Jit::LsxMachineImage& state) noexcept;
[[nodiscard]] std::string RenderDispatchFailureEvent(
    const DispatchFailureEvent& event);
[[nodiscard]] std::string RenderGuestFaultEvent(const GuestFaultEvent& event);
[[nodiscard]] std::string RenderExitFrontierEvent(const ExitFrontierEvent& event);
[[nodiscard]] std::string RenderTranslationProbeResult(
    const TranslationProbeResult& result);
[[nodiscard]] std::string RenderDifferentialMismatchEvent(
    const DifferentialMismatchEvent& event);
[[nodiscard]] std::string RenderHleBoundaryEvent(const HleBoundaryEvent& event);
[[nodiscard]] std::string RenderNativeTierRejectionEvent(
    const NativeTierRejectionEvent& event);

[[nodiscard]] std::string RenderInstructionEvent(
    const Executor::Jit::LsxDecodedOp& instruction);
[[nodiscard]] std::string RenderRegisterEvent(
    const Executor::Jit::LsxMachineImage& state);

}
