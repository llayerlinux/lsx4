// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/instruction_diagnostics.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/retiring_execution_core.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace Lsx4::Translation {
namespace {

const char* RegisterLabel(const std::uint32_t code) noexcept {
    const char* label = X86RegisterGetString(code);
    return label == nullptr ? "?" : label;
}

}

GuestRegisterSnapshot CaptureGuestRegisters(
    const Executor::Jit::LsxMachineImage& state) noexcept {
    GuestRegisterSnapshot snapshot{};
    snapshot.integer = state.gpr;
    snapshot.condition = state.rflags;
    snapshot.fs_origin = state.fs_base;
    snapshot.gs_origin = state.gs_base;
    return snapshot;
}

namespace {

void AppendMachine(std::ostringstream& output, const GuestRegisterSnapshot& machine) {
    static constexpr std::array<std::string_view, 16> names{
        "a", "c", "d", "b", "sp", "bp", "si", "di",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"};
    output << "machine={condition:" << std::showbase << std::hex << machine.condition;
    for (std::size_t index = 0; index < machine.integer.size(); ++index) {
        output << ',' << names[index] << ':' << machine.integer[index];
    }
    output << ",fs:" << machine.fs_origin << ",gs:" << machine.gs_origin
           << std::noshowbase << std::dec << '}';
}

}

std::string RenderDispatchFailureEvent(const DispatchFailureEvent& event) {
    std::ostringstream output;
    output << "translation.dispatch_failure{reason=\"" << event.reason
           << "\",region={pc:" << std::showbase << std::hex << event.region_pc
           << ",bytes:" << event.region_bytes << ",traits:" << event.region_traits
           << ",encoding:\"" << event.encoding << "\"},predecessor={pc:"
           << event.predecessor_pc << ",bytes:" << event.predecessor_bytes
           << ",outcome:" << event.predecessor_outcome
           << ",state_pc:" << event.predecessor_state_pc
           << ",successor:" << event.predecessor_successor << "},";
    AppendMachine(output, event.machine);
    output << std::noshowbase << std::dec << '}';
    return output.str();
}

std::string RenderGuestFaultEvent(const GuestFaultEvent& event) {
    std::ostringstream output;
    output << "translation.guest_fault{operation=\"" << event.operation
           << "\",memory={address:" << std::showbase << std::hex << event.address
           << ",bytes:" << event.byte_count << "},dispatch_pc:" << event.dispatch_pc
           << std::noshowbase << std::dec << '}';
    return output.str();
}

std::string RenderExitFrontierEvent(const ExitFrontierEvent& event) {
    std::ostringstream output;
    output << "translation.exit_frontier{retired=" << std::dec << event.retired_regions
           << ",region={pc:" << std::showbase << std::hex << event.region_pc
           << ",bytes:" << event.region_bytes << ",traits:" << event.region_traits
           << ",encoding:\"" << event.encoding << "\",summary:\""
           << event.region_summary << "\"},outcome:" << event.outcome
           << ",completion={pending:" << std::boolalpha << event.completion_pending
           << ",value:" << std::showbase << std::hex << event.completion_value << "},";
    AppendMachine(output, event.machine);
    output << std::noboolalpha << std::noshowbase << std::dec << '}';
    return output.str();
}

std::string RenderTranslationProbeResult(const TranslationProbeResult& result) {
    std::ostringstream output;
    output << R"({"execution":{"translatedReturn":"0x)" << std::hex
           << result.translated_return << R"(","bridge":{"compiled":)"
           << (result.bridge_compiled ? "true" : "false")
           << R"(,"successor":"0x)" << result.bridge_successor
           << R"("},"machineAccumulator":)" << std::dec << result.machine_accumulator
           << R"(,"replay":{"lastPc":"0x)" << std::hex << result.replay_last_pc
           << R"(","accumulator":)" << std::dec << result.replay_accumulator
           << R"(,"passed":)" << (result.replay_passed ? "true" : "false")
           << R"(}},"smc":{"initialResult":)" << result.smc_initial_result
           << R"(,"mutatedResult":)" << result.smc_mutated_result << "}}";
    return output.str();
}

std::string RenderDifferentialMismatchEvent(
    const DifferentialMismatchEvent& event) {
    std::ostringstream output;
    output << "translation.differential_mismatch{region={pc:" << std::showbase
           << std::hex << event.region_pc << ",summary:\""
           << event.region_summary << "\"},native={successor:"
           << event.native_successor << ",outcome:" << event.native_outcome << ',';
    AppendMachine(output, event.native_machine);
    output << "},semantic={successor:" << event.semantic_successor
           << ",outcome:" << event.semantic_outcome << ',';
    AppendMachine(output, event.semantic_machine);
    output << "}}" << std::noshowbase << std::dec;
    return output.str();
}

std::string RenderHleBoundaryEvent(const HleBoundaryEvent& event) {
    std::ostringstream output;
    output << "translation.hle_boundary{sequence=" << std::dec << event.sequence
           << ",mode=\"" << event.mode << "\",call={thunk:" << std::showbase
           << std::hex << event.thunk << ",host:" << event.host_function
           << ",continuation:" << event.continuation << "},stack={input:"
           << event.stack_input << ",output:" << event.stack_output << "},args=[";
    for (std::size_t index = 0; index < event.arguments.size(); ++index) {
        output << (index == 0 ? "" : ",") << event.arguments[index];
    }
    output << "],preserved=[";
    for (std::size_t index = 0; index < event.preserved.size(); ++index) {
        output << (index == 0 ? "" : ",") << event.preserved[index];
    }
    output << "]}" << std::noshowbase << std::dec;
    return output.str();
}

std::string RenderNativeTierRejectionEvent(
    const NativeTierRejectionEvent& event) {
    std::ostringstream output;
    output << "translation.native_tier_rejection{sequence=" << event.sequence
           << ",region={pc:" << std::showbase << std::hex << event.region_pc
           << ",instructions:" << std::dec << event.instruction_count
           << ",eligible:" << std::boolalpha << event.region_eligible
           << "},operation=\"" << event.operation << "\",instruction={eligible:"
           << event.instruction_eligible << ",family:" << event.family
           << ",operands:{declared:" << event.declared_operands
           << ",visible:" << event.visible_operands
           << ",decoded:" << event.decoded_operands << ",first_kind:"
           << event.operand_kind << ",first_width:" << event.operand_width
           << "},decode_failed:" << event.decode_failed << ",summary:\""
           << event.instruction << "\"},detail=\"" << event.detail << "\"}"
           << std::noboolalpha << std::noshowbase;
    return output.str();
}

std::string RenderInstructionEvent(const Executor::Jit::LsxDecodedOp& instruction) {
    std::ostringstream event;
    const char* operation =
        X86MnemonicGetString(static_cast<X86Mnemonic>(instruction.mnemonic));
    event << "{pc=" << std::hex << instruction.guest_rip
          << ",op=" << (operation == nullptr ? "?" : operation)
          << ",raw=";
    for (std::size_t index = 0;
         index < std::min<std::size_t>(instruction.length, instruction.bytes.size());
         ++index) {
        event << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(instruction.bytes[index]);
    }
    event << std::dec << ",args=[";
    const std::size_t count =
        std::min<std::size_t>(instruction.operand_count, instruction.operands.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            event << '|';
        }
        const auto& argument = instruction.operands[index];
        event << static_cast<unsigned>(argument.type) << '/' << argument.size;
        if (argument.type == X86_OPERAND_TYPE_REGISTER) {
            event << "@" << RegisterLabel(argument.reg.value);
        } else if (argument.type == X86_OPERAND_TYPE_MEMORY) {
            event << "@[" << RegisterLabel(argument.mem.segment) << ':'
                  << RegisterLabel(argument.mem.base) << '+'
                  << RegisterLabel(argument.mem.index) << '*'
                  << static_cast<unsigned>(argument.mem.scale) << '+'
                  << std::showbase << std::hex << argument.mem.disp.value
                  << std::noshowbase << std::dec << ']';
        } else if (argument.type == X86_OPERAND_TYPE_IMMEDIATE) {
            event << "@" << std::showbase << std::hex
                  << (argument.imm.is_signed
                          ? static_cast<std::uint64_t>(argument.imm.value.s)
                          : argument.imm.value.u)
                  << std::noshowbase << std::dec;
        }
    }
    event << "]}";
    return event.str();
}

std::string RenderRegisterEvent(const Executor::Jit::LsxMachineImage& state) {
    static constexpr std::array<const char*, 16> labels = {
        "a", "c", "d", "b", "sp", "bp", "si", "di",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    };
    std::ostringstream event;
    event << " registers={condition:" << std::showbase << std::hex << state.rflags;
    for (std::size_t index = 0; index < state.gpr.size(); ++index) {
        event << ',' << labels[index] << ':' << state.gpr[index];
    }
    event << ",fs:" << state.fs_base << ",gs:" << state.gs_base
          << std::noshowbase << std::dec << '}';
    return event.str();
}

}
