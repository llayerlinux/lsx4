// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/vector_lowering_policy.h"

#include "common/x86_decoder.h"

namespace Executor::Jit {
namespace {

bool IsVectorRegister(const LsxOperandRecord& operand) noexcept {
    if (operand.type != X86_OPERAND_TYPE_REGISTER) {
        return false;
    }
    const LsxRegisterCode reg = operand.reg.value;
    return (reg >= X86_REGISTER_XMM0 && reg <= X86_REGISTER_XMM15) ||
           (reg >= X86_REGISTER_YMM0 && reg <= X86_REGISTER_YMM15);
}

}

DirectVectorPlan PlanDirectVectorKernel(
    const LsxDecodedOp& instruction) noexcept {
    if (instruction.operand_count < 2 ||
        !IsVectorRegister(instruction.operands[0])) {
        return {};
    }
    const auto& source = instruction.operands[1];
    const bool source_is_vector = IsVectorRegister(source);
    const bool source_is_memory =
        source.type == X86_OPERAND_TYPE_MEMORY;
    const std::uint32_t width_bits = instruction.operands[0].size;
    if ((width_bits != 128 && width_bits != 256) ||
        (!source_is_vector && !source_is_memory) ||
        source.size != width_bits) {
        return {};
    }

    switch (static_cast<X86Mnemonic>(instruction.mnemonic)) {
    case X86_MNEMONIC_PTEST:
    case X86_MNEMONIC_VPTEST:
        return {DirectVectorKernel::PackedBitTest, width_bits / 8};
    case X86_MNEMONIC_MOVSLDUP:
    case X86_MNEMONIC_VMOVSLDUP:
        return {DirectVectorKernel::DuplicateEvenDwords, width_bits / 8};
    case X86_MNEMONIC_MOVSHDUP:
    case X86_MNEMONIC_VMOVSHDUP:
        return {DirectVectorKernel::DuplicateOddDwords, width_bits / 8};
    case X86_MNEMONIC_MOVDDUP:
    case X86_MNEMONIC_VMOVDDUP:
        return {DirectVectorKernel::DuplicateLowQwords, width_bits / 8};
    default:
        return {};
    }
}

std::optional<LanePermutationPlan> PlanLanePermutation(
    const LsxDecodedOp& instruction) noexcept {
    const auto opcode = static_cast<X86Mnemonic>(instruction.mnemonic);
    const std::uint8_t element_bytes = opcode == X86_MNEMONIC_VPERMILPD ? 8u
                                       : opcode == X86_MNEMONIC_VPERMILPS ? 4u
                                                                         : 0u;
    if (element_bytes == 0 || instruction.operand_count != 3 ||
        !IsVectorRegister(instruction.operands[0])) {
        return std::nullopt;
    }
    const auto vector_or_memory = [](const LsxOperandRecord& operand) {
        return IsVectorRegister(operand) || operand.type == X86_OPERAND_TYPE_MEMORY;
    };
    if (!vector_or_memory(instruction.operands[1])) {
        return std::nullopt;
    }

    const auto& control = instruction.operands[2];
    LanePermutationPlan plan{
        .source_operand = 1,
        .control_operand = 2,
        .element_bytes = element_bytes,
        .vector_bytes = instruction.operands[0].size / 8u,
    };
    if (control.type == X86_OPERAND_TYPE_IMMEDIATE) {
        plan.immediate = static_cast<std::uint8_t>(control.imm.value.u);
        plan.control_source = LaneControlSource::Immediate;
        return plan;
    }
    if (!vector_or_memory(control)) {
        return std::nullopt;
    }
    plan.control_source = LaneControlSource::VectorValue;
    return plan;
}

std::optional<ImmediateShufflePlan> PlanImmediateShuffle(
    const LsxDecodedOp& instruction) noexcept {
    const auto opcode = static_cast<X86Mnemonic>(instruction.mnemonic);
    ImmediateShuffleKind kind{};
    bool unary = false;
    bool vex_binary = false;
    switch (opcode) {
    case X86_MNEMONIC_PSHUFD:
    case X86_MNEMONIC_VPSHUFD:
        kind = ImmediateShuffleKind::Dwords;
        unary = true;
        break;
    case X86_MNEMONIC_PSHUFLW:
    case X86_MNEMONIC_VPSHUFLW:
        kind = ImmediateShuffleKind::LowWords;
        unary = true;
        break;
    case X86_MNEMONIC_PSHUFHW:
    case X86_MNEMONIC_VPSHUFHW:
        kind = ImmediateShuffleKind::HighWords;
        unary = true;
        break;
    case X86_MNEMONIC_SHUFPS:
        kind = ImmediateShuffleKind::SinglePrecisionPairs;
        break;
    case X86_MNEMONIC_VSHUFPS:
        kind = ImmediateShuffleKind::SinglePrecisionPairs;
        vex_binary = true;
        break;
    case X86_MNEMONIC_SHUFPD:
        kind = ImmediateShuffleKind::DoublePrecisionPairs;
        break;
    case X86_MNEMONIC_VSHUFPD:
        kind = ImmediateShuffleKind::DoublePrecisionPairs;
        vex_binary = true;
        break;
    default:
        return std::nullopt;
    }

    const std::uint8_t control_index = vex_binary ? 3u : 2u;
    if (instruction.operand_count <= control_index ||
        !IsVectorRegister(instruction.operands[0]) ||
        instruction.operands[control_index].type != X86_OPERAND_TYPE_IMMEDIATE) {
        return std::nullopt;
    }
    const std::uint8_t left_index = unary ? 1u : (vex_binary ? 1u : 0u);
    const std::uint8_t right_index = unary ? 1u : (vex_binary ? 2u : 1u);
    const auto readable = [](const LsxOperandRecord& operand) {
        return IsVectorRegister(operand) || operand.type == X86_OPERAND_TYPE_MEMORY;
    };
    if (!readable(instruction.operands[right_index]) ||
        (!unary && !readable(instruction.operands[left_index]))) {
        return std::nullopt;
    }
    return ImmediateShufflePlan{
        .left_operand = left_index,
        .right_operand = right_index,
        .control = static_cast<std::uint8_t>(
            instruction.operands[control_index].imm.value.u),
        .vector_bytes = instruction.operands[0].size / 8u,
        .kind = kind,
    };
}

VectorLogicOperation ClassifyVectorLogic(
    const LsxDecodedOp& instruction) noexcept {
    switch (static_cast<X86Mnemonic>(instruction.mnemonic)) {
    case X86_MNEMONIC_ANDPS:
    case X86_MNEMONIC_ANDPD:
    case X86_MNEMONIC_PAND:
    case X86_MNEMONIC_VANDPS:
    case X86_MNEMONIC_VANDPD:
    case X86_MNEMONIC_VPAND:
    case X86_MNEMONIC_VPANDD:
    case X86_MNEMONIC_VPANDQ:
        return VectorLogicOperation::And;
    case X86_MNEMONIC_ANDNPS:
    case X86_MNEMONIC_ANDNPD:
    case X86_MNEMONIC_PANDN:
    case X86_MNEMONIC_VANDNPS:
    case X86_MNEMONIC_VANDNPD:
    case X86_MNEMONIC_VPANDN:
    case X86_MNEMONIC_VPANDND:
    case X86_MNEMONIC_VPANDNQ:
        return VectorLogicOperation::AndNot;
    case X86_MNEMONIC_ORPS:
    case X86_MNEMONIC_ORPD:
    case X86_MNEMONIC_POR:
    case X86_MNEMONIC_VORPS:
    case X86_MNEMONIC_VORPD:
    case X86_MNEMONIC_VPOR:
    case X86_MNEMONIC_VPORD:
    case X86_MNEMONIC_VPORQ:
        return VectorLogicOperation::Or;
    case X86_MNEMONIC_XORPS:
    case X86_MNEMONIC_XORPD:
    case X86_MNEMONIC_PXOR:
    case X86_MNEMONIC_VXORPS:
    case X86_MNEMONIC_VXORPD:
    case X86_MNEMONIC_VPXOR:
    case X86_MNEMONIC_VPXORD:
    case X86_MNEMONIC_VPXORQ:
        return VectorLogicOperation::Xor;
    default:
        return VectorLogicOperation::Unsupported;
    }
}

}
