// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/vector_lowering_policy.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/floating_relation.h"

#include <algorithm>
#include <array>
#include <ranges>

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

struct BlendDescriptor {
    X86Mnemonic mnemonic;
    VectorBlendPlan plan;
};

constexpr std::array kBlendDescriptors{
    BlendDescriptor{X86_MNEMONIC_BLENDPS, {BlendControlKind::Immediate, 4, false, false}},
    BlendDescriptor{X86_MNEMONIC_BLENDPD, {BlendControlKind::Immediate, 8, false, false}},
    BlendDescriptor{X86_MNEMONIC_PBLENDW, {BlendControlKind::Immediate, 2, false, true}},
    BlendDescriptor{X86_MNEMONIC_VBLENDPS, {BlendControlKind::Immediate, 4, true, false}},
    BlendDescriptor{X86_MNEMONIC_VBLENDPD, {BlendControlKind::Immediate, 8, true, false}},
    BlendDescriptor{X86_MNEMONIC_VPBLENDW, {BlendControlKind::Immediate, 2, true, true}},
    BlendDescriptor{X86_MNEMONIC_VPBLENDD, {BlendControlKind::Immediate, 4, true, false}},
    BlendDescriptor{X86_MNEMONIC_BLENDVPS, {BlendControlKind::VectorMask, 4, false, false}},
    BlendDescriptor{X86_MNEMONIC_BLENDVPD, {BlendControlKind::VectorMask, 8, false, false}},
    BlendDescriptor{X86_MNEMONIC_PBLENDVB, {BlendControlKind::VectorMask, 1, false, false}},
    BlendDescriptor{X86_MNEMONIC_VBLENDVPS, {BlendControlKind::VectorMask, 4, true, false}},
    BlendDescriptor{X86_MNEMONIC_VBLENDVPD, {BlendControlKind::VectorMask, 8, true, false}},
    BlendDescriptor{X86_MNEMONIC_VPBLENDVB, {BlendControlKind::VectorMask, 1, true, false}},
};

struct ElementTransferDescriptor {
    X86Mnemonic mnemonic;
    ElementTransferPlan plan;
};

constexpr std::array kElementTransfers{
    ElementTransferDescriptor{X86_MNEMONIC_PINSRB, {ElementTransferDirection::Insert, 1, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PINSRW, {ElementTransferDirection::Insert, 2, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PINSRD, {ElementTransferDirection::Insert, 4, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PINSRQ, {ElementTransferDirection::Insert, 8, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VPINSRB, {ElementTransferDirection::Insert, 1, true}},
    ElementTransferDescriptor{X86_MNEMONIC_VPINSRW, {ElementTransferDirection::Insert, 2, true}},
    ElementTransferDescriptor{X86_MNEMONIC_VPINSRD, {ElementTransferDirection::Insert, 4, true}},
    ElementTransferDescriptor{X86_MNEMONIC_VPINSRQ, {ElementTransferDirection::Insert, 8, true}},
    ElementTransferDescriptor{X86_MNEMONIC_PEXTRB, {ElementTransferDirection::Extract, 1, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PEXTRW, {ElementTransferDirection::Extract, 2, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PEXTRD, {ElementTransferDirection::Extract, 4, false}},
    ElementTransferDescriptor{X86_MNEMONIC_PEXTRQ, {ElementTransferDirection::Extract, 8, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VPEXTRB, {ElementTransferDirection::Extract, 1, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VPEXTRW, {ElementTransferDirection::Extract, 2, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VPEXTRD, {ElementTransferDirection::Extract, 4, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VPEXTRQ, {ElementTransferDirection::Extract, 8, false}},
    ElementTransferDescriptor{X86_MNEMONIC_EXTRACTPS, {ElementTransferDirection::Extract, 4, false}},
    ElementTransferDescriptor{X86_MNEMONIC_VEXTRACTPS, {ElementTransferDirection::Extract, 4, false}},
};

bool IsReadableVector(const LsxOperandRecord& operand) noexcept {
    return IsVectorRegister(operand) || operand.type == X86_OPERAND_TYPE_MEMORY;
}

std::optional<VectorBlendPlan> PlanVectorBlend(
    const X86Mnemonic mnemonic) noexcept {
    const auto found = std::ranges::find(
        kBlendDescriptors, mnemonic, &BlendDescriptor::mnemonic);
    return found == kBlendDescriptors.end()
               ? std::nullopt
               : std::optional{found->plan};
}

std::optional<ElementTransferPlan> PlanElementTransfer(
    const X86Mnemonic mnemonic) noexcept {
    const auto found = std::ranges::find(
        kElementTransfers, mnemonic, &ElementTransferDescriptor::mnemonic);
    return found == kElementTransfers.end()
               ? std::nullopt
               : std::optional{found->plan};
}

std::optional<MaskedTransferPlan> PlanMaskedTransfer(
    const LsxDecodedOp& instruction) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(instruction.mnemonic);
    const bool qword_lanes = mnemonic == X86_MNEMONIC_VMASKMOVPD ||
                             mnemonic == X86_MNEMONIC_VPMASKMOVQ;
    const bool dword_lanes = mnemonic == X86_MNEMONIC_VMASKMOVPS ||
                             mnemonic == X86_MNEMONIC_VPMASKMOVD;
    if ((!qword_lanes && !dword_lanes) || instruction.operand_count < 3 ||
        !IsVectorRegister(instruction.operands[1])) {
        return std::nullopt;
    }
    const bool writes_memory =
        instruction.operands[0].type == X86_OPERAND_TYPE_MEMORY;
    const bool valid_payload = writes_memory
        ? IsVectorRegister(instruction.operands[2])
        : IsVectorRegister(instruction.operands[0]) &&
              instruction.operands[2].type == X86_OPERAND_TYPE_MEMORY;
    return valid_payload
        ? std::optional{MaskedTransferPlan{
              static_cast<std::uint8_t>(qword_lanes ? 8u : 4u),
              writes_memory}}
        : std::nullopt;
}

bool IsScalarIntegerSource(const LsxOperandRecord& operand) noexcept {
    return operand.type == X86_OPERAND_TYPE_REGISTER ||
           operand.type == X86_OPERAND_TYPE_MEMORY;
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
    const auto& source = instruction.operands[1];
    if (!vector_or_memory(source)) {
        return std::nullopt;
    }

    const auto& control = instruction.operands[2];
    LanePermutationPlan plan{
        .source_operand = 1,
        .control_operand = 2,
        .element_bytes = element_bytes,
        .vector_bytes = instruction.operands[0].size / 8u,
    };
    if ((plan.vector_bytes != 16 && plan.vector_bytes != 32) ||
        source.size != instruction.operands[0].size) {
        return std::nullopt;
    }
    if (control.type == X86_OPERAND_TYPE_IMMEDIATE) {
        plan.immediate = static_cast<std::uint8_t>(control.imm.value.u);
        plan.control_source = LaneControlSource::Immediate;
        return plan;
    }
    if (!vector_or_memory(control) ||
        control.size != instruction.operands[0].size) {
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

std::optional<ScalarInstructionPlan> PlanScalarInstruction(
    const LsxDecodedOp& instruction) noexcept {
    const auto operation = static_cast<X86Mnemonic>(instruction.mnemonic);
    const auto enough_operands = [&](const std::uint8_t last) {
        return instruction.operand_count > last;
    };

    if (const auto arithmetic =
            Lsx4::Translation::DescribeScalarBinaryOperation(instruction.mnemonic)) {
        if (enough_operands(arithmetic->right_operand) &&
            IsVectorRegister(instruction.operands[0]) &&
            IsVectorRegister(instruction.operands[arithmetic->merge_operand]) &&
            IsReadableVector(instruction.operands[arithmetic->right_operand])) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::BinaryArithmetic,
                .merge_operand = arithmetic->merge_operand,
                .source_operand = arithmetic->right_operand,
                .double_precision = arithmetic->double_precision,
            };
        }
        return std::nullopt;
    }
    if (const auto root =
            Lsx4::Translation::DescribeScalarRootOperation(instruction.mnemonic)) {
        if (enough_operands(root->source_operand) &&
            IsVectorRegister(instruction.operands[0]) &&
            IsVectorRegister(instruction.operands[root->merge_operand]) &&
            IsReadableVector(instruction.operands[root->source_operand])) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::RootArithmetic,
                .merge_operand = root->merge_operand,
                .source_operand = root->source_operand,
                .double_precision = root->double_precision,
            };
        }
        return std::nullopt;
    }

    const bool vector_move = operation == X86_MNEMONIC_VMOVSS ||
                             operation == X86_MNEMONIC_VMOVSD;
    if (operation == X86_MNEMONIC_MOVSS || operation == X86_MNEMONIC_MOVSD ||
        vector_move) {
        const std::uint8_t source_index =
            static_cast<std::uint8_t>(vector_move && instruction.operand_count >= 3 ? 2 : 1);
        if (!enough_operands(source_index)) {
            return std::nullopt;
        }
        const auto& destination = instruction.operands[0];
        const auto& source = instruction.operands[source_index];
        const bool valid_destination = IsVectorRegister(destination) ||
                                       destination.type == X86_OPERAND_TYPE_MEMORY;
        if (!valid_destination || !IsReadableVector(source) ||
            (destination.type == X86_OPERAND_TYPE_MEMORY &&
             source.type == X86_OPERAND_TYPE_MEMORY)) {
            return std::nullopt;
        }
        return ScalarInstructionPlan{
            .family = ScalarInstructionFamily::MoveLane,
            .merge_operand = static_cast<std::uint8_t>(vector_move ? 1 : 0),
            .source_operand = source_index,
            .double_precision = operation == X86_MNEMONIC_MOVSD ||
                                operation == X86_MNEMONIC_VMOVSD,
        };
    }

    if (const auto conversion =
            Lsx4::Translation::DescribeIntegerToFloat(instruction.mnemonic)) {
        if (enough_operands(conversion->source_operand) &&
            IsVectorRegister(instruction.operands[0]) &&
            IsVectorRegister(instruction.operands[conversion->merge_operand]) &&
            IsScalarIntegerSource(instruction.operands[conversion->source_operand])) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::IntegerToFloating,
                .merge_operand = conversion->merge_operand,
                .source_operand = conversion->source_operand,
                .double_precision = conversion->double_precision,
            };
        }
        return std::nullopt;
    }
    if (const auto conversion =
            Lsx4::Translation::DescribeFloatWidthConversion(instruction.mnemonic)) {
        if (enough_operands(conversion->source_operand) &&
            IsVectorRegister(instruction.operands[0]) &&
            IsVectorRegister(instruction.operands[conversion->merge_operand]) &&
            IsReadableVector(instruction.operands[conversion->source_operand])) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::ChangeFloatingWidth,
                .merge_operand = conversion->merge_operand,
                .source_operand = conversion->source_operand,
                .double_precision = conversion->widen_to_double,
            };
        }
        return std::nullopt;
    }
    if (const auto conversion =
            Lsx4::Translation::DescribeFloatToInteger(instruction.mnemonic)) {
        if (instruction.operand_count >= 2 &&
            IsScalarIntegerSource(instruction.operands[0]) &&
            IsReadableVector(instruction.operands[1])) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::FloatingToInteger,
                .source_operand = 1,
                .double_precision = conversion->source_is_double,
                .uses_environment_rounding = conversion->honor_mxcsr_rounding,
            };
        }
        return std::nullopt;
    }

    const bool ordered_status = operation == X86_MNEMONIC_COMISS ||
        operation == X86_MNEMONIC_VCOMISS || operation == X86_MNEMONIC_COMISD ||
        operation == X86_MNEMONIC_VCOMISD || operation == X86_MNEMONIC_UCOMISS ||
        operation == X86_MNEMONIC_VUCOMISS || operation == X86_MNEMONIC_UCOMISD ||
        operation == X86_MNEMONIC_VUCOMISD;
    if (ordered_status && instruction.operand_count >= 2 &&
        IsVectorRegister(instruction.operands[0]) &&
        IsReadableVector(instruction.operands[1])) {
        return ScalarInstructionPlan{
            .family = ScalarInstructionFamily::StatusComparison,
            .source_operand = 1,
            .double_precision = operation == X86_MNEMONIC_COMISD ||
                operation == X86_MNEMONIC_VCOMISD || operation == X86_MNEMONIC_UCOMISD ||
                operation == X86_MNEMONIC_VUCOMISD,
        };
    }

    const bool legacy_predicate = operation == X86_MNEMONIC_CMPSS ||
                                  operation == X86_MNEMONIC_CMPSD;
    const bool vector_predicate = operation == X86_MNEMONIC_VCMPSS ||
                                  operation == X86_MNEMONIC_VCMPSD;
    if (legacy_predicate || vector_predicate) {
        const std::uint8_t rhs = static_cast<std::uint8_t>(vector_predicate ? 2 : 1);
        const std::uint8_t control = static_cast<std::uint8_t>(vector_predicate ? 3 : 2);
        if (enough_operands(control) && IsVectorRegister(instruction.operands[0]) &&
            (!vector_predicate || IsVectorRegister(instruction.operands[1])) &&
            IsReadableVector(instruction.operands[rhs]) &&
            instruction.operands[control].type == X86_OPERAND_TYPE_IMMEDIATE) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::PredicateComparison,
                .merge_operand = static_cast<std::uint8_t>(vector_predicate ? 1 : 0),
                .source_operand = rhs,
                .control_operand = control,
                .double_precision = operation == X86_MNEMONIC_CMPSD ||
                                    operation == X86_MNEMONIC_VCMPSD,
            };
        }
        return std::nullopt;
    }

    if (const auto rounding =
            Lsx4::Translation::DescribeScalarRound(instruction.mnemonic)) {
        if (enough_operands(rounding->control_operand) &&
            IsVectorRegister(instruction.operands[0]) &&
            IsVectorRegister(instruction.operands[rounding->merge_operand]) &&
            IsReadableVector(instruction.operands[rounding->source_operand]) &&
            instruction.operands[rounding->control_operand].type ==
                X86_OPERAND_TYPE_IMMEDIATE) {
            return ScalarInstructionPlan{
                .family = ScalarInstructionFamily::ControlledRounding,
                .merge_operand = rounding->merge_operand,
                .source_operand = rounding->source_operand,
                .control_operand = rounding->control_operand,
                .double_precision = rounding->double_precision,
            };
        }
    }
    return std::nullopt;
}

std::optional<VectorWritePlan> PlanVectorWrite(
    const DecodeSummary& instruction, const LsxOperandRecord& destination) noexcept {
    const bool ymm_register = destination.type == X86_OPERAND_TYPE_REGISTER &&
                              destination.reg.value >= X86_REGISTER_YMM0 &&
                              destination.reg.value <= X86_REGISTER_YMM15;
    const std::uint8_t payload_bytes =
        static_cast<std::uint8_t>(ymm_register || destination.size > 128 ? 32 : 16);
    if (IsVectorRegister(destination)) {
        const auto encoding = instruction.encoding;
        const bool vector_encoding = encoding == X86_INSTRUCTION_ENCODING_VEX ||
                                     encoding == X86_INSTRUCTION_ENCODING_EVEX ||
                                     encoding == X86_INSTRUCTION_ENCODING_MVEX ||
                                     encoding == X86_INSTRUCTION_ENCODING_XOP;
        return VectorWritePlan{
            .destination = VectorDestinationKind::RegisterFile,
            .payload_bytes = payload_bytes,
            .clear_register_tail = payload_bytes < 32 && vector_encoding,
        };
    }
    if (destination.type == X86_OPERAND_TYPE_MEMORY) {
        return VectorWritePlan{
            .destination = VectorDestinationKind::GuestMemory,
            .payload_bytes = payload_bytes,
        };
    }
    return std::nullopt;
}

}
