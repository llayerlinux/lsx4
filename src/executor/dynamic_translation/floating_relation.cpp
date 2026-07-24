// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/floating_relation.h"

#include "common/x86_decoder.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace Lsx4::Translation {

namespace {
template <typename Float>
Float ApplyFloatingBinaryTyped(const Float left, const Float right,
                               const FloatingBinaryOperation operation) noexcept {
    if (operation == FloatingBinaryOperation::Minimum ||
        operation == FloatingBinaryOperation::Maximum) {
        if (std::isnan(left) || std::isnan(right)) {
            return right;
        }
        const bool choose_left = operation == FloatingBinaryOperation::Minimum
                                     ? left < right
                                     : left > right;
        return choose_left ? left : right;
    }
    switch (operation) {
    case FloatingBinaryOperation::Difference: return left - right;
    case FloatingBinaryOperation::Product: return left * right;
    case FloatingBinaryOperation::Quotient: return left / right;
    default: return left + right;
    }
}
}

float ApplyFloatingBinary(const float left, const float right,
                          const FloatingBinaryOperation operation) noexcept {
    return ApplyFloatingBinaryTyped(left, right, operation);
}

double ApplyFloatingBinary(const double left, const double right,
                           const FloatingBinaryOperation operation) noexcept {
    return ApplyFloatingBinaryTyped(left, right, operation);
}

FloatingOrder ClassifyFloatingOrder(const double left, const double right) noexcept {
    if (std::isunordered(left, right)) {
        return FloatingOrder::Unordered;
    }
    if (left < right) {
        return FloatingOrder::Less;
    }
    if (left > right) {
        return FloatingOrder::Greater;
    }
    return FloatingOrder::Equal;
}

FloatingCompareFlags FlagsForFloatingOrder(const FloatingOrder order) noexcept {
    switch (order) {
    case FloatingOrder::Unordered: return {true, true, true};
    case FloatingOrder::Less: return {false, false, true};
    case FloatingOrder::Equal: return {true, false, false};
    case FloatingOrder::Greater: return {};
    }
    return {};
}

bool FloatingRelationMatches(const double left, const double right,
                             const std::uint32_t immediate) noexcept {
    enum RelationBit : std::uint8_t {
        Unordered = 1u,
        Equal = 2u,
        Less = 4u,
        Greater = 8u,
    };

    constexpr std::array<std::uint8_t, 16> accepted_relations = {
        Equal,
        Less,
        static_cast<std::uint8_t>(Less | Equal),
        Unordered,
        static_cast<std::uint8_t>(Unordered | Less | Greater),
        static_cast<std::uint8_t>(Unordered | Equal | Greater),
        static_cast<std::uint8_t>(Unordered | Greater),
        static_cast<std::uint8_t>(Less | Equal | Greater),
        static_cast<std::uint8_t>(Unordered | Equal),
        static_cast<std::uint8_t>(Unordered | Less),
        static_cast<std::uint8_t>(Unordered | Less | Equal),
        0,
        static_cast<std::uint8_t>(Less | Greater),
        static_cast<std::uint8_t>(Equal | Greater),
        Greater,
        static_cast<std::uint8_t>(Unordered | Less | Equal | Greater),
    };

    std::uint8_t observed = 0;
    switch (ClassifyFloatingOrder(left, right)) {
    case FloatingOrder::Unordered:
        observed = Unordered;
        break;
    case FloatingOrder::Less:
        observed = Less;
        break;
    case FloatingOrder::Equal:
        observed = Equal;
        break;
    case FloatingOrder::Greater:
        observed = Greater;
        break;
    }
    return (accepted_relations[immediate & 0x0fu] & observed) != 0;
}

double RoundFloatingValue(const double value, const std::uint32_t mode) noexcept {
    constexpr std::array<double (*)(double), 4> operations = {
        static_cast<double (*)(double)>(std::nearbyint),
        static_cast<double (*)(double)>(std::floor),
        static_cast<double (*)(double)>(std::ceil),
        static_cast<double (*)(double)>(std::trunc),
    };
    return operations[mode & 3u](value);
}

std::optional<ScalarRootPlan> DescribeScalarRootOperation(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool sqrt_single = mnemonic == X86_MNEMONIC_SQRTSS ||
                             mnemonic == X86_MNEMONIC_VSQRTSS;
    const bool sqrt_double = mnemonic == X86_MNEMONIC_SQRTSD ||
                             mnemonic == X86_MNEMONIC_VSQRTSD;
    const bool reciprocal = mnemonic == X86_MNEMONIC_RCPSS ||
                            mnemonic == X86_MNEMONIC_VRCPSS;
    const bool reciprocal_root = mnemonic == X86_MNEMONIC_RSQRTSS ||
                                 mnemonic == X86_MNEMONIC_VRSQRTSS;
    if (!sqrt_single && !sqrt_double && !reciprocal && !reciprocal_root) {
        return std::nullopt;
    }

    const bool vector_prefix = mnemonic == X86_MNEMONIC_VSQRTSS ||
                               mnemonic == X86_MNEMONIC_VSQRTSD ||
                               mnemonic == X86_MNEMONIC_VRCPSS ||
                               mnemonic == X86_MNEMONIC_VRSQRTSS;
    return ScalarRootPlan{
        .operation = reciprocal_root ? ScalarRootOperation::ReciprocalSquareRoot
                                     : reciprocal ? ScalarRootOperation::Reciprocal
                                                  : ScalarRootOperation::SquareRoot,
        .merge_operand = static_cast<std::uint8_t>(vector_prefix ? 1u : 0u),
        .source_operand = static_cast<std::uint8_t>(vector_prefix ? 2u : 1u),
        .double_precision = sqrt_double,
    };
}

std::optional<ScalarBinaryPlan> DescribeScalarBinaryOperation(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    FloatingBinaryOperation operation{};
    switch (mnemonic) {
    case X86_MNEMONIC_ADDSS: case X86_MNEMONIC_ADDSD:
    case X86_MNEMONIC_VADDSS: case X86_MNEMONIC_VADDSD:
        operation = FloatingBinaryOperation::Sum; break;
    case X86_MNEMONIC_SUBSS: case X86_MNEMONIC_SUBSD:
    case X86_MNEMONIC_VSUBSS: case X86_MNEMONIC_VSUBSD:
        operation = FloatingBinaryOperation::Difference; break;
    case X86_MNEMONIC_MULSS: case X86_MNEMONIC_MULSD:
    case X86_MNEMONIC_VMULSS: case X86_MNEMONIC_VMULSD:
        operation = FloatingBinaryOperation::Product; break;
    case X86_MNEMONIC_DIVSS: case X86_MNEMONIC_DIVSD:
    case X86_MNEMONIC_VDIVSS: case X86_MNEMONIC_VDIVSD:
        operation = FloatingBinaryOperation::Quotient; break;
    case X86_MNEMONIC_MINSS: case X86_MNEMONIC_MINSD:
    case X86_MNEMONIC_VMINSS: case X86_MNEMONIC_VMINSD:
        operation = FloatingBinaryOperation::Minimum; break;
    case X86_MNEMONIC_MAXSS: case X86_MNEMONIC_MAXSD:
    case X86_MNEMONIC_VMAXSS: case X86_MNEMONIC_VMAXSD:
        operation = FloatingBinaryOperation::Maximum; break;
    default:
        return std::nullopt;
    }
    const bool vector_prefix = mnemonic == X86_MNEMONIC_VADDSS ||
        mnemonic == X86_MNEMONIC_VADDSD || mnemonic == X86_MNEMONIC_VSUBSS ||
        mnemonic == X86_MNEMONIC_VSUBSD || mnemonic == X86_MNEMONIC_VMULSS ||
        mnemonic == X86_MNEMONIC_VMULSD || mnemonic == X86_MNEMONIC_VDIVSS ||
        mnemonic == X86_MNEMONIC_VDIVSD || mnemonic == X86_MNEMONIC_VMINSS ||
        mnemonic == X86_MNEMONIC_VMINSD || mnemonic == X86_MNEMONIC_VMAXSS ||
        mnemonic == X86_MNEMONIC_VMAXSD;
    const bool double_precision = mnemonic == X86_MNEMONIC_ADDSD ||
        mnemonic == X86_MNEMONIC_SUBSD || mnemonic == X86_MNEMONIC_MULSD ||
        mnemonic == X86_MNEMONIC_DIVSD || mnemonic == X86_MNEMONIC_MINSD ||
        mnemonic == X86_MNEMONIC_MAXSD || mnemonic == X86_MNEMONIC_VADDSD ||
        mnemonic == X86_MNEMONIC_VSUBSD || mnemonic == X86_MNEMONIC_VMULSD ||
        mnemonic == X86_MNEMONIC_VDIVSD || mnemonic == X86_MNEMONIC_VMINSD ||
        mnemonic == X86_MNEMONIC_VMAXSD;
    return ScalarBinaryPlan{operation,
                            static_cast<std::uint8_t>(vector_prefix ? 1 : 0),
                            static_cast<std::uint8_t>(vector_prefix ? 2 : 1),
                            double_precision};
}

std::optional<IntegerToFloatPlan> DescribeIntegerToFloat(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool legacy = mnemonic == X86_MNEMONIC_CVTSI2SS ||
                        mnemonic == X86_MNEMONIC_CVTSI2SD;
    const bool vector = mnemonic == X86_MNEMONIC_VCVTSI2SS ||
                        mnemonic == X86_MNEMONIC_VCVTSI2SD;
    if (!legacy && !vector) {
        return std::nullopt;
    }
    return IntegerToFloatPlan{
        static_cast<std::uint8_t>(vector ? 1 : 0),
        static_cast<std::uint8_t>(vector ? 2 : 1),
        mnemonic == X86_MNEMONIC_CVTSI2SD ||
            mnemonic == X86_MNEMONIC_VCVTSI2SD};
}

std::optional<FloatWidthPlan> DescribeFloatWidthConversion(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool vector = mnemonic == X86_MNEMONIC_VCVTSS2SD ||
                        mnemonic == X86_MNEMONIC_VCVTSD2SS;
    const bool legacy = mnemonic == X86_MNEMONIC_CVTSS2SD ||
                        mnemonic == X86_MNEMONIC_CVTSD2SS;
    if (!legacy && !vector) {
        return std::nullopt;
    }
    return FloatWidthPlan{
        static_cast<std::uint8_t>(vector ? 1 : 0),
        static_cast<std::uint8_t>(vector ? 2 : 1),
        mnemonic == X86_MNEMONIC_CVTSS2SD ||
            mnemonic == X86_MNEMONIC_VCVTSS2SD};
}

std::optional<FloatToIntegerPlan> DescribeFloatToInteger(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool single = mnemonic == X86_MNEMONIC_CVTSS2SI ||
        mnemonic == X86_MNEMONIC_VCVTSS2SI || mnemonic == X86_MNEMONIC_CVTTSS2SI ||
        mnemonic == X86_MNEMONIC_VCVTTSS2SI;
    const bool dual = mnemonic == X86_MNEMONIC_CVTSD2SI ||
        mnemonic == X86_MNEMONIC_VCVTSD2SI || mnemonic == X86_MNEMONIC_CVTTSD2SI ||
        mnemonic == X86_MNEMONIC_VCVTTSD2SI;
    if (!single && !dual) {
        return std::nullopt;
    }
    const bool rounded = mnemonic == X86_MNEMONIC_CVTSS2SI ||
        mnemonic == X86_MNEMONIC_VCVTSS2SI || mnemonic == X86_MNEMONIC_CVTSD2SI ||
        mnemonic == X86_MNEMONIC_VCVTSD2SI;
    return FloatToIntegerPlan{dual, rounded};
}

std::optional<ScalarRoundPlan> DescribeScalarRound(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool vector = mnemonic == X86_MNEMONIC_VROUNDSS ||
                        mnemonic == X86_MNEMONIC_VROUNDSD;
    const bool legacy = mnemonic == X86_MNEMONIC_ROUNDSS ||
                        mnemonic == X86_MNEMONIC_ROUNDSD;
    if (!legacy && !vector) {
        return std::nullopt;
    }
    return ScalarRoundPlan{
        static_cast<std::uint8_t>(vector ? 1 : 0),
        static_cast<std::uint8_t>(vector ? 2 : 1),
        static_cast<std::uint8_t>(vector ? 3 : 2),
        mnemonic == X86_MNEMONIC_ROUNDSD ||
            mnemonic == X86_MNEMONIC_VROUNDSD};
}

double EvaluateScalarRoot(const ScalarRootOperation operation,
                          const double value) noexcept {
    switch (operation) {
    case ScalarRootOperation::SquareRoot:
        return std::sqrt(value);
    case ScalarRootOperation::Reciprocal:
        return 1.0 / value;
    case ScalarRootOperation::ReciprocalSquareRoot:
        return 1.0 / std::sqrt(value);
    }
    return value;
}

float EvaluateScalarRoot(const ScalarRootOperation operation,
                         const float value) noexcept {
    switch (operation) {
    case ScalarRootOperation::SquareRoot:
        return std::sqrt(value);
    case ScalarRootOperation::Reciprocal:
        return 1.0f / value;
    case ScalarRootOperation::ReciprocalSquareRoot:
        return 1.0f / std::sqrt(value);
    }
    return value;
}

}

namespace Lsx4::Translation {

bool EncodeSignedIntegerAsFloating(void* const destination,
                                   const std::uint64_t encoded_integer,
                                   const std::uint32_t integer_bits,
                                   const bool double_precision) noexcept {
    if (destination == nullptr || (integer_bits != 32 && integer_bits != 64)) {
        return false;
    }
    const std::int64_t value = integer_bits == 64
        ? static_cast<std::int64_t>(encoded_integer)
        : static_cast<std::int64_t>(static_cast<std::int32_t>(encoded_integer));
    if (double_precision) {
        const double converted = static_cast<double>(value);
        std::memcpy(destination, &converted, sizeof(converted));
    } else {
        const float converted = static_cast<float>(value);
        std::memcpy(destination, &converted, sizeof(converted));
    }
    return true;
}

bool ChangeScalarFloatingWidth(void* const destination,
                               const std::uint64_t source_bits,
                               const bool widen_to_double) noexcept {
    if (destination == nullptr) {
        return false;
    }
    if (widen_to_double) {
        const float source = std::bit_cast<float>(static_cast<std::uint32_t>(source_bits));
        const double converted = static_cast<double>(source);
        std::memcpy(destination, &converted, sizeof(converted));
    } else {
        const double source = std::bit_cast<double>(source_bits);
        const float converted = static_cast<float>(source);
        std::memcpy(destination, &converted, sizeof(converted));
    }
    return true;
}

bool EncodeFloatingRelationMask(void* const destination, const bool relation_holds,
                                const bool double_precision) noexcept {
    if (destination == nullptr) {
        return false;
    }
    if (double_precision) {
        const std::uint64_t mask = relation_holds ? UINT64_MAX : 0;
        std::memcpy(destination, &mask, sizeof(mask));
    } else {
        const std::uint32_t mask = relation_holds ? UINT32_MAX : 0;
        std::memcpy(destination, &mask, sizeof(mask));
    }
    return true;
}

bool EncodeRoundedScalar(void* const destination, const std::uint64_t source_bits,
                         const bool double_precision, const std::uint32_t mode) noexcept {
    if (destination == nullptr) {
        return false;
    }
    if (double_precision) {
        const double rounded = RoundFloatingValue(std::bit_cast<double>(source_bits), mode);
        std::memcpy(destination, &rounded, sizeof(rounded));
    } else {
        const float source = std::bit_cast<float>(static_cast<std::uint32_t>(source_bits));
        const float rounded = static_cast<float>(RoundFloatingValue(source, mode));
        std::memcpy(destination, &rounded, sizeof(rounded));
    }
    return true;
}

bool EncodeScalarBinaryOperation(void* const destination, const void* const left,
                                 const void* const right, const bool double_precision,
                                 const FloatingBinaryOperation operation) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr) {
        return false;
    }
    const bool selects_operand = operation == FloatingBinaryOperation::Minimum ||
                                 operation == FloatingBinaryOperation::Maximum;
    if (double_precision) {
        std::uint64_t left_bits{};
        std::uint64_t right_bits{};
        std::memcpy(&left_bits, left, sizeof(left_bits));
        std::memcpy(&right_bits, right, sizeof(right_bits));
        const double lhs = std::bit_cast<double>(left_bits);
        const double rhs = std::bit_cast<double>(right_bits);
        if (selects_operand) {
            const bool use_left = !std::isnan(lhs) && !std::isnan(rhs) &&
                (operation == FloatingBinaryOperation::Minimum ? lhs < rhs : lhs > rhs);
            const std::uint64_t selected = use_left ? left_bits : right_bits;
            std::memcpy(destination, &selected, sizeof(selected));
        } else {
            const double value = ApplyFloatingBinary(lhs, rhs, operation);
            std::memcpy(destination, &value, sizeof(value));
        }
        return true;
    }
    std::uint32_t left_bits{};
    std::uint32_t right_bits{};
    std::memcpy(&left_bits, left, sizeof(left_bits));
    std::memcpy(&right_bits, right, sizeof(right_bits));
    const float lhs = std::bit_cast<float>(left_bits);
    const float rhs = std::bit_cast<float>(right_bits);
    if (selects_operand) {
        const bool use_left = !std::isnan(lhs) && !std::isnan(rhs) &&
            (operation == FloatingBinaryOperation::Minimum ? lhs < rhs : lhs > rhs);
        const std::uint32_t selected = use_left ? left_bits : right_bits;
        std::memcpy(destination, &selected, sizeof(selected));
    } else {
        const float value = ApplyFloatingBinary(lhs, rhs, operation);
        std::memcpy(destination, &value, sizeof(value));
    }
    return true;
}

}
