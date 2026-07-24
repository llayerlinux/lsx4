// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/floating_operation.h"

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

enum class FloatingOrder : std::uint8_t {
    Unordered,
    Less,
    Equal,
    Greater,
};

struct FloatingCompareFlags {
    bool zero{};
    bool parity{};
    bool carry{};
};

enum class ScalarRootOperation : std::uint8_t {
    SquareRoot,
    Reciprocal,
    ReciprocalSquareRoot,
};

struct ScalarRootPlan {
    ScalarRootOperation operation{};
    std::uint8_t merge_operand{};
    std::uint8_t source_operand{};
    bool double_precision{};
};

struct ScalarBinaryPlan {
    FloatingBinaryOperation operation{};
    std::uint8_t merge_operand{};
    std::uint8_t right_operand{};
    bool double_precision{};
};

struct IntegerToFloatPlan {
    std::uint8_t merge_operand{};
    std::uint8_t source_operand{};
    bool double_precision{};
};

struct FloatWidthPlan {
    std::uint8_t merge_operand{};
    std::uint8_t source_operand{};
    bool widen_to_double{};
};

struct FloatToIntegerPlan {
    bool source_is_double{};
    bool honor_mxcsr_rounding{};
};

struct ScalarRoundPlan {
    std::uint8_t merge_operand{};
    std::uint8_t source_operand{};
    std::uint8_t control_operand{};
    bool double_precision{};
};

[[nodiscard]] FloatingOrder ClassifyFloatingOrder(double left, double right) noexcept;
[[nodiscard]] FloatingCompareFlags FlagsForFloatingOrder(
    FloatingOrder order) noexcept;

[[nodiscard]] bool FloatingRelationMatches(double left, double right,
                                           std::uint32_t immediate) noexcept;

[[nodiscard]] double RoundFloatingValue(double value, std::uint32_t mode) noexcept;

bool EncodeSignedIntegerAsFloating(void* destination, std::uint64_t encoded_integer,
                                   std::uint32_t integer_bits,
                                   bool double_precision) noexcept;
bool ChangeScalarFloatingWidth(void* destination, std::uint64_t source_bits,
                               bool widen_to_double) noexcept;
bool EncodeFloatingRelationMask(void* destination, bool relation_holds,
                                bool double_precision) noexcept;
bool EncodeRoundedScalar(void* destination, std::uint64_t source_bits,
                         bool double_precision, std::uint32_t mode) noexcept;
bool EncodeScalarBinaryOperation(void* destination, const void* left,
                                 const void* right, bool double_precision,
                                 FloatingBinaryOperation operation) noexcept;

[[nodiscard]] std::optional<ScalarRootPlan> DescribeScalarRootOperation(
    std::uint32_t mnemonic) noexcept;
[[nodiscard]] std::optional<ScalarBinaryPlan> DescribeScalarBinaryOperation(
    std::uint32_t mnemonic) noexcept;
[[nodiscard]] std::optional<IntegerToFloatPlan> DescribeIntegerToFloat(
    std::uint32_t mnemonic) noexcept;
[[nodiscard]] std::optional<FloatWidthPlan> DescribeFloatWidthConversion(
    std::uint32_t mnemonic) noexcept;
[[nodiscard]] std::optional<FloatToIntegerPlan> DescribeFloatToInteger(
    std::uint32_t mnemonic) noexcept;
[[nodiscard]] std::optional<ScalarRoundPlan> DescribeScalarRound(
    std::uint32_t mnemonic) noexcept;

[[nodiscard]] double EvaluateScalarRoot(ScalarRootOperation operation,
                                        double value) noexcept;
[[nodiscard]] float EvaluateScalarRoot(ScalarRootOperation operation,
                                       float value) noexcept;

}
