// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/retiring_execution_core.h"

#include <optional>

namespace Executor::Jit {

enum class DirectVectorKernel : std::uint8_t {
    None,
    PackedBitTest,
    DuplicateEvenDwords,
    DuplicateOddDwords,
    DuplicateLowQwords,
};

struct DirectVectorPlan {
    DirectVectorKernel kernel{DirectVectorKernel::None};
    std::uint32_t width_bytes{};

    explicit operator bool() const noexcept {
        return kernel != DirectVectorKernel::None;
    }
};

DirectVectorPlan PlanDirectVectorKernel(const LsxDecodedOp& instruction) noexcept;

enum class LaneControlSource : std::uint8_t {
    Immediate,
    VectorValue,
};

struct LanePermutationPlan {
    std::uint8_t source_operand{};
    std::uint8_t control_operand{};
    std::uint8_t immediate{};
    std::uint8_t element_bytes{};
    std::uint32_t vector_bytes{};
    LaneControlSource control_source{LaneControlSource::Immediate};
};

[[nodiscard]] std::optional<LanePermutationPlan> PlanLanePermutation(
    const LsxDecodedOp& instruction) noexcept;

enum class ImmediateShuffleKind : std::uint8_t {
    Dwords,
    LowWords,
    HighWords,
    SinglePrecisionPairs,
    DoublePrecisionPairs,
};

struct ImmediateShufflePlan {
    std::uint8_t left_operand{};
    std::uint8_t right_operand{};
    std::uint8_t control{};
    std::uint32_t vector_bytes{};
    ImmediateShuffleKind kind{ImmediateShuffleKind::Dwords};
};

[[nodiscard]] std::optional<ImmediateShufflePlan> PlanImmediateShuffle(
    const LsxDecodedOp& instruction) noexcept;

enum class VectorLogicOperation : std::uint8_t {
    Unsupported,
    And,
    AndNot,
    Or,
    Xor,
};

[[nodiscard]] VectorLogicOperation ClassifyVectorLogic(
    const LsxDecodedOp& instruction) noexcept;

}
