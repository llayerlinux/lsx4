// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <span>

namespace Executor::Jit {

struct LsxDecodedOp;

enum class ByteLensResult : std::uint8_t {
    Accepted,
    Rejected,
    Unavailable,
};

[[nodiscard]] ByteLensResult LiftOneMachineInstruction(
    std::span<const std::uint8_t> bytes, LsxDecodedOp& operation) noexcept;

[[nodiscard]] bool HasMachineInstruction(std::span<const std::uint8_t> bytes) noexcept;

[[nodiscard]] const char* MachineRegisterName(std::uint32_t code) noexcept;

[[nodiscard]] bool UsesSignExtendedMoveImmediate(const LsxDecodedOp& operation) noexcept;

}
