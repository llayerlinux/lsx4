// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/retiring_execution_core.h"

#include <array>
#include <cstdint>

namespace Executor::Jit {

struct HleInvocationFrame {
    std::array<std::uint64_t, 6> integer_arguments{};
    std::array<std::uint64_t, 4> floating_arguments{};
    std::array<std::uint64_t, 6> preserved_registers{};

    [[nodiscard]] static HleInvocationFrame Capture(
        const LsxMachineImage& image) noexcept;

    void Commit(LsxMachineImage& image, std::uint64_t result,
                std::uint64_t stack_after, std::uint64_t next_pc) const noexcept;
};

void CommitSyntheticHleReturn(LsxMachineImage& image, std::uint64_t result,
                              std::uint64_t stack_after,
                              std::uint64_t next_pc) noexcept;

}
