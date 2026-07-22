// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/hle_invocation_frame.h"

#include <cstring>

namespace Executor::Jit {
namespace {

constexpr std::array kArgumentRegisters{
    LsxGpr::Rdi, LsxGpr::Rsi, LsxGpr::Rdx,
    LsxGpr::Rcx, LsxGpr::R8, LsxGpr::R9,
};
constexpr std::array kPreservedRegisters{
    LsxGpr::Rbx, LsxGpr::Rbp, LsxGpr::R12,
    LsxGpr::R13, LsxGpr::R14, LsxGpr::R15,
};

}

HleInvocationFrame HleInvocationFrame::Capture(
    const LsxMachineImage& image) noexcept {
    HleInvocationFrame frame{};
    for (std::size_t index = 0; index != kArgumentRegisters.size(); ++index) {
        frame.integer_arguments[index] =
            ReadGuestGpr64(image, kArgumentRegisters[index]);
    }
    for (std::size_t index = 0; index != frame.floating_arguments.size(); ++index) {
        std::memcpy(&frame.floating_arguments[index], image.ymm[index].data(),
                    sizeof(frame.floating_arguments[index]));
    }
    for (std::size_t index = 0; index != kPreservedRegisters.size(); ++index) {
        frame.preserved_registers[index] =
            ReadGuestGpr64(image, kPreservedRegisters[index]);
    }
    return frame;
}

void HleInvocationFrame::Commit(LsxMachineImage& image, const std::uint64_t result,
                                const std::uint64_t stack_after,
                                const std::uint64_t next_pc) const noexcept {
    for (std::size_t index = 0; index != kPreservedRegisters.size(); ++index) {
        WriteGuestGpr64(image, kPreservedRegisters[index], preserved_registers[index]);
    }
    CommitSyntheticHleReturn(image, result, stack_after, next_pc);
}

void CommitSyntheticHleReturn(LsxMachineImage& image, const std::uint64_t result,
                              const std::uint64_t stack_after,
                              const std::uint64_t next_pc) noexcept {
    WriteGuestGpr64(image, LsxGpr::Rax, result);
    WriteGuestGpr64(image, LsxGpr::Rsp, stack_after);
    image.rip_or_exit = next_pc;
}

}
