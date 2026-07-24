// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/x86_decoder.h"

#include <xbyak_aarch64/xbyak_aarch64.h>

#include <cstdint>

namespace Executor::Jit::NativeCondition {

bool Materialize(Xbyak_aarch64::CodeGenerator& code, X86Mnemonic mnemonic,
                 const XReg& flags, const XReg& result,
                 const XReg& scratch_a, const XReg& scratch_b) noexcept;
bool Evaluate(std::uint64_t flags, X86Mnemonic mnemonic) noexcept;
bool EmitGuestBranchReturn(Xbyak_aarch64::CodeGenerator& code,
                           X86Mnemonic mnemonic,
                           std::uint32_t flags_state_offset,
                           std::uint64_t fallthrough,
                           std::uint64_t taken) noexcept;
void EmitGuestReturn(Xbyak_aarch64::CodeGenerator& code,
                     std::uint32_t stack_state_offset,
                     std::uint32_t stack_adjustment) noexcept;
void EmitLinkedGuestTail(Xbyak_aarch64::CodeGenerator& code,
                         Xbyak_aarch64::Label& linked,
                         Xbyak_aarch64::Label& unresolved,
                         const XReg& host_target,
                         std::uint64_t guest_target,
                         std::uint32_t guest_pc_offset) noexcept;
void EmitSharedFrameReturn(Xbyak_aarch64::CodeGenerator& code,
                           std::uint64_t result,
                           std::uint32_t frame_bytes) noexcept;

}
