// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/x86_decoder.h"

#include <xbyak_aarch64/xbyak_aarch64.h>

namespace Executor::Jit::NativeCondition {

bool Materialize(Xbyak_aarch64::CodeGenerator& code, X86Mnemonic mnemonic,
                 const XReg& flags, const XReg& result,
                 const XReg& scratch_a, const XReg& scratch_b) noexcept;

}
