// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/partial_move_plan.h"

namespace Xbyak_aarch64 {
class CodeGenerator;
}

namespace Executor::Jit {

void EmitPartialMovePayload(Xbyak_aarch64::CodeGenerator& code,
                            const PartialMovePlan& plan);

}
