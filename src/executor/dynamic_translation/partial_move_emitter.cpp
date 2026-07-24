// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/partial_move_emitter.h"

#include <xbyak_aarch64/xbyak_aarch64.h>

namespace Executor::Jit {

void EmitPartialMovePayload(Xbyak_aarch64::CodeGenerator& code,
                            const PartialMovePlan& plan) {
    const auto load_state_qword = [&](const XReg& destination,
                                      const std::uint32_t offset) {
        code.ldr(destination, Xbyak_aarch64::ptr(code.x0, offset));
    };
    const auto store_state_qword = [&](const XReg& source,
                                       const std::uint32_t offset) {
        code.str(source, Xbyak_aarch64::ptr(code.x0, offset));
    };

    if (plan.writes_memory) {
        load_state_qword(code.x10, plan.source_offset);
        code.str(code.x10, Xbyak_aarch64::ptr(code.x11));
        return;
    }

    code.ldr(code.x10, Xbyak_aarch64::ptr(code.x11));
    if (plan.has_merge_lane) {
        load_state_qword(code.x12, plan.merge_offset);
        const std::uint32_t companion_lane = plan.destination_lane_offset == 0
            ? plan.destination_offset + sizeof(std::uint64_t)
            : plan.destination_offset - sizeof(std::uint64_t);
        store_state_qword(code.x12, companion_lane);
    }
    store_state_qword(code.x10, plan.destination_offset);
    if (plan.clears_upper_half) {
        const std::uint32_t register_base =
            plan.destination_offset - plan.destination_lane_offset;
        code.eor(code.v31.b16, code.v31.b16, code.v31.b16);
        const std::uint32_t upper_half_offset =
            register_base + 2u * static_cast<std::uint32_t>(sizeof(std::uint64_t));
        code.str(code.q31,
                 Xbyak_aarch64::ptr(code.x0, upper_half_offset));
    }
}

}
