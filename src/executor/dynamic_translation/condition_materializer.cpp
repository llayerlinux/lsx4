// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/condition_materializer.h"

#include <cstdint>
#include <array>
#include <optional>

namespace Executor::Jit::NativeCondition {
namespace {

enum class Relation : std::uint8_t { Flag, EitherFlag, DifferentFlags, DifferentOrFlag };

struct Recipe {
    Relation relation{};
    std::uint8_t first_bit{};
    std::uint8_t second_bit{};
    std::uint8_t third_bit{};
    bool invert{};
};

constexpr std::uint8_t kCarry = 0;
constexpr std::uint8_t kParity = 2;
constexpr std::uint8_t kZero = 6;
constexpr std::uint8_t kSign = 7;
constexpr std::uint8_t kOverflow = 11;

std::optional<Recipe> SelectRecipe(const X86Mnemonic mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_JB: case X86_MNEMONIC_CMOVB: case X86_MNEMONIC_SETB:
        return Recipe{Relation::Flag, kCarry};
    case X86_MNEMONIC_JBE: case X86_MNEMONIC_CMOVBE: case X86_MNEMONIC_SETBE:
        return Recipe{Relation::EitherFlag, kCarry, kZero};
    case X86_MNEMONIC_JL: case X86_MNEMONIC_CMOVL: case X86_MNEMONIC_SETL:
        return Recipe{Relation::DifferentFlags, kSign, kOverflow};
    case X86_MNEMONIC_JLE: case X86_MNEMONIC_CMOVLE: case X86_MNEMONIC_SETLE:
        return Recipe{Relation::DifferentOrFlag, kSign, kOverflow, kZero};
    case X86_MNEMONIC_JNB: case X86_MNEMONIC_CMOVNB: case X86_MNEMONIC_SETNB:
        return Recipe{Relation::Flag, kCarry, 0, 0, true};
    case X86_MNEMONIC_JNBE: case X86_MNEMONIC_CMOVNBE: case X86_MNEMONIC_SETNBE:
        return Recipe{Relation::EitherFlag, kCarry, kZero, 0, true};
    case X86_MNEMONIC_JNL: case X86_MNEMONIC_CMOVNL: case X86_MNEMONIC_SETNL:
        return Recipe{Relation::DifferentFlags, kSign, kOverflow, 0, true};
    case X86_MNEMONIC_JNLE: case X86_MNEMONIC_CMOVNLE: case X86_MNEMONIC_SETNLE:
        return Recipe{Relation::DifferentOrFlag, kSign, kOverflow, kZero, true};
    case X86_MNEMONIC_JNO: case X86_MNEMONIC_CMOVNO: case X86_MNEMONIC_SETNO:
        return Recipe{Relation::Flag, kOverflow, 0, 0, true};
    case X86_MNEMONIC_JNP: case X86_MNEMONIC_CMOVNP: case X86_MNEMONIC_SETNP:
        return Recipe{Relation::Flag, kParity, 0, 0, true};
    case X86_MNEMONIC_JNS: case X86_MNEMONIC_CMOVNS: case X86_MNEMONIC_SETNS:
        return Recipe{Relation::Flag, kSign, 0, 0, true};
    case X86_MNEMONIC_JNZ: case X86_MNEMONIC_CMOVNZ: case X86_MNEMONIC_SETNZ:
        return Recipe{Relation::Flag, kZero, 0, 0, true};
    case X86_MNEMONIC_JO: case X86_MNEMONIC_CMOVO: case X86_MNEMONIC_SETO:
        return Recipe{Relation::Flag, kOverflow};
    case X86_MNEMONIC_JP: case X86_MNEMONIC_CMOVP: case X86_MNEMONIC_SETP:
        return Recipe{Relation::Flag, kParity};
    case X86_MNEMONIC_JS: case X86_MNEMONIC_CMOVS: case X86_MNEMONIC_SETS:
        return Recipe{Relation::Flag, kSign};
    case X86_MNEMONIC_JZ: case X86_MNEMONIC_CMOVZ: case X86_MNEMONIC_SETZ:
        return Recipe{Relation::Flag, kZero};
    default:
        return std::nullopt;
    }
}

}

bool Materialize(Xbyak_aarch64::CodeGenerator& code, const X86Mnemonic mnemonic,
                 const XReg& flags, const XReg& result,
                 const XReg& scratch_a, const XReg& scratch_b) noexcept {
    const auto recipe = SelectRecipe(mnemonic);
    if (!recipe) {
        return false;
    }
    code.ubfx(scratch_a, flags, recipe->first_bit, 1);
    if (recipe->relation == Relation::Flag) {
        code.mov(result, scratch_a);
    } else {
        code.ubfx(scratch_b, flags, recipe->second_bit, 1);
        if (recipe->relation == Relation::EitherFlag) {
            code.orr(result, scratch_a, scratch_b);
        } else {
            code.eor(result, scratch_a, scratch_b);
            if (recipe->relation == Relation::DifferentOrFlag) {
                code.ubfx(scratch_b, flags, recipe->third_bit, 1);
                code.orr(result, result, scratch_b);
            }
        }
    }
    if (recipe->invert) {
        code.eor(result, result, 1);
    }
    return true;
}

bool Evaluate(const std::uint64_t flags, const X86Mnemonic mnemonic) noexcept {
    const auto recipe = SelectRecipe(mnemonic);
    if (!recipe) {
        return false;
    }
    const auto bit = [flags](const std::uint8_t position) {
        return ((flags >> position) & 1u) != 0;
    };
    const bool first = bit(recipe->first_bit);
    const bool second = bit(recipe->second_bit);
    bool result = first;
    switch (recipe->relation) {
    case Relation::Flag:
        break;
    case Relation::EitherFlag:
        result = first || second;
        break;
    case Relation::DifferentFlags:
        result = first != second;
        break;
    case Relation::DifferentOrFlag:
        result = first != second || bit(recipe->third_bit);
        break;
    }
    return result != recipe->invert;
}

bool EmitGuestBranchReturn(Xbyak_aarch64::CodeGenerator& code,
                           const X86Mnemonic mnemonic,
                           const std::uint32_t flags_state_offset,
                           const std::uint64_t fallthrough,
                           const std::uint64_t taken) noexcept {
    code.ldr(code.x9, Xbyak_aarch64::ptr(code.x0, flags_state_offset));
    if (!Materialize(code, mnemonic, code.x9, code.x10, code.x11, code.x12)) {
        return false;
    }
    code.mov(code.x11, fallthrough);
    code.mov(code.x12, taken);
    code.cmp(code.x10, 0);
    code.csel(code.x0, code.x12, code.x11, Xbyak_aarch64::NE);
    code.ret();
    return true;
}

void EmitGuestReturn(Xbyak_aarch64::CodeGenerator& code,
                     const std::uint32_t stack_state_offset,
                     const std::uint32_t stack_adjustment) noexcept {
    code.ldr(code.x8, Xbyak_aarch64::ptr(code.x0, stack_state_offset));
    code.ldr(code.x9, Xbyak_aarch64::ptr(code.x8));
    code.add(code.x8, code.x8, stack_adjustment);
    code.str(code.x8, Xbyak_aarch64::ptr(code.x0, stack_state_offset));
    code.mov(code.x0, code.x9);
    code.ret();
}

void EmitLinkedGuestTail(Xbyak_aarch64::CodeGenerator& code,
                         Xbyak_aarch64::Label& linked,
                         Xbyak_aarch64::Label& unresolved,
                         const XReg& host_target,
                         const std::uint64_t guest_target,
                         const std::uint32_t guest_pc_offset) noexcept {
    code.L(linked);
    code.mov(code.x10, guest_target);
    code.str(code.x10, Xbyak_aarch64::ptr(code.x0, guest_pc_offset));
    code.br(host_target);
    code.L(unresolved);
    code.mov(code.x0, guest_target);
    code.ret();
}

void EmitSharedFrameReturn(Xbyak_aarch64::CodeGenerator& code,
                           const std::uint64_t result,
                           const std::uint32_t frame_bytes) noexcept {
    struct SavedPair {
        std::uint8_t first;
        std::uint8_t second;
        std::uint8_t offset;
    };
    constexpr std::array saved{
        SavedPair{27, 28, 80}, SavedPair{25, 26, 64},
        SavedPair{23, 24, 48}, SavedPair{21, 22, 32},
        SavedPair{19, 20, 16},
    };
    code.mov(code.x0, result);
    for (const auto& pair : saved) {
        code.ldp(XReg(pair.first), XReg(pair.second),
                 Xbyak_aarch64::ptr(code.sp, pair.offset));
    }
    const auto caller_frame =
        Xbyak_aarch64::post_ptr(code.sp, frame_bytes);
    code.ldp(code.x29, code.x30, caller_frame);
    code.ret();
}

}
