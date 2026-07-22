// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/condition_materializer.h"

#include <cstdint>
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

}
