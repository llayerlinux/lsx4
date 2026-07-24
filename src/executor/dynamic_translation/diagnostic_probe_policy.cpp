// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/diagnostic_probe_policy.h"

#include <bit>

namespace Lsx4::Translation {

bool TraceAllocatorOffset(const std::uint64_t module_offset) noexcept {
    (void)module_offset;
    return false;
}

bool TraceTranslationOffset(const std::uint64_t module_offset) noexcept {
    (void)module_offset;
    return false;
}

bool ClaimDiagnosticSample(std::atomic<std::uint64_t>& sequence) noexcept {
    return ClaimDiagnosticTicket(sequence) != kNoDiagnosticTicket;
}

std::uint64_t ClaimDiagnosticTicket(
    std::atomic<std::uint64_t>& sequence) noexcept {
    const std::uint64_t ticket =
        sequence.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t ordinal = ticket + 1;
    return ordinal <= 4 || std::has_single_bit(ordinal)
               ? ticket
               : kNoDiagnosticTicket;
}

}
