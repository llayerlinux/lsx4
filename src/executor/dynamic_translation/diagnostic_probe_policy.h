// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace Lsx4::Translation {

[[nodiscard]] bool TraceAllocatorOffset(std::uint64_t module_offset) noexcept;
[[nodiscard]] bool TraceTranslationOffset(std::uint64_t module_offset) noexcept;
[[nodiscard]] bool ClaimDiagnosticSample(
    std::atomic<std::uint64_t>& sequence) noexcept;
inline constexpr std::uint64_t kNoDiagnosticTicket = UINT64_MAX;
[[nodiscard]] std::uint64_t ClaimDiagnosticTicket(
    std::atomic<std::uint64_t>& sequence) noexcept;

}
