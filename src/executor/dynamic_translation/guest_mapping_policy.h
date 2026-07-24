// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <cstddef>

namespace Lsx4::Translation {

[[nodiscard]] bool IsExecutableGuestAddress(std::uint64_t address) noexcept;
[[nodiscard]] bool IsGuestAddressRange(std::uint64_t address,
                                       std::size_t byte_count) noexcept;

}
