// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace Lsx4::Translation {

[[nodiscard]] bool IsAddressRangeRepresentable(std::uint64_t address,
                                                std::size_t byte_count) noexcept;
[[nodiscard]] bool IsProcessImageMemory(std::uint64_t address,
                                        std::size_t byte_count) noexcept;
[[nodiscard]] bool QueryProcessImageBounds(std::uint64_t& first,
                                           std::uint64_t& last) noexcept;
[[nodiscard]] void* MapProcessMemoryBelow4GiB(std::size_t byte_count) noexcept;
[[nodiscard]] bool ReadProcessGuestMemory(std::uint64_t address, void* destination,
                                          std::size_t byte_count) noexcept;
[[nodiscard]] bool ReadProcessGuestMemoryWithFaultDispatch(
    std::uint64_t address, void* destination, std::size_t byte_count);
[[nodiscard]] bool WriteProcessGuestMemory(std::uint64_t address, const void* source,
                                           std::size_t byte_count) noexcept;
[[nodiscard]] bool WriteProcessGuestMemoryWithFaultDispatch(
    std::uint64_t address, const void* source, std::size_t byte_count);
[[nodiscard]] bool ReadProcessGuestScalar(std::uint64_t address,
                                          std::uint64_t& value) noexcept;
[[nodiscard]] bool WriteProcessGuestScalar(std::uint64_t address,
                                           std::uint64_t value) noexcept;

}
