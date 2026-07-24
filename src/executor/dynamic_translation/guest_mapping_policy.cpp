// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/guest_mapping_policy.h"

#include "core/memory.h"

#include <limits>

namespace Lsx4::Translation {

bool IsGuestAddressRange(const std::uint64_t address,
                         const std::size_t byte_count) noexcept {
    constexpr std::uint64_t kFirstUserspacePage = 0x10000ull;
    constexpr std::uint64_t kCanonicalUpperHalf = 0xffff000000000000ull;
    if (address < kFirstUserspacePage || address >= kCanonicalUpperHalf) {
        return false;
    }
    const std::uint64_t span = byte_count == 0 ? 0 : byte_count - 1u;
    return span <= std::numeric_limits<std::uint64_t>::max() - address;
}

bool IsExecutableGuestAddress(const std::uint64_t address) noexcept {
    Core::MemoryManager* const memory = Core::Memory::Instance();
    if (address == 0 || memory == nullptr || !memory->IsValidMapping(address, 1)) {
        return false;
    }
    u32 protection = 0;
    const int query_result =
        memory->QueryProtection(address, nullptr, nullptr, &protection);
    const auto execute_bit = static_cast<u32>(Core::MemoryProt::CpuExec);
    return query_result == 0 && (protection & execute_bit) != 0;
}

}
