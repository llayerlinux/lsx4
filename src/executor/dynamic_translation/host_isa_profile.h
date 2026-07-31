// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Executor::Jit {

enum class HostIsaFeature : std::uint64_t {
    Lse = UINT64_C(1) << 0,
    FlagM = UINT64_C(1) << 1,
    FlagM2 = UINT64_C(1) << 2,
    Lrcpc = UINT64_C(1) << 3,
    Ilrcpc = UINT64_C(1) << 4,
    Lrcpc3 = UINT64_C(1) << 5,
    Sve = UINT64_C(1) << 6,
    Sve2 = UINT64_C(1) << 7,
};

struct HostIsaProfile {
    std::uint64_t hwcap{};
    std::uint64_t hwcap2{};
    std::uint64_t features{};
    std::uint32_t sve_vector_bytes{};

    [[nodiscard]] bool Has(HostIsaFeature feature) const noexcept {
        return (features & static_cast<std::uint64_t>(feature)) != 0;
    }
};

[[nodiscard]] const HostIsaProfile& GetHostIsaProfile() noexcept;
[[nodiscard]] std::uint32_t CurrentThreadSveVectorBytes() noexcept;

}
