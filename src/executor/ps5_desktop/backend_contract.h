// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Ps5Desktop {

struct BackendCompatibility {
    bool ready{};
    bool cpu_state_compatible{};
    bool guest_page_size_compatible{};
    bool guest_address_limit_compatible{};
    std::uint64_t missing_instruction_requirements{};
};

[[nodiscard]] BackendCompatibility InspectBackendCompatibility() noexcept;

} // namespace Lsx4::Ps5Desktop
