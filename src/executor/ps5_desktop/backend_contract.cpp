// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/backend_contract.h"

#include "executor/dynamic_translation/code_witness.h"
#include "ps5_desktop/guest_contract.h"

namespace Lsx4::Ps5Desktop {
namespace {

constexpr std::uint32_t CurrentJitGuestPageSize =
    static_cast<std::uint32_t>(
        Lsx4::Translation::MutationLedger::kGuestPageSize);
constexpr std::uint64_t CurrentJitGuestAddressLimit = 0x10000000000ull;
constexpr std::uint64_t SupportedCpuState =
    Funnel::Ps5Desktop::CpuStateGprAndFlags |
    Funnel::Ps5Desktop::CpuStateFsGsBase |
    Funnel::Ps5Desktop::CpuStateX87 |
    Funnel::Ps5Desktop::CpuStateYmm;

constexpr std::uint64_t SupportedInstructionRequirements =
    Funnel::Ps5Desktop::InstructionSse4aExtrqInsertq |
    Funnel::Ps5Desktop::InstructionMonitorxMwaitx;

}

BackendCompatibility InspectBackendCompatibility() noexcept {
    const auto guest = Funnel::Ps5Desktop::QueryGuestContract();
    const bool cpu_state_compatible =
        (guest.required_cpu_state & ~SupportedCpuState) == 0;
    const bool guest_page_size_compatible =
        guest.guest_page_size == CurrentJitGuestPageSize;
    const bool guest_address_limit_compatible =
        guest.guest_address_limit <= CurrentJitGuestAddressLimit;
    const auto missing_instructions =
        guest.required_instructions & ~SupportedInstructionRequirements;

    return {
        .ready = guest.abi_version == Funnel::Ps5Desktop::GuestContractAbiVersion &&
                 cpu_state_compatible && guest_page_size_compatible &&
                 guest_address_limit_compatible && missing_instructions == 0,
        .cpu_state_compatible = cpu_state_compatible,
        .guest_page_size_compatible = guest_page_size_compatible,
        .guest_address_limit_compatible = guest_address_limit_compatible,
        .missing_instruction_requirements = missing_instructions,
    };
}

}
