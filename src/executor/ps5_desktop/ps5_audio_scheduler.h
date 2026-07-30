// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace Lsx4::Ps5Desktop {

struct Ps5AudioPortConfig {
    std::uint32_t buffer_frames{};
    std::uint32_t sample_rate{};
    std::int32_t format{};
    std::uint32_t guest_channels{};
    std::uint32_t bytes_per_sample{};
    std::int32_t port_type{};
};

struct Ps5AudioPort;
using Ps5AudioPortHandle = std::shared_ptr<Ps5AudioPort>;
using Ps5AudioGuestReadCallback =
    bool (*)(std::uint64_t address, void* destination,
             std::size_t byte_count);

/// Opens (or reuses) the host stream for a compatible PS5 audio cadence.
/// Compatible guest ports share one backend and are mixed by one scheduler.
[[nodiscard]] Ps5AudioPortHandle OpenPs5AudioPort(
    const Ps5AudioPortConfig& config);

/// Stops accepting new buffers and waits for the scheduler to retire both
/// handoff slots before releasing the shared backend.
void ClosePs5AudioPort(Ps5AudioPortHandle& port) noexcept;

/// Copies one guest buffer directly into a reusable handoff slot. Conversion,
/// mixing, pacing and the host write run on the shared scheduler thread.
[[nodiscard]] bool QueuePs5AudioBuffer(
    const Ps5AudioPortHandle& port, std::uint64_t source_address,
    float gain, std::uint64_t sequence,
    Ps5AudioGuestReadCallback read_guest) noexcept;

} // namespace Lsx4::Ps5Desktop
