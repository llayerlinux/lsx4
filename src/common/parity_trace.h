// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace Common::ParityTrace {

enum class Domain {
    Loader,
    Vm,
    Hle,
    Thread,
    Fs,
    Audio,
    Gnm,
    Gpu,
    VideoOut,
    Frame,
};

enum class Contract {
    MustEqual,
    Capability,
    Health,
};

enum class Stage {
    None,
    Graphics,
    Vs,
    Ps,
    Gs,
    Es,
    Hs,
    Ls,
    Cs,
};

struct Key {
    std::int64_t submit{-1};
    std::int64_t draw{-1};
    Stage stage{Stage::None};
    std::int64_t slot{-1};
    std::uint64_t ordinal{};
};

using Json = nlohmann::json;

// Tracing is disabled unless SHADPS4_PARITY_TRACE names an output file and all
// three SHADPS4_PARITY_*_SHA256 identity variables contain valid hashes. This
// intentionally fails closed: two unknown runs must never appear comparable.
[[nodiscard]] bool IsEnabled() noexcept;

// Canonical lowercase hexadecimal for guest-visible addresses and bit patterns.
[[nodiscard]] std::string Hex(std::uint64_t value);

// Emits one bounded, crash-durable JSONL record. Semantic records containing a
// float or a host-only field are rejected by the shared writer rather than left
// for each call site to normalize independently.
void Emit(Domain domain, std::string_view event, Contract contract, const Key& key,
          Json data) noexcept;

void Flush() noexcept;
[[nodiscard]] std::uint64_t EventCount() noexcept;
[[nodiscard]] bool WasTruncated() noexcept;

} // namespace Common::ParityTrace
