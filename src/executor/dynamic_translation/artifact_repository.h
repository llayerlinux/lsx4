// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/retiring_execution_core.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace Executor::Jit {

inline constexpr std::uint32_t kJitIrCacheFormatVersion = 4;
inline constexpr std::size_t kJitIrCacheContentKeySize = 32;
inline constexpr std::uint32_t kJitNativeNoSegment = UINT32_MAX;
inline constexpr std::uint32_t kJitNativeFlagValid = 1u << 0;
inline constexpr std::uint32_t kJitNativeFlagLeafHleFused = 1u << 1;
inline constexpr std::uint32_t kJitNativeFlagScalarFloatDirectV1 = 1u << 2;
inline constexpr std::uint32_t kJitNativeFlagScalarFloatDirectV2 = 1u << 3;
inline constexpr std::uint32_t kJitNativeFlagPackedFloatDirectV3 = 1u << 4;
inline constexpr std::uint32_t kJitNativeFlagMixedHotDirectV4 = 1u << 5;
inline constexpr std::uint32_t kJitNativeFlagWideHotDirectV5 = 1u << 6;
inline constexpr std::uint32_t kJitNativeFlagPostLoadDirectV6 = 1u << 7;
inline constexpr std::uint32_t kJitNativeFlagPostLoadDirectV7 = 1u << 8;
inline constexpr std::uint32_t kJitNativeFlagPostLoadDirectV8 = 1u << 9;
inline constexpr std::uint32_t kJitNativeFlagPostLoadDirectV9 = 1u << 10;
inline constexpr std::uint32_t kJitNativeFlagDirectFaultSlotV10 = 1u << 11;
inline constexpr std::uint32_t kJitNativeFlagInlineLseXchgV11 = 1u << 12;
inline constexpr std::uint32_t kJitNativeFlagDirectLseXchgV12 = 1u << 13;
inline constexpr std::uint32_t kJitNativeFlagSharedChainAbiV13 = 1u << 14;
inline constexpr std::uint32_t kJitNativeFlagScalarLogicDirectV14 = 1u << 15;
inline constexpr std::uint32_t kJitNativeFlagVpslldImmDirectV15 = 1u << 16;
inline constexpr std::uint32_t kJitNativeFlagInsertPsDirectV16 = 1u << 17;
inline constexpr std::uint32_t kJitNativeFlagInsertPsDirectV17 = 1u << 18;
inline constexpr std::uint32_t kJitNativeFlagDirectRetV18 = 1u << 19;
inline constexpr std::uint32_t kJitNativeFlagHotStateDirectV19 = 1u << 20;
inline constexpr std::uint32_t kJitNativeFlagHotAtomicPermuteV20 = 1u << 21;

enum class JitNativeRelocationKind : std::uint8_t {
    ModuleRelative = 0,
    IrBlock = 1,
    IrInstruction = 2,
    OutboundEdgeState = 3,
    SegmentAddress = 4,
    BackendRelative = 5,
    StableExternal = 6,
    IndirectEdgeFanout = 7,
};

struct JitNativeRelocation {
    JitNativeRelocationKind kind = JitNativeRelocationKind::ModuleRelative;
    std::uint32_t code_offset = 0;
    std::uint8_t register_index = 0;
    std::uint32_t target_index = 0;
    std::int64_t addend = 0;

    friend bool operator==(const JitNativeRelocation&,
                           const JitNativeRelocation&) = default;
};

struct JitNativeSegment {
    std::vector<std::uint8_t> bytes{};
    std::vector<JitNativeRelocation> relocs{};

    friend bool operator==(const JitNativeSegment&, const JitNativeSegment&) = default;
};

struct JitIrCacheIdentity {
    std::array<std::uint8_t, kJitIrCacheContentKeySize> content_key{};
    std::uint64_t jit_abi_version = 0;

    friend bool operator==(const JitIrCacheIdentity&,
                           const JitIrCacheIdentity&) = default;
};

struct JitIrCacheRecord {
    std::uint64_t guest_rip = 0;
    std::uint64_t guest_hash = 0;
    LsxDecodedRegion block{};
    std::vector<JitNativeSegment> native_segments{};
    std::uint32_t entry_segment_index = kJitNativeNoSegment;
    std::uint32_t direct_segment_index = kJitNativeNoSegment;
    std::uint32_t native_flags = 0;
    bool loaded_ir_validated = false;
    std::uint64_t loaded_order = 0;
    bool needs_identity_migration = false;
};

struct JitIrCacheLimits {
    std::uint64_t max_file_bytes = 512ull * 1024ull * 1024ull;
    std::uint64_t max_records = 1'000'000;
    std::uint64_t max_total_instructions = 8'000'000;
    std::uint32_t max_instructions_per_block = 128;
    std::uint32_t max_diagnostic_bytes = 64 * 1024;
    std::uint8_t max_synthetic_instruction_length = 64;
    std::uint32_t max_native_segments_per_record = 16;
    std::uint32_t max_native_segment_bytes = 2 * 1024 * 1024;
    std::uint32_t max_native_relocations_per_segment = 64 * 1024;
    std::uint64_t max_total_native_bytes = 384ull * 1024ull * 1024ull;
    std::uint64_t max_total_native_relocations = 8'000'000;
};

enum class JitIrCacheStatus {
    Ok,
    NotFound,
    IoError,
    IdentityMismatch,
    UnsupportedVersion,
    Corrupt,
    LimitExceeded,
};

struct JitIrCacheLoadResult {
    JitIrCacheStatus status = JitIrCacheStatus::Corrupt;
    std::vector<JitIrCacheRecord> records{};
    std::string detail{};

    [[nodiscard]] bool ok() const noexcept {
        return status == JitIrCacheStatus::Ok;
    }
};

[[nodiscard]] bool SerializeJitIrCache(const JitIrCacheIdentity& identity,
                                            std::span<const JitIrCacheRecord> records,
                                            std::vector<std::uint8_t>& output, std::string& error,
                                            const JitIrCacheLimits& limits = {});

[[nodiscard]] JitIrCacheLoadResult DeserializeJitIrCache(
    std::span<const std::uint8_t> bytes, const JitIrCacheIdentity& expected_identity,
    const JitIrCacheLimits& limits = {});

[[nodiscard]] JitIrCacheLoadResult LoadJitIrCacheFile(
    const std::filesystem::path& path, const JitIrCacheIdentity& expected_identity,
    const JitIrCacheLimits& limits = {});

[[nodiscard]] bool SaveJitIrCacheFileAtomic(const std::filesystem::path& path,
                                                 const JitIrCacheIdentity& identity,
                                                 std::span<const JitIrCacheRecord> records,
                                                 std::string& error,
                                                 const JitIrCacheLimits& limits = {});

}
