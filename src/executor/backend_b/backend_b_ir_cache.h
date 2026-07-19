// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "lsx_translation_engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace Executor::BackendB {

// Increment this whenever the on-disk layout changes. JIT semantic changes belong in
// BackendBIrCacheIdentity::jit_abi_version so an old file is rejected without attempting to
// deserialize records compiled for a different lowering contract.
inline constexpr std::uint32_t kBackendBIrCacheFormatVersion = 4;
inline constexpr std::size_t kBackendBIrCacheContentKeySize = 32;
inline constexpr std::uint32_t kBackendBNativeNoSegment = UINT32_MAX;
inline constexpr std::uint32_t kBackendBNativeFlagValid = 1u << 0;
// The native payload contains the fail-closed CALL->PLT/GOT->leaf-HLE fusion.  Old records lack
// this bit and are selectively re-emitted only when their live terminator resolves to a leaf HLE;
// unrelated cached blocks remain warm across the upgrade.
inline constexpr std::uint32_t kBackendBNativeFlagLeafHleFused = 1u << 1;
// The native payload was emitted after scalar ADDSS/SUBSS/MULSS gained a direct AArch64 lowering.
// Older relocatable records remain valid for every other block and are selectively re-emitted
// only when their decoded IR contains this family.
inline constexpr std::uint32_t kBackendBNativeFlagScalarFloatDirectV1 = 1u << 2;
// The native payload was emitted after the next measured scalar/SIMD frontier gained direct
// AArch64 lowerings: COMISS/UCOMISS flag compares, CVTSI2SS, and INSERTPS. Keep this separate
// from V1 so blocks already upgraded for ADDSS/SUBSS/MULSS are selectively rebuilt once.
inline constexpr std::uint32_t kBackendBNativeFlagScalarFloatDirectV2 = 1u << 3;
// The payload contains direct packed FP arithmetic/horizontal-add and exact truncating
// scalar-float-to-integer conversion. V3 is independent so V1/V2 cache upgrades remain warm.
inline constexpr std::uint32_t kBackendBNativeFlagPackedFloatDirectV3 = 1u << 4;
// Direct scalar div/min/max (plus double-lane arithmetic), SHUFPS, and the integer sign-extension
// family. These are the next cache-hot semantic leaves observed after V3 reached the level path.
inline constexpr std::uint32_t kBackendBNativeFlagMixedHotDirectV4 = 1u << 5;
// Wide post-load CPU front: scalar bit counts plus the packed compare/minmax/dot/blend/move
// families used by game render loops.  Keep the upgrade selective so unrelated cached blocks do
// not pay a rebuild when this common engine workload moves from semantic calls to AArch64.
inline constexpr std::uint32_t kBackendBNativeFlagWideHotDirectV5 = 1u << 6;
// The first engine workload after large-level load: packed dword multiply, packed integer
// widening, 128-bit AVX lane extraction, and scalar precision conversions.  These families are
// shared by render/animation middleware, so retain the selective cache-upgrade contract instead
// of invalidating every title's native cache.
inline constexpr std::uint32_t kBackendBNativeFlagPostLoadDirectV6 = 1u << 7;
// Byte-table shuffles, packed dword-to-float conversion, and immediate blend families form the
// next post-load render/animation front.  Version the upgrade independently so existing native
// shards are rebuilt only when they actually contain one of these instructions.
inline constexpr std::uint32_t kBackendBNativeFlagPostLoadDirectV7 = 1u << 8;
// Engine timing/spin and memory-ordering instructions no longer cross the generic scalar
// semantic thunk.  Keep this selective: only cached blocks containing the service family need
// to be rebuilt, while the rest of a title's native cache remains reusable.
inline constexpr std::uint32_t kBackendBNativeFlagPostLoadDirectV8 = 1u << 9;
// CMPXCHG now enters its specialized CAS helper directly from a native block, with aligned
// locked memory using one host atomic compare-exchange instead of the global semantic mutex.
inline constexpr std::uint32_t kBackendBNativeFlagPostLoadDirectV9 = 1u << 10;
// Faultable native blocks now resolve the thread-local fault IR slot once per Execute and load
// that address directly from LsxMachineImage. Older cached bodies still call the C++ TLS resolver in
// every block prologue, so selectively rebuild only blocks which can touch guest memory.
inline constexpr std::uint32_t kBackendBNativeFlagDirectFaultSlotV10 = 1u << 11;
// Byte XCHG on LSE-capable AArch64 hosts is emitted as one guarded SWPALB instead of crossing the
// C++/compiler-rt outline-atomic path. Only blocks containing this exact form need rebuilding.
inline constexpr std::uint32_t kBackendBNativeFlagInlineLseXchgV11 = 1u << 12;
// V11 emitted SWPALB only after the generic scalar classifier had accepted the instruction.
// Explicit LOCK XCHG was still diverted to the scalar semantic leaf before reaching that emitter.
// Rebuild the same small block family once more after making memory/register XCHG a first-class
// native atomic irrespective of its redundant LOCK prefix.
inline constexpr std::uint32_t kBackendBNativeFlagDirectLseXchgV12 = 1u << 13;
// The persisted direct segment is the generic shared-frame chain entry rather than an ordinary
// Arm64BlockEntry entry. The execution segment remains the normal ABI wrapper.
inline constexpr std::uint32_t kBackendBNativeFlagSharedChainAbiV13 = 1u << 14;
// Scalar ANDN now stays inside the generic AArch64 block, and TEST can read AH/CH/DH/BH directly
// before using the established inline logic-flags path. Keep a separate bit so restored bodies
// which still call the scalar semantic leaf are selectively rebuilt without invalidating
// unrelated native shards.
inline constexpr std::uint32_t kBackendBNativeFlagScalarLogicDirectV14 = 1u << 15;
// VEX VPSLLD with an immediate count now lowers to lane-wise NEON SHL (or exact all-zero for
// counts >= 32). Version only this strict operand family so existing unrelated vector blocks stay
// warm while cached semantic-thunk bodies are rebuilt once.
inline constexpr std::uint32_t kBackendBNativeFlagVpslldImmDirectV15 = 1u << 16;

// A cached native segment never contains a process-local pointer after serialization. Each
// relocation points at the beginning of a fixed-width AArch64 MOVZ/MOVK materialization sequence
// (four 32-bit instructions) which the loader patches before publishing executable code.
enum class BackendBNativeRelocationKind : std::uint8_t {
    ModuleRelative = 0,
    IrBlock = 1,
    IrInstruction = 2,
    ChainPatchCell = 3,
    SegmentAddress = 4,
    // Legacy v2 relocation kinds retained so old shards can still be read as IR-only. Native
    // restore rejects both because arbitrary module offsets are not stable across relinks.
    BackendRelative = 5,
    // Versioned logical helper id resolved by the current runtime. New native captures use this
    // for every callable module target and reject any helper missing from the stable-id table.
    StableExternal = 6,
    // The emitted block referenced its per-source indirect branch/return PIC. The raw pointer is
    // never serialized; restore supplies a freshly constructed process-local site and patches it
    // into the fixed MOVZ/MOVK materialization.
    PolymorphicChainSite = 7,
};

struct BackendBNativeRelocation {
    BackendBNativeRelocationKind kind = BackendBNativeRelocationKind::ModuleRelative;
    std::uint32_t code_offset = 0;
    std::uint8_t register_index = 0;
    std::uint32_t target_index = 0;
    std::int64_t addend = 0;

    friend bool operator==(const BackendBNativeRelocation&,
                           const BackendBNativeRelocation&) = default;
};

struct BackendBNativeSegment {
    std::vector<std::uint8_t> bytes{};
    std::vector<BackendBNativeRelocation> relocs{};

    friend bool operator==(const BackendBNativeSegment&, const BackendBNativeSegment&) = default;
};

struct BackendBIrCacheIdentity {
    // A stable digest chosen by the caller. It should cover the title/module contents and any
    // launch configuration that can change decoded guest code. It must not contain process-local
    // host addresses.
    std::array<std::uint8_t, kBackendBIrCacheContentKeySize> content_key{};
    std::uint64_t jit_abi_version = 0;

    friend bool operator==(const BackendBIrCacheIdentity&,
                           const BackendBIrCacheIdentity&) = default;
};

struct BackendBIrCacheRecord {
    std::uint64_t guest_rip = 0;
    std::uint64_t guest_hash = 0;
    LsxDecodedRegion block{};
    std::vector<BackendBNativeSegment> native_segments{};
    std::uint32_t entry_segment_index = kBackendBNativeNoSegment;
    std::uint32_t direct_segment_index = kBackendBNativeNoSegment;
    std::uint32_t native_flags = 0;
    // Process-local preparation bit. It is deliberately not serialized: the loader validates
    // and canonicalizes IR records in parallel after merging shards, then the first-execution
    // path only has to compare live guest bytes instead of re-running the full classifier for
    // every one of a game's hundreds of thousands of blocks.
    bool loaded_ir_validated = false;
    // Process-local chronological order assigned while shards are merged. The writer preserves
    // first-touch order inside its monotonically named shards; retaining that order lets Android
    // prewarm the exact startup working set instead of materializing an arbitrary hash-table
    // subset. Deliberately not serialized.
    std::uint64_t loaded_order = 0;
    // Runtime-only provenance. This is deliberately not serialized: records imported from an
    // older source-build directory are rewritten under the stable semantic ABI after a successful
    // native restore, so subsequent development builds do not scan the same legacy shards again.
    bool needs_identity_migration = false;
};

// These limits are part of the trust boundary. Files are rejected before any attacker-controlled
// count is used for an unbounded allocation.
struct BackendBIrCacheLimits {
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

enum class BackendBIrCacheStatus {
    Ok,
    NotFound,
    IoError,
    IdentityMismatch,
    UnsupportedVersion,
    Corrupt,
    LimitExceeded,
};

struct BackendBIrCacheLoadResult {
    BackendBIrCacheStatus status = BackendBIrCacheStatus::Corrupt;
    std::vector<BackendBIrCacheRecord> records{};
    std::string detail{};

    [[nodiscard]] bool ok() const noexcept {
        return status == BackendBIrCacheStatus::Ok;
    }
};

// All integers in the file are encoded little-endian and all decoded operands are serialized field
// by field. No compiler padding, std::vector internals, or process-local pointers enter the file.
[[nodiscard]] bool SerializeBackendBIrCache(const BackendBIrCacheIdentity& identity,
                                            std::span<const BackendBIrCacheRecord> records,
                                            std::vector<std::uint8_t>& output, std::string& error,
                                            const BackendBIrCacheLimits& limits = {});

[[nodiscard]] BackendBIrCacheLoadResult DeserializeBackendBIrCache(
    std::span<const std::uint8_t> bytes, const BackendBIrCacheIdentity& expected_identity,
    const BackendBIrCacheLimits& limits = {});

[[nodiscard]] BackendBIrCacheLoadResult LoadBackendBIrCacheFile(
    const std::filesystem::path& path, const BackendBIrCacheIdentity& expected_identity,
    const BackendBIrCacheLimits& limits = {});

// Rewrites via a uniquely named file in the destination directory and then renames it over the
// destination. The rename is atomic on Android/POSIX; MoveFileEx(REPLACE_EXISTING) provides the
// corresponding operation on Windows. A failed write leaves the previous cache intact.
[[nodiscard]] bool SaveBackendBIrCacheFileAtomic(const std::filesystem::path& path,
                                                 const BackendBIrCacheIdentity& identity,
                                                 std::span<const BackendBIrCacheRecord> records,
                                                 std::string& error,
                                                 const BackendBIrCacheLimits& limits = {});

} // namespace Executor::BackendB
