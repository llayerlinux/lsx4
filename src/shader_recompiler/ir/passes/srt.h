// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include <vector>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
using PFN_SrtMemoryRead = bool (*)(VAddr /*address*/, void* /*dest*/, u64 /*size*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

// Portable (non-x86) SRT walker recipe. The x86 path JITs a machine-code walker; on other hosts
// (Android arm64) we instead record the pointer-tree traversal as data and replay it at runtime in
// Info::RefreshFlatBuf. Each node is a pointer level (root pointer lives in user_data at ptr_off_dw;
// nested pointers live at ptr_off_dw inside their parent's pointed-to table). Each copy moves one dword
// from a node's table into the flattened buffer. Order MUST mirror x86 VisitPointer (a node's copies
// before its children) so dst_off_dw equals the sharp_idx assigned by resource tracking.
struct PortableSrtNode {
    s32 parent;       // -1 = root (base pointer read from user_data)
    u32 ptr_off_dw;   // dword offset of this node's 64-bit pointer within its parent base
};
struct PortableSrtCopy {
    u32 node;         // index into portable_nodes whose table we read from
    u32 src_off_dw;   // source dword within that node's table
    u32 dst_off_dw;   // destination dword in flattened_ud_buf
};

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs
    std::vector<PortableSrtNode> portable_nodes;   // non-x86 SRT walker recipe (empty on x86 path)
    std::vector<PortableSrtCopy> portable_copies;

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

struct SrtMemoryRange {
    VAddr address{};
    u64 size{};
};
// A shader normally touches only a handful of SRT tables. Inline storage keeps the live
// GetProgram path allocation-free while preserving a vector fallback for unusually deep recipes.
using SrtMemoryRanges = boost::container::small_vector<SrtMemoryRange, 16>;

// Execute the portable SRT walker recipe: deref the guest pointer tree (relocating each level through
// the replay overlay when a .gnmcap replay is active) and fill flattened_ud_buf[16..]. No-op if the
// recipe is empty (x86 path uses walker_func instead).
void RunPortableSrtWalker(const u32* user_data, u32* flat, const PersistentSrtInfo& srt,
                           u32 stage, u64 pgm_hash,
                           SrtMemoryRanges* memory_ranges = nullptr,
                           PFN_SrtMemoryRead memory_reader = nullptr);

} // namespace Shader
