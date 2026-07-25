// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" int executor_jit_lookup_hle_thunk(
    std::uint64_t thunk, std::uint64_t* native_function);
extern "C" int executor_jit_lookup_leaf_hle_thunk(
    std::uint64_t thunk, std::uint64_t* native_function);
extern "C" int executor_jit_hle_thunk_returns_zero(std::uint64_t thunk);
extern "C" int executor_jit_classify_hle_thunk(std::uint64_t thunk);
extern "C" int executor_jit_resolve_hle_thunk(
    std::uint64_t thunk, std::uint64_t* native_function);
extern "C" int executor_jit_hle_fp_result_kind(std::uint64_t native_function);
extern "C" int executor_jit_hle_fp_bridge_selftest();

extern "C" void executor_lsx4_android_register_guest_readable_range(
    const void* base, std::size_t size, const char* label);
using GuestWindowExtent = std::size_t;
extern "C" void executor_lsx4_android_note_guest_stack_window(
    const void* base, GuestWindowExtent size, const char* label);

extern "C" bool ExecutorJitReadGuestBytes(
    std::uint64_t address, void* destination, std::size_t size);
extern "C" bool ExecutorJitReadGuestBytesStable(
    std::uint64_t address, void* destination, std::size_t size);
extern "C" bool ExecutorJitWriteGuestBytes(
    std::uint64_t address, const void* source, std::size_t size);
extern "C" bool ExecutorJitIsReadableGuestRange(
    std::uint64_t address, std::size_t size);
extern "C" bool ExecutorJitIsExecutableGuestAddress(std::uint64_t address);

extern "C" std::uint64_t executor_jit_ensure_hle_thunk_slab();
extern "C" std::uint64_t executor_jit_hle_thunk_slab_base();
extern "C" std::uint64_t executor_jit_hle_thunk_slab_size();
extern "C" std::uint64_t executor_jit_get_hle_stub_for_native(
    const char* label, std::uint64_t native_function);
extern "C" std::uint64_t executor_jit_stable_libc_strcmp();

#ifdef __ANDROID__
extern "C" int executor_lsx4_android_dispatch_deferred_guest_signal(
    std::int32_t native_sig, std::int32_t si_code, std::int32_t si_errno,
    std::int32_t source_pid, std::uint32_t source_uid,
    std::uint64_t fault_addr, std::uint64_t guest_rip,
    std::int32_t is_write);
extern "C" int executor_jit_defer_synchronous_guest_fault(
    std::int32_t native_sig, std::int32_t si_code, std::int32_t si_errno,
    std::int32_t source_pid, std::uint32_t source_uid,
    std::uint64_t fault_addr, std::int32_t is_write);
#endif
