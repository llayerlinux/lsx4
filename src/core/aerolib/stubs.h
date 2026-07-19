// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

#ifdef __ANDROID__
#include <string>
#endif

namespace Core::AeroLib {

u64 UnresolvedStub();

u64 GetStub(const char* nid);

#ifdef __ANDROID__
struct ExecutorAllocationSnapshot {
    u64 user_base{};
    u64 mapping_base{};
    u64 mapping_size{};
    u64 requested_size{};
    u64 allocated_size{};
    u64 mspace_handle{};
    u64 serial{};
    u64 offset{};
    u64 region_handle{};
    u64 region_base{};
    u64 region_size{};
    u64 region_offset{};
    bool found{};
    bool exact{};
    bool in_region{};
};

bool DescribeExecutorAllocation(u64 guest_ptr, ExecutorAllocationSnapshot* out);
void TraceExecutorAllocationsContaining(u64 guest_ptr, const char* phase);

// Guest-visible storage owned by the Executor libc mspace.  HLE objects whose addresses are
// returned to translated guest code (for example OrbisFILE and its read buffer) must use this
// allocator instead of host new/malloc so every pointer embedded in their ABI-visible layout is a
// valid guest virtual address.  Keep allocation and release on the same family.
u64 ExecutorLibcMalloc(u64 size);
u64 ExecutorLibcCalloc(u64 count, u64 size);
u64 ExecutorLibcFree(u64 ptr);

u64 EnsureAndroidX64ZeroStubSlab();
u64 GetAndroidX64ZeroStubSlabBase();
u64 GetAndroidX64ZeroStubSlabSize();
u64 GetAndroidX64HleStubForNative(const char* name, u64 native_function);
std::string GetAndroidX64HleNameForNative(u64 native_function);
bool IsAndroidX64NativeHleStub(u64 address, std::string* name);
bool TryGetAndroidX64NativeHleTarget(u64 address, u64* native_function);
bool IsAndroidX64NativeHleReturnZeroStub(u64 address);
bool TryGetAndroidX64ExecutorOverride(const char* nid, u64* virtual_address, std::string* name);
// True for process-heap and C++ allocation entry points that must all resolve to one owner.
bool IsLibcAllocFamilyNid(const char* nid);
// True for every operation that accepts or produces the FILE object returned by the Executor
// fopen bridge.  These imports must resolve as one family: mixing a host std::FILE* producer with
// guest-libc consumers is an ABI violation even when the individual calls appear to succeed.
bool IsLibcStdioFamilyNid(const char* nid);
// Same ownership rule for opaque directory objects (DIR* and dirent views).
bool IsLibcDirectoryFamilyNid(const char* nid);
// Selects the process-wide owner of the libc allocator family exactly once, after the caller has
// determined whether a usable guest libc.prx is available. Returns true when this request installed
// the owner or agrees with the owner already installed; false means an earlier call latched the
// opposite owner. Once selected, the owner cannot change during the process lifetime.
bool ExecutorConfigureLibcAllocatorOwner(bool guest_libc_available);
// True when the latched owner is guest libc.prx. Calling this before configuration is a lifecycle
// error: it atomically latches the safe Executor-HLE owner so no later relocation can create a mixed
// allocator family.
bool ExecutorUseRealLibcAlloc();
// Builds a guest tail-call interposer that forces a Mono path-setter's arg0 to /app0/Media/Managed
// then jmps to the real mono fn (used to fix AOT-module dependency resolution). If
// register_preload_hook, the stub also installs our Mono assembly preload hook (needs the address
// set via ExecutorSetMonoInstallPreloadHook). Returns real_addr unchanged on failure.
u64 ExecutorMakeMonoPathInterposer(u64 real_addr, bool register_preload_hook);
u64 ExecutorMakeMonoConfigInterposer(u64 real_addr);
// TEMP DIAG: interposer for mono_domain_assembly_open that logs the assembly name then tail-calls real.
u64 ExecutorMakeMonoDomainOpenInterposer(u64 real_addr);
void ExecutorSetMonoInstallPreloadHook(u64 addr);
// Mono corlib-AOT-self-ref fix: addresses captured at link time so the mono_set_dirs interposer can
// register a SEARCH hook (mono_install_assembly_search_hook) that resolves the mscorlib self-ref by
// wrapping the already-registered corlib image: mono_assembly_load_from_full(mono_image_loaded(
// "mscorlib"), path, &status, 0) -> sets image->assembly and returns the in-flight corlib (guarded
// against re-entry; no struct offsets).
void ExecutorSetMonoInstallSearchHook(u64 addr);
void ExecutorSetMonoImageLoaded(u64 addr);
void ExecutorSetMonoImageGetAssembly(u64 addr);
void ExecutorSetMonoLoadFromFull(u64 addr);
// mono_add_internal_call (NID -Vt1ihrzI0Q): captured so the domain-open interposer can register the
// unbound UnityEngine.UnityLogWriter::WriteStringToUnityLog icall to a no-op (fixes the unhandled NRE).
void ExecutorSetMonoAddInternalCall(u64 addr);
void ExecutorSetMonoStringNew(u64 addr);
void ExecutorSetMonoDomainGet(u64 addr);
void ExecutorSetMonoDomainAssemblyOpen(u64 addr);
u64 ExecutorGetEnvironmentNewLineIcallStubCell();
s64 ExecutorLibcStrcmp(u64 lhs, u64 rhs);
s64 ExecutorLibcStrncmp(u64 lhs, u64 rhs, u64 size);
u64 ExecutorLibcBsearch(u64 key, u64 base, u64 nmemb, u64 size, u64 compar);
#endif

} // namespace Core::AeroLib
