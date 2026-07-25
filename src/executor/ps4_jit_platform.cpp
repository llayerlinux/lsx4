// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/runtime_bridge_api.h"

#include "core/aerolib/stubs.h"
#include "core/memory.h"

#include <cstdint>

extern "C" bool ExecutorJitIsExecutableGuestAddress(
    const std::uint64_t address) {
    Core::MemoryManager* const memory = Core::Memory::Instance();
    if (address == 0 || memory == nullptr || !memory->IsValidMapping(address, 1)) {
        return false;
    }
    u32 protection = 0;
    const int query_result =
        memory->QueryProtection(address, nullptr, nullptr, &protection);
    const auto execute_bit = static_cast<u32>(Core::MemoryProt::CpuExec);
    return query_result == 0 && (protection & execute_bit) != 0;
}

extern "C" std::uint64_t executor_jit_ensure_hle_thunk_slab() {
#ifdef __ANDROID__
    return Core::AeroLib::EnsureAndroidX64ZeroStubSlab();
#else
    return 0;
#endif
}

extern "C" std::uint64_t executor_jit_hle_thunk_slab_base() {
#ifdef __ANDROID__
    return Core::AeroLib::GetAndroidX64ZeroStubSlabBase();
#else
    return 0;
#endif
}

extern "C" std::uint64_t executor_jit_hle_thunk_slab_size() {
#ifdef __ANDROID__
    return Core::AeroLib::GetAndroidX64ZeroStubSlabSize();
#else
    return 0;
#endif
}

extern "C" std::uint64_t executor_jit_get_hle_stub_for_native(
    const char* label, const std::uint64_t native_function) {
#ifdef __ANDROID__
    return Core::AeroLib::GetAndroidX64HleStubForNative(label, native_function);
#else
    (void)label;
    (void)native_function;
    return 0;
#endif
}

extern "C" std::uint64_t executor_jit_stable_libc_strcmp() {
    return reinterpret_cast<std::uint64_t>(&Core::AeroLib::ExecutorLibcStrcmp);
}
