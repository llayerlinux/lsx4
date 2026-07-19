// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/logging/backend.h"

#include <cstdio>

#ifdef __ANDROID__
#include <atomic>
#include <android/log.h>
#endif

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

void assert_fail_impl() {
#ifdef __ANDROID__
    // Reference (release) parity: the reference libshadps4.so is built with asserts effectively
    // non-fatal (verified: its Shader::Gcn::CFG::EmitLabels has zero brk instructions). Our
    // RelWithDebInfo build was SIGTRAP-crashing the GpuCommandProcessor thread mid-frame on shader
    // recompiler CFG asserts (e.g. unresolvable S_SETPC_B64 / missing branch-target block) fed by
    // the RSDK embedded/fullscreen shaders. Make a failed assert NON-fatal so execution continues
    // exactly as it does in the reference, instead of aborting the whole render thread. Rate-limited
    // so a hot bad path can't flood logcat.
    static std::atomic<unsigned> s_budget{0};
    const unsigned n = s_budget.fetch_add(1, std::memory_order_relaxed);
    if (n < 200) {
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Assert",
                            "[EXECUTOR_ASSERT_NONFATAL] n=%u continuing (reference-release parity)",
                            n);
    }
    std::fflush(stdout);
    return;
#else
    Common::Log::Stop();
    std::fflush(stdout);
    Crash();
#endif
}

[[noreturn]] void unreachable_impl() {
    Common::Log::Stop();
    std::fflush(stdout);
    Crash();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_FATAL, "LSX4Assert",
                        "[EXECUTOR_ASSERT_FAIL] %s", msg ? msg : "<null>");
#endif
    std::fprintf(stderr, "[EXECUTOR_ASSERT_FAIL] %s\n", msg ? msg : "<null>");
    std::fflush(stderr);
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
