// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/runtime_api.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <signal.h>
#include <string>
#include <ucontext.h>
#include <unistd.h>
#include <vector>

namespace {

template <typename Function>
Function Resolve(void* const library, const char* const name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

struct DiagnosticBinding {
    std::uint64_t thunk{};
    std::string symbol;
};

std::vector<DiagnosticBinding> g_bindings;
std::atomic<std::uint64_t> g_hle_calls{};

void CrashDiagnostic(
    const int signal, siginfo_t* const information,
    void* const raw_context) {
    const auto* const context =
        static_cast<const ucontext_t*>(raw_context);
    char line[2048]{};
#if defined(__aarch64__)
    const auto& registers = context->uc_mcontext;
    const int count = std::snprintf(
        line, sizeof(line),
        "PS5_PROBE_CRASH signal=%d code=%d fault=%p "
        "pc=0x%llx sp=0x%llx lr=0x%llx "
        "x0=0x%llx x1=0x%llx x2=0x%llx x3=0x%llx "
        "x19=0x%llx x20=0x%llx x21=0x%llx x22=0x%llx\n",
        signal, information != nullptr ? information->si_code : 0,
        information != nullptr ? information->si_addr : nullptr,
        static_cast<unsigned long long>(registers.pc),
        static_cast<unsigned long long>(registers.sp),
        static_cast<unsigned long long>(registers.regs[30]),
        static_cast<unsigned long long>(registers.regs[0]),
        static_cast<unsigned long long>(registers.regs[1]),
        static_cast<unsigned long long>(registers.regs[2]),
        static_cast<unsigned long long>(registers.regs[3]),
        static_cast<unsigned long long>(registers.regs[19]),
        static_cast<unsigned long long>(registers.regs[20]),
        static_cast<unsigned long long>(registers.regs[21]),
        static_cast<unsigned long long>(registers.regs[22]));
#else
    const int count = std::snprintf(
        line, sizeof(line),
        "PS5_PROBE_CRASH signal=%d code=%d fault=%p\n",
        signal, information != nullptr ? information->si_code : 0,
        information != nullptr ? information->si_addr : nullptr);
#endif
    if (count > 0) {
        (void)write(
            STDERR_FILENO, line,
            std::min<std::size_t>(
                static_cast<std::size_t>(count), sizeof(line) - 1));
    }
    _exit(128 + signal);
}

void InstallCrashDiagnostics() {
    struct sigaction action {};
    action.sa_sigaction = &CrashDiagnostic;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO;
    (void)sigaction(SIGSEGV, &action, nullptr);
    (void)sigaction(SIGBUS, &action, nullptr);
    (void)sigaction(SIGILL, &action, nullptr);
}

int BindDiagnosticImport(void*, const char* const symbol,
                         const std::uint64_t thunk,
                         std::uint64_t* const native_function) {
    if (symbol == nullptr || native_function == nullptr) {
        return -1;
    }
    if (thunk != 0) {
        g_bindings.push_back({thunk, symbol});
    }
    *native_function = 0;
    return 0;
}

std::uint64_t InvokeDiagnosticHle(
    void*, const Lsx4Ps5HleCall* const call) {
    if (call == nullptr) {
        return 0;
    }
    const auto sequence = ++g_hle_calls;
    const auto binding = std::ranges::find(
        g_bindings, call->function, &DiagnosticBinding::thunk);
    const char* const symbol =
        binding != g_bindings.end() ? binding->symbol.c_str() : "?";
    static const std::uint64_t log_limit = [] {
        const auto* const raw =
            std::getenv("PS5_PROBE_HLE_LOG_LIMIT");
        if (raw == nullptr || *raw == '\0') {
            return UINT64_C(4096);
        }
        char* end{};
        const auto parsed = std::strtoull(raw, &end, 10);
        return end != raw && *end == '\0'
            ? static_cast<std::uint64_t>(parsed)
            : UINT64_C(4096);
    }();
    if (sequence <= log_limit) {
        std::fprintf(
            stderr,
            "PS5_HLE_CALL #%llu nid=%s thunk=0x%llx "
            "args=%llx,%llx,%llx,%llx,%llx,%llx stack=0x%llx\n",
            static_cast<unsigned long long>(sequence), symbol,
            static_cast<unsigned long long>(call->function),
            static_cast<unsigned long long>(call->integer_arguments[0]),
            static_cast<unsigned long long>(call->integer_arguments[1]),
            static_cast<unsigned long long>(call->integer_arguments[2]),
            static_cast<unsigned long long>(call->integer_arguments[3]),
            static_cast<unsigned long long>(call->integer_arguments[4]),
            static_cast<unsigned long long>(call->integer_arguments[5]),
            static_cast<unsigned long long>(call->guest_stack));
        std::fflush(stderr);
    }
    return 0;
}

}

int main(const int argc, char** const argv) {
    InstallCrashDiagnostics();
    if (argc < 3 || argc > 4) {
        std::fprintf(
            stderr,
            "usage: %s liblsx4_executor_ps5_android.so "
            "<game-directory-or-eboot.bin> [--launch]\n",
            argv[0]);
        return 2;
    }
    const bool launch_requested =
        argc == 4 && std::strcmp(argv[3], "--launch") == 0;
    if (argc == 4 && !launch_requested) {
        std::fprintf(stderr, "unknown option: %s\n", argv[3]);
        return 2;
    }

    void* const library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 3;
    }
    const auto initialize = Resolve<
        decltype(&executor_lsx4_ps5_runtime_initialize)>(
        library, "executor_lsx4_ps5_runtime_initialize");
    const auto set_callbacks = Resolve<
        decltype(&executor_lsx4_ps5_runtime_set_callbacks)>(
        library, "executor_lsx4_ps5_runtime_set_callbacks");
    const auto load_game = Resolve<
        decltype(&executor_lsx4_ps5_runtime_load_game)>(
        library, "executor_lsx4_ps5_runtime_load_game");
    const auto probe_report = Resolve<
        decltype(&executor_lsx4_ps5_runtime_probe_report)>(
        library, "executor_lsx4_ps5_runtime_probe_report");
    const auto launch = Resolve<
        decltype(&executor_lsx4_ps5_runtime_launch_eboot)>(
        library, "executor_lsx4_ps5_runtime_launch_eboot");
    const auto unload = Resolve<
        decltype(&executor_lsx4_ps5_runtime_unload_eboot)>(
        library, "executor_lsx4_ps5_runtime_unload_eboot");
    const auto status = Resolve<
        decltype(&executor_lsx4_ps5_runtime_status)>(
        library, "executor_lsx4_ps5_runtime_status");
    const auto reset = Resolve<
        decltype(&executor_lsx4_ps5_runtime_reset)>(
        library, "executor_lsx4_ps5_runtime_reset");
    if (initialize == nullptr || set_callbacks == nullptr ||
        load_game == nullptr || probe_report == nullptr ||
        launch == nullptr || unload == nullptr || status == nullptr ||
        reset == nullptr) {
        std::fprintf(stderr, "PS5 runtime ABI v7 exports are incomplete\n");
        dlclose(library);
        return 4;
    }

    Lsx4Ps5RuntimeConfig config{};
    config.title_id = "GAME_PROBE";
    if (initialize(&config) != 0) {
        std::fprintf(stderr, "initialize failed: %s\n", status());
        dlclose(library);
        return 5;
    }
    Lsx4Ps5RuntimeCallbacks callbacks{};
    callbacks.bind_import = &BindDiagnosticImport;
    callbacks.invoke_hle = &InvokeDiagnosticHle;
    if (set_callbacks(&callbacks) != 0) {
        std::fprintf(stderr, "callbacks failed: %s\n", status());
        reset();
        dlclose(library);
        return 6;
    }

    Lsx4Ps5LoadedExecutable executable{};
    Lsx4Ps5GameProbeReport report{};
    const auto load_status =
        load_game(argv[2], &executable, &report);
    if (load_status != 0) {
        std::fprintf(stderr, "load failed: %s\n", status());
        reset();
        dlclose(library);
        return 7;
    }
    std::array<char, 16384> text{};
    if (probe_report(
            executable.handle, &report, text.data(), text.size()) != 0) {
        std::fprintf(stderr, "report failed: %s\n", status());
        unload(executable.handle);
        reset();
        dlclose(library);
        return 8;
    }
    std::printf("%s\n", text.data());
    std::printf(
        "eboot base=0x%llx entry=0x%llx mapped=0x%llx flags=0x%x "
        "initializers=%u\n",
        static_cast<unsigned long long>(executable.base),
        static_cast<unsigned long long>(executable.entry),
        static_cast<unsigned long long>(executable.mapped_size),
        executable.flags, executable.initializer_count);
    if (std::getenv("PS5_PROBE_DUMP_MAPPED") != nullptr) {
        constexpr char DumpPath[] =
            "/data/local/tmp/lsx4-ps5-release/eboot-mapped.bin";
        if (FILE* const dump = std::fopen(DumpPath, "wb");
            dump != nullptr) {
            const auto written = std::fwrite(
                reinterpret_cast<const void*>(executable.base), 1,
                static_cast<std::size_t>(executable.mapped_size), dump);
            std::fclose(dump);
            std::fprintf(
                stderr, "PS5_PROBE_DUMP path=%s bytes=0x%zx\n",
                DumpPath, written);
        }
    }

    int exit_code = 0;
    if (launch_requested) {
        if ((report.flags & LSX4_PS5_GAME_READY_TO_LAUNCH) == 0) {
            std::fprintf(
                stderr,
                "launch blocked by compatibility report: %s\n", status());
            exit_code = 9;
        } else {
            Lsx4Ps5GuestResult result{};
            const auto launch_status =
                launch(executable.handle, nullptr, &result);
            std::printf(
                "launch status=%d value=0x%llx runtime=%s\n",
                launch_status,
                static_cast<unsigned long long>(result.value), status());
            if (launch_status != LSX4_PS5_EXECUTE_OK) {
                exit_code = 10;
            }
            if (const char* const wait_text =
                    std::getenv("PS5_PROBE_RUN_SECONDS");
                wait_text != nullptr) {
                const auto seconds = std::strtoul(
                    wait_text, nullptr, 10);
                if (seconds != 0 && seconds <= 3600) {
                    std::fprintf(
                        stderr,
                        "PS5_PROBE_WAIT seconds=%lu\n", seconds);
                    sleep(static_cast<unsigned int>(seconds));
                }
            }
        }
    }

    unload(executable.handle);
    reset();
    dlclose(library);
    return exit_code;
}
