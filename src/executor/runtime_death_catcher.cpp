// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef __ANDROID__

#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#include <unwind.h>

#include <atomic>
#include <cxxabi.h>
#include <exception>
#include <mutex>

namespace {

constexpr const char* kTag = "LSX4Native";
constexpr int kSignals[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE, SIGSYS};
constexpr std::size_t kAltStackSize = 256 * 1024;

thread_local void* g_runtime_death_alt_stack = nullptr;
thread_local std::size_t g_runtime_death_alt_stack_size = 0;
struct sigaction g_previous_actions[64]{};
std::atomic<int> g_death_logging{0};
std::atomic<int> g_maps_logged{0};
std::atomic<int> g_seccomp_exit_trap_installed{0};
volatile sig_atomic_t g_signal_death_logging = 0;
int g_raw_death_log_fd = -1;

void RuntimeDeathSignalHandler(int signo, siginfo_t* info, void* raw_context);

void LogRuntimeDeath(const char* format, ...) {
    char buffer[1536];
    va_list args;
    va_start(args, format);
    const int count = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (count <= 0) {
        return;
    }
    buffer[sizeof(buffer) - 1] = '\0';
    __android_log_write(ANDROID_LOG_ERROR, kTag, buffer);
    write(STDERR_FILENO, buffer, strnlen(buffer, sizeof(buffer)));
    write(STDERR_FILENO, "\n", 1);
}

uintptr_t ContextPc(void* raw_context) {
    if (raw_context == nullptr) {
        return 0;
    }
    auto* context = static_cast<ucontext_t*>(raw_context);
#if defined(__aarch64__)
    return static_cast<uintptr_t>(context->uc_mcontext.pc);
#elif defined(__arm__)
    return static_cast<uintptr_t>(context->uc_mcontext.arm_pc);
#elif defined(__x86_64__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_RIP]);
#elif defined(__i386__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_EIP]);
#else
    return 0;
#endif
}

uintptr_t ContextSp(void* raw_context) {
    if (raw_context == nullptr) {
        return 0;
    }
    auto* context = static_cast<ucontext_t*>(raw_context);
#if defined(__aarch64__)
    return static_cast<uintptr_t>(context->uc_mcontext.sp);
#elif defined(__arm__)
    return static_cast<uintptr_t>(context->uc_mcontext.arm_sp);
#elif defined(__x86_64__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_RSP]);
#elif defined(__i386__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_ESP]);
#else
    return 0;
#endif
}

uintptr_t ContextFp(void* raw_context) {
    if (raw_context == nullptr) {
        return 0;
    }
    auto* context = static_cast<ucontext_t*>(raw_context);
#if defined(__aarch64__)
    return static_cast<uintptr_t>(context->uc_mcontext.regs[29]);
#elif defined(__arm__)
    return static_cast<uintptr_t>(context->uc_mcontext.arm_fp);
#elif defined(__x86_64__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_RBP]);
#elif defined(__i386__)
    return static_cast<uintptr_t>(context->uc_mcontext.gregs[REG_EBP]);
#else
    return 0;
#endif
}

uintptr_t ContextLr(void* raw_context) {
    if (raw_context == nullptr) {
        return 0;
    }
    auto* context = static_cast<ucontext_t*>(raw_context);
#if defined(__aarch64__)
    return static_cast<uintptr_t>(context->uc_mcontext.regs[30]);
#elif defined(__arm__)
    return static_cast<uintptr_t>(context->uc_mcontext.arm_lr);
#else
    return 0;
#endif
}

uintptr_t ContextReg(void* raw_context, int reg) {
    if (raw_context == nullptr || reg < 0) {
        return 0;
    }
    auto* context = static_cast<ucontext_t*>(raw_context);
#if defined(__aarch64__)
    return reg < 31 ? static_cast<uintptr_t>(context->uc_mcontext.regs[reg]) : 0;
#else
    (void)context;
    return 0;
#endif
}

void EnsureRawDeathLogFile() {
    if (g_raw_death_log_fd >= 0) {
        return;
    }
    const char* path = getenv("EXECUTOR_RUNTIME_DEATH_LOG");
    if (path == nullptr || path[0] == '\0') {
        path = "/data/data/app.lsx4.android/files/executor-runtime-death.log";
    }
    const int fd = open(path, O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0600);
    if (fd >= 0) {
        g_raw_death_log_fd = fd;
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] raw death log path=%s fd=%d", path, fd);
    } else {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] raw death log open failed path=%s errno=%d",
                        path, errno);
    }
}

void RawWriteBytes(const char* data, size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    write(STDERR_FILENO, data, size);
    if (g_raw_death_log_fd >= 0) {
        write(g_raw_death_log_fd, data, size);
    }
}

char* RawAppendLiteral(char* out, char* end, const char* text) {
    if (text == nullptr) {
        text = "?";
    }
    while (out < end && *text != '\0') {
        *out++ = *text++;
    }
    return out;
}

char* RawAppendDec(char* out, char* end, uint64_t value) {
    char temp[32];
    int count = 0;
    do {
        temp[count++] = static_cast<char>('0' + (value % 10));
        value /= 10;
    } while (value != 0 && count < static_cast<int>(sizeof(temp)));
    while (out < end && count > 0) {
        *out++ = temp[--count];
    }
    return out;
}

char* RawAppendHex(char* out, char* end, uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    out = RawAppendLiteral(out, end, "0x");
    bool started = false;
    for (int shift = 60; shift >= 0; shift -= 4) {
        const unsigned nibble = static_cast<unsigned>((value >> shift) & 0xfu);
        if (nibble != 0 || started || shift == 0) {
            if (out < end) {
                *out++ = digits[nibble];
            }
            started = true;
        }
    }
    return out;
}

void RawLogLine(const char* label, uint64_t a = UINT64_MAX, const char* label_b = nullptr,
                uint64_t b = UINT64_MAX, const char* label_c = nullptr,
                uint64_t c = UINT64_MAX, const char* label_d = nullptr,
                uint64_t d = UINT64_MAX) {
    char buffer[1024];
    char* out = buffer;
    char* end = buffer + sizeof(buffer) - 2;
    out = RawAppendLiteral(out, end, "[EXECUTOR_RUNTIME_DEATH_RAW] ");
    out = RawAppendLiteral(out, end, label);
    if (a != UINT64_MAX) {
        out = RawAppendLiteral(out, end, " ");
        out = RawAppendHex(out, end, a);
    }
    if (label_b != nullptr) {
        out = RawAppendLiteral(out, end, " ");
        out = RawAppendLiteral(out, end, label_b);
        out = RawAppendLiteral(out, end, "=");
        out = RawAppendHex(out, end, b);
    }
    if (label_c != nullptr) {
        out = RawAppendLiteral(out, end, " ");
        out = RawAppendLiteral(out, end, label_c);
        out = RawAppendLiteral(out, end, "=");
        out = RawAppendHex(out, end, c);
    }
    if (label_d != nullptr) {
        out = RawAppendLiteral(out, end, " ");
        out = RawAppendLiteral(out, end, label_d);
        out = RawAppendLiteral(out, end, "=");
        out = RawAppendHex(out, end, d);
    }
    *out++ = '\n';
    RawWriteBytes(buffer, static_cast<size_t>(out - buffer));
}

void RawLogString(const char* label, const char* value) {
    char buffer[1024];
    char* out = buffer;
    char* end = buffer + sizeof(buffer) - 2;
    out = RawAppendLiteral(out, end, "[EXECUTOR_RUNTIME_DEATH_RAW] ");
    out = RawAppendLiteral(out, end, label);
    out = RawAppendLiteral(out, end, "=");
    out = RawAppendLiteral(out, end, value);
    *out++ = '\n';
    RawWriteBytes(buffer, static_cast<size_t>(out - buffer));
}

void RawLogSignalHeader(int signo, siginfo_t* info, void* raw_context) {
    char buffer[1536];
    char* out = buffer;
    char* end = buffer + sizeof(buffer) - 2;
    out = RawAppendLiteral(out, end, "[EXECUTOR_RUNTIME_DEATH_RAW] signal=");
    out = RawAppendDec(out, end, static_cast<uint64_t>(signo));
    out = RawAppendLiteral(out, end, " tid=");
    out = RawAppendDec(out, end, static_cast<uint64_t>(syscall(__NR_gettid)));
    out = RawAppendLiteral(out, end, " si_code=");
    out = RawAppendDec(out, end, info != nullptr ? static_cast<uint64_t>(info->si_code) : 0);
    out = RawAppendLiteral(out, end, " fault=");
    out = RawAppendHex(out, end, reinterpret_cast<uintptr_t>(info != nullptr ? info->si_addr : nullptr));
    out = RawAppendLiteral(out, end, " pc=");
    out = RawAppendHex(out, end, ContextPc(raw_context));
    out = RawAppendLiteral(out, end, " lr=");
    out = RawAppendHex(out, end, ContextLr(raw_context));
    out = RawAppendLiteral(out, end, " sp=");
    out = RawAppendHex(out, end, ContextSp(raw_context));
    out = RawAppendLiteral(out, end, " fp=");
    out = RawAppendHex(out, end, ContextFp(raw_context));
    out = RawAppendLiteral(out, end, " ctx=");
    out = RawAppendHex(out, end, reinterpret_cast<uintptr_t>(raw_context));
    if (signo == SIGSYS && info != nullptr) {
        out = RawAppendLiteral(out, end, " sys=");
        out = RawAppendDec(out, end, static_cast<uint64_t>(info->si_syscall));
        out = RawAppendLiteral(out, end, " arch=");
        out = RawAppendHex(out, end, static_cast<uint64_t>(info->si_arch));
        out = RawAppendLiteral(out, end, " call_addr=");
        out = RawAppendHex(out, end, reinterpret_cast<uintptr_t>(info->si_call_addr));
    }
    *out++ = '\n';
    RawWriteBytes(buffer, static_cast<size_t>(out - buffer));
}

void RawLogAarch64Regs(void* raw_context) {
#if defined(__aarch64__)
    RawLogLine("regs0", ContextReg(raw_context, 0), "x1", ContextReg(raw_context, 1), "x2",
               ContextReg(raw_context, 2), "x3", ContextReg(raw_context, 3));
    RawLogLine("regs4", ContextReg(raw_context, 4), "x5", ContextReg(raw_context, 5), "x6",
               ContextReg(raw_context, 6), "x7", ContextReg(raw_context, 7));
    RawLogLine("regs16", ContextReg(raw_context, 16), "x17", ContextReg(raw_context, 17), "x18",
               ContextReg(raw_context, 18), "x19", ContextReg(raw_context, 19));
    RawLogLine("regs20", ContextReg(raw_context, 20), "x21", ContextReg(raw_context, 21), "x22",
               ContextReg(raw_context, 22), "x23", ContextReg(raw_context, 23));
    RawLogLine("regs24", ContextReg(raw_context, 24), "x25", ContextReg(raw_context, 25), "x26",
               ContextReg(raw_context, 26), "x27", ContextReg(raw_context, 27));
    RawLogLine("regs28", ContextReg(raw_context, 28), "x29", ContextReg(raw_context, 29), "x30",
               ContextReg(raw_context, 30));
#else
    (void)raw_context;
#endif
}

void RawLogResolvedFrame(int index, uintptr_t raw_ret) {
    const uintptr_t addr = raw_ret & 0x7FFFFFFFFFull;
    Dl_info info{};
    if (addr == 0 || dladdr(reinterpret_cast<void*>(addr), &info) == 0 ||
        info.dli_fname == nullptr) {
        return;
    }
    const char* base_name = strrchr(info.dli_fname, '/');
    base_name = base_name != nullptr ? base_name + 1 : info.dli_fname;
    char buffer[512];
    char* out = buffer;
    char* end = buffer + sizeof(buffer) - 2;
    out = RawAppendLiteral(out, end, "[EXECUTOR_RUNTIME_DEATH_RAW] fp-sym ");
    out = RawAppendDec(out, end, static_cast<uint64_t>(index));
    out = RawAppendLiteral(out, end, " ");
    out = RawAppendHex(out, end, addr);
    out = RawAppendLiteral(out, end, " ");
    out = RawAppendLiteral(out, end, base_name);
    out = RawAppendLiteral(out, end, "+");
    out = RawAppendHex(out, end,
                       addr - reinterpret_cast<uintptr_t>(info.dli_fbase));
    if (info.dli_sname != nullptr) {
        out = RawAppendLiteral(out, end, " ");
        out = RawAppendLiteral(out, end, info.dli_sname);
        out = RawAppendLiteral(out, end, "+");
        out = RawAppendHex(out, end,
                           addr - reinterpret_cast<uintptr_t>(info.dli_saddr));
    }
    *out++ = '\n';
    RawWriteBytes(buffer, static_cast<size_t>(out - buffer));
}

void RawWalkFramePointer(uintptr_t fp, uintptr_t sp) {
    if (fp == 0 || sp == 0 || (fp & 0xfu) != 0 || fp < sp || fp - sp > 16 * 1024 * 1024) {
        RawLogLine("fp-chain-unavailable", fp, "sp", sp);
        return;
    }
    uintptr_t rets[12] = {};
    int frame_count = 0;
    for (int i = 0; i < 12; ++i) {
        const auto* frame = reinterpret_cast<const uintptr_t*>(fp);
        const uintptr_t next_fp = frame[0];
        const uintptr_t ret = frame[1];
        RawLogLine("fp-frame", static_cast<uint64_t>(i), "fp", fp, "ret", ret, "next", next_fp);
        rets[frame_count++] = ret;
        if (next_fp <= fp || (next_fp & 0xfu) != 0 || next_fp - sp > 16 * 1024 * 1024) {
            break;
        }
        fp = next_fp;
    }
    for (int i = 0; i < frame_count; ++i) {
        RawLogResolvedFrame(i, rets[i]);
    }
}

void LogInterestingMaps(const char* reason) {
    FILE* file = fopen("/proc/self/maps", "r");
    if (file == nullptr) {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] maps open failed reason=%s errno=%d",
                        reason != nullptr ? reason : "?", errno);
        return;
    }
    char line[1024];
    while (fgets(line, sizeof(line), file) != nullptr) {
        if (strstr(line, "liblsx4_executor_android") || strstr(line, "libbox64_executor_embed") ||
            strstr(line, "libGLES") || strstr(line, "libEGL") || strstr(line, "libc.so") ||
            strstr(line, "scudo") || strstr(line, "vulkan") || strstr(line, "kgsl")) {
            const size_t length = strnlen(line, sizeof(line));
            if (length > 0 && line[length - 1] == '\n') {
                line[length - 1] = '\0';
            }
            LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH_MAP] reason=%s %s",
                            reason != nullptr ? reason : "?", line);
        }
    }
    fclose(file);
}

void LogSymbolizedAddress(const char* prefix, void* address) {
    Dl_info info{};
    if (address != nullptr && dladdr(address, &info) != 0 && info.dli_fname != nullptr) {
        const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
        const auto value = reinterpret_cast<uintptr_t>(address);
        const auto symbol = reinterpret_cast<uintptr_t>(info.dli_saddr);
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] %s pc=%p so=%s base=%p off=0x%llx sym=%s symoff=0x%llx",
                        prefix != nullptr ? prefix : "frame", address, info.dli_fname,
                        info.dli_fbase, static_cast<unsigned long long>(value - base),
                        info.dli_sname != nullptr ? info.dli_sname : "?",
                        symbol != 0 ? static_cast<unsigned long long>(value - symbol) : 0ull);
        return;
    }
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] %s pc=%p dladdr=miss",
                    prefix != nullptr ? prefix : "frame", address);
}

struct BacktraceState {
    void** current;
    void** end;
};

_Unwind_Reason_Code BacktraceCallback(_Unwind_Context* context, void* arg) {
    auto* state = static_cast<BacktraceState*>(arg);
    const uintptr_t pc = _Unwind_GetIP(context);
    if (pc != 0 && state->current < state->end) {
        *state->current++ = reinterpret_cast<void*>(pc);
    }
    return state->current < state->end ? _URC_NO_REASON : _URC_END_OF_STACK;
}

int CaptureBacktrace(void** frames, int max_frames) {
    if (frames == nullptr || max_frames <= 0) {
        return 0;
    }
    BacktraceState state{frames, frames + max_frames};
    _Unwind_Backtrace(BacktraceCallback, &state);
    return static_cast<int>(state.current - frames);
}

void LogBacktrace(const char* reason) {
    void* frames[32]{};
    const int count = CaptureBacktrace(frames, 32);
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] backtrace reason=%s frames=%d",
                    reason != nullptr ? reason : "?", count);
    for (int i = 0; i < count && i < 24; ++i) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "bt#%02d", i);
        LogSymbolizedAddress(prefix, frames[i]);
    }
}

bool SeccompExitTrapEnabled() {
    const char* env = getenv("EXECUTOR_RUNTIME_SECCOMP_EXIT_TRAP");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}

bool SeccompExitTrapUseTsync() {
    const char* env = getenv("EXECUTOR_RUNTIME_SECCOMP_EXIT_TRAP_TSYNC");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}

void InstallSeccompExitTrap() {
#if defined(__aarch64__)
    if (!SeccompExitTrapEnabled()) {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] seccomp exit trap disabled by env");
        return;
    }
    int expected = 0;
    if (!g_seccomp_exit_trap_installed.compare_exchange_strong(expected, 1,
                                                               std::memory_order_acq_rel)) {
        return;
    }

    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 static_cast<unsigned int>(offsetof(struct seccomp_data, arch))),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_AARCH64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 static_cast<unsigned int>(offsetof(struct seccomp_data, nr))),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 1, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program {};
    program.len = static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0]));
    program.filter = filter;

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        const int err = errno;
        g_seccomp_exit_trap_installed.store(0, std::memory_order_release);
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] seccomp PR_SET_NO_NEW_PRIVS failed errno=%d",
                        err);
        return;
    }
    long seccomp_result = -1;
    int seccomp_errno = 0;
#ifdef __NR_seccomp
    if (SeccompExitTrapUseTsync()) {
        seccomp_result = syscall(__NR_seccomp, SECCOMP_SET_MODE_FILTER,
                                 SECCOMP_FILTER_FLAG_TSYNC, &program);
        seccomp_errno = errno;
    }
#endif
    if (seccomp_result != 0) {
        if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) {
            const int err = errno;
            g_seccomp_exit_trap_installed.store(0, std::memory_order_release);
            LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] seccomp install failed tsync_errno=%d prctl_errno=%d",
                            seccomp_errno, err);
            return;
        }
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] installed seccomp exit trap via prctl current-thread tsync_errno=%d ret_trap=0x%x",
                        seccomp_errno, SECCOMP_RET_TRAP);
    } else {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] installed seccomp exit trap via seccomp TSYNC ret_trap=0x%x",
                        SECCOMP_RET_TRAP);
    }
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] installed seccomp exit trap arch=0x%x exit=%d exit_group=%d action=RET_TRAP(0x%x) tsync=%d",
                    AUDIT_ARCH_AARCH64, __NR_exit, __NR_exit_group, SECCOMP_RET_TRAP,
                    SeccompExitTrapUseTsync() ? 1 : 0);
    RawLogLine("installed-seccomp-exit-trap", static_cast<uint64_t>(AUDIT_ARCH_AARCH64), "exit",
               static_cast<uint64_t>(__NR_exit), "exit_group",
               static_cast<uint64_t>(__NR_exit_group), "ret_trap", SECCOMP_RET_TRAP);
#else
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] seccomp exit trap unsupported on this arch");
#endif
}

void LogDeathFromWrapper(const char* reason, int code_or_signal, void* caller) {
    if (g_death_logging.exchange(1, std::memory_order_acq_rel) != 0) {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] recursive reason=%s code=%d caller=%p",
                        reason != nullptr ? reason : "?", code_or_signal, caller);
        return;
    }
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] wrapper reason=%s code=%d tid=%ld caller=%p",
                    reason != nullptr ? reason : "?", code_or_signal,
                    static_cast<long>(syscall(__NR_gettid)), caller);
    LogSymbolizedAddress("wrapper-caller", caller);
    LogBacktrace(reason);
    g_death_logging.store(0, std::memory_order_release);
}

__attribute__((noreturn)) void RawReraiseSignal(int signo) {
    struct sigaction action {};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    sigaction(signo, &action, nullptr);

    const pid_t pid = getpid();
    const pid_t tid = static_cast<pid_t>(syscall(__NR_gettid));
    syscall(__NR_tgkill, pid, tid, signo);
#ifdef __NR_exit_group
    syscall(__NR_exit_group, 128 + signo);
#endif
    syscall(__NR_exit, 128 + signo);
    for (;;) {
    }
}

bool IsAndroidRuntimeSignalPc(uintptr_t pc) {
    if (pc == 0) {
        return false;
    }
    Dl_info info {};
    if (dladdr(reinterpret_cast<void*>(pc), &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }
    return strstr(info.dli_fname, "/libart.so") != nullptr ||
           strstr(info.dli_fname, "/libartbase.so") != nullptr ||
           strstr(info.dli_fname, "/libsigchain.so") != nullptr;
}

bool ChainPreviousSignalHandler(int signo, siginfo_t* info, void* raw_context) {
    constexpr int previous_action_count =
        static_cast<int>(sizeof(g_previous_actions) / sizeof(g_previous_actions[0]));
    if (signo <= 0 || signo >= previous_action_count) {
        return false;
    }
    const struct sigaction& previous = g_previous_actions[signo];
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH_CHAIN] signal=%d pc=%p previous_handler=%p previous_sigaction=%p flags=0x%x",
                    signo, reinterpret_cast<void*>(ContextPc(raw_context)),
                    reinterpret_cast<void*>(previous.sa_handler),
                    reinterpret_cast<void*>(previous.sa_sigaction), previous.sa_flags);
    if ((previous.sa_flags & SA_SIGINFO) && previous.sa_sigaction &&
        previous.sa_sigaction != RuntimeDeathSignalHandler) {
        previous.sa_sigaction(signo, info, raw_context);
        return true;
    }
    if (previous.sa_handler == SIG_IGN) {
        return true;
    }
    if (previous.sa_handler && previous.sa_handler != SIG_DFL &&
        reinterpret_cast<void*>(previous.sa_handler) !=
            reinterpret_cast<void*>(RuntimeDeathSignalHandler)) {
        previous.sa_handler(signo);
        return true;
    }
    return false;
}

void RuntimeDeathSignalHandler(int signo, siginfo_t* info, void* raw_context) {
    if ((signo == SIGSEGV || signo == SIGBUS || signo == SIGILL) &&
        IsAndroidRuntimeSignalPc(ContextPc(raw_context)) &&
        ChainPreviousSignalHandler(signo, info, raw_context)) {
        return;
    }
    if (g_signal_death_logging != 0) {
        RawLogLine("recursive-signal", static_cast<uint64_t>(signo), "fault",
                   reinterpret_cast<uintptr_t>(info != nullptr ? info->si_addr : nullptr));
        RawReraiseSignal(signo);
    }
    g_signal_death_logging = 1;

    RawLogSignalHeader(signo, info, raw_context);
    RawLogAarch64Regs(raw_context);
    RawWalkFramePointer(ContextFp(raw_context), ContextSp(raw_context));
    RawLogLine("reraise", static_cast<uint64_t>(signo), "pc", ContextPc(raw_context), "lr",
               ContextLr(raw_context));
    RawReraiseSignal(signo);
}

void EnsureRuntimeDeathAltStack() {
    if (g_runtime_death_alt_stack != nullptr) {
        return;
    }
    void* memory = mmap(nullptr, kAltStackSize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] altstack mmap failed tid=%ld errno=%d",
                        static_cast<long>(syscall(__NR_gettid)), errno);
        return;
    }

    stack_t alt_stack{};
    alt_stack.ss_sp = memory;
    alt_stack.ss_size = kAltStackSize;
    alt_stack.ss_flags = 0;
    if (sigaltstack(&alt_stack, nullptr) != 0) {
        const int err = errno;
        munmap(memory, kAltStackSize);
        LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] sigaltstack install failed tid=%ld errno=%d",
                        static_cast<long>(syscall(__NR_gettid)), err);
        return;
    }

    g_runtime_death_alt_stack = memory;
    g_runtime_death_alt_stack_size = kAltStackSize;
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] installed runtime altstack tid=%ld sp=%p size=%zu",
                    static_cast<long>(syscall(__NR_gettid)), g_runtime_death_alt_stack,
                    g_runtime_death_alt_stack_size);
}

void InstallRuntimeDeathCatcher() {
    EnsureRawDeathLogFile();
    EnsureRuntimeDeathAltStack();

    static std::terminate_handler previous_terminate = nullptr;
    static std::once_flag terminate_once;
    std::call_once(terminate_once, [] {
        previous_terminate = std::set_terminate([] {
            const std::type_info* type = __cxxabiv1::__cxa_current_exception_type();
            const char* type_name = type != nullptr ? type->name() : "<none>";
            LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] terminate exceptionType=%s", type_name);
            RawLogString("terminate_type", type_name);
            if (type != nullptr) {
                try {
                    throw;
                } catch (const std::exception& e) {
                    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] terminate what=%s", e.what());
                    RawLogString("terminate_what", e.what());
                } catch (...) {
                }
            }
            if (previous_terminate != nullptr) {
                previous_terminate();
            }
            abort();
        });
    });

    const char* catch_fault_signals_env = getenv("EXECUTOR_RUNTIME_DEATH_CATCH_FAULT_SIGNALS");
    const bool catch_fault_signals =
        catch_fault_signals_env != nullptr && catch_fault_signals_env[0] != '\0' &&
        strcmp(catch_fault_signals_env, "0") != 0;

    struct sigaction action {};
    action.sa_sigaction = RuntimeDeathSignalHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
#ifdef SA_RESTART
    action.sa_flags |= SA_RESTART;
#endif
    for (const int signo : kSignals) {
        if (!catch_fault_signals &&
            (signo == SIGSEGV || signo == SIGBUS || signo == SIGILL)) {
            LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] skip fault signal=%d in Android mixed ART/Box64 process",
                            signo);
            continue;
        }
        constexpr int previous_action_count =
            static_cast<int>(sizeof(g_previous_actions) / sizeof(g_previous_actions[0]));
        if (signo > 0 && signo < previous_action_count) {
            if (sigaction(signo, &action, &g_previous_actions[signo]) != 0) {
                LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] sigaction install failed signal=%d errno=%d",
                                signo, errno);
            }
        }
    }
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] installed runtime death catcher tid=%ld",
                    static_cast<long>(syscall(__NR_gettid)));
    RawLogLine("installed", static_cast<uint64_t>(syscall(__NR_gettid)));
    if (getenv("EXECUTOR_TRACE_RUNTIME_DEATH_MAPS") != nullptr &&
        g_maps_logged.exchange(1, std::memory_order_acq_rel) == 0) {
        LogInterestingMaps("install");
    }
}

}

extern "C" {

void __real_abort(void);
void __real_exit(int status);
void __real__exit(int status);
void __real__Exit(int status);
void __real___stack_chk_fail(void);
void __real___assert2(const char* file, int line, const char* function, const char* expression);

__attribute__((constructor)) void executor_runtime_death_catcher_constructor() {
    InstallRuntimeDeathCatcher();
}

void executor_runtime_death_catcher_reinstall() {
    InstallRuntimeDeathCatcher();
}

void executor_runtime_death_catcher_install_seccomp_exit_trap() {
    EnsureRawDeathLogFile();
    InstallSeccompExitTrap();
}

__attribute__((noreturn)) void __wrap_abort(void) {
    LogDeathFromWrapper("abort", SIGABRT, __builtin_return_address(0));
    __real_abort();
    __builtin_unreachable();
}

__attribute__((noreturn)) void __wrap_exit(int status) {
    LogDeathFromWrapper("exit", status, __builtin_return_address(0));
    __real_exit(status);
    __builtin_unreachable();
}

__attribute__((noreturn)) void __wrap__exit(int status) {
    LogDeathFromWrapper("_exit", status, __builtin_return_address(0));
    __real__exit(status);
    __builtin_unreachable();
}

__attribute__((noreturn)) void __wrap__Exit(int status) {
    LogDeathFromWrapper("_Exit", status, __builtin_return_address(0));
    __real__Exit(status);
    __builtin_unreachable();
}

__attribute__((noreturn)) void __wrap_exit_group(int status) {
    LogDeathFromWrapper("exit_group", status, __builtin_return_address(0));
#ifdef __NR_exit_group
    syscall(__NR_exit_group, status);
#endif
    syscall(__NR_exit, status);
    __builtin_unreachable();
}

int __wrap_tgkill(int tgid, int tid, int sig) {
    LogDeathFromWrapper("tgkill", sig, __builtin_return_address(0));
    return static_cast<int>(syscall(__NR_tgkill, tgid, tid, sig));
}

__attribute__((noreturn)) void __wrap___stack_chk_fail(void) {
    LogDeathFromWrapper("__stack_chk_fail", 0, __builtin_return_address(0));
    __real___stack_chk_fail();
    __builtin_unreachable();
}

__attribute__((noreturn)) void __wrap___assert2(const char* file, int line, const char* function,
                                                const char* expression) {
    LogRuntimeDeath("[EXECUTOR_RUNTIME_DEATH] __assert2 file=%s line=%d function=%s expression=%s",
                    file != nullptr ? file : "?", line, function != nullptr ? function : "?",
                    expression != nullptr ? expression : "?");
    LogDeathFromWrapper("__assert2", line, __builtin_return_address(0));
    __real___assert2(file, line, function, expression);
    __builtin_unreachable();
}

}

#endif
