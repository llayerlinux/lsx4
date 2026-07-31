// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/android_performance_hint.h"

#if defined(__ANDROID__)
#include <android/api-level.h>
#include <android/log.h>
#include <android/performance_hint.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace Executor::AndroidPerformanceHint {
namespace {

constexpr std::int64_t TargetFrameDurationNs = 33'333'333;
constexpr std::int64_t MaximumReportedFrameDurationNs = 1'000'000'000;
constexpr auto ThreadRefreshPeriod = std::chrono::seconds{2};
constexpr std::uint64_t StatusLogPeriod = 300;

std::atomic_bool g_enabled{false};

#if defined(__ANDROID__)

using GetManagerFn = APerformanceHintManager* (*)();
using CreateSessionFn = APerformanceHintSession* (*)(
    APerformanceHintManager*, const std::int32_t*, std::size_t, std::int64_t);
using GetPreferredRateFn = std::int64_t (*)(APerformanceHintManager*);
using UpdateTargetFn = int (*)(APerformanceHintSession*, std::int64_t);
using ReportActualFn = int (*)(APerformanceHintSession*, std::int64_t);
using CloseSessionFn = void (*)(APerformanceHintSession*);
using SetThreadsFn = int (*)(APerformanceHintSession*, const pid_t*, std::size_t);

struct ThreadSet {
    std::vector<std::int32_t> tids;
    std::string description;
};

struct State {
    std::mutex mutex;
    bool load_attempted{};
    bool api_ready{};
    bool unsupported_logged{};
    void* library{};
    GetManagerFn get_manager{};
    CreateSessionFn create_session{};
    GetPreferredRateFn get_preferred_rate{};
    UpdateTargetFn update_target{};
    ReportActualFn report_actual{};
    CloseSessionFn close_session{};
    SetThreadsFn set_threads{};
    APerformanceHintManager* manager{};
    APerformanceHintSession* session{};
    std::vector<std::int32_t> tids;
    std::chrono::steady_clock::time_point last_present{};
    std::chrono::steady_clock::time_point next_thread_refresh{};
    std::uint64_t reports{};
    std::uint64_t report_errors{};
    std::uint64_t session_generation{};
    std::int64_t preferred_rate_ns{};
};

State& GetState() {
    static State* state = new State{};
    return *state;
}

template <typename Function>
Function Resolve(void* library, const char* name) {
    return reinterpret_cast<Function>(dlsym(library, name));
}

void LogUnsupportedOnce(State& state, const char* reason) {
    if (state.unsupported_logged) {
        return;
    }
    state.unsupported_logged = true;
    __android_log_print(ANDROID_LOG_INFO, "LSX4-ADPF",
                        "[EXECUTOR_ADPF] supported=0 enabled=1 reason=%s api=%d",
                        reason, android_get_device_api_level());
}

bool LoadApi(State& state) {
    if (state.load_attempted) {
        return state.api_ready;
    }
    state.load_attempted = true;
    if (android_get_device_api_level() < 31) {
        LogUnsupportedOnce(state, "android_api_below_31");
        return false;
    }

    state.library = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (state.library == nullptr) {
        LogUnsupportedOnce(state, "libandroid_unavailable");
        return false;
    }
    state.get_manager =
        Resolve<GetManagerFn>(state.library, "APerformanceHint_getManager");
    state.create_session =
        Resolve<CreateSessionFn>(state.library, "APerformanceHint_createSession");
    state.get_preferred_rate = Resolve<GetPreferredRateFn>(
        state.library, "APerformanceHint_getPreferredUpdateRateNanos");
    state.update_target = Resolve<UpdateTargetFn>(
        state.library, "APerformanceHint_updateTargetWorkDuration");
    state.report_actual = Resolve<ReportActualFn>(
        state.library, "APerformanceHint_reportActualWorkDuration");
    state.close_session = Resolve<CloseSessionFn>(
        state.library, "APerformanceHint_closeSession");
    state.set_threads =
        Resolve<SetThreadsFn>(state.library, "APerformanceHint_setThreads");

    state.api_ready = state.get_manager != nullptr &&
                      state.create_session != nullptr &&
                      state.get_preferred_rate != nullptr &&
                      state.update_target != nullptr &&
                      state.report_actual != nullptr &&
                      state.close_session != nullptr;
    if (!state.api_ready) {
        LogUnsupportedOnce(state, "required_symbols_unavailable");
        return false;
    }
    state.manager = state.get_manager();
    if (state.manager == nullptr) {
        state.api_ready = false;
        LogUnsupportedOnce(state, "manager_unavailable");
        return false;
    }
    state.preferred_rate_ns = state.get_preferred_rate(state.manager);
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-ADPF",
        "[EXECUTOR_ADPF] supported=1 enabled=1 api=%d targetNs=%lld "
        "preferredRateNs=%lld dynamicThreads=%d",
        android_get_device_api_level(),
        static_cast<long long>(TargetFrameDurationNs),
        static_cast<long long>(state.preferred_rate_ns),
        state.set_threads != nullptr ? 1 : 0);
    return true;
}

std::string ReadThreadName(const std::int32_t tid) {
    char path[64]{};
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/comm", tid);
    FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return {};
    }
    char name[64]{};
    const char* result = std::fgets(name, sizeof(name), file);
    std::fclose(file);
    if (result == nullptr) {
        return {};
    }
    std::string value{name};
    while (!value.empty() &&
           (value.back() == '\n' || value.back() == '\r')) {
        value.pop_back();
    }
    return value;
}

bool IsNumberedGuestWorker(std::string_view name) {
    std::string lowered{name};
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](const unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return lowered.starts_with("thread") &&
           lowered.size() > std::string_view{"thread"}.size() &&
           std::ranges::all_of(
               std::string_view{lowered}.substr(
                   std::string_view{"thread"}.size()),
               [](const unsigned char value) {
                   return std::isdigit(value) != 0;
               });
}

bool IsNamedHotRuntimeThread(std::string_view name) {
    std::string lowered{name};
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](const unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return lowered.find("game:main") != std::string::npos ||
           lowered.find("gpucomm") != std::string::npos ||
           lowered.find("fmod") != std::string::npos ||
           lowered.find("havok") != std::string::npos ||
           lowered.find("worker") != std::string::npos;
}

std::uint64_t ReadThreadCpuTicks(const std::int32_t tid) {
    char path[64]{};
    std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
    FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return 0;
    }
    char record[1024]{};
    const char* result = std::fgets(record, sizeof(record), file);
    std::fclose(file);
    if (result == nullptr) {
        return 0;
    }
    const std::string_view line{record};
    const std::size_t comm_end = line.rfind(')');
    if (comm_end == std::string_view::npos ||
        comm_end + 2 >= line.size()) {
        return 0;
    }
    std::istringstream fields{
        std::string{line.substr(comm_end + 2)}};
    std::string token;
    std::uint64_t user_ticks = 0;
    std::uint64_t system_ticks = 0;
    for (std::size_t index = 0; index <= 12; ++index) {
        if (!(fields >> token)) {
            return 0;
        }
        if (index == 11) {
            user_ticks = std::strtoull(token.c_str(), nullptr, 10);
        } else if (index == 12) {
            system_ticks = std::strtoull(token.c_str(), nullptr, 10);
        }
    }
    return user_ticks + system_ticks;
}

ThreadSet FindHotThreads() {
    ThreadSet result;
    std::vector<std::pair<std::int32_t, std::string>> selected;
    std::vector<std::tuple<std::uint64_t, std::int32_t, std::string>>
        numbered_workers;
    DIR* directory = opendir("/proc/self/task");
    if (directory != nullptr) {
        while (dirent* entry = readdir(directory)) {
            char* end{};
            const long parsed = std::strtol(entry->d_name, &end, 10);
            if (end == entry->d_name || *end != '\0' || parsed <= 0 ||
                parsed > INT32_MAX) {
                continue;
            }
            const auto tid = static_cast<std::int32_t>(parsed);
            std::string name = ReadThreadName(tid);
            if (IsNamedHotRuntimeThread(name)) {
                selected.emplace_back(tid, std::move(name));
            } else if (IsNumberedGuestWorker(name)) {
                numbered_workers.emplace_back(
                    ReadThreadCpuTicks(tid), tid, std::move(name));
            }
        }
        closedir(directory);
    }
    constexpr std::size_t MaximumNumberedWorkers = 12;
    std::sort(
        numbered_workers.begin(), numbered_workers.end(),
        [](const auto& left, const auto& right) {
            if (std::get<0>(left) != std::get<0>(right)) {
                return std::get<0>(left) > std::get<0>(right);
            }
            return std::get<1>(left) < std::get<1>(right);
        });
    const std::size_t numbered_count = std::min(
        numbered_workers.size(), MaximumNumberedWorkers);
    for (std::size_t index = 0; index < numbered_count; ++index) {
        selected.emplace_back(
            std::get<1>(numbered_workers[index]),
            std::move(std::get<2>(numbered_workers[index])));
    }

    if (selected.empty()) {
        const auto tid = static_cast<std::int32_t>(syscall(SYS_gettid));
        selected.emplace_back(tid, ReadThreadName(tid));
    }
    std::sort(selected.begin(), selected.end(),
              [](const auto& left, const auto& right) {
                  return left.first < right.first;
              });
    selected.erase(std::unique(selected.begin(), selected.end(),
                               [](const auto& left, const auto& right) {
                                   return left.first == right.first;
                               }),
                   selected.end());

    for (const auto& [tid, name] : selected) {
        result.tids.push_back(tid);
        if (!result.description.empty()) {
            result.description += ',';
        }
        result.description += name.empty() ? "unnamed" : name;
        result.description += ':';
        result.description += std::to_string(tid);
    }
    return result;
}

void CloseSession(State& state) {
    if (state.session != nullptr && state.close_session != nullptr) {
        state.close_session(state.session);
    }
    state.session = nullptr;
    state.tids.clear();
}

bool CreateSession(State& state, const ThreadSet& threads) {
    if (threads.tids.empty()) {
        return false;
    }
    state.session = state.create_session(
        state.manager, threads.tids.data(), threads.tids.size(),
        TargetFrameDurationNs);
    if (state.session == nullptr) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4-ADPF",
            "[EXECUTOR_ADPF] sessionActive=0 createFailed=1 tids=%s",
            threads.description.c_str());
        return false;
    }
    state.tids = threads.tids;
    ++state.session_generation;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-ADPF",
        "[EXECUTOR_ADPF] sessionActive=1 generation=%llu targetNs=%lld "
        "hotTidCount=%zu tids=%s",
        static_cast<unsigned long long>(state.session_generation),
        static_cast<long long>(TargetFrameDurationNs), state.tids.size(),
        threads.description.c_str());
    return true;
}

void RefreshThreads(State& state,
                    const std::chrono::steady_clock::time_point now) {
    if (now < state.next_thread_refresh && state.session != nullptr) {
        return;
    }
    state.next_thread_refresh = now + ThreadRefreshPeriod;
    ThreadSet threads = FindHotThreads();
    if (threads.tids == state.tids && state.session != nullptr) {
        return;
    }

    if (state.session != nullptr && state.set_threads != nullptr &&
        android_get_device_api_level() >= 34) {
        const int result = state.set_threads(
            state.session, reinterpret_cast<const pid_t*>(threads.tids.data()),
            threads.tids.size());
        if (result == 0) {
            state.tids = threads.tids;
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-ADPF",
                "[EXECUTOR_ADPF] sessionActive=1 threadsUpdated=1 "
                "hotTidCount=%zu tids=%s",
                state.tids.size(), threads.description.c_str());
            return;
        }
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4-ADPF",
            "[EXECUTOR_ADPF] setThreadsFailed=%d recreate=1", result);
    }

    CloseSession(state);
    CreateSession(state, threads);
}

#endif

} // namespace

void SetEnabled(const bool enabled) noexcept {
    g_enabled.store(enabled, std::memory_order_release);
#if defined(__ANDROID__)
    State& state = GetState();
    const std::lock_guard lock{state.mutex};
    state.last_present = {};
    state.next_thread_refresh = {};
    if (!enabled) {
        CloseSession(state);
        __android_log_print(ANDROID_LOG_INFO, "LSX4-ADPF",
                            "[EXECUTOR_ADPF] enabled=0 sessionActive=0");
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (LoadApi(state)) {
        state.next_thread_refresh = {};
        RefreshThreads(state, now);
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4-ADPF",
                        "[EXECUTOR_ADPF] enabled=1 sessionDeferred=0 "
                        "sessionActive=%d",
                        state.session != nullptr ? 1 : 0);
#else
    (void)enabled;
#endif
}

bool IsEnabled() noexcept {
    return g_enabled.load(std::memory_order_acquire);
}

void ReportFramePresented() noexcept {
#if defined(__ANDROID__)
    if (!g_enabled.load(std::memory_order_acquire)) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    State& state = GetState();
    const std::lock_guard lock{state.mutex};
    if (!g_enabled.load(std::memory_order_relaxed) || !LoadApi(state)) {
        return;
    }
    RefreshThreads(state, now);
    if (state.session == nullptr) {
        return;
    }

    if (state.last_present.time_since_epoch().count() == 0) {
        state.last_present = now;
        return;
    }
    const auto actual_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now - state.last_present).count();
    state.last_present = now;
    if (actual_ns <= 0 || actual_ns > MaximumReportedFrameDurationNs) {
        return;
    }

    const int result = state.report_actual(state.session, actual_ns);
    if (result != 0) {
        ++state.report_errors;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4-ADPF",
            "[EXECUTOR_ADPF] reportResult=%d actualNs=%lld errors=%llu",
            result, static_cast<long long>(actual_ns),
            static_cast<unsigned long long>(state.report_errors));
        CloseSession(state);
        state.next_thread_refresh = {};
        return;
    }

    ++state.reports;
    if (state.reports == 1 || state.reports % StatusLogPeriod == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-ADPF",
            "[EXECUTOR_ADPF] reportResult=0 reports=%llu actualNs=%lld "
            "targetNs=%lld hotTidCount=%zu generation=%llu",
            static_cast<unsigned long long>(state.reports),
            static_cast<long long>(actual_ns),
            static_cast<long long>(TargetFrameDurationNs), state.tids.size(),
            static_cast<unsigned long long>(state.session_generation));
    }
#endif
}

} // namespace Executor::AndroidPerformanceHint
