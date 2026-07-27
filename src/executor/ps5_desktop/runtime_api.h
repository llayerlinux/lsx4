// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

inline constexpr std::uint32_t LSX4_PS5_RUNTIME_ABI_VERSION = 7;
inline constexpr std::uint32_t LSX4_PS5_GUEST_PAGE_SIZE = 0x4000;

enum Lsx4Ps5GuestProtection : std::uint32_t {
    LSX4_PS5_GUEST_READ = 1u << 0,
    LSX4_PS5_GUEST_WRITE = 1u << 1,
    LSX4_PS5_GUEST_EXECUTE = 1u << 2,
};

enum Lsx4Ps5ExecuteStatus : std::int32_t {
    LSX4_PS5_EXECUTE_OK = 0,
    LSX4_PS5_EXECUTE_INVALID_ARGUMENT = -1,
    LSX4_PS5_EXECUTE_NOT_INITIALIZED = -2,
    LSX4_PS5_EXECUTE_UNMAPPED_ENTRY = -3,
    LSX4_PS5_EXECUTE_BACKEND_NOT_READY = -4,
    LSX4_PS5_EXECUTE_INTERNAL_ERROR = -5,
};

struct Lsx4Ps5RuntimeConfig {
    std::uint32_t size{sizeof(Lsx4Ps5RuntimeConfig)};
    std::uint32_t abi_version{LSX4_PS5_RUNTIME_ABI_VERSION};
    std::uint32_t guest_page_size{LSX4_PS5_GUEST_PAGE_SIZE};
    std::uint32_t reserved{};
    std::uint64_t guest_address_limit{0x10000000000ull};
    std::uint64_t executable_identity{};
    const char* artifact_directory{};
    const char* title_id{};
};

struct Lsx4Ps5GuestMapping {
    std::uint32_t size{sizeof(Lsx4Ps5GuestMapping)};
    std::uint32_t protection{};
    std::uint64_t guest_address{};
    std::uint64_t byte_count{};
    const char* label{};
};

struct Lsx4Ps5HleCall {
    std::uint64_t function{};
    std::uint64_t integer_arguments[6]{};
    std::uint64_t floating_arguments[8]{};
    std::uint64_t guest_stack{};
};

using Lsx4Ps5ResolveHle =
    int (*)(void* context, std::uint64_t thunk,
            std::uint64_t* native_function);
using Lsx4Ps5InvokeHle =
    std::uint64_t (*)(void* context, const Lsx4Ps5HleCall* call);
using Lsx4Ps5ClassifyHle =
    int (*)(void* context, std::uint64_t thunk);
using Lsx4Ps5HleReturnsZero =
    int (*)(void* context, std::uint64_t thunk);
using Lsx4Ps5HleFpResultKind =
    int (*)(void* context, std::uint64_t native_function);
using Lsx4Ps5DispatchSignal =
    int (*)(void* context, std::int32_t native_signal,
            std::int32_t signal_code, std::int32_t signal_errno,
            std::int32_t source_pid, std::uint32_t source_uid,
            std::uint64_t fault_address, std::uint64_t guest_rip,
            std::int32_t is_write);
using Lsx4Ps5BindImport =
    int (*)(void* context, const char* symbol, std::uint64_t thunk,
            std::uint64_t* native_function);

struct Lsx4Ps5RuntimeCallbacks {
    std::uint32_t size{sizeof(Lsx4Ps5RuntimeCallbacks)};
    std::uint32_t abi_version{LSX4_PS5_RUNTIME_ABI_VERSION};
    void* context{};
    Lsx4Ps5ResolveHle resolve_hle{};
    Lsx4Ps5InvokeHle invoke_hle{};
    Lsx4Ps5ClassifyHle classify_hle{};
    Lsx4Ps5HleReturnsZero hle_returns_zero{};
    Lsx4Ps5HleFpResultKind hle_fp_result_kind{};
    Lsx4Ps5DispatchSignal dispatch_signal{};
    Lsx4Ps5BindImport bind_import{};
};

struct Lsx4Ps5GuestEntry {
    std::uint32_t size{sizeof(Lsx4Ps5GuestEntry)};
    std::uint32_t argument_count{};
    std::uint64_t address{};
    std::uint64_t arguments[8]{};
    std::uint64_t fs_base{};
    std::uint64_t stack_base{};
    std::uint64_t stack_size{};
};

struct Lsx4Ps5GuestResult {
    std::uint32_t size{sizeof(Lsx4Ps5GuestResult)};
    std::int32_t status{LSX4_PS5_EXECUTE_INTERNAL_ERROR};
    std::uint64_t value{};
};

struct Lsx4Ps5LoadedExecutable {
    std::uint32_t size{sizeof(Lsx4Ps5LoadedExecutable)};
    std::uint32_t flags{};
    std::uint64_t handle{};
    std::uint64_t base{};
    std::uint64_t entry{};
    std::uint64_t mapped_size{};
    std::uint64_t tls_image{};
    std::uint64_t tls_file_size{};
    std::uint64_t tls_memory_size{};
    std::uint64_t tls_alignment{};
    std::uint64_t tls_static_offset{};
    std::uint64_t tls_module_id{};
    std::uint64_t process_param{};
    std::uint64_t owner_handle{};
    std::uint32_t initializer_count{};
    std::uint32_t reserved{};
};

struct Lsx4Ps5GuestThreadContext {
    std::uint32_t size{sizeof(Lsx4Ps5GuestThreadContext)};
    std::uint32_t reserved{};
    std::uint64_t handle{};
    std::uint64_t program_handle{};
    std::uint64_t fs_base{};
    std::uint64_t mapping_base{};
    std::uint64_t mapping_size{};
    std::uint64_t tls_static_offset{};
};

enum Lsx4Ps5LoadedExecutableFlag : std::uint32_t {
    LSX4_PS5_EXECUTABLE_NEXT_GEN = 1u << 0,
    LSX4_PS5_EXECUTABLE_SELF = 1u << 1,
    LSX4_PS5_EXECUTABLE_MODULE = 1u << 2,
    LSX4_PS5_EXECUTABLE_STARTED = 1u << 3,
};

enum Lsx4Ps5GameProbeFlag : std::uint32_t {
    LSX4_PS5_GAME_EBOOT_MAPPED = 1u << 0,
    LSX4_PS5_GAME_MODULES_SCANNED = 1u << 1,
    LSX4_PS5_GAME_IMPORTS_REBOUND = 1u << 2,
    LSX4_PS5_GAME_READY_TO_LAUNCH = 1u << 3,
};

struct Lsx4Ps5GameProbeReport {
    std::uint32_t size{sizeof(Lsx4Ps5GameProbeReport)};
    std::uint32_t flags{};
    std::uint64_t owner_handle{};
    std::uint32_t discovered_modules{};
    std::uint32_t loaded_modules{};
    std::uint32_t failed_modules{};
    std::uint32_t exported_symbols{};
    std::uint32_t dependencies{};
    std::uint32_t deferred_function_imports{};
    std::uint32_t unresolved_data_imports{};
    std::uint32_t missing_dependencies{};
    std::uint32_t reserved{};
};

extern "C" const char* executor_lsx4_ps5_runtime_abi();
extern "C" const char* executor_lsx4_ps5_runtime_status();
extern "C" int executor_lsx4_ps5_runtime_attach_surface(
    void* native_window);
extern "C" int executor_lsx4_ps5_runtime_detach_surface();
extern "C" int executor_lsx4_ps5_runtime_initialize_android(
    const char* root_directory, const char* user_id);
extern "C" int executor_lsx4_ps5_runtime_scan_game(
    const char* game_path);
extern "C" int executor_lsx4_ps5_runtime_launch_game_jit(
    const char* game_path);
extern "C" int executor_lsx4_ps5_runtime_set_pad_button(
    std::uint32_t button_mask, int pressed);
extern "C" int executor_lsx4_ps5_runtime_set_pad_axis(
    int axis, int value);
extern "C" int executor_lsx4_ps5_runtime_hud_stats(
    std::uint64_t* values, std::size_t value_count);
extern "C" int executor_lsx4_ps5_runtime_set_managed_optimization(
    int option, int enabled);
extern "C" int executor_lsx4_ps5_runtime_initialize(
    const Lsx4Ps5RuntimeConfig* config);
extern "C" int executor_lsx4_ps5_runtime_set_callbacks(
    const Lsx4Ps5RuntimeCallbacks* callbacks);
extern "C" int executor_lsx4_ps5_runtime_register_mapping(
    const Lsx4Ps5GuestMapping* mapping);
extern "C" int executor_lsx4_ps5_runtime_unregister_mapping(
    std::uint64_t guest_address, std::uint64_t byte_count);
extern "C" int executor_lsx4_ps5_runtime_execute(
    const Lsx4Ps5GuestEntry* entry, Lsx4Ps5GuestResult* result);
extern "C" int executor_lsx4_ps5_runtime_load_eboot(
    const char* path, Lsx4Ps5LoadedExecutable* executable);
extern "C" int executor_lsx4_ps5_runtime_load_module(
    std::uint64_t owner_handle, const char* path,
    Lsx4Ps5LoadedExecutable* module);
extern "C" int executor_lsx4_ps5_runtime_load_game(
    const char* game_path, Lsx4Ps5LoadedExecutable* executable,
    Lsx4Ps5GameProbeReport* report);
extern "C" int executor_lsx4_ps5_runtime_probe_report(
    std::uint64_t owner_handle, Lsx4Ps5GameProbeReport* report,
    char* text, std::size_t text_capacity);
extern "C" int executor_lsx4_ps5_runtime_unload_eboot(
    std::uint64_t handle);
extern "C" int executor_lsx4_ps5_runtime_start_module(
    std::uint64_t module_handle, Lsx4Ps5GuestResult* result);
extern "C" int executor_lsx4_ps5_runtime_stop_module(
    std::uint64_t module_handle, Lsx4Ps5GuestResult* result);
extern "C" int executor_lsx4_ps5_runtime_launch_eboot(
    std::uint64_t handle, const Lsx4Ps5GuestEntry* entry,
    Lsx4Ps5GuestResult* result);
extern "C" int executor_lsx4_ps5_runtime_create_thread_context(
    std::uint64_t program_handle,
    Lsx4Ps5GuestThreadContext* thread_context);
extern "C" int executor_lsx4_ps5_runtime_destroy_thread_context(
    std::uint64_t thread_handle);
extern "C" void executor_lsx4_ps5_runtime_reset();
