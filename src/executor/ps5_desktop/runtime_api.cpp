// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/runtime_api.h"
#include "executor/ps5_desktop/gen5_compute_recompiler.h"

#include "executor/dynamic_translation/runtime_bridge_api.h"
#include "executor/dynamic_translation/runtime_gateway.h"
#include "executor/dynamic_translation/hle_thunk_identity.h"
#include "executor/dynamic_translation/live_state_port.h"
#include "executor/dynamic_translation/retiring_execution_core.h"
#include "executor/dynamic_translation/stack_windows.h"
#include "executor/ps5_desktop/backend_contract.h"
#include "executor/ps5_desktop/bc7decomp.h"
#include "executor/ps5_desktop/vulkan_presenter.h"
#include "ps5_desktop/nextgen_loader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <aaudio/AAudio.h>
#include <android/log.h>
#include <android/native_window.h>
#include <png.h>
#endif

extern "C" std::uint64_t executor_jit_current_fault_guest_rip();
extern "C" std::uint64_t executor_jit_current_fault_active_block_rip();
extern "C" std::uint64_t executor_jit_current_fault_instruction();
extern "C" std::uint32_t executor_jit_current_fault_instruction_meta();
extern "C" std::uint64_t executor_jit_current_fault_gpr(std::uint32_t index);
extern "C" std::uint32_t executor_jit_current_fault_recent_rips(
    std::uint64_t* output, std::uint32_t capacity);
extern "C" void executor_jit_dump_thread_states(const char* reason);

namespace {

constexpr std::uint64_t Ps5HleSlabHint = 0x8f0000000ull;
constexpr std::uint64_t Ps5HleSlabStride = 0x100000ull;
constexpr std::uint64_t Ps5HleSlabSlots = 128;
constexpr std::uint64_t Ps5ImageBase = 0x900000000ull;
constexpr std::uint64_t Ps5ImageSessionStride = 0x20000000ull;
constexpr std::uint64_t Ps5ImageSessionSlots = 32;
constexpr std::size_t Ps5HleSlabSize = 0x100000;
constexpr std::size_t Ps5HleStubSize = 0x100;
constexpr std::uint64_t Ps5TlsRegionHint = 0x8d0000000ull;
constexpr std::uint64_t Ps5TlsRegionStride = 0x100000ull;
constexpr std::uint64_t Ps5HleDataHint = 0x8e0000000ull;
constexpr std::uint64_t Ps5HleDataStride = 0x100000ull;
constexpr std::size_t Ps5HleDataSize = 0x100000;
constexpr std::size_t Ps5MutexObjectPoolOffset = 0x4000;
constexpr std::size_t Ps5MutexObjectSize = 0x100;
constexpr std::uint64_t Ps5DirectMapHint = 0x2000000000ull;
constexpr std::uint64_t Ps5DirectMapStride = 0x20000000ull;
// Gen5's flexible virtual allocator starts below the direct-memory aperture.
// Keeping it in the 0x20... direct-map range eventually exhausts every coarse
// search slot in games which map large direct heaps during startup.
constexpr std::uint64_t Ps5FlexibleMapHint = 0x600000000ull;
constexpr std::uint64_t Ps5LibcHeapHint = 0x3000000000ull;
constexpr std::uint64_t Ps5LibcHeapStride = 0x20000000ull;
constexpr std::uint64_t Ps5DirectMemorySize =
    16384ull * 1024ull * 1024ull;
constexpr std::size_t Ps5StackGuardOffset = 0;
constexpr std::size_t Ps5ProgramNamePointerOffset = 0x20;
constexpr std::size_t Ps5ProgramNameBufferOffset = 0x40;
constexpr std::size_t Ps5ProgramNameBufferSize = 0x200;
constexpr std::size_t Ps5LibcNeedFlagOffset = 0x300;
constexpr std::size_t Ps5LibcInternalNeedFlagOffset = 0x304;
constexpr std::size_t Ps5ProcessEntryParamsOffset = 0x320;
constexpr std::size_t Ps5StaticTlsReservation = 0x20000;
constexpr std::size_t Ps5TcbAndDtvSize = 0x4000;
constexpr std::size_t Ps5TlsRegionSize =
    Ps5StaticTlsReservation + Ps5TcbAndDtvSize;
constexpr std::uint64_t Ps5StackCanary = 0xc0dec0decafeba00ull;

struct RegisteredMapping {
    std::uint64_t address{};
    std::uint64_t byte_count{};
    std::uint32_t protection{};
    std::string label;
    bool internal{};
};

class GuestMappingRegistry {
public:
    static constexpr std::uint64_t WideMappingBytes = 1ull << 20;
    static constexpr std::size_t ProbeBudget = 16;

    using Container = std::multimap<std::uint64_t, RegisteredMapping>;

    [[nodiscard]] const RegisteredMapping* Find(
        const std::uint64_t address, const std::uint64_t byte_count,
        const std::uint32_t access) const {
        if (const auto* const narrow =
                FindIn(narrow_, address, byte_count, access, ProbeBudget);
            narrow != nullptr) {
            return narrow;
        }
        return FindIn(wide_, address, byte_count, access,
                      std::numeric_limits<std::size_t>::max());
    }

    [[nodiscard]] std::size_t NarrowCount() const noexcept {
        return narrow_.size();
    }

    [[nodiscard]] std::size_t WideCount() const noexcept {
        return wide_.size();
    }

    void Insert(RegisteredMapping mapping) {
        const auto address = mapping.address;
        auto& index = mapping.byte_count >= WideMappingBytes ? wide_ : narrow_;
        index.emplace(address, std::move(mapping));
        generation_.fetch_add(1, std::memory_order_release);
    }

    std::size_t EraseExact(const std::uint64_t address,
                           const std::uint64_t byte_count,
                           const bool internal) {
        const auto removed = EraseExactIn(
                                 byte_count >= WideMappingBytes ? wide_ : narrow_,
                                 address, byte_count, internal);
        if (removed != 0) {
            generation_.fetch_add(1, std::memory_order_release);
        }
        return removed;
    }

    std::size_t EraseAt(const std::uint64_t address,
                        const std::uint64_t byte_count) {
        auto& index = byte_count >= WideMappingBytes ? wide_ : narrow_;
        const auto range = index.equal_range(address);
        std::size_t removed{};
        for (auto entry = range.first; entry != range.second;) {
            if (entry->second.byte_count == byte_count) {
                entry = index.erase(entry);
                ++removed;
            } else {
                ++entry;
            }
        }
        if (removed != 0) {
            generation_.fetch_add(1, std::memory_order_release);
        }
        return removed;
    }

    void Clear() {
        narrow_.clear();
        wide_.clear();
        generation_.fetch_add(1, std::memory_order_release);
    }

    template <typename Visitor>
    [[nodiscard]] const RegisteredMapping* FindIf(Visitor&& visitor) const {
        for (const auto* const index : {&narrow_, &wide_}) {
            for (const auto& [key, mapping] : *index) {
                if (visitor(mapping)) {
                    return &mapping;
                }
            }
        }
        return nullptr;
    }

    [[nodiscard]] std::size_t Size() const noexcept {
        return narrow_.size() + wide_.size();
    }

    [[nodiscard]] std::uint64_t Generation() const noexcept {
        return generation_.load(std::memory_order_acquire);
    }

private:
    static bool Contains(const RegisteredMapping& mapping,
                         const std::uint64_t address,
                         const std::uint64_t byte_count) noexcept {
        return address >= mapping.address && byte_count <= mapping.byte_count &&
               address - mapping.address <= mapping.byte_count - byte_count;
    }

    static const RegisteredMapping* FindIn(const Container& index,
                                           const std::uint64_t address,
                                           const std::uint64_t byte_count,
                                           const std::uint32_t access,
                                           std::size_t budget) {
        auto entry = index.upper_bound(address);
        while (entry != index.begin() && budget-- != 0) {
            --entry;
            const auto& mapping = entry->second;
            if ((mapping.protection & access) == access &&
                Contains(mapping, address, byte_count)) {
                return &mapping;
            }
        }
        return nullptr;
    }

    static std::size_t EraseExactIn(Container& index,
                                    const std::uint64_t address,
                                    const std::uint64_t byte_count,
                                    const bool internal) {
        const auto range = index.equal_range(address);
        std::size_t removed{};
        for (auto entry = range.first; entry != range.second;) {
            if (entry->second.byte_count == byte_count &&
                entry->second.internal == internal) {
                entry = index.erase(entry);
                ++removed;
            } else {
                ++entry;
            }
        }
        return removed;
    }

    Container narrow_;
    Container wide_;
    std::atomic<std::uint64_t> generation_{1};
};

struct LoadedProgram {
    std::uint64_t handle{};
    std::uint64_t owner_handle{};
    bool module{};
    bool started{};
    bool start_at_boot{};
    std::string path;
    bool module_scan_complete{};
    std::uint32_t discovered_modules{};
    std::uint32_t failed_modules{};
    Funnel::Ps5Desktop::LoadedImage image;
};

struct GuestThreadContext {
    std::uint64_t handle{};
    std::uint64_t pthread_handle{};
    std::uint64_t program_handle{};
    std::uint8_t* allocation{};
    std::size_t mapping_size{};
    std::uint64_t fs_base{};
    std::uint64_t tls_static_offset{};
};

struct HleBinding {
    std::uint64_t owner_handle{};
    std::uint64_t native_function{};
    std::string symbol;
};

struct DirectAllocation {
    std::uint64_t offset{};
    std::uint64_t byte_count{};
    std::uint64_t alignment{};
    std::int32_t memory_type{};
};

struct OwnedGuestMapping {
    std::uint8_t* allocation{};
    std::size_t byte_count{};
};

struct LibcHeapAllocation {
    std::uint8_t* allocation{};
    std::size_t mapped_size{};
    std::uint64_t address{};
    std::uint64_t requested_size{};
    std::uint64_t alignment{};
    std::array<std::uint64_t, 4> allocation_returns{};
};

struct GuestStdioFile {
    std::uint64_t guest_handle{};
    std::FILE* stream{};
};

struct GuestStdioHandlePage {
    std::uint8_t* allocation{};
    std::size_t mapped_size{};
    std::uint64_t base{};
    std::size_t used{};
};

struct GuestKernelFile {
    std::int32_t handle{};
    std::filesystem::path path;
    std::FILE* stream{};
};

struct GuestMutex {
    std::recursive_timed_mutex mutex;
};

struct GuestPthreadAttribute {
    std::uint64_t affinity_mask{UINT64_C(0x7f)};
    std::int32_t detach_state{};
    std::uint64_t stack_address{};
    std::uint64_t stack_size{UINT64_C(0x100000)};
    std::uint64_t guard_size{UINT64_C(0x1000)};
    std::int32_t inherit_sched{4};
    std::int32_t sched_policy{1};
    std::int32_t sched_priority{700};
};

struct GuestSemaphore {
    std::mutex mutex;
    std::condition_variable condition;
    std::int32_t initial_count{};
    std::int32_t maximum_count{};
    std::int32_t count{};
    std::uint32_t waiting_threads{};
    std::uint64_t cancel_generation{};
    bool deleted{};
};

struct GuestConditionVariable {
    std::mutex mutex;
    std::condition_variable condition;
    std::uint32_t waiting_threads{};
    std::uint32_t pending_signals{};
    bool deleted{};
};

struct GuestEventFlag {
    std::mutex mutex;
    std::condition_variable condition;
    std::atomic<std::uint64_t> bits{};
    std::uint32_t attributes{};
    std::atomic<std::uint32_t> waiting_threads{};
    std::atomic<std::uint64_t> waiting_patterns{};
    std::atomic<bool> deleted{};
};

struct GuestVideoOutPort {
    struct FlipEventRegistration {
        std::uint64_t event_queue{};
        std::uint64_t user_data{};
    };

    std::array<std::uint64_t, 16> buffer_addresses{};
    std::uint64_t pixel_format{};
    std::uint32_t tiling_mode{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t option{};
    std::int32_t flip_rate{};
    std::int32_t current_buffer{-1};
    std::uint64_t flip_count{};
    std::uint64_t last_flip_argument{};
    std::uint64_t last_flip_process_time{};
    std::uint64_t last_flip_process_counter{};
    std::uint64_t last_flip_submit_counter{};
    std::uint64_t vblank_count{};
    std::chrono::steady_clock::time_point opened_at{
        std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point last_vblank{
        std::chrono::steady_clock::now()};
    std::array<std::int32_t, 16> buffer_group_indices{
        -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<FlipEventRegistration> flip_events;
    std::vector<FlipEventRegistration> vblank_events;
};

struct GuestAudioOutPort {
    std::uint32_t buffer_length{};
    std::uint32_t frequency{};
    std::int32_t format{};
    std::uint32_t channels{};
    std::uint32_t bytes_per_sample{};
    float volume{1.0f};
    std::uint64_t output_count{};
#ifdef __ANDROID__
    AAudioStream* stream{};
#endif
};

struct GuestPresenterTexture {
    std::uint64_t address{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t tile_mode{};
    std::uint32_t unified_format{};
};

struct AgcCpuFrame {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t rendered_draws{};
    std::uint64_t covered_pixels{};
    std::uint64_t eligible_draw_mask{};
    std::uint64_t rendered_draw_mask{};
    std::array<std::uint64_t, 64> draw_coverage{};
    std::shared_ptr<Lsx4::Ps5Desktop::VulkanGuestFrame> gpu_frame;
    std::vector<std::uint8_t> rgba;
};

bool HasAgcFrameContent(const AgcCpuFrame& frame) {
    return !frame.rgba.empty() ||
        (frame.gpu_frame != nullptr &&
         !frame.gpu_frame->draws.empty());
}

constexpr std::uint32_t AgcCpuFrameWidth = 640;
constexpr std::uint32_t AgcCpuFrameHeight = 360;
constexpr std::uint32_t AgcCpuFrameCaptureIntervalMs = 16u;

bool UseVulkanGuestRaster() {
#ifdef __ANDROID__
    static const bool enabled = []() {
        const char* const value =
            std::getenv("PS5_VULKAN_GUEST_RASTER");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
#else
    return false;
#endif
}

bool UseFullGen5Graphics() {
#ifdef __ANDROID__
    const char* const value =
        std::getenv("PS5_FULL_GEN5_GRAPHICS");
    return value != nullptr && std::strcmp(value, "1") == 0;
#else
    return false;
#endif
}
// A gameplay frame commonly references more than 24 atlas textures.  Keeping
// less than one full working set makes the FIFO cache decode the same BC
// atlases again on every submit.
constexpr std::uint32_t AgcCpuTextureCacheLimit = 64u;

struct AgcDiagnosticDraw {
    std::uint64_t texture_address{};
    std::uint64_t render_target_address{};
    std::uint64_t index_buffer_address{};
    std::uint32_t texture_width{};
    std::uint32_t texture_height{};
    std::uint32_t texture_tile_mode{};
    std::uint32_t texture_format{};
    std::uint32_t render_target_width{};
    std::uint32_t render_target_height{};
    std::uint32_t render_target_format{};
    std::uint32_t render_target_number_type{};
    std::uint32_t render_target_channel_order{};
    std::uint32_t render_target_tile_mode{};
    std::uint32_t render_target_slot{};
    std::uint32_t index_buffer_count{};
    std::uint32_t index_size{};
    std::uint32_t index_offset{};
    std::uint32_t draw_count{};
    std::uint32_t instance_count{1};
    std::uint32_t primitive_type{};
    std::uint32_t cx_target_mask{};
    std::uint32_t cx_blend0_control{};
    std::uint32_t cx_color_base{};
    std::uint32_t cx_color_base_ext{};
    std::uint32_t cx_color_info{};
    std::uint32_t cx_color_attrib2{};
    std::uint32_t cx_color_attrib3{};
    std::uint32_t sh_ps_program_lo{};
    std::uint32_t sh_ps_program_hi{};
    std::uint32_t sh_es_program_lo{};
    std::uint32_t sh_es_program_hi{};
    std::array<std::uint32_t, 16> sh_vs_user_data{};
    std::array<std::uint32_t, 16> sh_gs_user_data{};
    std::array<std::uint32_t, 16> sh_es_user_data{};
    std::array<std::uint32_t, 16> sh_export_user_data{};
    std::array<std::uint32_t, 16> sh_ps_user_data{};
};

struct LoaderBindingContext {
    std::uint64_t owner_handle{};
    std::vector<Funnel::Ps5Desktop::LoadedSymbol> exports;
    bool rebind_only{};
};

struct AprCommandBufferState {
    std::uint64_t buffer{};
    std::uint64_t size{};
    std::uint64_t write_offset{};
    std::uint64_t command_count{};
};

struct AprCompletionEvent {
    std::uint64_t event_queue{};
    std::uint64_t ident{};
    std::uint64_t completion_token{};
    std::uint64_t user_data{};
};

struct RuntimeState {
    std::mutex mutex;
    bool initialized{};
    std::uint64_t guest_address_limit{0x10000000000ull};
    std::uint64_t preferred_image_base{Ps5ImageBase};
    Lsx4Ps5RuntimeCallbacks callbacks{};
    GuestMappingRegistry mappings;
    std::vector<LoadedProgram> programs;
    std::vector<GuestThreadContext> threads;
    std::unordered_map<std::uint64_t, HleBinding> hle_bindings;
    std::vector<DirectAllocation> direct_allocations;
    std::vector<OwnedGuestMapping> owned_guest_mappings;
    std::unordered_map<std::uint64_t, LibcHeapAllocation>
        libc_heap_allocations;
    std::vector<std::thread> guest_host_threads;
    std::unordered_map<std::uint64_t, std::shared_ptr<GuestMutex>>
        guest_mutexes;
    std::unordered_map<std::uint64_t, GuestPthreadAttribute>
        guest_pthread_attributes;
    std::unordered_map<std::uint32_t, std::shared_ptr<GuestSemaphore>>
        guest_semaphores;
    std::unordered_map<std::uint64_t,
                       std::shared_ptr<GuestConditionVariable>>
        guest_condition_variables;
    std::unordered_map<std::uint64_t, std::shared_ptr<GuestEventFlag>>
        guest_event_flags;
    std::unordered_map<std::int32_t, GuestVideoOutPort>
        guest_video_out_ports;
    std::unordered_map<std::int32_t, GuestAudioOutPort>
        guest_audio_out_ports;
    std::unordered_map<
        std::uint64_t,
        std::deque<std::array<std::uint8_t, 0x20>>>
        guest_event_queues;
    std::condition_variable guest_event_queue_condition;
    bool guest_vblank_thread_started{};
    std::atomic<bool> diagnostic_second_event_queue_created{false};
    std::atomic<bool> guest_login_event_delivered{false};
    std::vector<GuestStdioFile> stdio_files;
    std::vector<GuestStdioHandlePage> stdio_handle_pages;
    std::vector<GuestKernelFile> kernel_files;
    std::unordered_map<std::uint32_t, std::filesystem::path>
        apr_files;
    std::unordered_map<std::string, std::uint32_t>
        apr_file_ids_by_path;
    std::unordered_map<std::uint32_t, std::uint64_t>
        apr_submissions;
    std::unordered_map<
        std::uint64_t,
        std::vector<std::pair<std::uint64_t, std::uint64_t>>>
        apr_completion_writes;
    std::unordered_map<std::uint64_t, AprCommandBufferState>
        apr_command_buffers;
    std::unordered_map<std::uint64_t, std::vector<AprCompletionEvent>>
        apr_completion_events;
    std::unordered_map<std::string, std::filesystem::path>
        resolved_guest_paths;
    std::unordered_map<std::string, std::uint64_t>
        resolved_guest_file_sizes;
    std::unordered_map<std::string, std::uint64_t>
        indexed_guest_file_sizes;
    std::filesystem::path app0_directory;
    std::filesystem::path save_data_directory;
#ifdef __ANDROID__
    bool game_splash_active{};
    bool game_splash_presented{};
#endif
    std::uint64_t agc_register_defaults{};
    std::uint64_t agc_internal_register_defaults{};
    std::atomic<std::int32_t> msg_dialog_status{0};
    std::atomic<std::int32_t> save_data_dialog_status{0};
    std::atomic<std::int32_t> save_data_dialog_mode{0};
    std::atomic<std::uint64_t> save_data_dialog_user_data{0};
    std::array<char, 32> save_data_dialog_dir_name{};
    std::atomic<bool> shutdown_requested{false};
    std::uint64_t next_program_handle{1};
    std::uint64_t next_thread_handle{1};
    std::uint64_t next_pthread_attribute_handle{1};
    std::uint32_t next_semaphore_handle{1};
    std::uint64_t next_event_flag_handle{1};
    std::int32_t next_video_out_handle{1};
    std::int32_t next_audio_out_handle{1};
    std::int32_t next_net_pool_handle{1};
    std::int32_t next_ssl_handle{1};
    std::int32_t next_http2_handle{1};
    std::uint64_t next_event_queue_handle{1};
    std::int32_t next_kernel_file_handle{3};
    std::uint32_t next_apr_file_id{1};
    std::uint32_t next_apr_submission_id{1};
    std::uint64_t apr_read_count{};
    std::uint64_t agc_flip_count{};
    std::uint64_t agc_dumped_frames{};
    std::uint64_t agc_dumped_dcbs{};
    std::uint64_t agc_dumped_draw_states{};
    std::uint64_t agc_draw_count{};
    std::uint64_t agc_dumped_textures{};
    std::uint64_t agc_submit_count{};
    std::uint64_t agc_last_draw_signature{};
    std::vector<std::uint64_t> agc_logged_draw_signatures;
    std::uint32_t agc_draw_signature_logs{};
    std::unordered_map<std::uint32_t, std::uint32_t>
        agc_cx_registers;
    std::unordered_map<std::uint32_t, std::uint32_t>
        agc_sh_registers;
    std::unordered_map<std::uint32_t, std::uint32_t>
        agc_uc_registers;
    std::uint64_t agc_index_buffer_address{};
    std::uint32_t agc_index_buffer_count{};
    std::uint32_t agc_index_size{};
    std::uint32_t agc_instance_count{1};
    std::uint64_t agc_dispatch_indirect_args_base{};
    std::uint64_t agc_invalid_flip_count{};
    std::chrono::steady_clock::time_point agc_next_flip{};
    std::chrono::steady_clock::time_point agc_cpu_next_render{};
    std::uint64_t agc_cpu_texture_cache_epoch{1};
    bool agc_cpu_capture_active{};
    std::uint64_t agc_cpu_capture_target{};
    std::uint64_t agc_cpu_pending_capture_target{};
    GuestPresenterTexture presenter_texture{};
    std::shared_ptr<AgcCpuFrame> agc_cpu_frame;
    std::unordered_map<std::uint64_t, std::shared_ptr<AgcCpuFrame>>
        agc_cpu_surfaces;
    std::unordered_map<std::uint64_t, std::shared_ptr<AgcCpuFrame>>
        agc_cpu_working_surfaces;
    std::uint32_t agc_cpu_quality_command_draws{};
    std::uint32_t agc_cpu_best_rendered_draws{};
    std::uint64_t agc_cpu_best_coverage{};
#ifdef __ANDROID__
    ANativeWindow* android_window{};
#endif
    std::uint64_t next_direct_offset{};
    std::uint64_t next_direct_map_hint{Ps5DirectMapHint};
    std::uint64_t next_flexible_map_hint{Ps5FlexibleMapHint};
    std::uint64_t next_libc_heap_hint{Ps5LibcHeapHint};
    std::uint8_t* hle_slab{};
    std::size_t hle_slab_used{};
    std::uint64_t hle_slab_slot{};
    std::uint8_t* hle_data{};
    std::size_t next_mutex_object_offset{Ps5MutexObjectPoolOffset};
    std::string status{"PS5 runtime is not initialized"};
};

RuntimeState g_runtime;

bool EntryDirectlyCallsInitializer(
    const Funnel::Ps5Desktop::LoadedImage& image,
    const std::span<const std::uint64_t> initializers) {
    if (image.entry == 0 || initializers.empty()) {
        return false;
    }
    constexpr std::uint64_t ProbeBytes = 0x100;
    for (const auto& range : image.registered_ranges) {
        if (image.entry < range.address ||
            image.entry - range.address >= range.byte_count) {
            continue;
        }
        const auto available = std::min<std::uint64_t>(
            ProbeBytes,
            range.byte_count - (image.entry - range.address));
        if (available < 5) {
            return false;
        }
        const auto* const bytes =
            reinterpret_cast<const std::uint8_t*>(image.entry);
        for (std::uint64_t offset = 0;
             offset + 5 <= available; ++offset) {
            if (bytes[offset] != UINT8_C(0xe8)) {
                continue;
            }
            std::int32_t displacement{};
            std::memcpy(
                &displacement, bytes + offset + 1,
                sizeof(displacement));
            const auto next = image.entry + offset + 5;
            const auto target = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(next) +
                displacement);
            if (std::ranges::contains(
                    initializers, target)) {
                return true;
            }
        }
        return false;
    }
    return false;
}

thread_local std::uint64_t g_current_guest_thread_handle{};
thread_local std::uint64_t g_current_guest_fs_base{};
std::mutex g_module_load_mutex;
std::mutex g_agc_submit_mutex;
std::atomic<std::uint64_t> g_translation_session_generation{1};
std::atomic<std::uint32_t> g_ps5_pad_buttons{};
std::array<std::atomic<std::uint8_t>, 6> g_ps5_pad_axes{
    std::uint8_t{128}, std::uint8_t{128},
    std::uint8_t{128}, std::uint8_t{128},
    std::uint8_t{0}, std::uint8_t{0}};
const auto g_ps5_process_start =
    std::chrono::steady_clock::now();
std::atomic<std::uint64_t> g_ps5_pad_timestamp{};
std::atomic<std::uint64_t> g_hle_binding_generation{1};
std::atomic<std::uint64_t> g_event_flag_generation{1};
std::array<struct sigaction, 3> g_ps5_previous_fault_actions{};
bool g_ps5_diagnostic_fault_handlers_installed{};

struct Ngs2SystemState {
    std::uint32_t grain_samples{256};
};

struct Ngs2RackState {
    std::uint64_t system_handle{};
    std::uint32_t rack_id{};
};

struct Ngs2VoiceState {
    std::uint64_t rack_handle{};
    std::uint32_t voice_index{};
    std::vector<std::int16_t> pcm;
    std::uint64_t source_address{};
    std::uint32_t source_rate{48000};
    double position{};
    bool playing{};
    std::int32_t loop_start{-1};
    std::int32_t loop_end{};
    float gain{1.0f};
};

std::mutex g_ngs2_mutex;
std::unordered_map<std::uint64_t, Ngs2SystemState> g_ngs2_systems;
std::unordered_map<std::uint64_t, Ngs2RackState> g_ngs2_racks;
std::unordered_map<std::uint64_t, Ngs2VoiceState> g_ngs2_voices;

constexpr std::array Ps5DiagnosticFaultSignals{
    SIGSEGV, SIGBUS, SIGILL};

bool Ps5DiagnosticFaultProbeEnabled() {
    if (std::getenv("EXECUTOR_DIAG_JIT_RIP_PROBE") != nullptr) {
        return true;
    }
#ifdef __ANDROID__
    return access(
               "/data/user/0/app.lsx4.android/files/lsx4-home/"
               "diag-jit-rip-probe",
               F_OK) == 0 ||
           access(
               "/data/data/app.lsx4.android/files/lsx4-home/"
               "diag-jit-rip-probe",
               F_OK) == 0;
#else
    return false;
#endif
}

void Ps5DiagnosticFaultHandler(
    const int signal, siginfo_t* const information,
    void* const raw_context) {
    std::array<std::uint64_t, 12> recent{};
    const auto recent_count =
        executor_jit_current_fault_recent_rips(
            recent.data(), recent.size());
    const auto guest_rip =
        executor_jit_current_fault_guest_rip();
    const auto active_rip =
        executor_jit_current_fault_active_block_rip();
    const auto instruction =
        executor_jit_current_fault_instruction();
    const auto instruction_meta =
        executor_jit_current_fault_instruction_meta();
    std::uint64_t host_pc{};
#if defined(__aarch64__)
    if (raw_context != nullptr) {
        host_pc = static_cast<const ucontext_t*>(
            raw_context)->uc_mcontext.pc;
    }
#endif
    char line[3072]{};
    int count = std::snprintf(
        line, sizeof(line),
        "PS5_RUNTIME_FAULT signal=%d code=%d fault=%p "
        "host_pc=0x%llx guest_rip=0x%llx active=0x%llx "
        "instruction=0x%llx meta=0x%x thread=0x%llx fs=0x%llx "
        "rax=0x%llx rcx=0x%llx rdx=0x%llx rbx=0x%llx "
        "rsp=0x%llx rbp=0x%llx rsi=0x%llx rdi=0x%llx "
        "r8=0x%llx r9=0x%llx r10=0x%llx r11=0x%llx "
        "r12=0x%llx r13=0x%llx r14=0x%llx r15=0x%llx recent=",
        signal, information != nullptr ? information->si_code : 0,
        information != nullptr ? information->si_addr : nullptr,
        static_cast<unsigned long long>(host_pc),
        static_cast<unsigned long long>(guest_rip),
        static_cast<unsigned long long>(active_rip),
        static_cast<unsigned long long>(instruction),
        instruction_meta,
        static_cast<unsigned long long>(
            g_current_guest_thread_handle),
        static_cast<unsigned long long>(
            g_current_guest_fs_base),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(0)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(1)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(2)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(3)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(4)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(5)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(6)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(7)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(8)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(9)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(10)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(11)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(12)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(13)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(14)),
        static_cast<unsigned long long>(
            executor_jit_current_fault_gpr(15)));
    for (std::uint32_t index = 0;
         index < recent_count &&
         count > 0 &&
         static_cast<std::size_t>(count) < sizeof(line);
         ++index) {
        count += std::snprintf(
            line + count, sizeof(line) - count,
            "%s0x%llx", index == 0 ? "" : ",",
            static_cast<unsigned long long>(recent[index]));
    }
    if (guest_rip >= UINT64_C(0xfe36) &&
        (instruction & UINT64_C(0xffffffff)) == UINT64_C(0x077ffac5) &&
        executor_jit_current_fault_gpr(7) == 0 &&
        count > 0 &&
        static_cast<std::size_t>(count) < sizeof(line)) {
        const auto libc_base = guest_rip - UINT64_C(0xfe36);
        const auto allocator_state =
            libc_base + UINT64_C(0x1142d8);
        constexpr std::array<std::uint64_t, 12> offsets{
            0x0, 0x8, 0x10, 0x18, 0x20, 0x28,
            0x38, 0x350, 0x358, 0x360, 0x368, 0x370};
        count += std::snprintf(
            line + count, sizeof(line) - count,
            " allocator=0x%llx",
            static_cast<unsigned long long>(allocator_state));
        for (const auto offset : offsets) {
            if (count <= 0 ||
                static_cast<std::size_t>(count) >= sizeof(line)) {
                break;
            }
            std::uint64_t value{};
            std::memcpy(
                &value,
                reinterpret_cast<const void*>(
                    allocator_state + offset),
                sizeof(value));
            count += std::snprintf(
                line + count, sizeof(line) - count,
                " +%llx=0x%llx",
                static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(value));
        }
    }
    if (count > 0 &&
        static_cast<std::size_t>(count) < sizeof(line) - 1) {
        line[count++] = '\n';
    }
    if (count > 0) {
        (void)write(
            STDERR_FILENO, line,
            std::min<std::size_t>(
                static_cast<std::size_t>(count), sizeof(line) - 1));
#ifdef __ANDROID__
        const int diagnostic_fd = open(
            "/data/user/0/app.lsx4.android/files/lsx4-home/"
            "ps5-runtime-fault.log",
            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (diagnostic_fd >= 0) {
            (void)write(
                diagnostic_fd, line,
                std::min<std::size_t>(
                    static_cast<std::size_t>(count),
                    sizeof(line) - 1));
            (void)close(diagnostic_fd);
        }
#endif
    }
    const auto found = std::ranges::find(
        Ps5DiagnosticFaultSignals, signal);
    if (found != Ps5DiagnosticFaultSignals.end()) {
        const auto index = static_cast<std::size_t>(
            found - Ps5DiagnosticFaultSignals.begin());
        const auto& previous =
            g_ps5_previous_fault_actions[index];
        if ((previous.sa_flags & SA_SIGINFO) != 0 &&
            previous.sa_sigaction != nullptr &&
            previous.sa_sigaction != &Ps5DiagnosticFaultHandler) {
            previous.sa_sigaction(signal, information, raw_context);
            return;
        }
        if (previous.sa_handler == SIG_IGN) {
            return;
        }
        if (previous.sa_handler != nullptr &&
            previous.sa_handler != SIG_DFL) {
            previous.sa_handler(signal);
            return;
        }
    }
    (void)::signal(signal, SIG_DFL);
    (void)::raise(signal);
}

void InstallPs5DiagnosticFaultHandlers() {
    if (!Ps5DiagnosticFaultProbeEnabled()) {
        return;
    }
    static std::mutex install_mutex;
    const std::lock_guard install_lock{install_mutex};
    struct sigaction action {};
    action.sa_sigaction = &Ps5DiagnosticFaultHandler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO;
    for (std::size_t index = 0;
         index < Ps5DiagnosticFaultSignals.size(); ++index) {
        struct sigaction current {};
        (void)sigaction(
            Ps5DiagnosticFaultSignals[index], nullptr, &current);
        if ((current.sa_flags & SA_SIGINFO) != 0 &&
            current.sa_sigaction == &Ps5DiagnosticFaultHandler) {
            continue;
        }
        g_ps5_previous_fault_actions[index] = current;
        (void)sigaction(
            Ps5DiagnosticFaultSignals[index], &action, nullptr);
    }
    g_ps5_diagnostic_fault_handlers_installed = true;
}

void RestorePs5DiagnosticFaultHandlers() {
    if (!g_ps5_diagnostic_fault_handlers_installed) {
        return;
    }
    for (std::size_t index = 0;
         index < Ps5DiagnosticFaultSignals.size(); ++index) {
        (void)sigaction(
            Ps5DiagnosticFaultSignals[index],
            &g_ps5_previous_fault_actions[index], nullptr);
    }
    g_ps5_diagnostic_fault_handlers_installed = false;
}

bool RangeIsRepresentable(const std::uint64_t address,
                          const std::uint64_t byte_count) noexcept {
    return address >= 0x10000 && byte_count != 0 &&
           byte_count - 1 <=
               std::numeric_limits<std::uint64_t>::max() - address;
}

bool RangeContains(const RegisteredMapping& mapping,
                   const std::uint64_t address,
                   const std::uint64_t byte_count) noexcept {
    return RangeIsRepresentable(address, byte_count) &&
           address >= mapping.address &&
           byte_count <= mapping.byte_count &&
           address - mapping.address <= mapping.byte_count - byte_count;
}

bool HasAccessLocked(const std::uint64_t address,
                     const std::uint64_t byte_count,
                     const std::uint32_t access) {
    return RangeIsRepresentable(address, byte_count) &&
           g_runtime.mappings.Find(address, byte_count, access) != nullptr;
}

struct GuestAccessCacheEntry {
    std::uint64_t generation{};
    std::uint64_t address{};
    std::uint64_t byte_count{};
    std::uint32_t protection{};
};

// Guest code routinely alternates between its stack, TLS and heap mappings.
// A single-entry cache turns that normal pattern into a global runtime mutex
// acquisition for almost every HLE copy.
thread_local std::array<GuestAccessCacheEntry, 32>
    g_guest_access_cache{};
thread_local std::size_t g_guest_access_cache_cursor{};

bool Ps5TraceHleEnabled() {
    static const bool enabled = [] {
        const char* const requested = std::getenv("PS5_TRACE_HLE");
        return requested != nullptr && requested[0] != '0';
    }();
    return enabled;
}

bool GuestAccessCacheEnabled() {
    static const bool enabled = [] {
        const char* const strict = std::getenv("PS5_MAPPING_STRICT");
        return strict == nullptr || strict[0] == '0';
    }();
    return enabled;
}

bool HasCachedAccess(const std::uint64_t address,
                     const std::uint64_t byte_count,
                     const std::uint32_t access) {
    if (!GuestAccessCacheEnabled() ||
        !RangeIsRepresentable(address, byte_count)) {
        return false;
    }
    const auto generation = g_runtime.mappings.Generation();
    return std::ranges::any_of(
        g_guest_access_cache,
        [&](const GuestAccessCacheEntry& cached) {
            return cached.generation != 0 &&
                   cached.generation == generation &&
                   (cached.protection & access) == access &&
                   address >= cached.address &&
                   byte_count <= cached.byte_count &&
                   address - cached.address <=
                       cached.byte_count - byte_count;
        });
}

bool CopyGuestBytes(const std::uint64_t address, void* const destination,
                    const void* const source, const std::size_t byte_count,
                    const std::uint32_t access) {
    if (HasCachedAccess(address, byte_count, access)) {
        std::memcpy(destination, source, byte_count);
        return true;
    }
    if (!RangeIsRepresentable(address, byte_count)) {
        return false;
    }
    const std::lock_guard lock{g_runtime.mutex};
    const auto generation = g_runtime.mappings.Generation();
    const auto* const mapping =
        g_runtime.mappings.Find(address, byte_count, access);
    if (mapping == nullptr) {
        return false;
    }
    g_guest_access_cache[
        g_guest_access_cache_cursor++ %
        g_guest_access_cache.size()] = {
        .generation = generation,
        .address = mapping->address,
        .byte_count = mapping->byte_count,
        .protection = mapping->protection,
    };
    std::memcpy(destination, source, byte_count);
    return true;
}

bool RegisterMappingLocked(const std::uint64_t address,
                           const std::uint64_t byte_count,
                           const std::uint32_t protection,
                           const char* label, const bool internal) {
    if (!RangeIsRepresentable(address, byte_count) ||
        address >= g_runtime.guest_address_limit ||
        byte_count > g_runtime.guest_address_limit - address) {
        return false;
    }
    if (!internal &&
        ((address & (LSX4_PS5_GUEST_PAGE_SIZE - 1u)) != 0 ||
         (byte_count & (LSX4_PS5_GUEST_PAGE_SIZE - 1u)) != 0)) {
        return false;
    }
    if ((protection & ~(LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE |
                        LSX4_PS5_GUEST_EXECUTE)) != 0 ||
        protection == 0) {
        return false;
    }

    (void)g_runtime.mappings.EraseAt(address, byte_count);
    g_runtime.mappings.Insert({
        .address = address,
        .byte_count = byte_count,
        .protection = protection,
        .label = label != nullptr ? label : "",
        .internal = internal,
    });
    return true;
}

std::size_t HostPageSize() {
    const long queried = sysconf(_SC_PAGESIZE);
    return queried > 0 ? static_cast<std::size_t>(queried) : 0x1000u;
}

std::uint8_t* AllocateLowGuestRegionLocked(
    const std::uint64_t first_hint, const std::uint64_t stride,
    const std::size_t byte_count,
    const std::uint64_t required_alignment =
        LSX4_PS5_GUEST_PAGE_SIZE,
    const std::uint64_t attempt_limit = 256) {
    if (required_alignment == 0 ||
        (required_alignment & (required_alignment - 1u)) != 0) {
        return nullptr;
    }
    for (std::uint64_t attempt = 0;
         attempt != attempt_limit; ++attempt) {
        const auto unaligned = first_hint + attempt * stride;
        if (unaligned < first_hint ||
            unaligned >
                std::numeric_limits<std::uint64_t>::max() -
                    (required_alignment - 1u)) {
            break;
        }
        const auto desired =
            (unaligned + required_alignment - 1u) &
            ~(required_alignment - 1u);
        if (desired >= g_runtime.guest_address_limit ||
            byte_count > g_runtime.guest_address_limit - desired) {
            break;
        }
        int mapping_flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        mapping_flags |= MAP_NORESERVE;
#endif
#ifdef MAP_FIXED_NOREPLACE
        mapping_flags |= MAP_FIXED_NOREPLACE;
#endif
        void* const mapping = mmap(
            reinterpret_cast<void*>(desired), byte_count,
            PROT_READ | PROT_WRITE, mapping_flags, -1, 0);
        if (mapping == MAP_FAILED) {
            continue;
        }
        const auto address = reinterpret_cast<std::uint64_t>(mapping);
        if ((address & (required_alignment - 1u)) == 0 &&
            address < g_runtime.guest_address_limit &&
            byte_count <= g_runtime.guest_address_limit - address) {
            return static_cast<std::uint8_t*>(mapping);
        }
        munmap(mapping, byte_count);
    }
    return nullptr;
}

bool EnsureHleDataLocked() {
    if (g_runtime.hle_data != nullptr) {
        return true;
    }
    auto* const allocation = AllocateLowGuestRegionLocked(
        Ps5HleDataHint, Ps5HleDataStride, Ps5HleDataSize);
    if (allocation == nullptr) {
        return false;
    }
    const auto address = reinterpret_cast<std::uint64_t>(allocation);
    if (!RegisterMappingLocked(
            address, Ps5HleDataSize,
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE,
            "ps5-hle-data", true)) {
        munmap(allocation, Ps5HleDataSize);
        return false;
    }

    std::fill_n(allocation, Ps5HleDataSize, std::uint8_t{});
    std::memcpy(
        allocation + Ps5StackGuardOffset, &Ps5StackCanary,
        sizeof(Ps5StackCanary));
    std::memcpy(
        allocation + Ps5StackGuardOffset + sizeof(Ps5StackCanary),
        &Ps5StackCanary, sizeof(Ps5StackCanary));
    constexpr char ProcessName[] = "eboot.bin";
    std::memcpy(
        allocation + Ps5ProgramNameBufferOffset, ProcessName,
        sizeof(ProcessName));
    const auto process_name_address =
        address + Ps5ProgramNameBufferOffset;
    std::memcpy(
        allocation + Ps5ProgramNamePointerOffset,
        &process_name_address, sizeof(process_name_address));
    constexpr std::uint32_t NeedFlag = 1;
    std::memcpy(
        allocation + Ps5LibcNeedFlagOffset, &NeedFlag,
        sizeof(NeedFlag));
    std::memcpy(
        allocation + Ps5LibcInternalNeedFlagOffset, &NeedFlag,
        sizeof(NeedFlag));
    g_runtime.hle_data = allocation;
    return true;
}

bool CreateThreadContextLocked(
    const std::uint64_t program_handle,
    Lsx4Ps5GuestThreadContext& output) {
    const auto program = std::ranges::find(
        g_runtime.programs, program_handle, &LoadedProgram::handle);
    if (program == g_runtime.programs.end()) {
        return false;
    }
    const auto root_handle =
        program->module ? program->owner_handle : program->handle;
    std::vector<const LoadedProgram*> tls_programs;
    std::uint64_t maximum_static_offset{};
    std::uint64_t maximum_module_id{};
    for (const auto& candidate : g_runtime.programs) {
        if ((candidate.handle != root_handle &&
             candidate.owner_handle != root_handle) ||
            candidate.image.tls_memory_size == 0) {
            continue;
        }
        tls_programs.push_back(&candidate);
        maximum_static_offset = std::max(
            maximum_static_offset, candidate.image.tls_static_offset);
        maximum_module_id = std::max(
            maximum_module_id, candidate.image.tls_module_id);
    }
    const auto dtv_size =
        2 * sizeof(std::uint64_t) +
        maximum_module_id * sizeof(std::uint64_t);
    if (maximum_static_offset > Ps5StaticTlsReservation ||
        dtv_size > Ps5TcbAndDtvSize - 0x100) {
        return false;
    }

    auto* const allocation = AllocateLowGuestRegionLocked(
        Ps5TlsRegionHint, Ps5TlsRegionStride, Ps5TlsRegionSize);
    if (allocation == nullptr) {
        return false;
    }
    const auto mapping_base =
        reinterpret_cast<std::uint64_t>(allocation);
    if (!RegisterMappingLocked(
            mapping_base, Ps5TlsRegionSize,
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE,
            "ps5-thread-tls", true)) {
        munmap(allocation, Ps5TlsRegionSize);
        return false;
    }

    std::memset(allocation, 0, Ps5TlsRegionSize);
    const auto fs_base = mapping_base + Ps5StaticTlsReservation;
    const auto pthread_handle = fs_base + 0x2000;
    for (const auto* const tls_program : tls_programs) {
        const auto& image = tls_program->image;
        const auto tls_address = fs_base - image.tls_static_offset;
        if (image.tls_file_size != 0) {
            std::memcpy(
                reinterpret_cast<void*>(tls_address),
                reinterpret_cast<const void*>(image.tls_image),
                static_cast<std::size_t>(image.tls_file_size));
        }
    }

    const auto dtv_address = fs_base + 0x100;
    const std::uint64_t generation =
        static_cast<std::uint64_t>(tls_programs.size());
    std::memcpy(reinterpret_cast<void*>(fs_base + 0x00),
                &fs_base, sizeof(fs_base));
    std::memcpy(reinterpret_cast<void*>(fs_base + 0x08),
                &dtv_address, sizeof(dtv_address));
    std::memcpy(reinterpret_cast<void*>(fs_base + 0x10),
                &fs_base, sizeof(fs_base));
    std::memcpy(reinterpret_cast<void*>(fs_base + 0x28),
                &Ps5StackCanary, sizeof(Ps5StackCanary));
    std::memcpy(reinterpret_cast<void*>(fs_base + 0x60),
                &fs_base, sizeof(fs_base));
    std::memcpy(reinterpret_cast<void*>(dtv_address + 0x00),
                &generation, sizeof(generation));
    std::memcpy(reinterpret_cast<void*>(dtv_address + 0x08),
                &maximum_module_id, sizeof(maximum_module_id));
    for (const auto* const tls_program : tls_programs) {
        const auto& image = tls_program->image;
        const auto module_address =
            fs_base - image.tls_static_offset;
        const auto slot = dtv_address + 0x10 +
                          (image.tls_module_id - 1) *
                              sizeof(std::uint64_t);
        std::memcpy(reinterpret_cast<void*>(slot),
                    &module_address, sizeof(module_address));
    }

    auto thread_handle = g_runtime.next_thread_handle++;
    if (thread_handle == 0) {
        thread_handle = g_runtime.next_thread_handle++;
    }
    g_runtime.threads.push_back({
        .handle = thread_handle,
        .pthread_handle = pthread_handle,
        .program_handle = root_handle,
        .allocation = allocation,
        .mapping_size = Ps5TlsRegionSize,
        .fs_base = fs_base,
        .tls_static_offset = maximum_static_offset,
    });
    output = {
        .size = sizeof(output),
        .handle = thread_handle,
        .program_handle = root_handle,
        .fs_base = fs_base,
        .mapping_base = mapping_base,
        .mapping_size = Ps5TlsRegionSize,
        .tls_static_offset = maximum_static_offset,
    };
    return true;
}

bool InstallLateTlsModuleLocked(
    const std::uint64_t owner_handle,
    const Funnel::Ps5Desktop::LoadedImage& image,
    std::string& error) {
    if (image.tls_memory_size == 0) {
        return true;
    }
    const auto dtv_size =
        2 * sizeof(std::uint64_t) +
        image.tls_module_id * sizeof(std::uint64_t);
    if (image.tls_static_offset > Ps5StaticTlsReservation ||
        image.tls_memory_size > image.tls_static_offset ||
        dtv_size > Ps5TcbAndDtvSize - 0x100) {
        error = "late PS5 module TLS exceeds the reserved TCB/DTV layout";
        return false;
    }
    for (auto& thread : g_runtime.threads) {
        if (thread.program_handle != owner_handle) {
            continue;
        }
        const auto module_address =
            thread.fs_base - image.tls_static_offset;
        std::memset(
            reinterpret_cast<void*>(module_address), 0,
            static_cast<std::size_t>(image.tls_memory_size));
        if (image.tls_file_size != 0) {
            std::memcpy(
                reinterpret_cast<void*>(module_address),
                reinterpret_cast<const void*>(image.tls_image),
                static_cast<std::size_t>(image.tls_file_size));
        }
        const auto dtv_address = thread.fs_base + 0x100;
        std::uint64_t generation{};
        std::uint64_t maximum_module_id{};
        std::memcpy(
            &generation,
            reinterpret_cast<const void*>(dtv_address + 0x00),
            sizeof(generation));
        std::memcpy(
            &maximum_module_id,
            reinterpret_cast<const void*>(dtv_address + 0x08),
            sizeof(maximum_module_id));
        ++generation;
        maximum_module_id =
            std::max(maximum_module_id, image.tls_module_id);
        const auto slot =
            dtv_address + 0x10 +
            (image.tls_module_id - 1) * sizeof(std::uint64_t);
        std::memcpy(
            reinterpret_cast<void*>(dtv_address + 0x00),
            &generation, sizeof(generation));
        std::memcpy(
            reinterpret_cast<void*>(dtv_address + 0x08),
            &maximum_module_id, sizeof(maximum_module_id));
        std::memcpy(
            reinterpret_cast<void*>(slot),
            &module_address, sizeof(module_address));
        thread.tls_static_offset =
            std::max(thread.tls_static_offset, image.tls_static_offset);
    }
    return true;
}

void RollbackLateTlsModuleLocked(
    const std::uint64_t owner_handle,
    const std::uint64_t module_handle,
    const Funnel::Ps5Desktop::LoadedImage& image) {
    if (image.tls_memory_size == 0) {
        return;
    }
    std::uint64_t maximum_module_id{};
    std::uint64_t maximum_static_offset{};
    for (const auto& program : g_runtime.programs) {
        if (program.handle == module_handle ||
            (program.handle != owner_handle &&
             program.owner_handle != owner_handle) ||
            program.image.tls_memory_size == 0) {
            continue;
        }
        maximum_module_id =
            std::max(maximum_module_id, program.image.tls_module_id);
        maximum_static_offset =
            std::max(maximum_static_offset,
                     program.image.tls_static_offset);
    }
    for (auto& thread : g_runtime.threads) {
        if (thread.program_handle != owner_handle) {
            continue;
        }
        const auto module_address =
            thread.fs_base - image.tls_static_offset;
        std::memset(
            reinterpret_cast<void*>(module_address), 0,
            static_cast<std::size_t>(image.tls_memory_size));
        const auto dtv_address = thread.fs_base + 0x100;
        std::uint64_t generation{};
        std::memcpy(
            &generation,
            reinterpret_cast<const void*>(dtv_address),
            sizeof(generation));
        ++generation;
        const std::uint64_t empty_slot{};
        const auto slot =
            dtv_address + 0x10 +
            (image.tls_module_id - 1) * sizeof(std::uint64_t);
        std::memcpy(
            reinterpret_cast<void*>(dtv_address),
            &generation, sizeof(generation));
        std::memcpy(
            reinterpret_cast<void*>(dtv_address + 0x08),
            &maximum_module_id, sizeof(maximum_module_id));
        std::memcpy(
            reinterpret_cast<void*>(slot),
            &empty_slot, sizeof(empty_slot));
        thread.tls_static_offset = maximum_static_offset;
    }
}

bool RemoveThreadContextLocked(
    const std::uint64_t thread_handle, GuestThreadContext& removed) {
    const auto found = std::ranges::find(
        g_runtime.threads, thread_handle, &GuestThreadContext::handle);
    if (found == g_runtime.threads.end()) {
        return false;
    }
    removed = *found;
    (void)g_runtime.mappings.EraseExact(
        reinterpret_cast<std::uint64_t>(removed.allocation),
        removed.mapping_size, true);
    g_runtime.threads.erase(found);
    return true;
}

void EmitPs5HleStub(std::uint8_t* const code,
                    const std::uint64_t thunk) {
    std::fill_n(code, Ps5HleStubSize, std::uint8_t{0xcc});
    std::size_t cursor{};
    const auto emit = [&](const std::initializer_list<std::uint8_t> bytes) {
        std::ranges::copy(bytes, code + cursor);
        cursor += bytes.size();
    };
    const auto emit_u64 = [&](const std::uint64_t value) {
        std::memcpy(code + cursor, &value, sizeof(value));
        cursor += sizeof(value);
    };

    emit({0x48, 0x81, 0xec, 0x88, 0x00, 0x00, 0x00});
    emit({0x48, 0xb8});
    emit_u64(thunk);
    emit({0x48, 0x89, 0x44, 0x24, 0x08});
    emit({0x48, 0x89, 0x7c, 0x24, 0x10});
    emit({0x48, 0x89, 0x74, 0x24, 0x18});
    emit({0x48, 0x89, 0x54, 0x24, 0x20});
    emit({0x48, 0x89, 0x4c, 0x24, 0x28});
    emit({0x4c, 0x89, 0x44, 0x24, 0x30});
    emit({0x4c, 0x89, 0x4c, 0x24, 0x38});
    emit({0x48, 0x8d, 0x84, 0x24, 0x88, 0x00, 0x00, 0x00});
    emit({0x48, 0x89, 0x44, 0x24, 0x40});
    emit({0x4c, 0x89, 0x5c, 0x24, 0x48});
    emit({0x48, 0x89, 0x5c, 0x24, 0x50});
    emit({0x48, 0x89, 0x6c, 0x24, 0x58});
    emit({0x4c, 0x89, 0x64, 0x24, 0x60});
    emit({0x4c, 0x89, 0x6c, 0x24, 0x68});
    emit({0x4c, 0x89, 0x74, 0x24, 0x70});
    emit({0x4c, 0x89, 0x7c, 0x24, 0x78});
    emit({0x48, 0x8b, 0x00});
    emit({0x48, 0x89, 0x84, 0x24, 0x80, 0x00, 0x00, 0x00});
    emit({0x48, 0x89, 0xe7});
    emit({0x0f, 0x3f});
    std::ranges::copy(
        Lsx4::Translation::TranslationThunkIdentity, code + cursor);
    cursor += Lsx4::Translation::TranslationThunkIdentity.size();
    (void)cursor;
}

std::uint64_t EnsureHleSlabLocked() {
    if (g_runtime.hle_slab != nullptr) {
        return reinterpret_cast<std::uint64_t>(g_runtime.hle_slab);
    }
    void* const allocation = mmap(
        reinterpret_cast<void*>(
            Ps5HleSlabHint +
            (g_runtime.hle_slab_slot % Ps5HleSlabSlots) *
                Ps5HleSlabStride),
        Ps5HleSlabSize,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (allocation == MAP_FAILED) {
        return 0;
    }
    const auto address = reinterpret_cast<std::uint64_t>(allocation);
    if ((address & (LSX4_PS5_GUEST_PAGE_SIZE - 1u)) != 0 ||
        address >= g_runtime.guest_address_limit ||
        Ps5HleSlabSize > g_runtime.guest_address_limit - address ||
        !RegisterMappingLocked(
            address, Ps5HleSlabSize,
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_EXECUTE,
            "ps5-hle-thunks", true)) {
        munmap(allocation, Ps5HleSlabSize);
        return 0;
    }
    g_runtime.hle_slab = static_cast<std::uint8_t*>(allocation);
    g_runtime.hle_slab_used = 0;
    return address;
}

std::uint64_t CreateHleStubLocked(
    const std::uint64_t owner_handle, const char* const symbol,
    const std::uint64_t native_function) {
    const std::string_view requested =
        symbol != nullptr ? std::string_view{symbol} : std::string_view{};
    const auto existing = std::ranges::find_if(
        g_runtime.hle_bindings, [&](const auto& pair) {
            return pair.second.owner_handle == owner_handle &&
                   pair.second.symbol == requested;
        });
    if (existing != g_runtime.hle_bindings.end()) {
        if (native_function != 0) {
            existing->second.native_function = native_function;
        }
        return existing->first;
    }
    if (EnsureHleSlabLocked() == 0 ||
        g_runtime.hle_slab_used >
            Ps5HleSlabSize - Ps5HleStubSize) {
        return 0;
    }

    auto* const code =
        g_runtime.hle_slab + g_runtime.hle_slab_used;
    const auto thunk = reinterpret_cast<std::uint64_t>(code);
    const auto host_page_size = HostPageSize();
    const auto page = thunk & ~(static_cast<std::uint64_t>(
                                    host_page_size) - 1u);
    if (mprotect(reinterpret_cast<void*>(page), host_page_size,
                 PROT_READ | PROT_WRITE) != 0) {
        return 0;
    }
    EmitPs5HleStub(code, thunk);
    __builtin___clear_cache(
        reinterpret_cast<char*>(code),
        reinterpret_cast<char*>(code + Ps5HleStubSize));
    if (mprotect(reinterpret_cast<void*>(page), host_page_size,
                 PROT_READ | PROT_EXEC) != 0) {
        return 0;
    }
#ifdef __ANDROID__
    if (g_runtime.hle_slab_used >= 0x7200 &&
        g_runtime.hle_slab_used <= 0x7600) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-BIND",
            "offset=0x%zx thunk=0x%llx symbol=%s native=0x%llx",
            g_runtime.hle_slab_used,
            static_cast<unsigned long long>(thunk),
            requested.empty() ? "(empty)" : requested.data(),
            static_cast<unsigned long long>(native_function));
    }
#endif
    g_runtime.hle_slab_used += Ps5HleStubSize;
    g_runtime.hle_bindings.emplace(
        thunk, HleBinding{
                   owner_handle, native_function, std::string{requested}});
    return thunk;
}

int LoaderRegisterMapping(void*, const std::uint64_t address,
                          const std::uint64_t byte_count,
                          const std::uint32_t protection,
                          const char* const label) {
    const std::lock_guard lock{g_runtime.mutex};
    return g_runtime.initialized &&
                   RegisterMappingLocked(
                       address, byte_count, protection, label, false)
        ? 0
        : -1;
}

int LoaderUnregisterMapping(void*, const std::uint64_t address,
                            const std::uint64_t byte_count) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto removed =
        g_runtime.mappings.EraseExact(address, byte_count, false);
    return removed != 0 ? 0 : -1;
}

bool PreferPs5BuiltinHle(const std::string_view symbol) {
    constexpr std::array<std::string_view, 19> symbols{
        "gQX+4GDQjpM",
        "tIhsqj0qsFE",
        "2X5agFjKxMc",
        "Y7aJ1uydPMo",
        "Ujf3KzMvRmI",
        "2Btkg8k24Zg",
        "cVSk9y8URbc",
        "xeYO4u7uyJ0",
        "rQFVBXp-Cxg",
        "Qazy8LmXTvw",
        "uodLYyUip20",
        "lbB+UlZqVG0",
        "bzQExy189ZI",
        "8G2LB+A3rzg",
        "tsvEmnenz48",
        "H2e8t5ScQGc",
        "kbw4UHHSYy0",
        "uMei1W9uyNo",
        "XKRegsFpEpk",
    };
    return std::ranges::contains(symbols, symbol);
}

int LoaderBindImport(void* const opaque_context,
                     const char* const symbol,
                     const std::uint32_t flags,
                     std::uint64_t* const value,
                     std::uint64_t* const symbol_size) {
    if (value == nullptr || symbol_size == nullptr) {
        return Funnel::Ps5Desktop::LoaderBindError;
    }
    *value = 0;
    *symbol_size = 0;
    const auto* const context =
        static_cast<const LoaderBindingContext*>(opaque_context);
    if (context != nullptr) {
        const std::string_view requested =
            symbol != nullptr ? std::string_view{symbol} :
                                std::string_view{};
        if (!PreferPs5BuiltinHle(requested)) {
            const auto exported = std::ranges::find_if(
                context->exports,
                [&](const Funnel::Ps5Desktop::LoadedSymbol& candidate) {
                    return candidate.name == requested;
                });
            if (exported != context->exports.end()) {
                *value = exported->address;
                *symbol_size = exported->size;
                if (Ps5DiagnosticFaultProbeEnabled()) {
                    std::fprintf(
                        stderr,
                        "PS5_IMPORT_RESOLVED name=%s address=0x%llx "
                        "size=0x%llx flags=0x%x\n",
                        exported->name.c_str(),
                        static_cast<unsigned long long>(
                            exported->address),
                        static_cast<unsigned long long>(exported->size),
                        flags);
                }
                return Funnel::Ps5Desktop::LoaderBindResolved;
            }
        }
        if (context->rebind_only) {
            return Funnel::Ps5Desktop::LoaderBindDeferred;
        }
    }

    Lsx4Ps5RuntimeCallbacks callbacks{};
    std::uint64_t thunk{};
    const auto function_import =
        (flags & Funnel::Ps5Desktop::LoaderImportFunction) != 0;
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            return Funnel::Ps5Desktop::LoaderBindError;
        }
        callbacks = g_runtime.callbacks;
        if (function_import) {
            thunk = CreateHleStubLocked(
                context != nullptr ? context->owner_handle : 0,
                symbol, 0);
        }
    }
    if (function_import && thunk == 0) {
        return Funnel::Ps5Desktop::LoaderBindError;
    }

    std::uint64_t native_function{};
    if (callbacks.bind_import != nullptr &&
        callbacks.bind_import(
            callbacks.context, symbol, thunk, &native_function) != 0) {
        return Funnel::Ps5Desktop::LoaderBindError;
    }
    if (native_function != 0) {
        if (!function_import) {
            *value = native_function;
            return Funnel::Ps5Desktop::LoaderBindResolved;
        }
        {
            const std::lock_guard lock{g_runtime.mutex};
            if (const auto found = g_runtime.hle_bindings.find(thunk);
                found != g_runtime.hle_bindings.end()) {
                found->second.native_function = native_function;
            }
        }
        *value = thunk;
        return Funnel::Ps5Desktop::LoaderBindResolved;
    }
    if (function_import) {
        *value = thunk;
    }
    return Funnel::Ps5Desktop::LoaderBindDeferred;
}

Funnel::Ps5Desktop::LoaderHost Ps5LoaderHost(
    LoaderBindingContext* const context = nullptr) {
    return {
        .context = context,
        .register_mapping = &LoaderRegisterMapping,
        .unregister_mapping = &LoaderUnregisterMapping,
        .bind_import = &LoaderBindImport,
    };
}

bool ProgramBelongsToOwner(const LoadedProgram& program,
                           const std::uint64_t owner_handle) {
    return program.handle == owner_handle ||
           program.owner_handle == owner_handle;
}

std::string LowerAscii(std::string value) {
    std::ranges::transform(
        value, value.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

std::string ProgramFileName(const LoadedProgram& program) {
    return LowerAscii(
        std::filesystem::path{program.path}.filename().string());
}

bool ModuleStartsAtBoot(const std::filesystem::path& path) {
    const auto file_name = LowerAscii(path.filename().string());
    if (file_name == "libfmod.prx" ||
        file_name == "libfmodstudio.prx") {
        return true;
    }
    const auto parent = LowerAscii(path.parent_path().filename().string());
    return parent != "plugins";
}

LoaderBindingContext SnapshotLoaderContext(
    const std::uint64_t owner_handle, const bool rebind_only) {
    LoaderBindingContext context{
        .owner_handle = owner_handle,
        .rebind_only = rebind_only,
    };
    const std::lock_guard lock{g_runtime.mutex};
    if (g_runtime.hle_data != nullptr) {
        const auto base =
            reinterpret_cast<std::uint64_t>(g_runtime.hle_data);
        context.exports.push_back({
            "f7uOxY9mM1U", base + Ps5StackGuardOffset,
            2 * sizeof(Ps5StackCanary)});
        context.exports.push_back({
            "djxxOmW6-aw", base + Ps5ProgramNamePointerOffset,
            sizeof(std::uint64_t)});
        context.exports.push_back({
            "P330P3dFF68", base + Ps5LibcNeedFlagOffset,
            sizeof(std::uint32_t)});
        context.exports.push_back({
            "ZT4ODD2Ts9o", base + Ps5LibcInternalNeedFlagOffset,
            sizeof(std::uint32_t)});
    }
    for (const auto& program : g_runtime.programs) {
        if (!ProgramBelongsToOwner(program, owner_handle)) {
            continue;
        }
        for (const auto& symbol : program.image.exported_symbols) {
            if (!std::ranges::any_of(
                    context.exports,
                    [&](const Funnel::Ps5Desktop::LoadedSymbol& existing) {
                        return existing.name == symbol.name;
                    })) {
                context.exports.push_back(symbol);
            }
        }
    }
    return context;
}

bool RebindProgramFamily(const std::uint64_t owner_handle,
                         std::string& error) {
    auto context = SnapshotLoaderContext(owner_handle, true);
    const auto host = Ps5LoaderHost(&context);
    const std::lock_guard lock{g_runtime.mutex};
    for (auto& program : g_runtime.programs) {
        if (ProgramBelongsToOwner(program, owner_handle) &&
            !Funnel::Ps5Desktop::RebindNextGenImports(
                host, program.image, error)) {
            return false;
        }
    }
    return true;
}

bool FillGameProbeLocked(const std::uint64_t owner_handle,
                         Lsx4Ps5GameProbeReport& report,
                         std::string& text) {
    const auto owner = std::ranges::find(
        g_runtime.programs, owner_handle, &LoadedProgram::handle);
    if (owner == g_runtime.programs.end() || owner->module) {
        return false;
    }

    report = {};
    report.size = sizeof(report);
    report.owner_handle = owner_handle;
    report.flags = LSX4_PS5_GAME_EBOOT_MAPPED;
    report.discovered_modules = owner->discovered_modules;
    report.failed_modules = owner->failed_modules;
    if (owner->module_scan_complete) {
        report.flags |= LSX4_PS5_GAME_MODULES_SCANNED;
    }

    std::vector<std::string> loaded_names;
    std::vector<std::string> deferred_functions;
    std::vector<std::string> unresolved_data;
    std::vector<std::string> missing_dependencies;
    for (const auto& program : g_runtime.programs) {
        if (!ProgramBelongsToOwner(program, owner_handle)) {
            continue;
        }
        loaded_names.push_back(ProgramFileName(program));
        if (program.module) {
            ++report.loaded_modules;
        }
        report.exported_symbols +=
            static_cast<std::uint32_t>(
                program.image.exported_symbols.size());
        report.dependencies +=
            static_cast<std::uint32_t>(
                program.image.dependencies.size());
        for (const auto& import : program.image.deferred_imports) {
            auto& names =
                (import.flags &
                 (Funnel::Ps5Desktop::LoaderImportData |
                  Funnel::Ps5Desktop::LoaderImportTls)) != 0
                ? unresolved_data
                : deferred_functions;
            if (!std::ranges::contains(names, import.name)) {
                names.push_back(import.name);
            }
        }
    }
    for (const auto& program : g_runtime.programs) {
        if (!ProgramBelongsToOwner(program, owner_handle)) {
            continue;
        }
        for (const auto& dependency : program.image.dependencies) {
            const auto normalized = LowerAscii(
                std::filesystem::path{dependency}.filename().string());
            if (!normalized.empty() &&
                !std::ranges::contains(loaded_names, normalized) &&
                !std::ranges::contains(
                    missing_dependencies, normalized)) {
                missing_dependencies.push_back(normalized);
            }
        }
    }
    report.deferred_function_imports =
        static_cast<std::uint32_t>(deferred_functions.size());
    report.unresolved_data_imports =
        static_cast<std::uint32_t>(unresolved_data.size());
    report.missing_dependencies =
        static_cast<std::uint32_t>(missing_dependencies.size());
    if (unresolved_data.empty()) {
        report.flags |= LSX4_PS5_GAME_IMPORTS_REBOUND;
    }
    const auto has_hle_dispatch =
        g_runtime.callbacks.invoke_hle != nullptr ||
        g_runtime.callbacks.bind_import != nullptr ||
        g_runtime.callbacks.resolve_hle != nullptr;
    const auto imports_can_defer =
#ifdef __ANDROID__
        true;
#else
        has_hle_dispatch;
#endif
    if (owner->module_scan_complete && owner->failed_modules == 0 &&
        (unresolved_data.empty() || imports_can_defer) &&
        (deferred_functions.empty() || imports_can_defer) &&
        (missing_dependencies.empty() || imports_can_defer)) {
        report.flags |= LSX4_PS5_GAME_READY_TO_LAUNCH;
    }

    std::ostringstream stream;
    stream << "PS5 game probe: owner=" << owner_handle
           << " modules=" << report.loaded_modules << "/"
           << report.discovered_modules
           << " failed=" << report.failed_modules
           << " exports=" << report.exported_symbols
           << " dependencies=" << report.dependencies
           << " deferred_functions="
           << report.deferred_function_imports
           << " unresolved_data=" << report.unresolved_data_imports
           << " missing_dependencies="
           << report.missing_dependencies
           << " ready="
           << ((report.flags & LSX4_PS5_GAME_READY_TO_LAUNCH) != 0
                   ? "yes"
                   : "no");
    const auto append_names =
        [&](const char* const label,
            const std::vector<std::string>& names) {
            if (names.empty()) {
                return;
            }
            stream << "\n" << label << ": ";
            for (std::size_t index = 0; index < names.size(); ++index) {
                if (index != 0) {
                    stream << ", ";
                }
                stream << names[index];
            }
        };
    append_names("deferred HLE/function imports", deferred_functions);
    append_names("unresolved data/TLS imports", unresolved_data);
    append_names("dependencies without mapped PRX", missing_dependencies);
    text = stream.str();
    return true;
}

std::vector<std::uint64_t> StartupModuleOrderLocked(
    const std::uint64_t owner_handle) {
    std::vector<const LoadedProgram*> modules;
    for (const auto& program : g_runtime.programs) {
        if (program.module &&
            program.owner_handle == owner_handle) {
            modules.push_back(&program);
        }
    }
    std::vector<std::uint64_t> visiting;
    std::vector<std::uint64_t> ordered;
    const auto visit =
        [&](const auto& self, const LoadedProgram& module) -> void {
            if (std::ranges::contains(ordered, module.handle) ||
                std::ranges::contains(visiting, module.handle)) {
                return;
            }
            visiting.push_back(module.handle);
            for (const auto& dependency : module.image.dependencies) {
                const auto normalized = LowerAscii(
                    std::filesystem::path{dependency}.filename().string());
                const auto found = std::ranges::find_if(
                    modules, [&](const LoadedProgram* candidate) {
                        return ProgramFileName(*candidate) == normalized;
                    });
                if (found != modules.end()) {
                    self(self, **found);
                }
            }
            std::erase(visiting, module.handle);
            ordered.push_back(module.handle);
        };
    for (const auto* const module : modules) {
        if (module->start_at_boot) {
            visit(visit, *module);
        }
    }
    return ordered;
}

void DumpProgramFamilyForDiagnostics(
    const std::uint64_t owner_handle) {
    if (std::getenv("PS5_PROBE_DUMP_MAPPED") == nullptr) {
        return;
    }
    struct Dump {
        std::string name;
        std::uint64_t base{};
        std::uint64_t byte_count{};
    };
    std::vector<Dump> dumps;
    {
        const std::lock_guard lock{g_runtime.mutex};
        for (const auto& program : g_runtime.programs) {
            if (!ProgramBelongsToOwner(program, owner_handle)) {
                continue;
            }
            dumps.push_back({
                .name = std::filesystem::path{
                    program.path}.filename().string(),
                .base = program.image.base,
                .byte_count = program.image.mapped_size,
            });
        }
    }
    for (const auto& dump : dumps) {
        const auto path =
            "/data/local/tmp/lsx4-ps5-release/mapped-" +
            dump.name + ".bin";
        if (FILE* const output = std::fopen(path.c_str(), "wb");
            output != nullptr) {
            const auto written = std::fwrite(
                reinterpret_cast<const void*>(dump.base), 1,
                static_cast<std::size_t>(dump.byte_count), output);
            std::fclose(output);
            std::fprintf(
                stderr,
                "PS5_RUNTIME_DUMP path=%s base=0x%llx bytes=0x%zx\n",
                path.c_str(),
                static_cast<unsigned long long>(dump.base), written);
        }
    }
}

Lsx4Ps5RuntimeCallbacks SnapshotCallbacks() {
    const std::lock_guard lock{g_runtime.mutex};
    return g_runtime.callbacks;
}

bool IsPowerOfTwo(const std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool AlignUp(const std::uint64_t value, const std::uint64_t alignment,
             std::uint64_t& output) noexcept {
    if (!IsPowerOfTwo(alignment) ||
        value > std::numeric_limits<std::uint64_t>::max() -
                    (alignment - 1)) {
        return false;
    }
    output = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool GuestWriteU64Locked(const std::uint64_t address,
                         const std::uint64_t value) {
    if (!HasAccessLocked(
            address, sizeof(value), LSX4_PS5_GUEST_WRITE)) {
        return false;
    }
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    return true;
}

bool GuestReadU64Locked(const std::uint64_t address,
                        std::uint64_t& value) {
    if (!HasAccessLocked(
            address, sizeof(value), LSX4_PS5_GUEST_READ)) {
        return false;
    }
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return true;
}

bool TryAllocateLibcHeap(
    std::uint64_t requested_size,
    std::uint64_t requested_alignment,
    std::uint64_t& address,
    std::uint64_t guest_stack = 0);
void FreeLibcHeap(
    std::uint64_t address,
    std::uint64_t guest_stack = 0);
std::uint64_t OrbisError(std::uint32_t value) noexcept;

bool TryWriteGuestBytes(
    const std::uint64_t address, const void* const source,
    const std::size_t byte_count) {
    if (source == nullptr || byte_count == 0) {
        return byte_count == 0;
    }
    return CopyGuestBytes(
        address, reinterpret_cast<void*>(address), source, byte_count,
        LSX4_PS5_GUEST_WRITE);
}

bool TryReadGuestBytes(
    const std::uint64_t address, void* const destination,
    const std::size_t byte_count) {
    if (destination == nullptr || byte_count == 0) {
        return byte_count == 0;
    }
    return CopyGuestBytes(
        address, destination, reinterpret_cast<const void*>(address),
        byte_count, LSX4_PS5_GUEST_READ);
}

#ifdef __ANDROID__
void LogDreamingForeachState(
    const char* const phase, const std::uint64_t hle_result = 0) {
    std::uint64_t executable_base{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto executable = std::ranges::find(
            g_runtime.programs, false, &LoadedProgram::module);
        if (executable != g_runtime.programs.end()) {
            executable_base = executable->image.base;
        }
    }
    std::int32_t depth{};
    std::uint64_t stack_begin{};
    std::uint64_t stack_end{};
    std::uint64_t stack_capacity{};
    const auto readable =
        executable_base != 0 &&
        TryReadGuestBytes(
            executable_base + UINT64_C(0x4a0058),
            &depth, sizeof(depth)) &&
        TryReadGuestBytes(
            executable_base + UINT64_C(0x4bf9f8),
            &stack_begin, sizeof(stack_begin)) &&
        TryReadGuestBytes(
            executable_base + UINT64_C(0x4bfa00),
            &stack_end, sizeof(stack_end)) &&
        TryReadGuestBytes(
            executable_base + UINT64_C(0x4bfa08),
            &stack_capacity, sizeof(stack_capacity));
    __android_log_print(
        readable ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
        "LSX4-PS5-FOREACH",
        "phase=%s hle_result=0x%llx base=0x%llx "
        "depth=%d begin=0x%llx end=0x%llx capacity=0x%llx",
        phase != nullptr ? phase : "unknown",
        static_cast<unsigned long long>(hle_result),
        static_cast<unsigned long long>(executable_base),
        depth,
        static_cast<unsigned long long>(stack_begin),
        static_cast<unsigned long long>(stack_end),
        static_cast<unsigned long long>(stack_capacity));
}

void WatchDreamingForeachState(
    const std::string_view symbol, const std::uint64_t hle_result,
    const std::uint64_t guest_stack) {
    std::uint64_t executable_base{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto executable = std::ranges::find(
            g_runtime.programs, false, &LoadedProgram::module);
        if (executable != g_runtime.programs.end()) {
            executable_base = executable->image.base;
        }
    }
    std::int32_t depth{};
    std::uint64_t stack_begin{};
    std::uint64_t stack_end{};
    std::uint64_t stack_capacity{};
    if (executable_base == 0 ||
        !TryReadGuestBytes(
            executable_base + UINT64_C(0x4a0058),
            &depth, sizeof(depth)) ||
        !TryReadGuestBytes(
            executable_base + UINT64_C(0x4bf9f8),
            &stack_begin, sizeof(stack_begin)) ||
        !TryReadGuestBytes(
            executable_base + UINT64_C(0x4bfa00),
            &stack_end, sizeof(stack_end)) ||
        !TryReadGuestBytes(
            executable_base + UINT64_C(0x4bfa08),
            &stack_capacity, sizeof(stack_capacity))) {
        return;
    }
    static std::mutex watch_mutex;
    static std::uint64_t previous_begin =
        std::numeric_limits<std::uint64_t>::max();
    static std::uint64_t previous_end =
        std::numeric_limits<std::uint64_t>::max();
    static std::uint64_t previous_capacity =
        std::numeric_limits<std::uint64_t>::max();
    static std::int32_t previous_depth =
        std::numeric_limits<std::int32_t>::min();
    const std::lock_guard watch_lock{watch_mutex};
    if (stack_begin == previous_begin &&
        stack_end == previous_end &&
        stack_capacity == previous_capacity &&
        depth == previous_depth) {
        return;
    }
    std::array<std::uint64_t, 4> guest_stack_words{};
    (void)TryReadGuestBytes(
        guest_stack, guest_stack_words.data(),
        sizeof(guest_stack_words));
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-PS5-FOREACH",
        "transition after=%.*s result=0x%llx "
        "returns=0x%llx,0x%llx,0x%llx,0x%llx "
        "depth=%d begin=0x%llx end=0x%llx capacity=0x%llx",
        static_cast<int>(symbol.size()), symbol.data(),
        static_cast<unsigned long long>(hle_result),
        static_cast<unsigned long long>(guest_stack_words[0]),
        static_cast<unsigned long long>(guest_stack_words[1]),
        static_cast<unsigned long long>(guest_stack_words[2]),
        static_cast<unsigned long long>(guest_stack_words[3]),
        depth,
        static_cast<unsigned long long>(stack_begin),
        static_cast<unsigned long long>(stack_end),
        static_cast<unsigned long long>(stack_capacity));
    previous_begin = stack_begin;
    previous_end = stack_end;
    previous_capacity = stack_capacity;
    previous_depth = depth;
}
#endif

bool TryReadGuestCString(
    const std::uint64_t address, const std::size_t maximum_size,
    std::string& value) {
    value.clear();
    if (address == 0 || maximum_size == 0) {
        return false;
    }
    const std::lock_guard lock{g_runtime.mutex};
    auto current = address;
    auto remaining = maximum_size;
    while (remaining != 0) {
        const auto* const mapping = g_runtime.mappings.Find(
            current, 1, LSX4_PS5_GUEST_READ);
        if (mapping == nullptr ||
            current < mapping->address) {
            value.clear();
            return false;
        }
        const auto available = mapping->byte_count -
            (current - mapping->address);
        const auto chunk = std::min<std::uint64_t>(
            available, remaining);
        const auto* const bytes =
            reinterpret_cast<const char*>(current);
        const auto* const terminator = static_cast<const char*>(
            std::memchr(bytes, '\0',
                       static_cast<std::size_t>(chunk)));
        if (terminator != nullptr) {
            value.append(
                bytes, static_cast<std::size_t>(
                           terminator - bytes));
            return true;
        }
        value.append(bytes, static_cast<std::size_t>(chunk));
        if (chunk == 0 ||
            current >
                std::numeric_limits<std::uint64_t>::max() -
                    chunk) {
            value.clear();
            return false;
        }
        current += chunk;
        remaining -= static_cast<std::size_t>(chunk);
    }
    value.clear();
    return false;
}

bool TryResolveGuestPath(
    const std::string_view guest_path,
    std::filesystem::path& host_path) {
    std::filesystem::path root;
    std::string_view relative;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto cached =
            g_runtime.resolved_guest_paths.find(
                std::string{guest_path});
        if (cached !=
            g_runtime.resolved_guest_paths.end()) {
            host_path = cached->second;
            return true;
        }
        if (guest_path == "/app0") {
            root = g_runtime.app0_directory;
        } else if (guest_path.starts_with("/app0/")) {
            root = g_runtime.app0_directory;
            relative = guest_path.substr(6);
        } else if (guest_path == "/savedata0") {
            root = g_runtime.save_data_directory;
        } else if (guest_path.starts_with("/savedata0/")) {
            root = g_runtime.save_data_directory;
            relative = guest_path.substr(11);
        } else if (guest_path == "/hostapp") {
            root = g_runtime.save_data_directory;
        } else if (guest_path.starts_with("/hostapp/")) {
            root = g_runtime.save_data_directory;
            relative = guest_path.substr(9);
        } else {
            return false;
        }
    }
    if (root.empty()) {
        return false;
    }
    while (!relative.empty() &&
           (relative.front() == '/' || relative.front() == '\\')) {
        relative.remove_prefix(1);
    }
    std::filesystem::path suffix{relative};
    for (const auto& component : suffix) {
        if (component == ".." || component == "/" ||
            component == "\\") {
            return false;
        }
    }
    host_path = relative.empty()
        ? root
        : (root / suffix).lexically_normal();
    std::error_code error;
    if (!relative.empty() &&
        !std::filesystem::exists(host_path, error)) {
        std::filesystem::path resolved = root;
        bool fully_resolved = true;
        const auto equals_ignoring_ascii_case =
            [](const std::string_view left,
               const std::string_view right) {
                if (left.size() != right.size()) {
                    return false;
                }
                return std::ranges::equal(
                    left, right,
                    [](const unsigned char lhs,
                       const unsigned char rhs) {
                        return std::tolower(lhs) ==
                               std::tolower(rhs);
                    });
            };
        for (const auto& component : suffix) {
            const auto direct = resolved / component;
            error.clear();
            if (std::filesystem::exists(direct, error) &&
                !error) {
                resolved = direct;
                continue;
            }
            bool matched{};
            error.clear();
            for (std::filesystem::directory_iterator iterator{
                     resolved, error},
                 end;
                 !error && iterator != end;
                 iterator.increment(error)) {
                if (equals_ignoring_ascii_case(
                        iterator->path().filename().string(),
                        component.string())) {
                    resolved = iterator->path();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                fully_resolved = false;
                break;
            }
        }
        if (fully_resolved) {
            host_path = resolved.lexically_normal();
        }
    }
    error.clear();
    if (std::filesystem::exists(host_path, error) &&
        !error) {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.resolved_guest_paths.insert_or_assign(
            std::string{guest_path}, host_path);
    }
    return true;
}

bool TryResolveGuestRegularFile(
    const std::string_view guest_path,
    std::filesystem::path& host_path,
    std::uint64_t& file_size) {
    std::filesystem::path root;
    std::string_view relative;
    const std::string path_key{guest_path};
    bool has_indexed_size{};
    std::uint64_t indexed_size{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (const auto cached =
                g_runtime.resolved_guest_paths.find(path_key);
            cached != g_runtime.resolved_guest_paths.end()) {
            host_path = cached->second;
            if (const auto cached_size =
                    g_runtime.resolved_guest_file_sizes.find(
                        path_key);
                cached_size !=
                    g_runtime.resolved_guest_file_sizes.end()) {
                file_size = cached_size->second;
                return true;
            }
        } else if (guest_path.starts_with("/app0/")) {
            root = g_runtime.app0_directory;
            relative = guest_path.substr(6);
        } else if (guest_path.starts_with("/savedata0/")) {
            root = g_runtime.save_data_directory;
            relative = guest_path.substr(11);
        } else if (guest_path.starts_with("/hostapp/")) {
            root = g_runtime.save_data_directory;
            relative = guest_path.substr(9);
        } else {
            return false;
        }
        if (const auto indexed =
                g_runtime.indexed_guest_file_sizes.find(path_key);
            indexed !=
                g_runtime.indexed_guest_file_sizes.end()) {
            has_indexed_size = true;
            indexed_size = indexed->second;
        }
    }
    if (host_path.empty()) {
        while (!relative.empty() &&
               (relative.front() == '/' ||
                relative.front() == '\\')) {
            relative.remove_prefix(1);
        }
        if (root.empty() || relative.empty()) {
            return false;
        }
        std::filesystem::path suffix{relative};
        for (const auto& component : suffix) {
            if (component == ".." || component == "/" ||
                component == "\\") {
                return false;
            }
        }
        host_path = (root / suffix).lexically_normal();
    }
    if (has_indexed_size) {
        file_size = indexed_size;
        return true;
    }
    std::error_code error;
    const auto size = std::filesystem::file_size(
        host_path, error);
    if (!error) {
        file_size = static_cast<std::uint64_t>(size);
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.resolved_guest_paths.insert_or_assign(
            path_key, host_path);
        g_runtime.resolved_guest_file_sizes.insert_or_assign(
            path_key, file_size);
        return true;
    }

    host_path.clear();
    if (!TryResolveGuestPath(guest_path, host_path)) {
        return false;
    }
    error.clear();
    const auto fallback_size =
        std::filesystem::file_size(host_path, error);
    if (error) {
        return false;
    }
    file_size = static_cast<std::uint64_t>(fallback_size);
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.resolved_guest_file_sizes.insert_or_assign(
            path_key, file_size);
    }
    return true;
}

bool LoadGuestFileSizeIndex(
    const std::filesystem::path& manifest_path,
    std::unordered_map<std::string, std::uint64_t>& index) {
    constexpr std::array<char, 8> ExpectedMagic{
        'L', 'S', 'X', '4', 'F', 'S', '1', '\0'};
    constexpr std::uint64_t MaximumEntries = 2'000'000;
    constexpr std::uint32_t MaximumPathLength = 4096;
    auto* const stream =
        std::fopen(manifest_path.string().c_str(), "rb");
    if (stream == nullptr) {
        return false;
    }
    std::array<char, 8> magic{};
    std::uint64_t count{};
    const bool valid_header =
        std::fread(magic.data(), 1, magic.size(), stream) ==
            magic.size() &&
        magic == ExpectedMagic &&
        std::fread(&count, sizeof(count), 1, stream) == 1 &&
        count <= MaximumEntries;
    if (!valid_header) {
        std::fclose(stream);
        return false;
    }
    index.clear();
    index.reserve(static_cast<std::size_t>(count));
    std::string path;
    for (std::uint64_t entry = 0; entry < count; ++entry) {
        std::uint32_t path_length{};
        std::uint64_t file_size{};
        if (std::fread(
                &path_length, sizeof(path_length), 1, stream) !=
                1 ||
            std::fread(
                &file_size, sizeof(file_size), 1, stream) != 1 ||
            path_length == 0 ||
            path_length > MaximumPathLength) {
            index.clear();
            std::fclose(stream);
            return false;
        }
        path.resize(path_length);
        if (std::fread(
                path.data(), 1, path.size(), stream) !=
            path.size()) {
            index.clear();
            std::fclose(stream);
            return false;
        }
        index.emplace(path, file_size);
    }
    std::fclose(stream);
    return true;
}

bool TryWriteGuestKernelStat(
    const std::uint64_t stat_address,
    const std::filesystem::path& host_path) {
    constexpr std::size_t KernelStatSize = 120;
    constexpr std::size_t ModeOffset = 8;
    constexpr std::size_t LinkCountOffset = 10;
    constexpr std::size_t SizeOffset = 72;
    constexpr std::size_t BlocksOffset = 80;
    constexpr std::size_t BlockSizeOffset = 88;
    constexpr std::uint16_t DirectoryMode = 0x41ff;
    constexpr std::uint16_t RegularMode = 0x81ff;

    std::error_code error;
    const auto status = std::filesystem::status(host_path, error);
    if (error || !std::filesystem::exists(status)) {
        return false;
    }
    const bool directory = std::filesystem::is_directory(status);
    std::uint64_t size{};
    if (!directory) {
        const auto host_size =
            std::filesystem::file_size(host_path, error);
        if (error ||
            host_size >
                std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        size = static_cast<std::uint64_t>(host_size);
    }

    std::array<std::uint8_t, KernelStatSize> payload{};
    const std::uint16_t mode =
        directory ? DirectoryMode : RegularMode;
    constexpr std::uint16_t LinkCount = 1;
    const std::int64_t signed_size =
        static_cast<std::int64_t>(size);
    const std::int64_t blocks = directory
        ? 128
        : static_cast<std::int64_t>((size + 511) / 512);
    const std::uint32_t block_size =
        directory ? 65536u : 512u;
    std::memcpy(
        payload.data() + ModeOffset, &mode, sizeof(mode));
    std::memcpy(
        payload.data() + LinkCountOffset,
        &LinkCount, sizeof(LinkCount));
    std::memcpy(
        payload.data() + SizeOffset,
        &signed_size, sizeof(signed_size));
    std::memcpy(
        payload.data() + BlocksOffset,
        &blocks, sizeof(blocks));
    std::memcpy(
        payload.data() + BlockSizeOffset,
        &block_size, sizeof(block_size));
    return TryWriteGuestBytes(
        stat_address, payload.data(), payload.size());
}

bool TryGuestKernelStat(
    const std::uint64_t path_address,
    const std::uint64_t stat_address,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (path_address == 0 || stat_address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::string guest_path;
    if (!TryReadGuestCString(
            path_address, 4096, guest_path)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    std::filesystem::path host_path;
    if (!TryResolveGuestPath(guest_path, host_path)) {
        result = OrbisError(NotFound);
        return true;
    }
    if (!TryWriteGuestKernelStat(
            stat_address, host_path)) {
        result = OrbisError(NotFound);
        return true;
    }
#ifdef __ANDROID__
    if (std::getenv(
            "EXECUTOR_DIAG_JIT_RIP_PROBE") != nullptr) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-STAT",
            "guest=%s host=%s output=0x%llx",
            guest_path.c_str(), host_path.string().c_str(),
            static_cast<unsigned long long>(stat_address));
    }
#endif
    result = 0;
    return true;
}

bool TryGuestKernelOpen(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    const auto path_address = request.integer_arguments[0];
    const auto flags =
        static_cast<std::uint32_t>(request.integer_arguments[1]);
    if (path_address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::string guest_path;
    std::filesystem::path host_path;
    if (!TryReadGuestCString(
            path_address, 4096, guest_path) ||
        !TryResolveGuestPath(guest_path, host_path)) {
        result = OrbisError(NotFound);
        return true;
    }
    constexpr std::uint32_t WriteOnly = 0x0001;
    constexpr std::uint32_t ReadWrite = 0x0002;
    constexpr std::uint32_t Append = 0x0008;
    constexpr std::uint32_t Create = 0x0200;
    constexpr std::uint32_t Truncate = 0x0400;
    const bool writes =
        (flags & (WriteOnly | ReadWrite)) != 0;
    if (writes && guest_path.starts_with("/app0")) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    if ((flags & Create) != 0) {
        std::error_code error;
        std::filesystem::create_directories(
            host_path.parent_path(), error);
        if (error) {
            result = OrbisError(NotFound);
            return true;
        }
    }
    const char* mode = "rb";
    if (writes) {
        mode = (flags & Append) != 0
            ? "ab+"
            : (flags & Truncate) != 0
                ? "wb+"
                : "rb+";
    }
    auto* stream = std::fopen(
        host_path.string().c_str(), mode);
    if (stream == nullptr && writes &&
        (flags & Create) != 0) {
        stream = std::fopen(
            host_path.string().c_str(), "wb+");
    }
    if (stream == nullptr) {
        result = OrbisError(NotFound);
        return true;
    }
    std::int32_t handle{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        handle = g_runtime.next_kernel_file_handle++;
        g_runtime.kernel_files.push_back({
            .handle = handle,
            .path = host_path,
            .stream = stream,
        });
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-FS",
            "open guest=%s host=%s flags=0x%x fd=%d",
            guest_path.c_str(), host_path.string().c_str(),
            flags, handle);
    }
#endif
    result = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(handle));
    return true;
}

bool TryGuestKernelClose(
    const std::int32_t handle, std::uint64_t& result) {
    std::FILE* stream{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.kernel_files, handle,
            &GuestKernelFile::handle);
        if (found == g_runtime.kernel_files.end()) {
            result = OrbisError(UINT32_C(0x80020009));
            return true;
        }
        stream = found->stream;
        g_runtime.kernel_files.erase(found);
    }
    result =
        stream == nullptr || std::fclose(stream) == 0
            ? 0
            : OrbisError(UINT32_C(0x80020009));
    return true;
}

bool TryGuestKernelFstat(
    const std::int32_t handle,
    const std::uint64_t stat_address,
    std::uint64_t& result) {
    if (stat_address == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    std::filesystem::path host_path;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.kernel_files, handle,
            &GuestKernelFile::handle);
        if (found == g_runtime.kernel_files.end()) {
            result = OrbisError(UINT32_C(0x80020009));
            return true;
        }
        host_path = found->path;
    }
    result = TryWriteGuestKernelStat(
                 stat_address, host_path)
        ? 0
        : OrbisError(UINT32_C(0x80020002));
    return true;
}

bool TryGuestKernelRead(
    const std::int32_t handle,
    const std::uint64_t destination,
    const std::uint64_t byte_count,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t BadFileDescriptor = 0x80020009;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (byte_count == 0) {
        result = 0;
        return true;
    }
    if (destination == 0 ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(byte_count));
    std::size_t bytes_read{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.kernel_files, handle,
            &GuestKernelFile::handle);
        if (found == g_runtime.kernel_files.end() ||
            found->stream == nullptr) {
            result = OrbisError(BadFileDescriptor);
            return true;
        }
        bytes_read = std::fread(
            bytes.data(), 1, bytes.size(), found->stream);
        if (bytes_read == 0 && std::ferror(found->stream) != 0) {
            result = OrbisError(BadFileDescriptor);
            return true;
        }
    }
    if (!TryWriteGuestBytes(
            destination, bytes.data(), bytes_read)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    result = bytes_read;
    return true;
}

bool TryGuestKernelWrite(
    const std::int32_t handle,
    const std::uint64_t source,
    const std::uint64_t byte_count,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t BadFileDescriptor = 0x80020009;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (byte_count == 0) {
        result = 0;
        return true;
    }
    if (source == 0 ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(byte_count));
    if (!TryReadGuestBytes(
            source, bytes.data(), bytes.size())) {
        result = OrbisError(MemoryFault);
        return true;
    }
    std::size_t bytes_written{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.kernel_files, handle,
            &GuestKernelFile::handle);
        if (found == g_runtime.kernel_files.end() ||
            found->stream == nullptr) {
            result = OrbisError(BadFileDescriptor);
            return true;
        }
        bytes_written = std::fwrite(
            bytes.data(), 1, bytes.size(), found->stream);
        std::fflush(found->stream);
        if (bytes_written == 0 &&
            std::ferror(found->stream) != 0) {
            result = OrbisError(BadFileDescriptor);
            return true;
        }
    }
    result = bytes_written;
    return true;
}

bool TryResolveAprGuestPath(
    const std::uint64_t path_list_address,
    const std::uint64_t index,
    std::string& guest_path) {
    const auto try_pointer =
        [&](const std::uint64_t pointer_address) {
            std::uint64_t candidate{};
            return TryReadGuestBytes(
                       pointer_address, &candidate,
                       sizeof(candidate)) &&
                   TryReadGuestCString(
                       candidate, 4096, guest_path) &&
                   !guest_path.empty();
        };
    const auto indexed_address =
        path_list_address + index * sizeof(std::uint64_t);
    if (indexed_address >= path_list_address &&
        try_pointer(indexed_address)) {
        return true;
    }
    if (index != 0) {
        return false;
    }
    if (TryReadGuestCString(
            path_list_address, 4096, guest_path) &&
        !guest_path.empty()) {
        return true;
    }
    constexpr std::uint64_t ScanLimit = 0x40;
    for (std::uint64_t offset = 0;
         offset < ScanLimit;
         offset += sizeof(std::uint64_t)) {
        if (try_pointer(path_list_address + offset)) {
            return true;
        }
    }
    return false;
}

std::uint32_t RegisterAprFile(
    const std::filesystem::path& host_path) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto path_key = host_path.string();
    if (const auto existing =
            g_runtime.apr_file_ids_by_path.find(path_key);
        existing != g_runtime.apr_file_ids_by_path.end()) {
        return existing->second;
    }
    std::uint32_t identifier{};
    do {
        identifier = g_runtime.next_apr_file_id++;
    } while (identifier == 0 ||
             g_runtime.apr_files.contains(identifier));
    g_runtime.apr_files.emplace(identifier, host_path);
    g_runtime.apr_file_ids_by_path.emplace(path_key, identifier);
    return identifier;
}

bool TryResolveAprFilepathsToIds(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto path_list_address =
        request.integer_arguments[0];
    const auto count = request.integer_arguments[1];
    const auto identifiers_address =
        request.integer_arguments[2];
    if (path_list_address == 0 || count == 0 ||
        identifiers_address == 0 || count > 1024) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        constexpr std::uint32_t InvalidIdentifier =
            std::numeric_limits<std::uint32_t>::max();
        const auto output_address =
            identifiers_address +
            index * sizeof(std::uint32_t);
        if (output_address < identifiers_address ||
            !TryWriteGuestBytes(
                output_address, &InvalidIdentifier,
                sizeof(InvalidIdentifier))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        std::string guest_path;
        if (!TryResolveAprGuestPath(
                path_list_address, index, guest_path)) {
#ifdef __ANDROID__
            if (Ps5DiagnosticFaultProbeEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-APR",
                    "resolve failure=guest-path list=0x%llx "
                    "index=%llu output=0x%llx",
                    static_cast<unsigned long long>(
                        path_list_address),
                    static_cast<unsigned long long>(index),
                    static_cast<unsigned long long>(
                        output_address));
            }
#endif
            result = OrbisError(MemoryFault);
            return true;
        }
        std::filesystem::path host_path;
        std::uint64_t ignored_size{};
        if (!TryResolveGuestRegularFile(
                guest_path, host_path, ignored_size)) {
#ifdef __ANDROID__
            if (Ps5DiagnosticFaultProbeEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-APR",
                    "resolve failure=host-file guest=%s host=%s "
                    "error=%d",
                    guest_path.c_str(),
                    host_path.string().c_str(),
                    errno);
            }
#endif
            result = OrbisError(NotFound);
            return true;
        }
        const auto identifier =
            RegisterAprFile(host_path);
        if (!TryWriteGuestBytes(
                output_address, &identifier,
                sizeof(identifier))) {
            result = OrbisError(MemoryFault);
            return true;
        }
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled() &&
            (identifier <= 64u ||
             (identifier & 0x3ffu) == 0u)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-APR",
                "resolve guest=%s host=%s id=0x%x",
                guest_path.c_str(),
                host_path.string().c_str(), identifier);
        }
#endif
    }
    result = 0;
    return true;
}

bool TryResolveAprFilepathsToIdsAndSizes(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto path_list_address = request.integer_arguments[0];
    const auto count = request.integer_arguments[1];
    const auto identifiers_address = request.integer_arguments[2];
    const auto sizes_address = request.integer_arguments[3];
    const auto error_index_address = request.integer_arguments[4];
    if (path_list_address == 0 || count == 0 ||
        sizes_address == 0 || count > 1024) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    for (std::uint64_t index = 0; index < count; ++index) {
        constexpr std::uint32_t InvalidIdentifier =
            std::numeric_limits<std::uint32_t>::max();
        if (identifiers_address != 0 &&
            !TryWriteGuestBytes(
                identifiers_address +
                    index * sizeof(std::uint32_t),
                &InvalidIdentifier,
                sizeof(InvalidIdentifier))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        std::string guest_path;
        std::filesystem::path host_path;
        std::uint64_t file_size{};
        if (!TryResolveAprGuestPath(
                path_list_address, index, guest_path) ||
            !TryResolveGuestRegularFile(
                guest_path, host_path, file_size)) {
            const std::uint64_t zero{};
            if (!TryWriteGuestBytes(
                    sizes_address +
                        index * sizeof(std::uint64_t),
                    &zero, sizeof(zero)) ||
                (error_index_address != 0 &&
                 !TryWriteGuestBytes(
                     error_index_address,
                     &index, sizeof(std::uint32_t)))) {
                result = OrbisError(MemoryFault);
                return true;
            }
#ifdef __ANDROID__
            if (Ps5DiagnosticFaultProbeEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-APR",
                    "resolve-size missing index=%llu guest=%s host=%s",
                    static_cast<unsigned long long>(index),
                    guest_path.empty() ? "<invalid>" : guest_path.c_str(),
                    host_path.empty()
                        ? "<invalid>"
                        : host_path.string().c_str());
            }
#endif
            result = std::numeric_limits<std::uint64_t>::max();
            return true;
        }
        const auto identifier = RegisterAprFile(host_path);
        if ((identifiers_address != 0 &&
             !TryWriteGuestBytes(
                 identifiers_address +
                     index * sizeof(std::uint32_t),
                 &identifier, sizeof(identifier))) ||
            !TryWriteGuestBytes(
                sizes_address +
                    index * sizeof(std::uint64_t),
                &file_size, sizeof(file_size))) {
            result = OrbisError(MemoryFault);
            return true;
        }
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled() &&
            (identifier <= 64u ||
             (identifier & 0x3ffu) == 0u)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-APR",
                "resolve-size guest=%s id=0x%x size=0x%llx",
                guest_path.c_str(), identifier,
                static_cast<unsigned long long>(file_size));
        }
#endif
    }
    result = 0;
    return true;
}

bool TryGetAprFileStat(
    const std::uint32_t identifier,
    const std::uint64_t stat_address,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    if (stat_address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::filesystem::path host_path;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.apr_files.find(identifier);
        if (found == g_runtime.apr_files.end()) {
            result = OrbisError(NotFound);
            return true;
        }
        host_path = found->second;
    }
    if (!TryWriteGuestKernelStat(
            stat_address, host_path)) {
        result = OrbisError(NotFound);
        return true;
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-APR",
            "stat id=0x%x host=%s output=0x%llx",
            identifier, host_path.string().c_str(),
            static_cast<unsigned long long>(stat_address));
    }
#endif
    result = 0;
    return true;
}

constexpr std::uint64_t AprReadFileRecordSize = 0x30;
constexpr std::uint64_t AprKernelEventQueueRecordSize = 0x30;
constexpr std::uint64_t AprWriteAddressRecordSize = 0x20;

bool TryAppendAprRecord(
    const std::uint64_t command_buffer,
    const void* const record,
    const std::size_t record_size) {
    if (record == nullptr || record_size == 0) {
        return false;
    }
    const std::lock_guard lock{g_runtime.mutex};
    const auto found =
        g_runtime.apr_command_buffers.find(command_buffer);
    if (found == g_runtime.apr_command_buffers.end()) {
        return false;
    }
    auto& state = found->second;
    if (state.buffer == 0 ||
        state.write_offset > state.size ||
        record_size > state.size - state.write_offset) {
        return false;
    }
    const auto record_address = state.buffer + state.write_offset;
    if (record_address < state.buffer ||
        !HasAccessLocked(
            record_address, record_size, LSX4_PS5_GUEST_WRITE)) {
        return false;
    }
    std::memcpy(
        reinterpret_cast<void*>(record_address),
        record, record_size);
    state.write_offset += record_size;
    ++state.command_count;
    return true;
}

bool TryReadAprFile(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    const auto identifier = static_cast<std::uint32_t>(
        request.integer_arguments[3]);
    const auto destination = request.integer_arguments[4];
    const auto requested_size = request.integer_arguments[5];
    if (command_buffer == 0 ||
        (destination == 0 && requested_size != 0) ||
        requested_size >
            std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::uint64_t file_offset{};
    if (!TryReadGuestBytes(
            request.guest_stack + sizeof(std::uint64_t),
            &file_offset, sizeof(file_offset))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (file_offset >
        static_cast<std::uint64_t>(
            std::numeric_limits<off_t>::max())) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    std::filesystem::path host_path;
    std::uint64_t read_index{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.apr_files.find(identifier);
        if (found == g_runtime.apr_files.end()) {
            result = OrbisError(NotFound);
            return true;
        }
        if (requested_size != 0 &&
            !HasAccessLocked(
                destination, requested_size,
                LSX4_PS5_GUEST_WRITE)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        host_path = found->second;
        read_index = ++g_runtime.apr_read_count;
    }

    std::size_t bytes_read{};
    if (requested_size != 0) {
        auto* const stream =
            std::fopen(host_path.string().c_str(), "rb");
        if (stream == nullptr) {
            result = OrbisError(NotFound);
            return true;
        }
        const auto seek_result = ::fseeko(
            stream, static_cast<off_t>(file_offset), SEEK_SET);
        if (seek_result == 0) {
            bytes_read = std::fread(
                reinterpret_cast<void*>(destination), 1,
                static_cast<std::size_t>(requested_size), stream);
        }
        const auto read_error = std::ferror(stream);
        std::fclose(stream);
        if (seek_result != 0 || read_error != 0) {
            result = OrbisError(NotFound);
            return true;
        }
    }
    std::array<std::uint8_t, AprReadFileRecordSize> record{};
    constexpr std::uint32_t ReadFileRecordType = 1;
    const auto bytes_read_u64 =
        static_cast<std::uint64_t>(bytes_read);
    std::memcpy(record.data() + 0x00, &ReadFileRecordType,
                sizeof(ReadFileRecordType));
    std::memcpy(record.data() + 0x04, &identifier,
                sizeof(identifier));
    std::memcpy(record.data() + 0x08, &destination,
                sizeof(destination));
    std::memcpy(record.data() + 0x10, &requested_size,
                sizeof(requested_size));
    std::memcpy(record.data() + 0x18, &file_offset,
                sizeof(file_offset));
    std::memcpy(record.data() + 0x20, &bytes_read_u64,
                sizeof(bytes_read_u64));
    if (!TryAppendAprRecord(
            command_buffer, record.data(), record.size())) {
        result = OrbisError(MemoryFault);
        return true;
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled() &&
        (read_index <= 64 || (read_index & 0x3ffu) == 0)) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-APR",
            "read index=%llu id=0x%x host=%s offset=0x%llx "
            "size=0x%llx bytes=0x%zx command=0x%llx",
            static_cast<unsigned long long>(read_index),
            identifier, host_path.string().c_str(),
            static_cast<unsigned long long>(file_offset),
            static_cast<unsigned long long>(requested_size),
            bytes_read,
            static_cast<unsigned long long>(command_buffer));
    }
#endif
    result = 0;
    return true;
}

bool TryConstructAprCommandBuffer(
    const Lsx4::Translation::HleBridgeRequest& request,
    const bool preserve_buffer,
    std::uint64_t& result) {
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    if (command_buffer == 0) {
        result = 0;
        return true;
    }
    std::array<std::uint64_t, 5> header{};
    if (preserve_buffer &&
        !TryReadGuestBytes(
            command_buffer, header.data(), sizeof(header))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    header[0] = command_buffer;
    if (preserve_buffer) {
        header[3] = request.integer_arguments[1];
        header[4] = request.integer_arguments[2];
    } else {
        header[1] = request.integer_arguments[1];
        header[2] = request.integer_arguments[2];
    }
    if (!TryWriteGuestBytes(
            command_buffer, header.data(), sizeof(header))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_command_buffers[command_buffer] = {
            .buffer = header[1],
            .size = header[2],
        };
        g_runtime.apr_completion_writes.erase(command_buffer);
        g_runtime.apr_completion_events.erase(command_buffer);
    }
    result = command_buffer;
    return true;
}

bool TryDestroyAprCommandBuffer(
    const std::uint64_t command_buffer,
    const bool auxiliary_only,
    std::uint64_t& result) {
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (command_buffer == 0) {
        result = 0;
        return true;
    }
    const std::array<std::uint64_t, 2> zeros{};
    const auto address = auxiliary_only
        ? command_buffer + 3 * sizeof(std::uint64_t)
        : command_buffer + sizeof(std::uint64_t);
    if (!TryWriteGuestBytes(
            address, zeros.data(), sizeof(zeros))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (!auxiliary_only) {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_command_buffers.erase(command_buffer);
        g_runtime.apr_completion_writes.erase(command_buffer);
        g_runtime.apr_completion_events.erase(command_buffer);
    }
    result = 0;
    return true;
}

bool TrySetAprCommandBuffer(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    const auto buffer = request.integer_arguments[1];
    const auto size = request.integer_arguments[2];
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const std::array<std::uint64_t, 3> visible_pointers{
        command_buffer, buffer, size};
    if (!TryWriteGuestBytes(
            command_buffer, visible_pointers.data(),
            sizeof(visible_pointers))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_command_buffers[command_buffer] = {
            .buffer = buffer,
            .size = size,
        };
        g_runtime.apr_completion_writes.erase(command_buffer);
        g_runtime.apr_completion_events.erase(command_buffer);
    }
    result = 0;
    return true;
}

bool TryClearAprCommandBuffer(
    const std::uint64_t command_buffer,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::uint64_t buffer{};
    if (!TryReadGuestBytes(
            command_buffer + sizeof(std::uint64_t),
            &buffer, sizeof(buffer))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    const std::array<std::uint64_t, 3> visible_pointers{
        command_buffer, 0, 0};
    if (!TryWriteGuestBytes(
            command_buffer, visible_pointers.data(),
            sizeof(visible_pointers))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_command_buffers.erase(command_buffer);
        g_runtime.apr_completion_writes.erase(command_buffer);
        g_runtime.apr_completion_events.erase(command_buffer);
    }
    result = buffer;
    return true;
}

bool TryAppendAprCompletionWrite(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    const auto address = request.integer_arguments[1];
    const auto value = request.integer_arguments[2];
    if (command_buffer == 0 || address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::array<std::uint8_t, AprWriteAddressRecordSize> record{};
    constexpr std::uint32_t WriteAddressRecordType = 3;
    std::memcpy(record.data() + 0x00, &WriteAddressRecordType,
                sizeof(WriteAddressRecordType));
    std::memcpy(record.data() + 0x08, &address, sizeof(address));
    std::memcpy(record.data() + 0x10, &value, sizeof(value));
    if (!TryAppendAprRecord(
            command_buffer, record.data(), record.size())) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!HasAccessLocked(
                address, sizeof(std::uint64_t),
                LSX4_PS5_GUEST_WRITE)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        g_runtime.apr_completion_writes[command_buffer]
            .emplace_back(address, value);
    }
    result = 0;
    return true;
}

bool TryAppendAprCompletionEvent(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const AprCompletionEvent completion{
        .event_queue = request.integer_arguments[1],
        .ident = request.integer_arguments[2],
        .completion_token = request.integer_arguments[3],
        .user_data = request.integer_arguments[4],
    };
    std::array<std::uint8_t, AprKernelEventQueueRecordSize> record{};
    constexpr std::uint32_t KernelEventQueueRecordType = 2;
    constexpr std::int16_t KernelEventFilterAmpr = -16;
    std::memcpy(record.data() + 0x00, &KernelEventQueueRecordType,
                sizeof(KernelEventQueueRecordType));
    std::memcpy(record.data() + 0x04, &KernelEventFilterAmpr,
                sizeof(KernelEventFilterAmpr));
    std::memcpy(record.data() + 0x08, &completion.event_queue,
                sizeof(completion.event_queue));
    std::memcpy(record.data() + 0x10, &completion.ident,
                sizeof(completion.ident));
    std::memcpy(record.data() + 0x18, &completion.user_data,
                sizeof(completion.user_data));
    std::memcpy(record.data() + 0x20, &completion.completion_token,
                sizeof(completion.completion_token));
    if (!TryAppendAprRecord(
            command_buffer, record.data(), record.size())) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_completion_events[command_buffer]
            .push_back(completion);
    }
    result = 0;
    return true;
}

bool TryGetAprCommandBufferProperty(
    const std::uint64_t command_buffer,
    const std::uint32_t property,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const std::lock_guard lock{g_runtime.mutex};
    const auto found =
        g_runtime.apr_command_buffers.find(command_buffer);
    if (found == g_runtime.apr_command_buffers.end()) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (property == 0) {
        result = found->second.size;
    } else if (property == 1) {
        result = found->second.write_offset;
    } else {
        result = found->second.command_count;
    }
    return true;
}

bool TrySubmitAprCommandBuffer(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    const auto command_buffer = request.integer_arguments[0];
    const auto result_address = request.integer_arguments[2];
    const auto identifier_address = request.integer_arguments[3];
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::uint32_t identifier{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        do {
            identifier =
                g_runtime.next_apr_submission_id++;
        } while (
            identifier == 0 ||
            g_runtime.apr_submissions.contains(identifier));
        g_runtime.apr_submissions.emplace(
            identifier, command_buffer);
    }
    std::vector<std::uint8_t> command_records;
    std::vector<AprCompletionEvent> completion_events;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto state =
            g_runtime.apr_command_buffers.find(command_buffer);
        if (state == g_runtime.apr_command_buffers.end() ||
            state->second.write_offset > state->second.size ||
            state->second.write_offset >
                std::numeric_limits<std::size_t>::max()) {
            g_runtime.apr_submissions.erase(identifier);
            result = OrbisError(MemoryFault);
            return true;
        }
        command_records.resize(
            static_cast<std::size_t>(
                state->second.write_offset));
        if (!command_records.empty()) {
            if (state->second.buffer == 0 ||
                !HasAccessLocked(
                    state->second.buffer,
                    command_records.size(),
                    LSX4_PS5_GUEST_READ)) {
                g_runtime.apr_submissions.erase(identifier);
                result = OrbisError(MemoryFault);
                return true;
            }
            std::memcpy(
                command_records.data(),
                reinterpret_cast<const void*>(
                    state->second.buffer),
                command_records.size());
        }
        if (const auto found =
                g_runtime.apr_completion_events.find(command_buffer);
            found != g_runtime.apr_completion_events.end()) {
            completion_events = found->second;
        }
    }
    const auto forget_submission = [&] {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.apr_submissions.erase(identifier);
    };
    for (std::size_t offset = 0;
         offset < command_records.size();) {
        if (command_records.size() - offset <
            sizeof(std::uint32_t)) {
            forget_submission();
            result = OrbisError(MemoryFault);
            return true;
        }
        std::uint32_t record_type{};
        std::memcpy(
            &record_type,
            command_records.data() + offset,
            sizeof(record_type));
        std::size_t record_size{};
        if (record_type == 1) {
            record_size = AprReadFileRecordSize;
        } else if (record_type == 2) {
            record_size = AprKernelEventQueueRecordSize;
        } else if (record_type == 3) {
            record_size = AprWriteAddressRecordSize;
        } else {
            forget_submission();
            result = OrbisError(InvalidArgument);
            return true;
        }
        if (record_size > command_records.size() - offset) {
            forget_submission();
            result = OrbisError(MemoryFault);
            return true;
        }
        if (record_type == 3) {
            std::uint64_t address{};
            std::uint64_t value{};
            std::memcpy(
                &address,
                command_records.data() + offset + 0x08,
                sizeof(address));
            std::memcpy(
                &value,
                command_records.data() + offset + 0x10,
                sizeof(value));
            if (address == 0 ||
                !TryWriteGuestBytes(
                    address, &value, sizeof(value))) {
                forget_submission();
                result = OrbisError(MemoryFault);
                return true;
            }
        }
        offset += record_size;
    }
    bool queued_completion_event = false;
    {
        const std::lock_guard lock{g_runtime.mutex};
        for (const auto& completion : completion_events) {
            const auto queue = g_runtime.guest_event_queues.find(
                completion.event_queue);
            if (queue == g_runtime.guest_event_queues.end()) {
                continue;
            }
            std::array<std::uint8_t, 0x20> event{};
            constexpr std::int16_t KernelEventFilterAmpr = -16;
            constexpr std::uint16_t KernelEventFlags = 0x20;
            std::memcpy(event.data() + 0x00, &completion.ident,
                        sizeof(completion.ident));
            std::memcpy(event.data() + 0x08, &KernelEventFilterAmpr,
                        sizeof(KernelEventFilterAmpr));
            std::memcpy(event.data() + 0x0a, &KernelEventFlags,
                        sizeof(KernelEventFlags));
            std::memcpy(event.data() + 0x10,
                        &completion.completion_token,
                        sizeof(completion.completion_token));
            std::memcpy(event.data() + 0x18, &completion.user_data,
                        sizeof(completion.user_data));
            queue->second.push_back(event);
            queued_completion_event = true;
        }
    }
    if (queued_completion_event) {
        g_runtime.guest_event_queue_condition.notify_all();
    }
    if (identifier_address != 0 &&
        !TryWriteGuestBytes(
            identifier_address, &identifier,
            sizeof(identifier))) {
        forget_submission();
        result = OrbisError(MemoryFault);
        return true;
    }
    if (result_address != 0) {
        constexpr std::uint64_t CompletionResult = 0;
        if (!TryWriteGuestBytes(
                result_address, &CompletionResult,
                sizeof(CompletionResult))) {
            forget_submission();
            result = OrbisError(MemoryFault);
            return true;
        }
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled() &&
        (identifier <= 64u ||
         (identifier & 0x3ffu) == 0u)) {
        std::array<std::uint64_t, 2> result_qwords{};
        const bool result_qwords_ok =
            result_address != 0 &&
            TryReadGuestBytes(
                result_address, result_qwords.data(),
                sizeof(result_qwords));
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-APR",
            "submit id=0x%x command=0x%llx priority=0x%llx "
            "result=0x%llx output=0x%llx q0=0x%llx "
            "q1=0x%llx ok=%d",
            identifier,
            static_cast<unsigned long long>(command_buffer),
            static_cast<unsigned long long>(
                request.integer_arguments[1]),
            static_cast<unsigned long long>(result_address),
            static_cast<unsigned long long>(identifier_address),
            static_cast<unsigned long long>(result_qwords[0]),
            static_cast<unsigned long long>(result_qwords[1]),
            result_qwords_ok ? 1 : 0);
    }
#endif
    result = 0;
    return true;
}

bool TryWaitAprCommandBuffer(
    const std::uint32_t identifier,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    std::uint64_t command_buffer{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.apr_submissions.find(identifier);
        if (found == g_runtime.apr_submissions.end()) {
            result = OrbisError(NotFound);
            return true;
        }
        command_buffer = found->second;
        g_runtime.apr_submissions.erase(found);
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        static std::atomic<std::uint64_t> wait_logs{};
        const auto wait_index =
            wait_logs.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (wait_index <= 32u ||
            (wait_index & (wait_index - 1u)) == 0u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-APR",
                "wait count=%llu id=0x%x command=0x%llx",
                static_cast<unsigned long long>(wait_index),
                identifier,
                static_cast<unsigned long long>(command_buffer));
        }
    }
#endif
    result = 0;
    return true;
}

bool TryResetAprCommandBuffer(
    const std::uint64_t command_buffer,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (command_buffer == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::array<std::uint64_t, 2> data_and_size{};
    if (!TryReadGuestBytes(
            command_buffer + sizeof(std::uint64_t),
            data_and_size.data(), sizeof(data_and_size))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    const std::array<std::uint64_t, 3> visible_pointers{
        command_buffer, data_and_size[0], data_and_size[1]};
    if (!TryWriteGuestBytes(
            command_buffer, visible_pointers.data(),
            sizeof(visible_pointers))) {
        result = OrbisError(MemoryFault);
        return true;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.apr_command_buffers.find(command_buffer);
        if (found != g_runtime.apr_command_buffers.end()) {
            found->second.write_offset = 0;
            found->second.command_count = 0;
        } else {
            g_runtime.apr_command_buffers[command_buffer] = {
                .buffer = data_and_size[0],
                .size = data_and_size[1],
            };
        }
        g_runtime.apr_completion_writes.erase(command_buffer);
        g_runtime.apr_completion_events.erase(command_buffer);
    }
    result = 0;
    return true;
}

bool TryOpenGuestStdio(
    const std::uint64_t path_address,
    const std::uint64_t mode_address,
    std::uint64_t& result) {
    result = 0;
    std::string guest_path;
    std::string mode;
    if (!TryReadGuestCString(path_address, 4096, guest_path) ||
        !TryReadGuestCString(mode_address, 16, mode)) {
        return true;
    }
    std::filesystem::path host_path;
    if (!TryResolveGuestPath(guest_path, host_path)) {
        return true;
    }
    const auto writes =
        !mode.empty() && mode.front() != 'r';
    if (writes && guest_path.starts_with("/app0")) {
        return true;
    }
    if (writes) {
        std::error_code error;
        std::filesystem::create_directories(
            host_path.parent_path(), error);
        if (error) {
            return true;
        }
    }
    auto* const stream =
        std::fopen(host_path.string().c_str(), mode.c_str());
    if (stream == nullptr) {
        if (std::getenv("EXECUTOR_DIAG_JIT_RIP_PROBE") != nullptr) {
            std::fprintf(
                stderr,
                "PS5_STDIO_OPEN guest=%s host=%s mode=%s "
                "errno=%d\n",
                guest_path.c_str(), host_path.string().c_str(),
                mode.c_str(), errno);
        }
        return true;
    }
    constexpr std::size_t GuestFileObjectSize = 0x100;
    std::uint64_t guest_handle{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        auto* page = g_runtime.stdio_handle_pages.empty()
            ? nullptr
            : &g_runtime.stdio_handle_pages.back();
        if (page == nullptr ||
            page->used + GuestFileObjectSize > page->mapped_size) {
            std::uint64_t first_hint{};
            if (!AlignUp(
                    g_runtime.next_libc_heap_hint,
                    LSX4_PS5_GUEST_PAGE_SIZE, first_hint)) {
                std::fclose(stream);
                return true;
            }
            constexpr auto PageSize =
                static_cast<std::size_t>(
                    LSX4_PS5_GUEST_PAGE_SIZE);
            auto* const allocation =
                AllocateLowGuestRegionLocked(
                    first_hint, Ps5LibcHeapStride, PageSize);
            if (allocation == nullptr) {
                std::fclose(stream);
                return true;
            }
            const auto base =
                reinterpret_cast<std::uint64_t>(allocation);
            if (!RegisterMappingLocked(
                    base, PageSize,
                    LSX4_PS5_GUEST_READ |
                        LSX4_PS5_GUEST_WRITE,
                    "ps5-stdio-handle", true)) {
                munmap(allocation, PageSize);
                std::fclose(stream);
                return true;
            }
            g_runtime.stdio_handle_pages.push_back({
                .allocation = allocation,
                .mapped_size = PageSize,
                .base = base,
                .used = 0,
            });
            g_runtime.next_libc_heap_hint = base + PageSize;
            page = &g_runtime.stdio_handle_pages.back();
        }
        guest_handle = page->base + page->used;
        std::memset(
            page->allocation + page->used, 0,
            GuestFileObjectSize);
        page->used += GuestFileObjectSize;
        g_runtime.stdio_files.push_back({
            .guest_handle = guest_handle,
            .stream = stream,
        });
    }
    if (guest_handle == 0) {
        std::fclose(stream);
        return true;
    }
    if (std::getenv("EXECUTOR_DIAG_JIT_RIP_PROBE") != nullptr) {
        std::fprintf(
            stderr,
            "PS5_STDIO_OPEN guest=%s host=%s mode=%s "
            "handle=0x%llx\n",
            guest_path.c_str(), host_path.string().c_str(),
            mode.c_str(),
            static_cast<unsigned long long>(guest_handle));
    }
    result = guest_handle;
    return true;
}

std::FILE* FindGuestStdio(const std::uint64_t guest_handle) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto found = std::ranges::find(
        g_runtime.stdio_files, guest_handle,
        &GuestStdioFile::guest_handle);
    return found != g_runtime.stdio_files.end()
        ? found->stream
        : nullptr;
}

struct AgcRegisterDefaultRecord {
    std::uint32_t group{};
    std::uint32_t space{};
    std::uint32_t index{};
    std::uint32_t type{};
    std::uint32_t register_index{};
    std::uint32_t offset{};
    std::uint32_t value{};
};

#include "agc_register_defaults.inc"

bool TryBuildAgcRegisterDefaults(
    const std::span<const AgcRegisterDefaultRecord> records,
    const std::uint32_t group_count,
    const std::uint32_t cx_table_length,
    const std::uint32_t sh_table_length,
    const std::uint32_t uc_table_length,
    std::uint64_t& result) {
    constexpr std::size_t HeaderBytes = 0x40;
    constexpr std::size_t RegisterBlockBytes = 16u * 8u;
    const auto align_up = [](const std::size_t value,
                             const std::size_t alignment) {
        return (value + alignment - 1u) & ~(alignment - 1u);
    };
    const auto cx_table_offset =
        align_up(HeaderBytes, sizeof(std::uint64_t));
    const auto sh_table_offset =
        cx_table_offset +
        static_cast<std::size_t>(cx_table_length) *
            sizeof(std::uint64_t);
    const auto uc_table_offset =
        sh_table_offset +
        static_cast<std::size_t>(sh_table_length) *
            sizeof(std::uint64_t);
    const auto types_offset = align_up(
        uc_table_offset +
            static_cast<std::size_t>(uc_table_length) *
                sizeof(std::uint64_t),
        sizeof(std::uint32_t));
    const auto register_blocks_offset = align_up(
        types_offset +
            static_cast<std::size_t>(group_count) *
                3u * sizeof(std::uint32_t),
        sizeof(std::uint64_t));
    const auto blob_length =
        register_blocks_offset +
        static_cast<std::size_t>(group_count) *
            RegisterBlockBytes;

    std::uint64_t allocation{};
    if (!TryAllocateLibcHeap(
            blob_length, LSX4_PS5_GUEST_PAGE_SIZE,
            allocation)) {
        result = 0;
        return true;
    }
    std::vector<std::uint8_t> blob(blob_length);
    const auto write_u32 =
        [&](const std::size_t offset,
            const std::uint32_t value) {
            if (offset + sizeof(value) > blob.size()) {
                return false;
            }
            std::memcpy(blob.data() + offset, &value,
                        sizeof(value));
            return true;
        };
    const auto write_u64 =
        [&](const std::size_t offset,
            const std::uint64_t value) {
            if (offset + sizeof(value) > blob.size()) {
                return false;
            }
            std::memcpy(blob.data() + offset, &value,
                        sizeof(value));
            return true;
        };
    bool valid =
        write_u64(0x00, allocation + cx_table_offset) &&
        write_u64(0x08, allocation + sh_table_offset) &&
        write_u64(0x10, allocation + uc_table_offset) &&
        write_u64(0x30, allocation + types_offset) &&
        write_u32(0x38, group_count);
    for (const auto& record : records) {
        if (!valid || record.group >= group_count ||
            record.register_index >= 16u) {
            valid = false;
            break;
        }
        std::size_t table_offset{};
        std::uint32_t table_length{};
        if (record.space == 0u) {
            table_offset = cx_table_offset;
            table_length = cx_table_length;
        } else if (record.space == 1u) {
            table_offset = sh_table_offset;
            table_length = sh_table_length;
        } else if (record.space == 2u) {
            table_offset = uc_table_offset;
            table_length = uc_table_length;
        } else {
            valid = false;
            break;
        }
        if (record.index >= table_length) {
            valid = false;
            break;
        }
        const auto block_offset =
            register_blocks_offset +
            static_cast<std::size_t>(record.group) *
                RegisterBlockBytes;
        if (record.register_index == 0u) {
            const auto type_offset =
                types_offset +
                static_cast<std::size_t>(record.group) *
                    3u * sizeof(std::uint32_t);
            valid =
                write_u64(
                    table_offset +
                        static_cast<std::size_t>(
                            record.index) *
                            sizeof(std::uint64_t),
                    allocation + block_offset) &&
                write_u32(type_offset, record.type) &&
                write_u32(
                    type_offset + sizeof(std::uint32_t),
                    record.index * 4u + record.space);
        }
        valid =
            valid &&
            write_u32(
                block_offset +
                    static_cast<std::size_t>(
                        record.register_index) *
                        2u * sizeof(std::uint32_t),
                record.offset) &&
            write_u32(
                block_offset +
                    static_cast<std::size_t>(
                        record.register_index) *
                        2u * sizeof(std::uint32_t) +
                    sizeof(std::uint32_t),
                record.value);
    }
    if (!valid ||
        !TryWriteGuestBytes(
            allocation, blob.data(), blob.size())) {
        FreeLibcHeap(allocation);
        result = 0;
        return true;
    }
    result = allocation;
    return true;
}

bool TryGetAgcRegisterDefaults(
    const bool internal_defaults,
    std::uint64_t& result) {
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto cached = internal_defaults
            ? g_runtime.agc_internal_register_defaults
            : g_runtime.agc_register_defaults;
        if (cached != 0) {
            result = cached;
            return true;
        }
    }
    std::uint64_t allocation{};
    const auto built = internal_defaults
        ? TryBuildAgcRegisterDefaults(
              AgcInternalRegisterDefaults, 22u,
              4u, 15u, 3u, allocation)
        : TryBuildAgcRegisterDefaults(
              AgcPrimaryRegisterDefaults, 127u,
              78u, 29u, 20u, allocation);
    if (!built || allocation == 0) {
        result = allocation;
        return built;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        auto& cached = internal_defaults
            ? g_runtime.agc_internal_register_defaults
            : g_runtime.agc_register_defaults;
        if (cached == 0) {
            cached = allocation;
            result = allocation;
            return true;
        }
        result = cached;
    }
    FreeLibcHeap(allocation);
    return true;
}

bool TryRelocateAgcShaderPointer(
    const std::uint64_t field_address) {
    std::uint64_t relative_address{};
    if (!TryReadGuestBytes(
            field_address, &relative_address,
            sizeof(relative_address))) {
        return false;
    }
    if (relative_address == 0) {
        return true;
    }
    if (relative_address >
        std::numeric_limits<std::uint64_t>::max() -
            field_address) {
        return false;
    }
    const auto absolute_address =
        field_address + relative_address;
    return TryWriteGuestBytes(
        field_address, &absolute_address,
        sizeof(absolute_address));
}

bool TryCreateAgcShader(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t ShaderFileHeader = 0x34333231;
    constexpr std::uint32_t ShaderVersion = 0x18;
    constexpr std::array<std::uint64_t, 6> PointerOffsets{
        0x18, 0x20, 0x08, 0x28, 0x30, 0x38};
    const auto destination_address =
        request.integer_arguments[0];
    const auto header_address = request.integer_arguments[1];
    const auto code_address = request.integer_arguments[2];
    if (header_address == 0 || code_address == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    std::array<std::uint32_t, 2> identity{};
    if (!TryReadGuestBytes(
            header_address, identity.data(), sizeof(identity))) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (identity[0] != ShaderFileHeader ||
        identity[1] != ShaderVersion) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    for (const auto offset : PointerOffsets) {
        if (!TryRelocateAgcShaderPointer(
                header_address + offset)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
    }
    if (!TryWriteGuestBytes(
            header_address + 0x10, &code_address,
            sizeof(code_address))) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }

    std::uint64_t user_data_address{};
    if (!TryReadGuestBytes(
            header_address + 0x08, &user_data_address,
            sizeof(user_data_address))) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (user_data_address != 0) {
        for (std::uint64_t offset = 0;
             offset <= 0x20; offset += 0x08) {
            if (!TryRelocateAgcShaderPointer(
                    user_data_address + offset)) {
                result = OrbisError(UINT32_C(0x80020101));
                return true;
            }
        }
    }

    std::uint64_t sh_registers_address{};
    std::uint8_t shader_type{};
    std::uint8_t register_count{};
    if (!TryReadGuestBytes(
            header_address + 0x20, &sh_registers_address,
            sizeof(sh_registers_address)) ||
        !TryReadGuestBytes(
            header_address + 0x5a, &shader_type,
            sizeof(shader_type)) ||
        !TryReadGuestBytes(
            header_address + 0x5c, &register_count,
            sizeof(register_count)) ||
        sh_registers_address == 0 || register_count < 2) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    std::uint32_t lo_register{};
    std::uint32_t hi_register{};
    if (!TryReadGuestBytes(
            sh_registers_address, &lo_register,
            sizeof(lo_register)) ||
        !TryReadGuestBytes(
            sh_registers_address + 0x08, &hi_register,
            sizeof(hi_register))) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    std::uint32_t expected_lo{};
    switch (shader_type) {
    case 0:
        expected_lo = 0x20c;
        break;
    case 1:
        expected_lo = 0x08;
        break;
    case 2:
    case 6:
        expected_lo = 0xc8;
        break;
    case 4:
        expected_lo = 0x8a;
        break;
    case 7:
        expected_lo = 0x148;
        break;
    default:
        break;
    }
    if (expected_lo == 0 ||
        lo_register != expected_lo ||
        hi_register != expected_lo + 1) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    const auto lo_value =
        static_cast<std::uint32_t>(code_address >> 8);
    const auto hi_value =
        static_cast<std::uint32_t>((code_address >> 40) & 0xff);
    if (!TryWriteGuestBytes(
            sh_registers_address + 0x04, &lo_value,
            sizeof(lo_value)) ||
        !TryWriteGuestBytes(
            sh_registers_address + 0x0c, &hi_value,
            sizeof(hi_value)) ||
        (destination_address != 0 &&
         !TryWriteGuestBytes(
             destination_address, &header_address,
             sizeof(header_address)))) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    result = 0;
    return true;
}

template <typename Value>
bool TryReadGuestValue(const std::uint64_t address, Value& value) {
    return TryReadGuestBytes(address, &value, sizeof(value));
}

template <typename Value>
bool TryWriteGuestValue(const std::uint64_t address, const Value value) {
    return TryWriteGuestBytes(address, &value, sizeof(value));
}

constexpr std::uint64_t AgcCommandBufferCursorUpOffset = 0x10;
constexpr std::uint64_t AgcCommandBufferCursorDownOffset = 0x18;
constexpr std::uint64_t AgcCommandBufferReservedDwOffset = 0x30;
constexpr std::uint32_t AgcItNop = 0x10;
constexpr std::uint32_t AgcItSetBase = 0x11;
constexpr std::uint32_t AgcItDispatchDirect = 0x15;
constexpr std::uint32_t AgcItDispatchIndirect = 0x16;
constexpr std::uint32_t AgcComputePgmLo = 0x20c;
constexpr std::uint32_t AgcComputePgmHi = 0x20d;
constexpr std::uint32_t AgcComputeStartX = 0x204;
constexpr std::uint32_t AgcComputeStartY = 0x205;
constexpr std::uint32_t AgcComputeStartZ = 0x206;
constexpr std::uint32_t AgcComputeNumThreadX = 0x207;
constexpr std::uint32_t AgcComputeNumThreadY = 0x208;
constexpr std::uint32_t AgcComputeNumThreadZ = 0x209;
constexpr std::uint32_t AgcComputePgmRsrc1 = 0x212;
constexpr std::uint32_t AgcComputePgmRsrc2 = 0x213;
constexpr std::uint32_t AgcComputeUserData = 0x240;
constexpr std::uint32_t AgcItWaitRegMem = 0x3c;
constexpr std::uint32_t AgcItIndexBufferSize = 0x13;
constexpr std::uint32_t AgcItIndexBase = 0x26;
constexpr std::uint32_t AgcItIndexType = 0x2a;
constexpr std::uint32_t AgcItNumInstances = 0x2f;
constexpr std::uint32_t AgcItDrawIndexOffset2 = 0x35;
constexpr std::uint32_t AgcItWriteData = 0x37;
constexpr std::uint32_t AgcItEventWrite = 0x46;
constexpr std::uint32_t AgcItReleaseMem = 0x49;
constexpr std::uint32_t AgcItDmaData = 0x50;
constexpr std::uint32_t AgcItSetContextReg = 0x69;
constexpr std::uint32_t AgcItSetShReg = 0x76;
constexpr std::uint32_t AgcItSetUconfigReg = 0x79;
constexpr std::uint32_t AgcItGetLodStats = 0x8e;
constexpr std::uint32_t AgcRZero = 0x00;
constexpr std::uint32_t AgcRDrawIndexAuto = 0x04;
constexpr std::uint32_t AgcRDrawReset = 0x05;
constexpr std::uint32_t AgcRWaitFlipDone = 0x06;
constexpr std::uint32_t AgcRAcbReset = 0x09;
constexpr std::uint32_t AgcRWaitMem32 = 0x0a;
constexpr std::uint32_t AgcRPopMarker = 0x0c;
constexpr std::uint32_t AgcRShRegsIndirect = 0x11;
constexpr std::uint32_t AgcRCxRegsIndirect = 0x12;
constexpr std::uint32_t AgcRUcRegsIndirect = 0x13;
constexpr std::uint32_t AgcRAcquireMem = 0x14;
constexpr std::uint32_t AgcRWriteData = 0x15;
constexpr std::uint32_t AgcRWaitMem64 = 0x16;
constexpr std::uint32_t AgcRFlip = 0x17;
constexpr std::uint32_t AgcRReleaseMem = 0x18;
constexpr std::uint32_t AgcRDmaData = 0x19;
constexpr std::uint32_t AgcRIndexCount = 0x1c;
constexpr std::uint32_t AgcCbSetShRegisterRangeMarker = 0x6875000d;
constexpr std::uint32_t AgcCbTargetMask = 0x8e;
constexpr std::uint32_t AgcCbBlend0Control = 0x1e0;
constexpr std::uint32_t AgcSpiPsInputCntl0 = 0x191;
constexpr std::uint32_t AgcSpiPsInputEna = 0x1b3;
constexpr std::uint32_t AgcSpiPsInputAddr = 0x1b4;
constexpr std::uint32_t AgcSpiPsInControl = 0x1b6;
constexpr std::uint32_t AgcSpiShaderColFormat = 0x1c5;
constexpr std::uint32_t AgcDbShaderControl = 0x203;
constexpr std::uint32_t AgcCbColor0Base = 0x318;
constexpr std::uint32_t AgcCbColorRegisterStride = 15;
constexpr std::uint32_t AgcCbColor0Info = 0x31c;
constexpr std::uint32_t AgcCbColor0BaseExt = 0x390;
constexpr std::uint32_t AgcCbColor0Attrib2 = 0x3b0;
constexpr std::uint32_t AgcCbColor0Attrib3 = 0x3b8;

std::uint32_t AgcPm4(const std::uint32_t length_dwords,
                     const std::uint32_t opcode,
                     const std::uint32_t packet_register) {
    return UINT32_C(0xc0000000) |
           (((length_dwords - 2u) & 0x3fffu) << 16u) |
           ((opcode & 0xffu) << 8u) |
           ((packet_register & 0x3fu) << 2u);
}

bool TryAllocateAgcCommandDwords(
    const std::uint64_t command_buffer,
    const std::uint32_t dword_count,
    std::uint64_t& command_address) {
    command_address = 0;
    std::uint64_t cursor_up{};
    std::uint64_t cursor_down{};
    std::uint32_t reserved_dwords{};
    if (command_buffer == 0 || dword_count == 0 ||
        !TryReadGuestValue(
            command_buffer + AgcCommandBufferCursorUpOffset, cursor_up) ||
        !TryReadGuestValue(
            command_buffer + AgcCommandBufferCursorDownOffset, cursor_down) ||
        !TryReadGuestValue(
            command_buffer + AgcCommandBufferReservedDwOffset,
            reserved_dwords) ||
        cursor_down < cursor_up) {
        return false;
    }
    const auto available_dwords =
        (cursor_down - cursor_up) / sizeof(std::uint32_t);
    if (available_dwords <= reserved_dwords ||
        dword_count > available_dwords - reserved_dwords ||
        cursor_up > std::numeric_limits<std::uint64_t>::max() -
                        static_cast<std::uint64_t>(dword_count) *
                            sizeof(std::uint32_t)) {
        return false;
    }
    const auto next_cursor =
        cursor_up + static_cast<std::uint64_t>(dword_count) *
                        sizeof(std::uint32_t);
    if (!TryWriteGuestValue(
            command_buffer + AgcCommandBufferCursorUpOffset,
            next_cursor)) {
        return false;
    }
    command_address = cursor_up;
    return true;
}

bool TryAgcResetDrawQueue(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    if (request.integer_arguments[0] == 0 ||
        request.integer_arguments[1] != 0x3ff ||
        request.integer_arguments[2] != 0) {
        result = 0;
        return true;
    }
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(2, AgcItNop, AgcRDrawReset), 0};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcType2Packet(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 1, command) ||
        !TryWriteGuestValue(command, UINT32_C(0x80000000))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcNop(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto dword_count =
        static_cast<std::uint32_t>(request.integer_arguments[1]);
    std::uint64_t command{};
    if (dword_count < 2 || dword_count > 0x4001 ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], dword_count, command) ||
        !TryWriteGuestValue(
            command, AgcPm4(dword_count, AgcItNop, AgcRZero))) {
        result = 0;
        return true;
    }
    const std::uint32_t zero{};
    for (std::uint32_t index = 1; index < dword_count; ++index) {
        if (!TryWriteGuestValue(
                command + static_cast<std::uint64_t>(index) * 4u,
                zero)) {
            result = 0;
            return true;
        }
    }
    result = command;
    return true;
}

bool TryAgcEventWrite(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto event_type =
        static_cast<std::uint32_t>(request.integer_arguments[1] & 0xffu);
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(2, AgcItEventWrite, 0), event_type};
    if (event_type > 0x3f || request.integer_arguments[2] != 0 ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcAcquireMem(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto engine =
        static_cast<std::uint32_t>(request.integer_arguments[1] & 0xffu);
    const auto size_bytes = request.integer_arguments[5];
    const auto base_address = request.integer_arguments[4];
    std::uint32_t poll_cycles{};
    std::uint64_t command{};
    const bool no_size =
        size_bytes == std::numeric_limits<std::uint64_t>::max();
    if (engine > 1 || (!no_size && (size_bytes & 0xffu) != 0) ||
        (!no_size && (size_bytes >> 40u) != 0) ||
        (base_address & 0xffu) != 0 ||
        (base_address >> 40u) != 0 ||
        !TryReadGuestValue(
            request.guest_stack + 8u, poll_cycles)) {
        result = 0;
        return true;
    }
    const std::uint32_t words[]{
        AgcPm4(8, AgcItNop, AgcRAcquireMem),
        (engine << 31u) |
            static_cast<std::uint32_t>(request.integer_arguments[2]),
        no_size ? 0u : static_cast<std::uint32_t>(size_bytes >> 8u),
        0,
        static_cast<std::uint32_t>(base_address >> 8u),
        0,
        poll_cycles / 40u,
        static_cast<std::uint32_t>(request.integer_arguments[3])};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 8, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcReleaseMem(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::array<std::uint64_t, 6> stack{};
    if (!TryReadGuestBytes(
            request.guest_stack + 8u, stack.data(),
            sizeof(stack))) {
        result = 0;
        return true;
    }
    const auto data_selection =
        static_cast<std::uint32_t>(stack[0] & 0xffu);
    const auto data = stack[1];
    const auto gds_offset =
        static_cast<std::uint32_t>(stack[2] & 0xffffu);
    const auto gds_size =
        static_cast<std::uint32_t>(stack[3] & 0xffffu);
    const auto interrupt =
        static_cast<std::uint32_t>(stack[4] & 0xffu);
    const auto destination =
        static_cast<std::uint32_t>(request.integer_arguments[3] & 0xffu);
    if (destination > 1 || data_selection > 3 ||
        gds_offset != 0 || gds_size > 2 || interrupt > 3) {
        result = 0;
        return true;
    }
    const auto destination_address = request.integer_arguments[5];
    const std::uint32_t words[]{
        AgcPm4(8, AgcItNop, AgcRReleaseMem),
        static_cast<std::uint32_t>(
            request.integer_arguments[1] & 0xffu) |
            (static_cast<std::uint32_t>(
                 request.integer_arguments[4] & 0xffu) << 8u),
        static_cast<std::uint32_t>(
            request.integer_arguments[2] & 0xffffu) |
            (data_selection << 16u) | (interrupt << 24u),
        static_cast<std::uint32_t>(destination_address),
        static_cast<std::uint32_t>(destination_address >> 32u),
        static_cast<std::uint32_t>(data),
        static_cast<std::uint32_t>(data >> 32u),
        static_cast<std::uint32_t>(stack[5])};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 8, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcWaitRegMem(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::array<std::uint64_t, 3> stack{};
    if (!TryReadGuestBytes(
            request.guest_stack + 8u, stack.data(),
            sizeof(stack))) {
        result = 0;
        return true;
    }
    const auto size =
        static_cast<std::uint32_t>(request.integer_arguments[1] & 0xffu);
    const auto compare =
        static_cast<std::uint32_t>(request.integer_arguments[2] & 0xffu);
    const auto operation =
        static_cast<std::uint32_t>(request.integer_arguments[3] & 0xffu);
    const auto cache =
        static_cast<std::uint32_t>(request.integer_arguments[4] & 0xffu);
    const auto address = request.integer_arguments[5];
    const auto reference = stack[0];
    const auto mask = stack[1];
    const auto poll_cycles = static_cast<std::uint32_t>(stack[2]);
    if (size > 1 || compare > 7 || operation > 4 || cache > 3) {
        result = 0;
        return true;
    }
    const auto control = size == 0
        ? UINT32_C(0x10) | compare |
              ((operation & 3u) << 8u) |
              ((operation & 0xcu) << 4u) | (cache << 25u)
        : UINT32_C(0x10) | compare |
              ((operation & 1u) << 8u) |
              ((operation & 6u) << 5u) | (cache << 25u);
    const auto poll = std::min(poll_cycles >> 4u, 0xffffu);
    std::array<std::uint32_t, 9> words{
        AgcPm4(
            size == 0 ? 7u : 9u, AgcItNop,
            size == 0 ? AgcRWaitMem32 : AgcRWaitMem64),
        static_cast<std::uint32_t>(address) &
            (size == 0 ? ~3u : ~7u),
        static_cast<std::uint32_t>(address >> 32u) & 0x3ffffu,
        static_cast<std::uint32_t>(mask)};
    if (size == 0) {
        words[4] = static_cast<std::uint32_t>(reference);
        words[5] = control;
        words[6] = poll;
    } else {
        words[4] = static_cast<std::uint32_t>(mask >> 32u);
        words[5] = static_cast<std::uint32_t>(reference);
        words[6] = static_cast<std::uint32_t>(reference >> 32u);
        words[7] = control;
        words[8] = poll;
    }
    const auto dword_count = size == 0 ? 7u : 9u;
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], dword_count, command) ||
        !TryWriteGuestBytes(
            command, words.data(),
            static_cast<std::size_t>(dword_count) * 4u)) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDmaData(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::array<std::uint64_t, 6> stack{};
    if (!TryReadGuestBytes(
            request.guest_stack + 8u, stack.data(),
            sizeof(stack))) {
        result = 0;
        return true;
    }
    const auto byte_count = static_cast<std::uint32_t>(stack[2]);
    if (byte_count == 0 || (byte_count & 3u) != 0) {
        result = 0;
        return true;
    }
    const auto engine = static_cast<std::uint32_t>(
        request.integer_arguments[1] & 0x1u);
    const auto destination = static_cast<std::uint32_t>(
        request.integer_arguments[2] & 0xffu);
    const auto destination_cache_policy =
        static_cast<std::uint32_t>(
            request.integer_arguments[3] & 0x3u);
    const auto destination_address = request.integer_arguments[4];
    const auto source = static_cast<std::uint32_t>(
        request.integer_arguments[5] & 0xffu);
    const auto source_cache_policy =
        static_cast<std::uint32_t>(stack[0] & 0x3u);
    const auto source_address = stack[1];
    const std::uint32_t words[]{
        AgcPm4(7, AgcItDmaData, 0),
        engine |
            (source_cache_policy << 13u) |
            ((destination & 0x3u) << 20u) |
            (destination_cache_policy << 25u) |
            ((source & 0x3u) << 29u) |
            ((static_cast<std::uint32_t>(stack[5]) & 0x1u) << 31u),
        static_cast<std::uint32_t>(source_address),
        static_cast<std::uint32_t>(source_address >> 32u),
        static_cast<std::uint32_t>(destination_address),
        static_cast<std::uint32_t>(destination_address >> 32u),
        (byte_count & 0x03ffffffu) |
            ((source & 0x4u) << 24u) |
            ((destination & 0x4u) << 25u) |
            ((source & 0x8u) << 25u) |
            ((destination & 0x8u) << 26u) |
            ((static_cast<std::uint32_t>(stack[3]) & 0x1u) << 30u) |
            ((static_cast<std::uint32_t>(stack[4]) & 0x1u) << 31u)};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 7, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcPatchDmaDataAddress(
    const Lsx4::Translation::HleBridgeRequest& request,
    const bool source,
    std::uint64_t& result) {
    const auto command_or_field = request.integer_arguments[0];
    const auto value = request.integer_arguments[1];
    std::uint32_t header{};
    if (TryReadGuestValue(command_or_field, header)) {
        const auto opcode = (header >> 8u) & 0xffu;
        const auto packet_register = (header >> 2u) & 0x3fu;
        if ((header & UINT32_C(0xc0000000)) ==
                UINT32_C(0xc0000000) &&
            opcode == AgcItDmaData) {
            result = TryWriteGuestValue(
                         command_or_field +
                             (source ? 8u : 16u),
                         value)
                ? 0
                : OrbisError(UINT32_C(0x80020101));
            return true;
        }
        if ((header & UINT32_C(0xc0000000)) ==
                UINT32_C(0xc0000000) &&
            opcode == AgcItNop &&
            packet_register == AgcRDmaData) {
            const auto packet_length =
                ((header >> 16u) & 0x3fffu) + 2u;
            const auto field_offset =
                packet_length == 7u
                ? (source ? 12u : 4u)
                : (source ? 24u : 16u);
            result = TryWriteGuestValue(
                         command_or_field + field_offset, value)
                ? 0
                : OrbisError(UINT32_C(0x80020101));
            return true;
        }
    }

    result = command_or_field != 0 &&
                     TryWriteGuestValue(command_or_field, value)
        ? 0
        : OrbisError(command_or_field == 0
                         ? UINT32_C(0x80020003)
                         : UINT32_C(0x80020101));
    return true;
}

bool TryAgcPatchWaitRegMemAddress(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto command_or_field = request.integer_arguments[0];
    const auto address = request.integer_arguments[1];
    std::uint32_t header{};
    if (TryReadGuestValue(command_or_field, header) &&
        (header & UINT32_C(0xc0000000)) == UINT32_C(0xc0000000)) {
        const auto opcode = (header >> 8u) & 0xffu;
        const auto packet_register = (header >> 2u) & 0x3fu;
        if (opcode == AgcItWaitRegMem) {
            result = TryWriteGuestValue(command_or_field + 8u, address)
                ? 0
                : OrbisError(UINT32_C(0x80020101));
            return true;
        }
        if (opcode == AgcItNop &&
            (packet_register == AgcRWaitMem32 ||
             packet_register == AgcRWaitMem64)) {
            const auto low_alignment_mask =
                packet_register == AgcRWaitMem32 ? ~3u : ~7u;
            const std::array<std::uint32_t, 2> words{
                static_cast<std::uint32_t>(address) & low_alignment_mask,
                static_cast<std::uint32_t>(address >> 32u) & 0x3ffffu};
            result = TryWriteGuestBytes(
                         command_or_field + 4u,
                         words.data(),
                         sizeof(words))
                ? 0
                : OrbisError(UINT32_C(0x80020101));
            return true;
        }
    }

    result = command_or_field != 0 &&
                     TryWriteGuestValue(command_or_field, address)
        ? 0
        : OrbisError(command_or_field == 0
                         ? UINT32_C(0x80020003)
                         : UINT32_C(0x80020101));
    return true;
}

bool TryAgcSetRegistersIndirect(
    const Lsx4::Translation::HleBridgeRequest& request,
    const std::uint32_t packet_register,
    std::uint64_t& result) {
#ifdef __ANDROID__
    if (packet_register == AgcRCxRegsIndirect) {
        static std::atomic<std::uint32_t> cx_table_dumps{};
        const auto dump_index =
            cx_table_dumps.fetch_add(1, std::memory_order_relaxed);
        if (dump_index < 2u) {
            const auto registers_address = request.integer_arguments[1];
            const auto register_count =
                static_cast<std::uint32_t>(
                    request.integer_arguments[2]);
            std::uint64_t return_address{};
            (void)TryReadGuestValue(
                request.guest_stack, return_address);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cx_table[%u] address=0x%llx count=%u "
                "stack=0x%llx return=0x%llx",
                dump_index,
                static_cast<unsigned long long>(
                    registers_address),
                register_count,
                static_cast<unsigned long long>(
                    request.guest_stack),
                static_cast<unsigned long long>(
                    return_address));
            for (std::uint32_t index = 0;
                 index < register_count; ++index) {
                std::uint64_t raw{};
                if (!TryReadGuestValue(
                        registers_address +
                            static_cast<std::uint64_t>(index) * 8u,
                        raw)) {
                    break;
                }
                const auto low = static_cast<std::uint32_t>(raw);
                const auto high =
                    static_cast<std::uint32_t>(raw >> 32u);
                if (index < 24u || low >= 0x300u ||
                    high >= 0x300u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "cx_table[%u] entry=%u low=%08x "
                        "high=%08x",
                        dump_index, index, low, high);
                }
            }
        }
    }
#endif
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 4, command)) {
        result = 0;
        return true;
    }
    const std::uint32_t words[]{
        AgcPm4(4, AgcItNop, packet_register),
        static_cast<std::uint32_t>(request.integer_arguments[2]),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        static_cast<std::uint32_t>(request.integer_arguments[1] >> 32u),
    };
    result = TryWriteGuestBytes(command, words, sizeof(words))
        ? command
        : 0;
    return true;
}

bool TryAgcPatchIndirectAddress(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto command = request.integer_arguments[0];
    const auto registers = request.integer_arguments[1];
    if (command == 0 || registers == 0 ||
        !TryWriteGuestValue(
            command + 8, static_cast<std::uint32_t>(registers)) ||
        !TryWriteGuestValue(
            command + 12,
            static_cast<std::uint32_t>(registers >> 32u))) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    result = 0;
    return true;
}

bool TryAgcPatchIndirectCount(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto command = request.integer_arguments[0];
    std::uint32_t current_count{};
    if (command == 0 ||
        !TryReadGuestValue(command + 4, current_count) ||
        !TryWriteGuestValue(
            command + 4,
            current_count +
                static_cast<std::uint32_t>(
                    request.integer_arguments[1]))) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    result = 0;
    return true;
}

bool TryAgcSetShRegisterRange(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto command_buffer = request.integer_arguments[0];
    const auto offset =
        static_cast<std::uint32_t>(request.integer_arguments[1]);
    const auto values = request.integer_arguments[2];
    const auto value_count =
        static_cast<std::uint32_t>(request.integer_arguments[3]);
    if (command_buffer == 0 || offset == 0 || offset > 0x3ff ||
        value_count == 0 ||
        value_count >
            (std::numeric_limits<std::uint32_t>::max() - 2u)) {
        result = 0;
        return true;
    }
    std::uint64_t marker{};
    std::uint64_t command{};
    const std::uint32_t marker_words[]{
        AgcPm4(2, AgcItNop, AgcRZero),
        AgcCbSetShRegisterRangeMarker};
    if (!TryAllocateAgcCommandDwords(command_buffer, 2, marker) ||
        !TryWriteGuestBytes(marker, marker_words, sizeof(marker_words)) ||
        !TryAllocateAgcCommandDwords(
            command_buffer, value_count + 2u, command) ||
        !TryWriteGuestValue(
            command, AgcPm4(
                         value_count + 2u, AgcItSetShReg, 0)) ||
        !TryWriteGuestValue(command + 4, offset)) {
        result = 0;
        return true;
    }
    for (std::uint32_t index = 0; index < value_count; ++index) {
        std::uint32_t value{};
        if ((values != 0 &&
             !TryReadGuestValue(
                 values + static_cast<std::uint64_t>(index) * 4u,
                 value)) ||
            !TryWriteGuestValue(
                command + 8u +
                    static_cast<std::uint64_t>(index) * 4u,
                value)) {
            result = 0;
            return true;
        }
    }
    result = command;
    return true;
}

bool TryCopyAgcShaderRegister(
    const std::uint64_t source,
    const std::uint64_t destination) {
    std::array<std::uint32_t, 2> entry{};
    return TryReadGuestBytes(
               source, entry.data(), sizeof(entry)) &&
           TryWriteGuestBytes(
               destination, entry.data(), sizeof(entry));
}

bool TryAgcCreatePrimState(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint64_t ShaderSpecialsOffset = 0x28;
    constexpr std::uint64_t ShaderTypeOffset = 0x5a;
    constexpr std::uint64_t SpecialGeCntlOffset = 0x00;
    constexpr std::uint64_t SpecialVgtShaderStagesEnOffset = 0x08;
    constexpr std::uint64_t SpecialVgtGsOutPrimTypeOffset = 0x20;
    constexpr std::uint64_t SpecialGeUserVgprEnOffset = 0x28;
    constexpr std::uint32_t VgtPrimitiveType = 0x242;

    const auto cx_registers = request.integer_arguments[0];
    const auto uc_registers = request.integer_arguments[1];
    const auto hull_shader = request.integer_arguments[2];
    const auto geometry_shader = request.integer_arguments[3];
    const auto primitive_type =
        static_cast<std::uint32_t>(request.integer_arguments[4]);
    if (cx_registers == 0 || uc_registers == 0 ||
        hull_shader != 0 || geometry_shader == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    std::uint8_t shader_type{};
    std::uint64_t specials{};
    if (!TryReadGuestValue(
            geometry_shader + ShaderTypeOffset, shader_type) ||
        (shader_type != 2 && shader_type != 6) ||
        !TryReadGuestValue(
            geometry_shader + ShaderSpecialsOffset, specials) ||
        specials == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    if (!TryCopyAgcShaderRegister(
            specials + SpecialVgtShaderStagesEnOffset,
            cx_registers) ||
        !TryCopyAgcShaderRegister(
            specials + SpecialVgtGsOutPrimTypeOffset,
            cx_registers + 8u) ||
        !TryCopyAgcShaderRegister(
            specials + SpecialGeCntlOffset,
            uc_registers) ||
        !TryCopyAgcShaderRegister(
            specials + SpecialGeUserVgprEnOffset,
            uc_registers + 8u) ||
        !TryWriteGuestValue(
            uc_registers + 16u, VgtPrimitiveType) ||
        !TryWriteGuestValue(
            uc_registers + 20u, primitive_type)) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    result = 0;
    return true;
}

bool TryAgcCreateInterpolantMapping(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint64_t ShaderInputSemanticsOffset = 0x30;
    constexpr std::uint64_t ShaderOutputSemanticsOffset = 0x38;
    constexpr std::uint64_t ShaderNumOutputSemanticsOffset = 0x56;
    constexpr std::uint32_t SpiPsInputCntl0 = 0x191;

    const auto registers = request.integer_arguments[0];
    const auto geometry_shader = request.integer_arguments[1];
    const auto pixel_shader = request.integer_arguments[2];
    if (registers == 0 || geometry_shader == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    std::uint64_t output_semantics{};
    std::uint32_t output_count{};
    if (!TryReadGuestValue(
            geometry_shader + ShaderOutputSemanticsOffset,
            output_semantics) ||
        !TryReadGuestValue(
            geometry_shader + ShaderNumOutputSemanticsOffset,
            output_count)) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    std::uint64_t input_semantics{};
    if (pixel_shader != 0 &&
        !TryReadGuestValue(
            pixel_shader + ShaderInputSemanticsOffset,
            input_semantics)) {
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }
    for (std::uint32_t index = 0; index < 32; ++index) {
        std::uint32_t value{};
        if (index < output_count && output_semantics != 0) {
            bool flat{};
            std::uint32_t input_semantic{};
            if (pixel_shader != 0 && input_semantics != 0 &&
                TryReadGuestValue(
                    input_semantics +
                        static_cast<std::uint64_t>(index) * 4u,
                    input_semantic)) {
                flat = ((input_semantic >> 22u) & 1u) != 0;
            }
            value = index | (flat ? 0x400u : 0u);
        }
        const std::array<std::uint32_t, 2> entry{
            SpiPsInputCntl0 + index, value};
        if (!TryWriteGuestBytes(
                registers +
                    static_cast<std::uint64_t>(index) * 8u,
                entry.data(), sizeof(entry))) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
    }
    result = 0;
    return true;
}

bool TryAgcGetDataPacketPayload(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto output = request.integer_arguments[0];
    const auto command = request.integer_arguments[1];
    const auto type =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    if (output == 0 || command == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }
    auto payload = command + 8u;
    std::uint32_t header{};
    if (type == 0) {
        if (!TryReadGuestValue(command, header)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        payload = (header & UINT32_C(0x3fff0000)) ==
                          UINT32_C(0x3fff0000)
            ? 0
            : command + 4u;
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        static std::atomic<std::uint64_t> payload_logs{};
        const auto log_index =
            payload_logs.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (log_index <= 8u ||
            (log_index & (log_index - 1u)) == 0u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-AGC",
                "payload count=%llu command=0x%llx type=%u header=0x%08x "
                "output=0x%llx payload=0x%llx",
                static_cast<unsigned long long>(log_index),
                static_cast<unsigned long long>(command),
                type, header,
                static_cast<unsigned long long>(output),
                static_cast<unsigned long long>(payload));
        }
    }
#endif
    result = TryWriteGuestValue(output, payload)
        ? 0
        : OrbisError(UINT32_C(0x80020101));
    return true;
}

bool TryAgcWaitUntilSafeForRendering(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(7, AgcItNop, AgcRWaitFlipDone),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        static_cast<std::uint32_t>(request.integer_arguments[2]),
        0, 0, 0, 0};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 7, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcSetFlip(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const auto flip_argument = request.integer_arguments[4];
    const std::uint32_t words[]{
        AgcPm4(6, AgcItNop, AgcRFlip),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        static_cast<std::uint32_t>(request.integer_arguments[2]),
        static_cast<std::uint32_t>(request.integer_arguments[3]),
        static_cast<std::uint32_t>(flip_argument),
        static_cast<std::uint32_t>(flip_argument >> 32u)};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 6, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcSetIndexSize(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    if (request.integer_arguments[0] == 0 ||
        request.integer_arguments[2] != 0) {
        result = 0;
        return true;
    }
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(2, AgcItIndexType, 0),
        static_cast<std::uint32_t>(
            request.integer_arguments[1] & 0xffu)};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcSetIndexCount(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(2, AgcItNop, AgcRIndexCount),
        static_cast<std::uint32_t>(request.integer_arguments[1])};
    if (request.integer_arguments[0] == 0 ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcSetNumInstances(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const std::uint32_t words[]{
        AgcPm4(2, AgcItNumInstances, 0),
        static_cast<std::uint32_t>(request.integer_arguments[1])};
    if (request.integer_arguments[0] == 0 ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcSetIndexBuffer(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const auto index_buffer = request.integer_arguments[1];
    const std::uint32_t words[]{
        AgcPm4(3, AgcItIndexBase, 0),
        static_cast<std::uint32_t>(index_buffer),
        static_cast<std::uint32_t>(index_buffer >> 32u),
        AgcPm4(2, AgcItIndexBufferSize, 0),
        static_cast<std::uint32_t>(request.integer_arguments[2])};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 5, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDrawIndexOffset(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t command{};
    const auto index_count =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    const std::uint32_t words[]{
        AgcPm4(5, AgcItDrawIndexOffset2, 0),
        index_count,
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        index_count,
        static_cast<std::uint32_t>(
            request.integer_arguments[3] & UINT32_C(0xe0000001))};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 5, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcCbDispatch(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto modifier =
        static_cast<std::uint32_t>(request.integer_arguments[4]);
    const std::uint32_t words[]{
        AgcPm4(5, AgcItDispatchDirect, 0),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        static_cast<std::uint32_t>(request.integer_arguments[2]),
        static_cast<std::uint32_t>(request.integer_arguments[3]),
        (modifier & UINT32_C(0xa038)) | UINT32_C(0x41)};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 5, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcCbSetShRegistersDirect(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto command_buffer = request.integer_arguments[0];
    const auto records_address = request.integer_arguments[1];
    const auto record_count =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    if (command_buffer == 0 || records_address == 0 ||
        record_count == 0 || record_count > 4096) {
        result = 0;
        return true;
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> records(
        record_count);
    for (std::uint32_t index = 0; index < record_count; ++index) {
        const auto record =
            records_address + static_cast<std::uint64_t>(index) * 8u;
        if (!TryReadGuestValue(record, records[index].first) ||
            !TryReadGuestValue(record + 4u, records[index].second)) {
            result = 0;
            return true;
        }
    }
    std::ranges::sort(
        records, {}, &std::pair<std::uint32_t, std::uint32_t>::first);
    std::uint64_t first_command{};
    std::size_t begin{};
    while (begin < records.size()) {
        auto end = begin + 1u;
        while (end < records.size() &&
               records[end].first == records[end - 1u].first + 1u) {
            ++end;
        }
        const auto value_count =
            static_cast<std::uint32_t>(end - begin);
        const auto dword_count = value_count + 2u;
        std::uint64_t command{};
        if (!TryAllocateAgcCommandDwords(
                command_buffer, dword_count, command) ||
            !TryWriteGuestValue(
                command,
                AgcPm4(dword_count, AgcItSetShReg, 0)) ||
            !TryWriteGuestValue(
                command + 4u, records[begin].first & 0xffffu)) {
            result = 0;
            return true;
        }
        if (first_command == 0) {
            first_command = command;
        }
        for (std::size_t index = begin; index < end; ++index) {
            if (!TryWriteGuestValue(
                    command + 8u +
                        static_cast<std::uint64_t>(index - begin) * 4u,
                    records[index].second)) {
                result = 0;
                return true;
            }
        }
        begin = end;
    }
    result = first_command;
    return true;
}

bool TryAgcAcbResetQueue(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const std::uint32_t words[]{
        AgcPm4(2, AgcItNop, AgcRAcbReset), 0};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcAcbEventWrite(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto event_type =
        static_cast<std::uint32_t>(request.integer_arguments[1] & 0xffu);
    const auto event_address = request.integer_arguments[2];
    const bool has_address = (event_type & ~1u) == 0x38u;
    const auto dword_count = has_address ? 4u : 2u;
    std::array<std::uint32_t, 4> words{
        AgcPm4(dword_count, AgcItEventWrite, 0),
        has_address ? event_type | 0x100u : event_type & 0x3fu,
        static_cast<std::uint32_t>(event_address) & ~7u,
        static_cast<std::uint32_t>(event_address >> 32u)};
    std::uint64_t command{};
    if (event_type >= 0x40u ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], dword_count, command) ||
        !TryWriteGuestBytes(
            command, words.data(),
            static_cast<std::size_t>(dword_count) * 4u)) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcAcbAcquireMem(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto gcr_control =
        static_cast<std::uint32_t>(request.integer_arguments[1]);
    const auto base_address = request.integer_arguments[2];
    const auto size_bytes = request.integer_arguments[3];
    const auto poll_cycles =
        static_cast<std::uint32_t>(request.integer_arguments[4]);
    const bool no_size =
        size_bytes == std::numeric_limits<std::uint64_t>::max();
    const std::uint32_t words[]{
        AgcPm4(8, AgcItNop, AgcRAcquireMem),
        UINT32_C(0x80000000),
        no_size ? 0u : static_cast<std::uint32_t>(size_bytes >> 8u),
        0,
        static_cast<std::uint32_t>(base_address >> 8u),
        0,
        poll_cycles / 40u,
        gcr_control};
    std::uint64_t command{};
    if ((!no_size && (size_bytes & 0xffu) != 0) ||
        (!no_size && (size_bytes >> 40u) != 0) ||
        (base_address & 0xffu) != 0 ||
        (base_address >> 40u) != 0 ||
        !TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 8, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcAcbWaitRegMem(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto size =
        static_cast<std::uint32_t>(request.integer_arguments[1] & 0xffu);
    const auto compare =
        static_cast<std::uint32_t>(request.integer_arguments[2] & 0xffu);
    const auto cache =
        static_cast<std::uint32_t>(request.integer_arguments[3] & 0xffu);
    const auto address = request.integer_arguments[4];
    const auto reference = request.integer_arguments[5];
    std::uint64_t mask{};
    std::uint32_t poll_cycles{};
    if (size > 1 || compare > 7 || cache > 3 ||
        !TryReadGuestValue(request.guest_stack + 8u, mask) ||
        !TryReadGuestValue(request.guest_stack + 16u, poll_cycles)) {
        result = 0;
        return true;
    }
    const auto control =
        UINT32_C(0x10) | compare | (cache << 25u);
    const auto poll = std::min(poll_cycles >> 4u, 0xffffu);
    std::array<std::uint32_t, 9> words{
        AgcPm4(
            size == 0 ? 7u : 9u, AgcItNop,
            size == 0 ? AgcRWaitMem32 : AgcRWaitMem64),
        static_cast<std::uint32_t>(address) &
            (size == 0 ? ~3u : ~7u),
        static_cast<std::uint32_t>(address >> 32u) & 0x3ffffu,
        static_cast<std::uint32_t>(mask)};
    if (size == 0) {
        words[4] = static_cast<std::uint32_t>(reference);
        words[5] = control;
        words[6] = poll;
    } else {
        words[4] = static_cast<std::uint32_t>(mask >> 32u);
        words[5] = static_cast<std::uint32_t>(reference);
        words[6] = static_cast<std::uint32_t>(reference >> 32u);
        words[7] = control;
        words[8] = poll;
    }
    const auto dword_count = size == 0 ? 7u : 9u;
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], dword_count, command) ||
        !TryWriteGuestBytes(
            command, words.data(),
            static_cast<std::size_t>(dword_count) * 4u)) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcAcbDispatchIndirect(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto arguments_address = request.integer_arguments[1];
    const auto modifier =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    const std::uint32_t words[]{
        AgcPm4(4, AgcItDispatchIndirect, 0),
        static_cast<std::uint32_t>(arguments_address),
        static_cast<std::uint32_t>(arguments_address >> 32u),
        (modifier & UINT32_C(0xa038)) | UINT32_C(0x41)};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 4, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcWriteData(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto destination_address = request.integer_arguments[3];
    const auto data_address = request.integer_arguments[4];
    const auto data_count =
        static_cast<std::uint32_t>(request.integer_arguments[5]);
    std::uint64_t increment_raw{};
    std::uint64_t confirm_raw{};
    if (destination_address == 0 || data_address == 0 ||
        data_count > 0x3ffdu ||
        !TryReadGuestValue(
            request.guest_stack + 8u, increment_raw) ||
        !TryReadGuestValue(
            request.guest_stack + 16u, confirm_raw)) {
        result = 0;
        return true;
    }
    const auto dword_count = data_count + 4u;
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], dword_count, command)) {
        result = 0;
        return true;
    }
    const std::uint32_t prefix[]{
        AgcPm4(dword_count, AgcItNop, AgcRWriteData),
        static_cast<std::uint32_t>(
            request.integer_arguments[1] & 0xffu) |
            (static_cast<std::uint32_t>(
                 request.integer_arguments[2] & 0xffu) << 8u) |
            (static_cast<std::uint32_t>(
                 increment_raw & 0xffu) << 16u) |
            (static_cast<std::uint32_t>(
                 confirm_raw & 0xffu) << 24u),
        static_cast<std::uint32_t>(destination_address),
        static_cast<std::uint32_t>(destination_address >> 32u)};
    if (!TryWriteGuestBytes(command, prefix, sizeof(prefix))) {
        result = 0;
        return true;
    }
    for (std::uint32_t index = 0; index < data_count; ++index) {
        std::uint32_t value{};
        if (!TryReadGuestValue(
                data_address + static_cast<std::uint64_t>(index) * 4u,
                value) ||
            !TryWriteGuestValue(
                command + 16u +
                    static_cast<std::uint64_t>(index) * 4u,
                value)) {
            result = 0;
            return true;
        }
    }
    result = command;
    return true;
}

bool TryAgcDcbDrawIndexAuto(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    if (request.integer_arguments[2] != UINT32_C(0x40000000)) {
        result = 0;
        return true;
    }
    const std::uint32_t words[]{
        AgcPm4(7, AgcItNop, AgcRDrawIndexAuto),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        0, 0, 0, 0, 0};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 7, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDcbSetBaseIndirectArgs(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto base_index =
        static_cast<std::uint32_t>(request.integer_arguments[1]);
    const auto address = request.integer_arguments[2];
    const std::uint32_t words[]{
        AgcPm4(4, AgcItSetBase, 0) | (base_index << 1u),
        1,
        static_cast<std::uint32_t>(address) & ~7u,
        static_cast<std::uint32_t>(address >> 32u)};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 4, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDcbDispatchIndirect(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto modifier =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    const std::uint32_t words[]{
        AgcPm4(3, AgcItDispatchIndirect, 0),
        static_cast<std::uint32_t>(request.integer_arguments[1]),
        (modifier & UINT32_C(0xa038)) | UINT32_C(0x41)};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 3, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDcbPopMarker(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const std::uint32_t words[]{
        AgcPm4(2, AgcItNop, AgcRPopMarker), 0};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 2, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

bool TryAgcDcbGetLodStats(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    std::uint64_t enable_raw{};
    std::uint64_t counter_select_raw{};
    if (!TryReadGuestValue(
            request.guest_stack + 8u, enable_raw) ||
        !TryReadGuestValue(
            request.guest_stack + 16u, counter_select_raw)) {
        result = 0;
        return true;
    }
    const auto destination_address = request.integer_arguments[2];
    const auto packet_control =
        (static_cast<std::uint32_t>(
             request.integer_arguments[1] & 3u) << 28u) |
        (static_cast<std::uint32_t>(enable_raw & 1u) << 19u) |
        (static_cast<std::uint32_t>(
             request.integer_arguments[5] & 1u) << 18u) |
        (static_cast<std::uint32_t>(
             request.integer_arguments[4] & 0xffu) << 10u) |
        (static_cast<std::uint32_t>(
             counter_select_raw & 0xffu) << 2u);
    const std::uint32_t words[]{
        AgcPm4(5, AgcItGetLodStats, 0),
        static_cast<std::uint32_t>(request.integer_arguments[3]),
        static_cast<std::uint32_t>(destination_address) & ~0x3fu,
        static_cast<std::uint32_t>(destination_address >> 32u),
        packet_control};
    std::uint64_t command{};
    if (!TryAllocateAgcCommandDwords(
            request.integer_arguments[0], 5, command) ||
        !TryWriteGuestBytes(command, words, sizeof(words))) {
        result = 0;
        return true;
    }
    result = command;
    return true;
}

struct AgcCpuVertex {
    float x{};
    float y{};
    float z{};
    float u{};
    float v{};
    std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f};
};

struct AgcCpuDecodedTexture {
    std::uint64_t address{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t format{};
    std::uint32_t tile_mode{};
    std::uint64_t source_probe{};
    std::uint64_t source_probe_serial{};
    std::uint64_t signature{};
    std::shared_ptr<std::vector<std::uint8_t>> rgba;
};

std::uint64_t AgcTextureSignature(
    const std::vector<std::uint8_t>& rgba) {
    std::uint64_t signature = UINT64_C(1469598103934665603);
    const auto stride = std::max<std::size_t>(
        rgba.size() / 8192u, 1u);
    for (std::size_t offset = 0; offset < rgba.size();
         offset += stride) {
        signature ^= rgba[offset];
        signature *= UINT64_C(1099511628211);
    }
    signature ^= rgba.size();
    signature *= UINT64_C(1099511628211);
    return signature;
}

bool TryResolveAgcTextureDescriptor(
    AgcDiagnosticDraw& draw,
    const std::span<const AgcDiagnosticDraw> preceding_draws) {
    struct Candidate {
        std::uint64_t address{};
        std::uint32_t width{};
        std::uint32_t height{};
        std::uint32_t tile_mode{};
        std::uint32_t format{};
        std::uint64_t score{};
    };
    Candidate best{};
    Candidate fallback{};
    std::vector<std::uint64_t> tables;
    tables.reserve(64u);
    const auto append_table =
        [&](const std::uint64_t address) {
            if (address < UINT64_C(0x10000) ||
                (address & 3u) != 0u ||
                tables.size() >= 64u ||
                std::ranges::contains(tables, address)) {
                return;
            }
            tables.push_back(address);
        };
    const auto append_user_data =
        [&](const auto& user_data) {
            for (std::size_t word = 0;
                 word + 1u < user_data.size(); ++word) {
                append_table(
                    static_cast<std::uint64_t>(user_data[word]) |
                    (static_cast<std::uint64_t>(
                         user_data[word + 1u] & UINT32_C(0xffff))
                     << 32u));
            }
        };
    append_user_data(draw.sh_ps_user_data);
    append_user_data(draw.sh_export_user_data);
    append_user_data(draw.sh_gs_user_data);
    append_user_data(draw.sh_es_user_data);
    append_user_data(draw.sh_vs_user_data);
    if (draw.texture_address != 0u) {
        append_table(draw.texture_address >> 8u);
    }

    constexpr std::size_t TableBytes = 0x400u;
    std::array<std::uint32_t, TableBytes / sizeof(std::uint32_t)>
        words{};
    std::array<std::uint8_t, 4> source_probe{};
    for (std::size_t table_index = 0;
         table_index < tables.size() && table_index < 64u;
         ++table_index) {
        if (!TryReadGuestBytes(
                tables[table_index], words.data(), sizeof(words))) {
            continue;
        }
        for (std::size_t word = 0;
             word + 3u < words.size(); ++word) {
            const auto address =
                ((static_cast<std::uint64_t>(
                      words[word + 1u] & UINT32_C(0xff))
                  << 32u) |
                 words[word]) << 8u;
            const auto width =
                (((words[word + 1u] >> 30u) & 3u) |
                 ((words[word + 2u] & UINT32_C(0x3fff)) << 2u)) +
                1u;
            const auto height =
                ((words[word + 2u] >> 14u) & UINT32_C(0xffff)) +
                1u;
            const auto format =
                (words[word + 1u] >> 20u) & UINT32_C(0x1ff);
            const auto tile_mode =
                (words[word + 3u] >> 20u) & UINT32_C(0x1f);
            const bool supported_format =
                format == UINT32_C(0x1) ||
                format == UINT32_C(0x38) ||
                format == UINT32_C(0xad) ||
                format == UINT32_C(0xb5);
            const bool supported_hdr_surface =
                format == UINT32_C(0x47) &&
                tile_mode == UINT32_C(27);
            const auto preceding_target =
                std::ranges::find_if(
                    preceding_draws,
                    [&](const AgcDiagnosticDraw& preceding) {
                        return address ==
                            preceding.render_target_address;
                    });
            const bool matches_preceding_target =
                preceding_target != preceding_draws.end();
            const bool plausible_dimensions =
                (width == draw.render_target_width &&
                 height == draw.render_target_height) ||
                (width * 2u == draw.render_target_width &&
                 height * 2u == draw.render_target_height) ||
                (width == draw.render_target_width * 2u &&
                 height == draw.render_target_height * 2u);
            if (address != 0u &&
                address != draw.render_target_address &&
                width >= 16u && height >= 16u &&
                width <= 8192u && height <= 8192u &&
                plausible_dimensions &&
                TryReadGuestBytes(
                    address, source_probe.data(),
                    source_probe.size())) {
                auto fallback_score =
                    static_cast<std::uint64_t>(width) * height;
                fallback_score += UINT64_C(1) << 50u;
                if (format != 0u) {
                    fallback_score += UINT64_C(1) << 48u;
                }
                if (fallback_score > fallback.score) {
                    fallback = {
                        .address = address,
                        .width = width,
                        .height = height,
                        .tile_mode = tile_mode,
                        .format = format,
                        .score = fallback_score};
                }
            }
            if (address == 0u ||
                address == draw.render_target_address ||
                width < 16u || height < 16u ||
                width > 8192u || height > 8192u ||
                (!matches_preceding_target &&
                 ((!supported_format && !supported_hdr_surface) ||
                  (supported_format &&
                   tile_mode != 0u && tile_mode != 5u) ||
                  (supported_hdr_surface &&
                   tile_mode != UINT32_C(27)) ||
                  !TryReadGuestBytes(
                      address, source_probe.data(),
                      source_probe.size())))) {
                continue;
            }

            std::uint64_t score =
                static_cast<std::uint64_t>(width) * height;
            if (width == draw.render_target_width &&
                height == draw.render_target_height) {
                score += UINT64_C(1) << 50u;
            } else if (
                width * 2u == draw.render_target_width &&
                height * 2u == draw.render_target_height) {
                score += UINT64_C(1) << 49u;
            } else if (
                width == draw.render_target_width * 2u &&
                height == draw.render_target_height * 2u) {
                score += UINT64_C(1) << 48u;
            }
            if (matches_preceding_target) {
                score += UINT64_C(1) << 60u;
                if (width == preceding_target->render_target_width &&
                    height == preceding_target->render_target_height) {
                    score += UINT64_C(1) << 55u;
                }
            }
            if (score > best.score) {
                best = {
                    .address = address,
                    .width = width,
                    .height = height,
                    .tile_mode = tile_mode,
                    .format = format,
                    .score = score};
            }
        }
        for (std::size_t word = 0;
             word + 1u < words.size(); ++word) {
            append_table(
                static_cast<std::uint64_t>(words[word]) |
                (static_cast<std::uint64_t>(
                     words[word + 1u] & UINT32_C(0xffff))
                 << 32u));
        }
    }
    if (best.score == 0u) {
#ifdef __ANDROID__
        if (fallback.score != 0u) {
            static std::atomic<std::uint32_t> fallback_logs{};
            if (fallback_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 8u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "texture resolver fallback target=0x%llx/%ux%u "
                    "texture=0x%llx/%ux%u/f%02x/t%u",
                    static_cast<unsigned long long>(
                        draw.render_target_address),
                    draw.render_target_width,
                    draw.render_target_height,
                    static_cast<unsigned long long>(
                        fallback.address),
                    fallback.width, fallback.height,
                    fallback.format, fallback.tile_mode);
            }
        }
#endif
        return false;
    }
    draw.texture_address = best.address;
    draw.texture_width = best.width;
    draw.texture_height = best.height;
    draw.texture_tile_mode = best.tile_mode;
    draw.texture_format = best.format;
    return true;
}

bool TryAgcTextureSourceProbe(
    const AgcDiagnosticDraw& draw,
    std::uint64_t& signature) {
    std::uint32_t element_width = draw.texture_width;
    std::uint32_t element_height = draw.texture_height;
    std::uint32_t element_bytes{};
    if (draw.texture_format == UINT32_C(0x38)) {
        element_bytes = 4u;
    } else if (draw.texture_format == UINT32_C(0x1)) {
        element_bytes = 1u;
    } else if (draw.texture_format == UINT32_C(0xad) ||
               draw.texture_format == UINT32_C(0xb5)) {
        element_width = (element_width + 3u) / 4u;
        element_height = (element_height + 3u) / 4u;
        element_bytes = 16u;
    } else {
        return false;
    }

    std::uint64_t source_bytes{};
    if (draw.texture_tile_mode == 0u) {
        const auto row_bytes =
            static_cast<std::uint64_t>(element_width) *
            element_bytes;
        source_bytes =
            ((row_bytes + UINT64_C(255)) & ~UINT64_C(255)) *
            element_height;
    } else if (draw.texture_tile_mode == 5u) {
        const auto block_extent =
            element_bytes == 1u ? 64u
            : element_bytes == 4u ? 32u
                                  : 16u;
        source_bytes =
            ((static_cast<std::uint64_t>(element_width) +
              block_extent - 1u) /
             block_extent) *
            ((static_cast<std::uint64_t>(element_height) +
              block_extent - 1u) /
             block_extent) *
            UINT64_C(4096);
    } else {
        return false;
    }
    if (source_bytes == 0u) {
        return false;
    }

    // GameMaker reuses the same GPU allocation for unrelated texture pages
    // across rooms. Sparse probes can miss those atlas updates completely,
    // leaving menu/title data visible in gameplay. Hash the whole source,
    // but only periodically at the cache call site below.
    constexpr std::size_t ProbeBytes = 64u * 1024u;
    static thread_local std::vector<std::uint8_t> sample;
    sample.resize(static_cast<std::size_t>(
        std::min<std::uint64_t>(ProbeBytes, source_bytes)));
    signature = UINT64_C(1469598103934665603);
    for (std::uint64_t offset = 0u; offset < source_bytes;) {
        const auto byte_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                sample.size(), source_bytes - offset));
        if (!TryReadGuestBytes(
                draw.texture_address + offset,
                sample.data(), byte_count)) {
            return false;
        }
        for (std::size_t index = 0; index < byte_count; ++index) {
            signature ^= sample[index];
            signature *= UINT64_C(1099511628211);
        }
        signature ^= offset;
        signature *= UINT64_C(1099511628211);
        offset += byte_count;
    }
    return true;
}

class AgcCpuRowWorkerPool {
public:
    AgcCpuRowWorkerPool() {
        constexpr std::size_t WorkerCount = 5u;
        workers_.reserve(WorkerCount);
        for (std::size_t index = 0; index < WorkerCount; ++index) {
            workers_.emplace_back([this]() { WorkerMain(); });
        }
    }

    ~AgcCpuRowWorkerPool() {
        {
            const std::lock_guard lock{mutex_};
            stopping_ = true;
            ++generation_;
        }
        start_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    void Run(const int first_row,
             const int last_row,
             std::function<void(int)> work) {
        {
            const std::lock_guard lock{mutex_};
            work_ = std::move(work);
            next_row_.store(first_row, std::memory_order_relaxed);
            end_row_ = last_row + 1;
            active_workers_ = workers_.size();
            ++generation_;
        }
        start_.notify_all();
        RunAvailableRows();
        std::unique_lock lock{mutex_};
        finished_.wait(lock, [&]() { return active_workers_ == 0u; });
        work_ = {};
    }

private:
    void RunAvailableRows() {
        for (;;) {
            const auto row =
                next_row_.fetch_add(1, std::memory_order_relaxed);
            if (row >= end_row_) {
                return;
            }
            work_(row);
        }
    }

    void WorkerMain() {
        std::uint64_t observed_generation{};
        for (;;) {
            {
                std::unique_lock lock{mutex_};
                start_.wait(lock, [&]() {
                    return stopping_ ||
                        generation_ != observed_generation;
                });
                if (stopping_) {
                    return;
                }
                observed_generation = generation_;
            }
            RunAvailableRows();
            {
                const std::lock_guard lock{mutex_};
                if (--active_workers_ == 0u) {
                    finished_.notify_one();
                }
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable finished_;
    std::vector<std::thread> workers_;
    std::function<void(int)> work_;
    std::atomic<int> next_row_{};
    int end_row_{};
    std::size_t active_workers_{};
    std::uint64_t generation_{};
    bool stopping_{};
};

void RunAgcCpuRows(const int first_row,
                   const int last_row,
                   const std::uint64_t candidate_pixels,
                   std::function<void(int)> work) {
    if (last_row - first_row < 7 ||
        candidate_pixels < 1024u) {
        for (auto row = first_row; row <= last_row; ++row) {
            work(row);
        }
        return;
    }
    static AgcCpuRowWorkerPool pool;
    pool.Run(first_row, last_row, std::move(work));
}

bool TryReadAgcCpuVertexBuffer(
    const AgcDiagnosticDraw& draw,
    const std::vector<std::uint32_t>& indices,
    std::vector<std::uint8_t>& vertices,
    std::uint64_t& vertex_address,
    std::uint32_t& vertex_stride,
    std::uint64_t& descriptor_table_address,
    bool& game_maker_vertex_group) {
    const auto vertex_count =
        indices.empty()
        ? 0u
        : *std::ranges::max_element(indices) + 1u;
    vertex_address = 0;
    vertex_stride = 0;
    descriptor_table_address = 0;
    game_maker_vertex_group = false;
    if (vertex_count == 0 || vertex_count > 65536u ||
        draw.sh_es_program_hi != 0 ||
        (draw.sh_es_program_lo & UINT32_C(0xff000000)) !=
            UINT32_C(0x20000000)) {
        return false;
    }
    const std::array<std::uint64_t, 4> legacy_srt_addresses{
        (static_cast<std::uint64_t>(
             draw.sh_export_user_data[1] & 0xffu) << 32u) |
            draw.sh_export_user_data[0],
        (static_cast<std::uint64_t>(
             draw.sh_gs_user_data[1] & 0xffu) << 32u) |
            draw.sh_gs_user_data[0],
        (static_cast<std::uint64_t>(
             draw.sh_es_user_data[1] & 0xffu) << 32u) |
            draw.sh_es_user_data[0],
        (static_cast<std::uint64_t>(
             draw.sh_vs_user_data[1] & 0xffu) << 32u) |
            draw.sh_vs_user_data[0]};
    std::array<std::uint32_t, 4> last_descriptor{};
    std::uint64_t used_srt_address{};
    std::uint64_t used_descriptor_offset{};
    std::uint64_t best_score{};
    std::uint32_t descriptor_reject_stage{};
    std::uint32_t rejected_candidate_stride{};
    std::uint32_t rejected_candidate_elements{};
    std::uint32_t rejected_candidate_format{};
    bool rejected_candidate_group{};
    std::uint64_t rejected_candidate_address{};
    std::uint32_t rejected_vertex_index{};
    std::array<float, 4> rejected_vertex_values{};
    const auto shader_suffix =
        draw.sh_es_program_lo & UINT32_C(0xfff);
    const auto pixel_shader_suffix =
        draw.sh_ps_program_lo & UINT32_C(0xfff);
    const auto game_maker_shader =
        pixel_shader_suffix == UINT32_C(0xf13) ||
        pixel_shader_suffix == UINT32_C(0x311) ||
        pixel_shader_suffix == UINT32_C(0x459);
    const auto unity_c40_shader =
        shader_suffix == UINT32_C(0xc40);
    const auto unity_aee_shader =
        shader_suffix == UINT32_C(0xaee);
    const auto unity_world_sprite_shader =
        shader_suffix == UINT32_C(0xbaa) ||
        unity_c40_shader ||
        unity_aee_shader;
    const auto unity_ui_shader =
        unity_world_sprite_shader ||
        shader_suffix == UINT32_C(0xc2b) ||
        shader_suffix == UINT32_C(0xaee);
    const auto try_descriptor =
        [&](const std::uint64_t srt_address,
            const std::uint64_t descriptor_offset) {
            std::array<std::uint32_t, 4> descriptor{};
            if (!TryReadGuestBytes(
                    srt_address + descriptor_offset,
                    descriptor.data(), sizeof(descriptor))) {
                return false;
            }
            last_descriptor = descriptor;
            const auto candidate_address =
                descriptor[0] |
                (static_cast<std::uint64_t>(
                     descriptor[1] & 0xffffu) << 32u);
            const auto candidate_stride =
                (descriptor[1] >> 16u) & 0x3fffu;
            const auto candidate_elements = descriptor[2];
            const auto candidate_unified_format =
                (descriptor[3] >> 12u) & 0x7fu;
            bool interleaved_float2_pair{};
            bool stride24_sprite_group{};
            if (!unity_ui_shader &&
                candidate_stride == 16u &&
                candidate_unified_format == 64u &&
                descriptor_offset + 32u <= UINT64_C(0x400)) {
                std::array<std::uint32_t, 4> sibling{};
                if (TryReadGuestBytes(
                        srt_address + descriptor_offset + 16u,
                        sibling.data(), sizeof(sibling))) {
                    const auto sibling_address =
                        sibling[0] |
                        (static_cast<std::uint64_t>(
                             sibling[1] & 0xffffu) << 32u);
                    const auto sibling_stride =
                        (sibling[1] >> 16u) & 0x3fffu;
                    const auto sibling_format =
                        (sibling[3] >> 12u) & 0x7fu;
                    interleaved_float2_pair =
                        sibling_address == candidate_address + 8u &&
                        sibling_stride == candidate_stride &&
                        sibling[2] == candidate_elements &&
                        sibling_format == candidate_unified_format;
                }
            }
            if (candidate_stride == 24u &&
                candidate_unified_format == 74u &&
                descriptor_offset + 48u <= UINT64_C(0x400)) {
                std::array<std::uint32_t, 8> siblings{};
                if (TryReadGuestBytes(
                        srt_address + descriptor_offset + 16u,
                        siblings.data(), sizeof(siblings))) {
                    const auto color_address =
                        siblings[0] |
                        (static_cast<std::uint64_t>(
                             siblings[1] & 0xffffu) << 32u);
                    const auto uv_address =
                        siblings[4] |
                        (static_cast<std::uint64_t>(
                             siblings[5] & 0xffffu) << 32u);
                    const auto color_stride =
                        (siblings[1] >> 16u) & 0x3fffu;
                    const auto uv_stride =
                        (siblings[5] >> 16u) & 0x3fffu;
                    const auto color_format =
                        (siblings[3] >> 12u) & 0x7fu;
                    const auto uv_format =
                        (siblings[7] >> 12u) & 0x7fu;
                    stride24_sprite_group =
                        color_address == candidate_address + 12u &&
                        uv_address == candidate_address + 16u &&
                        color_stride == candidate_stride &&
                        uv_stride == candidate_stride &&
                        siblings[2] == candidate_elements &&
                        siblings[6] == candidate_elements &&
                        color_format == 22u &&
                        uv_format == 64u;
                }
            }
            bool has_unity_vertex_group =
                !unity_ui_shader ||
                (unity_aee_shader &&
                 candidate_stride == 112u &&
                 candidate_unified_format == 77u) ||
                (unity_c40_shader &&
                 candidate_stride == 16u &&
                 candidate_unified_format == 74u);
            if (unity_ui_shader && !unity_c40_shader &&
                !unity_aee_shader &&
                descriptor_offset + 48u <= UINT64_C(0x400)) {
                std::array<std::uint32_t, 8> siblings{};
                if (TryReadGuestBytes(
                        srt_address + descriptor_offset + 16u,
                        siblings.data(), sizeof(siblings))) {
                    const auto sibling_format_0 =
                        (siblings[3] >> 12u) & 0x7fu;
                    const auto sibling_format_1 =
                        (siblings[7] >> 12u) & 0x7fu;
                    const auto same_stream =
                        siblings[0] == descriptor[0] &&
                        siblings[1] == descriptor[1] &&
                        siblings[2] == descriptor[2] &&
                        siblings[4] == descriptor[0] &&
                        siblings[5] == descriptor[1] &&
                        siblings[6] == descriptor[2];
                    has_unity_vertex_group =
                        same_stream &&
                        ((sibling_format_0 == 64u &&
                          (sibling_format_1 == 56u ||
                           sibling_format_1 == 77u)) ||
                         (sibling_format_1 == 64u &&
                          (sibling_format_0 == 56u ||
                           sibling_format_0 == 77u)));
                }
            }
            rejected_candidate_address = candidate_address;
            rejected_candidate_stride = candidate_stride;
            rejected_candidate_elements = candidate_elements;
            rejected_candidate_format = candidate_unified_format;
            rejected_candidate_group = has_unity_vertex_group;
            if (candidate_address == 0) {
                descriptor_reject_stage = 2u;
                return false;
            }
            if (candidate_stride != 24u &&
                !interleaved_float2_pair &&
                !(unity_c40_shader &&
                  candidate_stride == 16u) &&
                !(unity_aee_shader &&
                  candidate_stride == 112u) &&
                candidate_stride != 36u) {
                descriptor_reject_stage = 3u;
                return false;
            }
            if (candidate_elements < vertex_count) {
                descriptor_reject_stage = 4u;
                return false;
            }
            if (game_maker_shader &&
                !stride24_sprite_group) {
                descriptor_reject_stage = 6u;
                return false;
            }
            if (candidate_unified_format != 74u &&
                !(interleaved_float2_pair &&
                  candidate_unified_format == 64u) &&
                !stride24_sprite_group &&
                !(unity_aee_shader &&
                  candidate_unified_format == 77u)) {
                descriptor_reject_stage = 5u;
                return false;
            }
            if (!has_unity_vertex_group) {
                descriptor_reject_stage = 6u;
                return false;
            }
            if (vertex_count >
                std::numeric_limits<std::size_t>::max() /
                    candidate_stride) {
                descriptor_reject_stage = 7u;
                return false;
            }
            const auto byte_count =
                static_cast<std::size_t>(vertex_count) *
                candidate_stride;
            std::vector<std::uint8_t> probe(byte_count);
            if (!TryReadGuestBytes(
                    candidate_address, probe.data(), probe.size())) {
                descriptor_reject_stage = 8u;
                return false;
            }
#ifdef __ANDROID__
            if (game_maker_shader &&
                draw.texture_width == 64u &&
                draw.texture_height == 128u) {
                static std::atomic<std::uint32_t>
                    game_maker_descriptor_probe_logs{};
                if (game_maker_descriptor_probe_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 40u) {
                    float values[6]{};
                    for (std::size_t word = 0; word < 6u &&
                         word * sizeof(float) + sizeof(float) <=
                             probe.size(); ++word) {
                        std::memcpy(
                            values + word,
                            probe.data() + word * sizeof(float),
                            sizeof(float));
                    }
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "gm descriptor probe srt=0x%llx off=0x%llx "
                        "desc=%08x,%08x,%08x,%08x "
                        "address=0x%llx stride=%u elements=%u "
                        "format=%u group=%u values="
                        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                        static_cast<unsigned long long>(srt_address),
                        static_cast<unsigned long long>(
                            descriptor_offset),
                        descriptor[0], descriptor[1],
                        descriptor[2], descriptor[3],
                        static_cast<unsigned long long>(
                            candidate_address),
                        candidate_stride, candidate_elements,
                        candidate_unified_format,
                        stride24_sprite_group ? 1u : 0u,
                        values[0], values[1], values[2],
                        values[3], values[4], values[5]);
                }
            }
#endif
            const auto read_float =
                [&](const std::uint32_t index,
                    const std::uint32_t byte_offset) {
                    float value{};
                    std::memcpy(
                        &value,
                        probe.data() +
                            static_cast<std::size_t>(index) *
                                candidate_stride +
                            byte_offset,
                        sizeof(value));
                    return value;
                };
            const auto position_limit_x =
                (unity_world_sprite_shader || game_maker_shader)
                ? 10000000.0f
                : static_cast<float>(
                      std::max(draw.render_target_width, 1920u)) *
                      2.0f;
            const auto position_limit_y =
                (unity_world_sprite_shader || game_maker_shader)
                ? 10000000.0f
                : static_cast<float>(
                      std::max(draw.render_target_height, 1080u)) *
                      2.0f;
            std::uint64_t score{};
            for (const auto index : indices) {
                const auto x = read_float(index, 0u);
                const auto y = read_float(index, 4u);
                float u{};
                float v{};
                if (unity_c40_shader &&
                    candidate_stride == 16u) {
                    const auto attribute =
                        static_cast<std::size_t>(index) *
                            candidate_stride +
                        12u;
                    u = static_cast<float>(probe[attribute]) /
                        255.0f;
                    v = static_cast<float>(probe[attribute + 1u]) /
                        255.0f;
                } else if (unity_aee_shader) {
                    u = read_float(index, 64u);
                    v = read_float(index, 68u);
                } else {
                    const auto uv_offset =
                        interleaved_float2_pair
                            ? 8u
                            : (candidate_stride == 36u ? 28u : 16u);
                    u = read_float(index, uv_offset);
                    v = read_float(index, uv_offset + 4u);
                }
                if (!std::isfinite(x) || !std::isfinite(y) ||
                    !std::isfinite(u) || !std::isfinite(v) ||
                    x < -position_limit_x ||
                    x > position_limit_x ||
                    y < -position_limit_y ||
                    y > position_limit_y ||
                    std::abs(u) >
                        (game_maker_shader ? 65536.0f : 16.0f) ||
                    std::abs(v) >
                        (game_maker_shader ? 65536.0f : 16.0f)) {
                    descriptor_reject_stage = 9u;
                    rejected_vertex_index = index;
                    rejected_vertex_values = {x, y, u, v};
                    return false;
                }
                score +=
                    x >= -1.0f &&
                            x <= static_cast<float>(
                                     draw.render_target_width) &&
                            y >= -1.0f &&
                            y <= static_cast<float>(
                                     draw.render_target_height)
                    ? 4u
                    : 1u;
                score +=
                    u >= -0.001f && u <= 1.001f &&
                            v >= -0.001f && v <= 1.001f
                    ? 4u
                    : 0u;
            }
            std::uint32_t nondegenerate_triangles{};
            for (std::size_t first = 0;
                 first + 2u < indices.size(); first += 3u) {
                const auto x0 = read_float(indices[first], 0u);
                const auto y0 = read_float(indices[first], 4u);
                const auto x1 = read_float(indices[first + 1u], 0u);
                const auto y1 = read_float(indices[first + 1u], 4u);
                const auto x2 = read_float(indices[first + 2u], 0u);
                const auto y2 = read_float(indices[first + 2u], 4u);
                const auto area =
                    (x1 - x0) * (y2 - y0) -
                    (y1 - y0) * (x2 - x0);
                if (std::isfinite(area) &&
                    std::abs(area) >= 0.0001f) {
                    ++nondegenerate_triangles;
                }
            }
            if (nondegenerate_triangles == 0u) {
                descriptor_reject_stage = 10u;
                return false;
            }
            score +=
                static_cast<std::uint64_t>(
                    nondegenerate_triangles) *
                1024u;
            if (score <= best_score) {
                descriptor_reject_stage = 11u;
                return false;
            }
            best_score = score;
            vertex_address = candidate_address;
            if (interleaved_float2_pair) {
                constexpr std::uint32_t canonical_stride = 24u;
                vertices.assign(
                    static_cast<std::size_t>(vertex_count) *
                        canonical_stride,
                    0u);
                for (std::uint32_t index = 0;
                     index < vertex_count; ++index) {
                    const auto* const source =
                        probe.data() +
                        static_cast<std::size_t>(index) *
                            candidate_stride;
                    auto* const destination =
                        vertices.data() +
                        static_cast<std::size_t>(index) *
                            canonical_stride;
                    std::memcpy(destination, source, 8u);
                    std::memset(destination + 12u, 0xff, 4u);
                    std::memcpy(destination + 16u, source + 8u, 8u);
                }
                vertex_stride = canonical_stride;
            } else {
                vertex_stride = candidate_stride;
                vertices = std::move(probe);
            }
            used_srt_address = srt_address;
            used_descriptor_offset = descriptor_offset;
            game_maker_vertex_group = stride24_sprite_group;
            return true;
        };
    if (unity_aee_shader) {
        const auto descriptor_table =
            (static_cast<std::uint64_t>(
                 draw.sh_gs_user_data[9] & 0xffu) << 32u) |
            draw.sh_gs_user_data[8];
        const auto parameter_block =
            (static_cast<std::uint64_t>(
                 draw.sh_gs_user_data[11] & 0xffu) << 32u) |
            draw.sh_gs_user_data[10];
        std::array<std::uint32_t, 4> parameters{};
        if (descriptor_table != 0u &&
            parameter_block != 0u &&
            TryReadGuestBytes(
                parameter_block, parameters.data(),
                sizeof(parameters))) {
            const auto descriptor_offset =
                static_cast<std::uint64_t>(
                    parameters[2] & 0x1fu) *
                16u;
            (void)try_descriptor(
                descriptor_table, descriptor_offset);
        }
    } else if (unity_ui_shader) {
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 2>
            table_bases{{
                {
                    (static_cast<std::uint64_t>(
                         draw.sh_gs_user_data[5] & 0xffu) << 32u) |
                        draw.sh_gs_user_data[4],
                    (static_cast<std::uint64_t>(
                         draw.sh_gs_user_data[7] & 0xffu) << 32u) |
                        draw.sh_gs_user_data[6],
                },
                {
                    (static_cast<std::uint64_t>(
                         draw.sh_es_user_data[13] & 0xffu) << 32u) |
                        draw.sh_es_user_data[12],
                    (static_cast<std::uint64_t>(
                         draw.sh_es_user_data[15] & 0xffu) << 32u) |
                        draw.sh_es_user_data[14],
                },
            }};
        bool resolved{};
        for (const auto& [eud_address, srt_address] :
             table_bases) {
            std::uint32_t resource_index_word{};
            if (eud_address == 0 || srt_address == 0 ||
                !TryReadGuestValue(
                    srt_address, resource_index_word)) {
                continue;
            }
            const auto descriptor_offset =
                static_cast<std::uint64_t>(
                    resource_index_word & 0x1fu) *
                16u;
            if (try_descriptor(
                    eud_address, descriptor_offset)) {
                resolved = true;
#ifdef __ANDROID__
                static std::atomic<std::uint32_t>
                    exact_descriptor_logs{};
                if (exact_descriptor_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 4u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "cpu unity descriptor eud=0x%llx "
                        "srt=0x%llx index_word=%08x "
                        "offset=0x%llx vertex=0x%llx stride=%u",
                        static_cast<unsigned long long>(
                            eud_address),
                        static_cast<unsigned long long>(
                            srt_address),
                        resource_index_word,
                        static_cast<unsigned long long>(
                            descriptor_offset),
                        static_cast<unsigned long long>(
                            vertex_address),
                        vertex_stride);
                }
#endif
                break;
            }
        }
        if (!resolved) {
#ifdef __ANDROID__
            if (shader_suffix == UINT32_C(0xaee)) {
                static std::atomic<std::uint32_t>
                    aee_user_data_logs{};
                if (aee_user_data_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 4u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "cpu aee user data gs="
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x",
                        draw.sh_gs_user_data[0],
                        draw.sh_gs_user_data[1],
                        draw.sh_gs_user_data[2],
                        draw.sh_gs_user_data[3],
                        draw.sh_gs_user_data[4],
                        draw.sh_gs_user_data[5],
                        draw.sh_gs_user_data[6],
                        draw.sh_gs_user_data[7],
                        draw.sh_gs_user_data[8],
                        draw.sh_gs_user_data[9],
                        draw.sh_gs_user_data[10],
                        draw.sh_gs_user_data[11],
                        draw.sh_gs_user_data[12],
                        draw.sh_gs_user_data[13],
                        draw.sh_gs_user_data[14],
                        draw.sh_gs_user_data[15]);
                }
            }
            if (shader_suffix == UINT32_C(0xbaa)) {
                static std::atomic<std::uint32_t>
                    baa_user_data_logs{};
                if (baa_user_data_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 8u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "cpu baa descriptor miss tex=0x%llx "
                        "gs=%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x "
                        "es=%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x "
                        "reject=%u last=%08x,%08x,%08x,%08x "
                        "candidate=0x%llx/%u/%u/%u group=%u "
                        "bad=%u:%.3f,%.3f,%.3f,%.3f",
                        static_cast<unsigned long long>(
                            draw.texture_address),
                        draw.sh_gs_user_data[0],
                        draw.sh_gs_user_data[1],
                        draw.sh_gs_user_data[2],
                        draw.sh_gs_user_data[3],
                        draw.sh_gs_user_data[4],
                        draw.sh_gs_user_data[5],
                        draw.sh_gs_user_data[6],
                        draw.sh_gs_user_data[7],
                        draw.sh_gs_user_data[8],
                        draw.sh_gs_user_data[9],
                        draw.sh_gs_user_data[10],
                        draw.sh_gs_user_data[11],
                        draw.sh_gs_user_data[12],
                        draw.sh_gs_user_data[13],
                        draw.sh_gs_user_data[14],
                        draw.sh_gs_user_data[15],
                        draw.sh_es_user_data[0],
                        draw.sh_es_user_data[1],
                        draw.sh_es_user_data[2],
                        draw.sh_es_user_data[3],
                        draw.sh_es_user_data[4],
                        draw.sh_es_user_data[5],
                        draw.sh_es_user_data[6],
                        draw.sh_es_user_data[7],
                        draw.sh_es_user_data[8],
                        draw.sh_es_user_data[9],
                        draw.sh_es_user_data[10],
                        draw.sh_es_user_data[11],
                        draw.sh_es_user_data[12],
                        draw.sh_es_user_data[13],
                        draw.sh_es_user_data[14],
                        draw.sh_es_user_data[15],
                        descriptor_reject_stage,
                        last_descriptor[0],
                        last_descriptor[1],
                        last_descriptor[2],
                        last_descriptor[3],
                        static_cast<unsigned long long>(
                            rejected_candidate_address),
                        rejected_candidate_stride,
                        rejected_candidate_elements,
                        rejected_candidate_format,
                        rejected_candidate_group ? 1u : 0u,
                        rejected_vertex_index,
                        rejected_vertex_values[0],
                        rejected_vertex_values[1],
                        rejected_vertex_values[2],
                        rejected_vertex_values[3]);
                }
            }
            static std::atomic<std::uint32_t>
                exact_descriptor_failure_logs{};
            if (exact_descriptor_failure_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 16u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu unity descriptor unresolved "
                    "gseud4-7=%08x,%08x,%08x,%08x "
                    "esud12-15=%08x,%08x,%08x,%08x",
                    draw.sh_gs_user_data[4],
                    draw.sh_gs_user_data[5],
                    draw.sh_gs_user_data[6],
                    draw.sh_gs_user_data[7],
                    draw.sh_es_user_data[12],
                    draw.sh_es_user_data[13],
                    draw.sh_es_user_data[14],
                    draw.sh_es_user_data[15]);
            }
#endif
        }
    } else {
        const auto scan_descriptor_table =
            [&](const std::uint64_t table_address) {
                if (table_address == 0) {
                    return false;
                }
                for (const auto descriptor_offset :
                     {UINT64_C(0x180), UINT64_C(0x80)}) {
                    if (try_descriptor(
                            table_address, descriptor_offset)) {
                        return true;
                    }
                }
                for (std::uint64_t descriptor_offset = 0;
                     descriptor_offset < UINT64_C(0x200);
                     descriptor_offset += 4u) {
                    if (descriptor_offset == UINT64_C(0x180) ||
                        descriptor_offset == UINT64_C(0x80)) {
                        continue;
                    }
                    if (try_descriptor(
                            table_address, descriptor_offset)) {
                        return true;
                    }
                }
                return false;
            };
        bool resolved{};
        if (game_maker_shader) {
            std::vector<std::uint64_t> table_addresses;
            const auto append_user_data_addresses =
                [&](const auto& user_data) {
                    for (std::size_t word = 0;
                         word + 1u < user_data.size(); ++word) {
                        const auto address =
                            (static_cast<std::uint64_t>(
                                 user_data[word + 1u] & 0xffu)
                             << 32u) |
                            user_data[word];
                        if (address != 0 &&
                            !std::ranges::contains(
                                table_addresses, address)) {
                            table_addresses.push_back(address);
                        }
                    }
                };
            append_user_data_addresses(
                draw.sh_export_user_data);
            append_user_data_addresses(draw.sh_gs_user_data);
            append_user_data_addresses(draw.sh_es_user_data);
            append_user_data_addresses(draw.sh_vs_user_data);
            for (const auto table_address : table_addresses) {
                if (scan_descriptor_table(table_address)) {
                    resolved = true;
                    break;
                }
            }
            if (!resolved) {
                const auto direct_addresses = table_addresses;
                for (const auto source_address :
                     direct_addresses) {
                    std::array<std::uint32_t, 64> pointers{};
                    if (!TryReadGuestBytes(
                            source_address, pointers.data(),
                            sizeof(pointers))) {
                        continue;
                    }
                    for (std::size_t word = 0;
                         word + 1u < pointers.size(); ++word) {
                        const auto nested_address =
                            (static_cast<std::uint64_t>(
                                 pointers[word + 1u] & 0xffu)
                             << 32u) |
                            pointers[word];
                        if (scan_descriptor_table(
                                nested_address)) {
                            resolved = true;
                            break;
                        }
                    }
                    if (resolved) {
                        break;
                    }
                }
            }
        } else {
            for (const auto srt_address :
                 legacy_srt_addresses) {
                if (scan_descriptor_table(srt_address)) {
                    resolved = true;
                    break;
                }
            }
        }
    }
    if (vertex_address == 0 || vertex_stride == 0 ||
        vertex_count >
            std::numeric_limits<std::size_t>::max() /
                vertex_stride) {
#ifdef __ANDROID__
        if (game_maker_shader) {
            static std::atomic<std::uint32_t>
                game_maker_vertex_logs{};
            if (game_maker_vertex_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 4u) {
                std::array<std::uint32_t, 16> probe{};
                (void)TryReadGuestBytes(
                    legacy_srt_addresses[0],
                    probe.data(), sizeof(probe));
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu gm vertex miss srt=0x%llx "
                    "gs=%08x,%08x,%08x,%08x "
                    "es=%08x,%08x,%08x,%08x "
                    "probe=%08x,%08x,%08x,%08x,"
                    "%08x,%08x,%08x,%08x,"
                    "%08x,%08x,%08x,%08x,"
                    "%08x,%08x,%08x,%08x",
                    static_cast<unsigned long long>(
                        legacy_srt_addresses[0]),
                    draw.sh_gs_user_data[0],
                    draw.sh_gs_user_data[1],
                    draw.sh_gs_user_data[2],
                    draw.sh_gs_user_data[3],
                    draw.sh_es_user_data[0],
                    draw.sh_es_user_data[1],
                    draw.sh_es_user_data[2],
                    draw.sh_es_user_data[3],
                    probe[0], probe[1], probe[2], probe[3],
                    probe[4], probe[5], probe[6], probe[7],
                    probe[8], probe[9], probe[10], probe[11],
                    probe[12], probe[13], probe[14], probe[15]);
            }
        }
        static std::atomic<std::uint32_t> descriptor_failure_logs{};
        const auto log_index = descriptor_failure_logs.fetch_add(
            1u, std::memory_order_relaxed);
        if (log_index < 16u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu vertex descriptor miss srt=0x%llx/0x%llx "
                "words=%08x,%08x,%08x,%08x shader=%03x "
                "vertices=%u reject=%u candidate=0x%llx/%u/%u/%u "
                "group=%u bad_vertex=%u values=%.3f,%.3f,%.3f,%.3f",
                static_cast<unsigned long long>(
                    legacy_srt_addresses[0]),
                static_cast<unsigned long long>(
                    legacy_srt_addresses[1]),
                last_descriptor[0], last_descriptor[1],
                last_descriptor[2], last_descriptor[3],
                shader_suffix, vertex_count,
                descriptor_reject_stage,
                static_cast<unsigned long long>(
                    rejected_candidate_address),
                rejected_candidate_stride,
                rejected_candidate_elements,
                rejected_candidate_format,
                rejected_candidate_group ? 1u : 0u,
                rejected_vertex_index,
                rejected_vertex_values[0],
                rejected_vertex_values[1],
                rejected_vertex_values[2],
                rejected_vertex_values[3]);
        }
#endif
        return false;
    }
#ifdef __ANDROID__
    if (used_descriptor_offset != UINT64_C(0x180) &&
        used_descriptor_offset != UINT64_C(0x80)) {
        static std::atomic<std::uint32_t>
            relocated_descriptor_logs{};
        if (relocated_descriptor_logs.fetch_add(
                1u, std::memory_order_relaxed) < 32u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu vertex descriptor relocated srt=0x%llx "
                "offset=0x%llx vertex=0x%llx stride=%u score=%llu "
                "format=74",
                static_cast<unsigned long long>(
                    used_srt_address),
                static_cast<unsigned long long>(
                    used_descriptor_offset),
                static_cast<unsigned long long>(
                    vertex_address),
                vertex_stride,
                static_cast<unsigned long long>(best_score));
        }
    }
#endif
    descriptor_table_address = used_srt_address;
    return true;
}

bool TryReadAgcCpuTextureElements(
    const AgcDiagnosticDraw& draw,
    const std::uint32_t element_width,
    const std::uint32_t element_height,
    const std::uint32_t element_bytes,
    std::vector<std::uint8_t>& linear) {
    const auto element_count =
        static_cast<std::uint64_t>(element_width) *
        element_height;
    if (element_count == 0 ||
        element_bytes == 0 ||
        element_count >
            std::numeric_limits<std::size_t>::max() /
                element_bytes) {
        return false;
    }
    linear.resize(
        static_cast<std::size_t>(element_count) *
        element_bytes);
    if (draw.texture_tile_mode == 0u) {
        const auto row_bytes =
            static_cast<std::uint64_t>(element_width) *
            element_bytes;
        const auto pitch_bytes =
            (row_bytes + UINT64_C(255)) & ~UINT64_C(255);
        if (pitch_bytes == row_bytes) {
            return TryReadGuestBytes(
                draw.texture_address, linear.data(), linear.size());
        }
        if (pitch_bytes >
            std::numeric_limits<std::size_t>::max() /
                element_height) {
            return false;
        }
        std::vector<std::uint8_t> pitched(
            static_cast<std::size_t>(pitch_bytes) *
            element_height);
        if (!TryReadGuestBytes(
                draw.texture_address, pitched.data(),
                pitched.size())) {
            return false;
        }
        for (std::uint32_t row = 0; row < element_height; ++row) {
            std::memcpy(
                linear.data() +
                    static_cast<std::size_t>(row * row_bytes),
                pitched.data() +
                    static_cast<std::size_t>(row * pitch_bytes),
                static_cast<std::size_t>(row_bytes));
        }
        return true;
    }
    if (draw.texture_tile_mode == UINT32_C(27) &&
        element_bytes == 8u) {
        constexpr std::uint32_t BlockWidth = 128u;
        constexpr std::uint32_t BlockHeight = 64u;
        constexpr std::uint64_t BlockBytes = UINT64_C(65536);
        constexpr std::array<std::uint32_t, 16> XMask{
            0u, 0u, 0u, 1u << 0u,
            0u, 1u << 1u, 1u << 2u, 0u,
            1u << 7u, 1u << 4u, 1u << 6u, 1u << 5u,
            0u, 1u << 3u, 1u << 7u, 1u << 6u};
        constexpr std::array<std::uint32_t, 16> YMask{
            0u, 0u, 0u, 0u,
            1u << 0u, 0u, 0u, 1u << 1u,
            (1u << 4u) | (1u << 7u), 1u << 4u,
            1u << 5u, 1u << 6u,
            1u << 2u, 0u, 1u << 3u, 1u << 6u};
        const auto blocks_wide =
            (static_cast<std::uint64_t>(element_width) +
             BlockWidth - 1u) /
            BlockWidth;
        const auto blocks_high =
            (static_cast<std::uint64_t>(element_height) +
             BlockHeight - 1u) /
            BlockHeight;
        if (blocks_wide == 0u || blocks_high == 0u ||
            blocks_wide >
                std::numeric_limits<std::size_t>::max() /
                    blocks_high ||
            blocks_wide * blocks_high >
                std::numeric_limits<std::size_t>::max() /
                    BlockBytes) {
            return false;
        }
        static thread_local std::vector<std::uint8_t> tiled;
        tiled.resize(static_cast<std::size_t>(
            blocks_wide * blocks_high * BlockBytes));
        if (!TryReadGuestBytes(
                draw.texture_address, tiled.data(), tiled.size())) {
            return false;
        }
        const auto axis_term =
            [](const std::uint32_t coordinate,
               const auto& masks) {
                std::uint32_t offset{};
                for (std::uint32_t bit = 0u; bit < 16u; ++bit) {
                    offset |=
                        (static_cast<std::uint32_t>(
                             __builtin_popcount(
                                 coordinate & masks[bit])) &
                         1u)
                        << bit;
                }
                return offset;
            };
        std::vector<std::uint32_t> x_terms(element_width);
        std::vector<std::uint32_t> y_terms(element_height);
        for (std::uint32_t x = 0u; x < element_width; ++x) {
            x_terms[x] = axis_term(x, XMask);
        }
        for (std::uint32_t y = 0u; y < element_height; ++y) {
            y_terms[y] = axis_term(y, YMask);
        }
        for (std::uint32_t y = 0u; y < element_height; ++y) {
            const auto block_y = y / BlockHeight;
            for (std::uint32_t x = 0u; x < element_width; ++x) {
                const auto block_x = x / BlockWidth;
                const auto block_index =
                    static_cast<std::uint64_t>(block_y) *
                        blocks_wide +
                    block_x;
                const auto source =
                    block_index * BlockBytes +
                    (x_terms[x] ^ y_terms[y]);
                const auto destination =
                    (static_cast<std::size_t>(y) *
                         element_width +
                     x) *
                    element_bytes;
                if (source + element_bytes <= tiled.size()) {
                    std::memcpy(
                        linear.data() + destination,
                        tiled.data() +
                            static_cast<std::size_t>(source),
                        element_bytes);
                }
            }
        }
        return true;
    }
    if (draw.texture_tile_mode != 5u ||
        (element_bytes != 1u &&
         element_bytes != 4u &&
         element_bytes != 16u)) {
        return false;
    }

    std::array<std::uint32_t, 12> x_masks{};
    std::array<std::uint32_t, 12> y_masks{};
    std::uint32_t block_width{};
    std::uint32_t block_height{};
    if (element_bytes == 1u) {
        x_masks = {
            1u << 0u, 1u << 1u, 1u << 2u, 1u << 3u,
            0u, 0u, 0u, 0u,
            0u, 1u << 4u, 0u, 1u << 5u};
        y_masks = {
            0u, 0u, 0u, 0u,
            1u << 0u, 1u << 1u, 1u << 2u, 1u << 3u,
            1u << 4u, 0u, 1u << 5u, 0u};
        block_width = 64u;
        block_height = 64u;
    } else if (element_bytes == 4u) {
        x_masks = {
            0u, 0u, 1u << 0u, 1u << 1u,
            0u, 0u, 0u, 1u << 2u,
            0u, 1u << 3u, 0u, 1u << 4u};
        y_masks = {
            0u, 0u, 0u, 0u,
            1u << 0u, 1u << 1u, 1u << 2u, 0u,
            1u << 3u, 0u, 1u << 4u, 0u};
        block_width = 32u;
        block_height = 32u;
    } else {
        x_masks = {
            0u, 0u, 0u, 0u,
            0u, 0u, 1u << 0u, 1u << 1u,
            0u, 1u << 2u, 0u, 1u << 3u};
        y_masks = {
            0u, 0u, 0u, 0u,
            1u << 0u, 1u << 1u, 0u, 0u,
            1u << 2u, 0u, 1u << 3u, 0u};
        block_width = 16u;
        block_height = 16u;
    }
    constexpr std::uint64_t TileBytes = 4096u;
    const auto tiles_wide =
        (static_cast<std::uint64_t>(element_width) +
         block_width - 1u) /
        block_width;
    const auto tiles_high =
        (static_cast<std::uint64_t>(element_height) +
         block_height - 1u) /
        block_height;
    if (tiles_wide == 0 || tiles_high == 0 ||
        tiles_wide >
            std::numeric_limits<std::size_t>::max() /
                tiles_high ||
        tiles_wide * tiles_high >
            std::numeric_limits<std::size_t>::max() /
                TileBytes) {
        return false;
    }
    std::vector<std::uint8_t> tiled(
        static_cast<std::size_t>(
            tiles_wide * tiles_high * TileBytes));
    if (!TryReadGuestBytes(
            draw.texture_address, tiled.data(), tiled.size())) {
        return false;
    }
    const auto axis_term =
        [](const std::uint32_t coordinate,
           const std::array<std::uint32_t, 12>& masks) {
            std::uint32_t offset{};
            for (std::uint32_t bit = 0; bit < 12u; ++bit) {
                const auto parity = static_cast<std::uint32_t>(
                    __builtin_popcount(coordinate & masks[bit])) &
                    1u;
                offset |= parity << bit;
            }
            return offset;
        };
    std::vector<std::uint32_t> x_terms(element_width);
    for (std::uint32_t x = 0; x < element_width; ++x) {
        x_terms[x] = axis_term(x, x_masks);
    }
    for (std::uint32_t y = 0; y < element_height; ++y) {
        const auto tile_y = y / block_height;
        const auto y_term = axis_term(y, y_masks);
        for (std::uint32_t x = 0; x < element_width; ++x) {
            const auto tile_x = x / block_width;
            const auto tile_index =
                static_cast<std::uint64_t>(tile_y) *
                    tiles_wide +
                tile_x;
            const auto source =
                tile_index * TileBytes +
                (x_terms[x] ^ y_term);
            const auto destination =
                (static_cast<std::uint64_t>(y) *
                     element_width +
                 x) *
                element_bytes;
            if (source + element_bytes <= tiled.size()) {
                std::memcpy(
                    linear.data() +
                        static_cast<std::size_t>(destination),
                    tiled.data() +
                        static_cast<std::size_t>(source),
                    element_bytes);
            }
        }
    }
    return true;
}

void DecodeAgcBc3Block(
    const std::uint8_t* const block,
    std::array<std::array<std::uint8_t, 4>, 16>& pixels) {
    std::array<std::uint8_t, 8> alphas{
        block[0], block[1], 0u, 0u, 0u, 0u, 0u, 0u};
    if (alphas[0] > alphas[1]) {
        for (std::uint32_t index = 1; index <= 6u; ++index) {
            alphas[index + 1u] = static_cast<std::uint8_t>(
                ((7u - index) * alphas[0] +
                 index * alphas[1]) /
                7u);
        }
    } else {
        for (std::uint32_t index = 1; index <= 4u; ++index) {
            alphas[index + 1u] = static_cast<std::uint8_t>(
                ((5u - index) * alphas[0] +
                 index * alphas[1]) /
                5u);
        }
        alphas[6] = 0u;
        alphas[7] = 255u;
    }
    std::uint64_t alpha_indices{};
    for (std::uint32_t byte = 0; byte < 6u; ++byte) {
        alpha_indices |=
            static_cast<std::uint64_t>(block[2u + byte]) <<
            (byte * 8u);
    }
    const auto color_0 =
        static_cast<std::uint16_t>(block[8]) |
        (static_cast<std::uint16_t>(block[9]) << 8u);
    const auto color_1 =
        static_cast<std::uint16_t>(block[10]) |
        (static_cast<std::uint16_t>(block[11]) << 8u);
    const auto expand_565 =
        [](const std::uint16_t color) {
            return std::array<std::uint8_t, 3>{
                static_cast<std::uint8_t>(
                    ((color >> 11u) & 0x1fu) * 255u / 31u),
                static_cast<std::uint8_t>(
                    ((color >> 5u) & 0x3fu) * 255u / 63u),
                static_cast<std::uint8_t>(
                    (color & 0x1fu) * 255u / 31u)};
        };
    std::array<std::array<std::uint8_t, 3>, 4> colors{
        expand_565(color_0), expand_565(color_1), {}, {}};
    for (std::size_t channel = 0; channel < 3u; ++channel) {
        colors[2][channel] = static_cast<std::uint8_t>(
            (2u * colors[0][channel] +
             colors[1][channel]) /
            3u);
        colors[3][channel] = static_cast<std::uint8_t>(
            (colors[0][channel] +
             2u * colors[1][channel]) /
            3u);
    }
    const auto color_indices =
        static_cast<std::uint32_t>(block[12]) |
        (static_cast<std::uint32_t>(block[13]) << 8u) |
        (static_cast<std::uint32_t>(block[14]) << 16u) |
        (static_cast<std::uint32_t>(block[15]) << 24u);
    for (std::uint32_t pixel = 0; pixel < 16u; ++pixel) {
        const auto color_index =
            (color_indices >> (pixel * 2u)) & 3u;
        const auto alpha_index =
            (alpha_indices >> (pixel * 3u)) & 7u;
        pixels[pixel] = {
            colors[color_index][0],
            colors[color_index][1],
            colors[color_index][2],
            alphas[alpha_index]};
    }
}

bool TryReadAgcCpuTextureRgba(
    const AgcDiagnosticDraw& draw,
    std::vector<std::uint8_t>& rgba) {
    const auto pixel_count =
        static_cast<std::uint64_t>(draw.texture_width) *
        draw.texture_height;
    if (pixel_count == 0 ||
        pixel_count >
            std::numeric_limits<std::size_t>::max() / 4u) {
        return false;
    }
    rgba.assign(static_cast<std::size_t>(pixel_count) * 4u, 0u);
    if (draw.texture_format == UINT32_C(0x47)) {
        std::vector<std::uint8_t> half_rgba;
        if (!TryReadAgcCpuTextureElements(
                draw, draw.texture_width, draw.texture_height,
                8u, half_rgba)) {
            return false;
        }
        const auto half_to_float =
            [](const std::uint16_t half) {
                const auto sign =
                    static_cast<std::uint32_t>(half & 0x8000u)
                    << 16u;
                auto exponent =
                    static_cast<std::uint32_t>(
                        (half >> 10u) & 0x1fu);
                auto mantissa =
                    static_cast<std::uint32_t>(half & 0x3ffu);
                std::uint32_t bits{};
                if (exponent == 0u) {
                    if (mantissa == 0u) {
                        bits = sign;
                    } else {
                        std::int32_t unbiased = -14;
                        while ((mantissa & 0x400u) == 0u) {
                            mantissa <<= 1u;
                            --unbiased;
                        }
                        mantissa &= 0x3ffu;
                        bits = sign |
                            (static_cast<std::uint32_t>(
                                 unbiased + 127)
                             << 23u) |
                            (mantissa << 13u);
                    }
                } else if (exponent == 0x1fu) {
                    bits = sign | UINT32_C(0x7f800000) |
                        (mantissa << 13u);
                } else {
                    bits = sign |
                        ((exponent + 112u) << 23u) |
                        (mantissa << 13u);
                }
                float value{};
                std::memcpy(&value, &bits, sizeof(value));
                return value;
            };
        for (std::size_t pixel = 0u;
             pixel < static_cast<std::size_t>(pixel_count);
             ++pixel) {
            for (std::size_t channel = 0u; channel < 4u;
                 ++channel) {
                std::uint16_t half{};
                std::memcpy(
                    &half,
                    half_rgba.data() + pixel * 8u + channel * 2u,
                    sizeof(half));
                auto value = half_to_float(half);
                if (!std::isfinite(value)) {
                    value = 0.0f;
                }
                value = std::clamp(value, 0.0f, 1.0f);
                if (channel != 3u) {
                    value = value <= 0.0031308f
                        ? value * 12.92f
                        : 1.055f *
                              std::pow(value, 1.0f / 2.4f) -
                              0.055f;
                }
                rgba[pixel * 4u + channel] =
                    static_cast<std::uint8_t>(
                        std::clamp(
                            value * 255.0f + 0.5f,
                            0.0f, 255.0f));
            }
        }
        return true;
    }
    if (draw.texture_format == UINT32_C(0x38)) {
        return TryReadAgcCpuTextureElements(
            draw, draw.texture_width, draw.texture_height,
            4u, rgba);
    }
    if (draw.texture_format == UINT32_C(0x1)) {
        std::vector<std::uint8_t> red;
        if (!TryReadAgcCpuTextureElements(
                draw, draw.texture_width, draw.texture_height,
                1u, red)) {
            return false;
        }
        for (std::size_t pixel = 0; pixel < red.size(); ++pixel) {
            rgba[pixel * 4u + 0u] = 255u;
            rgba[pixel * 4u + 1u] = 255u;
            rgba[pixel * 4u + 2u] = 255u;
            rgba[pixel * 4u + 3u] = red[pixel];
        }
        return true;
    }
    if (draw.texture_format == UINT32_C(0xad) ||
        draw.texture_format == UINT32_C(0xb5)) {
        const auto blocks_wide =
            (draw.texture_width + 3u) / 4u;
        const auto blocks_high =
            (draw.texture_height + 3u) / 4u;
        std::vector<std::uint8_t> blocks;
        if (!TryReadAgcCpuTextureElements(
                draw, blocks_wide, blocks_high,
                16u, blocks)) {
            return false;
        }
        std::array<std::array<std::uint8_t, 4>, 16>
            decoded{};
        for (std::uint32_t block_y = 0;
             block_y < blocks_high; ++block_y) {
            for (std::uint32_t block_x = 0;
                 block_x < blocks_wide; ++block_x) {
                const auto source =
                    (static_cast<std::size_t>(block_y) *
                         blocks_wide +
                     block_x) *
                    16u;
                if (draw.texture_format == UINT32_C(0xb5)) {
                    if (!bc7decomp::unpack_bc7(
                            blocks.data() + source,
                            reinterpret_cast<
                                bc7decomp::color_rgba*>(
                                decoded.data()))) {
                        return false;
                    }
                } else {
                    DecodeAgcBc3Block(
                        blocks.data() + source, decoded);
                }
                for (std::uint32_t local_y = 0;
                     local_y < 4u; ++local_y) {
                    const auto y = block_y * 4u + local_y;
                    if (y >= draw.texture_height) {
                        continue;
                    }
                    for (std::uint32_t local_x = 0;
                         local_x < 4u; ++local_x) {
                        const auto x =
                            block_x * 4u + local_x;
                        if (x >= draw.texture_width) {
                            continue;
                        }
                        const auto destination =
                            (static_cast<std::size_t>(y) *
                                 draw.texture_width +
                             x) *
                            4u;
                        const auto& source_pixel =
                            decoded[local_y * 4u + local_x];
                        std::memcpy(
                            rgba.data() + destination,
                            source_pixel.data(),
                            source_pixel.size());
                    }
                }
            }
        }
        return true;
    }
    return false;
}

bool TryReadGameMakerTransform(
    const AgcDiagnosticDraw& draw,
    std::uint64_t descriptor_table_address,
    std::array<float, 32>& transform,
    std::uint64_t& transform_address);

bool TryCompositeAgcCpuDraw(
    const AgcDiagnosticDraw& draw,
    AgcCpuFrame& frame,
    const std::size_t draw_ordinal,
    const bool ignore_black_rgb_tint,
    std::vector<AgcCpuDecodedTexture>& texture_cache,
    std::vector<Lsx4::Ps5Desktop::VulkanGuestPass>* const gpu_passes,
    const bool target_source = false,
    const std::uint64_t texture_probe_serial = 0u) {
    constexpr std::uint32_t FrameWidth = AgcCpuFrameWidth;
    constexpr std::uint32_t FrameHeight = AgcCpuFrameHeight;
    static std::atomic<std::uint64_t> next_batch_id{1u};
    const auto game_maker_fill =
        draw.draw_count == 6u &&
        (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
            UINT32_C(0xf13);
    const auto game_maker_sprite =
        (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
            UINT32_C(0x311);
    const auto game_maker_solid =
        game_maker_fill &&
        draw.texture_address != 0 &&
        draw.texture_width == 5u &&
        draw.texture_height == 1u &&
        draw.texture_tile_mode == 0u;
    const auto is_clear_packet =
        draw.draw_count == 3u &&
        draw.texture_address != 0 &&
        draw.texture_width == 5u &&
        draw.texture_height == 1u &&
        draw.texture_tile_mode == 0u;
    if (is_clear_packet) {
        std::array<float, 4> clear{};
        const auto clear_address = draw.texture_address >> 8u;
        if (!TryReadGuestBytes(
                clear_address, clear.data(), sizeof(clear)) ||
            !std::ranges::all_of(
                clear, [](const float value) {
                    return std::isfinite(value) &&
                        value >= 0.0f && value <= 1.0f;
                })) {
            return false;
        }
        std::array<std::uint8_t, 4> rgba{};
        for (std::size_t channel = 0; channel < rgba.size();
             ++channel) {
            rgba[channel] = static_cast<std::uint8_t>(
                std::clamp(
                    clear[channel] * 255.0f + 0.5f,
                    0.0f, 255.0f));
        }
        if (frame.gpu_frame == nullptr) {
            frame.gpu_frame = std::make_shared<
                Lsx4::Ps5Desktop::VulkanGuestFrame>();
        }
        frame.gpu_frame->batch_id =
            next_batch_id.fetch_add(1u, std::memory_order_relaxed);
        frame.gpu_frame->base_batch_id = 0u;
        frame.gpu_frame->target_key =
            draw.render_target_address;
        frame.gpu_frame->first_new_draw = 0u;
        frame.gpu_frame->preserve_target = false;
        frame.gpu_frame->clear_target = true;
        frame.gpu_frame->width = FrameWidth;
        frame.gpu_frame->height = FrameHeight;
        std::memcpy(
            &frame.gpu_frame->clear_rgba,
            rgba.data(), sizeof(frame.gpu_frame->clear_rgba));
        frame.gpu_frame->draws.clear();
        if (gpu_passes != nullptr) {
            gpu_passes->push_back({
                .target_key = draw.render_target_address,
                .clear_rgba = frame.gpu_frame->clear_rgba,
                .clear_target = true});
        }
        frame.width = FrameWidth;
        frame.height = FrameHeight;
        frame.rgba.clear();
        frame.rendered_draws = 1u;
        frame.covered_pixels =
            static_cast<std::uint64_t>(FrameWidth) * FrameHeight;
        if (draw_ordinal < 64u) {
            const auto bit = UINT64_C(1) << draw_ordinal;
            frame.eligible_draw_mask |= bit;
            frame.rendered_draw_mask |= bit;
            frame.draw_coverage[draw_ordinal] =
                frame.covered_pixels;
        }
        return true;
    }
    if (draw.texture_address == 0 ||
        (!game_maker_solid && !target_source &&
         (draw.texture_width < 16u ||
          draw.texture_height < 16u ||
          draw.texture_width > 4096u ||
          draw.texture_height > 4096u ||
          !(((draw.texture_tile_mode == 0u ||
              draw.texture_tile_mode == 5u) &&
             (draw.texture_format == UINT32_C(0x1) ||
              draw.texture_format == UINT32_C(0x38) ||
              draw.texture_format == UINT32_C(0xad) ||
              draw.texture_format == UINT32_C(0xb5))) ||
            (draw.texture_tile_mode == UINT32_C(27) &&
             draw.texture_format == UINT32_C(0x47))))) ||
        draw.sh_ps_program_hi != 0u ||
        (draw.sh_ps_program_lo & UINT32_C(0xff000000)) !=
            UINT32_C(0x20000000) ||
        (draw.index_buffer_address == 0 &&
         draw.draw_count != 3u &&
         draw.draw_count != 6u &&
         !((draw.primitive_type == UINT32_C(0x6) ||
            draw.primitive_type == UINT32_C(0x11)) &&
           draw.draw_count == 4u)) ||
        draw.draw_count < 3u || draw.draw_count > 65536u ||
        (draw.index_buffer_address != 0u &&
         draw.index_size != 0u && draw.index_size != 1u)) {
        return false;
    }
    if (draw_ordinal < 64u) {
        frame.eligible_draw_mask |=
            UINT64_C(1) << draw_ordinal;
    }
#ifdef __ANDROID__
    const auto log_cpu_failure =
        [&](const char* const stage,
            const std::uint32_t detail = 0u) {
            static std::atomic<std::uint32_t> failure_logs{};
            const auto index = failure_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (index < 4u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu draw skip stage=%s tex=0x%llx %ux%u "
                    "tile=%u count=%u offset=%u detail=%u",
                    stage,
                    static_cast<unsigned long long>(
                        draw.texture_address),
                    draw.texture_width, draw.texture_height,
                    draw.texture_tile_mode, draw.draw_count,
                    draw.index_offset, detail);
            }
        };
#endif

    const auto index_bytes =
        draw.index_size == 1u ? sizeof(std::uint32_t)
                              : sizeof(std::uint16_t);
    if (draw.draw_count >
        std::numeric_limits<std::size_t>::max() / index_bytes) {
        return false;
    }
    const auto index_address =
        draw.index_buffer_address +
        static_cast<std::uint64_t>(draw.index_offset) *
            index_bytes;
    static thread_local std::vector<std::uint8_t> raw_indices;
    const bool synthetic_strip =
        draw.index_buffer_address == 0u &&
        (draw.primitive_type == UINT32_C(0x6) ||
         draw.primitive_type == UINT32_C(0x11)) &&
        draw.draw_count == 4u;
    const bool synthetic_indices =
        draw.index_buffer_address == 0u &&
        (draw.draw_count == 3u || draw.draw_count == 6u ||
         synthetic_strip);
    raw_indices.resize(
        static_cast<std::size_t>(draw.draw_count) * index_bytes);
    if (!synthetic_indices &&
        !TryReadGuestBytes(
            index_address, raw_indices.data(),
            raw_indices.size())) {
#ifdef __ANDROID__
        log_cpu_failure("indices");
#endif
        return false;
    }
    static thread_local std::vector<std::uint32_t> indices;
    indices.resize(static_cast<std::size_t>(draw.draw_count));
    std::uint32_t maximum_index{};
    for (std::uint32_t index = 0; index < draw.draw_count;
         ++index) {
        if (synthetic_indices) {
            constexpr std::array<std::uint32_t, 6> QuadIndices{
                0u, 1u, 2u, 2u, 3u, 0u};
            indices[index] =
                draw.draw_count == 3u || synthetic_strip
                ? index
                : QuadIndices[index];
        } else if (draw.index_size == 1u) {
            std::memcpy(
                &indices[index],
                raw_indices.data() +
                    static_cast<std::size_t>(index) * 4u,
                sizeof(std::uint32_t));
        } else {
            std::uint16_t value{};
            std::memcpy(
                &value,
                raw_indices.data() +
                    static_cast<std::size_t>(index) * 2u,
                sizeof(value));
            indices[index] = value;
        }
        maximum_index = std::max(maximum_index, indices[index]);
    }
    static thread_local std::vector<std::uint8_t> raw_vertices;
    std::uint64_t vertex_address{};
    std::uint32_t vertex_stride{};
    std::uint64_t vertex_descriptor_table{};
    bool game_maker_vertex_group{};
    static thread_local std::vector<std::uint8_t>
        cached_raw_vertices;
    static thread_local std::uint64_t cached_vertex_address{};
    static thread_local std::uint32_t cached_vertex_stride{};
    static thread_local std::uint32_t cached_vertex_count{};
    static thread_local std::uint32_t cached_vertex_shader{};
    static thread_local bool cached_game_maker_vertex_group{};
    if (synthetic_indices &&
        (draw.draw_count == 3u || synthetic_strip)) {
        constexpr std::uint32_t SyntheticStride = 24u;
        constexpr std::array<std::array<float, 6>, 4>
            FullscreenGeometry{{
                {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
                {0.0f, 2.0f, 0.0f, 1.0f, 0.0f, 2.0f},
                {2.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f},
                {2.0f, 2.0f, 0.0f, 1.0f, 2.0f, 2.0f},
            }};
        raw_vertices.resize(
            (synthetic_strip ? 4u : 3u) * SyntheticStride);
        for (std::size_t index = 0;
             index < (synthetic_strip ? 4u : 3u); ++index) {
            std::memcpy(
                raw_vertices.data() + index * SyntheticStride,
                FullscreenGeometry[index].data(),
                SyntheticStride);
        }
        vertex_stride = SyntheticStride;
    } else if (TryReadAgcCpuVertexBuffer(
            draw, indices, raw_vertices,
            vertex_address, vertex_stride,
            vertex_descriptor_table,
            game_maker_vertex_group)) {
        cached_raw_vertices = raw_vertices;
        cached_vertex_address = vertex_address;
        cached_vertex_stride = vertex_stride;
        cached_vertex_count = maximum_index + 1u;
        cached_vertex_shader = draw.sh_es_program_lo;
        cached_game_maker_vertex_group =
            game_maker_vertex_group;
    } else if (
               cached_vertex_shader == draw.sh_es_program_lo &&
               cached_vertex_count >= maximum_index + 1u &&
               cached_vertex_stride != 0u &&
               ((draw.texture_width ==
                     draw.render_target_width &&
                 draw.texture_height ==
                     draw.render_target_height) ||
                (draw.render_target_width >= 2560u &&
                 draw.render_target_height >= 1440u &&
                 draw.texture_width ==
                     draw.render_target_width / 2u &&
                 draw.texture_height ==
                     draw.render_target_height / 2u)) &&
               cached_raw_vertices.size() >=
                   static_cast<std::size_t>(maximum_index + 1u) *
                       cached_vertex_stride) {
        raw_vertices.assign(
            cached_raw_vertices.begin(),
            cached_raw_vertices.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(
                        maximum_index + 1u) *
                    cached_vertex_stride));
        vertex_address = cached_vertex_address;
        vertex_stride = cached_vertex_stride;
        game_maker_vertex_group =
            cached_game_maker_vertex_group;
    } else {
#ifdef __ANDROID__
        log_cpu_failure("vertices", maximum_index);
#endif
        return false;
    }
    static thread_local std::vector<AgcCpuVertex> vertices;
    static thread_local std::vector<std::array<float, 2>> raw_positions;
    vertices.resize(maximum_index + 1u);
    raw_positions.resize(maximum_index + 1u);
    std::array<float, 4> raw_color0{
        1.0f, 1.0f, 1.0f, 1.0f};
    for (std::uint32_t index = 0; index <= maximum_index;
         ++index) {
        const auto* const source =
            raw_vertices.data() +
            static_cast<std::size_t>(index) * vertex_stride;
        std::memcpy(&vertices[index].x, source + 0u, 4u);
        std::memcpy(&vertices[index].y, source + 4u, 4u);
        std::memcpy(&vertices[index].z, source + 8u, 4u);
        raw_positions[index] = {
            vertices[index].x, vertices[index].y};
        if (vertex_stride == 16u &&
            (draw.sh_es_program_lo & UINT32_C(0xfff)) ==
                UINT32_C(0xc40)) {
            vertices[index].u =
                static_cast<float>(source[12u]) / 255.0f;
            vertices[index].v =
                static_cast<float>(source[13u]) / 255.0f;
            vertices[index].color = {
                1.0f, 1.0f, 1.0f, 1.0f};
        } else {
            const auto uv_offset =
                vertex_stride == 36u ? 28u : 16u;
            std::memcpy(
                &vertices[index].u,
                source + uv_offset + 0u, 4u);
            std::memcpy(
                &vertices[index].v,
                source + uv_offset + 4u, 4u);
        }
        if (vertex_stride == 112u &&
            (draw.sh_es_program_lo & UINT32_C(0xfff)) ==
                UINT32_C(0xaee)) {
            std::memcpy(
                &vertices[index].u, source + 64u, 4u);
            std::memcpy(
                &vertices[index].v, source + 68u, 4u);
            for (std::size_t channel = 0; channel < 4u;
                 ++channel) {
                float color{};
                std::memcpy(
                    &color, source + 16u + channel * 4u, 4u);
                vertices[index].color[channel] =
                    std::clamp(color, 0.0f, 1.0f);
            }
        } else if (vertex_stride == 36u) {
            std::memcpy(
                vertices[index].color.data(),
                source + 12u, sizeof(float) * 4u);
            if (index == 0u) {
                raw_color0 = vertices[index].color;
            }
            for (auto& channel : vertices[index].color) {
                channel = std::clamp(channel, 0.0f, 1.0f);
            }
            if (ignore_black_rgb_tint &&
                vertices[index].color[0] == 0.0f &&
                vertices[index].color[1] == 0.0f &&
                vertices[index].color[2] == 0.0f &&
                vertices[index].color[3] > 0.0f) {
                vertices[index].color[0] = 1.0f;
                vertices[index].color[1] = 1.0f;
                vertices[index].color[2] = 1.0f;
            }
        } else if (vertex_stride == 24u) {
            float multiplier{};
            std::memcpy(
                &multiplier, source + 12u, sizeof(multiplier));
            vertices[index].color = {
                multiplier, multiplier,
                multiplier, multiplier};
        } else if (vertex_stride != 16u) {
            for (std::size_t channel = 0; channel < 4u;
                 ++channel) {
                vertices[index].color[channel] =
                    static_cast<float>(source[12u + channel]) /
                    255.0f;
            }
        }
        if (!std::isfinite(vertices[index].x) ||
            !std::isfinite(vertices[index].y) ||
            !std::isfinite(vertices[index].z) ||
            !std::isfinite(vertices[index].u) ||
            !std::isfinite(vertices[index].v)) {
            return false;
        }
    }
    if (game_maker_solid) {
        std::array<float, 4> solid_color{};
        const auto solid_address = draw.texture_address >> 8u;
        if (!TryReadGuestBytes(
                solid_address, solid_color.data(),
                sizeof(solid_color)) ||
            !std::ranges::all_of(
                solid_color,
                [](const float value) {
                    return std::isfinite(value) &&
                        value >= 0.0f && value <= 1.0f;
                })) {
            return false;
        }
        for (auto& vertex : vertices) {
            vertex.color = solid_color;
        }
    }
#ifdef __ANDROID__
    static std::atomic<bool> observed_game_maker_character{};
    if (draw.texture_width == 256u &&
        draw.texture_height == 512u &&
        draw.draw_count == 150u) {
        observed_game_maker_character.store(
            true, std::memory_order_relaxed);
    }
    if (observed_game_maker_character.load(
            std::memory_order_relaxed) &&
        ((draw.texture_width == 980u &&
          draw.texture_height == 347u) ||
         (draw.texture_width == 256u &&
          draw.texture_height == 512u))) {
        static std::atomic<std::uint32_t> gameplay_sprite_logs{};
        if (gameplay_sprite_logs.fetch_add(
                1u, std::memory_order_relaxed) < 12u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "gm gameplay sprite=%ux%u count=%u "
                "color=%.6f p0=%.1f,%.1f uv0=%.4f,%.4f "
                "p1=%.1f,%.1f",
                draw.texture_width, draw.texture_height,
                draw.draw_count, vertices[0].color[0],
                vertices[0].x, vertices[0].y,
                vertices[0].u, vertices[0].v,
                vertices.size() > 1u ? vertices[1].x : 0.0f,
                vertices.size() > 1u ? vertices[1].y : 0.0f);
        }
    }
    if (game_maker_solid) {
        static std::atomic<std::uint32_t> solid_vertex_logs{};
        if (solid_vertex_logs.fetch_add(
                1u, std::memory_order_relaxed) < 12u) {
            const auto* const source = raw_vertices.data();
            std::uint32_t packed_color{};
            float color_as_float{};
            std::memcpy(
                &packed_color, source + 12u,
                sizeof(packed_color));
            std::memcpy(
                &color_as_float, source + 12u,
                sizeof(color_as_float));
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "gm solid vertex=0x%llx stride=%u "
                "p0=%.1f,%.1f packed=%08x float=%g "
                "rgba=%.3f,%.3f,%.3f,%.3f",
                static_cast<unsigned long long>(vertex_address),
                vertex_stride, vertices[0].x, vertices[0].y,
                packed_color, color_as_float,
                vertices[0].color[0], vertices[0].color[1],
                vertices[0].color[2], vertices[0].color[3]);
        }
    }
#endif
    bool positions_are_render_target_coordinates{};
    const auto vertex_shader_suffix =
        draw.sh_es_program_lo & UINT32_C(0xfff);
    if (game_maker_vertex_group && !target_source) {
        std::array<float, 32> transforms{};
        std::uint64_t transform_address{};
        if (TryReadGameMakerTransform(
                draw, vertex_descriptor_table,
                transforms, transform_address)) {
            bool valid_transform = true;
            for (std::size_t index = 0;
                 index < vertices.size(); ++index) {
                const std::array<float, 4> input{
                    vertices[index].x,
                    vertices[index].y,
                    vertices[index].z,
                    1.0f};
                std::array<float, 4> view{};
                std::array<float, 4> clip{};
                for (std::size_t row = 0; row < 4u; ++row) {
                    for (std::size_t column = 0;
                         column < 4u; ++column) {
                        view[row] +=
                            transforms[
                                16u + column * 4u + row] *
                            input[column];
                    }
                }
                for (std::size_t row = 0; row < 4u; ++row) {
                    for (std::size_t column = 0;
                         column < 4u; ++column) {
                        clip[row] +=
                            transforms[column * 4u + row] *
                            view[column];
                    }
                }
                if (!std::ranges::all_of(
                        clip,
                        [](const float value) {
                            return std::isfinite(value);
                        }) ||
                    std::abs(clip[3]) < 0.000001f) {
                    valid_transform = false;
                    break;
                }
                const auto reciprocal_w = 1.0f / clip[3];
                vertices[index].x =
                    (clip[0] * reciprocal_w * 0.5f + 0.5f) *
                    static_cast<float>(draw.render_target_width);
                vertices[index].y =
                    (0.5f - clip[1] * reciprocal_w * 0.5f) *
                    static_cast<float>(draw.render_target_height);
            }
            if (valid_transform) {
                positions_are_render_target_coordinates = true;
            } else {
                for (std::size_t index = 0;
                     index < vertices.size(); ++index) {
                    vertices[index].x =
                        raw_positions[index][0];
                    vertices[index].y =
                        raw_positions[index][1];
                }
            }
        }
    } else if (vertex_shader_suffix == UINT32_C(0xbaa) ||
        vertex_shader_suffix == UINT32_C(0xaee) ||
        vertex_shader_suffix == UINT32_C(0xc40)) {
        const auto transform_address =
            static_cast<std::uint64_t>(
                draw.sh_gs_user_data[0]) |
            (static_cast<std::uint64_t>(
                 draw.sh_gs_user_data[1] & 0xffffu) << 32u);
        std::array<float, 32> transforms{};
        const auto has_transforms =
            transform_address != 0u &&
            TryReadGuestBytes(
                transform_address, transforms.data(),
                sizeof(transforms)) &&
            std::ranges::all_of(
                transforms,
                [](const float value) {
                    return std::isfinite(value);
                });
        if (has_transforms) {
            bool valid_transform = true;
            std::vector<std::array<float, 4>> clip_positions(
                vertices.size());
            for (std::size_t index = 0;
                 index < vertices.size(); ++index) {
                const std::array<float, 4> input{
                    vertices[index].x,
                    vertices[index].y,
                    vertices[index].z,
                    1.0f};
                std::array<float, 4> object{};
                std::array<float, 4> clip{};
                for (std::size_t row = 0; row < 4u; ++row) {
                    object[row] =
                        transforms[row * 4u + 0u] * input[0] +
                        transforms[row * 4u + 1u] * input[1] +
                        transforms[row * 4u + 2u] * input[2] +
                        transforms[row * 4u + 3u] * input[3];
                }
                for (std::size_t row = 0; row < 4u; ++row) {
                    clip[row] =
                        transforms[16u + row * 4u + 0u] *
                            object[0] +
                        transforms[16u + row * 4u + 1u] *
                            object[1] +
                        transforms[16u + row * 4u + 2u] *
                            object[2] +
                        transforms[16u + row * 4u + 3u] *
                            object[3];
                }
                clip_positions[index] = clip;
                if (!std::ranges::all_of(
                        clip,
                        [](const float value) {
                            return std::isfinite(value);
                        }) ||
                    std::abs(clip[3]) < 0.000001f) {
                    valid_transform = false;
                    break;
                }
                const auto reciprocal_w = 1.0f / clip[3];
                vertices[index].x =
                    (clip[0] * reciprocal_w * 0.5f + 0.5f) *
                    static_cast<float>(
                        draw.render_target_width);
                vertices[index].y =
                    (0.5f - clip[1] * reciprocal_w * 0.5f) *
                    static_cast<float>(
                        draw.render_target_height);
                if (!std::isfinite(vertices[index].x) ||
                    !std::isfinite(vertices[index].y)) {
                    valid_transform = false;
                    break;
                }
            }
            if (valid_transform) {
                positions_are_render_target_coordinates = true;
#ifdef __ANDROID__
                static std::atomic<std::uint32_t>
                    unity_transform_logs{};
                if (unity_transform_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 4u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "cpu unity transform cb=0x%llx "
                        "vertex=0x%llx raw=%.3f,%.3f,%.3f "
                        "clip=%.6f,%.6f,%.6f,%.6f "
                        "screen=%.3f,%.3f",
                        static_cast<unsigned long long>(
                            transform_address),
                        static_cast<unsigned long long>(
                            vertex_address),
                        raw_positions[0][0],
                        raw_positions[0][1],
                        vertices[0].z,
                        clip_positions[0][0],
                        clip_positions[0][1],
                        clip_positions[0][2],
                        clip_positions[0][3],
                        vertices[0].x,
                        vertices[0].y);
                }
#endif
            } else {
                for (std::size_t index = 0;
                     index < vertices.size(); ++index) {
                    vertices[index].x =
                        raw_positions[index][0];
                    vertices[index].y =
                        raw_positions[index][1];
                }
            }
        }
    }
    if (draw.render_target_width != 0u &&
        draw.render_target_height != 0u) {
        float minimum_x = std::numeric_limits<float>::infinity();
        float minimum_y = std::numeric_limits<float>::infinity();
        float maximum_x =
            -std::numeric_limits<float>::infinity();
        float maximum_y =
            -std::numeric_limits<float>::infinity();
        for (const auto& vertex : vertices) {
            minimum_x = std::min(minimum_x, vertex.x);
            minimum_y = std::min(minimum_y, vertex.y);
            maximum_x = std::max(maximum_x, vertex.x);
            maximum_y = std::max(maximum_y, vertex.y);
        }
        const auto uses_centered_target_coordinates =
            target_source &&
            std::abs(
                minimum_x +
                static_cast<float>(draw.render_target_width) * 0.5f) <=
                1.0f &&
            std::abs(
                maximum_x -
                static_cast<float>(draw.render_target_width) * 0.5f) <=
                1.0f &&
            std::abs(
                minimum_y +
                static_cast<float>(draw.render_target_height) * 0.5f) <=
                1.0f &&
            std::abs(
                maximum_y -
                static_cast<float>(draw.render_target_height) * 0.5f) <=
                1.0f;
        if (uses_centered_target_coordinates) {
            for (auto& vertex : vertices) {
                vertex.x +=
                    static_cast<float>(draw.render_target_width) * 0.5f;
                vertex.y +=
                    static_cast<float>(draw.render_target_height) * 0.5f;
            }
            minimum_x = 0.0f;
            minimum_y = 0.0f;
            maximum_x = static_cast<float>(draw.render_target_width);
            maximum_y = static_cast<float>(draw.render_target_height);
            positions_are_render_target_coordinates = true;
        }
        const auto uses_clip_coordinates =
            !positions_are_render_target_coordinates &&
            minimum_x >= -1.001f && maximum_x <= 1.001f &&
            minimum_y >= -1.001f && maximum_y <= 1.001f &&
            minimum_x < -0.001f && minimum_y < -0.001f &&
            maximum_x > 0.001f && maximum_y > 0.001f;
        if (uses_clip_coordinates) {
            for (auto& vertex : vertices) {
                vertex.x =
                    (vertex.x * 0.5f + 0.5f) *
                    static_cast<float>(
                        draw.render_target_width);
                vertex.y =
                    (0.5f - vertex.y * 0.5f) *
                    static_cast<float>(
                        draw.render_target_height);
            }
            positions_are_render_target_coordinates = true;
        }
        const auto uses_logical_coordinates =
            !positions_are_render_target_coordinates &&
            draw.render_target_width >= 2560u &&
            draw.render_target_height >= 1440u &&
            maximum_x <=
                static_cast<float>(
                    draw.render_target_width / 2u) +
                    1.0f &&
            maximum_y <=
                static_cast<float>(
                    draw.render_target_height / 2u) +
                    1.0f;
        const auto coordinate_width =
            uses_logical_coordinates
            ? draw.render_target_width / 2u
            : draw.render_target_width;
        const auto coordinate_height =
            uses_logical_coordinates
            ? draw.render_target_height / 2u
            : draw.render_target_height;
        for (auto& vertex : vertices) {
            vertex.x *=
                static_cast<float>(FrameWidth) /
                static_cast<float>(coordinate_width);
            vertex.y *=
                static_cast<float>(FrameHeight) /
                static_cast<float>(coordinate_height);
        }
#ifdef __ANDROID__
        if (target_source) {
            static std::atomic<std::uint32_t> copy_geometry_logs{};
            if (copy_geometry_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 20u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gm copy geometry draw=%zu vertex=0x%llx "
                    "stride=%u p=%.1f,%.1f;%.1f,%.1f;"
                    "%.1f,%.1f;%.1f,%.1f uv=%.3f,%.3f;"
                    "%.3f,%.3f",
                    draw_ordinal,
                    static_cast<unsigned long long>(vertex_address),
                    vertex_stride,
                    vertices[0].x, vertices[0].y,
                    vertices[1].x, vertices[1].y,
                    vertices[2].x, vertices[2].y,
                    vertices[3].x, vertices[3].y,
                    vertices[0].u, vertices[0].v,
                    vertices[1].u, vertices[1].v);
            }
        }
#endif
#ifdef __ANDROID__
        if (uses_clip_coordinates) {
            static std::atomic<std::uint32_t> clip_geometry_logs{};
            if (clip_geometry_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 4u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu clip geometry es=%03x raw0=%.3f,%.3f "
                    "raw1=%.3f,%.3f screen0=%.1f,%.1f "
                    "screen1=%.1f,%.1f",
                    vertex_shader_suffix,
                    raw_positions[0][0], raw_positions[0][1],
                    raw_positions[
                        std::min<std::size_t>(
                            1u, raw_positions.size() - 1u)][0],
                    raw_positions[
                        std::min<std::size_t>(
                            1u, raw_positions.size() - 1u)][1],
                    vertices[0].x, vertices[0].y,
                    vertices[
                        std::min<std::size_t>(
                            1u, vertices.size() - 1u)].x,
                    vertices[
                        std::min<std::size_t>(
                            1u, vertices.size() - 1u)].y);
            }
        }
#endif
#ifdef __ANDROID__
        if ((draw.render_target_address & UINT64_C(0xffffffff)) ==
            UINT64_C(0xd0a20000)) {
            static std::atomic<std::uint32_t>
                gameplay_geometry_logs{};
            const auto log_index = gameplay_geometry_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (log_index < 32u) {
                const auto second =
                    std::min<std::size_t>(1u, vertices.size() - 1u);
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gameplay geometry draw=%zu es=%03x ps=%03x "
                    "count=%u stride=%u transformed=%u logical=%u "
                    "raw0=%.2f,%.2f raw1=%.2f,%.2f "
                    "screen0=%.2f,%.2f screen1=%.2f,%.2f "
                    "tex=0x%llx/%ux%u/f%02x",
                    draw_ordinal,
                    draw.sh_es_program_lo & UINT32_C(0xfff),
                    draw.sh_ps_program_lo & UINT32_C(0xfff),
                    draw.draw_count, vertex_stride,
                    positions_are_render_target_coordinates ? 1u : 0u,
                    uses_logical_coordinates ? 1u : 0u,
                    raw_positions[0][0], raw_positions[0][1],
                    raw_positions[second][0],
                    raw_positions[second][1],
                    vertices[0].x, vertices[0].y,
                    vertices[second].x, vertices[second].y,
                    static_cast<unsigned long long>(
                        draw.texture_address),
                    draw.texture_width, draw.texture_height,
                    draw.texture_format);
            }
        }
#endif
    }
    {
        float minimum_x = std::numeric_limits<float>::infinity();
        float minimum_y = std::numeric_limits<float>::infinity();
        float maximum_x =
            -std::numeric_limits<float>::infinity();
        float maximum_y =
            -std::numeric_limits<float>::infinity();
        for (const auto index : indices) {
            minimum_x = std::min(minimum_x, vertices[index].x);
            minimum_y = std::min(minimum_y, vertices[index].y);
            maximum_x = std::max(maximum_x, vertices[index].x);
            maximum_y = std::max(maximum_y, vertices[index].y);
        }
        if (maximum_x < 0.0f || maximum_y < 0.0f ||
            minimum_x >= static_cast<float>(FrameWidth) ||
            minimum_y >= static_cast<float>(FrameHeight)) {
            if (draw_ordinal < 64u) {
                frame.rendered_draw_mask |=
                    UINT64_C(1) << draw_ordinal;
                frame.draw_coverage[draw_ordinal] = 0u;
            }
            return true;
        }
    }
#ifdef __ANDROID__
    if (vertex_shader_suffix == UINT32_C(0xc40)) {
        static std::atomic<std::uint32_t> c40_geometry_logs{};
        if (c40_geometry_logs.fetch_add(
                1u, std::memory_order_relaxed) < 16u) {
            const auto vertex_log =
                [&](const std::size_t slot) -> const AgcCpuVertex& {
                    return vertices[
                        std::min(slot, vertices.size() - 1u)];
                };
            const auto& v0 = vertex_log(0u);
            const auto& v1 = vertex_log(1u);
            const auto& v2 = vertex_log(2u);
            const auto& v3 = vertex_log(3u);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu c40 geometry ps=0x%08x vertex=0x%llx "
                "stride=%u count=%u indices=%u,%u,%u,%u "
                "p0=%.2f,%.2f/%.3f,%.3f "
                "p1=%.2f,%.2f/%.3f,%.3f "
                "p2=%.2f,%.2f/%.3f,%.3f "
                "p3=%.2f,%.2f/%.3f,%.3f",
                draw.sh_ps_program_lo,
                static_cast<unsigned long long>(vertex_address),
                vertex_stride, draw.draw_count,
                indices.size() > 0u ? indices[0] : 0u,
                indices.size() > 1u ? indices[1] : 0u,
                indices.size() > 2u ? indices[2] : 0u,
                indices.size() > 3u ? indices[3] : 0u,
                v0.x, v0.y, v0.u, v0.v,
                v1.x, v1.y, v1.u, v1.v,
                v2.x, v2.y, v2.u, v2.v,
                v3.x, v3.y, v3.u, v3.v);
        }
    }
#endif

    float draw_minimum_x = static_cast<float>(FrameWidth);
    float draw_minimum_y = static_cast<float>(FrameHeight);
    float draw_maximum_x{};
    float draw_maximum_y{};
    for (const auto index : indices) {
        draw_minimum_x =
            std::min(draw_minimum_x, vertices[index].x);
        draw_minimum_y =
            std::min(draw_minimum_y, vertices[index].y);
        draw_maximum_x =
            std::max(draw_maximum_x, vertices[index].x);
        draw_maximum_y =
            std::max(draw_maximum_y, vertices[index].y);
    }
    std::shared_ptr<std::vector<std::uint8_t>> texture_rgba;
    std::uint64_t texture_signature{};
    if (target_source) {
        static const auto white_texture =
            std::make_shared<std::vector<std::uint8_t>>(
                static_cast<std::size_t>(4u), UINT8_C(255));
        texture_rgba = white_texture;
        texture_signature = AgcTextureSignature(*texture_rgba);
    } else if (game_maker_solid) {
        static const auto white_texture =
            std::make_shared<std::vector<std::uint8_t>>(
                static_cast<std::size_t>(5u * 4u), UINT8_C(255));
        texture_rgba = white_texture;
        texture_signature = AgcTextureSignature(*texture_rgba);
    } else {
        auto cached_texture = std::ranges::find_if(
            texture_cache,
            [&](const AgcCpuDecodedTexture& texture) {
                return texture.address == draw.texture_address &&
                    texture.width == draw.texture_width &&
                    texture.height == draw.texture_height &&
                    texture.format == draw.texture_format &&
                    texture.tile_mode == draw.texture_tile_mode;
            });
        constexpr std::uint64_t FullProbeInterval = 15u;
        std::uint64_t current_source_probe{};
        const auto should_probe_source =
            game_maker_sprite &&
            (cached_texture == texture_cache.end() ||
             texture_probe_serial == 0u ||
             texture_probe_serial -
                     cached_texture->source_probe_serial >=
                 FullProbeInterval);
        const auto has_current_source_probe =
            should_probe_source &&
            TryAgcTextureSourceProbe(draw, current_source_probe);
        if (cached_texture != texture_cache.end() &&
            has_current_source_probe &&
            cached_texture->source_probe != current_source_probe) {
            texture_cache.erase(cached_texture);
            cached_texture = texture_cache.end();
        } else if (cached_texture != texture_cache.end() &&
                   has_current_source_probe) {
            cached_texture->source_probe_serial =
                texture_probe_serial;
        }
        if (cached_texture == texture_cache.end()) {
            AgcCpuDecodedTexture decoded{
                .address = draw.texture_address,
                .width = draw.texture_width,
                .height = draw.texture_height,
                .format = draw.texture_format,
                .tile_mode = draw.texture_tile_mode,
                .source_probe =
                    has_current_source_probe
                    ? current_source_probe
                    : 0u,
                .source_probe_serial = texture_probe_serial,
                .rgba =
                    std::make_shared<std::vector<std::uint8_t>>()};
            if (!TryReadAgcCpuTextureRgba(draw, *decoded.rgba)) {
#ifdef __ANDROID__
                log_cpu_failure("texture");
#endif
                return false;
            }
            decoded.signature =
                AgcTextureSignature(*decoded.rgba);
            texture_cache.emplace_back(std::move(decoded));
            cached_texture = std::prev(texture_cache.end());
        }
        texture_rgba = cached_texture->rgba;
        texture_signature = cached_texture->signature;
    }
    const auto& texture = *texture_rgba;
#ifdef __ANDROID__
    if (draw.texture_width == 250u &&
        draw.texture_height == 250u) {
        static std::atomic<std::uint32_t> texture250_logs{};
        if (texture250_logs.fetch_add(
                1u, std::memory_order_relaxed) < 8u) {
            std::uint32_t packed_multiplier{};
            std::memcpy(
                &packed_multiplier,
                raw_vertices.data() + 12u,
                sizeof(packed_multiplier));
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "gm250 vertex stride=%u packed=%08x color=%g "
                "p=%.1f,%.1f;%.1f,%.1f;%.1f,%.1f;%.1f,%.1f",
                vertex_stride, packed_multiplier,
                vertices[0].color[0],
                vertices[0].x, vertices[0].y,
                vertices[1].x, vertices[1].y,
                vertices[2].x, vertices[2].y,
                vertices[3].x, vertices[3].y);
        }
        static std::atomic<bool> dumped_menu_texture{};
        if (!dumped_menu_texture.exchange(
                true, std::memory_order_relaxed)) {
            if (auto* output = std::fopen(
                    "/data/user/0/app.lsx4.android/files/gm250.rgba",
                    "wb");
                output != nullptr) {
                (void)std::fwrite(
                    texture.data(), 1u, texture.size(), output);
                std::fclose(output);
            }
        }
    }
    if (draw.texture_width == 320u &&
        draw.texture_height == 512u) {
        static std::atomic<bool> dumped_ground_texture{};
        if (!dumped_ground_texture.exchange(
                true, std::memory_order_relaxed)) {
            if (auto* output = std::fopen(
                    "/data/user/0/app.lsx4.android/files/gm320x512.rgba",
                    "wb");
                output != nullptr) {
                (void)std::fwrite(
                    texture.data(), 1u, texture.size(), output);
                std::fclose(output);
            }
        }
    }
    if (game_maker_sprite) {
        static std::atomic<std::uint32_t> game_maker_texture_logs{};
        if (game_maker_texture_logs.fetch_add(
                1u, std::memory_order_relaxed) < 10u) {
            std::uint64_t alpha_zero{};
            std::uint64_t alpha_full{};
            for (std::size_t alpha = 3u;
                 alpha < texture.size(); alpha += 4u) {
                alpha_zero += texture[alpha] == 0u;
                alpha_full += texture[alpha] == 255u;
            }
            const auto& v0 = vertices[0];
            const auto& v1 = vertices[
                std::min<std::size_t>(1u, vertices.size() - 1u)];
            const auto& v2 = vertices[
                std::min<std::size_t>(2u, vertices.size() - 1u)];
            const auto& v3 = vertices[
                std::min<std::size_t>(3u, vertices.size() - 1u)];
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "gm texture=0x%llx %ux%u alpha0=%llu "
                "alpha255=%llu uv0=%.6f,%.6f uv1=%.6f,%.6f "
                "uv2=%.6f,%.6f uv3=%.6f,%.6f",
                static_cast<unsigned long long>(
                    draw.texture_address),
                draw.texture_width, draw.texture_height,
                static_cast<unsigned long long>(alpha_zero),
                static_cast<unsigned long long>(alpha_full),
                v0.u, v0.v, v1.u, v1.v,
                v2.u, v2.v, v3.u, v3.v);
        }
    }
    {
        static std::atomic<std::uint32_t> texture_probe_logs{};
        const auto probe_index = texture_probe_logs.fetch_add(
            1u, std::memory_order_relaxed);
        if (probe_index < 2u) {
            std::array<std::uint64_t, 4> nonzero{};
            std::array<std::uint8_t, 4> maximum{};
            const auto pixel_stride = std::max<std::size_t>(
                std::min<std::size_t>(
                    static_cast<std::size_t>(
                        draw.texture_width) *
                        draw.texture_height,
                    texture.size() / 4u) /
                    4096u,
                1u);
            const auto texture_pixel_count =
                std::min<std::size_t>(
                    static_cast<std::size_t>(
                        draw.texture_width) *
                        draw.texture_height,
                    texture.size() / 4u);
            for (std::size_t pixel = 0;
                 pixel < texture_pixel_count;
                 pixel += pixel_stride) {
                for (std::size_t channel = 0; channel < 4u;
                     ++channel) {
                    const auto value =
                        texture[pixel * 4u + channel];
                    nonzero[channel] += value != 0u;
                    maximum[channel] =
                        std::max(maximum[channel], value);
                }
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu texture probe=0x%llx nz=%llu,%llu,%llu,%llu "
                "max=%u,%u,%u,%u raw_color0=%.3f,%.3f,%.3f,%.3f "
                "color0=%.3f,%.3f,%.3f,%.3f es=0x%08x "
                "ps=0x%08x size=%ux%u",
                static_cast<unsigned long long>(
                    draw.texture_address),
                static_cast<unsigned long long>(nonzero[0]),
                static_cast<unsigned long long>(nonzero[1]),
                static_cast<unsigned long long>(nonzero[2]),
                static_cast<unsigned long long>(nonzero[3]),
                maximum[0], maximum[1], maximum[2], maximum[3],
                raw_color0[0], raw_color0[1],
                raw_color0[2], raw_color0[3],
                vertices[0].color[0], vertices[0].color[1],
                vertices[0].color[2], vertices[0].color[3],
                draw.sh_es_program_lo, draw.sh_ps_program_lo,
                draw.texture_width, draw.texture_height);
        }
    }
#endif

    if (UseVulkanGuestRaster()) {
        if (frame.gpu_frame == nullptr) {
            frame.gpu_frame =
                std::make_shared<
                    Lsx4::Ps5Desktop::VulkanGuestFrame>();
            frame.gpu_frame->target_key =
                draw.render_target_address;
            frame.gpu_frame->width = FrameWidth;
            frame.gpu_frame->height = FrameHeight;
        }
        if (frame.rendered_draws == 0u) {
            frame.gpu_frame->batch_id =
                next_batch_id.fetch_add(
                    1u, std::memory_order_relaxed);
        }
        Lsx4::Ps5Desktop::VulkanGuestDraw gpu_draw{};
        gpu_draw.nearest_texture =
            game_maker_sprite ||
            (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
                UINT32_C(0x459);
        gpu_draw.repeat_texture =
            game_maker_sprite &&
            std::ranges::any_of(
                vertices,
                [](const AgcCpuVertex& vertex) {
                    return vertex.u < 0.0f || vertex.u > 1.0f ||
                        vertex.v < 0.0f || vertex.v > 1.0f;
                });
        gpu_draw.texture_key =
            draw.texture_address ^
            (static_cast<std::uint64_t>(draw.texture_width) << 32u) ^
            (static_cast<std::uint64_t>(draw.texture_height) << 48u) ^
            (static_cast<std::uint64_t>(draw.texture_format) << 24u) ^
            (gpu_draw.repeat_texture
                 ? UINT64_C(0x8000000000000000) : 0u) ^
            (gpu_draw.nearest_texture
                 ? UINT64_C(0x4000000000000000) : 0u);
        gpu_draw.texture_address = draw.texture_address;
        gpu_draw.texture_signature = texture_signature;
        gpu_draw.texture_width = draw.texture_width;
        gpu_draw.texture_height = draw.texture_height;
        gpu_draw.instance_count = draw.instance_count;
        gpu_draw.require_target_source = target_source;
        // SharpEmu/AGC defines CB_BLEND_CONTROL.ENABLE at bit 30.
        // The factor fields remain populated while blending is disabled
        // (notably 0x25040504 on GameMaker render-target copies), so using
        // SRC_BLEND==0 as the enable test incorrectly alpha-composites the
        // final surface over its previous contents.
        const auto blend_enabled =
            ((draw.cx_blend0_control >> 30u) & 1u) != 0u;
        const auto color_source_factor =
            draw.cx_blend0_control & UINT32_C(0x1f);
        const auto color_destination_factor =
            (draw.cx_blend0_control >> 8u) & UINT32_C(0x1f);
        const auto game_maker_blend =
            game_maker_fill || game_maker_sprite ||
            (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
                UINT32_C(0x459);
        gpu_draw.extended_blend_contract = game_maker_blend;
        if (game_maker_blend) {
            gpu_draw.opaque =
                game_maker_solid || !blend_enabled ||
                (color_destination_factor == 0u &&
                 (!((draw.cx_blend0_control >> 29u) & 1u) ||
                  ((draw.cx_blend0_control >> 24u) &
                    UINT32_C(0x1f)) == 0u));
            gpu_draw.premultiplied_alpha =
                blend_enabled &&
                color_source_factor == 1u &&
                color_destination_factor == 5u;
            gpu_draw.destination_source_alpha =
                blend_enabled && color_source_factor == 0u &&
                color_destination_factor == 4u;
            gpu_draw.destination_inverse_source_alpha =
                blend_enabled && color_source_factor == 0u &&
                color_destination_factor == 5u;
            gpu_draw.additive =
                blend_enabled && color_source_factor == 1u &&
                color_destination_factor == 1u;
        } else {
            // Preserve the pre-GameMaker path used by the first PS5 title.
            // Its 0x60040008 pass keeps populated blend fields with a
            // destination factor of zero and must not become an opaque draw.
            gpu_draw.opaque =
                (draw.cx_blend0_control & UINT32_C(0x1f)) == 0u;
            gpu_draw.premultiplied_alpha =
                (draw.cx_blend0_control & UINT32_C(0x1f)) == 1u &&
                ((draw.cx_blend0_control >> 8u) &
                 UINT32_C(0x1f)) == 5u;
        }
        gpu_draw.wave_effect =
            (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
            UINT32_C(0x459);
        if (gpu_draw.wave_effect) {
            const std::array<std::uint64_t, 5> srt_addresses{
                (static_cast<std::uint64_t>(
                     draw.sh_ps_user_data[1] & 0xffu) << 32u) |
                    draw.sh_ps_user_data[0],
                (static_cast<std::uint64_t>(
                     draw.sh_export_user_data[1] & 0xffu) << 32u) |
                    draw.sh_export_user_data[0],
                (static_cast<std::uint64_t>(
                     draw.sh_gs_user_data[1] & 0xffu) << 32u) |
                    draw.sh_gs_user_data[0],
                (static_cast<std::uint64_t>(
                     draw.sh_es_user_data[1] & 0xffu) << 32u) |
                    draw.sh_es_user_data[0],
                (static_cast<std::uint64_t>(
                     draw.sh_vs_user_data[1] & 0xffu) << 32u) |
                    draw.sh_vs_user_data[0]};
            bool found_parameters{};
            const auto try_parameters =
                [&](const std::uint64_t buffer_address) {
                    std::array<float, 32> constants{};
                    if (buffer_address == 0u ||
                        !TryReadGuestBytes(
                            buffer_address, constants.data(),
                            sizeof(constants)) ||
                        !std::isfinite(constants[0]) ||
                        !std::isfinite(constants[1]) ||
                        constants[0] <= 0.0f ||
                        constants[1] <= 0.0f ||
                        std::abs(
                            constants[0] * constants[10] -
                            0.5f) > 0.02f ||
                        std::abs(
                            constants[1] * constants[11] -
                            0.5f) > 0.02f ||
                        constants[10] < 160.0f ||
                        constants[10] > 2048.0f ||
                        constants[11] < 90.0f ||
                        constants[11] > 2048.0f) {
                        return false;
                    }
                    gpu_draw.wave_parameters = {
                        constants[0], constants[1],
                        constants[12], constants[16],
                        constants[17], constants[18],
                        constants[19], constants[20],
                        constants[21], constants[22],
                        constants[23], constants[24]};
                    found_parameters = true;
                    return true;
                };
            std::vector<std::uint64_t> tables;
            for (const auto srt_address : srt_addresses) {
                if (srt_address != 0u &&
                    !std::ranges::contains(tables, srt_address)) {
                    tables.push_back(srt_address);
                }
            }
            const auto append_user_data_addresses =
                [&](const auto& user_data) {
                    for (std::size_t word = 0;
                         word + 1u < user_data.size(); ++word) {
                        const auto address =
                            static_cast<std::uint64_t>(
                                user_data[word]) |
                            (static_cast<std::uint64_t>(
                                 user_data[word + 1u] & 0xffffu)
                             << 32u);
                        if (address != 0u &&
                            !std::ranges::contains(
                                tables, address)) {
                            tables.push_back(address);
                        }
                    }
                };
            append_user_data_addresses(draw.sh_ps_user_data);
            append_user_data_addresses(
                draw.sh_export_user_data);
            append_user_data_addresses(draw.sh_gs_user_data);
            append_user_data_addresses(draw.sh_es_user_data);
            append_user_data_addresses(draw.sh_vs_user_data);
            for (std::size_t table_index = 0;
                 table_index < tables.size() &&
                 table_index < 128u && !found_parameters;
                 ++table_index) {
                const auto table_address = tables[table_index];
                if (try_parameters(table_address)) {
                    break;
                }
                std::array<std::uint32_t, 256> words{};
                if (!TryReadGuestBytes(
                        table_address, words.data(),
                        sizeof(words))) {
                    continue;
                }
                for (std::size_t word = 0;
                     word + 1u < words.size() &&
                     !found_parameters; ++word) {
                    const auto nested_address =
                        static_cast<std::uint64_t>(words[word]) |
                        (static_cast<std::uint64_t>(
                             words[word + 1u] & 0xffffu) << 32u);
                    if (try_parameters(nested_address)) {
                        break;
                    }
                    if (nested_address != 0u &&
                        tables.size() < 128u &&
                        !std::ranges::contains(
                            tables, nested_address)) {
                        tables.push_back(nested_address);
                    }
                }
            }
#ifdef __ANDROID__
            static std::atomic<std::uint32_t> wave_parameter_logs{};
            if (wave_parameter_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 16u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gm wave uniforms found=%u values="
                    "%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,"
                    "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                    found_parameters ? 1u : 0u,
                    gpu_draw.wave_parameters[0],
                    gpu_draw.wave_parameters[1],
                    gpu_draw.wave_parameters[2],
                    gpu_draw.wave_parameters[3],
                    gpu_draw.wave_parameters[4],
                    gpu_draw.wave_parameters[5],
                    gpu_draw.wave_parameters[6],
                    gpu_draw.wave_parameters[7],
                    gpu_draw.wave_parameters[8],
                    gpu_draw.wave_parameters[9],
                    gpu_draw.wave_parameters[10],
                    gpu_draw.wave_parameters[11]);
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gm wave psud="
                    "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                    draw.sh_ps_user_data[0],
                    draw.sh_ps_user_data[1],
                    draw.sh_ps_user_data[2],
                    draw.sh_ps_user_data[3],
                    draw.sh_ps_user_data[4],
                    draw.sh_ps_user_data[5],
                    draw.sh_ps_user_data[6],
                    draw.sh_ps_user_data[7]);
                std::array<std::uint32_t, 16> ps_probe{};
                (void)TryReadGuestBytes(
                    srt_addresses[0], ps_probe.data(),
                    sizeof(ps_probe));
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gm wave psroot=0x%llx "
                    "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,"
                    "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                    static_cast<unsigned long long>(
                        srt_addresses[0]),
                    ps_probe[0], ps_probe[1], ps_probe[2],
                    ps_probe[3], ps_probe[4], ps_probe[5],
                    ps_probe[6], ps_probe[7], ps_probe[8],
                    ps_probe[9], ps_probe[10], ps_probe[11],
                    ps_probe[12], ps_probe[13], ps_probe[14],
                    ps_probe[15]);
            }
#endif
        }
        gpu_draw.texture_rgba = texture_rgba;
        gpu_draw.vertices.reserve(vertices.size());
        for (const auto& vertex : vertices) {
            const auto channel = [&](const std::size_t index) {
                return static_cast<std::uint32_t>(
                    std::clamp(
                        vertex.color[index] * 255.0f,
                        0.0f, 255.0f) +
                    0.5f);
            };
            gpu_draw.vertices.push_back({
                vertex.x,
                vertex.y,
                vertex.u,
                vertex.v,
                channel(0) |
                    (channel(1) << 8u) |
                    (channel(2) << 16u) |
                    (channel(3) << 24u)});
        }
        if ((draw.primitive_type == UINT32_C(0x6) ||
             draw.primitive_type == UINT32_C(0x11)) &&
            indices.size() >= 3u) {
            gpu_draw.indices.reserve(
                (indices.size() - 2u) * 3u);
            for (std::size_t index = 2u;
                 index < indices.size(); ++index) {
                if ((index & 1u) == 0u) {
                    gpu_draw.indices.insert(
                        gpu_draw.indices.end(),
                        {indices[index - 2u],
                         indices[index - 1u],
                         indices[index]});
                } else {
                    gpu_draw.indices.insert(
                        gpu_draw.indices.end(),
                        {indices[index - 1u],
                         indices[index - 2u],
                         indices[index]});
                }
            }
        } else {
            gpu_draw.indices.assign(
                indices.begin(), indices.end());
        }
        frame.width = FrameWidth;
        frame.height = FrameHeight;
        float minimum_x = static_cast<float>(FrameWidth);
        float minimum_y = static_cast<float>(FrameHeight);
        float maximum_x{};
        float maximum_y{};
        for (const auto index : indices) {
            minimum_x = std::min(minimum_x, vertices[index].x);
            minimum_y = std::min(minimum_y, vertices[index].y);
            maximum_x = std::max(maximum_x, vertices[index].x);
            maximum_y = std::max(maximum_y, vertices[index].y);
        }
        const auto first_x = std::clamp(
            static_cast<int>(std::floor(minimum_x)),
            0, static_cast<int>(FrameWidth));
        const auto last_x = std::clamp(
            static_cast<int>(std::ceil(maximum_x)),
            0, static_cast<int>(FrameWidth));
        const auto first_y = std::clamp(
            static_cast<int>(std::floor(minimum_y)),
            0, static_cast<int>(FrameHeight));
        const auto last_y = std::clamp(
            static_cast<int>(std::ceil(maximum_y)),
            0, static_cast<int>(FrameHeight));
        const auto coverage =
            static_cast<std::uint64_t>(
                std::max(last_x - first_x, 0)) *
            static_cast<std::uint64_t>(
                std::max(last_y - first_y, 0));
        if (gpu_passes != nullptr) {
            if (gpu_passes->empty() ||
                gpu_passes->back().target_key !=
                    draw.render_target_address) {
                gpu_passes->push_back({
                    .target_key = draw.render_target_address});
            }
            gpu_passes->back().draws.push_back(gpu_draw);
        }
        frame.gpu_frame->draws.emplace_back(std::move(gpu_draw));
        ++frame.rendered_draws;
        frame.covered_pixels += coverage;
        if (draw_ordinal < 64u) {
            frame.rendered_draw_mask |=
                UINT64_C(1) << draw_ordinal;
            frame.draw_coverage[draw_ordinal] = coverage;
        }
        return true;
    }

    if (frame.rgba.empty()) {
        frame.width = FrameWidth;
        frame.height = FrameHeight;
        frame.rgba.resize(
            static_cast<std::size_t>(FrameWidth) *
            FrameHeight * 4u);
        for (std::size_t alpha = 3u; alpha < frame.rgba.size();
             alpha += 4u) {
            frame.rgba[alpha] = 255u;
        }
    }

    const auto edge =
        [](const AgcCpuVertex& begin,
           const AgcCpuVertex& end,
           const float x,
           const float y) {
            return (x - begin.x) * (end.y - begin.y) -
                   (y - begin.y) * (end.x - begin.x);
        };
    std::uint64_t rasterized_pixels{};
    std::uint64_t covered_pixels{};
    const auto rasterize_axis_aligned_quad = [&]() {
        if (draw.draw_count != 6u) {
            return false;
        }
        std::array<std::uint32_t, 4> unique_indices{};
        std::size_t unique_count{};
        for (const auto index : indices) {
            if (std::find(
                    unique_indices.begin(),
                    unique_indices.begin() +
                        static_cast<std::ptrdiff_t>(unique_count),
                    index) ==
                unique_indices.begin() +
                    static_cast<std::ptrdiff_t>(unique_count)) {
                if (unique_count == unique_indices.size()) {
                    return false;
                }
                unique_indices[unique_count++] = index;
            }
        }
        if (unique_count != unique_indices.size()) {
            return false;
        }

        float minimum_x = std::numeric_limits<float>::infinity();
        float minimum_y = std::numeric_limits<float>::infinity();
        float maximum_x = -std::numeric_limits<float>::infinity();
        float maximum_y = -std::numeric_limits<float>::infinity();
        for (const auto index : unique_indices) {
            minimum_x = std::min(minimum_x, vertices[index].x);
            minimum_y = std::min(minimum_y, vertices[index].y);
            maximum_x = std::max(maximum_x, vertices[index].x);
            maximum_y = std::max(maximum_y, vertices[index].y);
        }
        const auto width = maximum_x - minimum_x;
        const auto height = maximum_y - minimum_y;
        if (!std::isfinite(width) || !std::isfinite(height) ||
            width < 0.001f || height < 0.001f) {
            return false;
        }
        const auto epsilon_x = std::max(0.05f, width * 0.0001f);
        const auto epsilon_y = std::max(0.05f, height * 0.0001f);
        std::array<const AgcCpuVertex*, 4> corners{};
        for (const auto index : unique_indices) {
            const auto& vertex = vertices[index];
            const bool left =
                std::abs(vertex.x - minimum_x) <= epsilon_x;
            const bool right =
                std::abs(vertex.x - maximum_x) <= epsilon_x;
            const bool top =
                std::abs(vertex.y - minimum_y) <= epsilon_y;
            const bool bottom =
                std::abs(vertex.y - maximum_y) <= epsilon_y;
            if ((!left && !right) || (!top && !bottom)) {
                return false;
            }
            const auto corner = (bottom ? 2u : 0u) +
                (right ? 1u : 0u);
            if (corners[corner] != nullptr) {
                return false;
            }
            corners[corner] = &vertex;
        }
        if (std::ranges::any_of(
                corners, [](const AgcCpuVertex* const vertex) {
                    return vertex == nullptr;
                })) {
            return false;
        }

        constexpr float AttributeEpsilon = 0.0001f;
        for (std::size_t corner = 1; corner < corners.size();
             ++corner) {
            for (std::size_t channel = 0; channel < 4u; ++channel) {
                if (std::abs(
                        corners[corner]->color[channel] -
                        corners[0]->color[channel]) >
                    AttributeEpsilon) {
                    return false;
                }
            }
        }
        if (std::abs(corners[0]->u - corners[2]->u) >
                AttributeEpsilon ||
            std::abs(corners[1]->u - corners[3]->u) >
                AttributeEpsilon ||
            std::abs(corners[0]->v - corners[1]->v) >
                AttributeEpsilon ||
            std::abs(corners[2]->v - corners[3]->v) >
                AttributeEpsilon) {
            return false;
        }

        const auto first_x = std::max(
            0, static_cast<int>(std::ceil(minimum_x - 0.5f)));
        const auto last_x = std::min(
            static_cast<int>(FrameWidth) - 1,
            static_cast<int>(std::floor(maximum_x - 0.5f)));
        const auto first_y = std::max(
            0, static_cast<int>(std::ceil(minimum_y - 0.5f)));
        const auto last_y = std::min(
            static_cast<int>(FrameHeight) - 1,
            static_cast<int>(std::floor(maximum_y - 0.5f)));
        if (first_x > last_x || first_y > last_y) {
            return false;
        }

        static thread_local
            std::array<std::uint32_t, FrameWidth> texture_x_lookup;
        static thread_local
            std::array<std::uint32_t, FrameHeight> texture_y_lookup;
        const auto u0 = corners[0]->u;
        const auto u1 = corners[1]->u;
        const auto v0 = corners[0]->v;
        const auto v1 = corners[2]->v;
        for (auto x = first_x; x <= last_x; ++x) {
            const auto t =
                (static_cast<float>(x) + 0.5f - minimum_x) / width;
            const auto u = std::clamp(
                u0 + (u1 - u0) * t, 0.0f, 1.0f);
            texture_x_lookup[static_cast<std::size_t>(x)] =
                std::min(
                    static_cast<std::uint32_t>(
                        u * draw.texture_width),
                    draw.texture_width - 1u);
        }
        for (auto y = first_y; y <= last_y; ++y) {
            const auto t =
                (static_cast<float>(y) + 0.5f - minimum_y) / height;
            const auto v = std::clamp(
                v0 + (v1 - v0) * t, 0.0f, 1.0f);
            texture_y_lookup[static_cast<std::size_t>(y)] =
                std::min(
                    static_cast<std::uint32_t>(
                        v * draw.texture_height),
                    draw.texture_height - 1u);
        }

        std::array<std::uint32_t, 4> vertex_multiplier{};
        for (std::size_t channel = 0; channel < 4u; ++channel) {
            vertex_multiplier[channel] =
                static_cast<std::uint32_t>(
                    std::clamp(
                        corners[0]->color[channel] * 255.0f,
                        0.0f, 255.0f) +
                    0.5f);
        }
        const bool coverage_alpha_from_rgb =
            draw.texture_width == 1920u &&
            draw.texture_height == 1080u;
        rasterized_pixels +=
            static_cast<std::uint64_t>(last_x - first_x + 1) *
            static_cast<std::uint64_t>(last_y - first_y + 1);
        for (auto y = first_y; y <= last_y; ++y) {
            const auto source_row =
                static_cast<std::size_t>(
                    texture_y_lookup[static_cast<std::size_t>(y)]) *
                draw.texture_width * 4u;
            auto destination =
                (static_cast<std::size_t>(y) * FrameWidth +
                 static_cast<std::size_t>(first_x)) *
                4u;
            for (auto x = first_x; x <= last_x;
                 ++x, destination += 4u) {
                const auto source = source_row +
                    static_cast<std::size_t>(
                        texture_x_lookup[static_cast<std::size_t>(x)]) *
                        4u;
                auto texture_alpha =
                    static_cast<std::uint32_t>(texture[source + 3u]);
                if (coverage_alpha_from_rgb) {
                    texture_alpha = std::max({
                        static_cast<std::uint32_t>(texture[source]),
                        static_cast<std::uint32_t>(texture[source + 1u]),
                        static_cast<std::uint32_t>(texture[source + 2u])});
                }
                const auto source_alpha =
                    (texture_alpha * vertex_multiplier[3] + 127u) /
                    255u;
                if (source_alpha == 0u) {
                    continue;
                }
                const auto source_red =
                    (static_cast<std::uint32_t>(texture[source]) *
                         vertex_multiplier[0] +
                     127u) /
                    255u;
                const auto source_green =
                    (static_cast<std::uint32_t>(texture[source + 1u]) *
                         vertex_multiplier[1] +
                     127u) /
                    255u;
                const auto source_blue =
                    (static_cast<std::uint32_t>(texture[source + 2u]) *
                         vertex_multiplier[2] +
                     127u) /
                    255u;
                if (source_alpha == 255u) {
                    frame.rgba[destination] =
                        static_cast<std::uint8_t>(source_red);
                    frame.rgba[destination + 1u] =
                        static_cast<std::uint8_t>(source_green);
                    frame.rgba[destination + 2u] =
                        static_cast<std::uint8_t>(source_blue);
                } else {
                    const auto inverse_alpha = 255u - source_alpha;
                    frame.rgba[destination] =
                        static_cast<std::uint8_t>(
                            (source_red * source_alpha +
                             static_cast<std::uint32_t>(
                                 frame.rgba[destination]) *
                                 inverse_alpha +
                             127u) /
                            255u);
                    frame.rgba[destination + 1u] =
                        static_cast<std::uint8_t>(
                            (source_green * source_alpha +
                             static_cast<std::uint32_t>(
                                 frame.rgba[destination + 1u]) *
                                 inverse_alpha +
                             127u) /
                            255u);
                    frame.rgba[destination + 2u] =
                        static_cast<std::uint8_t>(
                            (source_blue * source_alpha +
                             static_cast<std::uint32_t>(
                                 frame.rgba[destination + 2u]) *
                                 inverse_alpha +
                             127u) /
                            255u);
                }
                frame.rgba[destination + 3u] = 255u;
                ++covered_pixels;
            }
        }
        return true;
    };
    if (!rasterize_axis_aligned_quad()) {
        std::array<std::uint64_t, FrameHeight> draw_rasterized{};
        std::array<std::uint64_t, FrameHeight> draw_covered{};
        RunAgcCpuRows(
            0, static_cast<int>(FrameHeight) - 1,
            static_cast<std::uint64_t>(FrameWidth) * FrameHeight,
            [&](const int parallel_y) {
        for (std::uint32_t first = 0;
             first + 2u < draw.draw_count; first += 3u) {
        const auto& v0 = vertices[indices[first]];
        const auto& v1 = vertices[indices[first + 1u]];
        const auto& v2 = vertices[indices[first + 2u]];
        const auto area = edge(v0, v1, v2.x, v2.y);
        if (!std::isfinite(area) || std::abs(area) < 0.0001f) {
            continue;
        }
        const auto minimum_x = std::max(
            0, static_cast<int>(std::floor(
                   std::min({v0.x, v1.x, v2.x}))));
        const auto maximum_x = std::min(
            static_cast<int>(FrameWidth) - 1,
            static_cast<int>(std::ceil(
                std::max({v0.x, v1.x, v2.x}))));
        const auto minimum_y = std::max(
            0, static_cast<int>(std::floor(
                   std::min({v0.y, v1.y, v2.y}))));
        const auto maximum_y = std::min(
            static_cast<int>(FrameHeight) - 1,
            static_cast<int>(std::ceil(
                std::max({v0.y, v1.y, v2.y}))));
        if (minimum_x > maximum_x || minimum_y > maximum_y) {
            continue;
        }
        if (parallel_y < minimum_y || parallel_y > maximum_y) {
            continue;
        }
        const auto inverse_area = 1.0f / area;
        const auto w0_step_x = (v2.y - v1.y) * inverse_area;
        const auto w0_step_y = (v1.x - v2.x) * inverse_area;
        const auto w1_step_x = (v0.y - v2.y) * inverse_area;
        const auto w1_step_y = (v2.x - v0.x) * inverse_area;
        const auto first_sample_x =
            static_cast<float>(minimum_x) + 0.5f;
        const auto first_sample_y =
            static_cast<float>(minimum_y) + 0.5f;
        auto row_w0 =
            edge(v1, v2, first_sample_x, first_sample_y) *
            inverse_area;
        auto row_w1 =
            edge(v2, v0, first_sample_x, first_sample_y) *
            inverse_area;
        constexpr float ColorEpsilon = 0.0001f;
        bool constant_vertex_color = true;
        for (std::size_t channel = 0; channel < 4u; ++channel) {
            constant_vertex_color &=
                std::abs(
                    v0.color[channel] -
                    v1.color[channel]) <= ColorEpsilon &&
                std::abs(
                    v0.color[channel] -
                    v2.color[channel]) <= ColorEpsilon;
        }
        std::array<std::uint32_t, 4> constant_multiplier{};
        if (constant_vertex_color) {
            for (std::size_t channel = 0; channel < 4u; ++channel) {
                constant_multiplier[channel] =
                    static_cast<std::uint32_t>(
                        std::clamp(
                            v0.color[channel] * 255.0f,
                            0.0f, 255.0f) +
                        0.5f);
            }
        }
        const bool coverage_alpha_from_rgb =
            draw.texture_width == 1920u &&
            draw.texture_height == 1080u;
        const auto y = parallel_y;
            const auto sample_y = static_cast<float>(y) + 0.5f;
            float row_minimum_x =
                std::numeric_limits<float>::infinity();
            float row_maximum_x =
                -std::numeric_limits<float>::infinity();
            std::uint32_t row_intersections{};
            const auto intersect_edge =
                [&](const AgcCpuVertex& begin,
                    const AgcCpuVertex& end) {
                    const auto edge_minimum_y =
                        std::min(begin.y, end.y);
                    const auto edge_maximum_y =
                        std::max(begin.y, end.y);
                    if (sample_y < edge_minimum_y - 0.0001f ||
                        sample_y > edge_maximum_y + 0.0001f) {
                        return;
                    }
                    const auto delta_y = end.y - begin.y;
                    if (std::abs(delta_y) < 0.0001f) {
                        row_minimum_x = std::min({
                            row_minimum_x, begin.x, end.x});
                        row_maximum_x = std::max({
                            row_maximum_x, begin.x, end.x});
                        row_intersections += 2u;
                        return;
                    }
                    const auto t =
                        (sample_y - begin.y) / delta_y;
                    if (t < -0.0001f || t > 1.0001f) {
                        return;
                    }
                    const auto x =
                        begin.x + (end.x - begin.x) * t;
                    row_minimum_x = std::min(row_minimum_x, x);
                    row_maximum_x = std::max(row_maximum_x, x);
                    ++row_intersections;
                };
            intersect_edge(v0, v1);
            intersect_edge(v1, v2);
            intersect_edge(v2, v0);
            if (row_intersections < 2u) {
                continue;
            }
            const auto row_first_x = std::max(
                minimum_x,
                static_cast<int>(
                    std::ceil(row_minimum_x - 0.5f)) - 1);
            const auto row_last_x = std::min(
                maximum_x,
                static_cast<int>(
                    std::floor(row_maximum_x - 0.5f)) + 1);
            if (row_first_x > row_last_x) {
                continue;
            }
            auto w0 = row_w0 +
                static_cast<float>(y - minimum_y) * w0_step_y +
                static_cast<float>(row_first_x - minimum_x) *
                    w0_step_x;
            auto w1 = row_w1 +
                static_cast<float>(y - minimum_y) * w1_step_y +
                static_cast<float>(row_first_x - minimum_x) *
                    w1_step_x;
            for (auto x = row_first_x; x <= row_last_x; ++x) {
                const auto w2 = 1.0f - w0 - w1;
                if (w0 < -0.0001f || w1 < -0.0001f ||
                    w2 < -0.0001f) {
                    w0 += w0_step_x;
                    w1 += w1_step_x;
                    continue;
                }
                ++draw_rasterized[
                    static_cast<std::size_t>(y)];
                const auto u =
                    std::clamp(
                        w0 * v0.u + w1 * v1.u + w2 * v2.u,
                        0.0f, 1.0f);
                const auto v =
                    std::clamp(
                        w0 * v0.v + w1 * v1.v + w2 * v2.v,
                        0.0f, 1.0f);
                const auto texture_x = std::min(
                    static_cast<std::uint32_t>(
                        u * draw.texture_width),
                    draw.texture_width - 1u);
                const auto texture_y = std::min(
                    static_cast<std::uint32_t>(
                        v * draw.texture_height),
                    draw.texture_height - 1u);
                const auto source =
                    (static_cast<std::size_t>(texture_y) *
                         draw.texture_width +
                     texture_x) *
                    4u;
                const auto destination =
                    (static_cast<std::size_t>(y) * FrameWidth +
                     static_cast<std::size_t>(x)) *
                    4u;
                auto texture_alpha =
                    static_cast<std::uint32_t>(
                        texture[source + 3u]);
                if (coverage_alpha_from_rgb) {
                    texture_alpha = std::max({
                        static_cast<std::uint32_t>(
                            texture[source]),
                        static_cast<std::uint32_t>(
                            texture[source + 1u]),
                        static_cast<std::uint32_t>(
                            texture[source + 2u])});
                }
                std::array<std::uint32_t, 4> vertex_multiplier =
                    constant_multiplier;
                if (!constant_vertex_color) {
                    for (std::size_t channel = 0; channel < 4u;
                         ++channel) {
                        vertex_multiplier[channel] =
                            static_cast<std::uint32_t>(
                                std::clamp(
                                    (w0 * v0.color[channel] +
                                     w1 * v1.color[channel] +
                                     w2 * v2.color[channel]) *
                                        255.0f,
                                    0.0f, 255.0f) +
                                0.5f);
                    }
                }
                const auto source_alpha =
                    (texture_alpha * vertex_multiplier[3] + 127u) /
                    255u;
                if (source_alpha == 0) {
                    w0 += w0_step_x;
                    w1 += w1_step_x;
                    continue;
                }
                const auto inverse_alpha = 255u - source_alpha;
                if (source_alpha == 255u &&
                    vertex_multiplier[0] == 255u &&
                    vertex_multiplier[1] == 255u &&
                    vertex_multiplier[2] == 255u) {
                    frame.rgba[destination] = texture[source];
                    frame.rgba[destination + 1u] = texture[source + 1u];
                    frame.rgba[destination + 2u] = texture[source + 2u];
                } else {
                    const auto modulated_source_red =
                        (static_cast<std::uint32_t>(texture[source]) *
                         vertex_multiplier[0] +
                         127u) /
                        255u;
                    const auto modulated_source_green =
                        (static_cast<std::uint32_t>(
                             texture[source + 1u]) *
                         vertex_multiplier[1] +
                         127u) /
                        255u;
                    const auto modulated_source_blue =
                        (static_cast<std::uint32_t>(texture[source + 2u]) *
                         vertex_multiplier[2] +
                         127u) /
                        255u;
                    if (source_alpha == 255u) {
                        frame.rgba[destination] =
                            static_cast<std::uint8_t>(
                                modulated_source_red);
                        frame.rgba[destination + 1u] =
                            static_cast<std::uint8_t>(
                                modulated_source_green);
                        frame.rgba[destination + 2u] =
                            static_cast<std::uint8_t>(
                                modulated_source_blue);
                    } else {
                        frame.rgba[destination] =
                            static_cast<std::uint8_t>(
                                (modulated_source_red *
                                     source_alpha +
                                 static_cast<std::uint32_t>(
                                     frame.rgba[destination]) *
                                     inverse_alpha +
                                 127u) /
                                255u);
                        frame.rgba[destination + 1u] =
                            static_cast<std::uint8_t>(
                                (modulated_source_green *
                                     source_alpha +
                                 static_cast<std::uint32_t>(
                                     frame.rgba[destination +
                                                1u]) *
                                     inverse_alpha +
                                 127u) /
                                255u);
                        frame.rgba[destination + 2u] =
                            static_cast<std::uint8_t>(
                                (modulated_source_blue *
                                     source_alpha +
                                 static_cast<std::uint32_t>(
                                     frame.rgba[destination +
                                                2u]) *
                                     inverse_alpha +
                                 127u) /
                                255u);
                    }
                }
                frame.rgba[destination + 3u] = 255u;
                ++draw_covered[static_cast<std::size_t>(y)];
                w0 += w0_step_x;
                w1 += w1_step_x;
            }
        }
        });
        for (std::size_t y = 0; y < FrameHeight; ++y) {
            rasterized_pixels += draw_rasterized[y];
            covered_pixels += draw_covered[y];
        }
    }
    if (covered_pixels == 0) {
        if (rasterized_pixels != 0u ||
            positions_are_render_target_coordinates) {
            if (draw_ordinal < 64u) {
                frame.rendered_draw_mask |=
                    UINT64_C(1) << draw_ordinal;
                frame.draw_coverage[draw_ordinal] = 0u;
            }
#ifdef __ANDROID__
            static std::atomic<std::uint32_t> transparent_logs{};
            if (transparent_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 4u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu layer transparent ordinal=%zu "
                    "rasterized=%llu transformed=%u "
                    "texture=0x%llx %ux%u",
                    draw_ordinal,
                    static_cast<unsigned long long>(
                        rasterized_pixels),
                    positions_are_render_target_coordinates
                        ? 1u
                        : 0u,
                    static_cast<unsigned long long>(
                        draw.texture_address),
                    draw.texture_width, draw.texture_height);
            }
#endif
            return true;
        }
#ifdef __ANDROID__
        {
            static std::atomic<std::uint32_t>
                coverage_geometry_logs{};
            const auto geometry_log_index =
                coverage_geometry_logs.fetch_add(
                    1u, std::memory_order_relaxed);
            if (geometry_log_index < 2u) {
                float raw_minimum_x =
                    std::numeric_limits<float>::infinity();
                float raw_minimum_y =
                    std::numeric_limits<float>::infinity();
                float raw_maximum_x =
                    -std::numeric_limits<float>::infinity();
                float raw_maximum_y =
                    -std::numeric_limits<float>::infinity();
                float scaled_minimum_x = raw_minimum_x;
                float scaled_minimum_y = raw_minimum_y;
                float scaled_maximum_x = raw_maximum_x;
                float scaled_maximum_y = raw_maximum_y;
                for (const auto index : indices) {
                    raw_minimum_x = std::min(
                        raw_minimum_x, raw_positions[index][0]);
                    raw_minimum_y = std::min(
                        raw_minimum_y, raw_positions[index][1]);
                    raw_maximum_x = std::max(
                        raw_maximum_x, raw_positions[index][0]);
                    raw_maximum_y = std::max(
                        raw_maximum_y, raw_positions[index][1]);
                    scaled_minimum_x = std::min(
                        scaled_minimum_x, vertices[index].x);
                    scaled_minimum_y = std::min(
                        scaled_minimum_y, vertices[index].y);
                    scaled_maximum_x = std::max(
                        scaled_maximum_x, vertices[index].x);
                    scaled_maximum_y = std::max(
                        scaled_maximum_y, vertices[index].y);
                }
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu coverage geometry ordinal=%zu "
                    "vertex=0x%llx stride=%u max_index=%u "
                    "raw=[%.3f,%.3f]-[%.3f,%.3f] "
                    "scaled=[%.3f,%.3f]-[%.3f,%.3f] "
                    "rt=%ux%u es=0x%08x ps=0x%08x",
                    draw_ordinal,
                    static_cast<unsigned long long>(
                        vertex_address),
                    vertex_stride, maximum_index,
                    raw_minimum_x, raw_minimum_y,
                    raw_maximum_x, raw_maximum_y,
                    scaled_minimum_x, scaled_minimum_y,
                    scaled_maximum_x, scaled_maximum_y,
                    draw.render_target_width,
                    draw.render_target_height,
                    draw.sh_es_program_lo,
                    draw.sh_ps_program_lo);
                const auto i0 =
                    indices.size() > 0u ? indices[0] : 0u;
                const auto i1 =
                    indices.size() > 1u ? indices[1] : i0;
                const auto i2 =
                    indices.size() > 2u ? indices[2] : i1;
                const auto i3 =
                    indices.size() > 3u ? indices[3] : i2;
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu coverage vertices indices=%u,%u,%u,%u "
                    "p0=%.3f,%.3f p1=%.3f,%.3f "
                    "p2=%.3f,%.3f p3=%.3f,%.3f",
                    i0, i1, i2, i3,
                    raw_positions[i0][0],
                    raw_positions[i0][1],
                    raw_positions[i1][0],
                    raw_positions[i1][1],
                    raw_positions[i2][0],
                    raw_positions[i2][1],
                    raw_positions[i3][0],
                    raw_positions[i3][1]);
            }
        }
        log_cpu_failure("coverage");
#endif
        return false;
    }
    ++frame.rendered_draws;
    frame.covered_pixels += covered_pixels;
    if (draw_ordinal < 64u) {
        frame.rendered_draw_mask |=
            UINT64_C(1) << draw_ordinal;
        frame.draw_coverage[draw_ordinal] = covered_pixels;
    }
#ifdef __ANDROID__
    static std::atomic<std::uint32_t> cpu_draw_logs{};
    const auto cpu_draw_log_index = cpu_draw_logs.fetch_add(
        1u, std::memory_order_relaxed);
    if (cpu_draw_log_index < 4u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "cpu layer ordinal=%zu rendered=%u texture=0x%llx "
            "%ux%u vertex=0x%llx stride=%u vertices=%u "
            "indices=%u covered=%llu rt=%ux%u "
            "p0=%.1f,%.1f p1=%.1f,%.1f p2=%.1f,%.1f "
            "uv0=%.4f,%.4f",
            draw_ordinal, frame.rendered_draws,
            static_cast<unsigned long long>(
                draw.texture_address),
            draw.texture_width, draw.texture_height,
            static_cast<unsigned long long>(vertex_address),
            vertex_stride, maximum_index + 1u, draw.draw_count,
            static_cast<unsigned long long>(covered_pixels),
            draw.render_target_width,
            draw.render_target_height,
            vertices[0].x, vertices[0].y,
            vertices.size() > 1u ? vertices[1].x : 0.0f,
            vertices.size() > 1u ? vertices[1].y : 0.0f,
            vertices.size() > 2u ? vertices[2].x : 0.0f,
            vertices.size() > 2u ? vertices[2].y : 0.0f,
            vertices[0].u, vertices[0].v);
    }
#endif
    return true;
}

std::size_t AgcRgbaStorageBytes(
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t tile_mode) {
    if (width == 0 || height == 0 ||
        width > 8192u || height > 8192u) {
        return 0;
    }
    std::uint64_t bytes{};
    if (tile_mode == 0u) {
        bytes =
            static_cast<std::uint64_t>(width) * height * 4u;
    } else if (tile_mode == 27u) {
        const auto blocks_wide =
            (static_cast<std::uint64_t>(width) + 127u) / 128u;
        const auto blocks_high =
            (static_cast<std::uint64_t>(height) + 127u) / 128u;
        bytes = blocks_wide * blocks_high * 65536u;
    } else {
        return 0;
    }
    return bytes <= std::numeric_limits<std::size_t>::max()
        ? static_cast<std::size_t>(bytes)
        : 0;
}

bool TryApplyAgcCpuClearOrCopy(
    const AgcDiagnosticDraw& draw) {
    if (draw.render_target_address == 0 ||
        draw.render_target_width == 0 ||
        draw.render_target_height == 0 ||
        draw.render_target_format != 9u ||
        draw.render_target_number_type != 0u) {
        return false;
    }
    const auto target_bytes = AgcRgbaStorageBytes(
        draw.render_target_width,
        draw.render_target_height,
        draw.render_target_tile_mode);
    if (target_bytes == 0) {
        return false;
    }

    if (draw.draw_count == 3u &&
        draw.texture_address != 0 &&
        draw.texture_width == 5u &&
        draw.texture_height == 1u &&
        draw.texture_tile_mode == 0u) {
        std::array<float, 4> color{};
        const auto scalar_buffer_address =
            draw.texture_address >> 8u;
        if (!TryReadGuestBytes(
                scalar_buffer_address, color.data(),
                sizeof(color)) ||
            !std::ranges::all_of(
                color, [](const float value) {
                    return std::isfinite(value) &&
                           value >= 0.0f && value <= 1.0f;
                })) {
            return false;
        }
        std::array<std::uint8_t, 4> rgba{};
        for (std::size_t channel = 0;
             channel < rgba.size(); ++channel) {
            rgba[channel] = static_cast<std::uint8_t>(
                std::clamp(
                    color[channel] * 255.0f + 0.5f,
                    0.0f, 255.0f));
        }
        std::uint32_t packed{};
        std::memcpy(&packed, rgba.data(), sizeof(packed));
        bool capture_software_frame{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            if (!HasAccessLocked(
                    draw.render_target_address,
                    target_bytes,
                    LSX4_PS5_GUEST_WRITE)) {
                return false;
            }
            const auto now = std::chrono::steady_clock::now();
            capture_software_frame =
                g_runtime.agc_cpu_next_render
                        .time_since_epoch().count() == 0 ||
                now >= g_runtime.agc_cpu_next_render;
            const bool clear_guest_target =
                !UseVulkanGuestRaster() &&
                !(capture_software_frame &&
                  target_bytes >= (1920u * 1080u * 4u));
            if (clear_guest_target) {
                std::fill_n(
                    reinterpret_cast<std::uint32_t*>(
                        draw.render_target_address),
                    target_bytes / sizeof(std::uint32_t),
                    packed);
            }
            g_runtime.agc_cpu_capture_active =
                capture_software_frame;
            g_runtime.agc_cpu_capture_target =
                capture_software_frame
                ? draw.render_target_address
                : 0;
            if (capture_software_frame) {
                g_runtime.agc_cpu_next_render =
                    now + std::chrono::milliseconds(
                              AgcCpuFrameCaptureIntervalMs);
            }
        }
        if (capture_software_frame) {
            constexpr std::uint32_t CpuFrameWidth =
                AgcCpuFrameWidth;
            constexpr std::uint32_t CpuFrameHeight =
                AgcCpuFrameHeight;
            auto software_clear =
                std::make_shared<AgcCpuFrame>();
            software_clear->width = CpuFrameWidth;
            software_clear->height = CpuFrameHeight;
            software_clear->rgba.resize(
                static_cast<std::size_t>(CpuFrameWidth) *
                CpuFrameHeight * rgba.size());
            std::fill_n(
                reinterpret_cast<std::uint32_t*>(
                    software_clear->rgba.data()),
                static_cast<std::size_t>(CpuFrameWidth) *
                    CpuFrameHeight,
                packed);
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.agc_cpu_working_surfaces[
                draw.render_target_address] =
                std::move(software_clear);
        }
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> clear_logs{};
        const auto log_index =
            clear_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 4u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu clear target=0x%llx bytes=%zu "
                "buffer=0x%llx rgba=%u,%u,%u,%u "
                "floats=%.4f,%.4f,%.4f,%.4f",
                static_cast<unsigned long long>(
                    draw.render_target_address),
                target_bytes,
                static_cast<unsigned long long>(
                    scalar_buffer_address),
                rgba[0], rgba[1], rgba[2], rgba[3],
                color[0], color[1], color[2], color[3]);
        }
#endif
        return true;
    }

    if (draw.draw_count == 4u &&
        draw.texture_address != 0 &&
        draw.texture_width == draw.render_target_width &&
        draw.texture_height == draw.render_target_height &&
        draw.texture_tile_mode ==
            draw.render_target_tile_mode) {
        const auto source_bytes = AgcRgbaStorageBytes(
            draw.texture_width,
            draw.texture_height,
            draw.texture_tile_mode);
        if (source_bytes != target_bytes) {
            return false;
        }
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto software_source =
                g_runtime.agc_cpu_surfaces.find(
                    draw.texture_address);
            const bool has_software_source =
                software_source != g_runtime.agc_cpu_surfaces.end() &&
                software_source->second != nullptr &&
                !software_source->second->rgba.empty();
            if (!has_software_source &&
                !UseVulkanGuestRaster()) {
                if (!HasAccessLocked(
                        draw.texture_address, source_bytes,
                        LSX4_PS5_GUEST_READ) ||
                    !HasAccessLocked(
                        draw.render_target_address, target_bytes,
                        LSX4_PS5_GUEST_WRITE)) {
                    return false;
                }
            }
            if (!has_software_source &&
                !UseVulkanGuestRaster()) {
                std::memmove(
                    reinterpret_cast<void*>(
                        draw.render_target_address),
                    reinterpret_cast<const void*>(
                        draw.texture_address),
                    target_bytes);
            }
            if (has_software_source) {
                g_runtime.agc_cpu_surfaces[
                    draw.render_target_address] =
                    software_source->second;
            }
        }
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> copy_logs{};
        const auto log_index =
            copy_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_index < 4u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu final blit source=0x%llx target=0x%llx "
                "bytes=%zu",
                static_cast<unsigned long long>(
                    draw.texture_address),
                static_cast<unsigned long long>(
                    draw.render_target_address),
                target_bytes);
        }
#endif
        return true;
    }
    return false;
}

#ifdef __ANDROID__
std::uint32_t PresenterDecimationStep(const std::uint32_t width,
                                      const std::uint32_t height) {
    static const std::uint32_t cap_height = [] {
        const char* const requested = std::getenv("PS5_PRESENT_MAX_HEIGHT");
        if (requested == nullptr) {
            return 1080u;
        }
        char* end = nullptr;
        const auto parsed = std::strtoul(requested, &end, 10);
        if (end == requested || parsed < 240u || parsed > 4320u) {
            return 1080u;
        }
        return static_cast<std::uint32_t>(parsed);
    }();
    if (height == 0 || width == 0) {
        return 1u;
    }
    std::uint32_t step = 1u;
    while (step < 8u && height / step > cap_height) {
        ++step;
    }
    while (step > 1u && (width / step < 64u || height / step < 64u)) {
        --step;
    }
    return step;
}

template <typename RowWork>
void RunPresenterRows(const std::uint32_t rows, RowWork&& work) {
    const auto hardware = std::thread::hardware_concurrency();
    const auto workers = std::min<std::uint32_t>(
        rows, std::max<std::uint32_t>(hardware == 0 ? 1u : hardware / 2u, 1u));
    if (workers <= 1u) {
        work(0u, rows);
        return;
    }
    const auto span = (rows + workers - 1u) / workers;
    std::vector<std::thread> helpers;
    helpers.reserve(workers - 1u);
    for (std::uint32_t worker = 1u; worker < workers; ++worker) {
        const auto first = worker * span;
        if (first >= rows) {
            break;
        }
        helpers.emplace_back(
            [&work, first, last = std::min(first + span, rows)] {
                work(first, last);
            });
    }
    work(0u, std::min(span, rows));
    for (auto& helper : helpers) {
        helper.join();
    }
}

bool TryReadGuestPresenterRgba(
    const GuestPresenterTexture& texture,
    std::vector<std::uint8_t>& rgba,
    std::uint32_t& out_width,
    std::uint32_t& out_height) {
    static thread_local std::vector<std::uint32_t> cached_tiled_pixels;
    static thread_local std::vector<std::uint32_t> cached_x_terms;
    static thread_local std::vector<std::uint32_t> cached_y_terms;
    static thread_local std::uint32_t cached_mode27_width{};
    static thread_local std::uint32_t cached_mode27_height{};
    out_width = texture.width;
    out_height = texture.height;
    const auto pixel_count =
        static_cast<std::uint64_t>(texture.width) *
        texture.height;
    if (pixel_count >
        std::numeric_limits<std::size_t>::max() / 4u) {
        return false;
    }
    rgba.resize(static_cast<std::size_t>(pixel_count) * 4u);
    if (texture.tile_mode == 0) {
        return TryReadGuestBytes(
            texture.address, rgba.data(), rgba.size());
    }
    if (texture.tile_mode != 27) {
        return false;
    }

    constexpr std::uint32_t XMask[16]{
        0u, 0u, 1u << 0u, 1u << 1u,
        0u, 0u, 1u << 2u, 0u,
        1u << 7u, 1u << 4u, 1u << 6u, 1u << 5u,
        1u << 3u, 0u, 1u << 6u, 1u << 7u};
    constexpr std::uint32_t YMask[16]{
        0u, 0u, 0u, 0u,
        1u << 0u, 1u << 1u, 0u, 1u << 2u,
        (1u << 4u) | (1u << 7u), 1u << 4u,
        1u << 5u, 1u << 6u,
        0u, 1u << 3u, 1u << 7u, 1u << 6u};
    constexpr std::uint32_t BlockWidth = 128;
    constexpr std::uint32_t BlockHeight = 128;
    constexpr std::uint64_t BlockBytes = 65536;
    const auto blocks_wide =
        (static_cast<std::uint64_t>(texture.width) +
         BlockWidth - 1u) /
        BlockWidth;
    const auto blocks_high =
        (static_cast<std::uint64_t>(texture.height) +
         BlockHeight - 1u) /
        BlockHeight;
    if (blocks_wide == 0 || blocks_high == 0 ||
        blocks_wide >
            std::numeric_limits<std::size_t>::max() /
                blocks_high ||
        blocks_wide * blocks_high >
            std::numeric_limits<std::size_t>::max() /
                BlockBytes) {
        return false;
    }
    const auto cached_tiled_pixel_count =
        blocks_wide * blocks_high *
        (BlockBytes / sizeof(std::uint32_t));
    cached_tiled_pixels.resize(
        static_cast<std::size_t>(cached_tiled_pixel_count));
    if (!TryReadGuestBytes(
            texture.address,
            reinterpret_cast<std::uint8_t*>(
                cached_tiled_pixels.data()),
            cached_tiled_pixels.size() * sizeof(std::uint32_t))) {
        return false;
    }
    const auto axis_term =
        [](const std::uint32_t coordinate,
           const std::uint32_t* const masks) {
            std::uint32_t offset{};
            for (std::uint32_t bit = 0; bit < 16u; ++bit) {
                const auto parity = static_cast<std::uint32_t>(
                    __builtin_popcount(coordinate & masks[bit])) &
                    1u;
                offset |= parity << bit;
            }
            return offset;
        };
    if (cached_x_terms.size() < texture.width) {
        cached_x_terms.resize(texture.width);
    }
    if (cached_y_terms.size() < texture.height) {
        cached_y_terms.resize(texture.height);
    }
    auto x_terms = std::span<std::uint32_t>(
        cached_x_terms.data(), texture.width);
    auto y_terms = std::span<std::uint32_t>(
        cached_y_terms.data(), texture.height);
    if (cached_mode27_width != texture.width ||
        cached_mode27_height != texture.height ||
        cached_x_terms.size() < texture.width) {
        for (std::uint32_t x = 0; x < texture.width; ++x) {
            x_terms[x] = axis_term(x, XMask);
        }
        for (std::uint32_t y = 0; y < texture.height; ++y) {
            y_terms[y] = axis_term(y, YMask);
        }
        cached_mode27_width = texture.width;
        cached_mode27_height = texture.height;
    }
    const auto destination_pixels =
        reinterpret_cast<std::uint32_t*>(rgba.data());
    const auto source_pixels =
        cached_tiled_pixels.data();
    const auto source_pixel_count = cached_tiled_pixels.size();
    const auto block_pixel_stride = BlockBytes / 4u;
    for (std::uint32_t y = 0; y < texture.height; ++y) {
        const auto block_y = y / BlockHeight;
        const auto y_term = y_terms[y];
        const auto row_destination =
            static_cast<std::size_t>(y) * texture.width;
        const auto block_y_base =
            block_y * blocks_wide * block_pixel_stride;
        for (std::uint32_t block_x = 0; block_x < blocks_wide;
             ++block_x) {
            const auto block_left = block_x * BlockWidth;
            const auto block_right = std::min<std::uint32_t>(
                block_left + BlockWidth, texture.width);
            const auto source_block_base =
                block_y_base + block_x * block_pixel_stride;
            for (std::uint32_t x = block_left; x < block_right; ++x) {
                const auto source =
                    source_block_base +
                    (x_terms[x] ^ y_term);
                if (source >= source_pixel_count) {
                    continue;
                }
                destination_pixels[row_destination + x] =
                    source_pixels[static_cast<std::size_t>(source)];
            }
        }
    }
    return true;
}

void PresentGuestTextureToAndroid(
    ANativeWindow* const window,
    const GuestPresenterTexture& texture,
    const std::shared_ptr<AgcCpuFrame>& cpu_frame,
    const std::uint64_t flip_count) {
    static const std::uint64_t presentation_divisor = []() -> std::uint64_t {
        const char* divisor =
            std::getenv("PS5_ANDROID_PRESENT_DIVISOR");
        if (divisor == nullptr) {
            return 1ull;
        }
        char* end = nullptr;
        const auto parsed = std::strtoull(divisor, &end, 10);
        if (end == divisor || parsed == 0ull) {
            return 1ull;
        }
        return parsed;
    }();
    if (flip_count > 8u && cpu_frame == nullptr &&
        presentation_divisor > 1u &&
        flip_count % presentation_divisor != 0u) {
        return;
    }
    const auto has_gpu_frame =
        cpu_frame != nullptr &&
        cpu_frame->gpu_frame != nullptr &&
        cpu_frame->gpu_frame->width != 0 &&
        cpu_frame->gpu_frame->height != 0 &&
        !cpu_frame->gpu_frame->draws.empty();
    const auto has_cpu_frame =
        cpu_frame != nullptr &&
        cpu_frame->width != 0 && cpu_frame->height != 0 &&
        cpu_frame->rgba.size() ==
            static_cast<std::size_t>(cpu_frame->width) *
                cpu_frame->height * 4u;
    if (window == nullptr ||
        (!has_gpu_frame && !has_cpu_frame &&
         (texture.address == 0 ||
          texture.width == 0 || texture.height == 0 ||
          (texture.tile_mode != 0 && texture.tile_mode != 27) ||
          texture.width > 8192 || texture.height > 8192))) {
        return;
    }
    auto source_width =
        (has_gpu_frame || has_cpu_frame)
        ? cpu_frame->width : texture.width;
    auto source_height =
        (has_gpu_frame || has_cpu_frame)
        ? cpu_frame->height : texture.height;
    if (has_gpu_frame) {
        bool splash_active{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            splash_active = g_runtime.game_splash_active;
        }
        const auto draw_has_visible_content =
            [](const Lsx4::Ps5Desktop::VulkanGuestDraw& draw) {
                if (draw.texture_rgba != nullptr) {
                    const auto& rgba = *draw.texture_rgba;
                    for (std::size_t pixel = 0;
                         pixel + 2u < rgba.size(); pixel += 4u) {
                        if (rgba[pixel] != 0u ||
                            rgba[pixel + 1u] != 0u ||
                            rgba[pixel + 2u] != 0u) {
                            return true;
                        }
                    }
                    return false;
                }
                return std::ranges::any_of(
                    draw.vertices,
                    [](const auto& vertex) {
                        return (vertex.color &
                                UINT32_C(0x00ffffff)) != 0u;
                    });
            };
        bool visible =
            (cpu_frame->gpu_frame->clear_rgba &
             UINT32_C(0x00ffffff)) != 0u ||
            std::ranges::any_of(
                cpu_frame->gpu_frame->draws,
                draw_has_visible_content);
        for (const auto& pass : cpu_frame->gpu_frame->passes) {
            visible = visible ||
                ((pass.clear_rgba & UINT32_C(0x00ffffff)) != 0u) ||
                std::ranges::any_of(
                    pass.draws, draw_has_visible_content);
        }
        if (!splash_active || visible) {
            if (Lsx4::Ps5Desktop::PresentVulkanGuestFrame(
                    window, *cpu_frame->gpu_frame, flip_count)) {
                const std::lock_guard lock{g_runtime.mutex};
                g_runtime.game_splash_active = false;
                return;
            }
        } else {
            return;
        }
    }
    const std::uint8_t* rgba{};
    std::size_t rgba_size{};
    static thread_local std::vector<std::uint8_t> owned_rgba;
    if (has_cpu_frame) {
        rgba = cpu_frame->rgba.data();
        rgba_size = cpu_frame->rgba.size();
    } else {
        if (!TryReadGuestPresenterRgba(
                texture, owned_rgba, source_width, source_height)) {
            return;
        }
        rgba = owned_rgba.data();
        rgba_size = owned_rgba.size();
    }
    bool splash_active{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        splash_active = g_runtime.game_splash_active;
    }
    if (splash_active) {
        const bool guest_frame_is_black = std::ranges::all_of(
            std::span<const std::uint8_t>{rgba, rgba_size},
            [](const std::uint8_t value) { return value == 0; });
        if (guest_frame_is_black) {
            return;
        }
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.game_splash_active = false;
    }
    if (Lsx4::Ps5Desktop::PresentVulkanFrame(
            window, rgba, rgba_size,
            source_width, source_height, flip_count)) {
        return;
    }
    const bool log_fallback =
        flip_count <= 8u || flip_count % 60u == 0u;
    std::uint64_t fingerprint{};
    std::uint64_t sampled_nonzero{};
    if (log_fallback) {
        fingerprint = UINT64_C(1469598103934665603);
        const auto sample_stride =
            std::max<std::size_t>(rgba_size / 4096u, 1u);
        for (std::size_t index = 0; index < rgba_size;
             index += sample_stride) {
            const auto value = rgba[index];
            fingerprint ^= value;
            fingerprint *= UINT64_C(1099511628211);
            sampled_nonzero += value != 0;
        }
    }
    if (ANativeWindow_getWidth(window) !=
            static_cast<std::int32_t>(source_width) ||
        ANativeWindow_getHeight(window) !=
            static_cast<std::int32_t>(source_height) ||
        ANativeWindow_getFormat(window) != WINDOW_FORMAT_RGBA_8888) {
        (void)ANativeWindow_setBuffersGeometry(
            window, static_cast<std::int32_t>(source_width),
            static_cast<std::int32_t>(source_height),
            WINDOW_FORMAT_RGBA_8888);
    }
    ANativeWindow_Buffer buffer{};
    if (ANativeWindow_lock(window, &buffer, nullptr) != 0) {
        return;
    }
    const auto copy_width = std::min<std::uint32_t>(
        source_width, static_cast<std::uint32_t>(buffer.width));
    const auto copy_height = std::min<std::uint32_t>(
        source_height, static_cast<std::uint32_t>(buffer.height));
    const auto stride_pixels = static_cast<std::size_t>(buffer.stride);
    const auto stride_bytes = stride_pixels * 4u;
    const auto source_width_bytes =
        static_cast<std::size_t>(source_width) * 4u;
    const auto copy_width_bytes =
        static_cast<std::size_t>(copy_width) * 4u;
    const auto copy_row_bytes =
        static_cast<std::size_t>(copy_width) * 4u;
    const auto copy_height_rows =
        static_cast<std::size_t>(copy_height);
    auto* const destination =
        static_cast<std::uint8_t*>(buffer.bits);
    if (copy_width == static_cast<std::uint32_t>(buffer.width) &&
        copy_height == static_cast<std::uint32_t>(buffer.height)) {
        std::memcpy(
            destination, rgba,
            copy_width_bytes * copy_height_rows);
    } else {
        for (std::size_t row = 0; row < copy_height_rows; ++row) {
            const auto source_row =
                destination + row * stride_bytes;
            const auto source_offset =
                row * source_width_bytes;
            std::memcpy(
                source_row, rgba + source_offset,
                copy_row_bytes);
            if (copy_width < static_cast<std::uint32_t>(
                                  buffer.width)) {
                const auto right_margin = stride_bytes - copy_row_bytes;
                if (right_margin != 0u) {
                    std::memset(
                        source_row + copy_row_bytes, 0,
                        right_margin);
                }
            }
        }
        if (copy_height < static_cast<std::uint32_t>(buffer.height)) {
            const auto remaining_rows =
                static_cast<std::size_t>(buffer.height) -
                copy_height_rows;
            std::fill_n(
                reinterpret_cast<std::uint32_t*>(
                    destination +
                    copy_height_rows * stride_bytes),
                remaining_rows * stride_pixels,
                0u);
        }
    }
    (void)ANativeWindow_unlockAndPost(window);
    if (log_fallback) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "present count=%llu address=0x%llx source=%ux%u "
            "format=0x%x tile=%u cpu=%u/%u sample_nonzero=%llu "
            "hash=0x%llx",
            static_cast<unsigned long long>(flip_count),
            static_cast<unsigned long long>(texture.address),
            source_width, source_height, texture.unified_format,
            texture.tile_mode,
            has_cpu_frame ? cpu_frame->rendered_draws : 0u,
            has_cpu_frame ? 1u : 0u,
            static_cast<unsigned long long>(sampled_nonzero),
            static_cast<unsigned long long>(fingerprint));
        std::fprintf(
            stderr,
            "PS5_ANDROID_PRESENT count=%llu address=0x%llx "
            "source=%ux%u surface=%dx%d stride=%d\n",
            static_cast<unsigned long long>(flip_count),
            static_cast<unsigned long long>(texture.address),
            source_width, source_height,
            buffer.width, buffer.height, buffer.stride);
    }
}

void TryPresentGameSplashToAndroid() {
    std::filesystem::path splash_path;
    ANativeWindow* window{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (g_runtime.game_splash_presented ||
            g_runtime.app0_directory.empty() ||
            g_runtime.android_window == nullptr) {
            return;
        }
        splash_path =
            g_runtime.app0_directory / "sce_sys" / "pic0.png";
        window = g_runtime.android_window;
        ANativeWindow_acquire(window);
    }

    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    bool presented{};
    if (png_image_begin_read_from_file(
            &image, splash_path.string().c_str()) != 0 &&
        image.width != 0 && image.height != 0 &&
        image.width <= 8192u && image.height <= 8192u) {
        image.format = PNG_FORMAT_RGBA;
        std::vector<std::uint8_t> rgba(PNG_IMAGE_SIZE(image));
        if (png_image_finish_read(
                &image, nullptr, rgba.data(), 0, nullptr) != 0) {
            presented = Lsx4::Ps5Desktop::PresentVulkanFrame(
                window, rgba.data(), rgba.size(),
                image.width, image.height, 0);
        }
    }
    png_image_free(&image);
    ANativeWindow_release(window);

    if (presented) {
        {
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.game_splash_active = true;
            g_runtime.game_splash_presented = true;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "presented game splash path=%s size=%ux%u",
            splash_path.string().c_str(), image.width, image.height);
    }
}
#endif

bool EnqueueGuestVideoOutEventsLocked(
    const std::vector<GuestVideoOutPort::FlipEventRegistration>&
        registrations,
    const std::uint64_t ident,
    const std::uint64_t data_hint) {
    bool notify{};
    constexpr std::int16_t Filter = -13;
    constexpr std::uint16_t Flags = 0x20;
    const auto time_bits =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count()) &
        0xfffu;
    const auto event_data =
        time_bits | UINT64_C(0x1000) |
        (data_hint & UINT64_C(0xffffffffffff0000));
    for (const auto& registration : registrations) {
        const auto queue = g_runtime.guest_event_queues.find(
            registration.event_queue);
        if (queue == g_runtime.guest_event_queues.end()) {
            continue;
        }
        std::array<std::uint8_t, 0x20> event{};
        std::memcpy(event.data() + 0x00, &ident, sizeof(ident));
        std::memcpy(event.data() + 0x08, &Filter, sizeof(Filter));
        std::memcpy(event.data() + 0x0a, &Flags, sizeof(Flags));
        std::memcpy(
            event.data() + 0x10, &event_data, sizeof(event_data));
        std::memcpy(
            event.data() + 0x18, &registration.user_data,
            sizeof(registration.user_data));
        if (queue->second.empty()) {
            queue->second.push_back(event);
        } else {
            queue->second.back() = event;
        }
        notify = true;
    }
    return notify;
}

void StartGuestVblankThreadLocked() {
    if (g_runtime.guest_vblank_thread_started) {
        return;
    }
    g_runtime.guest_vblank_thread_started = true;
    g_runtime.guest_host_threads.emplace_back([] {
        auto next = std::chrono::steady_clock::now();
        constexpr auto Period = std::chrono::microseconds(16667);
        while (!g_runtime.shutdown_requested.load(
            std::memory_order_acquire)) {
            next += Period;
            std::this_thread::sleep_until(next);
            bool notify{};
            {
                const std::lock_guard lock{g_runtime.mutex};
                for (auto& [handle, port] :
                     g_runtime.guest_video_out_ports) {
                    (void)handle;
                    if (port.vblank_events.empty()) {
                        continue;
                    }
                    const auto count = ++port.vblank_count;
                    const auto data_hint =
                        (count &
                         UINT64_C(0x0000ffffffffffff))
                        << 16u;
                    notify |= EnqueueGuestVideoOutEventsLocked(
                        port.vblank_events, UINT64_C(0x40),
                        data_hint);
                }
            }
            if (notify) {
                g_runtime.guest_event_queue_condition.notify_all();
            }
            const auto now = std::chrono::steady_clock::now();
            if (next + std::chrono::milliseconds(100) < now) {
                next = now;
            }
        }
    });
}

void ObserveAgcFlip(const std::int32_t handle,
                    const std::int32_t buffer_index,
                    const std::uint32_t flip_mode,
                    const std::uint64_t flip_argument) {
    GuestVideoOutPort port{};
    std::uint64_t flip_count{};
    bool should_dump{};
    bool notify_event_queue{};
    GuestPresenterTexture presenter{};
    std::shared_ptr<AgcCpuFrame> cpu_frame;
    bool used_capture_surface{};
    bool queued_capture_surface{};
    std::uint64_t resolved_capture_target{};
    std::uint32_t resolved_capture_draws{};
    std::uint64_t resolved_capture_coverage{};
    std::uint64_t resolved_capture_eligible{};
    std::uint64_t resolved_capture_rendered{};
#ifdef __ANDROID__
    ANativeWindow* android_window{};
#endif
    std::chrono::steady_clock::time_point flip_deadline{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = g_runtime.guest_video_out_ports.find(handle);
        if (found == g_runtime.guest_video_out_ports.end() ||
            buffer_index < -2 || buffer_index >= 16 ||
            (buffer_index >= 0 &&
             static_cast<std::size_t>(buffer_index) >=
                 found->second.buffer_addresses.size())) {
            const auto invalid_count =
                ++g_runtime.agc_invalid_flip_count;
            std::fprintf(
                stderr,
                "PS5_AGC_FLIP_INVALID count=%llu handle=%d index=%d "
                "mode=%u arg=0x%llx\n",
                static_cast<unsigned long long>(invalid_count),
                handle, buffer_index, flip_mode,
                static_cast<unsigned long long>(flip_argument));
#ifdef __ANDROID__
            if (invalid_count <= 8u ||
                invalid_count % 60u == 0) {
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4-PS5",
                    "invalid flip count=%llu handle=%d index=%d "
                    "mode=%u arg=0x%llx",
                    static_cast<unsigned long long>(invalid_count),
                    handle, buffer_index, flip_mode,
                    static_cast<unsigned long long>(flip_argument));
            }
#endif
            return;
        }
        found->second.current_buffer = buffer_index;
        ++found->second.flip_count;
        found->second.last_flip_argument = flip_argument;
        const auto flip_now = std::chrono::steady_clock::now();
        found->second.last_flip_process_time =
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    flip_now - found->second.opened_at).count());
        found->second.last_flip_process_counter =
            static_cast<std::uint64_t>(
                flip_now.time_since_epoch().count());
        found->second.last_flip_submit_counter =
            found->second.last_flip_process_counter;
        port = found->second;
        presenter = g_runtime.presenter_texture;
        cpu_frame.reset();
        if (buffer_index >= 0) {
            const auto flip_address =
                port.buffer_addresses[
                    static_cast<std::size_t>(buffer_index)];
            const auto software_surface =
                g_runtime.agc_cpu_surfaces.find(flip_address);
            if (software_surface !=
                g_runtime.agc_cpu_surfaces.end()) {
                cpu_frame = software_surface->second;
            }
            if (g_runtime.agc_cpu_pending_capture_target != 0) {
                resolved_capture_target =
                    g_runtime.agc_cpu_pending_capture_target;
                auto captured =
                    g_runtime.agc_cpu_working_surfaces.find(
                        g_runtime.agc_cpu_pending_capture_target);
                const auto working_has_gpu =
                    captured !=
                        g_runtime.agc_cpu_working_surfaces.end() &&
                    captured->second != nullptr &&
                    captured->second->gpu_frame != nullptr &&
                    !captured->second->gpu_frame->draws.empty();
                if (!working_has_gpu) {
                    captured =
                        g_runtime.agc_cpu_surfaces.find(
                            g_runtime.agc_cpu_pending_capture_target);
                }
                const auto captured_end =
                    working_has_gpu
                    ? g_runtime.agc_cpu_working_surfaces.end()
                    : g_runtime.agc_cpu_surfaces.end();
                if (captured != captured_end &&
                    captured->second != nullptr) {
                    resolved_capture_draws =
                        captured->second->rendered_draws;
                    resolved_capture_coverage =
                        captured->second->covered_pixels;
                    resolved_capture_eligible =
                        captured->second->eligible_draw_mask;
                    resolved_capture_rendered =
                        captured->second->rendered_draw_mask;
                    if (HasAgcFrameContent(*captured->second) &&
                        captured->second->rendered_draws != 0u) {
                        cpu_frame = captured->second;
                        used_capture_surface = true;
                        g_runtime.agc_cpu_surfaces[flip_address] = cpu_frame;
                        g_runtime.agc_cpu_frame = cpu_frame;
                    }
                }
                g_runtime.agc_cpu_pending_capture_target = 0;
            }
            if (g_runtime.agc_cpu_capture_active &&
                g_runtime.agc_cpu_capture_target != 0) {
                g_runtime.agc_cpu_pending_capture_target =
                    g_runtime.agc_cpu_capture_target;
                g_runtime.agc_cpu_capture_active = false;
                g_runtime.agc_cpu_capture_target = 0;
                g_runtime.agc_cpu_next_render =
                    std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(16);
                queued_capture_surface = true;
            }
        }
#ifdef __ANDROID__
        android_window = g_runtime.android_window;
        if (android_window != nullptr) {
            ANativeWindow_acquire(android_window);
        }
#endif
        flip_count = ++g_runtime.agc_flip_count;
        const auto event_hint =
            UINT64_C(6) |
            ((flip_argument &
              UINT64_C(0x0000ffffffffffff))
             << 16u);
        notify_event_queue = EnqueueGuestVideoOutEventsLocked(
            port.flip_events, UINT64_C(6), event_hint);
        should_dump =
            std::getenv("PS5_RUNTIME_FRAME_DUMP_PATH") != nullptr &&
            g_runtime.agc_dumped_frames == 0;
        if (should_dump) {
            ++g_runtime.agc_dumped_frames;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto frame_period =
            std::chrono::microseconds(
                16667 * (std::clamp(port.flip_rate, 0, 2) + 1));
        if (g_runtime.agc_next_flip.time_since_epoch().count() == 0 ||
            now > g_runtime.agc_next_flip +
                      std::chrono::milliseconds(100)) {
            g_runtime.agc_next_flip = now + frame_period;
        }
        flip_deadline = g_runtime.agc_next_flip;
        g_runtime.agc_next_flip += frame_period;
    }
#ifdef __ANDROID__
    if (resolved_capture_target != 0 &&
        resolved_capture_draws != 0u) {
        static std::atomic<std::uint32_t> capture_resolution_logs{};
        if (capture_resolution_logs.fetch_add(
                1u, std::memory_order_relaxed) < 8u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "cpu capture resolve target=0x%llx draws=%u "
                "coverage=%llu eligible=0x%llx success=0x%llx used=%u",
                static_cast<unsigned long long>(
                    resolved_capture_target),
                resolved_capture_draws,
                static_cast<unsigned long long>(
                    resolved_capture_coverage),
                static_cast<unsigned long long>(
                    resolved_capture_eligible),
                static_cast<unsigned long long>(
                    resolved_capture_rendered),
                used_capture_surface ? 1u : 0u);
        }
    }
#endif
    if (notify_event_queue) {
        g_runtime.guest_event_queue_condition.notify_all();
    }

    const auto address = buffer_index >= 0
        ? port.buffer_addresses[
              static_cast<std::size_t>(buffer_index)]
        : 0u;
    // A degenerate diagnostic draw must not hide the real scanout buffer.
    // Large titles can emit a one-pixel setup primitive before their actual
    // display pass; treating it as a complete software frame produces an
    // indefinitely black image even though the registered VideoOut buffer is
    // being updated.
    if (cpu_frame != nullptr &&
        cpu_frame->covered_pixels <= 4u &&
        cpu_frame->rendered_draws <= 3u) {
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> weak_frame_logs{};
        if (weak_frame_logs.fetch_add(
                1u, std::memory_order_relaxed) < 8u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "ignore degenerate frame draws=%u coverage=%llu "
                "scanout=0x%llx",
                cpu_frame->rendered_draws,
                static_cast<unsigned long long>(
                    cpu_frame->covered_pixels),
                static_cast<unsigned long long>(address));
        }
#endif
        cpu_frame.reset();
    }
    if (buffer_index >= 0 &&
        (cpu_frame == nullptr ||
         !HasAgcFrameContent(*cpu_frame))) {
        presenter.address = address;
        presenter.width = port.width;
        presenter.height = port.height;
        if (presenter.tile_mode == 0u) {
            presenter.tile_mode = port.tiling_mode;
        }
    }
    if (flip_count <= 8 || flip_count % 60u == 0) {
        std::fprintf(
            stderr,
            "PS5_AGC_FLIP count=%llu handle=%d index=%d mode=%u "
            "arg=0x%llx address=0x%llx size=%ux%u format=0x%llx tile=%u\n",
            static_cast<unsigned long long>(flip_count),
            handle, buffer_index, flip_mode,
            static_cast<unsigned long long>(flip_argument),
            static_cast<unsigned long long>(address),
            port.width, port.height,
            static_cast<unsigned long long>(port.pixel_format),
            port.tiling_mode);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "video flip count=%llu handle=%d index=%d "
            "address=0x%llx size=%ux%u cpu_surface=%u capture=%u "
            "queued=%u "
            "surface_size=%ux%u",
            static_cast<unsigned long long>(flip_count),
            handle, buffer_index,
            static_cast<unsigned long long>(address),
            port.width, port.height,
            cpu_frame != nullptr &&
                    HasAgcFrameContent(*cpu_frame)
                ? 1u : 0u,
            used_capture_surface ? 1u : 0u,
            queued_capture_surface ? 1u : 0u,
            cpu_frame != nullptr ? cpu_frame->width : 0u,
            cpu_frame != nullptr ? cpu_frame->height : 0u);
#endif
    }
    std::this_thread::sleep_until(flip_deadline);

    if (buffer_index < 0) {
        return;
    }

#ifdef __ANDROID__
    PresentGuestTextureToAndroid(
        android_window, presenter, cpu_frame, flip_count);
    if (android_window != nullptr) {
        ANativeWindow_release(android_window);
    }
#endif

    const auto dump_address =
        presenter.address != 0 ? presenter.address : address;
    const auto dump_width =
        presenter.address != 0 ? presenter.width : port.width;
    const auto dump_height =
        presenter.address != 0 ? presenter.height : port.height;
    if (!should_dump || dump_address == 0 ||
        dump_width == 0 || dump_height == 0 ||
        dump_width > 8192 || dump_height > 8192) {
        return;
    }
    const auto pixel_count =
        static_cast<std::uint64_t>(dump_width) * dump_height;
    if (pixel_count >
        std::numeric_limits<std::size_t>::max() / 4u) {
        return;
    }
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(pixel_count) * 4u);
    if (!TryReadGuestBytes(
            dump_address, rgba.data(), rgba.size())) {
        std::fprintf(
            stderr,
            "PS5_AGC_FRAME_DUMP_FAILED reason=guest-read "
            "address=0x%llx bytes=%zu\n",
            static_cast<unsigned long long>(dump_address),
            rgba.size());
        return;
    }
    std::uint64_t fingerprint = UINT64_C(1469598103934665603);
    std::uint64_t nonzero_bytes{};
    for (const auto value : rgba) {
        fingerprint ^= value;
        fingerprint *= UINT64_C(1099511628211);
        nonzero_bytes += value != 0;
    }

    const char* const path =
        std::getenv("PS5_RUNTIME_FRAME_DUMP_PATH");
    FILE* const output = path != nullptr
        ? std::fopen(path, "wb")
        : nullptr;
    if (output == nullptr) {
        std::fprintf(
            stderr,
            "PS5_AGC_FRAME_DUMP_FAILED reason=open path=%s\n",
            path != nullptr ? path : "(null)");
        return;
    }
    std::fprintf(
        output, "P6\n%u %u\n255\n",
        dump_width, dump_height);
    std::array<std::uint8_t, 3> rgb{};
    for (std::size_t pixel = 0;
         pixel < static_cast<std::size_t>(pixel_count); ++pixel) {
        const auto source = pixel * 4u;
        rgb[0] = rgba[source + 0];
        rgb[1] = rgba[source + 1];
        rgb[2] = rgba[source + 2];
        (void)std::fwrite(rgb.data(), 1, rgb.size(), output);
    }
    std::fclose(output);
    std::fprintf(
        stderr,
        "PS5_AGC_FRAME_DUMP path=%s bytes=%zu nonzero=%llu "
        "fingerprint=0x%llx\n",
        path, rgba.size(),
        static_cast<unsigned long long>(nonzero_bytes),
        static_cast<unsigned long long>(fingerprint));
}

bool TryAgcDriverSubmitDcb(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto submit_started = std::chrono::steady_clock::now();
    const std::lock_guard submit_lock{g_agc_submit_mutex};
    const auto packet_address = request.integer_arguments[0];
    std::uint64_t command_address{};
    std::uint32_t dword_count{};
    if (packet_address == 0 ||
        !TryReadGuestValue(packet_address, command_address) ||
        !TryReadGuestValue(packet_address + 8u, dword_count) ||
        command_address == 0 || dword_count == 0 ||
        dword_count > (16u * 1024u * 1024u)) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }

    bool should_dump_dcb{};
    {
        std::uint32_t minimum_dwords{};
        if (const char* const minimum =
                std::getenv("PS5_RUNTIME_DCB_DUMP_MIN_DWORDS");
            minimum != nullptr) {
            minimum_dwords = static_cast<std::uint32_t>(
                std::strtoul(minimum, nullptr, 0));
        }
        const std::lock_guard lock{g_runtime.mutex};
        should_dump_dcb =
            std::getenv("PS5_RUNTIME_DCB_DUMP_PATH") != nullptr &&
            dword_count >= minimum_dwords &&
            g_runtime.agc_dumped_dcbs == 0;
        if (should_dump_dcb) {
            ++g_runtime.agc_dumped_dcbs;
        }
    }
    if (should_dump_dcb) {
        std::vector<std::uint32_t> dwords(dword_count);
        const char* const path =
            std::getenv("PS5_RUNTIME_DCB_DUMP_PATH");
        if (TryReadGuestBytes(
                command_address, dwords.data(),
                dwords.size() * sizeof(std::uint32_t))) {
            FILE* const output =
                path != nullptr ? std::fopen(path, "wb") : nullptr;
            if (output != nullptr) {
                (void)std::fwrite(
                    dwords.data(), sizeof(std::uint32_t),
                    dwords.size(), output);
                std::fclose(output);
                std::fprintf(
                    stderr,
                    "PS5_AGC_DCB_DUMP path=%s address=0x%llx "
                    "dwords=%u bytes=%zu\n",
                    path,
                    static_cast<unsigned long long>(command_address),
                    dword_count,
                    dwords.size() * sizeof(std::uint32_t));
            }
        }
    }

    static thread_local std::unordered_map<
        std::uint32_t, std::uint32_t> cx_registers;
    static thread_local std::unordered_map<
        std::uint32_t, std::uint32_t> sh_registers;
    static thread_local std::unordered_map<
        std::uint32_t, std::uint32_t> uc_registers;
    std::array<std::uint32_t, 256> opcode_counts{};
    std::array<std::uint32_t, 64> nop_register_counts{};
    std::uint32_t parsed_packet_count{};
    std::uint32_t parsed_draw_count{};
    struct DeferredReleaseWrite {
        bool standard_packet{};
        std::uint32_t destination_selection{};
        std::uint32_t data_selection{};
        std::uint64_t destination_address{};
        std::uint64_t data{};
    };
    std::vector<DeferredReleaseWrite> deferred_release_writes;
    deferred_release_writes.reserve(8u);
    struct DeferredDataWrite {
        bool standard_packet{};
        std::uint32_t destination_selection{};
        bool increment_address{};
        std::uint64_t destination_address{};
        std::vector<std::uint32_t> values;
    };
    std::vector<DeferredDataWrite> deferred_data_writes;
    deferred_data_writes.reserve(32u);
    std::uint64_t index_buffer_address{};
    std::uint32_t index_buffer_count{};
    std::uint32_t index_size{};
    std::uint32_t instance_count{1};
    std::uint64_t dispatch_indirect_args_base{};
    std::uint64_t runtime_texture_cache_epoch{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        cx_registers = g_runtime.agc_cx_registers;
        sh_registers = g_runtime.agc_sh_registers;
        uc_registers = g_runtime.agc_uc_registers;
        index_buffer_address =
            g_runtime.agc_index_buffer_address;
        index_buffer_count =
            g_runtime.agc_index_buffer_count;
        index_size = g_runtime.agc_index_size;
        instance_count = g_runtime.agc_instance_count;
        dispatch_indirect_args_base =
            g_runtime.agc_dispatch_indirect_args_base;
        runtime_texture_cache_epoch =
            g_runtime.agc_cpu_texture_cache_epoch;
    }
#ifdef __ANDROID__
    ANativeWindow* graphics_window{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        graphics_window = g_runtime.android_window;
        if (graphics_window != nullptr) {
            ANativeWindow_acquire(graphics_window);
        }
    }
    if (graphics_window != nullptr) {
        (void)Lsx4::Ps5Desktop::EnsureVulkanPresenter(
            graphics_window);
        ANativeWindow_release(graphics_window);
    }
#endif
    std::vector<AgcDiagnosticDraw> diagnostic_draws;
    diagnostic_draws.reserve(32u);
    const auto is_cpu_draw_candidate =
        [&](const AgcDiagnosticDraw& draw) {
            const auto game_maker_fill =
                draw.draw_count == 6u &&
                (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
                    UINT32_C(0xf13);
            if ((draw.draw_count == 3u || game_maker_fill) &&
                draw.texture_address != 0 &&
                draw.texture_width == 5u &&
                draw.texture_height == 1u &&
                draw.texture_tile_mode == 0u) {
                return true;
            }
            if (draw.primitive_type != UINT32_C(0x4) &&
                draw.primitive_type != UINT32_C(0x6) &&
                draw.primitive_type != UINT32_C(0x11)) {
                return false;
            }
            if ((draw.sh_es_program_lo & UINT32_C(0xfff)) ==
                    UINT32_C(0xc40) &&
                (draw.sh_ps_program_lo & UINT32_C(0xfff)) ==
                    UINT32_C(0xc38)) {
                return false;
            }
            const auto render_target_source =
                std::ranges::any_of(
                    diagnostic_draws,
                    [&](const AgcDiagnosticDraw& producer) {
                        return producer.render_target_address != 0u &&
                            producer.render_target_address ==
                                draw.texture_address;
                    });
            return draw.sh_es_program_hi == 0 &&
                draw.sh_es_program_lo != 0u &&
                draw.sh_ps_program_hi == 0 &&
                draw.sh_ps_program_lo != 0u &&
                (render_target_source ||
                 ((draw.texture_tile_mode == 0 ||
                   draw.texture_tile_mode == 5) &&
                  (draw.texture_format == UINT32_C(0x1) ||
                   draw.texture_format == UINT32_C(0x38) ||
                   draw.texture_format == UINT32_C(0xad) ||
                   draw.texture_format == UINT32_C(0xb5))) ||
                 (draw.texture_tile_mode == UINT32_C(27) &&
                  draw.texture_format == UINT32_C(0x47))) &&
                draw.texture_width >= 16u &&
                draw.texture_height >= 16u;
        };
    static thread_local std::vector<AgcCpuDecodedTexture>
        cpu_texture_cache;
    static thread_local std::uint64_t cpu_texture_cache_epoch{};
    static thread_local std::uint64_t cpu_texture_probe_serial{};
    static thread_local std::uint64_t gen5_route_epoch{};
    static thread_local std::unordered_set<std::uint64_t>
        gen5_only_shader_pairs;
    ++cpu_texture_probe_serial;
    if (cpu_texture_cache_epoch != runtime_texture_cache_epoch) {
        cpu_texture_cache.clear();
        cpu_texture_cache_epoch = runtime_texture_cache_epoch;
    }
    if (gen5_route_epoch != runtime_texture_cache_epoch) {
        gen5_only_shader_pairs.clear();
        gen5_route_epoch = runtime_texture_cache_epoch;
    }
    const auto gen5_shader_pair_key =
        [](const AgcDiagnosticDraw& draw) {
            const auto vertex =
                (static_cast<std::uint64_t>(
                     draw.sh_es_program_hi & 0xffu) << 40u) |
                (static_cast<std::uint64_t>(
                     draw.sh_es_program_lo) << 8u);
            const auto pixel =
                (static_cast<std::uint64_t>(
                     draw.sh_ps_program_hi & 0xffu) << 40u) |
                (static_cast<std::uint64_t>(
                     draw.sh_ps_program_lo) << 8u);
            std::uint64_t key =
                vertex ^ (pixel + UINT64_C(0x9e3779b97f4a7c15) +
                          (vertex << 6u) + (vertex >> 2u));
            return key != 0u ? key : UINT64_C(1);
        };
    std::shared_ptr<AgcCpuFrame> cpu_frame;
    std::uint64_t cpu_frame_target{};
    std::unordered_map<
        std::uint64_t, std::shared_ptr<AgcCpuFrame>>
        cpu_command_surfaces;
    bool cpu_render_decided{};
    bool cpu_render_allowed{};
    std::uint64_t cpu_capture_target{};
    std::array<std::uint32_t, 5> diagnostic_flip{};
    bool has_diagnostic_flip{};
    std::uint32_t release_write_count{};
    std::uint32_t release_write_failures{};
    std::uint32_t data_write_count{};
    std::uint32_t data_write_failures{};
    std::uint32_t dma_write_count{};
    std::uint32_t dma_write_failures{};
    std::uint32_t offset{};
    while (offset < dword_count) {
        std::uint32_t header{};
        if (!TryReadGuestValue(
                command_address +
                    static_cast<std::uint64_t>(offset) * 4u,
                header)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        if ((header & UINT32_C(0xc0000000)) != UINT32_C(0xc0000000)) {
            ++offset;
            continue;
        }
        const auto length = ((header >> 16u) & 0x3fffu) + 2u;
        const auto opcode = (header >> 8u) & 0xffu;
        const auto packet_register = (header >> 2u) & 0x3fu;
        ++opcode_counts[opcode];
        if (opcode == AgcItNop) {
            ++nop_register_counts[packet_register];
        }
        ++parsed_packet_count;
        if (length == 0 || length > dword_count - offset) {
            break;
        }
        const auto packet_address =
            command_address +
            static_cast<std::uint64_t>(offset) * 4u;
#ifdef __ANDROID__
        if (opcode == AgcItNop &&
            packet_register == AgcRReleaseMem) {
            static std::atomic<std::uint32_t> release_packet_logs{};
            const auto log_index = release_packet_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (log_index >= 16u) {
                goto skip_packet_diagnostic_log;
            }
            std::array<std::uint32_t, 8> words{};
            const auto readable_dwords =
                std::min<std::uint32_t>(length, words.size());
            const bool read_packet = TryReadGuestBytes(
                packet_address, words.data(),
                static_cast<std::size_t>(readable_dwords) * 4u);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "release_packet len=%u read=%d "
                "words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                length, read_packet ? 1 : 0,
                words[0], words[1], words[2], words[3],
                words[4], words[5], words[6], words[7]);
        } else if (opcode == AgcItNop &&
                   (packet_register == AgcRWaitMem32 ||
                    packet_register == AgcRWaitMem64)) {
            static std::atomic<std::uint32_t> wait_packet_logs{};
            const auto log_index = wait_packet_logs.fetch_add(
                1u, std::memory_order_relaxed);
            if (log_index >= 16u) {
                goto skip_packet_diagnostic_log;
            }
            std::array<std::uint32_t, 9> words{};
            const auto readable_dwords =
                std::min<std::uint32_t>(length, words.size());
            const bool read_packet = TryReadGuestBytes(
                packet_address, words.data(),
                static_cast<std::size_t>(readable_dwords) * 4u);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "wait_packet bits=%u len=%u read=%d "
                "words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                packet_register == AgcRWaitMem32 ? 32u : 64u,
                length, read_packet ? 1 : 0,
                words[0], words[1], words[2], words[3],
                words[4], words[5], words[6], words[7],
                words[8]);
        }
skip_packet_diagnostic_log:
#endif
        const auto apply_release_mem =
            [&](const bool standard_packet) {
                std::uint32_t control{};
                std::uint32_t destination_low{};
                std::uint32_t destination_high{};
                std::uint32_t data_low{};
                std::uint32_t data_high{};
                if (!TryReadGuestValue(
                        packet_address + 8u, control) ||
                    !TryReadGuestValue(
                        packet_address + 12u,
                        destination_low) ||
                    !TryReadGuestValue(
                        packet_address + 16u,
                        destination_high) ||
                    !TryReadGuestValue(
                        packet_address + 20u, data_low) ||
                    !TryReadGuestValue(
                        packet_address + 24u, data_high)) {
                    ++release_write_failures;
                    return;
                }
                const auto destination_address =
                    static_cast<std::uint64_t>(destination_low) |
                    (static_cast<std::uint64_t>(
                         destination_high) << 32u);
                const auto data_selection = standard_packet
                    ? ((control >> 29u) & 0x7u)
                    : ((control >> 16u) & 0xffu);
                const auto destination = standard_packet
                    ? ((control >> 16u) & 0x3u)
                    : 0u;
                const auto data =
                    static_cast<std::uint64_t>(data_low) |
                    (static_cast<std::uint64_t>(
                         data_high) << 32u);
                const bool destination_supported =
                    !standard_packet || destination <= 1u;
                const bool operation_supported =
                    data_selection == 1u ||
                    data_selection == 2u ||
                    data_selection == 3u ||
                    (standard_packet && data_selection == 4u);
                if (destination_supported &&
                    destination_address != 0 &&
                    operation_supported) {
                    deferred_release_writes.push_back({
                        .standard_packet = standard_packet,
                        .destination_selection = destination,
                        .data_selection = data_selection,
                        .destination_address = destination_address,
                        .data = data,
                    });
                } else {
                    ++release_write_failures;
                }
            };
        if (opcode == AgcItNop &&
            packet_register == AgcRReleaseMem &&
            length >= 7u) {
            apply_release_mem(false);
        } else if (opcode == AgcItReleaseMem &&
                   length >= 8u) {
            apply_release_mem(true);
        }
        const bool agc_write_data =
            opcode == AgcItNop &&
            packet_register == AgcRWriteData &&
            length >= 4u;
        const bool standard_write_data =
            opcode == AgcItWriteData &&
            length >= 4u;
        if (agc_write_data || standard_write_data) {
            std::uint32_t control{};
            std::uint64_t destination_address{};
            DeferredDataWrite write{
                .standard_packet = standard_write_data,
            };
            bool readable =
                TryReadGuestValue(
                    packet_address + 4u, control) &&
                TryReadGuestValue(
                    packet_address + 8u,
                    destination_address);
            if (standard_write_data) {
                write.destination_selection =
                    (control >> 8u) & 0x0fu;
                write.increment_address =
                    (control & (1u << 16u)) == 0;
            } else {
                write.destination_selection =
                    control & 0xffu;
                write.increment_address =
                    ((control >> 16u) & 0xffu) == 0;
            }
            write.destination_address =
                destination_address;
            const auto value_count = length - 4u;
            write.values.reserve(value_count);
            for (std::uint32_t value_index = 0;
                 readable && value_index < value_count;
                 ++value_index) {
                std::uint32_t value{};
                readable = TryReadGuestValue(
                    packet_address + 16u +
                        static_cast<std::uint64_t>(
                            value_index) * 4u,
                    value);
                if (readable) {
                    write.values.push_back(value);
                }
            }
            const bool supported_destination =
                write.destination_selection == 1u ||
                write.destination_selection == 2u ||
                write.destination_selection == 4u ||
                write.destination_selection == 5u;
            if (readable && supported_destination &&
                destination_address != 0 &&
                write.values.size() == value_count) {
                deferred_data_writes.push_back(
                    std::move(write));
            } else {
                ++data_write_failures;
            }
        }
        if (opcode == AgcItDmaData && length >= 7u) {
            std::uint32_t control{};
            std::uint32_t command{};
            std::uint64_t destination_address{};
            std::uint64_t source_address{};
            bool copied{};
            std::uint32_t byte_count{};
            std::uint32_t source_selection{};
            std::uint32_t destination_selection{};
            const bool readable =
                TryReadGuestValue(packet_address + 4u, control) &&
                TryReadGuestValue(
                    packet_address + 8u, source_address) &&
                TryReadGuestValue(
                    packet_address + 16u, destination_address) &&
                TryReadGuestValue(packet_address + 24u, command);
            if (readable) {
                byte_count = command & 0x03ffffffu;
                source_selection =
                    ((control >> 29u) & 0x3u) |
                    ((command >> 24u) & 0x4u) |
                    ((command >> 25u) & 0x8u);
                destination_selection =
                    ((control >> 20u) & 0x3u) |
                    ((command >> 25u) & 0x4u) |
                    ((command >> 26u) & 0x8u);
                const bool destination_memory =
                    destination_selection == 0u ||
                    destination_selection == 3u;
                const bool source_memory =
                    source_selection == 0u ||
                    source_selection == 3u;
                if (byte_count != 0 &&
                    byte_count <= 256u * 1024u * 1024u &&
                    destination_address != 0 &&
                    destination_memory) {
                    std::vector<std::uint8_t> bytes(byte_count);
                    if (source_selection == 2u) {
                        const auto fill =
                            static_cast<std::uint32_t>(
                                source_address);
                        for (std::size_t index = 0;
                             index < bytes.size(); ++index) {
                            bytes[index] =
                                static_cast<std::uint8_t>(
                                    fill >>
                                    ((index & 3u) * 8u));
                        }
                        copied = TryWriteGuestBytes(
                            destination_address,
                            bytes.data(), bytes.size());
                    } else if (source_memory &&
                               source_address != 0 &&
                               TryReadGuestBytes(
                                   source_address,
                                   bytes.data(), bytes.size())) {
                        copied = TryWriteGuestBytes(
                            destination_address,
                            bytes.data(), bytes.size());
                    }
                }
            }
            dma_write_count += copied;
            dma_write_failures += !copied;
#ifdef __ANDROID__
            static std::atomic<std::uint32_t> standard_dma_logs{};
            if (standard_dma_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 8u) {
                __android_log_print(
                    copied ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                    "LSX4-PS5",
                    "dma_data standard=1 dst=0x%llx src=0x%llx "
                    "bytes=%u src_sel=%u dst_sel=%u copied=%d",
                    static_cast<unsigned long long>(
                        destination_address),
                    static_cast<unsigned long long>(
                        source_address),
                    byte_count, source_selection,
                    destination_selection, copied ? 1 : 0);
            }
#endif
        }
        if (opcode == AgcItNop &&
            packet_register == AgcRDmaData &&
            length >= 7u) {
            const bool compact_layout = length == 7u;
            const auto byte_count_offset =
                compact_layout ? 20u : 12u;
            const auto destination_offset =
                compact_layout ? 4u : 16u;
            const auto source_offset =
                compact_layout ? 12u : 24u;
            const auto selector_offset =
                compact_layout ? 24u : 4u;
            std::uint32_t byte_count{};
            std::uint32_t selectors{};
            std::uint64_t destination_address{};
            std::uint64_t source_address{};
            bool copied{};
            if (TryReadGuestValue(
                    packet_address + byte_count_offset,
                    byte_count) &&
                TryReadGuestValue(
                    packet_address + selector_offset,
                    selectors) &&
                TryReadGuestValue(
                    packet_address + destination_offset,
                    destination_address) &&
                TryReadGuestValue(
                    packet_address + source_offset,
                    source_address) &&
                byte_count != 0 &&
                byte_count <= 256u * 1024u * 1024u &&
                destination_address != 0) {
                std::vector<std::uint8_t> bytes(byte_count);
                const auto source_selection =
                    compact_layout
                        ? selectors & 0xffu
                        : (selectors >> 16u) & 0xffu;
                const bool immediate_fill =
                    source_selection == 2u ||
                    (compact_layout &&
                     destination_address >= 0x10000u &&
                     source_address <=
                         std::numeric_limits<std::uint32_t>::max());
                if (immediate_fill) {
                    const auto fill =
                        static_cast<std::uint32_t>(
                            source_address);
                    for (std::size_t index = 0;
                         index < bytes.size(); ++index) {
                        bytes[index] = static_cast<std::uint8_t>(
                            fill >> ((index & 3u) * 8u));
                    }
                    copied = TryWriteGuestBytes(
                        destination_address,
                        bytes.data(), bytes.size());
                } else if (source_address != 0 &&
                           TryReadGuestBytes(
                               source_address,
                               bytes.data(), bytes.size())) {
                    copied = TryWriteGuestBytes(
                        destination_address,
                        bytes.data(), bytes.size());
                }
            }
            dma_write_count += copied;
            dma_write_failures += !copied;
#ifdef __ANDROID__
            static std::atomic<std::uint32_t> dma_logs{};
            if (dma_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 8u) {
                std::uint64_t observed{};
                const bool observed_ok =
                    destination_address != 0 &&
                    TryReadGuestValue(
                        destination_address, observed);
                __android_log_print(
                    copied ? ANDROID_LOG_INFO
                           : ANDROID_LOG_WARN,
                    "LSX4-PS5",
                    "dma_data compact=%d dst=0x%llx "
                    "src=0x%llx bytes=%u selectors=%08x copied=%d "
                    "observed=0x%llx read=%d",
                    compact_layout ? 1 : 0,
                    static_cast<unsigned long long>(
                        destination_address),
                    static_cast<unsigned long long>(
                        source_address),
                    byte_count, selectors, copied ? 1 : 0,
                    static_cast<unsigned long long>(observed),
                    observed_ok ? 1 : 0);
            }
#endif
        }
        if (opcode == AgcItNop &&
            packet_register == AgcRDrawReset) {
            cx_registers.clear();
            sh_registers.clear();
            uc_registers.clear();
            index_buffer_address = 0;
            index_buffer_count = 0;
            index_size = 0;
            instance_count = 1;
        }
        if (opcode == AgcItNop &&
            (packet_register == AgcRCxRegsIndirect ||
             packet_register == AgcRShRegsIndirect ||
             packet_register == AgcRUcRegsIndirect) &&
            length >= 4u) {
            std::uint32_t register_count{};
            std::uint64_t registers_address{};
            if (TryReadGuestValue(
                    packet_address + 4u, register_count) &&
                TryReadGuestValue(
                    packet_address + 8u, registers_address) &&
                register_count <= 4096u) {
                auto* destination =
                    packet_register == AgcRCxRegsIndirect
                    ? &cx_registers
                    : (packet_register == AgcRShRegsIndirect
                           ? &sh_registers
                           : &uc_registers);
                for (std::uint32_t index = 0;
                     index < register_count; ++index) {
                    std::uint32_t register_offset{};
                    std::uint32_t value{};
                    const auto entry =
                        registers_address +
                        static_cast<std::uint64_t>(index) * 8u;
                    if (!TryReadGuestValue(
                            entry, register_offset) ||
                        !TryReadGuestValue(
                            entry + 4u, value)) {
                        break;
                    }
                    (*destination)[register_offset] = value;
                }
            }
        }
        if ((opcode == AgcItSetContextReg ||
             opcode == AgcItSetShReg ||
             opcode == AgcItSetUconfigReg) &&
            length >= 3u) {
            std::uint32_t start_register{};
            if (TryReadGuestValue(
                    packet_address + 4u, start_register)) {
                auto* destination =
                    opcode == AgcItSetContextReg
                    ? &cx_registers
                    : (opcode == AgcItSetShReg
                           ? &sh_registers
                           : &uc_registers);
                for (std::uint32_t index = 0;
                     index < length - 2u; ++index) {
                    std::uint32_t value{};
                    if (!TryReadGuestValue(
                            packet_address + 8u +
                                static_cast<std::uint64_t>(index) *
                                    4u,
                            value)) {
                        break;
                    }
                    (*destination)[start_register + index] = value;
                }
            }
        }
        if (opcode == AgcItSetBase && length == 4u) {
            std::uint32_t base_index{};
            std::uint32_t address_low{};
            std::uint32_t address_high{};
            if (TryReadGuestValue(
                    packet_address + 4u, base_index) &&
                TryReadGuestValue(
                    packet_address + 8u, address_low) &&
                TryReadGuestValue(
                    packet_address + 12u, address_high) &&
                (base_index & 0xfu) == 1u) {
                // libSceAgc encodes Gnmp::ShaderType in PM4 header bit 1:
                // zero is draw, one is dispatch.
                const auto shader_type = (header >> 1u) & 0x3u;
                if (shader_type == 1u) {
                    dispatch_indirect_args_base =
                        static_cast<std::uint64_t>(
                            address_low & ~7u) |
                        (static_cast<std::uint64_t>(
                             address_high & 0xffffu)
                         << 32u);
                }
            }
        }
        if (opcode == AgcItIndexBase && length >= 3u) {
            (void)TryReadGuestValue(
                packet_address + 4u, index_buffer_address);
        } else if (opcode == AgcItIndexBufferSize &&
                   length >= 2u) {
            (void)TryReadGuestValue(
                packet_address + 4u, index_buffer_count);
        } else if (opcode == AgcItNop &&
                   packet_register == AgcRIndexCount &&
                   length >= 2u) {
            (void)TryReadGuestValue(
                packet_address + 4u, index_buffer_count);
        } else if (opcode == AgcItIndexType &&
                   length >= 2u) {
            (void)TryReadGuestValue(
                packet_address + 4u, index_size);
            index_size &= 3u;
        } else if (opcode == AgcItNumInstances &&
                   length >= 2u) {
            (void)TryReadGuestValue(
                packet_address + 4u, instance_count);
            instance_count = std::max(instance_count, 1u);
        }
#ifdef __ANDROID__
        if ((opcode == AgcItDispatchDirect ||
             opcode == AgcItDispatchIndirect) &&
            length >= 3u) {
            static std::atomic<std::uint32_t> dispatch_logs{};
            const auto log_index = dispatch_logs.fetch_add(
                1u, std::memory_order_relaxed);
            {
                std::uint64_t dimensions_address =
                    packet_address + 4u;
                std::uint32_t initiator{};
                if (opcode == AgcItDispatchIndirect) {
                    if (length >= 4u) {
                        // ACB packet: absolute 64-bit argument address.
                        (void)TryReadGuestValue(
                            packet_address + 4u,
                            dimensions_address);
                        (void)TryReadGuestValue(
                            packet_address + 12u, initiator);
                    } else {
                        // DCB packet: 32-bit byte offset from the
                        // dispatch argument base established by SET_BASE.
                        std::uint32_t data_offset{};
                        (void)TryReadGuestValue(
                            packet_address + 4u, data_offset);
                        dimensions_address =
                            dispatch_indirect_args_base +
                            data_offset;
                        (void)TryReadGuestValue(
                            packet_address + 8u, initiator);
                    }
                } else if (opcode == AgcItDispatchDirect &&
                           length >= 5u) {
                    (void)TryReadGuestValue(
                        packet_address + 16u, initiator);
                }
                std::array<std::uint32_t, 3> dimensions{};
                (void)TryReadGuestBytes(
                    dimensions_address, dimensions.data(),
                    sizeof(dimensions));
                if (opcode == AgcItDispatchDirect &&
                    length < 5u) {
                    dimensions[2] = 1u;
                    if (length >= 4u) {
                        (void)TryReadGuestValue(
                            packet_address + 12u, initiator);
                    }
                }
                for (auto& dimension : dimensions) {
                    if (dimension > 65535u) {
                        dimension = 1u;
                    }
                }
                const auto value =
                    [&](const std::uint32_t index) {
                        const auto found = sh_registers.find(index);
                        return found == sh_registers.end()
                            ? 0u : found->second;
                    };
                const auto shader_address =
                    (static_cast<std::uint64_t>(
                         value(AgcComputePgmHi)) << 40u) |
                    (static_cast<std::uint64_t>(
                         value(AgcComputePgmLo)) << 8u);
                struct CachedGen5Kernel {
                    bool attempted{};
                    bool compiled{};
                    std::vector<std::uint32_t> spirv;
                    Lsx4::Ps5Desktop::Gen5ComputeProgramInfo info;
                };
                static std::mutex gen5_cache_mutex;
                static std::unordered_map<
                    std::uint64_t, CachedGen5Kernel> gen5_cache;
                Lsx4::Ps5Desktop::Gen5ComputeProgramInfo
                    program_info{};
                std::vector<std::uint32_t> program_spirv;
                bool compiled{};
                if (shader_address != 0) {
                    auto kernel_key = shader_address;
                    for (std::uint32_t index = 0;
                         index < 64u; ++index) {
                        const auto word =
                            value(AgcComputeUserData + index);
                        kernel_key ^=
                            static_cast<std::uint64_t>(word) +
                            UINT64_C(0x9e3779b97f4a7c15) +
                            (kernel_key << 6u) +
                            (kernel_key >> 2u);
                    }
                    for (const auto word : {
                             value(AgcComputeNumThreadX),
                             value(AgcComputeNumThreadY),
                             value(AgcComputeNumThreadZ),
                             value(AgcComputePgmRsrc1),
                             value(AgcComputePgmRsrc2),
                             initiator & UINT32_C(0x20),
                             dimensions[0], dimensions[1],
                             dimensions[2]}) {
                        kernel_key ^=
                            static_cast<std::uint64_t>(word) +
                            UINT64_C(0x9e3779b97f4a7c15) +
                            (kernel_key << 6u) +
                            (kernel_key >> 2u);
                    }
                    const std::lock_guard cache_lock{
                        gen5_cache_mutex};
                    auto& cached = gen5_cache[kernel_key];
                    if (!cached.attempted &&
                        gen5_cache.size() <= 256u) {
                        cached.attempted = true;
                        std::array<std::uint32_t, 4096>
                            shader_code{};
                        if (TryReadGuestBytes(
                                shader_address, shader_code.data(),
                                sizeof(shader_code))) {
                            Lsx4::Ps5Desktop::
                                Gen5ComputeCompileRequest
                                    compile_request{};
                            compile_request.shader_address =
                                shader_address;
                            compile_request.code = shader_code;
                            for (std::uint32_t index = 0;
                                 index <
                                     compile_request.user_data.size();
                                 ++index) {
                                compile_request.user_data[index] =
                                    value(
                                        AgcComputeUserData + index);
                            }
                            compile_request.local_size = {
                                std::max(
                                    value(AgcComputeNumThreadX), 1u),
                                std::max(
                                    value(AgcComputeNumThreadY), 1u),
                                std::max(
                                    value(AgcComputeNumThreadZ), 1u)};
                            const auto compute_rsrc2 =
                                value(AgcComputePgmRsrc2);
                            compile_request.user_data_count =
                                (compute_rsrc2 >> 1u) &
                                UINT32_C(0x1f);
                            compile_request.compute_group_id = {
                                ((compute_rsrc2 >> 7u) & 1u) != 0u,
                                ((compute_rsrc2 >> 8u) & 1u) != 0u,
                                ((compute_rsrc2 >> 9u) & 1u) != 0u};
                            compile_request.compute_tg_size_en =
                                ((compute_rsrc2 >> 10u) & 1u) != 0u;
                            compile_request.compute_thread_ids_num =
                                ((compute_rsrc2 >> 11u) & 3u) + 1u;
                            compile_request.compute_workgroup_register =
                                compile_request.user_data_count;
                            compile_request
                                .dispatch_thread_dimensions =
                                    (initiator & UINT32_C(0x20)) != 0u;
                            compile_request.dispatch_threads =
                                dimensions;
                            compile_request.wave_size =
                                (value(AgcComputePgmRsrc1) &
                                 UINT32_C(0x40000000)) != 0
                                ? 32u : 64u;
                            compile_request.read_memory =
                                [](void*,
                                   const std::uint64_t address,
                                   std::uint32_t* const output) {
                                    return output != nullptr &&
                                        TryReadGuestValue(
                                            address, *output);
                            };
                            std::string compile_error;
                            __android_log_print(
                                ANDROID_LOG_INFO,
                                "LSX4-PS5-GEN5",
                                "compile begin shader=0x%llx wave=%u "
                                "local=%u/%u/%u",
                                static_cast<unsigned long long>(
                                    shader_address),
                                compile_request.wave_size,
                                compile_request.local_size[0],
                                compile_request.local_size[1],
                                compile_request.local_size[2]);
                            cached.compiled =
                                Lsx4::Ps5Desktop::
                                    TryCompileGen5Compute(
                                        compile_request, cached.spirv,
                                        &cached.info,
                                        compile_error);
                            __android_log_print(
                                cached.compiled
                                    ? ANDROID_LOG_INFO
                                    : ANDROID_LOG_WARN,
                                "LSX4-PS5-GEN5",
                                "compile shader=0x%llx wave=%u "
                                "local=%u/%u/%u ok=%u words=%zu "
                                "buffers=%zu images=%zu addresses=%zu "
                                "copy=%u image_copy=%u "
                                "src_ud=%u dst_ud=%u error=%s",
                                static_cast<unsigned long long>(
                                    shader_address),
                                compile_request.wave_size,
                                compile_request.local_size[0],
                                compile_request.local_size[1],
                                compile_request.local_size[2],
                                cached.compiled ? 1u : 0u,
                                cached.spirv.size(),
                                cached.info.buffers.size(),
                                cached.info.images.size(),
                                cached.info.address_count,
                                cached.info.simple_buffer_copy
                                    ? 1u : 0u,
                                cached.info.simple_image_copy
                                    ? 1u : 0u,
                                cached.info.copy_source_user_data,
                                cached.info
                                    .copy_destination_user_data,
                                compile_error.c_str());
                            if (cached.compiled) {
                                for (const auto& binding :
                                     cached.info.descriptors) {
                                    __android_log_print(
                                        ANDROID_LOG_INFO,
                                        "LSX4-PS5-GEN5",
                                        "layout shader=0x%llx set=%u "
                                        "binding=%u kind=%u count=%zu "
                                        "push=%u+%u",
                                        static_cast<unsigned long long>(
                                            shader_address),
                                        cached.info.descriptor_set,
                                        binding.binding, binding.kind,
                                        binding.resources.size(),
                                        cached.info.push_constant_offset,
                                        cached.info.push_constant_size);
                                }
                                for (std::uint32_t index = 0;
                                     index < cached.info.images.size();
                                     ++index) {
                                    const auto& image =
                                        cached.info.images[index];
                                    __android_log_print(
                                        ANDROID_LOG_INFO,
                                        "LSX4-PS5-GEN5",
                                        "image shader=0x%llx i=%u "
                                        "addr=0x%llx size=%ux%ux%u "
                                        "type=%u fmt=%u tile=%u kind=%u "
                                        "rw=%u/%u/%u",
                                        static_cast<unsigned long long>(
                                            shader_address),
                                        index,
                                        static_cast<unsigned long long>(
                                            image.guest_address),
                                        image.width, image.height,
                                        image.depth, image.type,
                                        image.format, image.tile_mode,
                                        image.resource_kind,
                                        image.read ? 1u : 0u,
                                        image.written ? 1u : 0u,
                                        image.atomic ? 1u : 0u);
                                }
                                for (std::uint32_t index = 0;
                                     index <
                                         cached.info.addresses.size();
                                     ++index) {
                                    const auto& address =
                                        cached.info.addresses[index];
                                    __android_log_print(
                                        ANDROID_LOG_INFO,
                                        "LSX4-PS5-GEN5",
                                        "address shader=0x%llx i=%u "
                                        "guest=0x%llx binding=0x%llx "
                                        "min=%d rw=%u/%u/%u",
                                        static_cast<unsigned long long>(
                                            shader_address),
                                        index,
                                        static_cast<unsigned long long>(
                                            address.guest_base),
                                        static_cast<unsigned long long>(
                                            address.binding_base),
                                        address.min_offset,
                                        address.read ? 1u : 0u,
                                        address.written ? 1u : 0u,
                                        address.atomic ? 1u : 0u);
                                }
                            }
                        }
                    }
                    compiled = cached.compiled;
                    program_info = cached.info;
                    program_spirv = cached.spirv;
                }
                if (compiled) {
                    for (auto& buffer : program_info.buffers) {
                        const auto offset =
                            buffer.user_data_dword;
                        if (offset == UINT32_MAX ||
                            offset + 2u >= 64u) {
                            continue;
                        }
                        for (std::uint32_t word = 0;
                             word <
                                 buffer.descriptor_dword_count &&
                             offset + word < 64u;
                             ++word) {
                            buffer.descriptor[word] =
                                value(
                                    AgcComputeUserData +
                                    offset + word);
                        }
                        buffer.guest_address =
                            static_cast<std::uint64_t>(
                                buffer.descriptor[0]) |
                            (static_cast<std::uint64_t>(
                                 buffer.descriptor[1] &
                                 UINT32_C(0xffff))
                             << 32u);
                        const auto stride =
                            (buffer.descriptor[1] >> 16u) &
                            UINT32_C(0x3fff);
                        buffer.byte_count =
                            static_cast<std::uint64_t>(
                                buffer.descriptor[2]) *
                            std::max(stride, 1u);
                    }
                    for (std::size_t index = 0;
                         index <
                             program_info.packed_user_data.size() &&
                         index <
                             program_info
                                 .packed_user_data_indices.size();
                         ++index) {
                        const auto source =
                            program_info
                                .packed_user_data_indices[index];
                        if (source < 64u) {
                            program_info.packed_user_data[index] =
                                value(
                                    AgcComputeUserData + source);
                        }
                    }
                }
                if ((initiator & UINT32_C(0x20)) != 0u) {
                    const std::array<std::uint32_t, 3> local_size{
                        std::max(
                            value(AgcComputeNumThreadX), 1u),
                        std::max(
                            value(AgcComputeNumThreadY), 1u),
                        std::max(
                            value(AgcComputeNumThreadZ), 1u)};
                    for (std::uint32_t axis = 0; axis < 3u;
                         ++axis) {
                        dimensions[axis] =
                            dimensions[axis] == 0u ? 0u :
                            (dimensions[axis] +
                             local_size[axis] - 1u) /
                                local_size[axis];
                    }
                }
                bool vulkan_compute{};
                if (compiled && !program_info.buffers.empty() &&
                    program_info.images.empty() &&
                    program_info.addresses.empty()) {
                    vulkan_compute =
                        Lsx4::Ps5Desktop::
                            ExecuteVulkanGen5ComputeBuffers(
                                program_spirv, program_info,
                                dimensions,
                                [](void*,
                                   const std::uint64_t address,
                                   void* const output,
                                   const std::size_t size) {
                                    return TryReadGuestBytes(
                                        address, output, size);
                                },
                                [](void*,
                                   const std::uint64_t address,
                                   const void* const input,
                                   const std::size_t size) {
                                    return TryWriteGuestBytes(
                                        address, input, size);
                                },
                                nullptr);
                    static std::atomic<std::uint32_t>
                        vulkan_compute_logs{};
                    const auto compute_index =
                        vulkan_compute_logs.fetch_add(
                            1u, std::memory_order_relaxed);
                    if (compute_index < 64u) {
                        __android_log_print(
                            vulkan_compute
                                ? ANDROID_LOG_INFO
                                : ANDROID_LOG_WARN,
                            "LSX4-PS5-GEN5",
                            "vulkan buffer dispatch n=%u "
                            "shader=0x%llx groups=%u/%u/%u "
                            "buffers=%zu ok=%u",
                            compute_index + 1u,
                            static_cast<unsigned long long>(
                                shader_address),
                            dimensions[0], dimensions[1],
                            dimensions[2],
                            program_info.buffers.size(),
                            vulkan_compute ? 1u : 0u);
                    }
                }
                bool vulkan_image_compute{};
                if (compiled &&
                    !program_info.images.empty()) {
                    vulkan_image_compute =
                        Lsx4::Ps5Desktop::
                            ExecuteVulkanGen5ComputeImages(
                                program_spirv, program_info,
                                dimensions,
                                [](void*,
                                   const std::uint64_t address,
                                   void* const output,
                                   const std::size_t size) {
                                    return TryReadGuestBytes(
                                        address, output, size);
                                },
                                [](void*,
                                   const std::uint64_t address,
                                   const void* const input,
                                   const std::size_t size) {
                                    return TryWriteGuestBytes(
                                        address, input, size);
                                },
                                nullptr);
                    static std::atomic<std::uint32_t>
                        vulkan_image_logs{};
                    const auto compute_index =
                        vulkan_image_logs.fetch_add(
                            1u, std::memory_order_relaxed);
                    if (compute_index < 64u) {
                        __android_log_print(
                            vulkan_image_compute
                                ? ANDROID_LOG_INFO
                                : ANDROID_LOG_WARN,
                            "LSX4-PS5-GEN5",
                            "vulkan image dispatch n=%u "
                            "shader=0x%llx groups=%u/%u/%u "
                            "images=%zu samplers=%zu ok=%u",
                            compute_index + 1u,
                            static_cast<unsigned long long>(
                                shader_address),
                            dimensions[0], dimensions[1],
                            dimensions[2],
                            program_info.images.size(),
                            program_info.samplers.size(),
                            vulkan_image_compute ? 1u : 0u);
                    }
                }
                if (compiled && !vulkan_compute &&
                    program_info.simple_buffer_copy) {
                    const auto decode_buffer =
                        [&](const std::uint32_t offset,
                            std::uint64_t& address,
                            std::uint64_t& byte_count) {
                            if (offset + 2u >= 64u) {
                                return false;
                            }
                            const auto word0 =
                                value(AgcComputeUserData + offset);
                            const auto word1 =
                                value(AgcComputeUserData + offset + 1u);
                            const auto records =
                                value(AgcComputeUserData + offset + 2u);
                            address =
                                static_cast<std::uint64_t>(word0) |
                                (static_cast<std::uint64_t>(
                                     word1 & UINT32_C(0xffff)) << 32u);
                            const auto stride =
                                (word1 >> 16u) & UINT32_C(0x3fff);
                            byte_count =
                                static_cast<std::uint64_t>(records) *
                                std::max(stride, 1u);
                            return address != 0 && byte_count != 0 &&
                                byte_count <= UINT64_C(0x10000000);
                        };
                    std::uint64_t source_address{};
                    std::uint64_t source_size{};
                    std::uint64_t destination_address{};
                    std::uint64_t destination_size{};
                    if (decode_buffer(
                            program_info.copy_source_user_data,
                            source_address, source_size) &&
                        decode_buffer(
                            program_info.copy_destination_user_data,
                            destination_address,
                            destination_size)) {
                        const auto copy_size = static_cast<std::size_t>(
                            std::min(source_size, destination_size));
                        thread_local std::vector<std::uint8_t>
                            copy_buffer;
                        copy_buffer.resize(copy_size);
                        const auto copied =
                            TryReadGuestBytes(
                                source_address, copy_buffer.data(),
                                copy_size) &&
                            TryWriteGuestBytes(
                                destination_address,
                                copy_buffer.data(), copy_size);
                        static std::atomic<std::uint32_t>
                            copy_logs{};
                        const auto copy_index =
                            copy_logs.fetch_add(
                                1u, std::memory_order_relaxed);
                        if (copy_index < 32u) {
                            __android_log_print(
                                copied ? ANDROID_LOG_INFO
                                       : ANDROID_LOG_WARN,
                                "LSX4-PS5-GEN5",
                                "copy n=%u shader=0x%llx "
                                "source=0x%llx destination=0x%llx "
                                "bytes=%zu ok=%u",
                                copy_index + 1u,
                                static_cast<unsigned long long>(
                                    shader_address),
                                static_cast<unsigned long long>(
                                    source_address),
                                static_cast<unsigned long long>(
                                    destination_address),
                                copy_size, copied ? 1u : 0u);
                        }
                    }
                }
                if (compiled && !vulkan_image_compute &&
                    program_info.simple_image_copy) {
                    const auto& source = program_info.images[
                        program_info.copy_source_image];
                    const auto& destination = program_info.images[
                        program_info.copy_destination_image];
                    const auto bytes_per_pixel =
                        source.format == UINT32_C(0x47) ? 8u :
                        source.format == UINT32_C(0x1) ? 1u : 4u;
                    const auto byte_count64 =
                        static_cast<std::uint64_t>(source.width) *
                        source.height * bytes_per_pixel;
                    if (byte_count64 != 0 &&
                        byte_count64 <= UINT64_C(0x10000000)) {
                        const auto byte_count =
                            static_cast<std::size_t>(byte_count64);
                        thread_local std::vector<std::uint8_t>
                            image_copy_buffer;
                        image_copy_buffer.resize(byte_count);
                        const auto copied =
                            TryReadGuestBytes(
                                source.guest_address,
                                image_copy_buffer.data(), byte_count) &&
                            TryWriteGuestBytes(
                                destination.guest_address,
                                image_copy_buffer.data(), byte_count);
                        static std::atomic<std::uint32_t>
                            image_copy_logs{};
                        const auto copy_index =
                            image_copy_logs.fetch_add(
                                1u, std::memory_order_relaxed);
                        if (copy_index < 32u) {
                            __android_log_print(
                                copied ? ANDROID_LOG_INFO
                                       : ANDROID_LOG_WARN,
                                "LSX4-PS5-GEN5",
                                "image copy n=%u shader=0x%llx "
                                "source=0x%llx destination=0x%llx "
                                "size=%ux%u format=%02x tile=%u "
                                "bytes=%zu ok=%u",
                                copy_index + 1u,
                                static_cast<unsigned long long>(
                                    shader_address),
                                static_cast<unsigned long long>(
                                    source.guest_address),
                                static_cast<unsigned long long>(
                                    destination.guest_address),
                                source.width, source.height,
                                source.format, source.tile_mode,
                                byte_count, copied ? 1u : 0u);
                        }
                    }
                }
                if (log_index < 96u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5-AGC",
                        "dispatch n=%u op=%02x packet=0x%llx "
                        "shader=0x%llx dims=%u/%u/%u init=%08x "
                        "start=%u/%u/%u local=%08x/%08x/%08x "
                        "ud=%08x:%08x:%08x:%08x:"
                        "%08x:%08x:%08x:%08x",
                        log_index + 1u, opcode,
                        static_cast<unsigned long long>(
                            packet_address),
                        static_cast<unsigned long long>(
                            shader_address),
                        dimensions[0], dimensions[1],
                        dimensions[2], initiator,
                        value(AgcComputeStartX),
                        value(AgcComputeStartY),
                        value(AgcComputeStartZ),
                        value(AgcComputeNumThreadX),
                        value(AgcComputeNumThreadY),
                        value(AgcComputeNumThreadZ),
                        value(AgcComputeUserData + 0u),
                        value(AgcComputeUserData + 1u),
                        value(AgcComputeUserData + 2u),
                        value(AgcComputeUserData + 3u),
                        value(AgcComputeUserData + 4u),
                        value(AgcComputeUserData + 5u),
                        value(AgcComputeUserData + 6u),
                        value(AgcComputeUserData + 7u));
                }
            }
        }
#endif
        const bool draw_index_offset =
            opcode == AgcItDrawIndexOffset2 && length >= 5u;
        const bool draw_index_auto =
            opcode == AgcItNop &&
            packet_register == AgcRDrawIndexAuto &&
            length >= 7u;
        if (draw_index_offset || draw_index_auto) {
            ++parsed_draw_count;
            const auto presenter_word0 = sh_registers[0x0c];
            const auto presenter_word1 = sh_registers[0x0d];
            const auto presenter_word2 = sh_registers[0x0e];
            const auto presenter_word3 = sh_registers[0x0f];
            const GuestPresenterTexture presenter{
                .address =
                    ((static_cast<std::uint64_t>(
                          presenter_word1 & 0xffu) << 32u) |
                     presenter_word0) << 8u,
                .width =
                    (((presenter_word1 >> 30u) & 3u) |
                     ((presenter_word2 & 0x3fffu) << 2u)) + 1u,
                .height =
                    ((presenter_word2 >> 14u) & 0xffffu) + 1u,
                .tile_mode =
                    (presenter_word3 >> 20u) & 0x1fu,
                .unified_format =
                    (presenter_word1 >> 20u) & 0x1ffu,
            };
            AgcDiagnosticDraw diagnostic{
                .texture_address = presenter.address,
                .index_buffer_address =
                    draw_index_auto ? 0u : index_buffer_address,
                .texture_width = presenter.width,
                .texture_height = presenter.height,
                .texture_tile_mode = presenter.tile_mode,
                .texture_format = presenter.unified_format,
                .index_buffer_count =
                    draw_index_auto ? 0u : index_buffer_count,
                .index_size = index_size,
                .instance_count = instance_count,
                .primitive_type =
                    uc_registers.contains(UINT32_C(0x242))
                    ? uc_registers[UINT32_C(0x242)]
                    : 0u,
            };
            const auto register_value =
                [](const auto& registers,
                   const std::uint32_t index) {
                    const auto found = registers.find(index);
                    return found != registers.end()
                        ? found->second
                        : 0u;
                };
            diagnostic.cx_target_mask =
                cx_registers.contains(AgcCbTargetMask)
                ? cx_registers.at(AgcCbTargetMask)
                : UINT32_MAX;
            diagnostic.cx_blend0_control =
                register_value(cx_registers, AgcCbBlend0Control);
            diagnostic.cx_color_base =
                register_value(cx_registers, AgcCbColor0Base);
            diagnostic.cx_color_base_ext =
                register_value(cx_registers, AgcCbColor0BaseExt);
            diagnostic.cx_color_info =
                register_value(cx_registers, AgcCbColor0Info);
            diagnostic.cx_color_attrib2 =
                register_value(cx_registers, AgcCbColor0Attrib2);
            diagnostic.cx_color_attrib3 =
                register_value(cx_registers, AgcCbColor0Attrib3);
            diagnostic.sh_ps_program_lo =
                register_value(sh_registers, 0x08);
            diagnostic.sh_ps_program_hi =
                register_value(sh_registers, 0x09);
            diagnostic.sh_es_program_lo =
                register_value(sh_registers, 0xc8);
            diagnostic.sh_es_program_hi =
                register_value(sh_registers, 0xc9);
#ifdef __ANDROID__
            static std::atomic<bool> dumped_first_context_registers{};
            if (!dumped_first_context_registers.exchange(
                    true, std::memory_order_relaxed)) {
                std::vector<std::pair<
                    std::uint32_t, std::uint32_t>> sorted_context(
                    cx_registers.begin(), cx_registers.end());
                std::ranges::sort(sorted_context);
                for (std::size_t start = 0;
                     start < sorted_context.size();
                     start += 6u) {
                    char line[512]{};
                    auto used = std::snprintf(
                        line, sizeof(line), "cx_registers");
                    for (std::size_t index = start;
                         index < std::min(
                             start + 6u,
                             sorted_context.size()) &&
                         used > 0 &&
                         static_cast<std::size_t>(used) <
                             sizeof(line);
                         ++index) {
                        used += std::snprintf(
                            line + used,
                            sizeof(line) -
                                static_cast<std::size_t>(used),
                            " %04x=%08x",
                            sorted_context[index].first,
                            sorted_context[index].second);
                    }
                    __android_log_write(
                        ANDROID_LOG_INFO, "LSX4-PS5", line);
                }
            }
#endif
            for (std::uint32_t word = 0; word < 16u; ++word) {
                diagnostic.sh_ps_user_data[word] =
                    register_value(
                        sh_registers, 0x0c + word);
                diagnostic.sh_vs_user_data[word] =
                    register_value(
                        sh_registers, 0x4c + word);
                diagnostic.sh_gs_user_data[word] =
                    register_value(
                        sh_registers, 0x8c + word);
                diagnostic.sh_es_user_data[word] =
                    register_value(
                        sh_registers, 0xcc + word);
            }
#ifdef __ANDROID__
            struct CompiledGraphicsStage {
                std::vector<std::uint32_t> spirv;
                Lsx4::Ps5Desktop::Gen5ComputeProgramInfo info;
                bool ok{};
            };
            const auto probe_graphics_stage =
                [&](const std::uint64_t shader_address,
                    const Lsx4::Ps5Desktop::Gen5ShaderStage stage,
                    const std::array<std::uint32_t, 16>& user_data,
                    const std::uint32_t rsrc1,
                    const std::uint32_t rsrc2)
                    -> std::shared_ptr<const CompiledGraphicsStage> {
                    if (shader_address == 0u) {
                        return {};
                    }
                    static std::mutex graphics_probe_mutex;
                    static std::unordered_map<
                        std::uint64_t,
                        std::shared_ptr<CompiledGraphicsStage>>
                        graphics_probe_cache;
                    auto cache_key =
                        shader_address |
                        (stage ==
                                 Lsx4::Ps5Desktop::Gen5ShaderStage::Pixel
                             ? UINT64_C(0x8000000000000000)
                             : 0u);
                    for (const auto word : user_data) {
                        cache_key ^=
                            static_cast<std::uint64_t>(word) +
                            UINT64_C(0x9e3779b97f4a7c15) +
                            (cache_key << 6u) +
                            (cache_key >> 2u);
                    }
                    cache_key ^=
                        static_cast<std::uint64_t>(rsrc1) << 1u;
                    cache_key ^=
                        static_cast<std::uint64_t>(rsrc2) << 33u;
                    if (stage ==
                        Lsx4::Ps5Desktop::Gen5ShaderStage::Pixel) {
                        constexpr std::array<std::uint32_t, 6>
                            pixel_state_registers{
                                AgcSpiPsInputEna,
                                AgcSpiPsInputAddr,
                                AgcSpiPsInControl,
                                AgcSpiShaderColFormat,
                                AgcDbShaderControl,
                                AgcCbTargetMask};
                        for (const auto reg :
                             pixel_state_registers) {
                            cache_key ^=
                                static_cast<std::uint64_t>(
                                    register_value(
                                        cx_registers, reg)) +
                                UINT64_C(0x9e3779b97f4a7c15) +
                                (cache_key << 6u) +
                                (cache_key >> 2u);
                        }
                        for (std::uint32_t input = 0;
                             input < 32u; ++input) {
                            cache_key ^=
                                static_cast<std::uint64_t>(
                                    register_value(
                                        cx_registers,
                                        AgcSpiPsInputCntl0 +
                                            input)) +
                                UINT64_C(0x9e3779b97f4a7c15) +
                                (cache_key << 6u) +
                                (cache_key >> 2u);
                        }
                    }
                    const std::lock_guard probe_lock{
                        graphics_probe_mutex};
                    if (const auto found =
                            graphics_probe_cache.find(cache_key);
                        found != graphics_probe_cache.end()) {
                        return found->second;
                    }
                    if (graphics_probe_cache.size() >= 256u) {
                        return {};
                    }
                    std::array<std::uint32_t, 4096> shader_code{};
                    if (!TryReadGuestBytes(
                            shader_address, shader_code.data(),
                            sizeof(shader_code))) {
                        return {};
                    }
                    Lsx4::Ps5Desktop::Gen5ComputeCompileRequest
                        request{};
                    request.shader_address = shader_address;
                    request.code = shader_code;
                    request.stage = stage;
                    request.wave_size =
                        (rsrc1 & UINT32_C(0x40000000)) != 0u
                        ? 32u : 64u;
                    request.user_data_base =
                        stage ==
                                Lsx4::Ps5Desktop::Gen5ShaderStage::Vertex
                        ? 8u : 0u;
                    request.descriptor_set =
                        stage ==
                                Lsx4::Ps5Desktop::Gen5ShaderStage::Pixel
                        ? 1u : 0u;
                    request.user_data_count =
                        ((rsrc2 >> 1u) & UINT32_C(0x1f)) |
                        (((rsrc2 >> 27u) & 1u) << 5u);
                    std::copy(
                        user_data.begin(), user_data.end(),
                        request.user_data.begin());
                    if (stage ==
                        Lsx4::Ps5Desktop::Gen5ShaderStage::Pixel) {
                        request.pixel_input_enable =
                            register_value(
                                cx_registers,
                                AgcSpiPsInputEna);
                        request.pixel_input_address =
                            register_value(
                                cx_registers,
                                AgcSpiPsInputAddr);
                        request.pixel_input_count =
                            register_value(
                                cx_registers,
                                AgcSpiPsInControl) &
                            UINT32_C(0x3f);
                        const auto shader_color_format =
                            register_value(
                                cx_registers,
                                AgcSpiShaderColFormat);
                        for (std::uint32_t slot = 0;
                             slot < 8u; ++slot) {
                            request.pixel_target_output_mode[slot] =
                                static_cast<std::uint8_t>(
                                    (shader_color_format >>
                                     (slot * 4u)) &
                                    0xfu);
                            request.pixel_interpolator_settings[slot] =
                                register_value(
                                    cx_registers,
                                    AgcSpiPsInputCntl0 +
                                        slot);
                        }
                        for (std::uint32_t input = 8u;
                             input < 32u; ++input) {
                            request.pixel_interpolator_settings[input] =
                                register_value(
                                    cx_registers,
                                    AgcSpiPsInputCntl0 +
                                        input);
                        }
                        const auto target_mask =
                            register_value(
                                cx_registers,
                                AgcCbTargetMask);
                        request.pixel_mrt_output_mask = 0u;
                        for (std::uint32_t slot = 0;
                             slot < 8u; ++slot) {
                            if (((target_mask >>
                                  (slot * 4u)) &
                                 0xfu) != 0u) {
                                request.pixel_mrt_output_mask |=
                                    1u << slot;
                            }
                            const auto color_info =
                                register_value(
                                    cx_registers,
                                    AgcCbColor0Info +
                                        slot *
                                            AgcCbColorRegisterStride);
                            const auto layout =
                                (color_info >> 2u) & 0x1fu;
                            const auto number_type =
                                (color_info >> 8u) & 0x7u;
                            const auto channel_order =
                                (color_info >> 11u) & 0x3u;
                            if (layout == 12u &&
                                number_type == 7u) {
                                if (channel_order == 1u) {
                                    request
                                        .pixel_target_export_mapping
                                            [slot] = 0xc6u;
                                } else if (
                                    channel_order == 2u) {
                                    request
                                        .pixel_target_export_mapping
                                            [slot] = 0x1bu;
                                }
                            }
                        }
                        const auto db_shader_control =
                            register_value(
                                cx_registers,
                                AgcDbShaderControl);
                        request.pixel_depth_export_enable =
                            (db_shader_control & 1u) != 0u;
                        const auto z_behavior =
                            (db_shader_control >> 4u) & 3u;
                        request.pixel_kill_enable =
                            (db_shader_control &
                             UINT32_C(0x40)) != 0u;
                        request.pixel_sample_mask_export_enable =
                            (db_shader_control &
                             UINT32_C(0x100)) != 0u;
                        request.pixel_execute_on_noop =
                            (db_shader_control &
                             UINT32_C(0x400)) != 0u;
                        request.pixel_early_z =
                            z_behavior == 1u &&
                            !request.pixel_kill_enable &&
                            !request.pixel_depth_export_enable &&
                            !request.pixel_sample_mask_export_enable;
                    }
                    request.read_memory =
                        [](void*, const std::uint64_t address,
                           std::uint32_t* const output) {
                            return output != nullptr &&
                                TryReadGuestValue(address, *output);
                        };
                    auto compiled =
                        std::make_shared<CompiledGraphicsStage>();
                    std::string error;
                    compiled->ok =
                        Lsx4::Ps5Desktop::TryCompileGen5Compute(
                            request, compiled->spirv,
                            &compiled->info, error);
                    graphics_probe_cache.emplace(
                        cache_key, compiled);
                    if (compiled->ok) {
                        const std::filesystem::path dump_directory{
                            "/data/user/0/app.lsx4.android/files/"
                            "lsx4-home/ps5-shader-dumps"};
                        std::error_code dump_error;
                        std::filesystem::create_directories(
                            dump_directory, dump_error);
                        const auto stem =
                            std::string{
                                stage ==
                                        Lsx4::Ps5Desktop::
                                            Gen5ShaderStage::Vertex
                                    ? "vs-" : "ps-"} +
                            std::to_string(shader_address);
                        if (auto* const stream = std::fopen(
                                (dump_directory /
                                 (stem + ".spv")).string().c_str(),
                                "wb");
                            stream != nullptr) {
                            (void)std::fwrite(
                                compiled->spirv.data(),
                                sizeof(std::uint32_t),
                                compiled->spirv.size(), stream);
                            std::fclose(stream);
                        }
                        if (auto* const stream = std::fopen(
                                (dump_directory /
                                 (stem + ".bin")).string().c_str(),
                                "wb");
                            stream != nullptr) {
                            (void)std::fwrite(
                                shader_code.data(),
                                sizeof(std::uint32_t),
                                shader_code.size(), stream);
                            std::fclose(stream);
                        }
                    }
                    __android_log_print(
                        compiled->ok
                            ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                        "LSX4-PS5-GFX",
                        "stage=%s shader=0x%llx ok=%u words=%zu "
                        "buffers=%zu images=%zu samplers=%zu "
                        "addresses=%zu in=%08x out=%08x mrt=%08x "
                        "frag=%u psabi=%08x/%08x/%u error=%s",
                        stage ==
                                Lsx4::Ps5Desktop::Gen5ShaderStage::Vertex
                            ? "vs" : "ps",
                        static_cast<unsigned long long>(
                            shader_address),
                        compiled->ok ? 1u : 0u,
                        compiled->spirv.size(),
                        compiled->info.buffers.size(),
                        compiled->info.images.size(),
                        compiled->info.samplers.size(),
                        compiled->info.addresses.size(),
                        compiled->info.parameter_input_mask,
                        compiled->info.parameter_output_mask,
                        compiled->info.mrt_output_mask,
                        compiled->info.uses_fragment_coordinates
                            ? 1u : 0u,
                        request.pixel_input_enable,
                        request.pixel_input_address,
                        request.pixel_input_count,
                        error.c_str());
                    if (compiled->ok) {
                        for (std::size_t image_index = 0;
                             image_index <
                                 compiled->info.images.size();
                             ++image_index) {
                            const auto& image =
                                compiled->info.images[image_index];
                            __android_log_print(
                                ANDROID_LOG_INFO,
                                "LSX4-PS5-GFX",
                                "resource shader=0x%llx i=%zu "
                                "addr=0x%llx size=%ux%ux%u "
                                "type=%u fmt=%u tile=%u rw=%u/%u/%u",
                                static_cast<unsigned long long>(
                                    shader_address),
                                image_index,
                                static_cast<unsigned long long>(
                                    image.guest_address),
                                image.width, image.height,
                                image.depth, image.type,
                                image.format, image.tile_mode,
                                image.read ? 1u : 0u,
                                image.written ? 1u : 0u,
                                image.atomic ? 1u : 0u);
                        }
                    }
                    return compiled;
                };
            const auto vertex_address =
                (static_cast<std::uint64_t>(
                     diagnostic.sh_es_program_hi & 0xffu) << 40u) |
                (static_cast<std::uint64_t>(
                     diagnostic.sh_es_program_lo) << 8u);
            const auto pixel_address =
                (static_cast<std::uint64_t>(
                     diagnostic.sh_ps_program_hi & 0xffu) << 40u) |
                (static_cast<std::uint64_t>(
                     diagnostic.sh_ps_program_lo) << 8u);
            std::shared_ptr<const CompiledGraphicsStage>
                compiled_vertex;
            std::shared_ptr<const CompiledGraphicsStage>
                compiled_pixel;
#endif
            const auto has_resource2 =
                [&](const std::uint32_t user_data_register) {
                    return sh_registers.contains(
                        user_data_register - 1u);
                };
            const auto has_user_data =
                [&](const std::uint32_t user_data_register) {
                    for (std::uint32_t word = 0;
                         word < 16u; ++word) {
                        if (sh_registers.contains(
                                user_data_register + word)) {
                            return true;
                        }
                    }
                    return false;
                };
            const std::array<std::uint32_t, 3>
                export_user_data_candidates{
                    0x8cu, 0xccu, 0x4cu};
            auto export_user_data_register = 0xccu;
            bool selected_export_bank{};
            for (const auto candidate :
                 export_user_data_candidates) {
                if (has_resource2(candidate)) {
                    export_user_data_register = candidate;
                    selected_export_bank = true;
                    break;
                }
            }
            if (!selected_export_bank) {
                for (const auto candidate :
                     export_user_data_candidates) {
                    if (has_user_data(candidate)) {
                        export_user_data_register = candidate;
                        break;
                    }
                }
            }
            for (std::uint32_t word = 0; word < 16u; ++word) {
                diagnostic.sh_export_user_data[word] =
                    register_value(
                        sh_registers,
                        export_user_data_register + word);
            }
            if (draw_index_auto) {
                diagnostic.index_offset = 0;
                (void)TryReadGuestValue(
                    packet_address + 4u,
                    diagnostic.draw_count);
            } else {
                (void)TryReadGuestValue(
                    packet_address + 8u,
                    diagnostic.index_offset);
                (void)TryReadGuestValue(
                    packet_address + 12u,
                    diagnostic.draw_count);
            }
            const auto target_mask =
                cx_registers.contains(AgcCbTargetMask)
                ? cx_registers[AgcCbTargetMask]
                : UINT32_MAX;
            for (std::uint32_t slot = 0; slot < 8u; ++slot) {
                const auto base_register =
                    AgcCbColor0Base +
                    slot * AgcCbColorRegisterStride;
                const auto info_register =
                    AgcCbColor0Info +
                    slot * AgcCbColorRegisterStride;
                if (!cx_registers.contains(base_register) ||
                    !cx_registers.contains(
                        AgcCbColor0BaseExt + slot) ||
                    !cx_registers.contains(
                        AgcCbColor0Attrib2 + slot) ||
                    !cx_registers.contains(
                        AgcCbColor0Attrib3 + slot) ||
                    !cx_registers.contains(info_register) ||
                    ((target_mask >> (slot * 4u)) & 0xfu) == 0) {
                    continue;
                }
                const auto base_low =
                    cx_registers[base_register];
                const auto base_high =
                    cx_registers[AgcCbColor0BaseExt + slot];
                const auto attrib2 =
                    cx_registers[AgcCbColor0Attrib2 + slot];
                const auto attrib3 =
                    cx_registers[AgcCbColor0Attrib3 + slot];
                const auto info = cx_registers[info_register];
                const auto target_address =
                    (static_cast<std::uint64_t>(
                         base_high & 0xffu) << 40u) |
                    (static_cast<std::uint64_t>(
                         base_low) << 8u);
                if (target_address == 0) {
                    continue;
                }
                diagnostic.render_target_address =
                    target_address;
                diagnostic.render_target_slot = slot;
                diagnostic.render_target_width =
                    ((attrib2 >> 14u) & 0x3fffu) + 1u;
                diagnostic.render_target_height =
                    (attrib2 & 0x3fffu) + 1u;
                diagnostic.render_target_format =
                    (info >> 2u) & 0x1fu;
                diagnostic.render_target_number_type =
                    (info >> 8u) & 0x7u;
                diagnostic.render_target_channel_order =
                    (info >> 11u) & 0x3u;
                diagnostic.render_target_tile_mode =
                    (attrib3 >> 14u) & 0x1fu;
                diagnostic.cx_blend0_control =
                    register_value(
                        cx_registers,
                        AgcCbBlend0Control + slot);
                break;
            }
            if (UseFullGen5Graphics()) {
                (void)TryResolveAgcTextureDescriptor(
                    diagnostic, diagnostic_draws);
            }
            diagnostic_draws.push_back(diagnostic);
            const bool prefer_cpu_fast_path =
                is_cpu_draw_candidate(diagnostic);
            const bool enable_gen5_graphics =
                UseFullGen5Graphics() &&
                gen5_only_shader_pairs.contains(
                    gen5_shader_pair_key(diagnostic));
            bool vulkan_graphics{};
#ifdef __ANDROID__
            if (!prefer_cpu_fast_path &&
                enable_gen5_graphics) {
                compiled_vertex = probe_graphics_stage(
                    vertex_address,
                    Lsx4::Ps5Desktop::Gen5ShaderStage::Vertex,
                    diagnostic.sh_gs_user_data,
                    register_value(sh_registers, 0x8a),
                    register_value(sh_registers, 0x8b));
                compiled_pixel = probe_graphics_stage(
                    pixel_address,
                    Lsx4::Ps5Desktop::Gen5ShaderStage::Pixel,
                    diagnostic.sh_ps_user_data,
                    register_value(sh_registers, 0x0a),
                    register_value(sh_registers, 0x0b));
            }
            if (compiled_vertex && compiled_vertex->ok &&
                compiled_pixel && compiled_pixel->ok &&
                diagnostic.draw_count != 0u) {
                const auto index_bytes =
                    diagnostic.index_size == 4u ? 4u : 2u;
                const Lsx4::Ps5Desktop::VulkanGen5GraphicsDraw
                    graphics_draw{
                        .target_address =
                            diagnostic.render_target_address,
                        .index_address =
                            diagnostic.index_buffer_address +
                            static_cast<std::uint64_t>(
                                diagnostic.index_offset) *
                                index_bytes,
                        .target_width =
                            diagnostic.render_target_width,
                        .target_height =
                            diagnostic.render_target_height,
                        .target_tile_mode =
                            diagnostic.render_target_tile_mode,
                        .target_format =
                            diagnostic.render_target_format,
                        .target_number_type =
                            diagnostic.render_target_number_type,
                        .target_channel_order =
                            diagnostic.render_target_channel_order,
                        .blend_control =
                            diagnostic.cx_blend0_control,
                        .color_write_mask =
                            (diagnostic.cx_target_mask >>
                             (diagnostic.render_target_slot * 4u)) &
                            0xfu,
                        .primitive_type =
                            diagnostic.primitive_type,
                        .vertex_count = diagnostic.draw_count,
                        .instance_count =
                            diagnostic.instance_count,
                        .index_size = index_bytes,
                        .indexed = !draw_index_auto,
                    };
                vulkan_graphics =
                    Lsx4::Ps5Desktop::ExecuteVulkanGen5Graphics(
                        compiled_vertex->spirv,
                        compiled_vertex->info,
                        compiled_pixel->spirv,
                        compiled_pixel->info,
                        graphics_draw,
                        [](void*, const std::uint64_t address,
                           void* const output,
                           const std::size_t size) {
                            return TryReadGuestBytes(
                                address, output, size);
                        },
                        [](void*, const std::uint64_t address,
                           const void* const input,
                           const std::size_t size) {
                            return TryWriteGuestBytes(
                                address, input, size);
                        },
                        nullptr);
                static std::atomic<std::uint32_t>
                    graphics_execution_logs{};
                const auto execution_index =
                    graphics_execution_logs.fetch_add(
                        1u, std::memory_order_relaxed);
                if (execution_index < 128u) {
                    __android_log_print(
                        vulkan_graphics
                            ? ANDROID_LOG_INFO
                            : ANDROID_LOG_WARN,
                        "LSX4-PS5-GFX",
                        "execute n=%u vs=0x%llx ps=0x%llx "
                        "target=0x%llx %ux%u tile=%u "
                        "primitive=%u count=%u indexed=%u ok=%u",
                        execution_index + 1u,
                        static_cast<unsigned long long>(
                            vertex_address),
                        static_cast<unsigned long long>(
                            pixel_address),
                        static_cast<unsigned long long>(
                            diagnostic.render_target_address),
                        diagnostic.render_target_width,
                        diagnostic.render_target_height,
                        diagnostic.render_target_tile_mode,
                        diagnostic.primitive_type,
                        diagnostic.draw_count,
                        draw_index_auto ? 0u : 1u,
                        vulkan_graphics ? 1u : 0u);
                }
            }
#endif
            if (!vulkan_graphics) {
                (void)TryApplyAgcCpuClearOrCopy(diagnostic);
            }
            bool should_dump_draw_state{};
            bool should_dump_texture{};
            {
                const std::lock_guard lock{g_runtime.mutex};
                if (diagnostic.render_target_address != 0 &&
                    diagnostic.render_target_width <= 8192u &&
                    diagnostic.render_target_height <= 8192u) {
                    g_runtime.presenter_texture = {
                        .address =
                            diagnostic.render_target_address,
                        .width =
                            diagnostic.render_target_width,
                        .height =
                            diagnostic.render_target_height,
                        .tile_mode =
                            diagnostic.render_target_tile_mode,
                        .unified_format =
                            diagnostic.render_target_format,
                    };
                } else if (presenter.address != 0 &&
                           presenter.width <= 8192u &&
                           presenter.height <= 8192u) {
                    g_runtime.presenter_texture = presenter;
                }
                const auto submitted_draw =
                    ++g_runtime.agc_draw_count;
                std::uint64_t texture_dump_after = 1;
                if (const char* const after = std::getenv(
                        "PS5_RUNTIME_TEXTURE_DUMP_AFTER_DRAWS");
                    after != nullptr) {
                    texture_dump_after =
                        std::strtoull(after, nullptr, 0);
                }
                should_dump_draw_state =
                    std::getenv(
                        "PS5_RUNTIME_DRAW_STATE_DUMP_PATH") !=
                        nullptr &&
                    g_runtime.agc_dumped_draw_states == 0;
                if (should_dump_draw_state) {
                    ++g_runtime.agc_dumped_draw_states;
                }
                should_dump_texture =
                    std::getenv(
                        "PS5_RUNTIME_TEXTURE_DUMP_PATH") !=
                        nullptr &&
                    submitted_draw >= texture_dump_after &&
                    g_runtime.agc_dumped_textures == 0;
                if (should_dump_texture) {
                    ++g_runtime.agc_dumped_textures;
                }
            }
            if (should_dump_draw_state) {
                const char* const path = std::getenv(
                    "PS5_RUNTIME_DRAW_STATE_DUMP_PATH");
                FILE* const output =
                    path != nullptr ? std::fopen(path, "wb") : nullptr;
                if (output != nullptr) {
                    std::uint32_t index_offset{};
                    std::uint32_t draw_count{};
                    (void)TryReadGuestValue(
                        packet_address + 8u, index_offset);
                    (void)TryReadGuestValue(
                        packet_address + 12u, draw_count);
                    std::fprintf(
                        output,
                        "draw_command=0x%llx\n"
                        "index_buffer=0x%llx\n"
                        "index_buffer_count=%u\n"
                        "index_size=%u\n"
                        "index_offset=%u\n"
                        "draw_count=%u\n",
                        static_cast<unsigned long long>(
                            packet_address),
                        static_cast<unsigned long long>(
                            index_buffer_address),
                        index_buffer_count, index_size,
                        index_offset, draw_count);
                    const auto dump_registers =
                        [&](const char* const name,
                            const auto& registers) {
                            std::vector<std::pair<
                                std::uint32_t,
                                std::uint32_t>> sorted(
                                registers.begin(),
                                registers.end());
                            std::ranges::sort(sorted);
                            for (const auto& [reg, value] :
                                 sorted) {
                                std::fprintf(
                                    output, "%s[%04x]=%08x\n",
                                    name, reg, value);
                            }
                        };
                    dump_registers("cx", cx_registers);
                    dump_registers("sh", sh_registers);
                    dump_registers("uc", uc_registers);
                    std::fclose(output);
                    std::fprintf(
                        stderr,
                        "PS5_AGC_DRAW_STATE_DUMP path=%s "
                        "cx=%zu sh=%zu uc=%zu count=%u\n",
                        path, cx_registers.size(),
                        sh_registers.size(),
                        uc_registers.size(), draw_count);
                }
            }
            if (should_dump_texture) {
                const auto word0 = sh_registers[0x0c];
                const auto word1 = sh_registers[0x0d];
                const auto word2 = sh_registers[0x0e];
                const auto word3 = sh_registers[0x0f];
                const auto texture_address =
                    ((static_cast<std::uint64_t>(
                          word1 & 0xffu) << 32u) |
                     word0) << 8u;
                const auto texture_width =
                    (((word1 >> 30u) & 3u) |
                     ((word2 & 0x3fffu) << 2u)) + 1u;
                const auto texture_height =
                    ((word2 >> 14u) & 0xffffu) + 1u;
                const auto tile_mode =
                    (word3 >> 20u) & 0x1fu;
                const auto pixel_count =
                    static_cast<std::uint64_t>(
                        texture_width) * texture_height;
                if (texture_address != 0 &&
                    texture_width <= 8192u &&
                    texture_height <= 8192u &&
                    tile_mode == 0 &&
                    pixel_count <=
                        std::numeric_limits<std::size_t>::max() /
                            4u) {
                    std::vector<std::uint8_t> rgba(
                        static_cast<std::size_t>(
                            pixel_count) * 4u);
                    if (TryReadGuestBytes(
                            texture_address, rgba.data(),
                            rgba.size())) {
                        std::uint64_t nonzero{};
                        for (const auto value : rgba) {
                            nonzero += value != 0;
                        }
                        const char* const path = std::getenv(
                            "PS5_RUNTIME_TEXTURE_DUMP_PATH");
                        FILE* const output =
                            path != nullptr
                            ? std::fopen(path, "wb")
                            : nullptr;
                        if (output != nullptr) {
                            std::fprintf(
                                output, "P6\n%u %u\n255\n",
                                texture_width, texture_height);
                            for (std::size_t pixel = 0;
                                 pixel <
                                 static_cast<std::size_t>(
                                     pixel_count);
                                 ++pixel) {
                                const auto source = pixel * 4u;
                                const std::array<std::uint8_t, 3>
                                    rgb{
                                        rgba[source],
                                        rgba[source + 1u],
                                        rgba[source + 2u]};
                                (void)std::fwrite(
                                    rgb.data(), 1, rgb.size(),
                                    output);
                            }
                            std::fclose(output);
                            std::fprintf(
                                stderr,
                                "PS5_AGC_TEXTURE_DUMP path=%s "
                                "address=0x%llx size=%ux%u "
                                "nonzero=%llu\n",
                                path,
                                static_cast<unsigned long long>(
                                    texture_address),
                                texture_width, texture_height,
                                static_cast<unsigned long long>(
                                    nonzero));
                        }
                    }
                }
            }
        }
        if (opcode == AgcItNop &&
            packet_register == AgcRFlip && length >= 6u) {
            std::array<std::uint32_t, 5> flip{};
            if (!TryReadGuestBytes(
                    command_address +
                        static_cast<std::uint64_t>(offset + 1u) * 4u,
                    flip.data(), sizeof(flip))) {
                result = OrbisError(UINT32_C(0x80020101));
                return true;
            }
            diagnostic_flip = flip;
            has_diagnostic_flip = true;
        }
        offset += length;
    }
    cpu_render_decided = std::ranges::any_of(
        diagnostic_draws, is_cpu_draw_candidate);
    if (UseFullGen5Graphics()) {
        if (cpu_render_decided) {
            for (const auto& draw : diagnostic_draws) {
                gen5_only_shader_pairs.erase(
                    gen5_shader_pair_key(draw));
            }
        } else {
            for (const auto& draw : diagnostic_draws) {
                if (draw.sh_es_program_lo != 0u &&
                    draw.sh_ps_program_lo != 0u) {
                    gen5_only_shader_pairs.insert(
                        gen5_shader_pair_key(draw));
                }
            }
        }
    }
    if (cpu_render_decided) {
        const std::lock_guard lock{g_runtime.mutex};
        if (g_runtime.agc_cpu_capture_active) {
            cpu_capture_target =
                g_runtime.agc_cpu_capture_target;
        }
    }
    if (UseVulkanGuestRaster() && cpu_render_decided) {
        const auto display_copy = std::ranges::find_if(
            diagnostic_draws,
            [](const AgcDiagnosticDraw& draw) {
                return draw.draw_count == 4u &&
                    draw.texture_address != 0 &&
                    draw.texture_width ==
                        draw.render_target_width &&
                    draw.texture_height ==
                        draw.render_target_height &&
                    draw.texture_tile_mode ==
                        draw.render_target_tile_mode;
            });
        const auto last_candidate = std::ranges::find_if(
            diagnostic_draws.rbegin(),
            diagnostic_draws.rend(),
            is_cpu_draw_candidate);
        if (display_copy != diagnostic_draws.end()) {
            cpu_capture_target = display_copy->texture_address;
        } else if (cpu_capture_target == 0) {
            if (last_candidate != diagnostic_draws.rend()) {
                cpu_capture_target =
                    last_candidate->render_target_address;
            }
        }
    }
    const auto is_cpu_target_relevant =
        [&](const AgcDiagnosticDraw& draw) {
            return draw.render_target_address ==
                       cpu_capture_target ||
                std::ranges::any_of(
                    diagnostic_draws,
                    [&](const AgcDiagnosticDraw& consumer) {
                        return consumer.texture_address ==
                            draw.render_target_address;
                    });
        };
    cpu_render_allowed =
        cpu_capture_target != 0 &&
        std::ranges::any_of(
            diagnostic_draws,
            [&](const AgcDiagnosticDraw& draw) {
                return is_cpu_draw_candidate(draw) &&
                    is_cpu_target_relevant(draw);
            });
#ifdef __ANDROID__
    if (!diagnostic_draws.empty()) {
        static std::atomic<bool> logged_gameplay_draws{};
        static std::atomic<bool> logged_gameplay_multi_draws{};
        auto& log_gate =
            diagnostic_draws.size() > 1u
            ? logged_gameplay_multi_draws
            : logged_gameplay_draws;
        if (!log_gate.exchange(
                true, std::memory_order_relaxed)) {
            for (std::size_t ordinal = 0;
                 ordinal < diagnostic_draws.size(); ++ordinal) {
                const auto& draw = diagnostic_draws[ordinal];
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "gameplay draw=%zu target=0x%llx/%ux%u/t%u "
                    "texture=0x%llx/%ux%u/f%02x/t%u "
                    "es=%08x ps=%08x indices=%u/%u/%u "
                    "instances=%u relevant=%u candidate=%u",
                    ordinal,
                    static_cast<unsigned long long>(
                        draw.render_target_address),
                    draw.render_target_width,
                    draw.render_target_height,
                    draw.render_target_tile_mode,
                    static_cast<unsigned long long>(
                        draw.texture_address),
                    draw.texture_width, draw.texture_height,
                    draw.texture_format, draw.texture_tile_mode,
                    draw.sh_es_program_lo, draw.sh_ps_program_lo,
                    draw.index_buffer_count, draw.index_offset,
                    draw.draw_count, draw.instance_count,
                    is_cpu_target_relevant(draw) ? 1u : 0u,
                    is_cpu_draw_candidate(draw) ? 1u : 0u);
            }
        }
        if (diagnostic_draws.size() >= 8u) {
            static std::atomic<bool> logged_large_ps5_batch{};
            if (!logged_large_ps5_batch.exchange(
                    true, std::memory_order_relaxed)) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "large draw batch count=%zu capture=0x%llx "
                    "decided=%u allowed=%u",
                    diagnostic_draws.size(),
                    static_cast<unsigned long long>(
                        cpu_capture_target),
                    cpu_render_decided ? 1u : 0u,
                    cpu_render_allowed ? 1u : 0u);
                for (std::size_t ordinal = 0;
                     ordinal < diagnostic_draws.size(); ++ordinal) {
                    const auto& draw = diagnostic_draws[ordinal];
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "large draw=%zu target=0x%llx/%ux%u/f%02x/t%u "
                        "texture=0x%llx/%ux%u/f%02x/t%u "
                        "prim=%u es=%08x/%08x ps=%08x/%08x "
                        "count=%u candidate=%u relevant=%u",
                        ordinal,
                        static_cast<unsigned long long>(
                            draw.render_target_address),
                        draw.render_target_width,
                        draw.render_target_height,
                        draw.render_target_format,
                        draw.render_target_tile_mode,
                        static_cast<unsigned long long>(
                            draw.texture_address),
                        draw.texture_width, draw.texture_height,
                        draw.texture_format, draw.texture_tile_mode,
                        draw.primitive_type,
                        draw.sh_es_program_lo,
                        draw.sh_es_program_hi,
                        draw.sh_ps_program_lo,
                        draw.sh_ps_program_hi,
                        draw.draw_count,
                        is_cpu_draw_candidate(draw) ? 1u : 0u,
                        is_cpu_target_relevant(draw) ? 1u : 0u);
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "large draw=%zu psud="
                        "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x "
                        "index=0x%llx/%u",
                        ordinal,
                        draw.sh_ps_user_data[0],
                        draw.sh_ps_user_data[1],
                        draw.sh_ps_user_data[2],
                        draw.sh_ps_user_data[3],
                        draw.sh_ps_user_data[4],
                        draw.sh_ps_user_data[5],
                        draw.sh_ps_user_data[6],
                        draw.sh_ps_user_data[7],
                        static_cast<unsigned long long>(
                            draw.index_buffer_address),
                        draw.index_buffer_count);
                    std::array<std::uint32_t, 16> ps_root{};
                    const auto ps_root_address =
                        static_cast<std::uint64_t>(
                            draw.sh_ps_user_data[0]) |
                        (static_cast<std::uint64_t>(
                             draw.sh_ps_user_data[1] &
                             UINT32_C(0xffff))
                         << 32u);
                    const auto ps_root_read =
                        TryReadGuestBytes(
                            ps_root_address, ps_root.data(),
                            sizeof(ps_root));
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "large draw=%zu psroot=0x%llx read=%u "
                        "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,"
                        "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                        ordinal,
                        static_cast<unsigned long long>(
                            ps_root_address),
                        ps_root_read ? 1u : 0u,
                        ps_root[0], ps_root[1], ps_root[2],
                        ps_root[3], ps_root[4], ps_root[5],
                        ps_root[6], ps_root[7], ps_root[8],
                        ps_root[9], ps_root[10], ps_root[11],
                        ps_root[12], ps_root[13], ps_root[14],
                        ps_root[15]);
                }
            }
        }
    }
#endif
    const auto cpu_render_started =
        std::chrono::steady_clock::now();
    if (cpu_render_allowed) {
        std::vector<Lsx4::Ps5Desktop::VulkanGuestPass>
            gpu_command_passes;
        std::erase_if(
            cpu_texture_cache,
            [&](const AgcCpuDecodedTexture& texture) {
                return std::ranges::any_of(
                    diagnostic_draws,
                    [&](const AgcDiagnosticDraw& draw) {
                        return texture.address ==
                                   draw.render_target_address ||
                            (texture.address ==
                                 draw.texture_address &&
                             draw.texture_width ==
                                 draw.render_target_width &&
                             draw.texture_height ==
                                 draw.render_target_height);
                    });
            });
        const bool has_splash_overlay = std::ranges::any_of(
            diagnostic_draws,
            [](const AgcDiagnosticDraw& draw) {
                return draw.texture_width == 1920u &&
                    draw.texture_height == 1080u &&
                    draw.texture_format == UINT32_C(0x38) &&
                    (draw.sh_es_program_lo & UINT32_C(0xfff)) ==
                        UINT32_C(0xbaa);
            });
        const bool has_game_maker_pass = std::ranges::any_of(
            diagnostic_draws,
            [](const AgcDiagnosticDraw& draw) {
                const auto suffix =
                    draw.sh_ps_program_lo & UINT32_C(0xfff);
                return suffix == UINT32_C(0xf13) ||
                    suffix == UINT32_C(0x311) ||
                    suffix == UINT32_C(0x459);
            });
#ifdef __ANDROID__
        static std::atomic<bool> observed_game_maker_pass{};
        static std::atomic<bool> observed_full_menu_pass{};
        if (has_game_maker_pass) {
            observed_game_maker_pass.store(
                true, std::memory_order_relaxed);
            if (diagnostic_draws.size() == 17u) {
                observed_full_menu_pass.store(
                    true, std::memory_order_relaxed);
            }
        }
        if (observed_full_menu_pass.load(
                std::memory_order_relaxed) &&
            diagnostic_draws.size() == 15u) {
            static std::atomic<bool> logged_gameplay_pass{};
            if (!logged_gameplay_pass.exchange(
                    true, std::memory_order_relaxed)) {
                for (std::size_t ordinal = 0;
                     ordinal < diagnostic_draws.size(); ++ordinal) {
                    const auto& draw = diagnostic_draws[ordinal];
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "gm gameplay draw=%zu target=0x%llx/%ux%u/t%u "
                        "texture=0x%llx/%ux%u/f%02x/t%u "
                        "es=%03x ps=%03x count=%u index=0x%llx",
                        ordinal,
                        static_cast<unsigned long long>(
                            draw.render_target_address),
                        draw.render_target_width,
                        draw.render_target_height,
                        draw.render_target_tile_mode,
                        static_cast<unsigned long long>(
                            draw.texture_address),
                        draw.texture_width, draw.texture_height,
                        draw.texture_format, draw.texture_tile_mode,
                        draw.sh_es_program_lo & UINT32_C(0xfff),
                        draw.sh_ps_program_lo & UINT32_C(0xfff),
                        draw.draw_count,
                        static_cast<unsigned long long>(
                            draw.index_buffer_address));
                }
            }
        }
        if (observed_game_maker_pass.load(
                std::memory_order_relaxed) &&
            (!has_game_maker_pass ||
             (observed_full_menu_pass.load(
                  std::memory_order_relaxed) &&
              diagnostic_draws.size() != 17u))) {
            static std::atomic<std::uint32_t> post_menu_draw_logs{};
            for (const auto& draw : diagnostic_draws) {
                if (post_menu_draw_logs.fetch_add(
                        1u, std::memory_order_relaxed) >= 60u) {
                    break;
                }
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "post-menu draw target=0x%llx/%ux%u/t%u "
                    "texture=0x%llx/%ux%u/f%02x/t%u "
                    "es=%03x ps=%03x count=%u index=0x%llx",
                    static_cast<unsigned long long>(
                        draw.render_target_address),
                    draw.render_target_width,
                    draw.render_target_height,
                    draw.render_target_tile_mode,
                    static_cast<unsigned long long>(
                        draw.texture_address),
                    draw.texture_width, draw.texture_height,
                    draw.texture_format, draw.texture_tile_mode,
                    draw.sh_es_program_lo & UINT32_C(0xfff),
                    draw.sh_ps_program_lo & UINT32_C(0xfff),
                    draw.draw_count,
                    static_cast<unsigned long long>(
                        draw.index_buffer_address));
            }
        }
#endif
        for (std::size_t ordinal = 0;
             ordinal < diagnostic_draws.size(); ++ordinal) {
            const auto& draw = diagnostic_draws[ordinal];
            const auto is_surface_copy =
                draw.texture_address != 0u &&
                draw.render_target_address != 0u &&
                draw.texture_address !=
                    draw.render_target_address &&
                (draw.draw_count == 4u ||
                 draw.draw_count == 6u) &&
                draw.texture_width >= 16u &&
                draw.texture_height >= 16u &&
                (draw.texture_width ==
                     draw.render_target_width ||
                 draw.render_target_width %
                         draw.texture_width ==
                     0u) &&
                (draw.texture_height ==
                     draw.render_target_height ||
                 draw.render_target_height %
                          draw.texture_height ==
                      0u);
            if ((!is_cpu_draw_candidate(draw) &&
                 !(has_game_maker_pass && is_surface_copy)) ||
                !is_cpu_target_relevant(draw)) {
                continue;
            }
            if (cpu_frame == nullptr ||
                cpu_frame_target != draw.render_target_address) {
                if (cpu_frame != nullptr &&
                    cpu_frame_target != 0 &&
                    HasAgcFrameContent(*cpu_frame)) {
                    cpu_command_surfaces[cpu_frame_target] =
                        cpu_frame;
                }
                auto accumulated =
                    std::make_shared<AgcCpuFrame>();
                const auto command_surface =
                    cpu_command_surfaces.find(
                        draw.render_target_address);
                if (command_surface !=
                        cpu_command_surfaces.end() &&
                    command_surface->second != nullptr &&
                    HasAgcFrameContent(*command_surface->second)) {
                    *accumulated = *command_surface->second;
                } else if (!has_game_maker_pass) {
                    const std::lock_guard lock{g_runtime.mutex};
                    auto existing =
                        g_runtime.agc_cpu_working_surfaces.find(
                            draw.render_target_address);
                    if (existing !=
                            g_runtime.agc_cpu_working_surfaces.end() &&
                        existing->second != nullptr &&
                        HasAgcFrameContent(*existing->second)) {
                        *accumulated = *existing->second;
                    } else {
                        existing = g_runtime.agc_cpu_surfaces.find(
                            draw.render_target_address);
                        if (existing !=
                                g_runtime.agc_cpu_surfaces.end() &&
                            existing->second != nullptr &&
                            HasAgcFrameContent(*existing->second)) {
                            *accumulated = *existing->second;
                        }
                    }
                }
                accumulated->rendered_draws = 0;
                accumulated->covered_pixels = 0;
                accumulated->eligible_draw_mask = 0;
                accumulated->rendered_draw_mask = 0;
                accumulated->draw_coverage.fill(0);
                accumulated->gpu_frame =
                    accumulated->gpu_frame != nullptr
                    ? std::make_shared<
                          Lsx4::Ps5Desktop::VulkanGuestFrame>(
                          *accumulated->gpu_frame)
                    : nullptr;
                if (accumulated->gpu_frame != nullptr) {
                    accumulated->gpu_frame->target_key =
                        draw.render_target_address;
                    accumulated->gpu_frame->base_batch_id =
                        accumulated->gpu_frame->batch_id;
                    accumulated->gpu_frame->first_new_draw =
                        accumulated->gpu_frame->draws.size();
                    accumulated->gpu_frame->preserve_target =
                        !accumulated->gpu_frame->draws.empty();
                }
                cpu_frame = std::move(accumulated);
                cpu_frame_target = draw.render_target_address;
            }
            const auto composited = TryCompositeAgcCpuDraw(
                draw, *cpu_frame, ordinal, !has_splash_overlay,
                cpu_texture_cache,
                has_game_maker_pass ? &gpu_command_passes : nullptr,
                has_game_maker_pass && is_surface_copy,
                cpu_texture_probe_serial);
#ifdef __ANDROID__
            if (has_game_maker_pass) {
                static std::atomic<std::uint32_t>
                    game_maker_composite_logs{};
                if (game_maker_composite_logs.fetch_add(
                        1u, std::memory_order_relaxed) < 40u) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5",
                        "gm composite draw=%zu copy=%u ok=%u "
                        "target=0x%llx passes=%zu",
                        ordinal, is_surface_copy ? 1u : 0u,
                        composited ? 1u : 0u,
                        static_cast<unsigned long long>(
                            draw.render_target_address),
                        gpu_command_passes.size());
                }
            }
#endif
            cpu_command_surfaces[cpu_frame_target] = cpu_frame;
            if (cpu_texture_cache.size() >
                AgcCpuTextureCacheLimit) {
                cpu_texture_cache.erase(cpu_texture_cache.begin());
            }
        }
        const auto final_blit = std::ranges::find_if(
            diagnostic_draws,
            [](const AgcDiagnosticDraw& draw) {
                return draw.draw_count == 4u &&
                    draw.texture_address != 0 &&
                    draw.texture_width ==
                        draw.render_target_width &&
                    draw.texture_height ==
                        draw.render_target_height &&
                    draw.texture_tile_mode ==
                    draw.render_target_tile_mode;
            });
        auto selected_final_blit = final_blit;
        if (has_game_maker_pass) {
            const auto display_blit = std::ranges::find_if(
                diagnostic_draws,
                [](const AgcDiagnosticDraw& draw) {
                    return draw.texture_address != 0u &&
                        draw.texture_width == 1280u &&
                        draw.texture_height == 720u &&
                        draw.render_target_width >= 2560u &&
                        draw.render_target_height >= 1440u;
                });
            if (display_blit != diagnostic_draws.end()) {
                selected_final_blit = display_blit;
            }
        }
        std::shared_ptr<AgcCpuFrame> final_blit_frame;
        if (selected_final_blit != diagnostic_draws.end()) {
            const auto final_surface_key =
                selected_final_blit->texture_address;
            const auto command_source =
                cpu_command_surfaces.find(
                    final_surface_key);
            if (command_source != cpu_command_surfaces.end()) {
                final_blit_frame = command_source->second;
            } else {
                const std::lock_guard lock{g_runtime.mutex};
                const auto working_source =
                    g_runtime.agc_cpu_working_surfaces.find(
                        final_surface_key);
                if (working_source !=
                        g_runtime.agc_cpu_working_surfaces.end() &&
                    working_source->second != nullptr &&
                    HasAgcFrameContent(*working_source->second)) {
                    final_blit_frame = working_source->second;
                } else {
                    const auto published_source =
                        g_runtime.agc_cpu_surfaces.find(
                        final_surface_key);
                    if (published_source !=
                            g_runtime.agc_cpu_surfaces.end() &&
                        published_source->second != nullptr &&
                        HasAgcFrameContent(*published_source->second)) {
                        final_blit_frame =
                            published_source->second;
                    }
                }
            }
        }
        if (selected_final_blit != diagnostic_draws.end() &&
            final_blit_frame != nullptr &&
            final_blit_frame->rendered_draws != 0u) {
            if (has_game_maker_pass &&
                final_blit_frame->gpu_frame != nullptr &&
                !gpu_command_passes.empty()) {
                final_blit_frame->gpu_frame =
                    std::make_shared<
                        Lsx4::Ps5Desktop::VulkanGuestFrame>(
                        *final_blit_frame->gpu_frame);
                final_blit_frame->gpu_frame->target_key =
                    selected_final_blit->texture_address;
                final_blit_frame->gpu_frame->passes =
                    std::move(gpu_command_passes);
            }
            bool accepted{};
            std::uint64_t best_coverage{};
            std::uint32_t best_rendered_draws{};
            {
                const std::lock_guard lock{g_runtime.mutex};
                g_runtime.agc_cpu_quality_command_draws =
                    static_cast<std::uint32_t>(
                        diagnostic_draws.size());
                const auto complete =
                    final_blit_frame->eligible_draw_mask != 0u &&
                    final_blit_frame->rendered_draw_mask ==
                        final_blit_frame->eligible_draw_mask;
                if (complete) {
                    g_runtime.agc_cpu_best_rendered_draws =
                        final_blit_frame->rendered_draws;
                    g_runtime.agc_cpu_best_coverage =
                        final_blit_frame->covered_pixels;
                    g_runtime.agc_cpu_surfaces[
                        selected_final_blit->texture_address] =
                            final_blit_frame;
                    g_runtime.agc_cpu_surfaces[
                        selected_final_blit->render_target_address] =
                            final_blit_frame;
                    g_runtime.agc_cpu_frame = final_blit_frame;
                    g_runtime.agc_cpu_capture_active = false;
                    g_runtime.agc_cpu_capture_target = 0;
                    g_runtime.agc_cpu_pending_capture_target = 0;
                    g_runtime.agc_cpu_next_render =
                        std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(16);
                    accepted = true;
                }
                best_rendered_draws =
                    g_runtime.agc_cpu_best_rendered_draws;
                best_coverage =
                    g_runtime.agc_cpu_best_coverage;
            }
#ifdef __ANDROID__
            static std::atomic<std::uint32_t> cpu_quality_logs{};
            if (cpu_quality_logs.fetch_add(
                    1u, std::memory_order_relaxed) < 4u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "cpu frame quality command_draws=%zu "
                    "rendered=%u best_rendered=%u "
                    "eligible=0x%llx success=0x%llx "
                    "coverage=%llu best=%llu accepted=%u",
                    diagnostic_draws.size(),
                    final_blit_frame->rendered_draws,
                    best_rendered_draws,
                    static_cast<unsigned long long>(
                        final_blit_frame->eligible_draw_mask),
                    static_cast<unsigned long long>(
                        final_blit_frame->rendered_draw_mask),
                    static_cast<unsigned long long>(
                        final_blit_frame->covered_pixels),
                    static_cast<unsigned long long>(
                        best_coverage),
                    accepted ? 1u : 0u);
            }
#endif
        }
    }
    const auto cpu_render_finished =
        std::chrono::steady_clock::now();
    for (const auto& write : deferred_data_writes) {
        bool wrote = true;
        for (std::size_t index = 0;
             wrote && index < write.values.size();
             ++index) {
            const auto target =
                write.destination_address +
                (write.increment_address
                     ? static_cast<std::uint64_t>(index) * 4u
                     : 0u);
            wrote = TryWriteGuestValue(
                target, write.values[index]);
        }
        data_write_count += wrote;
        data_write_failures += !wrote;
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> data_write_logs{};
        if (data_write_logs.fetch_add(
                1u, std::memory_order_relaxed) < 16u) {
            __android_log_print(
                wrote ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                "LSX4-PS5",
                "write_data deferred=1 standard=%d dst_sel=%u "
                "dst=0x%llx count=%zu increment=%d wrote=%d",
                write.standard_packet ? 1 : 0,
                write.destination_selection,
                static_cast<unsigned long long>(
                    write.destination_address),
                write.values.size(),
                write.increment_address ? 1 : 0,
                wrote ? 1 : 0);
        }
#endif
    }
    for (const auto& write : deferred_release_writes) {
        const auto timestamp = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now()
                .time_since_epoch()
                .count());
        const bool wrote =
            write.data_selection == 1u
            ? TryWriteGuestValue(
                  write.destination_address,
                  static_cast<std::uint32_t>(write.data))
            : (write.data_selection == 2u
                   ? TryWriteGuestValue(
                         write.destination_address, write.data)
                   : TryWriteGuestValue(
                         write.destination_address, timestamp));
        release_write_count += wrote;
        release_write_failures += !wrote;
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> release_logs{};
        if (release_logs.fetch_add(
                1u, std::memory_order_relaxed) < 32u) {
            __android_log_print(
                wrote ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                "LSX4-PS5",
                "release_mem deferred=1 standard=%d dst_sel=%u "
                "dst=0x%llx data_sel=%u data=0x%llx wrote=%d",
                write.standard_packet ? 1 : 0,
                write.destination_selection,
                static_cast<unsigned long long>(
                    write.destination_address),
                write.data_selection,
                static_cast<unsigned long long>(write.data),
                wrote ? 1 : 0);
        }
#endif
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        for (const auto& [surface_target, surface] :
             cpu_command_surfaces) {
            if (surface_target == 0 ||
                surface == nullptr ||
                !HasAgcFrameContent(*surface) ||
                surface->rendered_draws == 0u ||
                surface->covered_pixels == 0u) {
                continue;
            }
            g_runtime.agc_cpu_working_surfaces[surface_target] =
                surface;
            if (surface->gpu_frame != nullptr) {
                const auto extended_frame =
                    !surface->gpu_frame->passes.empty() ||
                    std::ranges::any_of(
                        surface->gpu_frame->draws,
                        [](const auto& draw) {
                            return draw.extended_blend_contract;
                        });
                if (!extended_frame) {
                    // The original single-pass contract publishes only the
                    // completed final blit below. Publishing its intermediate
                    // display surfaces exposes alternating partial frames.
                    continue;
                }
                const auto incoming_complete =
                    surface->eligible_draw_mask != 0u &&
                    surface->rendered_draw_mask ==
                        surface->eligible_draw_mask;
                const auto existing =
                    g_runtime.agc_cpu_surfaces.find(surface_target);
                const auto existing_complete =
                    existing != g_runtime.agc_cpu_surfaces.end() &&
                    existing->second != nullptr &&
                    existing->second->eligible_draw_mask != 0u &&
                    existing->second->rendered_draw_mask ==
                        existing->second->eligible_draw_mask;
                const auto is_display_surface =
                    std::ranges::any_of(
                        g_runtime.guest_video_out_ports,
                        [&](const auto& entry) {
                            return std::ranges::find(
                                       entry.second.buffer_addresses,
                                       surface_target) !=
                                entry.second.buffer_addresses.end();
                        });
                if (is_display_surface &&
                    (incoming_complete || !existing_complete)) {
                    g_runtime.agc_cpu_surfaces[surface_target] =
                        surface;
                    g_runtime.agc_cpu_frame = surface;
                }
                continue;
            }
            const auto incoming_complete =
                surface->eligible_draw_mask != 0u &&
                surface->rendered_draw_mask ==
                    surface->eligible_draw_mask;
            const auto existing =
                g_runtime.agc_cpu_surfaces.find(surface_target);
            const auto existing_complete =
                existing != g_runtime.agc_cpu_surfaces.end() &&
                existing->second != nullptr &&
                existing->second->eligible_draw_mask != 0u &&
                existing->second->rendered_draw_mask ==
                    existing->second->eligible_draw_mask;
            if (incoming_complete || !existing_complete) {
                g_runtime.agc_cpu_surfaces[surface_target] =
                    surface;
            }
        }
        g_runtime.agc_cx_registers = cx_registers;
        g_runtime.agc_sh_registers = sh_registers;
        g_runtime.agc_uc_registers = uc_registers;
        g_runtime.agc_index_buffer_address =
            index_buffer_address;
        g_runtime.agc_index_buffer_count =
            index_buffer_count;
        g_runtime.agc_index_size = index_size;
        g_runtime.agc_instance_count = instance_count;
        g_runtime.agc_dispatch_indirect_args_base =
            dispatch_indirect_args_base;
    }
    if (has_diagnostic_flip) {
        ObserveAgcFlip(
            static_cast<std::int32_t>(diagnostic_flip[0]),
            static_cast<std::int32_t>(diagnostic_flip[1]),
            diagnostic_flip[2],
            static_cast<std::uint64_t>(diagnostic_flip[3]) |
                (static_cast<std::uint64_t>(
                     diagnostic_flip[4]) << 32u));
    }
#ifdef __ANDROID__
    std::uint64_t submit_count{};
    bool log_draw_signature{};
    std::uint64_t draw_signature = UINT64_C(1469598103934665603);
    const auto mix_signature =
        [&](const std::uint64_t value) {
            draw_signature ^= value;
            draw_signature *= UINT64_C(1099511628211);
        };
    mix_signature(dword_count);
    mix_signature(parsed_draw_count);
    for (const auto& draw : diagnostic_draws) {
        mix_signature(
            static_cast<std::uint64_t>(draw.texture_width) << 32u |
            draw.texture_height);
        mix_signature(
            static_cast<std::uint64_t>(draw.texture_format) << 32u |
            draw.texture_tile_mode);
        mix_signature(
            static_cast<std::uint64_t>(
                draw.render_target_width) << 32u |
            draw.render_target_height);
        mix_signature(
            static_cast<std::uint64_t>(
                draw.render_target_format) << 32u |
            draw.render_target_tile_mode);
        mix_signature(
            static_cast<std::uint64_t>(
                draw.sh_es_program_hi) << 32u |
            draw.sh_es_program_lo);
        mix_signature(
            static_cast<std::uint64_t>(
                draw.sh_ps_program_hi) << 32u |
            draw.sh_ps_program_lo);
        mix_signature(draw.draw_count);
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        submit_count = ++g_runtime.agc_submit_count;
        log_draw_signature =
            !std::ranges::contains(
                g_runtime.agc_logged_draw_signatures,
                draw_signature) &&
            g_runtime.agc_draw_signature_logs < 4u;
        g_runtime.agc_last_draw_signature = draw_signature;
        if (log_draw_signature) {
            g_runtime.agc_logged_draw_signatures.push_back(
                draw_signature);
            ++g_runtime.agc_draw_signature_logs;
        }
    }
    static std::atomic<std::uint64_t> profile_total_us{};
    static std::atomic<std::uint64_t> profile_render_us{};
    profile_total_us.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() -
                submit_started).count()),
        std::memory_order_relaxed);
    profile_render_us.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                cpu_render_finished -
                cpu_render_started).count()),
        std::memory_order_relaxed);
    if (submit_count % 300u == 0u) {
        const auto total_us = profile_total_us.exchange(
            0u, std::memory_order_relaxed);
        const auto render_us = profile_render_us.exchange(
            0u, std::memory_order_relaxed);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "submit profile count=%llu total_avg_us=%llu "
            "render_avg_us=%llu",
            static_cast<unsigned long long>(submit_count),
            static_cast<unsigned long long>(total_us / 300u),
            static_cast<unsigned long long>(render_us / 300u));
    }
    if (log_draw_signature) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "draw signature=0x%llx submit=%llu dwords=%u draws=%zu "
            "flip=%d/%d mode=%u arg=0x%llx",
            static_cast<unsigned long long>(draw_signature),
            static_cast<unsigned long long>(submit_count),
            dword_count, diagnostic_draws.size(),
            has_diagnostic_flip
                ? static_cast<std::int32_t>(diagnostic_flip[0])
                : -1,
            has_diagnostic_flip
                ? static_cast<std::int32_t>(diagnostic_flip[1])
                : -1,
            has_diagnostic_flip ? diagnostic_flip[2] : 0u,
            static_cast<unsigned long long>(
                has_diagnostic_flip
                    ? static_cast<std::uint64_t>(
                          diagnostic_flip[3]) |
                          (static_cast<std::uint64_t>(
                               diagnostic_flip[4]) << 32u)
                    : 0u));
    }
    if (submit_count <= 8u || submit_count % 300u == 0) {
        std::array<std::pair<std::uint32_t, std::uint32_t>, 256>
            ranked{};
        for (std::uint32_t opcode = 0;
             opcode < ranked.size(); ++opcode) {
            ranked[opcode] = {opcode_counts[opcode], opcode};
        }
        std::ranges::sort(
            ranked, std::greater<>{});
        static std::mutex rate_mutex;
        static std::chrono::steady_clock::time_point rate_mark{};
        static std::uint64_t rate_count{};
        double submit_rate{};
        {
            const auto now = std::chrono::steady_clock::now();
            const std::lock_guard rate_lock{rate_mutex};
            if (rate_mark.time_since_epoch().count() != 0 &&
                submit_count > rate_count) {
                const auto seconds =
                    std::chrono::duration<double>(now - rate_mark).count();
                if (seconds > 0.0) {
                    submit_rate =
                        static_cast<double>(submit_count - rate_count) /
                        seconds;
                }
            }
            rate_mark = now;
            rate_count = submit_count;
        }
        std::size_t mapping_count{};
        std::size_t narrow_count{};
        std::size_t wide_count{};
        std::size_t heap_count{};
        {
            const std::lock_guard state_lock{g_runtime.mutex};
            mapping_count = g_runtime.mappings.Size();
            narrow_count = g_runtime.mappings.NarrowCount();
            wide_count = g_runtime.mappings.WideCount();
            heap_count = g_runtime.libc_heap_allocations.size();
        }
        char summary[512]{};
        auto used = std::snprintf(
            summary, sizeof(summary),
            "submit count=%llu rate=%.1f/s maps=%zu(n%zu/w%zu) heap=%zu "
            "dwords=%u packets=%u draws=%u "
            "cx=%zu sh=%zu uc=%zu ops=",
            static_cast<unsigned long long>(submit_count),
            submit_rate, mapping_count, narrow_count, wide_count,
            heap_count,
            dword_count, parsed_packet_count, parsed_draw_count,
            cx_registers.size(), sh_registers.size(),
            uc_registers.size());
        for (std::size_t index = 0;
             index < 10u && ranked[index].first != 0 &&
             used > 0 &&
             static_cast<std::size_t>(used) < sizeof(summary);
             ++index) {
            used += std::snprintf(
                summary + used, sizeof(summary) -
                    static_cast<std::size_t>(used),
                "%s%02x:%u", index == 0 ? "" : ",",
                ranked[index].second, ranked[index].first);
        }
        if (used > 0 &&
            static_cast<std::size_t>(used) < sizeof(summary)) {
            used += std::snprintf(
                summary + used, sizeof(summary) -
                    static_cast<std::size_t>(used),
                " nop=");
        }
        for (std::size_t reg = 0; reg < nop_register_counts.size() &&
             used > 0 &&
             static_cast<std::size_t>(used) < sizeof(summary);
             ++reg) {
            if (nop_register_counts[reg] == 0) {
                continue;
            }
            used += std::snprintf(
                summary + used, sizeof(summary) -
                    static_cast<std::size_t>(used),
                "%s%02zx:%u",
                std::ranges::count(
                    nop_register_counts.begin(),
                    nop_register_counts.begin() + reg,
                    std::uint32_t{0}) ==
                        static_cast<std::ptrdiff_t>(reg)
                    ? ""
                    : ",",
                reg, nop_register_counts[reg]);
        }
        __android_log_write(
            ANDROID_LOG_INFO, "LSX4-PS5", summary);
    }
#endif
    result = 0;
    return true;
}

bool TryAgcDriverSubmitAcb(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    auto submit = request;
    submit.integer_arguments[0] = request.integer_arguments[1];
    return TryAgcDriverSubmitDcb(submit, result);
}

#ifdef __ANDROID__
bool OpenGuestAudioOut(GuestAudioOutPort& port) {
    AAudioStreamBuilder* builder{};
    auto status = AAudio_createStreamBuilder(&builder);
    if (status != AAUDIO_OK || builder == nullptr) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=builder error=%s\n",
            AAudio_convertResultToText(status));
        return false;
    }
    const auto output_channels = port.channels == 1 ? 1 : 2;
    AAudioStreamBuilder_setDirection(
        builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(
        builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(
        builder, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setSampleRate(
        builder, static_cast<std::int32_t>(port.frequency));
    AAudioStreamBuilder_setChannelCount(
        builder, static_cast<std::int32_t>(output_channels));
    AAudioStreamBuilder_setFormat(
        builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setBufferCapacityInFrames(
        builder,
        static_cast<std::int32_t>(
            std::max<std::uint32_t>(
                port.buffer_length * 4u, 512u)));
    status = AAudioStreamBuilder_openStream(builder, &port.stream);
    AAudioStreamBuilder_delete(builder);
    if (status != AAUDIO_OK || port.stream == nullptr) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=open error=%s\n",
            AAudio_convertResultToText(status));
        port.stream = nullptr;
        return false;
    }
    status = AAudioStream_requestStart(port.stream);
    if (status != AAUDIO_OK) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=start error=%s\n",
            AAudio_convertResultToText(status));
        AAudioStream_close(port.stream);
        port.stream = nullptr;
        return false;
    }
    std::fprintf(
        stderr,
        "PS5_AUDIO_OPEN ok rate=%u frames=%u guest_channels=%u "
        "output_channels=%u format=%d\n",
        port.frequency, port.buffer_length, port.channels,
        output_channels, port.format);
    return true;
}

void CloseGuestAudioOut(GuestAudioOutPort& port) {
    if (port.stream == nullptr) {
        return;
    }
    (void)AAudioStream_requestStop(port.stream);
    (void)AAudioStream_close(port.stream);
    port.stream = nullptr;
}

bool OutputGuestAudio(const GuestAudioOutPort& port,
                      const std::uint64_t source_address,
                      const std::uint64_t output_count) {
    if (port.stream == nullptr || source_address == 0) {
        return false;
    }
    const auto sample_count =
        static_cast<std::uint64_t>(port.buffer_length) *
        port.channels;
    if (sample_count == 0 ||
        sample_count >
            std::numeric_limits<std::size_t>::max() /
                port.bytes_per_sample) {
        return false;
    }
    const auto source_size =
        static_cast<std::size_t>(sample_count) *
        port.bytes_per_sample;
    std::vector<std::uint8_t> source(source_size);
    if (!TryReadGuestBytes(
            source_address, source.data(), source.size())) {
        std::fprintf(
            stderr,
            "PS5_AUDIO_OUTPUT error=guest-read address=0x%llx bytes=%zu\n",
            static_cast<unsigned long long>(source_address),
            source.size());
        return false;
    }

    const auto output_channels = port.channels == 1 ? 1u : 2u;
    std::vector<float> output(
        static_cast<std::size_t>(port.buffer_length) *
        output_channels);
    const auto base_format =
        static_cast<std::uint32_t>(port.format) & 0xffu;
    const bool is_float =
        base_format >= 3u && base_format <= 5u ||
        base_format == 7u;
    const auto sample = [&](const std::size_t index) {
        if (is_float) {
            float value{};
            std::memcpy(
                &value,
                source.data() + index * sizeof(float),
                sizeof(value));
            return std::isfinite(value) ? value : 0.0f;
        }
        std::int16_t value{};
        std::memcpy(
            &value,
            source.data() + index * sizeof(std::int16_t),
            sizeof(value));
        return static_cast<float>(value) / 32768.0f;
    };
    float peak{};
    constexpr float Mix = 0.70710678f;
    for (std::uint32_t frame = 0;
         frame < port.buffer_length; ++frame) {
        const auto source_base =
            static_cast<std::size_t>(frame) * port.channels;
        const auto output_base =
            static_cast<std::size_t>(frame) * output_channels;
        if (port.channels == 8) {
            const auto center = sample(source_base + 2u) * Mix;
            output[output_base] =
                (sample(source_base) + center +
                 (sample(source_base + 4u) +
                  sample(source_base + 6u)) * Mix) *
                port.volume;
            output[output_base + 1u] =
                (sample(source_base + 1u) + center +
                 (sample(source_base + 5u) +
                  sample(source_base + 7u)) * Mix) *
                port.volume;
        } else {
            output[output_base] =
                sample(source_base) * port.volume;
            if (output_channels == 2) {
                output[output_base + 1u] =
                    sample(source_base + 1u) * port.volume;
            }
        }
        for (std::uint32_t channel = 0;
             channel < output_channels; ++channel) {
            auto& value = output[output_base + channel];
            value = std::clamp(value, -1.0f, 1.0f);
            peak = std::max(peak, std::abs(value));
        }
    }
    const auto written = AAudioStream_write(
        port.stream, output.data(),
        static_cast<std::int32_t>(port.buffer_length),
        INT64_C(20000000));
    if (output_count <= 8 || output_count % 200u == 0 ||
        written < 0) {
        std::fprintf(
            stderr,
            "PS5_AUDIO_OUTPUT count=%llu frames=%d/%u "
            "guest_channels=%u output_channels=%u float=%d "
            "peak=%.6f status=%s\n",
            static_cast<unsigned long long>(output_count),
            written >= 0 ? static_cast<int>(written) : 0,
            port.buffer_length, port.channels,
            output_channels, is_float ? 1 : 0, peak,
            written >= 0
                ? "ok"
                : AAudio_convertResultToText(
                      static_cast<aaudio_result_t>(written)));
    }
    return written >= 0;
}
#endif

bool TryAllocateLibcHeap(
    const std::uint64_t requested_size,
    const std::uint64_t requested_alignment,
    std::uint64_t& address,
    const std::uint64_t guest_stack) {
    address = 0;
    auto alignment = std::max<std::uint64_t>(
        requested_alignment, alignof(std::max_align_t));
    if (!IsPowerOfTwo(alignment)) {
        return false;
    }
    const auto payload_size = std::max<std::uint64_t>(
        requested_size, 1);
    if (payload_size >
        std::numeric_limits<std::uint64_t>::max() -
            (alignment - 1)) {
        return false;
    }
    std::uint64_t mapped_size_u64{};
    if (!AlignUp(
            payload_size + alignment - 1,
            LSX4_PS5_GUEST_PAGE_SIZE, mapped_size_u64) ||
        mapped_size_u64 > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const auto mapped_size =
        static_cast<std::size_t>(mapped_size_u64);

    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t first_hint{};
    if (!AlignUp(
            g_runtime.next_libc_heap_hint,
            std::max<std::uint64_t>(
                alignment, LSX4_PS5_GUEST_PAGE_SIZE),
            first_hint)) {
        return false;
    }
    auto* const allocation = AllocateLowGuestRegionLocked(
        first_hint,
        std::max(Ps5LibcHeapStride, mapped_size_u64),
        mapped_size);
    if (allocation == nullptr) {
        return false;
    }
    const auto mapping_base =
        reinterpret_cast<std::uint64_t>(allocation);
    if (!AlignUp(mapping_base, alignment, address) ||
        address < mapping_base ||
        payload_size > mapped_size_u64 - (address - mapping_base) ||
        !RegisterMappingLocked(
            mapping_base, mapped_size_u64,
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE,
            "ps5-libc-heap", true)) {
        munmap(allocation, mapped_size);
        address = 0;
        return false;
    }
    std::array<std::uint64_t, 4> allocation_returns{};
    if (guest_stack != 0 &&
        HasAccessLocked(
            guest_stack, sizeof(allocation_returns),
            LSX4_PS5_GUEST_READ)) {
        std::memcpy(
            allocation_returns.data(),
            reinterpret_cast<const void*>(guest_stack),
            sizeof(allocation_returns));
    }
    g_runtime.libc_heap_allocations.insert_or_assign(
        address,
        LibcHeapAllocation{
            .allocation = allocation,
            .mapped_size = mapped_size,
            .address = address,
            .requested_size = requested_size,
            .alignment = alignment,
            .allocation_returns = allocation_returns,
        });
    g_runtime.next_libc_heap_hint =
        mapping_base + mapped_size_u64;
#ifdef __ANDROID__
    if (mapping_base < Ps5LibcHeapHint + UINT64_C(0x100000)) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-HEAP",
            "alloc base=0x%llx address=0x%llx requested=0x%llx "
            "mapped=0x%zx alignment=0x%llx "
            "returns=0x%llx,0x%llx,0x%llx,0x%llx",
            static_cast<unsigned long long>(mapping_base),
            static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(requested_size),
            mapped_size,
            static_cast<unsigned long long>(alignment),
            static_cast<unsigned long long>(allocation_returns[0]),
            static_cast<unsigned long long>(allocation_returns[1]),
            static_cast<unsigned long long>(allocation_returns[2]),
            static_cast<unsigned long long>(allocation_returns[3]));
    }
#endif
    return true;
}

void FreeLibcHeap(
    const std::uint64_t address,
    const std::uint64_t guest_stack) {
    if (address == 0) {
        return;
    }
    LibcHeapAllocation removed{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = g_runtime.libc_heap_allocations.find(address);
        if (found == g_runtime.libc_heap_allocations.end()) {
            return;
        }
        removed = found->second;
        const auto mapping_base =
            reinterpret_cast<std::uint64_t>(removed.allocation);
#ifdef __ANDROID__
        if (mapping_base < Ps5LibcHeapHint + UINT64_C(0x100000)) {
            std::array<std::uint64_t, 4> free_returns{};
            if (guest_stack != 0 &&
                HasAccessLocked(
                    guest_stack, sizeof(free_returns),
                    LSX4_PS5_GUEST_READ)) {
                std::memcpy(
                    free_returns.data(),
                    reinterpret_cast<const void*>(guest_stack),
                    sizeof(free_returns));
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-HEAP",
                "free base=0x%llx address=0x%llx requested=0x%llx "
                "mapped=0x%zx "
                "alloc_returns=0x%llx,0x%llx,0x%llx,0x%llx "
                "free_returns=0x%llx,0x%llx,0x%llx,0x%llx",
                static_cast<unsigned long long>(mapping_base),
                static_cast<unsigned long long>(removed.address),
                static_cast<unsigned long long>(
                    removed.requested_size),
                removed.mapped_size,
                static_cast<unsigned long long>(
                    removed.allocation_returns[0]),
                static_cast<unsigned long long>(
                    removed.allocation_returns[1]),
                static_cast<unsigned long long>(
                    removed.allocation_returns[2]),
                static_cast<unsigned long long>(
                    removed.allocation_returns[3]),
                static_cast<unsigned long long>(free_returns[0]),
                static_cast<unsigned long long>(free_returns[1]),
                static_cast<unsigned long long>(free_returns[2]),
                static_cast<unsigned long long>(free_returns[3]));
        }
#endif
        (void)g_runtime.mappings.EraseExact(
            mapping_base, removed.mapped_size, true);
        g_runtime.libc_heap_allocations.erase(found);
    }
    munmap(removed.allocation, removed.mapped_size);
}

bool TryReallocateLibcHeap(
    const std::uint64_t existing_address,
    const std::uint64_t requested_size,
    std::uint64_t& result,
    const std::uint64_t guest_stack = 0) {
    if (existing_address == 0) {
        return TryAllocateLibcHeap(
            requested_size, alignof(std::max_align_t), result,
            guest_stack);
    }
    if (requested_size == 0) {
        FreeLibcHeap(existing_address, guest_stack);
        result = 0;
        return true;
    }

    std::uint64_t existing_size{};
    std::uint64_t alignment{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.libc_heap_allocations.find(existing_address);
        if (found == g_runtime.libc_heap_allocations.end()) {
            result = 0;
            return true;
        }
        existing_size = found->second.requested_size;
        alignment = found->second.alignment;
    }
    if (!TryAllocateLibcHeap(
            requested_size, alignment, result, guest_stack)) {
        result = 0;
        return true;
    }
    std::memcpy(
        reinterpret_cast<void*>(result),
        reinterpret_cast<const void*>(existing_address),
        static_cast<std::size_t>(
            std::min(existing_size, requested_size)));
    FreeLibcHeap(existing_address, guest_stack);
    return true;
}

bool TryWriteGuestU64(
    const std::uint64_t address, const std::uint64_t value) {
    const std::lock_guard lock{g_runtime.mutex};
    return GuestWriteU64Locked(address, value);
}

const GuestThreadContext* CurrentGuestThreadLocked() {
    if (g_runtime.threads.empty()) {
        return nullptr;
    }
    if (g_current_guest_thread_handle != 0) {
        const auto found = std::ranges::find(
            g_runtime.threads, g_current_guest_thread_handle,
            &GuestThreadContext::handle);
        if (found != g_runtime.threads.end()) {
            return &*found;
        }
    }
    return &g_runtime.threads.back();
}

bool TryResolveTlsAddress(
    const std::uint64_t descriptor_address,
    std::uint64_t& result) {
    result = 0;
    if (descriptor_address == 0) {
        return true;
    }
    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t module_id{};
    std::uint64_t offset{};
    if (!GuestReadU64Locked(descriptor_address, module_id) ||
        !GuestReadU64Locked(
            descriptor_address + sizeof(std::uint64_t), offset)) {
        return true;
    }
    const auto* const thread = CurrentGuestThreadLocked();
    if (module_id == 0 || thread == nullptr) {
        return true;
    }
    const auto module = std::ranges::find_if(
        g_runtime.programs,
        [&](const LoadedProgram& program) {
            return ProgramBelongsToOwner(
                       program, thread->program_handle) &&
                   program.image.tls_module_id == module_id;
        });
    if (module == g_runtime.programs.end() ||
        offset >= module->image.tls_memory_size ||
        thread->fs_base < module->image.tls_static_offset) {
        return true;
    }
    result = thread->fs_base -
             module->image.tls_static_offset + offset;
    return true;
}

std::uint64_t CurrentGuestPthreadHandle() {
    const std::lock_guard lock{g_runtime.mutex};
    const auto* const thread = CurrentGuestThreadLocked();
    return thread != nullptr ? thread->pthread_handle : 0;
}

std::uint64_t GuestPthreadHandleForRuntimeThread(
    const std::uint64_t runtime_thread_handle) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto found = std::ranges::find(
        g_runtime.threads, runtime_thread_handle,
        &GuestThreadContext::handle);
    return found != g_runtime.threads.end()
        ? found->pthread_handle
        : 0;
}

constexpr std::uint64_t Ps5SyntheticPthreadAttributeHandleBase =
    UINT64_C(0x0000600400000000);
constexpr std::uint64_t Ps5GuestStackSlotSize = UINT64_C(0x100000);

GuestPthreadAttribute GuestPthreadAttributeForAddressLocked(
    const std::uint64_t address) {
    if (const auto found =
            g_runtime.guest_pthread_attributes.find(address);
        found != g_runtime.guest_pthread_attributes.end()) {
        return found->second;
    }
    std::uint64_t pointed_handle{};
    if (GuestReadU64Locked(address, pointed_handle)) {
        if (const auto found =
                g_runtime.guest_pthread_attributes.find(pointed_handle);
            found != g_runtime.guest_pthread_attributes.end()) {
            return found->second;
        }
    }
    return {};
}

std::uint64_t ResolveGuestPthreadAttributeAddressLocked(
    const std::uint64_t address) {
    if (g_runtime.guest_pthread_attributes.contains(address)) {
        return address;
    }
    std::uint64_t pointed_handle{};
    if (GuestReadU64Locked(address, pointed_handle) &&
        g_runtime.guest_pthread_attributes.contains(pointed_handle)) {
        return pointed_handle;
    }
    return address;
}

bool InferCurrentGuestStackLocked(
    const std::uint64_t guest_stack,
    std::uint64_t& stack_address,
    std::uint64_t& stack_size) {
    stack_address = 0;
    stack_size = 0;
    if (guest_stack == 0) {
        return false;
    }
    const auto* const mapping = g_runtime.mappings.FindIf(
        [&](const RegisteredMapping& candidate) {
            return guest_stack >= candidate.address &&
                   guest_stack - candidate.address <
                       candidate.byte_count &&
                   candidate.label.find("stack") != std::string::npos;
        });
    if (mapping == nullptr) {
        return false;
    }
    const auto slot_size = std::min<std::uint64_t>(
        Ps5GuestStackSlotSize, mapping->byte_count);
    if (slot_size == 0) {
        return false;
    }
    const auto offset = guest_stack - mapping->address;
    const auto slot_index = offset / slot_size;
    if (slot_index >
        (std::numeric_limits<std::uint64_t>::max() -
         mapping->address) /
            slot_size) {
        return false;
    }
    const auto candidate = mapping->address + slot_index * slot_size;
    const auto available =
        mapping->byte_count - (candidate - mapping->address);
    stack_address = candidate;
    stack_size = std::min(slot_size, available);
    return stack_size != 0;
}

bool TryGuestPthreadAttributeOperation(
    const std::string_view symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto attribute_address = request.integer_arguments[0];

    if (symbol == "nsYoNRywwNg" ||
        symbol == "wtkt-teR1so") {
        if (attribute_address == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        if (!HasAccessLocked(
                attribute_address, sizeof(std::uint64_t),
                LSX4_PS5_GUEST_WRITE)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        auto handle =
            Ps5SyntheticPthreadAttributeHandleBase +
            g_runtime.next_pthread_attribute_handle++;
        if (handle == Ps5SyntheticPthreadAttributeHandleBase) {
            handle =
                Ps5SyntheticPthreadAttributeHandleBase +
                g_runtime.next_pthread_attribute_handle++;
        }
        const GuestPthreadAttribute state{};
        g_runtime.guest_pthread_attributes[attribute_address] = state;
        g_runtime.guest_pthread_attributes[handle] = state;
        std::memcpy(
            reinterpret_cast<void*>(attribute_address),
            &handle, sizeof(handle));
        result = 0;
        return true;
    }

    if (symbol == "62KCwEMmzcM" ||
        symbol == "zHchY8ft5pk") {
        if (attribute_address == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto resolved =
            ResolveGuestPthreadAttributeAddressLocked(
                attribute_address);
        g_runtime.guest_pthread_attributes.erase(attribute_address);
        if (resolved != attribute_address) {
            g_runtime.guest_pthread_attributes.erase(resolved);
        }
        const std::uint64_t zero{};
        if (HasAccessLocked(
                attribute_address, sizeof(zero),
                LSX4_PS5_GUEST_WRITE)) {
            std::memcpy(
                reinterpret_cast<void*>(attribute_address),
                &zero, sizeof(zero));
        }
        result = 0;
        return true;
    }

    if (symbol == "x1X76arYMxU" ||
        symbol == "Ucsu-OK+els") {
        const auto thread = attribute_address;
        const auto output_address = request.integer_arguments[1];
        if (thread == 0 || output_address == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        GuestPthreadAttribute state{};
        if (thread == (CurrentGuestThreadLocked() != nullptr
                           ? CurrentGuestThreadLocked()->pthread_handle
                           : 0)) {
            (void)InferCurrentGuestStackLocked(
                request.guest_stack,
                state.stack_address, state.stack_size);
        }
        g_runtime.guest_pthread_attributes[output_address] = state;
        result = 0;
        return true;
    }

    constexpr std::array<std::string_view, 26> AttributeObjectSymbols{
        "8+s5BzZjxSg", "JaRMy+QcpeU", "txHtngJ+eyc", "Ru36fiTtJzA",
        "-fA+7ZlGDQs", "0qOtCR-ZHck", "VUT1ZSrHT0I", "JNkVVsVDmOk",
        "FXPWHNk8Of0", "qlk9pSLsUmM", "-quPa4SEJUw", "vQm4fDEsWi8",
        "3qxgM4ezETA", "-Wreprtu0Qs", "E+tyo3lp5Lw", "El+cQ20DynU",
        "JKyG3SWyA10", "eXbUSpEaTsA", "7ZlAakEf0Qg", "4+h9EzwKF4I",
        "JarMIy8kKEY", "UTXzJbWhhTE", "2Q0z6rnBrTE", "Bvn74vj6oLo",
        "DzES9hQF4f4", "euKRgm0Vn2M"};
    if (!std::ranges::contains(AttributeObjectSymbols, symbol)) {
        return false;
    }
    if (attribute_address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    if (symbol == "8+s5BzZjxSg" ||
        symbol == "JaRMy+QcpeU" ||
        symbol == "txHtngJ+eyc" ||
        symbol == "Ru36fiTtJzA" ||
        symbol == "-fA+7ZlGDQs" ||
        symbol == "0qOtCR-ZHck" ||
        symbol == "VUT1ZSrHT0I" ||
        symbol == "JNkVVsVDmOk" ||
        symbol == "FXPWHNk8Of0" ||
        symbol == "qlk9pSLsUmM") {
        const auto output_address = request.integer_arguments[1];
        if (output_address == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto state =
            GuestPthreadAttributeForAddressLocked(attribute_address);
        if (symbol == "JaRMy+QcpeU" ||
            symbol == "VUT1ZSrHT0I") {
            if (!HasAccessLocked(
                    output_address, sizeof(state.detach_state),
                    LSX4_PS5_GUEST_WRITE)) {
                result = OrbisError(MemoryFault);
                return true;
            }
            std::memcpy(
                reinterpret_cast<void*>(output_address),
                &state.detach_state, sizeof(state.detach_state));
            result = static_cast<std::uint32_t>(
                state.detach_state);
            return true;
        }
        if (symbol == "FXPWHNk8Of0" ||
            symbol == "qlk9pSLsUmM") {
            if (!HasAccessLocked(
                    output_address, sizeof(state.sched_priority),
                    LSX4_PS5_GUEST_WRITE)) {
                result = OrbisError(MemoryFault);
                return true;
            }
            std::memcpy(
                reinterpret_cast<void*>(output_address),
                &state.sched_priority,
                sizeof(state.sched_priority));
            result = 0;
            return true;
        }
        const auto value =
            symbol == "8+s5BzZjxSg" ? state.affinity_mask :
            symbol == "txHtngJ+eyc" ||
                    symbol == "JNkVVsVDmOk"
                ? state.guard_size :
            symbol == "Ru36fiTtJzA"
                ? state.stack_address
                : state.stack_size;
        if (!GuestWriteU64Locked(output_address, value)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        result = value;
        return true;
    }

    if (symbol == "-quPa4SEJUw" ||
        symbol == "vQm4fDEsWi8") {
        const auto output_stack_address =
            request.integer_arguments[1];
        const auto output_stack_size =
            request.integer_arguments[2];
        if (output_stack_address == 0 ||
            output_stack_size == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto state =
            GuestPthreadAttributeForAddressLocked(attribute_address);
        if (!GuestWriteU64Locked(
                output_stack_address, state.stack_address) ||
            !GuestWriteU64Locked(
                output_stack_size, state.stack_size)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        result = 0;
        return true;
    }

    if (symbol == "3qxgM4ezETA" ||
        symbol == "-Wreprtu0Qs" ||
        symbol == "E+tyo3lp5Lw" ||
        symbol == "El+cQ20DynU" ||
        symbol == "JKyG3SWyA10" ||
        symbol == "eXbUSpEaTsA" ||
        symbol == "7ZlAakEf0Qg" ||
        symbol == "4+h9EzwKF4I" ||
        symbol == "JarMIy8kKEY" ||
        symbol == "UTXzJbWhhTE" ||
        symbol == "2Q0z6rnBrTE") {
        const auto value = request.integer_arguments[1];
        const std::lock_guard lock{g_runtime.mutex};
        const auto resolved =
            ResolveGuestPthreadAttributeAddressLocked(
                attribute_address);
        auto state =
            GuestPthreadAttributeForAddressLocked(resolved);
        if (symbol == "3qxgM4ezETA") {
            state.affinity_mask = value;
        } else if (symbol == "-Wreprtu0Qs" ||
                   symbol == "E+tyo3lp5Lw") {
            state.detach_state =
                static_cast<std::int32_t>(value);
        } else if (symbol == "El+cQ20DynU" ||
                   symbol == "JKyG3SWyA10") {
            state.guard_size = value;
        } else if (symbol == "eXbUSpEaTsA" ||
                   symbol == "7ZlAakEf0Qg") {
            state.inherit_sched =
                static_cast<std::int32_t>(value);
        } else if (symbol == "4+h9EzwKF4I" ||
                   symbol == "JarMIy8kKEY") {
            state.sched_policy =
                static_cast<std::int32_t>(value);
        } else {
            state.stack_size = value;
        }
        g_runtime.guest_pthread_attributes[resolved] = state;
        if (resolved != attribute_address) {
            g_runtime.guest_pthread_attributes[attribute_address] =
                state;
        }
        result = 0;
        return true;
    }

    if (symbol == "Bvn74vj6oLo") {
        const auto stack_address = request.integer_arguments[1];
        const auto stack_size = request.integer_arguments[2];
        if (stack_address == 0 || stack_size == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto resolved =
            ResolveGuestPthreadAttributeAddressLocked(
                attribute_address);
        auto state =
            GuestPthreadAttributeForAddressLocked(resolved);
        state.stack_address = stack_address;
        state.stack_size = stack_size;
        g_runtime.guest_pthread_attributes[resolved] = state;
        if (resolved != attribute_address) {
            g_runtime.guest_pthread_attributes[attribute_address] =
                state;
        }
        result = 0;
        return true;
    }

    if (symbol == "DzES9hQF4f4" ||
        symbol == "euKRgm0Vn2M") {
        const auto parameter_address =
            request.integer_arguments[1];
        std::int32_t priority{};
        const std::lock_guard lock{g_runtime.mutex};
        if (parameter_address == 0 ||
            !HasAccessLocked(
                parameter_address, sizeof(priority),
                LSX4_PS5_GUEST_READ)) {
            result = OrbisError(MemoryFault);
            return true;
        }
        std::memcpy(
            &priority,
            reinterpret_cast<const void*>(parameter_address),
            sizeof(priority));
        const auto resolved =
            ResolveGuestPthreadAttributeAddressLocked(
                attribute_address);
        auto state =
            GuestPthreadAttributeForAddressLocked(resolved);
        state.sched_priority = priority;
        g_runtime.guest_pthread_attributes[resolved] = state;
        if (resolved != attribute_address) {
            g_runtime.guest_pthread_attributes[attribute_address] =
                state;
        }
        result = 0;
        return true;
    }

    return false;
}

std::shared_ptr<GuestMutex> FindOrCreateGuestMutex(
    const std::uint64_t address) {
    if (address == 0) {
        return {};
    }
    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t pointed_address{};
    if (GuestReadU64Locked(address, pointed_address)) {
        if (const auto pointed =
                g_runtime.guest_mutexes.find(pointed_address);
            pointed != g_runtime.guest_mutexes.end()) {
            return pointed->second;
        }
    }
    auto& mutex = g_runtime.guest_mutexes[address];
    if (mutex == nullptr) {
        mutex = std::make_shared<GuestMutex>();
    }
    return mutex;
}

bool TryInitializeGuestMutex(
    const std::uint64_t address, std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    if (address == 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const std::lock_guard lock{g_runtime.mutex};
    if (!HasAccessLocked(
            address, sizeof(std::uint64_t),
            LSX4_PS5_GUEST_WRITE)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (!EnsureHleDataLocked() ||
        g_runtime.next_mutex_object_offset >
            Ps5HleDataSize - Ps5MutexObjectSize) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    auto* const object =
        g_runtime.hle_data + g_runtime.next_mutex_object_offset;
    const auto object_address =
        reinterpret_cast<std::uint64_t>(object);
    g_runtime.next_mutex_object_offset += Ps5MutexObjectSize;
    std::fill_n(object, Ps5MutexObjectSize, std::uint8_t{});
    std::memcpy(
        reinterpret_cast<void*>(address),
        &object_address, sizeof(object_address));
    auto mutex = std::make_shared<GuestMutex>();
    g_runtime.guest_mutexes[address] = mutex;
    g_runtime.guest_mutexes[object_address] = std::move(mutex);
    result = 0;
    return true;
}

bool TryGuestMutexOperation(
    const std::string_view symbol,
    const std::uint64_t address,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t Busy = 0x80020010;
    if (symbol == "cmo1RIYva9o" ||
        symbol == "ttHNfU+qDBU") {
        return TryInitializeGuestMutex(address, result);
    }
    if (symbol == "2Of0f+3mhhE" ||
        symbol == "ltCfaGr2JGE") {
        if (address == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        std::uint64_t pointed_address{};
        (void)GuestReadU64Locked(address, pointed_address);
        g_runtime.guest_mutexes.erase(address);
        g_runtime.guest_mutexes.erase(pointed_address);
        constexpr std::uint64_t DestroyedMutex = 1;
        if (HasAccessLocked(
                address, sizeof(DestroyedMutex),
                LSX4_PS5_GUEST_WRITE)) {
            std::memcpy(
                reinterpret_cast<void*>(address),
                &DestroyedMutex, sizeof(DestroyedMutex));
        }
        result = 0;
        return true;
    }

    auto mutex = FindOrCreateGuestMutex(address);
    if (mutex == nullptr) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    if (symbol == "upoVrzMHFeE" ||
        symbol == "K-jXhbt2gn4") {
        result = mutex->mutex.try_lock()
            ? 0
            : OrbisError(Busy);
        return true;
    }
    if (symbol == "9UK1vLZQft4" ||
        symbol == "7H0iTOciTLo") {
        while (!g_runtime.shutdown_requested.load(
                   std::memory_order_acquire)) {
            if (mutex->mutex.try_lock_for(
                    std::chrono::milliseconds(10))) {
                result = 0;
                return true;
            }
        }
        result = 0;
        (void)Lsx4::Translation::CompleteCurrentGuestExecution(0);
        return true;
    }
    if (symbol == "tn3VlD0hG60" ||
        symbol == "2Z+PpY6CaJg") {
        mutex->mutex.unlock();
        result = 0;
        return true;
    }
    return false;
}

std::shared_ptr<GuestSemaphore> FindGuestSemaphore(
    const std::uint32_t handle) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto found = g_runtime.guest_semaphores.find(handle);
    return found != g_runtime.guest_semaphores.end()
        ? found->second
        : std::shared_ptr<GuestSemaphore>{};
}

bool TryGuestSemaphoreOperation(
    const std::string_view symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t Busy = 0x80020010;
    constexpr std::uint32_t TimedOut = 0x8002003c;
    constexpr std::uint32_t Canceled = 0x80020055;
    constexpr std::uint32_t MemoryFault = 0x80020101;

    if (symbol == "188x57JYp0g") {
        const auto output_address = request.integer_arguments[0];
        const auto name_address = request.integer_arguments[1];
        const auto attributes =
            static_cast<std::uint32_t>(request.integer_arguments[2]);
        const auto initial_count =
            static_cast<std::int32_t>(request.integer_arguments[3]);
        const auto maximum_count =
            static_cast<std::int32_t>(request.integer_arguments[4]);
        const auto option_address = request.integer_arguments[5];
        if (output_address == 0 || name_address == 0 ||
            attributes > 2 || initial_count < 0 ||
            maximum_count <= 0 || initial_count > maximum_count ||
            option_address != 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }

        auto semaphore = std::make_shared<GuestSemaphore>();
        semaphore->initial_count = initial_count;
        semaphore->maximum_count = maximum_count;
        semaphore->count = initial_count;
        std::uint32_t handle{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            do {
                handle = g_runtime.next_semaphore_handle++;
            } while (handle == 0 ||
                     g_runtime.guest_semaphores.contains(handle));
            g_runtime.guest_semaphores.emplace(handle, semaphore);
        }
        if (!TryWriteGuestBytes(
                output_address, &handle, sizeof(handle))) {
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.guest_semaphores.erase(handle);
            result = OrbisError(MemoryFault);
            return true;
        }
        result = 0;
        return true;
    }

    const auto handle =
        static_cast<std::uint32_t>(request.integer_arguments[0]);
    auto semaphore = FindGuestSemaphore(handle);
    if (semaphore == nullptr) {
        result = OrbisError(NotFound);
        return true;
    }

    if (symbol == "Zxa0VhQVTsk") {
        const auto need_count =
            static_cast<std::int32_t>(request.integer_arguments[1]);
        const auto timeout_address = request.integer_arguments[2];
        if (need_count < 1 ||
            need_count > semaphore->maximum_count) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        std::uint32_t timeout_microseconds{};
        if (timeout_address != 0 &&
            !TryReadGuestBytes(
                timeout_address, &timeout_microseconds,
                sizeof(timeout_microseconds))) {
            result = OrbisError(MemoryFault);
            return true;
        }

        std::unique_lock lock{semaphore->mutex};
        const auto cancel_generation =
            semaphore->cancel_generation;
        ++semaphore->waiting_threads;
        const auto ready = [&] {
            return semaphore->deleted ||
                   semaphore->cancel_generation !=
                       cancel_generation ||
                   semaphore->count >= need_count ||
                   g_runtime.shutdown_requested.load(
                       std::memory_order_acquire);
        };
        bool awakened{};
        if (timeout_address == 0) {
            while (!ready()) {
                semaphore->condition.wait_for(
                    lock, std::chrono::milliseconds(10));
            }
            awakened = true;
        } else {
            awakened = semaphore->condition.wait_for(
                lock,
                std::chrono::microseconds(timeout_microseconds),
                ready);
        }
        --semaphore->waiting_threads;
        if (g_runtime.shutdown_requested.load(
                std::memory_order_acquire)) {
            lock.unlock();
            result = 0;
            (void)Lsx4::Translation::CompleteCurrentGuestExecution(0);
            return true;
        }
        if (semaphore->deleted ||
            semaphore->cancel_generation != cancel_generation) {
            result = OrbisError(Canceled);
            return true;
        }
        if (!awakened || semaphore->count < need_count) {
            if (timeout_address != 0) {
                constexpr std::uint32_t Zero = 0;
                lock.unlock();
                (void)TryWriteGuestBytes(
                    timeout_address, &Zero, sizeof(Zero));
            }
            result = OrbisError(TimedOut);
            return true;
        }
        semaphore->count -= need_count;
        lock.unlock();
        if (timeout_address != 0) {
            constexpr std::uint32_t Zero = 0;
            (void)TryWriteGuestBytes(
                timeout_address, &Zero, sizeof(Zero));
        }
        result = 0;
        return true;
    }

    if (symbol == "12wOHk8ywb0") {
        const auto need_count =
            static_cast<std::int32_t>(request.integer_arguments[1]);
        if (need_count < 1 ||
            need_count > semaphore->maximum_count) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        const std::lock_guard lock{semaphore->mutex};
        if (semaphore->count < need_count) {
            result = OrbisError(Busy);
            return true;
        }
        semaphore->count -= need_count;
        result = 0;
        return true;
    }

    if (symbol == "4czppHBiriw") {
        const auto signal_count =
            static_cast<std::int32_t>(request.integer_arguments[1]);
        {
            const std::lock_guard lock{semaphore->mutex};
            if (signal_count <= 0 ||
                semaphore->count >
                    semaphore->maximum_count - signal_count) {
                result = OrbisError(InvalidArgument);
                return true;
            }
            semaphore->count += signal_count;
        }
        semaphore->condition.notify_all();
        result = 0;
        return true;
    }

    if (symbol == "4DM06U2BNEY") {
        const auto set_count =
            static_cast<std::int32_t>(request.integer_arguments[1]);
        const auto waiting_output = request.integer_arguments[2];
        std::uint32_t waiting_threads{};
        {
            const std::lock_guard lock{semaphore->mutex};
            if (set_count > semaphore->maximum_count) {
                result = OrbisError(InvalidArgument);
                return true;
            }
            waiting_threads = semaphore->waiting_threads;
            semaphore->count = set_count < 0
                ? semaphore->initial_count
                : set_count;
            ++semaphore->cancel_generation;
        }
        if (waiting_output != 0 &&
            !TryWriteGuestBytes(
                waiting_output, &waiting_threads,
                sizeof(waiting_threads))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        semaphore->condition.notify_all();
        result = 0;
        return true;
    }

    if (symbol == "R1Jvn8bSCW8") {
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto found =
                g_runtime.guest_semaphores.find(handle);
            if (found == g_runtime.guest_semaphores.end()) {
                result = OrbisError(NotFound);
                return true;
            }
            g_runtime.guest_semaphores.erase(found);
        }
        {
            const std::lock_guard lock{semaphore->mutex};
            semaphore->deleted = true;
        }
        semaphore->condition.notify_all();
        result = 0;
        return true;
    }
    return false;
}

std::shared_ptr<GuestConditionVariable>
FindOrCreateGuestConditionVariable(
    const std::uint64_t address) {
    if (address == 0) {
        return {};
    }
    const std::lock_guard lock{g_runtime.mutex};
    auto& condition = g_runtime.guest_condition_variables[address];
    if (condition == nullptr) {
        condition =
            std::make_shared<GuestConditionVariable>();
    }
    return condition;
}

bool ReacquireGuestMutex(
    const std::shared_ptr<GuestMutex>& mutex) {
    while (!g_runtime.shutdown_requested.load(
               std::memory_order_acquire)) {
        if (mutex->mutex.try_lock_for(
                std::chrono::milliseconds(10))) {
            return true;
        }
    }
    return false;
}

bool TryGuestConditionVariableOperation(
    const std::string_view symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t Busy = 0x80020010;
    constexpr std::uint32_t TimedOut = 0x8002003c;

    const auto condition_address =
        request.integer_arguments[0];
    if (symbol == "2Tb92quprl0" ||
        symbol == "0TyVk4MSLt0") {
        result =
            FindOrCreateGuestConditionVariable(
                condition_address) != nullptr
            ? 0
            : OrbisError(InvalidArgument);
        return true;
    }

    auto condition =
        FindOrCreateGuestConditionVariable(
            condition_address);
    if (condition == nullptr) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    if (symbol == "g+PZd2hiacg" ||
        symbol == "RXXqi4CtF8w") {
        {
            const std::lock_guard lock{condition->mutex};
            if (condition->waiting_threads != 0) {
                result = OrbisError(Busy);
                return true;
            }
            condition->deleted = true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.guest_condition_variables.erase(
            condition_address);
        result = 0;
        return true;
    }

    if (symbol == "kDh-NfxgMtE" ||
        symbol == "2MOy+rUfuhQ" ||
        symbol == "JGgj7Uvrl+A" ||
        symbol == "mkx2fVhNMsg") {
        const bool broadcast =
            symbol == "JGgj7Uvrl+A" ||
            symbol == "mkx2fVhNMsg";
        {
            const std::lock_guard lock{condition->mutex};
            const auto available =
                condition->waiting_threads -
                std::min(
                    condition->waiting_threads,
                    condition->pending_signals);
            if (broadcast) {
                condition->pending_signals += available;
            } else if (available != 0) {
                ++condition->pending_signals;
            }
        }
        if (broadcast) {
            condition->condition.notify_all();
        } else {
            condition->condition.notify_one();
        }
        result = 0;
        return true;
    }

    const bool sce_wait =
        symbol == "WKAXJ4XBPQ4" ||
        symbol == "BmMjYxmew1w";
    const bool posix_wait =
        symbol == "Op8TBGY5KHg" ||
        symbol == "27bAgiJmOh0";
    if (!sce_wait && !posix_wait) {
        return false;
    }
    const auto mutex_address = request.integer_arguments[1];
    auto guest_mutex = FindOrCreateGuestMutex(mutex_address);
    if (guest_mutex == nullptr) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    bool timed =
        symbol == "BmMjYxmew1w" ||
        symbol == "27bAgiJmOh0";
    std::chrono::microseconds timeout{};
    if (symbol == "BmMjYxmew1w") {
        timeout = std::chrono::microseconds{
            static_cast<std::uint32_t>(
                request.integer_arguments[2])};
    } else if (symbol == "27bAgiJmOh0") {
        timeout = std::chrono::milliseconds(1);
    }

    std::unique_lock condition_lock{condition->mutex};
    ++condition->waiting_threads;
    guest_mutex->mutex.unlock();
    bool signaled{};
    if (!timed) {
        while (!condition->deleted &&
               condition->pending_signals == 0 &&
               !g_runtime.shutdown_requested.load(
                   std::memory_order_acquire)) {
            condition->condition.wait_for(
                condition_lock,
                std::chrono::milliseconds(10));
        }
        signaled = condition->pending_signals != 0;
    } else {
        const auto deadline =
            std::chrono::steady_clock::now() + timeout;
        while (!condition->deleted &&
               condition->pending_signals == 0 &&
               !g_runtime.shutdown_requested.load(
                   std::memory_order_acquire)) {
            if (condition->condition.wait_until(
                    condition_lock, deadline) ==
                std::cv_status::timeout) {
                break;
            }
        }
        signaled = condition->pending_signals != 0;
    }
    if (signaled) {
        --condition->pending_signals;
    }
    --condition->waiting_threads;
    condition_lock.unlock();

    if (!ReacquireGuestMutex(guest_mutex)) {
        result = 0;
        (void)Lsx4::Translation::CompleteCurrentGuestExecution(0);
        return true;
    }
    result = signaled ? 0 : OrbisError(TimedOut);
    return true;
}

struct GuestFiberLayout {
    std::uint32_t magic_start{};
    std::uint32_t state{};
    std::uint64_t entry{};
    std::uint64_t argument_on_initialize{};
    std::uint64_t context_address{};
    std::uint64_t context_size{};
    std::array<char, 32> name{};
    std::uint64_t reserved_context{};
    std::uint32_t flags{};
    std::uint32_t padding{};
    std::uint64_t context_start{};
    std::uint64_t context_end{};
    std::uint32_t magic_end{};
    std::uint32_t tail_padding{};
};
static_assert(sizeof(GuestFiberLayout) == 112);

struct GuestFiberContinuation {
    Lsx4::Translation::CpuFrame machine{};
    Lsx4::Translation::StackWindow stack_window{};
    std::uint64_t resume_argument_output{};
    bool has_machine{};
    bool owns_stack_window{};
};

struct GuestFiberThreadRuntime {
    Lsx4::Translation::CpuFrame thread_machine{};
    std::uint64_t current_fiber{};
    std::uint64_t thread_return_output{};
    bool active{};
};

thread_local GuestFiberThreadRuntime g_guest_fiber_runtime{};

std::mutex g_guest_fiber_continuations_mutex;
std::unordered_map<
    std::uint64_t, std::shared_ptr<GuestFiberContinuation>>
    g_guest_fiber_continuations;

std::shared_ptr<GuestFiberContinuation> GuestFiberContinuationFor(
    const std::uint64_t fiber, const bool create) {
    const std::lock_guard lock{g_guest_fiber_continuations_mutex};
    if (const auto found = g_guest_fiber_continuations.find(fiber);
        found != g_guest_fiber_continuations.end()) {
        return found->second;
    }
    if (!create) {
        return {};
    }
    auto continuation = std::make_shared<GuestFiberContinuation>();
    g_guest_fiber_continuations.emplace(fiber, continuation);
    return continuation;
}

void EraseGuestFiberContinuation(const std::uint64_t fiber) {
    const std::lock_guard lock{g_guest_fiber_continuations_mutex};
    g_guest_fiber_continuations.erase(fiber);
}

bool CaptureGuestFiberContinuation(
    Lsx4::Translation::CpuFrame& destination,
    const std::uint64_t result) {
    if (!Lsx4::Translation::SnapshotContinuationState(destination)) {
        return false;
    }
    Lsx4::Translation::WriteInteger(
        destination, Lsx4::Translation::IntegerRegister::A,
        result, 64);
    destination.dispatch_phase = 0;
    return destination.resume_address != 0 &&
           Lsx4::Translation::ReadInteger(
               destination,
               Lsx4::Translation::IntegerRegister::Stack,
               64) != 0;
}

bool PublishGuestFiberState(
    const Lsx4::Translation::CpuFrame& source) {
    const auto current = Lsx4::Translation::PublishedLiveState();
    return current && current.ReplaceWith(source) &&
           Lsx4::Translation::RequestLiveStateReplacement();
}

bool BuildInitialGuestFiberState(
    const GuestFiberLayout& fiber,
    GuestFiberContinuation& continuation,
    const std::uint64_t argument_on_run,
    Lsx4::Translation::CpuFrame& destination) {
    const auto current = Lsx4::Translation::PublishedLiveState();
    if (!current || fiber.entry == 0 ||
        !current.CopyTo(destination)) {
        return false;
    }
    destination.integer.fill(0);
    destination.resume_address = fiber.entry;
    destination.condition_word = UINT64_C(0x202);
    destination.dispatch_phase = 0;
    destination.x87_status = 0;
    destination.x87_opcode = 0;
    destination.x87_code_address = 0;
    destination.x87_data_address = 0;
    if ((fiber.flags & UINT32_C(0x100)) != 0) {
        destination.x87_control = UINT16_C(0x037f);
        destination.simd_control = UINT32_C(0x9fc0);
    }
    Lsx4::Translation::WriteInteger(
        destination,
        Lsx4::Translation::IntegerRegister::Destination,
        fiber.argument_on_initialize, 64);
    Lsx4::Translation::WriteInteger(
        destination,
        Lsx4::Translation::IntegerRegister::Source,
        argument_on_run, 64);

    std::uint64_t stack_top{};
    if (fiber.context_address != 0 &&
        fiber.context_size != 0) {
        stack_top = fiber.context_address + fiber.context_size;
    } else {
        continuation.stack_window =
            Lsx4::Translation::AcquireStackWindow(current);
        continuation.owns_stack_window =
            continuation.stack_window.IsValid();
        stack_top = continuation.stack_window.stack_pointer;
    }
    stack_top &= ~UINT64_C(0xf);
    if (stack_top < sizeof(std::uint64_t)) {
        return false;
    }
    const auto stack_pointer =
        stack_top - sizeof(std::uint64_t);
    const std::uint64_t return_address{};
    if (!TryWriteGuestValue(stack_pointer, return_address)) {
        return false;
    }
    Lsx4::Translation::WriteInteger(
        destination,
        Lsx4::Translation::IntegerRegister::Stack,
        stack_pointer, 64);
    return true;
}

bool TryHandleGuestFiber(
    const std::string_view symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t FiberMagicStart = 0xdef1649c;
    constexpr std::uint32_t FiberMagicEnd = 0xb37592a0;
    constexpr std::uint32_t FiberOptionMagic = 0xbb40e64d;
    constexpr std::uint64_t FiberStackMagic =
        UINT64_C(0x7149f2ca7149f2ca);
    constexpr std::uint32_t FiberRun = 1;
    constexpr std::uint32_t FiberIdle = 2;
    constexpr std::uint32_t FiberTerminated = 3;
    constexpr std::uint32_t ErrorNull = 0x80590001;
    constexpr std::uint32_t ErrorAlignment = 0x80590002;
    constexpr std::uint32_t ErrorRange = 0x80590003;
    constexpr std::uint32_t ErrorInvalid = 0x80590004;
    constexpr std::uint32_t ErrorPermission = 0x80590005;
    constexpr std::uint32_t ErrorState = 0x80590006;

    const auto fail = [&](const std::uint32_t error) {
        result = OrbisError(error);
        return true;
    };
    if (symbol == "asjUJJ+aa8s") {
        if (request.integer_arguments[0] == 0) {
            return fail(ErrorNull);
        }
        if ((request.integer_arguments[0] & 7u) != 0) {
            return fail(ErrorAlignment);
        }
        if (!TryWriteGuestValue(
                request.integer_arguments[0],
                FiberOptionMagic)) {
            return fail(ErrorInvalid);
        }
        result = 0;
        return true;
    }
    if (symbol == "hVYD7Ou2pCQ" ||
        symbol == "7+OJIpko9RY") {
        const auto fiber_address = request.integer_arguments[0];
        const auto name_address = request.integer_arguments[1];
        const auto entry = request.integer_arguments[2];
        const auto argument_on_initialize =
            request.integer_arguments[3];
        const auto context_address =
            request.integer_arguments[4];
        const auto context_size =
            request.integer_arguments[5];
        std::uint64_t option_address{};
        std::uint64_t flags{};
        std::uint64_t build_version{};
        if (!TryReadGuestValue(
                request.guest_stack + 8u, option_address)) {
            return fail(ErrorInvalid);
        }
        if (symbol == "7+OJIpko9RY") {
            if (!TryReadGuestValue(
                    request.guest_stack + 16u, flags) ||
                !TryReadGuestValue(
                    request.guest_stack + 24u,
                    build_version)) {
                return fail(ErrorInvalid);
            }
        } else if (!TryReadGuestValue(
                       request.guest_stack + 16u,
                       build_version)) {
            return fail(ErrorInvalid);
        }
        if (fiber_address == 0 || name_address == 0 ||
            entry == 0) {
            return fail(ErrorNull);
        }
        if ((fiber_address & 7u) != 0 ||
            (context_address & 15u) != 0 ||
            (option_address & 7u) != 0) {
            return fail(ErrorAlignment);
        }
        if (context_size != 0 && context_size < 512u) {
            return fail(ErrorRange);
        }
        if ((context_size & 15u) != 0 ||
            ((context_address == 0) != (context_size == 0))) {
            return fail(ErrorInvalid);
        }
        if (option_address != 0) {
            std::uint32_t option_magic{};
            if (!TryReadGuestValue(
                    option_address, option_magic) ||
                option_magic != FiberOptionMagic) {
                return fail(ErrorInvalid);
            }
        }
        GuestFiberLayout fiber{};
        fiber.magic_start = FiberMagicStart;
        fiber.state = FiberIdle;
        fiber.entry = entry;
        fiber.argument_on_initialize =
            argument_on_initialize;
        fiber.context_address = context_address;
        fiber.context_size = context_size;
        fiber.flags = static_cast<std::uint32_t>(flags);
        if (build_version >= UINT64_C(0x3500000)) {
            fiber.flags |= UINT32_C(0x100);
        }
        fiber.context_start = context_address;
        fiber.context_end = context_address + context_size;
        fiber.magic_end = FiberMagicEnd;
        std::string name;
        if (TryReadGuestCString(name_address, 32, name)) {
            std::memcpy(
                fiber.name.data(), name.data(),
                std::min(name.size(), fiber.name.size() - 1u));
        }
        if (!TryWriteGuestBytes(
                fiber_address, &fiber, sizeof(fiber))) {
            return fail(ErrorInvalid);
        }
        if (context_address != 0 &&
            !TryWriteGuestValue(
                context_address, FiberStackMagic)) {
            return fail(ErrorInvalid);
        }
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled()) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-FIBER",
                "init fiber=0x%llx entry=0x%llx arg=0x%llx "
                "context=0x%llx size=0x%llx name=%s",
                static_cast<unsigned long long>(fiber_address),
                static_cast<unsigned long long>(entry),
                static_cast<unsigned long long>(
                    argument_on_initialize),
                static_cast<unsigned long long>(context_address),
                static_cast<unsigned long long>(context_size),
                name.empty() ? "<unnamed>" : name.c_str());
        }
#endif
        result = 0;
        return true;
    }

    if (symbol == "JzyT91ucGDc") {
        const auto fiber_address = request.integer_arguments[0];
        const auto name_address = request.integer_arguments[1];
        GuestFiberLayout fiber{};
        if (fiber_address == 0 || name_address == 0) {
            return fail(ErrorNull);
        }
        if (!TryReadGuestValue(fiber_address, fiber) ||
            fiber.magic_start != FiberMagicStart ||
            fiber.magic_end != FiberMagicEnd) {
            return fail(ErrorInvalid);
        }
        std::string name;
        if (!TryReadGuestCString(name_address, 32, name)) {
            return fail(ErrorInvalid);
        }
        fiber.name.fill(0);
        std::memcpy(
            fiber.name.data(), name.data(),
            std::min(name.size(), fiber.name.size() - 1u));
        result = TryWriteGuestValue(fiber_address, fiber)
            ? 0
            : OrbisError(ErrorInvalid);
        return true;
    }
    if (symbol == "uq2Y5BFz0PE") {
        const auto fiber_address = request.integer_arguments[0];
        const auto output = request.integer_arguments[1];
        GuestFiberLayout fiber{};
        std::uint64_t requested_size{};
        if (fiber_address == 0 || output == 0) {
            return fail(ErrorNull);
        }
        if (!TryReadGuestValue(fiber_address, fiber) ||
            !TryReadGuestValue(output, requested_size) ||
            requested_size != 128u ||
            fiber.magic_start != FiberMagicStart ||
            fiber.magic_end != FiberMagicEnd) {
            return fail(ErrorInvalid);
        }
        std::array<std::uint8_t, 128> info{};
        std::memcpy(info.data(), &requested_size, 8);
        std::memcpy(info.data() + 8, &fiber.entry, 8);
        std::memcpy(
            info.data() + 16,
            &fiber.argument_on_initialize, 8);
        std::memcpy(
            info.data() + 24, &fiber.context_address, 8);
        std::memcpy(
            info.data() + 32, &fiber.context_size, 8);
        std::memcpy(
            info.data() + 40, fiber.name.data(),
            fiber.name.size());
        const std::uint64_t unknown_margin = UINT64_MAX;
        std::memcpy(info.data() + 72, &unknown_margin, 8);
        result = TryWriteGuestBytes(
                     output, info.data(), info.size())
            ? 0
            : OrbisError(ErrorInvalid);
        return true;
    }
    if (symbol == "JeNX5F-NzQU") {
        const auto fiber_address = request.integer_arguments[0];
        GuestFiberLayout fiber{};
        if (fiber_address == 0) {
            return fail(ErrorNull);
        }
        if (!TryReadGuestValue(fiber_address, fiber) ||
            fiber.magic_start != FiberMagicStart ||
            fiber.magic_end != FiberMagicEnd) {
            return fail(ErrorInvalid);
        }
        if (fiber.state != FiberIdle) {
            return fail(ErrorState);
        }
        fiber.state = FiberTerminated;
        if (!TryWriteGuestValue(fiber_address, fiber)) {
            return fail(ErrorInvalid);
        }
        if (const auto continuation =
                GuestFiberContinuationFor(
                    fiber_address, false);
            continuation != nullptr &&
            continuation->owns_stack_window) {
            Lsx4::Translation::ReleaseStackWindow(
                continuation->stack_window);
        }
        EraseGuestFiberContinuation(fiber_address);
        result = 0;
        return true;
    }

    if (symbol == "p+zLIOg27zU") {
        const auto output = request.integer_arguments[0];
        if (output == 0) {
            return fail(ErrorNull);
        }
        if (!g_guest_fiber_runtime.active ||
            g_guest_fiber_runtime.current_fiber == 0) {
            return fail(ErrorPermission);
        }
        result = TryWriteGuestValue(
                     output,
                     g_guest_fiber_runtime.current_fiber)
            ? 0
            : OrbisError(ErrorInvalid);
        return true;
    }

    if (symbol == "B0ZX2hx9DMw") {
        auto& runtime = g_guest_fiber_runtime;
        if (!runtime.active || runtime.current_fiber == 0) {
            return fail(ErrorPermission);
        }
        GuestFiberLayout fiber{};
        if (!TryReadGuestValue(
                runtime.current_fiber, fiber)) {
            return fail(ErrorInvalid);
        }
        const auto continuation =
            GuestFiberContinuationFor(
                runtime.current_fiber, true);
        if (continuation == nullptr) {
            return fail(ErrorInvalid);
        }
        if (fiber.context_address != 0) {
            if (!CaptureGuestFiberContinuation(
                    continuation->machine, 0)) {
                return fail(ErrorInvalid);
            }
            continuation->resume_argument_output =
                request.integer_arguments[1];
            continuation->has_machine = true;
        } else {
            continuation->resume_argument_output = 0;
            continuation->has_machine = false;
        }
        if (runtime.thread_return_output != 0) {
            (void)TryWriteGuestValue(
                runtime.thread_return_output,
                request.integer_arguments[0]);
        }
        fiber.state = FiberIdle;
        (void)TryWriteGuestValue(
            runtime.current_fiber, fiber);
        Lsx4::Translation::WriteInteger(
            runtime.thread_machine,
            Lsx4::Translation::IntegerRegister::A,
            0, 64);
        runtime.current_fiber = 0;
        runtime.thread_return_output = 0;
        runtime.active = false;
        result = PublishGuestFiberState(
                     runtime.thread_machine)
            ? 0
            : OrbisError(ErrorInvalid);
        return true;
    }

    const bool run =
        symbol == "a0LLrZWac0M" ||
        symbol == "avfGJ94g36Q";
    const bool switch_fiber =
        symbol == "PFT2S-tJ7Uk" ||
        symbol == "ZqhZFuzKT6U";
    if (!run && !switch_fiber) {
        return false;
    }
    const auto fiber_address = request.integer_arguments[0];
    auto argument_on_run = request.integer_arguments[1];
    auto argument_output = request.integer_arguments[2];
    std::uint64_t attach_context{};
    std::uint64_t attach_size{};
    if (symbol == "avfGJ94g36Q" ||
        symbol == "ZqhZFuzKT6U") {
        attach_context = request.integer_arguments[1];
        attach_size = request.integer_arguments[2];
        argument_on_run = request.integer_arguments[3];
        argument_output = request.integer_arguments[4];
    }
    if (fiber_address == 0) {
        return fail(ErrorNull);
    }
    GuestFiberLayout fiber{};
    if (!TryReadGuestValue(fiber_address, fiber) ||
        fiber.magic_start != FiberMagicStart ||
        fiber.magic_end != FiberMagicEnd) {
        return fail(ErrorInvalid);
    }
    if (attach_context != 0 || attach_size != 0) {
        if ((attach_context & 15u) != 0 ||
            attach_size < 512u ||
            (attach_size & 15u) != 0 ||
            fiber.context_address != 0) {
            return fail(ErrorInvalid);
        }
        fiber.context_address = attach_context;
        fiber.context_size = attach_size;
        fiber.context_start = attach_context;
        fiber.context_end = attach_context + attach_size;
        (void)TryWriteGuestValue(
            attach_context, FiberStackMagic);
    }
    if (fiber.state != FiberIdle) {
        return fail(ErrorState);
    }
    const auto next_continuation =
        GuestFiberContinuationFor(
            fiber_address, true);
    if (next_continuation == nullptr) {
        return fail(ErrorInvalid);
    }
    auto& runtime = g_guest_fiber_runtime;
    if (run && runtime.active) {
        return fail(ErrorPermission);
    }

    if (switch_fiber) {
        if (!runtime.active ||
            runtime.current_fiber == 0) {
            return fail(ErrorPermission);
        }
        GuestFiberLayout previous{};
        const auto previous_continuation =
            GuestFiberContinuationFor(
                runtime.current_fiber, true);
        if (previous_continuation == nullptr ||
            !TryReadGuestValue(
                runtime.current_fiber, previous)) {
            return fail(ErrorInvalid);
        }
        if (previous.context_address != 0) {
            if (!CaptureGuestFiberContinuation(
                    previous_continuation->machine, 0)) {
                return fail(ErrorInvalid);
            }
            previous_continuation->resume_argument_output =
                argument_output;
            previous_continuation->has_machine = true;
        } else {
            previous_continuation->resume_argument_output = 0;
            previous_continuation->has_machine = false;
        }
        previous.state = FiberIdle;
        (void)TryWriteGuestValue(
            runtime.current_fiber, previous);
    }

    Lsx4::Translation::CpuFrame next{};
    if (fiber.context_address != 0 &&
        next_continuation->has_machine) {
        next = next_continuation->machine;
        if (next_continuation->resume_argument_output != 0) {
            (void)TryWriteGuestValue(
                next_continuation->resume_argument_output,
                argument_on_run);
        }
        Lsx4::Translation::WriteInteger(
            next, Lsx4::Translation::IntegerRegister::A,
            0, 64);
    } else if (!BuildInitialGuestFiberState(
                   fiber, *next_continuation,
                   argument_on_run, next)) {
        return fail(ErrorInvalid);
    }
    // Fibers may migrate between PS5 job-worker threads.  The saved CPU
    // continuation belongs to the fiber, while FS/GS belong to the thread
    // executing it (native setjmp/longjmp does not replace kernel TLS).
    // Preserve the destination worker's segment origins on every transfer.
    if (const auto current =
            Lsx4::Translation::PublishedLiveState()) {
        next.fs_origin = current.ReadScalar(
            Lsx4::Translation::FrameScalar::FsOrigin);
        next.gs_origin = current.ReadScalar(
            Lsx4::Translation::FrameScalar::GsOrigin);
    }
    if (run &&
        !CaptureGuestFiberContinuation(
            runtime.thread_machine, 0)) {
        return fail(ErrorInvalid);
    }
    fiber.state = FiberRun;
    if (!TryWriteGuestValue(fiber_address, fiber)) {
        return fail(ErrorInvalid);
    }
    runtime.current_fiber = fiber_address;
    if (run) {
        runtime.thread_return_output = argument_output;
        runtime.active = true;
    }
    next_continuation->has_machine = false;
    next_continuation->resume_argument_output = 0;
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-FIBER",
            "%s fiber=0x%llx entry=0x%llx arg=0x%llx "
            "context=0x%llx size=0x%llx",
            run ? "run" : "switch",
            static_cast<unsigned long long>(fiber_address),
            static_cast<unsigned long long>(fiber.entry),
            static_cast<unsigned long long>(argument_on_run),
            static_cast<unsigned long long>(
                fiber.context_address),
            static_cast<unsigned long long>(
                fiber.context_size));
    }
#endif
    result = PublishGuestFiberState(next)
        ? 0
        : OrbisError(ErrorInvalid);
    return true;
}

bool TryHandleGuestEventFlag(
    const std::string_view symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr std::uint32_t NotFound = 0x80020002;
    constexpr std::uint32_t InvalidArgument = 0x80020003;
    constexpr std::uint32_t MemoryFault = 0x80020101;
    constexpr std::uint32_t TimedOut = 0x8002003c;
    constexpr std::uint32_t Busy = 0x8002000a;
    constexpr std::uint32_t WaitAnd = 0x01;
    constexpr std::uint32_t WaitOr = 0x02;
    constexpr std::uint32_t ClearAll = 0x10;
    constexpr std::uint32_t ClearPattern = 0x20;

    if (symbol == "BpFoboUJoZU") {
        const auto output = request.integer_arguments[0];
        const auto attributes =
            static_cast<std::uint32_t>(request.integer_arguments[2]);
        const auto initial_pattern = request.integer_arguments[3];
        const auto option = request.integer_arguments[4];
        const auto queue_mode = attributes & 0x0fu;
        const auto thread_mode = attributes & 0xf0u;
        if (output == 0 || request.integer_arguments[1] == 0 ||
            option != 0 ||
            (queue_mode != 0 && queue_mode != 1 && queue_mode != 2) ||
            (thread_mode != 0 && thread_mode != 0x10 &&
             thread_mode != 0x20) ||
            (attributes & ~UINT32_C(0x33)) != 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        auto state = std::make_shared<GuestEventFlag>();
        state->bits = initial_pattern;
        state->attributes = attributes;
        std::uint64_t handle{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            do {
                handle = g_runtime.next_event_flag_handle++;
            } while (handle == 0 ||
                     g_runtime.guest_event_flags.contains(handle));
            g_runtime.guest_event_flags.emplace(handle, state);
        }
        if (!TryWriteGuestValue(output, handle)) {
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.guest_event_flags.erase(handle);
            result = OrbisError(MemoryFault);
            return true;
        }
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled()) {
            std::string name;
            (void)TryReadGuestCString(
                request.integer_arguments[1], 64, name);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-EVENTFLAG",
                "create handle=0x%llx name=%s attr=0x%x bits=0x%llx",
                static_cast<unsigned long long>(handle),
                name.empty() ? "<unnamed>" : name.c_str(),
                attributes,
                static_cast<unsigned long long>(initial_pattern));
        }
#endif
        result = 0;
        return true;
    }

    const auto handle = request.integer_arguments[0];
    struct EventFlagCache {
        std::uint64_t generation{};
        std::unordered_map<
            std::uint64_t, std::shared_ptr<GuestEventFlag>> entries;
    };
    thread_local EventFlagCache cache;
    const auto generation =
        g_event_flag_generation.load(std::memory_order_acquire);
    if (cache.generation != generation) {
        cache.entries.clear();
        cache.generation = generation;
    }
    std::shared_ptr<GuestEventFlag> state;
    if (const auto found = cache.entries.find(handle);
        found != cache.entries.end()) {
        state = found->second;
    } else {
        const std::lock_guard lock{g_runtime.mutex};
        const auto runtime_found =
            g_runtime.guest_event_flags.find(handle);
        if (runtime_found != g_runtime.guest_event_flags.end()) {
            state = runtime_found->second;
            cache.entries.emplace(handle, state);
        }
    }
    if (!state ||
        state->deleted.load(std::memory_order_acquire)) {
        result = OrbisError(NotFound);
        return true;
    }

    if (symbol == "8mql9OcQnd4") {
        {
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.guest_event_flags.erase(handle);
        }
        state->deleted.store(true, std::memory_order_release);
        state->condition.notify_all();
        result = 0;
        return true;
    }
    if (symbol == "IOnSvHzqu6A") {
        const auto requested_bits = request.integer_arguments[1];
        const auto before_bits =
            state->bits.load(std::memory_order_acquire);
        const auto waiters = state->waiting_threads.load(
            std::memory_order_acquire);
        if (waiters != 0) {
            std::uint64_t previous_bits{};
            {
                const std::lock_guard lock{state->mutex};
                previous_bits = state->bits.fetch_or(
                    request.integer_arguments[1],
                    std::memory_order_acq_rel);
            }
            const auto changed_bits =
                request.integer_arguments[1] & ~previous_bits;
            if ((changed_bits &
                 state->waiting_patterns.load(
                     std::memory_order_acquire)) != 0) {
                if (waiters == 1) {
                    state->condition.notify_one();
                } else {
                    state->condition.notify_all();
                }
            }
        } else {
            state->bits.fetch_or(
                requested_bits,
                std::memory_order_release);
        }
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled()) {
            static std::atomic<std::uint64_t> set_logs{};
            const auto log_index =
                set_logs.fetch_add(1u, std::memory_order_relaxed) + 1u;
            if (log_index <= 8u ||
                (log_index & (log_index - 1u)) == 0u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-EVENTFLAG",
                    "set count=%llu handle=0x%llx add=0x%llx "
                    "before=0x%llx after=0x%llx waiters=%u",
                    static_cast<unsigned long long>(log_index),
                    static_cast<unsigned long long>(handle),
                    static_cast<unsigned long long>(requested_bits),
                    static_cast<unsigned long long>(before_bits),
                    static_cast<unsigned long long>(
                        state->bits.load(std::memory_order_acquire)),
                    waiters);
            }
        }
#endif
        result = 0;
        return true;
    }
    if (symbol == "7uhBFWRAS60") {
        state->bits.fetch_and(
            request.integer_arguments[1],
            std::memory_order_acq_rel);
        result = 0;
        return true;
    }
    if (symbol == "PZku4ZrXJqg") {
        const auto waiter_output = request.integer_arguments[2];
        std::uint32_t waiters{};
        {
            const std::lock_guard lock{state->mutex};
            waiters = state->waiting_threads.load(
                std::memory_order_relaxed);
            state->bits.store(
                request.integer_arguments[1],
                std::memory_order_release);
        }
        state->condition.notify_all();
        result = waiter_output == 0 ||
                         TryWriteGuestValue(waiter_output, waiters)
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol != "JTvBflhYazQ" &&
        symbol != "9lvj5DjHZiA") {
        return false;
    }

    const auto pattern = request.integer_arguments[1];
    const auto wait_mode =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    const auto result_output = request.integer_arguments[3];
    const auto timeout_output = request.integer_arguments[4];
    const auto condition_mode = wait_mode & 0x0fu;
    const auto clear_mode = wait_mode & 0xf0u;
    if (pattern == 0 ||
        (condition_mode != WaitAnd && condition_mode != WaitOr) ||
        (clear_mode != 0 && clear_mode != ClearAll &&
         clear_mode != ClearPattern) ||
        (wait_mode & ~UINT32_C(0x33)) != 0) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    const auto satisfied = [&](const std::uint64_t bits) {
        return condition_mode == WaitAnd
            ? (bits & pattern) == pattern
            : (bits & pattern) != 0;
    };
    const auto publish_result = [&](const std::uint64_t bits) {
        return result_output == 0 ||
               TryWriteGuestValue(result_output, bits);
    };
    const auto try_consume = [&](std::uint64_t& observed) {
        observed = state->bits.load(std::memory_order_acquire);
        for (;;) {
            if (!satisfied(observed)) {
                return false;
            }
            auto desired = observed;
            if (clear_mode == ClearAll) {
                desired = 0;
            } else if (clear_mode == ClearPattern) {
                desired &= ~pattern;
            } else {
                return true;
            }
            if (state->bits.compare_exchange_weak(
                    observed, desired, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                return true;
            }
        }
    };

    std::uint64_t bits{};
    if (symbol == "9lvj5DjHZiA") {
        if (!try_consume(bits)) {
            if (!publish_result(bits)) {
                result = OrbisError(MemoryFault);
                return true;
            }
            result = OrbisError(Busy);
        } else {
            if (!publish_result(bits)) {
                result = OrbisError(MemoryFault);
                return true;
            }
            result = 0;
        }
        return true;
    }

    std::uint32_t timeout_microseconds{};
    if (timeout_output != 0 &&
        !TryReadGuestValue(timeout_output, timeout_microseconds)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (try_consume(bits)) {
        result = publish_result(bits)
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        static std::atomic<std::uint64_t> wait_logs{};
        const auto log_index =
            wait_logs.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (log_index <= 8u ||
            (log_index & (log_index - 1u)) == 0u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-EVENTFLAG",
                "wait count=%llu handle=0x%llx pattern=0x%llx "
                "mode=0x%x bits=0x%llx timeout=%u timed=%d",
                static_cast<unsigned long long>(log_index),
                static_cast<unsigned long long>(handle),
                static_cast<unsigned long long>(pattern), wait_mode,
                static_cast<unsigned long long>(bits),
                timeout_microseconds, timeout_output != 0 ? 1 : 0);
        }
    }
#endif

    // PS5 job-system flags are commonly set a few microseconds after the
    // waiter reaches this point.  A short adaptive spin avoids a kernel
    // futex sleep plus a producer-side futex wake for those hand-offs.
    if (timeout_output == 0 || timeout_microseconds >= 50u) {
        for (std::uint32_t spin = 0; spin < 192u; ++spin) {
#if defined(__aarch64__) || defined(__arm__)
            __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause");
#else
            std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
            if (try_consume(bits)) {
                result = publish_result(bits)
                    ? 0
                    : OrbisError(MemoryFault);
                return true;
            }
        }
    }

    std::unique_lock lock{state->mutex};
    state->waiting_patterns.fetch_or(
        pattern, std::memory_order_release);
    state->waiting_threads.fetch_add(
        1, std::memory_order_acq_rel);
    const auto release_waiter = [&] {
        const auto previous = state->waiting_threads.fetch_sub(
            1, std::memory_order_acq_rel);
        if (previous == 1) {
            state->waiting_patterns.store(
                0, std::memory_order_release);
        }
    };
    if (!try_consume(bits)) {
        if (timeout_output != 0) {
            const auto ready = state->condition.wait_for(
                lock,
                std::chrono::microseconds(timeout_microseconds),
                [&] {
                    return state->deleted.load(
                               std::memory_order_acquire) ||
                           satisfied(state->bits.load(
                               std::memory_order_acquire)) ||
                           g_runtime.shutdown_requested.load(
                               std::memory_order_acquire);
                });
            if (!ready || !try_consume(bits)) {
                release_waiter();
                lock.unlock();
                const std::uint32_t zero{};
                const bool wrote_timeout =
                    TryWriteGuestValue(timeout_output, zero);
                result = wrote_timeout && publish_result(bits)
                    ? OrbisError(TimedOut)
                    : OrbisError(MemoryFault);
                return true;
            }
        } else {
            while (!state->deleted.load(
                       std::memory_order_acquire) &&
                   !try_consume(bits) &&
                   !g_runtime.shutdown_requested.load(
                       std::memory_order_acquire)) {
                state->condition.wait_for(
                    lock, std::chrono::milliseconds(10));
            }
        }
    }
    if (state->deleted.load(std::memory_order_acquire) ||
        !satisfied(bits)) {
        release_waiter();
        result = OrbisError(NotFound);
        return true;
    }
    release_waiter();
    lock.unlock();
    result = publish_result(bits)
        ? 0
        : OrbisError(MemoryFault);
    return true;
}

std::uint64_t CurrentThreadTlsScratch(
    const std::uint64_t offset) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto* const thread = CurrentGuestThreadLocked();
    return thread != nullptr ? thread->fs_base + offset : 0;
}

bool TryCreateGuestPthread(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    const auto output_address = request.integer_arguments[0];
    const auto entry_address = request.integer_arguments[2];
    const auto argument = request.integer_arguments[3];
    std::string thread_name;
    (void)TryReadGuestCString(
        request.integer_arguments[4], 128, thread_name);
    std::uint64_t owner_handle{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (const auto* const thread =
                CurrentGuestThreadLocked()) {
            owner_handle = thread->program_handle;
        }
    }
    if (owner_handle == 0 || entry_address == 0) {
        result = OrbisError(UINT32_C(0x80020003));
        return true;
    }

    Lsx4Ps5GuestThreadContext thread_context{};
    const auto create_status =
        executor_lsx4_ps5_runtime_create_thread_context(
            owner_handle, &thread_context);
    if (create_status != 0) {
        result = OrbisError(UINT32_C(0x80020023));
        return true;
    }
    const auto pthread_handle =
        GuestPthreadHandleForRuntimeThread(thread_context.handle);
    if (pthread_handle == 0 ||
        (output_address != 0 &&
         !TryWriteGuestU64(output_address, pthread_handle))) {
        (void)executor_lsx4_ps5_runtime_destroy_thread_context(
            thread_context.handle);
        result = OrbisError(UINT32_C(0x80020101));
        return true;
    }

    Lsx4Ps5GuestEntry entry{};
    entry.address = entry_address;
    entry.argument_count = 1;
    entry.arguments[0] = argument;
    entry.fs_base = thread_context.fs_base;
    if (Ps5DiagnosticFaultProbeEnabled()) {
        std::array<std::uint64_t, 4> argument_words{};
        const auto has_argument_words =
            argument != 0 &&
            TryReadGuestBytes(
                argument, argument_words.data(),
                sizeof(argument_words));
        std::fprintf(
            stderr,
            "PS5_GUEST_THREAD_CREATE handle=0x%llx entry=0x%llx "
            "arg=0x%llx name=%s arg_words=%s"
            "0x%llx,0x%llx,0x%llx,0x%llx\n",
            static_cast<unsigned long long>(thread_context.handle),
            static_cast<unsigned long long>(entry_address),
            static_cast<unsigned long long>(argument),
            thread_name.empty() ? "<unnamed>" : thread_name.c_str(),
            has_argument_words ? "" : "<unmapped>",
            static_cast<unsigned long long>(argument_words[0]),
            static_cast<unsigned long long>(argument_words[1]),
            static_cast<unsigned long long>(argument_words[2]),
            static_cast<unsigned long long>(argument_words[3]));
        std::fflush(stderr);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-THREAD",
            "create handle=0x%llx entry=0x%llx arg=0x%llx "
            "name=%s words=0x%llx,0x%llx,0x%llx,0x%llx",
            static_cast<unsigned long long>(thread_context.handle),
            static_cast<unsigned long long>(entry_address),
            static_cast<unsigned long long>(argument),
            thread_name.empty() ? "<unnamed>" : thread_name.c_str(),
            static_cast<unsigned long long>(argument_words[0]),
            static_cast<unsigned long long>(argument_words[1]),
            static_cast<unsigned long long>(argument_words[2]),
            static_cast<unsigned long long>(argument_words[3]));
#endif
    }
    try {
        std::unique_lock lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            lock.unlock();
            (void)executor_lsx4_ps5_runtime_destroy_thread_context(
                thread_context.handle);
            result = OrbisError(UINT32_C(0x80020023));
            return true;
        }
        g_runtime.guest_host_threads.emplace_back(
            [entry, thread_handle = thread_context.handle,
             thread_name = std::move(thread_name)]() {
                if (Ps5DiagnosticFaultProbeEnabled()) {
                    std::fprintf(
                        stderr,
                        "PS5_GUEST_THREAD_START handle=0x%llx "
                        "entry=0x%llx arg=0x%llx fs=0x%llx\n",
                        static_cast<unsigned long long>(thread_handle),
                        static_cast<unsigned long long>(entry.address),
                        static_cast<unsigned long long>(
                            entry.arguments[0]),
                        static_cast<unsigned long long>(entry.fs_base));
                    std::fflush(stderr);
#ifdef __ANDROID__
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5-THREAD",
                        "start handle=0x%llx entry=0x%llx "
                        "arg=0x%llx fs=0x%llx name=%s",
                        static_cast<unsigned long long>(thread_handle),
                        static_cast<unsigned long long>(entry.address),
                        static_cast<unsigned long long>(
                            entry.arguments[0]),
                        static_cast<unsigned long long>(entry.fs_base),
                        thread_name.empty()
                            ? "<unnamed>" : thread_name.c_str());
#endif
                }
                Lsx4Ps5GuestResult thread_result{};
                (void)executor_lsx4_ps5_runtime_execute(
                    &entry, &thread_result);
                if (Ps5DiagnosticFaultProbeEnabled()) {
                    const std::string runtime_status =
                        executor_lsx4_ps5_runtime_status();
                    std::fprintf(
                        stderr,
                        "PS5_GUEST_THREAD_END handle=0x%llx "
                        "entry=0x%llx status=%d value=0x%llx "
                        "runtime=%s\n",
                        static_cast<unsigned long long>(thread_handle),
                        static_cast<unsigned long long>(entry.address),
                        thread_result.status,
                        static_cast<unsigned long long>(
                            thread_result.value),
                        runtime_status.c_str());
                    std::fflush(stderr);
#ifdef __ANDROID__
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4-PS5-THREAD",
                        "end handle=0x%llx entry=0x%llx status=%d "
                        "value=0x%llx name=%s",
                        static_cast<unsigned long long>(thread_handle),
                        static_cast<unsigned long long>(entry.address),
                        thread_result.status,
                        static_cast<unsigned long long>(
                            thread_result.value),
                        thread_name.empty()
                            ? "<unnamed>" : thread_name.c_str());
#endif
                }
                (void)executor_lsx4_ps5_runtime_destroy_thread_context(
                    thread_handle);
            });
    } catch (...) {
        (void)executor_lsx4_ps5_runtime_destroy_thread_context(
            thread_context.handle);
        result = OrbisError(UINT32_C(0x80020023));
        return true;
    }
    result = 0;
    return true;
}

std::string HleSymbol(const std::uint64_t thunk) {
    struct ThreadCache {
        std::uint64_t generation{};
        std::unordered_map<std::uint64_t, std::string> symbols;
    };
    thread_local ThreadCache cache;
    const auto generation =
        g_hle_binding_generation.load(std::memory_order_acquire);
    if (cache.generation != generation) {
        cache.symbols.clear();
        cache.generation = generation;
    }
    if (const auto cached = cache.symbols.find(thunk);
        cached != cache.symbols.end()) {
        return cached->second;
    }
    const std::lock_guard lock{g_runtime.mutex};
    const auto found = g_runtime.hle_bindings.find(thunk);
    auto symbol = found != g_runtime.hle_bindings.end()
        ? found->second.symbol
        : std::string{};
    cache.symbols.emplace(thunk, symbol);
    return symbol;
}

std::uint64_t OrbisError(const std::uint32_t value) noexcept {
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(static_cast<std::int32_t>(value)));
}

bool TryAllocateDirectMemory(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto TryAgain = UINT32_C(0x80020023);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto search_start_raw =
        static_cast<std::int64_t>(request.integer_arguments[0]);
    const auto search_end_raw =
        static_cast<std::int64_t>(request.integer_arguments[1]);
    const auto byte_count = request.integer_arguments[2];
    auto alignment = request.integer_arguments[3];
    const auto memory_type =
        static_cast<std::int32_t>(request.integer_arguments[4]);
    const auto output_address = request.integer_arguments[5];
#ifdef __ANDROID__
    const auto report_rejection = [&](const char* const reason) {
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4-PS5-DIRECT",
            "allocate rejected reason=%s start=0x%llx end=0x%llx "
            "bytes=0x%llx alignment=0x%llx type=%d out=0x%llx",
            reason,
            static_cast<unsigned long long>(search_start_raw),
            static_cast<unsigned long long>(search_end_raw),
            static_cast<unsigned long long>(byte_count),
            static_cast<unsigned long long>(alignment), memory_type,
            static_cast<unsigned long long>(output_address));
    };
#else
    const auto report_rejection = [](const char*) {};
#endif
    if (byte_count == 0 || byte_count > Ps5DirectMemorySize ||
        output_address == 0) {
        report_rejection("request-shape");
        result = OrbisError(InvalidArgument);
        return true;
    }
    if (alignment == 0) {
        alignment = LSX4_PS5_GUEST_PAGE_SIZE;
    }
    if (!IsPowerOfTwo(alignment) ||
        alignment < LSX4_PS5_GUEST_PAGE_SIZE) {
        report_rejection("alignment");
        result = OrbisError(InvalidArgument);
        return true;
    }

    auto search_start =
        search_start_raw < 0 ? UINT64_C(0)
                             : static_cast<std::uint64_t>(search_start_raw);
    auto search_end =
        search_end_raw <= 0
            ? Ps5DirectMemorySize
            : std::min(
                  static_cast<std::uint64_t>(search_end_raw),
                  Ps5DirectMemorySize);
    if (search_start >= search_end) {
        search_start = 0;
    }

    const std::lock_guard lock{g_runtime.mutex};
    if (!g_runtime.initialized) {
        report_rejection("uninitialized");
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::uint64_t candidate{};
    if (!AlignUp(
            search_start, alignment, candidate)) {
        result = OrbisError(TryAgain);
        return true;
    }
    while (candidate < search_end &&
           byte_count <= search_end - candidate) {
        const auto conflict = std::ranges::find_if(
            g_runtime.direct_allocations,
            [&](const DirectAllocation& allocation) {
                return candidate < allocation.offset +
                                       allocation.byte_count &&
                       allocation.offset < candidate + byte_count;
            });
        if (conflict == g_runtime.direct_allocations.end()) {
            break;
        }
        if (!AlignUp(
                conflict->offset + conflict->byte_count,
                alignment, candidate)) {
            candidate = search_end;
            break;
        }
    }
    if (candidate >= search_end ||
        byte_count > search_end - candidate) {
        report_rejection("exhausted");
        result = OrbisError(TryAgain);
        return true;
    }
    if (!GuestWriteU64Locked(output_address, candidate)) {
        report_rejection("output-write");
        result = OrbisError(MemoryFault);
        return true;
    }
    g_runtime.direct_allocations.push_back({
        .offset = candidate,
        .byte_count = byte_count,
        .alignment = alignment,
        .memory_type = memory_type,
    });
    g_runtime.next_direct_offset =
        std::max(g_runtime.next_direct_offset, candidate + byte_count);
    if (Ps5DiagnosticFaultProbeEnabled()) {
        std::fprintf(
            stderr,
            "PS5_DIRECT_ALLOC offset=0x%llx size=0x%llx "
            "alignment=0x%llx type=0x%x output=0x%llx\n",
            static_cast<unsigned long long>(candidate),
            static_cast<unsigned long long>(byte_count),
            static_cast<unsigned long long>(alignment),
            static_cast<unsigned int>(memory_type),
            static_cast<unsigned long long>(output_address));
        std::fflush(stderr);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "direct alloc offset=0x%llx size=0x%llx "
            "alignment=0x%llx type=0x%x output=0x%llx",
            static_cast<unsigned long long>(candidate),
            static_cast<unsigned long long>(byte_count),
            static_cast<unsigned long long>(alignment),
            static_cast<unsigned int>(memory_type),
            static_cast<unsigned long long>(output_address));
#endif
    }
    result = 0;
    return true;
}

bool TryAvailableDirectMemorySize(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto OutOfMemory = UINT32_C(0x8002000c);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto search_start =
        std::min(request.integer_arguments[0], Ps5DirectMemorySize);
    const auto search_end =
        std::min(request.integer_arguments[1], Ps5DirectMemorySize);
    const auto alignment = request.integer_arguments[2];
    const auto output_address = request.integer_arguments[3];
    const auto output_size = request.integer_arguments[4];
    if (search_start >= search_end || output_address == 0 ||
        output_size == 0 ||
        (alignment != 0 && !IsPowerOfTwo(alignment))) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    const std::lock_guard lock{g_runtime.mutex};
    auto allocations = g_runtime.direct_allocations;
    std::ranges::sort(
        allocations, {}, &DirectAllocation::offset);
    std::uint64_t best_address{};
    std::uint64_t best_size{};
    std::uint64_t cursor = search_start;
    const auto consider_gap =
        [&](const std::uint64_t gap_begin,
            const std::uint64_t gap_end) {
            std::uint64_t aligned_begin = gap_begin;
            if (alignment != 0 &&
                !AlignUp(gap_begin, alignment, aligned_begin)) {
                return;
            }
            if (aligned_begin < gap_end &&
                gap_end - aligned_begin > best_size) {
                best_address = aligned_begin;
                best_size = gap_end - aligned_begin;
            }
        };
    for (const auto& allocation : allocations) {
        const auto allocation_end =
            allocation.offset + allocation.byte_count;
        if (allocation_end <= cursor ||
            allocation.offset >= search_end) {
            continue;
        }
        consider_gap(
            cursor, std::min(allocation.offset, search_end));
        cursor = std::max(
            cursor, std::min(allocation_end, search_end));
        if (cursor >= search_end) {
            break;
        }
    }
    consider_gap(cursor, search_end);
    if (best_size == 0) {
        result = OrbisError(OutOfMemory);
        return true;
    }
    if (!GuestWriteU64Locked(output_address, best_address) ||
        !GuestWriteU64Locked(output_size, best_size)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    result = 0;
    return true;
}

bool TryReserveVirtualRange(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto in_out_address = request.integer_arguments[0];
    const auto byte_count = request.integer_arguments[1];
    const auto flags =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    auto alignment = request.integer_arguments[3];
    if (in_out_address == 0 || byte_count == 0 ||
        (byte_count & (LSX4_PS5_GUEST_PAGE_SIZE - 1u)) != 0 ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    if (alignment == 0) {
        alignment = LSX4_PS5_GUEST_PAGE_SIZE;
    }
    if (!IsPowerOfTwo(alignment) ||
        alignment < LSX4_PS5_GUEST_PAGE_SIZE) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t requested_address{};
    if (!GuestReadU64Locked(in_out_address, requested_address)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    constexpr std::uint32_t FixedMapping = 0x10u;
    const bool fixed_mapping = (flags & FixedMapping) != 0;
    if (fixed_mapping && requested_address != 0 &&
        g_runtime.mappings.Find(
            requested_address, byte_count,
            LSX4_PS5_GUEST_READ) != nullptr) {
        result = 0;
        return true;
    }
    const auto first_hint =
        requested_address != 0
            ? requested_address
            : g_runtime.next_flexible_map_hint;
    const auto stride = std::max<std::uint64_t>(
        alignment, byte_count);
    auto* const allocation = AllocateLowGuestRegionLocked(
        first_hint, stride, static_cast<std::size_t>(byte_count),
        alignment, fixed_mapping ? 1u : 4096u);
    if (allocation == nullptr) {
        result = OrbisError(UINT32_C(0x80020023));
        return true;
    }
    const auto address =
        reinterpret_cast<std::uint64_t>(allocation);
    if (!RegisterMappingLocked(
            address, byte_count,
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE,
            "ps5-flexible-memory", true) ||
        !GuestWriteU64Locked(in_out_address, address)) {
        (void)g_runtime.mappings.EraseExact(
            address, byte_count, true);
        munmap(allocation, static_cast<std::size_t>(byte_count));
        result = OrbisError(MemoryFault);
        return true;
    }
    g_runtime.owned_guest_mappings.push_back({
        .allocation = allocation,
        .byte_count = static_cast<std::size_t>(byte_count),
    });
    if (!fixed_mapping) {
        g_runtime.next_flexible_map_hint =
            std::max(
                g_runtime.next_flexible_map_hint,
                address + byte_count);
    }
    result = 0;
    return true;
}

bool TryMapFlexibleMemory(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto in_out_address = request.integer_arguments[0];
    const auto byte_count = request.integer_arguments[1];
    const auto orbis_protection =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    if (in_out_address == 0 || byte_count == 0 ||
        (byte_count & (LSX4_PS5_GUEST_PAGE_SIZE - 1u)) != 0 ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t requested_address{};
    if (!GuestReadU64Locked(
            in_out_address, requested_address)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    std::uint32_t guest_protection{};
    int host_protection{};
    if ((orbis_protection & 0x01u) != 0) {
        guest_protection |= LSX4_PS5_GUEST_READ;
        host_protection |= PROT_READ;
    }
    if ((orbis_protection & 0x02u) != 0) {
        guest_protection |=
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE;
        host_protection |= PROT_READ | PROT_WRITE;
    }
    if ((orbis_protection & 0x04u) != 0) {
        guest_protection |=
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_EXECUTE;
        host_protection |= PROT_READ | PROT_EXEC;
    }
    if (guest_protection == 0) {
        guest_protection =
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE;
        host_protection = PROT_READ | PROT_WRITE;
    }
    if (requested_address != 0 &&
        g_runtime.mappings.Find(
            requested_address, byte_count,
            guest_protection) != nullptr) {
        result = 0;
        return true;
    }
    const auto first_hint =
        requested_address != 0
            ? requested_address
            : g_runtime.next_flexible_map_hint;
    auto* const allocation = AllocateLowGuestRegionLocked(
        first_hint,
        std::max<std::uint64_t>(
            LSX4_PS5_GUEST_PAGE_SIZE, byte_count),
        static_cast<std::size_t>(byte_count),
        LSX4_PS5_GUEST_PAGE_SIZE,
        4096u);
    if (allocation == nullptr) {
        result = OrbisError(UINT32_C(0x80020023));
        return true;
    }
    const auto address =
        reinterpret_cast<std::uint64_t>(allocation);
    if (mprotect(
            allocation, static_cast<std::size_t>(byte_count),
            host_protection) != 0 ||
        !RegisterMappingLocked(
            address, byte_count, guest_protection,
            "ps5-flexible-memory", true) ||
        !GuestWriteU64Locked(in_out_address, address)) {
        (void)g_runtime.mappings.EraseExact(
            address, byte_count, true);
        munmap(allocation, static_cast<std::size_t>(byte_count));
        result = OrbisError(MemoryFault);
        return true;
    }
    g_runtime.owned_guest_mappings.push_back({
        .allocation = allocation,
        .byte_count = static_cast<std::size_t>(byte_count),
    });
    g_runtime.next_flexible_map_hint =
        std::max(
            g_runtime.next_flexible_map_hint,
            address + byte_count);
    result = 0;
    return true;
}

bool TryMapDirectMemory(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto NotFound = UINT32_C(0x80020002);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    const auto in_out_address = request.integer_arguments[0];
    const auto requested_size = request.integer_arguments[1];
    const auto orbis_protection =
        static_cast<std::uint32_t>(request.integer_arguments[2]);
    const auto direct_offset = request.integer_arguments[4];
    auto alignment = request.integer_arguments[5];
#ifdef __ANDROID__
    if (Ps5DiagnosticFaultProbeEnabled()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "direct map call inout=0x%llx size=0x%llx "
            "prot=0x%x flags=0x%llx direct=0x%llx "
            "alignment=0x%llx",
            static_cast<unsigned long long>(in_out_address),
            static_cast<unsigned long long>(requested_size),
            orbis_protection,
            static_cast<unsigned long long>(
                request.integer_arguments[3]),
            static_cast<unsigned long long>(direct_offset),
            static_cast<unsigned long long>(alignment));
    }
#endif
    if (in_out_address == 0 || requested_size == 0 ||
        requested_size > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    if (alignment == 0) {
        alignment = LSX4_PS5_GUEST_PAGE_SIZE;
    }
    if (!IsPowerOfTwo(alignment) ||
        alignment < LSX4_PS5_GUEST_PAGE_SIZE) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    std::uint64_t mapped_size_u64{};
    if (!AlignUp(
            requested_size, LSX4_PS5_GUEST_PAGE_SIZE,
            mapped_size_u64) ||
        mapped_size_u64 > std::numeric_limits<std::size_t>::max()) {
        result = OrbisError(InvalidArgument);
        return true;
    }
    const auto mapped_size = static_cast<std::size_t>(mapped_size_u64);

    const std::lock_guard lock{g_runtime.mutex};
    std::uint64_t requested_address{};
    if (!GuestReadU64Locked(in_out_address, requested_address)) {
        result = OrbisError(MemoryFault);
        return true;
    }
    if (direct_offset >= Ps5DirectMemorySize ||
        requested_size > Ps5DirectMemorySize - direct_offset) {
        result = OrbisError(InvalidArgument);
        return true;
    }

    const auto first_hint =
        requested_address != 0
            ? requested_address
            : g_runtime.next_direct_map_hint;
    auto* const allocation = AllocateLowGuestRegionLocked(
        first_hint,
        std::max(Ps5DirectMapStride, mapped_size_u64),
        mapped_size, alignment);
    if (allocation == nullptr) {
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled()) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4-PS5",
                "direct map reserve failed errno=%d direct=0x%llx "
                "size=0x%llx requested=0x%llx hint=0x%llx "
                "alignment=0x%llx",
                errno,
                static_cast<unsigned long long>(direct_offset),
                static_cast<unsigned long long>(mapped_size_u64),
                static_cast<unsigned long long>(requested_address),
                static_cast<unsigned long long>(first_hint),
                static_cast<unsigned long long>(alignment));
        }
#endif
        result = OrbisError(NotFound);
        return true;
    }
    const auto mapped_address =
        reinterpret_cast<std::uint64_t>(allocation);
    std::uint32_t guest_protection{};
    int host_protection{};
    if ((orbis_protection & 0x01u) != 0) {
        guest_protection |= LSX4_PS5_GUEST_READ;
        host_protection |= PROT_READ;
    }
    if ((orbis_protection & 0x02u) != 0) {
        guest_protection |=
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE;
        host_protection |= PROT_READ | PROT_WRITE;
    }
    if ((orbis_protection & 0x04u) != 0) {
        guest_protection |=
            LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_EXECUTE;
        host_protection |= PROT_READ | PROT_EXEC;
    }
    if (guest_protection == 0) {
        guest_protection = LSX4_PS5_GUEST_READ;
        host_protection = PROT_READ;
    }
    if (!RegisterMappingLocked(
            mapped_address, mapped_size_u64, guest_protection,
            "ps5-direct-memory", true) ||
        mprotect(allocation, mapped_size, host_protection) != 0) {
        (void)g_runtime.mappings.EraseExact(
            mapped_address, mapped_size_u64, true);
        munmap(allocation, mapped_size);
        result = OrbisError(NotFound);
        return true;
    }
    if (!GuestWriteU64Locked(in_out_address, mapped_address)) {
        (void)g_runtime.mappings.EraseExact(
            mapped_address, mapped_size_u64, true);
        munmap(allocation, mapped_size);
        result = OrbisError(MemoryFault);
        return true;
    }
    g_runtime.owned_guest_mappings.push_back({
        .allocation = allocation,
        .byte_count = mapped_size,
    });
    g_runtime.next_direct_map_hint =
        mapped_address + mapped_size_u64;
    if (Ps5DiagnosticFaultProbeEnabled()) {
        std::fprintf(
            stderr,
            "PS5_DIRECT_MAP direct=0x%llx size=0x%llx "
            "requested=0x%llx mapped=0x%llx next=0x%llx\n",
            static_cast<unsigned long long>(direct_offset),
            static_cast<unsigned long long>(mapped_size_u64),
            static_cast<unsigned long long>(requested_address),
            static_cast<unsigned long long>(mapped_address),
            static_cast<unsigned long long>(
                g_runtime.next_direct_map_hint));
        std::fflush(stderr);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "direct map direct=0x%llx size=0x%llx "
            "requested=0x%llx mapped=0x%llx next=0x%llx",
            static_cast<unsigned long long>(direct_offset),
            static_cast<unsigned long long>(mapped_size_u64),
            static_cast<unsigned long long>(requested_address),
            static_cast<unsigned long long>(mapped_address),
            static_cast<unsigned long long>(
                g_runtime.next_direct_map_hint));
#endif
    }
    result = 0;
    return true;
}

bool TryKernelBatchMap(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    constexpr std::uint64_t EntrySize = 0x20;
    constexpr std::uint64_t FixedMap = 0x10;
    const auto entries = request.integer_arguments[0];
    const auto entry_count =
        static_cast<std::int32_t>(request.integer_arguments[1]);
    const auto processed_output = request.integer_arguments[2];
    std::int32_t processed{};
    result = 0;

    if (entry_count < 0 || (entry_count != 0 && entries == 0)) {
        result = OrbisError(InvalidArgument);
    }
    for (std::int32_t index = 0;
         result == 0 && index < entry_count; ++index) {
        std::array<std::uint8_t, EntrySize> bytes{};
        const auto entry_address =
            entries + static_cast<std::uint64_t>(index) * EntrySize;
        if (entry_address < entries ||
            !TryReadGuestBytes(
                entry_address, bytes.data(), bytes.size())) {
            result = OrbisError(MemoryFault);
            break;
        }
        std::uint64_t start{};
        std::uint64_t offset{};
        std::uint64_t length{};
        std::uint32_t operation{};
        std::memcpy(&start, bytes.data() + 0x00, sizeof(start));
        std::memcpy(&offset, bytes.data() + 0x08, sizeof(offset));
        std::memcpy(&length, bytes.data() + 0x10, sizeof(length));
        const auto protection = bytes[0x18];
        const auto memory_type = bytes[0x19];
        std::memcpy(
            &operation, bytes.data() + 0x1c, sizeof(operation));
        if (length == 0 || operation > 4) {
            result = OrbisError(InvalidArgument);
            break;
        }
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> batch_map_logs{};
        if (batch_map_logs.fetch_add(
                1, std::memory_order_relaxed) < 8) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "batch map op=%u start=0x%llx offset=0x%llx "
                "length=0x%llx prot=0x%x type=0x%x",
                operation,
                static_cast<unsigned long long>(start),
                static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(length),
                static_cast<unsigned int>(protection),
                static_cast<unsigned int>(memory_type));
        }
#endif
        Lsx4::Translation::HleBridgeRequest translated{};
        translated.integer_arguments[0] = entry_address;
        translated.integer_arguments[1] = length;
        translated.integer_arguments[2] = protection;
        translated.integer_arguments[3] = FixedMap;
        switch (operation) {
        case 0:
            translated.integer_arguments[4] = offset;
            translated.integer_arguments[5] = 0;
            (void)TryMapDirectMemory(translated, result);
            break;
        case 1: {
            const auto page_size = HostPageSize();
            if (start == 0 || length >
                    std::numeric_limits<std::size_t>::max() ||
                (start & (page_size - 1u)) != 0 ||
                (length & (page_size - 1u)) != 0) {
                result = OrbisError(InvalidArgument);
                break;
            }
            std::uint8_t* allocation{};
            {
                const std::lock_guard lock{g_runtime.mutex};
                const auto found = std::ranges::find(
                    g_runtime.owned_guest_mappings,
                    start,
                    [](const OwnedGuestMapping& mapping) {
                        return reinterpret_cast<std::uint64_t>(
                            mapping.allocation);
                    });
                if (found !=
                        g_runtime.owned_guest_mappings.end() &&
                    found->byte_count ==
                        static_cast<std::size_t>(length)) {
                    allocation = found->allocation;
                    g_runtime.owned_guest_mappings.erase(found);
                    (void)g_runtime.mappings.EraseExact(
                        start, length, true);
                }
            }
            if (allocation != nullptr) {
                (void)munmap(
                    allocation, static_cast<std::size_t>(length));
            }
            result = 0;
            break;
        }
        case 2:
        case 4: {
            std::uint32_t guest_protection{};
            int host_protection{};
            if ((protection & 0x01u) != 0) {
                guest_protection |= LSX4_PS5_GUEST_READ;
                host_protection |= PROT_READ;
            }
            if ((protection & 0x02u) != 0) {
                guest_protection |=
                    LSX4_PS5_GUEST_READ |
                    LSX4_PS5_GUEST_WRITE;
                host_protection |= PROT_READ | PROT_WRITE;
            }
            if ((protection & 0x04u) != 0) {
                guest_protection |=
                    LSX4_PS5_GUEST_READ |
                    LSX4_PS5_GUEST_EXECUTE;
                host_protection |= PROT_READ | PROT_EXEC;
            }
            if (guest_protection == 0) {
                guest_protection = LSX4_PS5_GUEST_READ;
                host_protection = PROT_READ;
            }
            if (start == 0 ||
                length > std::numeric_limits<std::size_t>::max() ||
                mprotect(
                    reinterpret_cast<void*>(start),
                    static_cast<std::size_t>(length),
                    host_protection) != 0) {
                result = OrbisError(InvalidArgument);
                break;
            }
            {
                const std::lock_guard lock{g_runtime.mutex};
                (void)RegisterMappingLocked(
                    start, length, guest_protection,
                    operation == 4
                        ? "ps5-type-protected-memory"
                        : "ps5-protected-memory",
                    true);
            }
            result = 0;
            break;
        }
        case 3:
            (void)TryMapFlexibleMemory(translated, result);
            break;
        default:
            result = OrbisError(InvalidArgument);
            break;
        }
        if (result == 0) {
            ++processed;
        }
    }
    if (processed_output != 0 &&
        !TryWriteGuestValue(processed_output, processed)) {
        result = OrbisError(MemoryFault);
    }
    return true;
}

std::uint32_t ReadBigEndianU32(const std::uint8_t* const bytes) noexcept {
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

bool TryCreateNgs2Handle(
    const std::uint32_t type, const std::uint64_t owner,
    std::uint64_t& handle) {
    if (!TryAllocateLibcHeap(0x20, 16, handle) || handle == 0) {
        return false;
    }
    std::array<std::uint8_t, 0x20> storage{};
    std::memcpy(storage.data(), &handle, sizeof(handle));
    std::memcpy(storage.data() + 8, &owner, sizeof(owner));
    const std::uint32_t version = 1;
    std::memcpy(storage.data() + 16, &version, sizeof(version));
    std::memcpy(storage.data() + 24, &type, sizeof(type));
    if (TryWriteGuestBytes(handle, storage.data(), storage.size())) {
        return true;
    }
    FreeLibcHeap(handle);
    handle = 0;
    return false;
}

void RemoveNgs2RackLocked(const std::uint64_t rack_handle) {
    g_ngs2_racks.erase(rack_handle);
    for (auto voice = g_ngs2_voices.begin();
         voice != g_ngs2_voices.end();) {
        if (voice->second.rack_handle == rack_handle) {
            voice = g_ngs2_voices.erase(voice);
        } else {
            ++voice;
        }
    }
}

bool TryDecodeNgs2Vag(
    const std::uint64_t address, std::vector<std::int16_t>& samples,
    std::uint32_t& sample_rate, std::int32_t& loop_start,
    std::int32_t& loop_end) {
    constexpr std::size_t HeaderSize = 0x30;
    constexpr std::size_t MaximumDataSize = 8 * 1024 * 1024;
    std::array<std::uint8_t, HeaderSize> header{};
    if (!TryReadGuestBytes(address, header.data(), header.size()) ||
        std::memcmp(header.data(), "VAGp", 4) != 0) {
        return false;
    }
    const auto declared_size =
        std::min<std::size_t>(ReadBigEndianU32(header.data() + 0x0c),
                              MaximumDataSize);
    const auto frame_bytes = declared_size - (declared_size % 16);
    if (frame_bytes == 0) {
        return false;
    }
    std::vector<std::uint8_t> frames(frame_bytes);
    if (!TryReadGuestBytes(
            address + HeaderSize, frames.data(), frames.size())) {
        return false;
    }

    sample_rate = ReadBigEndianU32(header.data() + 0x10);
    if (sample_rate == 0) {
        sample_rate = 48000;
    }
    samples.clear();
    samples.reserve((frame_bytes / 16) * 28);
    loop_start = -1;
    loop_end = -1;
    constexpr std::array<std::int32_t, 5> Coeff0{0, 60, 115, 98, 122};
    constexpr std::array<std::int32_t, 5> Coeff1{0, 0, -52, -55, -60};
    std::int32_t history1{};
    std::int32_t history2{};
    bool ended{};
    for (std::size_t offset = 0;
         offset + 16 <= frames.size() && !ended; offset += 16) {
        const auto predictor = frames[offset];
        const auto shift = predictor & 0x0f;
        auto filter = (predictor >> 4) & 0x0f;
        if (filter > 4) {
            filter = 0;
        }
        const auto flags = frames[offset + 1];
        if (flags == 3) {
            loop_start = static_cast<std::int32_t>(samples.size());
        }
        for (std::size_t index = 0; index != 14; ++index) {
            const auto packed = frames[offset + 2 + index];
            for (std::uint32_t nibble_index = 0;
                 nibble_index != 2; ++nibble_index) {
                const auto nibble = static_cast<std::int32_t>(
                    nibble_index == 0 ? packed & 0x0f : packed >> 4);
                auto decoded = static_cast<std::int32_t>(
                    static_cast<std::int16_t>(nibble << 12));
                decoded >>= shift;
                decoded +=
                    (history1 * Coeff0[filter] +
                     history2 * Coeff1[filter]) >>
                    6;
                decoded = std::clamp(
                    decoded,
                    static_cast<std::int32_t>(
                        std::numeric_limits<std::int16_t>::min()),
                    static_cast<std::int32_t>(
                        std::numeric_limits<std::int16_t>::max()));
                samples.push_back(static_cast<std::int16_t>(decoded));
                history2 = history1;
                history1 = decoded;
            }
        }
        if (flags == 6) {
            loop_end = static_cast<std::int32_t>(samples.size());
        } else if (flags == 1 || flags == 7) {
            ended = true;
        }
    }
    if (loop_start >= 0 && loop_end <= loop_start) {
        loop_end = static_cast<std::int32_t>(samples.size());
    }
    return !samples.empty();
}

bool ClearNgs2GuestBuffer(
    const std::uint64_t address, const std::uint64_t byte_count) {
    static constexpr std::array<std::uint8_t, 4096> Zeroes{};
    for (std::uint64_t offset = 0; offset < byte_count;) {
        const auto chunk = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                Zeroes.size(), byte_count - offset));
        if (!TryWriteGuestBytes(
                address + offset, Zeroes.data(), chunk)) {
            return false;
        }
        offset += chunk;
    }
    return true;
}

bool TryReadGameMakerTransform(
    const AgcDiagnosticDraw& draw,
    const std::uint64_t descriptor_table_address,
    std::array<float, 32>& transform,
    std::uint64_t& transform_address) {
    const auto is_projection_view =
        [](const std::array<float, 32>& candidate) {
            if (!std::ranges::all_of(
                    candidate,
                    [](const float value) {
                        return std::isfinite(value);
                    })) {
                return false;
            }
            // GameMaker supplies a conventional column-major perspective
            // projection followed by a view matrix. Validate the contract,
            // not title-specific values.
            return candidate[0] > 0.01f &&
                candidate[0] < 100.0f &&
                candidate[5] > 0.01f &&
                candidate[5] < 100.0f &&
                std::abs(candidate[11] + 1.0f) < 0.02f &&
                std::abs(candidate[15]) < 0.02f &&
                std::abs(candidate[31] - 1.0f) < 0.02f &&
                std::abs(candidate[16]) < 100.0f &&
                std::abs(candidate[21]) < 100.0f &&
                std::abs(candidate[30]) > 0.01f;
        };
    const auto try_address =
        [&](const std::uint64_t address) {
            if (address == 0u ||
                !TryReadGuestBytes(
                    address, transform.data(),
                    sizeof(transform)) ||
                !is_projection_view(transform)) {
                return false;
            }
            transform_address = address;
            return true;
        };

    std::vector<std::uint64_t> roots;
    const auto append_address =
        [&](const std::uint64_t address) {
            if (address != 0u &&
                !std::ranges::contains(roots, address)) {
                roots.push_back(address);
            }
        };
    append_address(descriptor_table_address);
    const auto append_user_data =
        [&](const auto& user_data) {
            for (std::size_t word = 0;
                 word + 1u < user_data.size(); ++word) {
                append_address(
                    static_cast<std::uint64_t>(user_data[word]) |
                    (static_cast<std::uint64_t>(
                         user_data[word + 1u] & 0xffffu) << 32u));
            }
        };
    append_user_data(draw.sh_export_user_data);
    append_user_data(draw.sh_gs_user_data);
    append_user_data(draw.sh_es_user_data);
    append_user_data(draw.sh_vs_user_data);

    for (const auto root : roots) {
        if (try_address(root)) {
            return true;
        }
        std::array<std::uint32_t, 128> table{};
        if (!TryReadGuestBytes(
                root, table.data(), sizeof(table))) {
            continue;
        }
        for (std::size_t word = 0;
             word + 1u < table.size(); ++word) {
            const auto address =
                static_cast<std::uint64_t>(table[word]) |
                (static_cast<std::uint64_t>(
                     table[word + 1u] & 0xffffu) << 32u);
            if (try_address(address)) {
                return true;
            }
        }
    }
    return false;
}

void ApplyNgs2VoiceParameters(
    const std::uint64_t voice_handle,
    const std::uint64_t parameter_list) {
    if (parameter_list == 0) {
        return;
    }
    auto parameter = parameter_list;
    for (std::uint32_t guard = 0; guard != 32; ++guard) {
        std::uint32_t size{};
        std::uint32_t id{};
        if (!TryReadGuestBytes(parameter, &size, sizeof(size)) ||
            !TryReadGuestBytes(parameter + 4, &id, sizeof(id))) {
            return;
        }
        if (id == UINT32_C(0x10000001)) {
            std::uint64_t data_address{};
            if (TryReadGuestBytes(
                    parameter + 8, &data_address,
                    sizeof(data_address)) &&
                data_address > 0x10000) {
                {
                    const std::lock_guard lock{g_ngs2_mutex};
                    const auto voice = g_ngs2_voices.find(voice_handle);
                    if (voice == g_ngs2_voices.end()) {
                        return;
                    }
                    if (voice->second.source_address == data_address &&
                        !voice->second.pcm.empty()) {
                        data_address = 0;
                    }
                }
                if (data_address != 0) {
                    std::vector<std::int16_t> samples;
                    std::uint32_t sample_rate{};
                    std::int32_t loop_start{};
                    std::int32_t loop_end{};
                    if (TryDecodeNgs2Vag(
                            data_address, samples, sample_rate,
                            loop_start, loop_end)) {
                        const std::lock_guard lock{g_ngs2_mutex};
                        if (auto voice =
                                g_ngs2_voices.find(voice_handle);
                            voice != g_ngs2_voices.end()) {
                            voice->second.pcm = std::move(samples);
                            voice->second.source_address = data_address;
                            voice->second.source_rate = sample_rate;
                            voice->second.position = 0;
                            voice->second.playing = true;
                            voice->second.loop_start = loop_start;
                            voice->second.loop_end =
                                loop_end > 0
                                    ? loop_end
                                    : static_cast<std::int32_t>(
                                          voice->second.pcm.size());
                        }
                    }
                }
            }
        } else if (id == UINT32_C(0x20010001)) {
            std::uint32_t level_bits{};
            if (TryReadGuestBytes(
                    parameter + 12, &level_bits,
                    sizeof(level_bits))) {
                float level{};
                std::memcpy(&level, &level_bits, sizeof(level));
                if (std::isfinite(level) && level >= 0.0f &&
                    level <= 8.0f) {
                    const std::lock_guard lock{g_ngs2_mutex};
                    if (auto voice =
                            g_ngs2_voices.find(voice_handle);
                        voice != g_ngs2_voices.end()) {
                        voice->second.gain = level;
                    }
                }
            }
        }
        if (size < 8 || size > 0x1000) {
            return;
        }
        parameter += (static_cast<std::uint64_t>(size) + 7) & ~7ull;
    }
}

bool TryHandleNgs2(
    const std::string& symbol,
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    constexpr auto InvalidArgument = UINT32_C(0x80020003);
    constexpr auto MemoryFault = UINT32_C(0x80020101);
    constexpr auto InvalidOutAddress = UINT32_C(0x804a0053);
    constexpr auto InvalidSystemHandle = UINT32_C(0x804a0230);
    constexpr auto InvalidRackHandle = UINT32_C(0x804a0261);
    constexpr auto InvalidVoiceHandle = UINT32_C(0x804a0300);

    if (symbol == "mPYgU4oYpuY" || symbol == "koBbCMvOKWw") {
        const auto output = request.integer_arguments[2];
        if (output == 0) {
            result = OrbisError(InvalidOutAddress);
            return true;
        }
        std::uint64_t handle{};
        if (!TryCreateNgs2Handle(1, 0, handle) ||
            !TryWriteGuestBytes(output, &handle, sizeof(handle))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        {
            const std::lock_guard lock{g_ngs2_mutex};
            g_ngs2_systems.emplace(handle, Ngs2SystemState{});
        }
        result = 0;
        return true;
    }
    if (symbol == "u-WrYDaJA3k") {
        const auto handle = request.integer_arguments[0];
        std::vector<std::uint64_t> racks;
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (g_ngs2_systems.erase(handle) == 0) {
                result = OrbisError(InvalidSystemHandle);
                return true;
            }
            for (const auto& [rack_handle, rack] : g_ngs2_racks) {
                if (rack.system_handle == handle) {
                    racks.push_back(rack_handle);
                }
            }
            for (const auto rack : racks) {
                RemoveNgs2RackLocked(rack);
            }
        }
        FreeLibcHeap(handle);
        result = 0;
        return true;
    }
    if (symbol == "U546k6orxQo" || symbol == "cLV4aiT9JpA") {
        const auto system = request.integer_arguments[0];
        const auto rack_id =
            static_cast<std::uint32_t>(request.integer_arguments[1]);
        const auto output = request.integer_arguments[4];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_systems.contains(system)) {
                result = OrbisError(InvalidSystemHandle);
                return true;
            }
        }
        if (output == 0) {
            result = OrbisError(InvalidOutAddress);
            return true;
        }
        std::uint64_t handle{};
        if (!TryCreateNgs2Handle(2, system, handle) ||
            !TryWriteGuestBytes(output, &handle, sizeof(handle))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        {
            const std::lock_guard lock{g_ngs2_mutex};
            g_ngs2_racks.emplace(
                handle, Ngs2RackState{system, rack_id});
        }
        result = 0;
        return true;
    }
    if (symbol == "lCqD7oycmIM") {
        const auto handle = request.integer_arguments[0];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_racks.contains(handle)) {
                result = OrbisError(InvalidRackHandle);
                return true;
            }
            RemoveNgs2RackLocked(handle);
        }
        FreeLibcHeap(handle);
        result = 0;
        return true;
    }
    if (symbol == "MwmHz8pAdAo") {
        const auto rack_handle = request.integer_arguments[0];
        const auto voice_index =
            static_cast<std::uint32_t>(request.integer_arguments[1]);
        const auto output = request.integer_arguments[2];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_racks.contains(rack_handle)) {
                result = OrbisError(InvalidRackHandle);
                return true;
            }
            for (const auto& [handle, voice] : g_ngs2_voices) {
                if (voice.rack_handle == rack_handle &&
                    voice.voice_index == voice_index) {
                    result = TryWriteGuestBytes(
                                 output, &handle, sizeof(handle))
                        ? 0
                        : OrbisError(MemoryFault);
                    return true;
                }
            }
        }
        if (output == 0) {
            result = OrbisError(InvalidOutAddress);
            return true;
        }
        std::uint64_t handle{};
        if (!TryCreateNgs2Handle(4, rack_handle, handle) ||
            !TryWriteGuestBytes(output, &handle, sizeof(handle))) {
            result = OrbisError(MemoryFault);
            return true;
        }
        {
            const std::lock_guard lock{g_ngs2_mutex};
            g_ngs2_voices.emplace(
                handle,
                Ngs2VoiceState{
                    .rack_handle = rack_handle,
                    .voice_index = voice_index,
                });
        }
        result = 0;
        return true;
    }
    if (symbol == "uu94irFOGpA" || symbol == "AbYvTOZ8Pts") {
        const auto voice_handle = request.integer_arguments[0];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_voices.contains(voice_handle)) {
                result = OrbisError(InvalidVoiceHandle);
                return true;
            }
        }
        ApplyNgs2VoiceParameters(
            voice_handle, request.integer_arguments[1]);
        result = 0;
        return true;
    }
    if (symbol == "i0VnXM-C9fc") {
        const auto system_handle = request.integer_arguments[0];
        const auto buffer_info = request.integer_arguments[1];
        const auto buffer_count =
            static_cast<std::uint32_t>(request.integer_arguments[2]);
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_systems.contains(system_handle)) {
                result = OrbisError(InvalidSystemHandle);
                return true;
            }
        }
        if (buffer_count != 0 && buffer_info == 0) {
            result = OrbisError(InvalidArgument);
            return true;
        }
        for (std::uint32_t index = 0; index != buffer_count; ++index) {
            const auto entry = buffer_info +
                static_cast<std::uint64_t>(index) * 0x18;
            std::uint64_t address{};
            std::uint64_t byte_count{};
            std::uint32_t channels{2};
            if (!TryReadGuestBytes(entry, &address, sizeof(address)) ||
                !TryReadGuestBytes(
                    entry + 8, &byte_count, sizeof(byte_count)) ||
                !TryReadGuestBytes(
                    entry + 20, &channels, sizeof(channels))) {
                result = OrbisError(MemoryFault);
                return true;
            }
            if (address == 0 || byte_count == 0) {
                continue;
            }
            if (byte_count > 16 * 1024 * 1024ull ||
                !ClearNgs2GuestBuffer(address, byte_count)) {
                result = OrbisError(MemoryFault);
                return true;
            }
            if (channels == 0 || channels > 8) {
                channels = 2;
            }
            std::uint32_t grain{256};
            {
                const std::lock_guard lock{g_ngs2_mutex};
                grain = g_ngs2_systems.at(system_handle).grain_samples;
            }
            const auto frames = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    grain, byte_count /
                               (channels * sizeof(float))));
            if (frames == 0) {
                continue;
            }
            thread_local std::vector<float> mix;
            mix.assign(frames * channels, 0.0f);
            bool mixed{};
            {
                const std::lock_guard lock{g_ngs2_mutex};
                for (auto& [voice_handle, voice] : g_ngs2_voices) {
                    const auto rack =
                        g_ngs2_racks.find(voice.rack_handle);
                    if (!voice.playing || voice.pcm.empty() ||
                        rack == g_ngs2_racks.end() ||
                        rack->second.system_handle != system_handle) {
                        continue;
                    }
                    const auto limit =
                        voice.loop_end > 0 &&
                                static_cast<std::size_t>(
                                    voice.loop_end) <= voice.pcm.size()
                            ? voice.loop_end
                            : static_cast<std::int32_t>(
                                  voice.pcm.size());
                    const auto step =
                        static_cast<double>(voice.source_rate) / 48000.0;
                    auto position = voice.position;
                    const auto gain = voice.gain / 32768.0f;
                    for (std::size_t frame = 0; frame != frames; ++frame) {
                        auto sample_index =
                            static_cast<std::int32_t>(position);
                        if (sample_index >= limit) {
                            if (voice.loop_start >= 0 &&
                                voice.loop_start < limit) {
                                position = voice.loop_start;
                                sample_index = voice.loop_start;
                            } else {
                                voice.playing = false;
                                break;
                            }
                        }
                        if (sample_index < 0 ||
                            static_cast<std::size_t>(sample_index) >=
                                voice.pcm.size()) {
                            voice.playing = false;
                            break;
                        }
                        const auto sample =
                            voice.pcm[sample_index] * gain;
                        const auto destination = frame * channels;
                        mix[destination] += sample;
                        if (channels > 1) {
                            mix[destination + 1] += sample;
                        }
                        position += step;
                    }
                    voice.position = position;
                    mixed = true;
                }
            }
            if (mixed) {
                for (auto& sample : mix) {
                    sample = std::clamp(sample, -1.0f, 1.0f);
                }
                if (!TryWriteGuestBytes(
                        address, mix.data(),
                        mix.size() * sizeof(float))) {
                    result = OrbisError(MemoryFault);
                    return true;
                }
            }
        }
        result = 0;
        return true;
    }
    if (symbol == "pgFAiLR5qT4" || symbol == "0eFLVCfWVds") {
        const auto output = request.integer_arguments[
            symbol == "pgFAiLR5qT4" ? 1 : 2];
        if (output == 0) {
            result = OrbisError(InvalidOutAddress);
            return true;
        }
        std::array<std::uint8_t, 0x18> info{};
        const std::uint64_t size = 0x10000;
        const std::uint64_t alignment = 0x100;
        std::memcpy(info.data(), &size, sizeof(size));
        std::memcpy(info.data() + 8, &alignment, sizeof(alignment));
        result = TryWriteGuestBytes(output, info.data(), info.size())
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "l4Q2dWEH6UM") {
        const auto handle = request.integer_arguments[0];
        const auto grain =
            static_cast<std::uint32_t>(request.integer_arguments[1]);
        const std::lock_guard lock{g_ngs2_mutex};
        if (auto system = g_ngs2_systems.find(handle);
            system != g_ngs2_systems.end()) {
            if (grain != 0 && grain <= 8192) {
                system->second.grain_samples = grain;
            }
            result = 0;
        } else {
            result = OrbisError(InvalidSystemHandle);
        }
        return true;
    }
    if (symbol == "-tbc2SxQD60" || symbol == "gThZqM5PYlQ" ||
        symbol == "JXRC5n0RQls") {
        const std::lock_guard lock{g_ngs2_mutex};
        result = g_ngs2_systems.contains(request.integer_arguments[0])
            ? 0
            : OrbisError(InvalidSystemHandle);
        return true;
    }
    if (symbol == "-TOuuAQ-buE") {
        const auto handle = request.integer_arguments[0];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_voices.contains(handle)) {
                result = OrbisError(InvalidVoiceHandle);
                return true;
            }
        }
        const auto address = request.integer_arguments[1];
        const auto size = std::min<std::uint64_t>(
            request.integer_arguments[2], 0x400);
        result = address == 0 || size == 0 ||
                         ClearNgs2GuestBuffer(address, size)
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "rEh728kXk3w") {
        const auto handle = request.integer_arguments[0];
        {
            const std::lock_guard lock{g_ngs2_mutex};
            if (!g_ngs2_voices.contains(handle)) {
                result = OrbisError(InvalidVoiceHandle);
                return true;
            }
        }
        const std::uint64_t flags{};
        const auto address = request.integer_arguments[1];
        result = address == 0 ||
                         TryWriteGuestBytes(
                             address, &flags, sizeof(flags))
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "xa8oL9dmXkM" || symbol == "1WsleK-MTkE" ||
        symbol == "0lbbayqDNoE" || symbol == "7Lcfo8SmpsU") {
        result = 0;
        return true;
    }
    return false;
}

bool TryInvokeBuiltinHle(
    const Lsx4::Translation::HleBridgeRequest& request,
    std::uint64_t& result) {
    if (g_runtime.shutdown_requested.load(
            std::memory_order_acquire)) {
        result = 0;
        (void)Lsx4::Translation::CompleteCurrentGuestExecution(0);
        return true;
    }
    const auto symbol = HleSymbol(request.function);
    if (symbol == "j3YMu1MVNNo") {
        g_runtime.guest_login_event_delivered.store(
            false, std::memory_order_release);
        result = 0;
        return true;
    }
    if (symbol == "CdWp0oHWGr0") {
        constexpr std::int32_t PrimaryUser = 0x10000000;
        result = request.integer_arguments[0] != 0 &&
                TryWriteGuestValue(
                    request.integer_arguments[0], PrimaryUser)
            ? 0
            : OrbisError(UINT32_C(0x80960005));
        return true;
    }
    if (symbol == "fPhymKNvK-A") {
        constexpr std::array<std::int32_t, 4> Users{
            0x10000000, -1, -1, -1};
        result = request.integer_arguments[0] != 0 &&
                TryWriteGuestBytes(
                    request.integer_arguments[0],
                    Users.data(), sizeof(Users))
            ? 0
            : OrbisError(UINT32_C(0x80960005));
        return true;
    }
    if (symbol == "yH17Q6NWtVg") {
        const auto event = request.integer_arguments[0];
        if (event == 0) {
            result = OrbisError(UINT32_C(0x80960005));
            return true;
        }
        if (g_runtime.guest_login_event_delivered.exchange(
                true, std::memory_order_acq_rel)) {
            result = OrbisError(UINT32_C(0x80960007));
            return true;
        }
        constexpr std::array<std::int32_t, 2> LoginEvent{
            0, 0x10000000};
        result = TryWriteGuestBytes(
                     event, LoginEvent.data(), sizeof(LoginEvent))
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "eQH7nWPcAgc") {
        constexpr std::uint32_t SignedIn = 1;
        result = request.integer_arguments[1] != 0 &&
                TryWriteGuestValue(
                    request.integer_arguments[1], SignedIn)
            ? 0
            : OrbisError(UINT32_C(0x80020003));
        return true;
    }
    if (symbol == "iQw3iQPhvUQ") {
        result = 0;
        return true;
    }
    if (symbol == "uBPlr0lbuiI") {
        constexpr std::int32_t Disconnected = 0;
        result = request.integer_arguments[0] != 0 &&
                TryWriteGuestValue(
                    request.integer_arguments[0], Disconnected)
            ? 0
            : OrbisError(UINT32_C(0x80412102));
        return true;
    }
    if (symbol == "3RQ5aQfnstU") {
        const std::uint8_t skip_notice_screen{};
        result = request.integer_arguments[0] != 0 &&
                TryWriteGuestValue(
                    request.integer_arguments[0],
                    skip_notice_screen)
            ? 0
            : OrbisError(UINT32_C(0x80020003));
        return true;
    }
    if (symbol == "lQOCF84lvzw") {
        // Match the offline PS4 contract. Reporting success without an
        // HTTP response leaves the title's response factory uninitialized
        // and it later calls a null interface method.
        result = OrbisError(UINT32_C(0x80553407));
        return true;
    }
    if (symbol == "HwP3aM+c85c") {
        const std::uint64_t empty_length{};
        if (request.integer_arguments[2] != 0) {
            (void)TryWriteGuestValue(
                request.integer_arguments[2], empty_length);
        }
        result = OrbisError(UINT32_C(0x80553407));
        return true;
    }
    if (symbol == "sk54bi6FtYM" ||
        symbol == "3EI-OSJ65Xc" ||
        symbol == "egOOvrnF6mI" ||
        symbol == "vvzWO-DvG1s") {
        result = 0;
        return true;
    }
    if (symbol == "aesyjrHVWy4") {
        const auto left = request.integer_arguments[0];
        const auto right = request.integer_arguments[1];
        const auto limit = request.integer_arguments[2];
        if (limit == 0 || left == right) {
            result = 0;
            return true;
        }
        if (left == 0 || right == 0) {
            const std::int64_t compare =
                left == 0 ? -1 : 1;
            result = static_cast<std::uint64_t>(compare);
            return true;
        }
        const auto bounded_limit =
            std::min<std::uint64_t>(
                limit, UINT64_C(16) * 1024u * 1024u);
        for (std::uint64_t index = 0;
             index < bounded_limit; ++index) {
            std::uint8_t left_byte{};
            std::uint8_t right_byte{};
            if (!TryReadGuestBytes(
                    left + index, &left_byte, 1) ||
                !TryReadGuestBytes(
                    right + index, &right_byte, 1)) {
                result = 0;
                return true;
            }
            if (left_byte != right_byte) {
                const auto compare =
                    static_cast<std::int64_t>(
                        static_cast<std::int32_t>(left_byte) -
                        static_cast<std::int32_t>(right_byte));
                result = static_cast<std::uint64_t>(compare);
                return true;
            }
            if (left_byte == 0) {
                result = 0;
                return true;
            }
        }
        result = 0;
        return true;
    }
    if (TryHandleNgs2(symbol, request, result)) {
        return true;
    }
    constexpr std::int32_t MsgDialogStatusNone = 0;
    constexpr std::int32_t MsgDialogStatusInitialized = 1;
    constexpr std::int32_t MsgDialogStatusRunning = 2;
    constexpr std::int32_t MsgDialogStatusFinished = 3;
    constexpr std::uint32_t MsgDialogErrorNotInitialized =
        UINT32_C(0x80b80003);
    constexpr std::uint32_t MsgDialogErrorNotFinished =
        UINT32_C(0x80b80005);
    constexpr std::uint32_t MsgDialogErrorBusy =
        UINT32_C(0x80b80007);
    constexpr std::uint32_t MsgDialogErrorNotRunning =
        UINT32_C(0x80b8000b);
    constexpr std::uint32_t MsgDialogErrorArgNull =
        UINT32_C(0x80b8000d);
    constexpr std::uint32_t MemoryFault =
        UINT32_C(0x80020101);
    if (symbol == "Xjoosiw+XPI") {
        static std::atomic<std::uint64_t> uuid_counter{1};
        const auto destination = request.integer_arguments[0];
        std::array<std::uint8_t, 16> uuid{};
        const auto sequence =
            uuid_counter.fetch_add(1, std::memory_order_relaxed);
        const auto ticks = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        std::memcpy(uuid.data(), &ticks, sizeof(ticks));
        std::memcpy(
            uuid.data() + sizeof(ticks), &sequence, sizeof(sequence));
        uuid[6] = static_cast<std::uint8_t>(
            (uuid[6] & 0x0fu) | 0x40u);
        uuid[8] = static_cast<std::uint8_t>(
            (uuid[8] & 0x3fu) | 0x80u);
        result =
            destination != 0 &&
                    TryWriteGuestBytes(
                        destination, uuid.data(), uuid.size())
                ? 0
                : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "dgJBaeJnGpo" ||
        symbol == "hdpVEUDFW3s" ||
        symbol == "3JCe3lCbQ8A") {
        const std::lock_guard lock{g_runtime.mutex};
        auto* next_handle = symbol == "dgJBaeJnGpo"
            ? &g_runtime.next_net_pool_handle
            : (symbol == "hdpVEUDFW3s"
                   ? &g_runtime.next_ssl_handle
                   : &g_runtime.next_http2_handle);
        result = static_cast<std::uint64_t>((*next_handle)++);
        return true;
    }
    if (symbol == "T72hz6ffq08") {
        // Release the host core for a guest yield. Returning immediately
        // turns idle guest worker loops into mobile-CPU busy loops.
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        result = 0;
        return true;
    }
    if (symbol == "lDqxaY1UbEo") {
        auto expected = MsgDialogStatusNone;
        (void)g_runtime.msg_dialog_status.compare_exchange_strong(
            expected, MsgDialogStatusInitialized,
            std::memory_order_acq_rel);
        result = 0;
        return true;
    }
    if (symbol == "ePw-kqZmelo") {
        const auto previous =
            g_runtime.msg_dialog_status.exchange(
                MsgDialogStatusNone, std::memory_order_acq_rel);
        result = previous == MsgDialogStatusNone
            ? MsgDialogErrorNotInitialized
            : 0;
        return true;
    }
    if (symbol == "b06Hh0DPEaE") {
        if (request.integer_arguments[0] == 0) {
            result = MsgDialogErrorArgNull;
            return true;
        }
        const auto status =
            g_runtime.msg_dialog_status.load(
                std::memory_order_acquire);
        if (status == MsgDialogStatusNone) {
            result = MsgDialogErrorNotInitialized;
            return true;
        }
        if (status == MsgDialogStatusRunning) {
            result = MsgDialogErrorBusy;
            return true;
        }
        g_runtime.msg_dialog_status.store(
            MsgDialogStatusRunning, std::memory_order_release);
        result = 0;
        return true;
    }
    if (symbol == "CWVW78Qc3fI" ||
        symbol == "6fIC3XKt2k0") {
        auto expected = MsgDialogStatusRunning;
        (void)g_runtime.msg_dialog_status.compare_exchange_strong(
            expected, MsgDialogStatusFinished,
            std::memory_order_acq_rel);
        result = static_cast<std::uint32_t>(
            g_runtime.msg_dialog_status.load(
                std::memory_order_acquire));
        return true;
    }
    if (symbol == "Lr8ovHH9l6A") {
        if (request.integer_arguments[0] == 0) {
            result = MsgDialogErrorArgNull;
            return true;
        }
        if (g_runtime.msg_dialog_status.load(
                std::memory_order_acquire) !=
            MsgDialogStatusFinished) {
            result = MsgDialogErrorNotFinished;
            return true;
        }
        std::array<std::uint8_t, 0x20> dialog_result{};
        const std::uint32_t affirmative_button = 1;
        std::memcpy(
            dialog_result.data() + 0x08,
            &affirmative_button, sizeof(affirmative_button));
        result = TryWriteGuestBytes(
                     request.integer_arguments[0],
                     dialog_result.data(), dialog_result.size())
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "HTrcDKlFKuM") {
        auto expected = MsgDialogStatusRunning;
        if (!g_runtime.msg_dialog_status.compare_exchange_strong(
                expected, MsgDialogStatusFinished,
                std::memory_order_acq_rel)) {
            result = MsgDialogErrorNotRunning;
            return true;
        }
        result = 0;
        return true;
    }
    if (symbol == "wTpfglkmv34" ||
        symbol == "Gc5k1qcK4fs" ||
        symbol == "6H-71OdrpXM") {
        result = g_runtime.msg_dialog_status.load(
                     std::memory_order_acquire) ==
                MsgDialogStatusNone
            ? MsgDialogErrorNotInitialized
            : 0;
        return true;
    }
    if (symbol == "s9e3+YpRnzw") {
        g_runtime.save_data_dialog_status.store(
            MsgDialogStatusInitialized, std::memory_order_release);
        g_runtime.save_data_dialog_mode.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_user_data.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_dir_name.fill('\0');
        result = 0;
        return true;
    }
    if (symbol == "4tPhsP6FpDI") {
        const auto parameter = request.integer_arguments[0];
        if (parameter == 0) {
            result = MsgDialogErrorArgNull;
            return true;
        }
        std::int32_t mode{};
        std::uint64_t user_data{};
        std::uint64_t items{};
        (void)TryReadGuestValue(parameter + 0x34u, mode);
        (void)TryReadGuestValue(parameter + 0x48u, items);
        (void)TryReadGuestValue(parameter + 0x70u, user_data);
        g_runtime.save_data_dialog_mode.store(
            mode, std::memory_order_release);
        g_runtime.save_data_dialog_user_data.store(
            user_data, std::memory_order_release);
        g_runtime.save_data_dialog_dir_name.fill('\0');
        if (items != 0) {
            std::uint64_t dir_names{};
            std::uint32_t dir_names_num{};
            if (TryReadGuestValue(items + 0x10u, dir_names) &&
                TryReadGuestValue(
                    items + 0x18u, dir_names_num) &&
                dir_names != 0) {
                for (std::uint32_t index = 0;
                     index < std::min(dir_names_num, 64u);
                     ++index) {
                    std::array<char, 32> candidate{};
                    if (TryReadGuestBytes(
                            dir_names +
                                static_cast<std::uint64_t>(index) *
                                    candidate.size(),
                            candidate.data(), candidate.size()) &&
                        candidate[0] != '\0') {
                        g_runtime.save_data_dialog_dir_name =
                            candidate;
                        break;
                    }
                }
            }
        }
        g_runtime.save_data_dialog_status.store(
            MsgDialogStatusFinished, std::memory_order_release);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "save dialog open mode=%d items=0x%llx user=0x%llx "
            "dir=%s",
            mode,
            static_cast<unsigned long long>(items),
            static_cast<unsigned long long>(user_data),
            g_runtime.save_data_dialog_dir_name.data());
#endif
        result = 0;
        return true;
    }
    if (symbol == "ERKzksauAJA" ||
        symbol == "KK3Bdg1RWK0") {
        result = static_cast<std::uint32_t>(
            g_runtime.save_data_dialog_status.load(
                std::memory_order_acquire));
        return true;
    }
    if (symbol == "en7gNVnh878") {
        result = 1;
        return true;
    }
    if (symbol == "yEiJ-qqr6Cg") {
        const auto output = request.integer_arguments[0];
        if (output == 0) {
            result = MsgDialogErrorArgNull;
            return true;
        }
        std::array<std::uint8_t, 0x48> dialog_result{};
        if (!TryReadGuestBytes(
                output, dialog_result.data(),
                dialog_result.size())) {
            result = OrbisError(MemoryFault);
            return true;
        }
        const auto mode =
            g_runtime.save_data_dialog_mode.load(
                std::memory_order_acquire);
        const std::int32_t dialog_result_ok = 0;
        const std::uint32_t affirmative_button = 1;
        const auto user_data =
            g_runtime.save_data_dialog_user_data.load(
                std::memory_order_acquire);
        std::uint64_t dir_name{};
        std::memcpy(
            &dir_name, dialog_result.data() + 0x10,
            sizeof(dir_name));
        std::memcpy(
            dialog_result.data(), &mode, sizeof(mode));
        std::memcpy(
            dialog_result.data() + 0x04,
            &dialog_result_ok, sizeof(dialog_result_ok));
        std::memcpy(
            dialog_result.data() + 0x08,
            &affirmative_button, sizeof(affirmative_button));
        std::memcpy(
            dialog_result.data() + 0x20,
            &user_data, sizeof(user_data));
        const auto wrote_result = TryWriteGuestBytes(
            output, dialog_result.data(), dialog_result.size());
        const auto wrote_dir =
            dir_name == 0 ||
            g_runtime.save_data_dialog_dir_name[0] == '\0' ||
            TryWriteGuestBytes(
                dir_name,
                g_runtime.save_data_dialog_dir_name.data(),
                g_runtime.save_data_dialog_dir_name.size());
        result = wrote_result && wrote_dir
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "fH46Lag88XY") {
        g_runtime.save_data_dialog_status.store(
            MsgDialogStatusFinished, std::memory_order_release);
        result = 0;
        return true;
    }
    if (symbol == "YuH2FA7azqQ") {
        const auto previous =
            g_runtime.save_data_dialog_status.exchange(
                MsgDialogStatusNone, std::memory_order_acq_rel);
        g_runtime.save_data_dialog_mode.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_user_data.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_dir_name.fill('\0');
        result = previous == MsgDialogStatusNone
            ? MsgDialogErrorNotInitialized
            : 0;
        return true;
    }
    if (symbol == "V-uEeFKARJU" ||
        symbol == "hay1CfTmLyA") {
        result = 0;
        return true;
    }
    if (symbol == "18B2NS1y9UU" ||
        symbol == "LN3Zcb72Q0c" ||
        symbol == "Ot1DE3gif84" ||
        symbol == "zO9UL3qIINQ" ||
        symbol == "HWxHOdbM-Pg") {
        constexpr std::uint64_t UnixEpochMicroseconds =
            UINT64_C(62135596800000000);
        const auto output = request.integer_arguments[0];
        if (output == 0) {
            result = UINT32_C(0x80b50002);
            return true;
        }
        const auto unix_microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now()
                    .time_since_epoch())
                .count();
        const auto tick =
            UnixEpochMicroseconds +
            static_cast<std::uint64_t>(
                std::max<std::int64_t>(unix_microseconds, 0));
        result = TryWriteGuestU64(output, tick)
            ? 0
            : OrbisError(MemoryFault);
        return true;
    }
    if (symbol == "gQX+4GDQjpM") {
        (void)TryAllocateLibcHeap(
            request.integer_arguments[0],
            alignof(std::max_align_t), result,
            request.guest_stack);
#ifdef __ANDROID__
        if (request.integer_arguments[0] == UINT64_C(0x800)) {
            LogDreamingForeachState("malloc-0x800-return", result);
        }
#endif
        return true;
    }
    if (symbol == "bzQExy189ZI" ||
        symbol == "8G2LB+A3rzg" ||
        symbol == "tsvEmnenz48" ||
        symbol == "H2e8t5ScQGc" ||
        symbol == "kbw4UHHSYy0") {
        result = 0;
        return true;
    }
    if (symbol == "uMei1W9uyNo" ||
        symbol == "XKRegsFpEpk") {
        result = request.integer_arguments[0];
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-EXIT",
            "process exit symbol=%s value=0x%llx stack=0x%llx",
            symbol.c_str(),
            static_cast<unsigned long long>(result),
            static_cast<unsigned long long>(request.guest_stack));
#endif
        (void)Lsx4::Translation::CompleteCurrentGuestExecution(result);
        return true;
    }
    if (symbol == "tIhsqj0qsFE") {
        FreeLibcHeap(
            request.integer_arguments[0], request.guest_stack);
        result = 0;
        return true;
    }
    if (symbol == "2X5agFjKxMc") {
        const auto count = request.integer_arguments[0];
        const auto element_size = request.integer_arguments[1];
        if (count != 0 &&
            element_size >
                std::numeric_limits<std::uint64_t>::max() / count) {
            result = 0;
            return true;
        }
        (void)TryAllocateLibcHeap(
            count * element_size,
            alignof(std::max_align_t), result,
            request.guest_stack);
        return true;
    }
    if (symbol == "Y7aJ1uydPMo") {
        return TryReallocateLibcHeap(
            request.integer_arguments[0],
            request.integer_arguments[1], result,
            request.guest_stack);
    }
    if (symbol == "Ujf3KzMvRmI" ||
        symbol == "2Btkg8k24Zg") {
        const auto alignment = request.integer_arguments[0];
        const auto size = request.integer_arguments[1];
        if (symbol == "2Btkg8k24Zg" &&
            (alignment == 0 || size % alignment != 0)) {
            result = 0;
            return true;
        }
        (void)TryAllocateLibcHeap(
            size, alignment, result, request.guest_stack);
        return true;
    }
    if (symbol == "cVSk9y8URbc") {
        constexpr std::uint64_t InvalidArgument = 22;
        constexpr std::uint64_t OutOfMemory = 12;
        const auto output_address = request.integer_arguments[0];
        const auto alignment = request.integer_arguments[1];
        const auto size = request.integer_arguments[2];
        if (output_address == 0 ||
            !IsPowerOfTwo(alignment) ||
            alignment < sizeof(void*)) {
            result = InvalidArgument;
            return true;
        }
        std::uint64_t allocated{};
        if (!TryAllocateLibcHeap(
                size, alignment, allocated,
                request.guest_stack)) {
            result = OutOfMemory;
            return true;
        }
        if (!TryWriteGuestU64(output_address, allocated)) {
            FreeLibcHeap(allocated, request.guest_stack);
            result = OutOfMemory;
            return true;
        }
        result = 0;
        return true;
    }
    if (symbol == "xeYO4u7uyJ0") {
        return TryOpenGuestStdio(
            request.integer_arguments[0],
            request.integer_arguments[1], result);
    }
    if (symbol == "rQFVBXp-Cxg") {
        auto* const stream =
            FindGuestStdio(request.integer_arguments[0]);
        if (stream == nullptr) {
            result = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(-1));
            return true;
        }
        const auto offset = static_cast<std::int64_t>(
            request.integer_arguments[1]);
        const auto whence =
            static_cast<int>(request.integer_arguments[2]);
        result = std::fseek(
                     stream, static_cast<long>(offset), whence) == 0
            ? 0
            : static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(-1));
        return true;
    }
    if (symbol == "Qazy8LmXTvw") {
        auto* const stream =
            FindGuestStdio(request.integer_arguments[0]);
        if (stream == nullptr) {
            result = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(-1));
            return true;
        }
        const auto position = std::ftell(stream);
        result = position >= 0
            ? static_cast<std::uint64_t>(position)
            : static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(-1));
        return true;
    }
    if (symbol == "uodLYyUip20") {
        const auto guest_handle = request.integer_arguments[0];
        std::FILE* stream{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto found = std::ranges::find(
                g_runtime.stdio_files, guest_handle,
                &GuestStdioFile::guest_handle);
            if (found != g_runtime.stdio_files.end()) {
                stream = found->stream;
                g_runtime.stdio_files.erase(found);
            }
        }
        if (stream == nullptr) {
            result = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(-1));
            return true;
        }
        const auto close_status = std::fclose(stream);
        result = close_status == 0
            ? 0
            : static_cast<std::uint64_t>(
                  static_cast<std::int64_t>(-1));
        return true;
    }
    if (symbol == "lbB+UlZqVG0") {
        const auto destination = request.integer_arguments[0];
        const auto element_size = request.integer_arguments[1];
        const auto element_count = request.integer_arguments[2];
        auto* const stream =
            FindGuestStdio(request.integer_arguments[3]);
        if (element_size == 0 || element_count == 0) {
            result = 0;
            return true;
        }
        if (stream == nullptr ||
            element_count >
                std::numeric_limits<std::uint64_t>::max() /
                    element_size) {
            result = 0;
            return true;
        }
        const auto byte_count = element_size * element_count;
        if (byte_count >
            std::numeric_limits<std::size_t>::max()) {
            result = 0;
            return true;
        }
        {
            const std::lock_guard lock{g_runtime.mutex};
            if (!HasAccessLocked(
                    destination, byte_count,
                    LSX4_PS5_GUEST_WRITE)) {
                result = 0;
                return true;
            }
        }
        result = std::fread(
            reinterpret_cast<void*>(destination),
            static_cast<std::size_t>(element_size),
            static_cast<std::size_t>(element_count), stream);
        return true;
    }
    if (symbol == "eV9wAD2riIA") {
        return TryGuestKernelStat(
            request.integer_arguments[0],
            request.integer_arguments[1], result);
    }
    if (symbol == "1G3lF1Gg1k8") {
        return TryGuestKernelOpen(request, result);
    }
    if (symbol == "UK2Tl2DWUns") {
        return TryGuestKernelClose(
            static_cast<std::int32_t>(
                request.integer_arguments[0]),
            result);
    }
    if (symbol == "kBwCPsYX-m4") {
        return TryGuestKernelFstat(
            static_cast<std::int32_t>(
                request.integer_arguments[0]),
            request.integer_arguments[1], result);
    }
    if (symbol == "Cg4srZ6TKbU") {
        return TryGuestKernelRead(
            static_cast<std::int32_t>(
                request.integer_arguments[0]),
            request.integer_arguments[1],
            request.integer_arguments[2], result);
    }
    if (symbol == "4wSze92BhLI") {
        return TryGuestKernelWrite(
            static_cast<std::int32_t>(
                request.integer_arguments[0]),
            request.integer_arguments[1],
            request.integer_arguments[2], result);
    }
    if (symbol == "WT-5NKy42fw") {
        return TryResolveAprFilepathsToIds(
            request, result);
    }
    if (symbol == "gEpBkcwxUjw") {
        return TryResolveAprFilepathsToIdsAndSizes(
            request, result);
    }
    if (symbol == "ApkYaHb8Sek") {
        return TryGetAprFileStat(
            static_cast<std::uint32_t>(
                request.integer_arguments[0]),
            request.integer_arguments[1], result);
    }
    if (symbol == "mQ16-QdKv7k") {
        return TryReadAprFile(request, result);
    }
    if (symbol == "8aI7R7WaOlc") {
        return TryConstructAprCommandBuffer(
            request, false, result);
    }
    if (symbol == "a8uLzYY--tM") {
        return TryConstructAprCommandBuffer(
            request, true, result);
    }
    if (symbol == "Qs1xtplKo0U") {
        return TryDestroyAprCommandBuffer(
            request.integer_arguments[0], true, result);
    }
    if (symbol == "GuchCTefuZw") {
        return TryDestroyAprCommandBuffer(
            request.integer_arguments[0], false, result);
    }
    if (symbol == "N-FSPA4S3nI") {
        return TrySetAprCommandBuffer(request, result);
    }
    if (symbol == "ULvXMDz56po") {
        return TryClearAprCommandBuffer(
            request.integer_arguments[0], result);
    }
    if (symbol == "sJXyWHjP-F8") {
        return TryAppendAprCompletionWrite(request, result);
    }
    if (symbol == "j0+3uJMxYJY") {
        return TryAppendAprCompletionWrite(request, result);
    }
    if (symbol == "H896Pt-yB4I" ||
        symbol == "o67gODLFpls") {
        return TryAppendAprCompletionEvent(request, result);
    }
    if (symbol == "vWU-odnS+fU") {
        result = AprReadFileRecordSize;
        return true;
    }
    if (symbol == "sSAUCCU1dv4" ||
        symbol == "Zi3dBUjgyXI") {
        result = AprKernelEventQueueRecordSize;
        return true;
    }
    if (symbol == "C+IEj+BsAFM" ||
        symbol == "4fgtGfXDrFc") {
        result = AprWriteAddressRecordSize;
        return true;
    }
    if (symbol == "tZDDEo2tE5k") {
        return TryGetAprCommandBufferProperty(
            request.integer_arguments[0], 0, result);
    }
    if (symbol == "GnxKOHEawhk") {
        return TryGetAprCommandBufferProperty(
            request.integer_arguments[0], 1, result);
    }
    if (symbol == "gzndltBEzWc") {
        return TryGetAprCommandBufferProperty(
            request.integer_arguments[0], 2, result);
    }
    if (symbol == "ASoW5WE-UPo") {
        return TrySubmitAprCommandBuffer(request, result);
    }
    if (symbol == "rqwFKI4PAiM") {
        return TryWaitAprCommandBuffer(
            static_cast<std::uint32_t>(
                request.integer_arguments[0]),
            result);
    }
    if (symbol == "baQO9ez2gL4") {
        return TryResetAprCommandBuffer(
            request.integer_arguments[0], result);
    }
    if (symbol == "vNe1w4diLCs") {
        return TryResolveTlsAddress(
            request.integer_arguments[0], result);
    }
    if (symbol == "1jfXLRVzisc") {
        const auto microseconds = request.integer_arguments[0];
#ifdef __ANDROID__
        if (Ps5DiagnosticFaultProbeEnabled() &&
            microseconds == 1000) {
            static std::atomic<std::uint64_t>
                diagnostic_poll_sleep_count{};
            const auto count =
                diagnostic_poll_sleep_count.fetch_add(
                    1, std::memory_order_relaxed) + 1u;
            if (count <= 4 || (count & (count - 1u)) == 0) {
                std::array<std::uint64_t, 4> stack{};
                std::array<std::uint64_t, 4> context{};
                Executor::Jit::LsxMachineImage guest_state{};
                const auto state_ok =
                    Executor::Jit::SnapshotGuestStateAfterBridge(
                        guest_state);
                const auto guest_rbp = state_ok
                    ? Executor::Jit::ReadGuestGpr64(
                          guest_state,
                          Executor::Jit::LsxGpr::Rbp)
                    : 0;
                std::uint64_t completion_pointer{};
                std::uint64_t completion_value{};
                const auto completion_ok =
                    guest_rbp >= 0x50u &&
                    TryReadGuestValue(
                        guest_rbp - 0x50u,
                        completion_pointer) &&
                    completion_pointer != 0 &&
                    TryReadGuestValue(
                        completion_pointer,
                        completion_value);
                const auto stack_ok = TryReadGuestBytes(
                    request.guest_stack, stack.data(),
                    sizeof(stack));
                const auto context_ok =
                    request.integer_arguments[1] != 0 &&
                    TryReadGuestBytes(
                        request.integer_arguments[1],
                        context.data(), sizeof(context));
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-POLL",
                    "usleep count=%llu stack=0x%llx ok=%d "
                    "returns=%llx,%llx,%llx,%llx "
                    "context=0x%llx ok=%d "
                    "qwords=%llx,%llx,%llx,%llx "
                    "rbp=0x%llx completion=0x%llx "
                    "value=0x%llx ok=%d",
                    static_cast<unsigned long long>(count),
                    static_cast<unsigned long long>(
                        request.guest_stack),
                    stack_ok ? 1 : 0,
                    static_cast<unsigned long long>(stack[0]),
                    static_cast<unsigned long long>(stack[1]),
                    static_cast<unsigned long long>(stack[2]),
                    static_cast<unsigned long long>(stack[3]),
                    static_cast<unsigned long long>(
                        request.integer_arguments[1]),
                    context_ok ? 1 : 0,
                    static_cast<unsigned long long>(context[0]),
                    static_cast<unsigned long long>(context[1]),
                    static_cast<unsigned long long>(context[2]),
                    static_cast<unsigned long long>(context[3]),
                    static_cast<unsigned long long>(guest_rbp),
                    static_cast<unsigned long long>(
                        completion_pointer),
                    static_cast<unsigned long long>(
                        completion_value),
                    completion_ok ? 1 : 0);
            }
        }
#endif
        if (microseconds != 0) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(microseconds));
        } else {
            // Android's sched_yield keeps large guest worker pools runnable
            // and can starve the render thread. A short timed park preserves
            // zero-delay semantics while releasing a core to the main guest.
            std::this_thread::sleep_for(
                std::chrono::microseconds(50));
        }
        result = 0;
        return true;
    }
    if (symbol == "NhpspxdjEKU" ||
        symbol == "yS8U2TGCe1A" ||
        symbol == "QvsZxomvUHs") {
        const auto request_address = request.integer_arguments[0];
        const auto remain_address = request.integer_arguments[1];
        std::array<std::int64_t, 2> requested{};
        if (request_address == 0 ||
            !TryReadGuestBytes(
                request_address, requested.data(),
                sizeof(requested)) ||
            requested[0] < 0 || requested[1] < 0 ||
            requested[1] >= INT64_C(1000000000)) {
            result = symbol == "QvsZxomvUHs"
                ? OrbisError(UINT32_C(0x80020003))
                : std::numeric_limits<std::uint64_t>::max();
            return true;
        }
        if (requested[0] != 0 || requested[1] != 0) {
            const auto seconds =
                std::chrono::seconds(requested[0]);
            const auto nanoseconds =
                std::chrono::nanoseconds(requested[1]);
            std::this_thread::sleep_for(seconds + nanoseconds);
        } else {
            std::this_thread::sleep_for(
                std::chrono::microseconds(50));
        }
        if (remain_address != 0) {
            const std::array<std::int64_t, 2> remaining{};
            (void)TryWriteGuestBytes(
                remain_address, remaining.data(),
                sizeof(remaining));
        }
        result = 0;
        return true;
    }
    if (symbol == "1j3S3n-tTW4") {
#if defined(__aarch64__)
        std::uint64_t frequency{};
        asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
        result = frequency;
#else
        result = UINT64_C(1000000000);
#endif
        return true;
    }
    if (symbol == "-2IRUCO--PM") {
#if defined(__aarch64__)
        std::uint64_t counter{};
        asm volatile("mrs %0, cntvct_el0" : "=r"(counter));
        result = counter;
#else
        result = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#endif
        return true;
    }
    if (symbol == "4J2sUJmuHZQ") {
        result = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() -
                g_ps5_process_start).count());
        return true;
    }
    if (symbol == "fgxnMeTNUtY") {
        result = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() -
                g_ps5_process_start).count());
        return true;
    }
    if (symbol == "BNowx2l588E") {
        result = UINT64_C(1000000000);
        return true;
    }
    if (symbol == "959qrazPIrg") {
        const std::lock_guard lock{g_runtime.mutex};
        const auto* const thread = CurrentGuestThreadLocked();
        const auto owner_handle =
            thread != nullptr ? thread->program_handle : 0;
        const auto executable = std::ranges::find_if(
            g_runtime.programs,
            [&](const LoadedProgram& program) {
                return !program.module &&
                       (owner_handle == 0 ||
                        program.handle == owner_handle);
            });
        result = executable != g_runtime.programs.end()
            ? executable->image.process_param
            : 0;
        return true;
    }
    if (symbol == "Tz4RNUCBbGI" ||
        symbol == "8OnWXlgQlvo") {
        const auto counter_address =
            request.integer_arguments[0];
        if (counter_address == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        std::uint64_t value{};
        if (!GuestReadU64Locked(counter_address, value)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        if (symbol == "Tz4RNUCBbGI") {
            ++value;
        } else if (value != 0) {
            --value;
        }
        if (!GuestWriteU64Locked(counter_address, value)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        result = value;
        return true;
    }
    if (symbol == "p5EcQeEeJAE") {
        result = 0;
        return true;
    }
    if (symbol == "bnZxYgAFeA0") {
        constexpr std::size_t NewReplaceSize = 0x68;
        const auto address = CurrentThreadTlsScratch(0x300);
        std::array<std::uint8_t, NewReplaceSize> replacement{};
        const auto size = static_cast<std::uint64_t>(replacement.size());
        std::memcpy(replacement.data(), &size, sizeof(size));
        result =
            address != 0 &&
                    TryWriteGuestBytes(
                        address, replacement.data(),
                        replacement.size())
                ? address
                : 0;
        return true;
    }
    if (symbol == "NWtTN10cJzE") {
        constexpr std::uint64_t TraceInfoSize = 32;
        const auto info_address = request.integer_arguments[0];
        std::uint64_t supplied_size{};
        if (info_address == 0 ||
            !TryReadGuestBytes(
                info_address, &supplied_size,
                sizeof(supplied_size)) ||
            supplied_size != TraceInfoSize) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        const auto trace_storage =
            CurrentThreadTlsScratch(0x400);
        const auto trace_mask = trace_storage;
        const auto trace_table =
            trace_storage + sizeof(std::uint64_t);
        const bool wrote =
            trace_storage != 0 &&
            TryWriteGuestU64(
                info_address + 16, trace_mask) &&
            TryWriteGuestU64(
                info_address + 24, trace_table);
        result = wrote
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "f7KBOafysXo" ||
        symbol == "RpQJJVKTiFM") {
        constexpr std::size_t ModuleInfoExSize = 0x1a8;
        constexpr std::size_t ModuleInfoUnwindSize = 0x130;
        const auto queried_address =
            request.integer_arguments[0];
        const auto flags =
            static_cast<std::int32_t>(
                request.integer_arguments[1]);
        const auto output_address =
            request.integer_arguments[2];
        if (output_address == 0 ||
            flags < 0 || flags >= 3) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        std::uint64_t caller_size{};
        if (!GuestReadU64Locked(
                output_address, caller_size)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        const auto required_size =
            symbol == "f7KBOafysXo"
                ? ModuleInfoExSize
                : ModuleInfoUnwindSize;
        if (caller_size < required_size ||
            !HasAccessLocked(
                output_address, required_size,
                LSX4_PS5_GUEST_WRITE)) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        const auto program = std::ranges::find_if(
            g_runtime.programs,
            [&](const LoadedProgram& candidate) {
                return std::ranges::any_of(
                    candidate.image.registered_ranges,
                    [&](const Funnel::Ps5Desktop::LoadedRange& range) {
                        return queried_address >= range.address &&
                               queried_address - range.address <
                                   range.byte_count;
                    });
            });
        if (program == g_runtime.programs.end()) {
            result = OrbisError(UINT32_C(0x80020002));
            return true;
        }
        std::array<std::uint8_t, ModuleInfoExSize> payload{};
        const auto payload_size =
            static_cast<std::uint64_t>(required_size);
        std::memcpy(
            payload.data(), &payload_size,
            sizeof(payload_size));
        const auto module_name =
            std::filesystem::path{program->path}
                .filename().string();
        const auto name_bytes = std::min<std::size_t>(
            module_name.size(), 0xff);
        std::memcpy(
            payload.data() + 0x08,
            module_name.data(), name_bytes);
        const auto image_base = program->image.base;
        const auto image_size = program->image.mapped_size;
        if (symbol == "f7KBOafysXo") {
            const auto module_handle =
                static_cast<std::int32_t>(program->handle);
            std::memcpy(
                payload.data() + 0x108,
                &module_handle, sizeof(module_handle));
            std::memcpy(
                payload.data() + 0x128,
                &program->image.entry,
                sizeof(program->image.entry));
            std::memcpy(
                payload.data() + 0x160,
                &image_base, sizeof(image_base));
            const auto segment_size =
                static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(
                        image_size,
                        std::numeric_limits<std::uint32_t>::max()));
            constexpr std::int32_t SegmentProtection = 5;
            constexpr std::uint32_t SegmentCount = 1;
            std::memcpy(
                payload.data() + 0x168,
                &segment_size, sizeof(segment_size));
            std::memcpy(
                payload.data() + 0x16c,
                &SegmentProtection,
                sizeof(SegmentProtection));
            std::memcpy(
                payload.data() + 0x1a0,
                &SegmentCount, sizeof(SegmentCount));
        } else {
            std::memcpy(
                payload.data() + 0x120,
                &image_base, sizeof(image_base));
            std::memcpy(
                payload.data() + 0x128,
                &image_size, sizeof(image_size));
        }
        std::memcpy(
            reinterpret_cast<void*>(output_address),
            payload.data(), required_size);
        result = 0;
        return true;
    }
    if (symbol == "0NTHN1NKONI") {
        const auto local_seconds =
            request.integer_arguments[0];
        const auto utc_output =
            request.integer_arguments[2];
        const auto timezone_output =
            request.integer_arguments[3];
        const auto daylight_output =
            request.integer_arguments[4];
        if (timezone_output == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        const std::array<std::int32_t, 2> timezone{};
        const std::int32_t daylight{};
        const bool wrote =
            TryWriteGuestBytes(
                timezone_output, timezone.data(),
                sizeof(timezone)) &&
            (utc_output == 0 ||
             TryWriteGuestU64(
                 utc_output, local_seconds)) &&
            (daylight_output == 0 ||
             TryWriteGuestBytes(
                 daylight_output, &daylight,
                 sizeof(daylight)));
        result = wrote
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "-o5uEDpN+oY") {
        const auto utc_seconds =
            request.integer_arguments[0];
        const auto local_output =
            request.integer_arguments[1];
        const auto timesec_output =
            request.integer_arguments[2];
        const auto daylight_output =
            request.integer_arguments[3];
        if (local_output == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        std::array<std::uint8_t, 16> timesec{};
        std::memcpy(
            timesec.data(), &utc_seconds,
            sizeof(utc_seconds));
        const std::uint64_t daylight{};
        const bool wrote =
            TryWriteGuestU64(
                local_output, utc_seconds) &&
            (timesec_output == 0 ||
             TryWriteGuestBytes(
                 timesec_output, timesec.data(),
                 timesec.size())) &&
            (daylight_output == 0 ||
             TryWriteGuestU64(
                 daylight_output, daylight));
        result = wrote
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "QBi7HCK03hw" ||
        symbol == "lLMT9vJAck0") {
        const auto clock_id =
            static_cast<std::int32_t>(request.integer_arguments[0]);
        const auto destination = request.integer_arguments[1];
        std::array<std::int64_t, 2> value{};
        if (clock_id == 0 || symbol == "lLMT9vJAck0") {
            const auto now =
                std::chrono::system_clock::now().time_since_epoch();
            const auto nanoseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now).count();
            value[0] = nanoseconds / INT64_C(1000000000);
            value[1] = nanoseconds % INT64_C(1000000000);
        } else {
            const auto nanoseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() -
                    g_ps5_process_start).count();
            value[0] = nanoseconds / INT64_C(1000000000);
            value[1] = nanoseconds % INT64_C(1000000000);
        }
        result = TryWriteGuestBytes(
                     destination, value.data(), sizeof(value))
            ? 0
            : OrbisError(UINT32_C(0x80020016));
        return true;
    }
    if (symbol == "rPo6tV8D9bM") {
        const auto destination = request.integer_arguments[0];
        std::array<std::uint8_t, 0x0c> status{};
        status[0x06] = 1;
        result =
            destination != 0 &&
                    TryWriteGuestBytes(
                        destination, status.data(), status.size())
                ? 0
                : OrbisError(UINT32_C(0x80020016));
        return true;
    }
    if (symbol == "ejekcaNQNq0" ||
        symbol == "n88vx3C5nW8") {
        const auto destination = request.integer_arguments[0];
        const auto now =
            std::chrono::system_clock::now().time_since_epoch();
        const auto microseconds =
            std::chrono::duration_cast<std::chrono::microseconds>(
                now).count();
        const std::array<std::int64_t, 2> value{
            microseconds / INT64_C(1000000),
            microseconds % INT64_C(1000000)};
        const bool wrote_time =
            destination == 0 ||
            TryWriteGuestBytes(
                destination, value.data(), sizeof(value));
        const auto timezone = request.integer_arguments[1];
        const std::uint64_t zero{};
        const bool wrote_timezone =
            timezone == 0 ||
            TryWriteGuestBytes(timezone, &zero, sizeof(zero));
        result = wrote_time && wrote_timezone
            ? 0
            : (symbol == "n88vx3C5nW8"
                   ? std::numeric_limits<std::uint64_t>::max()
                   : OrbisError(UINT32_C(0x80020016)));
        return true;
    }
    if (symbol == "aI+OeCz8xrQ" ||
        symbol == "EotR8a3ASf4") {
        result = CurrentGuestPthreadHandle();
        return true;
    }
    if (symbol == "1tKyG7RlMJo") {
        const auto thread = request.integer_arguments[0];
        const auto priority_output = request.integer_arguments[1];
        if (thread == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        constexpr std::int32_t DefaultGuestPriority = 700;
        result =
            priority_output != 0 &&
                    TryWriteGuestBytes(
                        priority_output, &DefaultGuestPriority,
                        sizeof(DefaultGuestPriority))
                ? 0
                : OrbisError(UINT32_C(0x80020016));
        return true;
    }
    if (symbol == "W0Hpm2X0uPE") {
        result = request.integer_arguments[0] != 0
            ? 0
            : OrbisError(UINT32_C(0x80020003));
        return true;
    }
    if (TryGuestPthreadAttributeOperation(
            symbol, request, result)) {
        return true;
    }
    if (symbol == "3PtV6p3QNX4" ||
        symbol == "7Xl257M4VNI") {
        result = request.integer_arguments[0] ==
                         request.integer_arguments[1]
            ? 1
            : 0;
        return true;
    }
    if (symbol == "cmo1RIYva9o" ||
        symbol == "ttHNfU+qDBU" ||
        symbol == "2Of0f+3mhhE" ||
        symbol == "ltCfaGr2JGE" ||
        symbol == "9UK1vLZQft4" ||
        symbol == "7H0iTOciTLo" ||
        symbol == "upoVrzMHFeE" ||
        symbol == "K-jXhbt2gn4" ||
        symbol == "tn3VlD0hG60" ||
        symbol == "2Z+PpY6CaJg") {
        return TryGuestMutexOperation(
            symbol, request.integer_arguments[0], result);
    }
    if (symbol == "188x57JYp0g" ||
        symbol == "Zxa0VhQVTsk" ||
        symbol == "12wOHk8ywb0" ||
        symbol == "4czppHBiriw" ||
        symbol == "4DM06U2BNEY" ||
        symbol == "R1Jvn8bSCW8") {
        return TryGuestSemaphoreOperation(symbol, request, result);
    }
    if (symbol == "2Tb92quprl0" ||
        symbol == "0TyVk4MSLt0" ||
        symbol == "g+PZd2hiacg" ||
        symbol == "RXXqi4CtF8w" ||
        symbol == "WKAXJ4XBPQ4" ||
        symbol == "BmMjYxmew1w" ||
        symbol == "kDh-NfxgMtE" ||
        symbol == "JGgj7Uvrl+A" ||
        symbol == "Op8TBGY5KHg" ||
        symbol == "27bAgiJmOh0" ||
        symbol == "mkx2fVhNMsg" ||
        symbol == "2MOy+rUfuhQ") {
        return TryGuestConditionVariableOperation(
            symbol, request, result);
    }
    if (symbol == "hVYD7Ou2pCQ" ||
        symbol == "7+OJIpko9RY" ||
        symbol == "asjUJJ+aa8s" ||
        symbol == "JeNX5F-NzQU" ||
        symbol == "uq2Y5BFz0PE" ||
        symbol == "JzyT91ucGDc" ||
        symbol == "a0LLrZWac0M" ||
        symbol == "PFT2S-tJ7Uk" ||
        symbol == "p+zLIOg27zU" ||
        symbol == "B0ZX2hx9DMw" ||
        symbol == "avfGJ94g36Q" ||
        symbol == "ZqhZFuzKT6U") {
        return TryHandleGuestFiber(symbol, request, result);
    }
    if (symbol == "BpFoboUJoZU" ||
        symbol == "8mql9OcQnd4" ||
        symbol == "IOnSvHzqu6A" ||
        symbol == "7uhBFWRAS60" ||
        symbol == "JTvBflhYazQ" ||
        symbol == "9lvj5DjHZiA" ||
        symbol == "PZku4ZrXJqg") {
        return TryHandleGuestEventFlag(
            symbol, request, result);
    }
    if (symbol == "9BcDykPmo1I") {
        result = CurrentThreadTlsScratch(0x40);
        return true;
    }
    if (symbol == "FxVZqBAA7ks") {
        const auto descriptor =
            static_cast<std::int32_t>(request.integer_arguments[0]);
        const auto buffer = request.integer_arguments[1];
        const auto byte_count = request.integer_arguments[2];
        {
            const std::lock_guard lock{g_runtime.mutex};
            if (byte_count != 0 &&
                !HasAccessLocked(
                    buffer, byte_count, LSX4_PS5_GUEST_READ)) {
                result = OrbisError(UINT32_C(0x80020101));
                return true;
            }
        }
        if ((descriptor == STDOUT_FILENO ||
             descriptor == STDERR_FILENO) &&
            byte_count != 0) {
            auto* const stream =
                descriptor == STDOUT_FILENO ? stdout : stderr;
            (void)std::fwrite(
                reinterpret_cast<const void*>(buffer), 1,
                static_cast<std::size_t>(byte_count), stream);
            std::fflush(stream);
#ifdef __ANDROID__
            static std::atomic<std::uint32_t>
                dreaming_stdio_log_count{};
            if (dreaming_stdio_log_count.fetch_add(
                    1, std::memory_order_relaxed) < 8) {
                LogDreamingForeachState("guest-stdio");
            }
            constexpr std::size_t MaxAndroidLogBytes = 2048;
            const auto log_byte_count = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    byte_count, MaxAndroidLogBytes));
            std::string message{
                reinterpret_cast<const char*>(buffer),
                log_byte_count};
            for (auto& value : message) {
                const auto byte =
                    static_cast<unsigned char>(value);
                if (value == '\n' || value == '\r' ||
                    value == '\t') {
                    continue;
                }
                if (byte < 0x20 || byte == 0x7f) {
                    value = '.';
                }
            }
            __android_log_print(
                descriptor == STDERR_FILENO
                    ? ANDROID_LOG_ERROR
                    : ANDROID_LOG_INFO,
                "LSX4-PS5-STDIO",
                "fd=%d bytes=%llu%s %s",
                descriptor,
                static_cast<unsigned long long>(byte_count),
                byte_count > MaxAndroidLogBytes
                    ? " truncated"
                    : "",
                message.c_str());
            if (message.find(
                    "increase size of foreach_instancestack") !=
                std::string::npos) {
                std::uint64_t executable_base{};
                std::vector<std::pair<
                    std::uint64_t, std::uint64_t>> image_ranges;
                std::vector<Funnel::Ps5Desktop::LoadedRange>
                    executable_ranges;
                std::filesystem::path executable_dump_path;
                {
                    const std::lock_guard lock{g_runtime.mutex};
                    const auto executable =
                        std::ranges::find(
                            g_runtime.programs, false,
                            &LoadedProgram::module);
                    if (executable !=
                        g_runtime.programs.end()) {
                        executable_base =
                            executable->image.base;
                        executable_ranges =
                            executable->image.registered_ranges;
                        executable_dump_path =
                            g_runtime.save_data_directory /
                            "eboot-runtime-memory.bin";
                    }
                    image_ranges.reserve(
                        g_runtime.programs.size());
                    for (const auto& program :
                         g_runtime.programs) {
                        image_ranges.emplace_back(
                            program.image.base,
                            program.image.mapped_size);
                    }
                }
                std::int32_t depth{};
                std::uint64_t stack_begin{};
                std::uint64_t stack_end{};
                std::uint64_t stack_capacity{};
                std::array<std::uint64_t, 256> guest_stack_words{};
                (void)TryReadGuestBytes(
                    request.guest_stack,
                    guest_stack_words.data(),
                    sizeof(guest_stack_words));
                const auto readable =
                    executable_base != 0 &&
                    TryReadGuestValue(
                        executable_base + UINT64_C(0x4a0058),
                        depth) &&
                    TryReadGuestValue(
                        executable_base + UINT64_C(0x4bf9f8),
                        stack_begin) &&
                    TryReadGuestValue(
                        executable_base + UINT64_C(0x4bfa00),
                        stack_end) &&
                    TryReadGuestValue(
                        executable_base + UINT64_C(0x4bfa08),
                        stack_capacity);
                __android_log_print(
                    readable ? ANDROID_LOG_ERROR
                             : ANDROID_LOG_WARN,
                    "LSX4-PS5-STDIO",
                    "foreach diagnostics base=0x%llx "
                    "guest_stack=0x%llx "
                    "returns=0x%llx,0x%llx,0x%llx,0x%llx "
                    "depth=%d begin=0x%llx end=0x%llx "
                    "capacity=0x%llx entries=%lld reserved=%lld",
                    static_cast<unsigned long long>(
                        executable_base),
                    static_cast<unsigned long long>(
                        request.guest_stack),
                    static_cast<unsigned long long>(
                        guest_stack_words[0]),
                    static_cast<unsigned long long>(
                        guest_stack_words[1]),
                    static_cast<unsigned long long>(
                        guest_stack_words[2]),
                    static_cast<unsigned long long>(
                        guest_stack_words[3]),
                    depth,
                    static_cast<unsigned long long>(
                        stack_begin),
                    static_cast<unsigned long long>(
                        stack_end),
                    static_cast<unsigned long long>(
                        stack_capacity),
                    stack_end >= stack_begin
                        ? static_cast<long long>(
                              (stack_end - stack_begin) / 0x20u)
                        : -1ll,
                    stack_capacity >= stack_begin
                        ? static_cast<long long>(
                              (stack_capacity - stack_begin) /
                              0x20u)
                        : -1ll);
                for (std::size_t word_index = 0;
                     word_index < guest_stack_words.size();
                     ++word_index) {
                    const auto candidate =
                        guest_stack_words[word_index];
                    for (const auto& [image_base, image_size] :
                         image_ranges) {
                        if (image_base != executable_base) {
                            continue;
                        }
                        if (candidate < image_base ||
                            candidate - image_base >=
                                image_size) {
                            continue;
                        }
                        __android_log_print(
                            ANDROID_LOG_ERROR,
                            "LSX4-PS5-STDIO",
                            "foreach stack candidate "
                            "word=%zu address=0x%llx "
                            "image=0x%llx offset=0x%llx",
                            word_index,
                            static_cast<unsigned long long>(
                                candidate),
                            static_cast<unsigned long long>(
                                image_base),
                            static_cast<unsigned long long>(
                                candidate - image_base));
                    }
                }
                std::error_code dump_error;
                std::filesystem::create_directories(
                    executable_dump_path.parent_path(),
                    dump_error);
                if (!dump_error &&
                    !std::filesystem::exists(
                        executable_dump_path, dump_error) &&
                    !dump_error) {
                    FILE* const dump = std::fopen(
                        executable_dump_path.string().c_str(),
                        "wb");
                    bool dump_ok = dump != nullptr;
                    if (dump != nullptr) {
                        for (const auto& range :
                             executable_ranges) {
                            if (range.address < executable_base ||
                                range.byte_count == 0 ||
                                fseeko(
                                    dump,
                                    static_cast<off_t>(
                                        range.address -
                                        executable_base),
                                    SEEK_SET) != 0 ||
                                std::fwrite(
                                    reinterpret_cast<const void*>(
                                        range.address),
                                    1,
                                    static_cast<std::size_t>(
                                        range.byte_count),
                                    dump) != range.byte_count) {
                                dump_ok = false;
                                break;
                            }
                        }
                        std::fclose(dump);
                    }
                    __android_log_print(
                        dump_ok ? ANDROID_LOG_INFO
                                : ANDROID_LOG_ERROR,
                        "LSX4-PS5-STDIO",
                        "foreach executable dump path=%s "
                        "ranges=%zu status=%s",
                        executable_dump_path.string().c_str(),
                        executable_ranges.size(),
                        dump_ok ? "ok" : "failed");
                }
            }
#endif
            result = byte_count;
            return true;
        }
        result = OrbisError(UINT32_C(0x80020009));
        return true;
    }
    if (symbol == "CdWp0oHWGr0") {
        constexpr std::int32_t PrimaryUserId = 0x10000000;
        result = TryWriteGuestBytes(
                     request.integer_arguments[0],
                     &PrimaryUserId, sizeof(PrimaryUserId))
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "fPhymKNvK-A") {
        constexpr std::array<std::int32_t, 4> UserIds{
            0x10000000, -1, -1, -1};
        result = TryWriteGuestBytes(
                     request.integer_arguments[0],
                     UserIds.data(), sizeof(UserIds))
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "hv1luiJrqQM") {
        result = 0;
        return true;
    }
    if (symbol == "xk0AcarP3V4" ||
        symbol == "WFIiSfXGUq8" ||
        symbol == "u1GRHp+oWoY") {
        result = 1;
        return true;
    }
    if (symbol == "6ncge5+l5Qs" ||
        symbol == "clVvL4ZDntw" ||
        symbol == "vDLMoJLde8I" ||
        symbol == "W2G-yoyMF5U" ||
        symbol == "2JgFB2n9oUM") {
        result = 0;
        return true;
    }
    if (symbol == "gjP9-KQzoUk" ||
        symbol == "hGbf2QTBmqc") {
        std::array<std::uint8_t, 0x40> information{};
        constexpr float TouchDensity = 44.86f;
        constexpr std::uint16_t TouchWidth = 1920;
        constexpr std::uint16_t TouchHeight = 943;
        std::memcpy(
            information.data(), &TouchDensity,
            sizeof(TouchDensity));
        std::memcpy(
            information.data() + 0x04, &TouchWidth,
            sizeof(TouchWidth));
        std::memcpy(
            information.data() + 0x06, &TouchHeight,
            sizeof(TouchHeight));
        information[0x08] = 30;
        information[0x09] = 30;
        information[0x0b] = 1;
        information[0x0c] = 1;
        if (symbol == "hGbf2QTBmqc") {
            information[0x1d] = 1;
        }
        const auto bytes = symbol == "hGbf2QTBmqc"
            ? information.size()
            : std::size_t{0x1c};
        result = TryWriteGuestBytes(
                     request.integer_arguments[1],
                     information.data(), bytes)
            ? 0
            : OrbisError(UINT32_C(0x80020016));
        return true;
    }
    if (symbol == "AcslpN1jHR8") {
        const std::array<std::uint8_t, 0x20> information{};
        result = TryWriteGuestBytes(
                     request.integer_arguments[1],
                     information.data(), information.size())
            ? 0
            : OrbisError(UINT32_C(0x80020016));
        return true;
    }
    if (symbol == "YndgXqQVV7c" ||
        symbol == "q1cHNfGycLI" ||
        symbol == "5Wf4q349s+Q" ||
        symbol == "QjwkT2Ycmew") {
        std::array<std::uint8_t, 0x78> data{};
        const auto buttons =
            g_ps5_pad_buttons.load(std::memory_order_relaxed);
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> last_logged_buttons{
            std::numeric_limits<std::uint32_t>::max()};
        const auto previous_buttons =
            last_logged_buttons.exchange(
                buttons, std::memory_order_relaxed);
        if (previous_buttons != buttons) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "pad guest sample symbol=%s buttons=0x%08x "
                "previous=0x%08x data=0x%llx",
                symbol.c_str(), buttons, previous_buttons,
                static_cast<unsigned long long>(
                    request.integer_arguments[1]));
        }
        static std::array<std::uint8_t, 6> last_logged_axes{
            128, 128, 128, 128, 0, 0};
        std::array<std::uint8_t, 6> sampled_axes{};
        for (std::size_t axis = 0; axis < sampled_axes.size();
             ++axis) {
            sampled_axes[axis] =
                g_ps5_pad_axes[axis].load(
                    std::memory_order_relaxed);
        }
        if (sampled_axes != last_logged_axes) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "pad guest axes=%u,%u,%u,%u,%u,%u",
                sampled_axes[0], sampled_axes[1],
                sampled_axes[2], sampled_axes[3],
                sampled_axes[4], sampled_axes[5]);
            last_logged_axes = sampled_axes;
        }
#endif
        std::memcpy(data.data(), &buttons, sizeof(buttons));
        for (std::size_t axis = 0; axis < 6; ++axis) {
            data[0x04 + axis] =
                g_ps5_pad_axes[axis].load(
                    std::memory_order_relaxed);
        }
        constexpr float OrientationW = 1.0f;
        std::memcpy(
            data.data() + 0x18, &OrientationW,
            sizeof(OrientationW));
        data[0x4c] = 1;
        const auto timestamp =
            g_ps5_pad_timestamp.load(std::memory_order_relaxed);
        std::memcpy(
            data.data() + 0x50, &timestamp,
            sizeof(timestamp));
        data[0x68] = 1;
        if (!TryWriteGuestBytes(
                request.integer_arguments[1],
                data.data(), data.size())) {
            result = OrbisError(UINT32_C(0x80020016));
        } else {
            result =
                symbol == "q1cHNfGycLI" ||
                symbol == "QjwkT2Ycmew"
                ? 1
                : 0;
        }
        return true;
    }
    if (symbol == "Up36PTk687E") {
        constexpr auto InvalidValue = UINT32_C(0x80290001);
        const auto user_id = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto bus_type = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        const auto index = static_cast<std::int32_t>(
            request.integer_arguments[2]);
        if ((user_id != 0 && user_id != 255) ||
            bus_type != 0 || index != 0) {
            result = OrbisError(InvalidValue);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        auto handle = g_runtime.next_video_out_handle++;
        while (handle <= 0 ||
               g_runtime.guest_video_out_ports.contains(handle)) {
            handle = g_runtime.next_video_out_handle++;
        }
        g_runtime.guest_video_out_ports.emplace(
            handle, GuestVideoOutPort{});
        result = static_cast<std::uint64_t>(handle);
        return true;
    }
    if (symbol == "utPrVdxio-8") {
        constexpr auto InvalidAddress = UINT32_C(0x80290002);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto output = request.integer_arguments[1];
        if (output == 0) {
            result = OrbisError(InvalidAddress);
            return true;
        }
        std::array<std::uint8_t, 0x30> status{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port == g_runtime.guest_video_out_ports.end()) {
                result = OrbisError(InvalidHandle);
                return true;
            }
            const std::int32_t resolution_class =
                port->second.width >= 3840 ||
                        port->second.height >= 2160
                    ? 2
                    : 1;
            constexpr std::int32_t DynamicRange = 1;
            constexpr std::uint64_t RefreshRate59_94Hz = 3;
            std::memcpy(
                status.data() + 0x00, &resolution_class,
                sizeof(resolution_class));
            std::memcpy(
                status.data() + 0x04, &DynamicRange,
                sizeof(DynamicRange));
            std::memcpy(
                status.data() + 0x08, &RefreshRate59_94Hz,
                sizeof(RefreshRate59_94Hz));
        }
        result = TryWriteGuestBytes(
                     output, status.data(), status.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "j6RaAUlaLv0") {
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        auto now = std::chrono::steady_clock::now();
        auto target = now;
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port == g_runtime.guest_video_out_ports.end()) {
                result = OrbisError(InvalidHandle);
                return true;
            }
            constexpr auto Period =
                std::chrono::microseconds(16667);
            target = port->second.last_vblank + Period;
            if (target <= now || target > now + Period) {
                target = now;
            }
            port->second.last_vblank = target;
        }
        if (target > now) {
            std::this_thread::sleep_until(target);
        }
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port != g_runtime.guest_video_out_ports.end()) {
                ++port->second.vblank_count;
            }
        }
        result = 0;
        return true;
    }
    if (symbol == "1FZBKy8HeNU") {
        constexpr auto InvalidAddress = UINT32_C(0x80290002);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto output = request.integer_arguments[1];
        if (output == 0) {
            result = OrbisError(InvalidAddress);
            return true;
        }
        std::array<std::uint8_t, 0x28> status{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port == g_runtime.guest_video_out_ports.end()) {
                result = OrbisError(InvalidHandle);
                return true;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_us =
                std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    now - port->second.opened_at)
                    .count();
            const auto elapsed_count =
                static_cast<std::uint64_t>(
                    std::max<std::int64_t>(elapsed_us, 0) * 60 /
                    1000000);
            port->second.vblank_count = std::max(
                port->second.vblank_count, elapsed_count);
            const auto count = port->second.vblank_count;
            const auto process_counter = static_cast<std::uint64_t>(
                now.time_since_epoch().count());
            const auto elapsed =
                static_cast<std::uint64_t>(
                    std::max<std::int64_t>(elapsed_us, 0));
            std::memcpy(
                status.data() + 0x00, &count, sizeof(count));
            std::memcpy(
                status.data() + 0x08, &elapsed,
                sizeof(elapsed));
            std::memcpy(
                status.data() + 0x18, &process_counter,
                sizeof(process_counter));
        }
        result = TryWriteGuestBytes(
                     output, status.data(), status.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "SbU3dwp80lQ") {
        constexpr auto InvalidAddress = UINT32_C(0x80290002);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto output = request.integer_arguments[1];
        if (output == 0) {
            result = OrbisError(InvalidAddress);
            return true;
        }
        std::array<std::uint8_t, 0x80> status{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port == g_runtime.guest_video_out_ports.end()) {
                result = OrbisError(InvalidHandle);
                return true;
            }
            std::memcpy(
                status.data() + 0x00, &port->second.flip_count,
                sizeof(port->second.flip_count));
            std::memcpy(
                status.data() + 0x08,
                &port->second.last_flip_process_time,
                sizeof(port->second.last_flip_process_time));
            std::memcpy(
                status.data() + 0x18,
                &port->second.last_flip_argument,
                sizeof(port->second.last_flip_argument));
            std::memcpy(
                status.data() + 0x28,
                &port->second.last_flip_process_counter,
                sizeof(port->second.last_flip_process_counter));
            std::memcpy(
                status.data() + 0x38,
                &port->second.current_buffer,
                sizeof(port->second.current_buffer));
            std::memcpy(
                status.data() + 0x40,
                &port->second.last_flip_submit_counter,
                sizeof(port->second.last_flip_submit_counter));
        }
        result = TryWriteGuestBytes(
                     output, status.data(), status.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "zgXifHT9ErY" ||
        symbol == "MTxxrOCeSig") {
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.guest_video_out_ports.contains(handle)) {
            result = OrbisError(InvalidHandle);
            return true;
        }
        result = 0;
        return true;
    }
    if (symbol == "N5KDtkIjjJ4") {
        constexpr auto InvalidValue = UINT32_C(0x80290001);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto group_index = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        if (group_index < 0) {
            result = OrbisError(InvalidValue);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto port =
            g_runtime.guest_video_out_ports.find(handle);
        if (port == g_runtime.guest_video_out_ports.end()) {
            result = OrbisError(InvalidHandle);
            return true;
        }
        for (std::size_t index = 0;
             index < port->second.buffer_addresses.size(); ++index) {
            if (port->second.buffer_group_indices[index] ==
                group_index) {
                port->second.buffer_addresses[index] = 0;
                port->second.buffer_group_indices[index] = -1;
            }
        }
        result = 0;
        return true;
    }
    if (symbol == "Xru92wHJRmg") {
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        constexpr auto InvalidEventQueue = UINT32_C(0x8029000c);
        const auto event_queue = request.integer_arguments[0];
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        const auto user_data = request.integer_arguments[2];
        const std::lock_guard lock{g_runtime.mutex};
        const auto queue =
            g_runtime.guest_event_queues.find(event_queue);
        const auto port =
            g_runtime.guest_video_out_ports.find(handle);
        if (port == g_runtime.guest_video_out_ports.end()) {
            result = OrbisError(InvalidHandle);
            return true;
        }
        if (queue == g_runtime.guest_event_queues.end()) {
            result = OrbisError(InvalidEventQueue);
            return true;
        }
        auto& registrations = port->second.vblank_events;
        const auto existing = std::ranges::find(
            registrations, event_queue,
            &GuestVideoOutPort::FlipEventRegistration::event_queue);
        if (existing != registrations.end()) {
            existing->user_data = user_data;
        } else {
            registrations.push_back(
                {.event_queue = event_queue,
                 .user_data = user_data});
        }
        StartGuestVblankThreadLocked();
        result = 0;
        return true;
    }
    if (symbol == "CBiu4mCE1DA") {
        constexpr auto InvalidValue = UINT32_C(0x80290001);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto rate = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        if (rate < 0 || rate > 2) {
            result = OrbisError(InvalidValue);
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto port =
            g_runtime.guest_video_out_ports.find(handle);
        if (port == g_runtime.guest_video_out_ports.end()) {
            result = OrbisError(InvalidHandle);
            return true;
        }
        port->second.flip_rate = rate;
        result = 0;
        return true;
    }
    if (symbol == "U46NwOiJpys") {
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        constexpr auto InvalidIndex = UINT32_C(0x8029000d);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto buffer_index = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        const auto flip_mode = static_cast<std::uint32_t>(
            request.integer_arguments[2]);
        const auto flip_argument = request.integer_arguments[3];
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto port =
                g_runtime.guest_video_out_ports.find(handle);
            if (port == g_runtime.guest_video_out_ports.end()) {
                result = OrbisError(InvalidHandle);
                return true;
            }
            if (buffer_index < -2 || buffer_index >= 16 ||
                (buffer_index >= 0 &&
                 port->second.buffer_addresses[
                     static_cast<std::size_t>(buffer_index)] == 0)) {
                result = OrbisError(InvalidIndex);
                return true;
            }
        }
        ObserveAgcFlip(
            handle, buffer_index, flip_mode, flip_argument);
        result = 0;
        return true;
    }
    if (symbol == "PjS5uASwcV8") {
        std::array<std::uint8_t, 0x50> attribute{};
        const auto output = request.integer_arguments[0];
        const auto pixel_format = request.integer_arguments[1];
        const auto tiling_mode = static_cast<std::uint32_t>(
            request.integer_arguments[2]);
        const auto width = static_cast<std::uint32_t>(
            request.integer_arguments[3]);
        const auto height = static_cast<std::uint32_t>(
            request.integer_arguments[4]);
        const auto option = request.integer_arguments[5];
        std::uint32_t dcc_control{};
        std::uint64_t dcc_clear_color{};
        if (output == 0 ||
            !TryReadGuestBytes(
                request.guest_stack + 0x08,
                &dcc_control, sizeof(dcc_control)) ||
            !TryReadGuestBytes(
                request.guest_stack + 0x10,
                &dcc_clear_color, sizeof(dcc_clear_color))) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        std::memcpy(
            attribute.data() + 0x04,
            &tiling_mode, sizeof(tiling_mode));
        std::memcpy(
            attribute.data() + 0x0c,
            &width, sizeof(width));
        std::memcpy(
            attribute.data() + 0x10,
            &height, sizeof(height));
        std::memcpy(
            attribute.data() + 0x18,
            &option, sizeof(option));
        std::memcpy(
            attribute.data() + 0x20,
            &pixel_format, sizeof(pixel_format));
        std::memcpy(
            attribute.data() + 0x28,
            &dcc_clear_color, sizeof(dcc_clear_color));
        std::memcpy(
            attribute.data() + 0x30,
            &dcc_control, sizeof(dcc_control));
        result = TryWriteGuestBytes(
                     output, attribute.data(), attribute.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "rKBUtgRrtbk") {
        constexpr auto InvalidValue = UINT32_C(0x80290001);
        constexpr auto InvalidAddress = UINT32_C(0x80290002);
        constexpr auto InvalidHandle = UINT32_C(0x8029000b);
        constexpr auto InvalidOption = UINT32_C(0x8029001a);
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto set_index = static_cast<std::int32_t>(
            request.integer_arguments[1]);
        const auto start_index = static_cast<std::int32_t>(
            request.integer_arguments[2]);
        const auto buffers_address = request.integer_arguments[3];
        const auto buffer_count = static_cast<std::int32_t>(
            request.integer_arguments[4]);
        const auto attribute_address = request.integer_arguments[5];
        std::uint64_t category_raw{};
        std::uint64_t option{};
        std::array<std::uint8_t, 0x50> attribute{};
        if (!TryReadGuestBytes(
                request.guest_stack + 0x08,
                &category_raw, sizeof(category_raw)) ||
            !TryReadGuestBytes(
                request.guest_stack + 0x10,
                &option, sizeof(option))) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        if (buffers_address == 0) {
            result = OrbisError(InvalidAddress);
            return true;
        }
        if (attribute_address == 0) {
            result = OrbisError(InvalidOption);
            return true;
        }
        if (start_index < 0 || buffer_count <= 0 ||
            start_index >= 16 || buffer_count > 16 ||
            start_index + buffer_count > 16 ||
            static_cast<std::uint32_t>(category_raw) > 1 ||
            option != 0) {
            result = OrbisError(InvalidValue);
            return true;
        }
        if (!TryReadGuestBytes(
                attribute_address,
                attribute.data(), attribute.size())) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        std::array<std::uint64_t, 16> addresses{};
        for (std::int32_t buffer = 0;
             buffer < buffer_count; ++buffer) {
            if (!TryReadGuestBytes(
                    buffers_address +
                        static_cast<std::uint64_t>(buffer) * 0x20,
                    &addresses[static_cast<std::size_t>(buffer)],
                    sizeof(std::uint64_t))) {
                result = OrbisError(UINT32_C(0x80020101));
                return true;
            }
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto port =
            g_runtime.guest_video_out_ports.find(handle);
        if (port == g_runtime.guest_video_out_ports.end()) {
            result = OrbisError(InvalidHandle);
            return true;
        }
        auto& state = port->second;
        for (std::int32_t buffer = 0;
             buffer < buffer_count; ++buffer) {
            const auto slot =
                static_cast<std::size_t>(start_index + buffer);
            state.buffer_addresses[slot] =
                addresses[static_cast<std::size_t>(buffer)];
            state.buffer_group_indices[slot] = set_index;
        }
        std::memcpy(
            &state.tiling_mode,
            attribute.data() + 0x04, sizeof(state.tiling_mode));
        std::memcpy(
            &state.width,
            attribute.data() + 0x0c, sizeof(state.width));
        std::memcpy(
            &state.height,
            attribute.data() + 0x10, sizeof(state.height));
        std::memcpy(
            &state.option,
            attribute.data() + 0x18, sizeof(state.option));
        std::memcpy(
            &state.pixel_format,
            attribute.data() + 0x20, sizeof(state.pixel_format));
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5",
            "video register handle=%d start=%d count=%d "
            "address0=0x%llx address1=0x%llx size=%ux%u "
            "format=0x%llx tile=%u",
            handle, start_index, buffer_count,
            static_cast<unsigned long long>(addresses[0]),
            static_cast<unsigned long long>(
                buffer_count > 1 ? addresses[1] : 0),
            state.width, state.height,
            static_cast<unsigned long long>(state.pixel_format),
            state.tiling_mode);
#endif
        result = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(set_index));
        return true;
    }
    if (symbol == "D0OdFMjp46I") {
        const auto output = request.integer_arguments[0];
        if (output == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        std::uint64_t handle{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            handle = g_runtime.next_event_queue_handle++;
            if (handle == 0) {
                handle = g_runtime.next_event_queue_handle++;
            }
            g_runtime.guest_event_queues.try_emplace(handle);
            if (handle >= 2u) {
                g_runtime.diagnostic_second_event_queue_created.store(
                    true, std::memory_order_release);
            }
        }
        result = TryWriteGuestU64(output, handle)
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "HXzjK9yI30k") {
        const auto event_queue = request.integer_arguments[0];
        const auto handle =
            static_cast<std::int32_t>(request.integer_arguments[1]);
        const auto user_data = request.integer_arguments[2];
        const std::lock_guard lock{g_runtime.mutex};
        const auto queue =
            g_runtime.guest_event_queues.find(event_queue);
        const auto port =
            g_runtime.guest_video_out_ports.find(handle);
        if (queue == g_runtime.guest_event_queues.end() ||
            port == g_runtime.guest_video_out_ports.end()) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        auto& registrations = port->second.flip_events;
        const auto existing = std::ranges::find(
            registrations, event_queue,
            &GuestVideoOutPort::FlipEventRegistration::event_queue);
        if (existing != registrations.end()) {
            existing->user_data = user_data;
        } else {
            registrations.push_back(
                {.event_queue = event_queue,
                 .user_data = user_data});
        }
        result = 0;
        return true;
    }
    if (symbol == "fzyMKs9kim0") {
        const auto event_queue = request.integer_arguments[0];
        const auto events = request.integer_arguments[1];
        const auto capacity = request.integer_arguments[2];
        const auto out_count = request.integer_arguments[3];
        const auto timeout = request.integer_arguments[4];
#ifdef __ANDROID__
        static std::atomic<std::uint64_t> wait_trace_count{};
        const auto wait_index =
            wait_trace_count.fetch_add(1, std::memory_order_relaxed) + 1u;
        const bool trace_wait =
            Ps5DiagnosticFaultProbeEnabled() && wait_index <= 64u;
        if (trace_wait) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-EQUEUE",
                "wait enter index=%llu queue=0x%llx events=0x%llx "
                "capacity=%llu out=0x%llx timeout=0x%llx",
                static_cast<unsigned long long>(wait_index),
                static_cast<unsigned long long>(event_queue),
                static_cast<unsigned long long>(events),
                static_cast<unsigned long long>(capacity),
                static_cast<unsigned long long>(out_count),
                static_cast<unsigned long long>(timeout));
        }
#endif
        if (events == 0 || capacity == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        std::uint64_t timeout_microseconds{};
        if (timeout != 0 &&
            !TryReadGuestValue(timeout, timeout_microseconds)) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        std::array<std::uint8_t, 0x20> event{};
        {
            std::unique_lock lock{g_runtime.mutex};
            auto queue =
                g_runtime.guest_event_queues.find(event_queue);
            if (queue == g_runtime.guest_event_queues.end()) {
                result = OrbisError(UINT32_C(0x80020002));
                return true;
            }
            if (queue->second.empty()) {
                if (timeout != 0) {
                    (void)g_runtime.guest_event_queue_condition.wait_for(
                        lock,
                        std::chrono::microseconds(
                            timeout_microseconds),
                        [&] {
                            const auto current =
                                g_runtime.guest_event_queues.find(
                                    event_queue);
                            return current ==
                                       g_runtime.guest_event_queues.end() ||
                                   !current->second.empty() ||
                                   g_runtime.shutdown_requested.load(
                                       std::memory_order_acquire);
                        });
                } else {
                    g_runtime.guest_event_queue_condition.wait(
                        lock,
                        [&] {
                            const auto current =
                                g_runtime.guest_event_queues.find(
                                    event_queue);
                            return current ==
                                       g_runtime.guest_event_queues.end() ||
                                   !current->second.empty() ||
                                   g_runtime.shutdown_requested.load(
                                       std::memory_order_acquire);
                        });
                }
                queue = g_runtime.guest_event_queues.find(event_queue);
                if (queue == g_runtime.guest_event_queues.end()) {
                    result = OrbisError(UINT32_C(0x80020002));
                    return true;
                }
            }
            if (queue->second.empty()) {
                const std::uint32_t zero{};
                lock.unlock();
                if (out_count != 0) {
                    (void)TryWriteGuestValue(out_count, zero);
                }
                result = timeout != 0
                    ? OrbisError(UINT32_C(0x8002003c))
                    : 0;
                return true;
            }
            event = queue->second.front();
            queue->second.pop_front();
        }
        const std::uint32_t one = 1;
        if (!TryWriteGuestBytes(
                events, event.data(), event.size()) ||
            (out_count != 0 &&
             !TryWriteGuestValue(out_count, one))) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
#ifdef __ANDROID__
        if (trace_wait) {
            std::uint64_t ident{};
            std::int16_t filter{};
            std::uint16_t flags{};
            std::uint64_t data{};
            std::uint64_t user_data{};
            std::memcpy(&ident, event.data(), sizeof(ident));
            std::memcpy(&filter, event.data() + 0x08, sizeof(filter));
            std::memcpy(&flags, event.data() + 0x0a, sizeof(flags));
            std::memcpy(&data, event.data() + 0x10, sizeof(data));
            std::memcpy(
                &user_data, event.data() + 0x18, sizeof(user_data));
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5-EQUEUE",
                "wait deliver index=%llu queue=0x%llx ident=0x%llx "
                "filter=%d flags=0x%x data=0x%llx user=0x%llx",
                static_cast<unsigned long long>(wait_index),
                static_cast<unsigned long long>(event_queue),
                static_cast<unsigned long long>(ident),
                static_cast<int>(filter),
                static_cast<unsigned int>(flags),
                static_cast<unsigned long long>(data),
                static_cast<unsigned long long>(user_data));
        }
#endif
        result = 0;
        return true;
    }
    if (symbol == "clVvL4ZDntw") {
        result = 0;
        return true;
    }
    if (symbol == "gjP9-KQzoUk") {
        std::array<std::uint8_t, 0x1c> information{};
        const float touch_resolution = 44.86f;
        const std::uint16_t touch_width = 1920;
        const std::uint16_t touch_height = 943;
        std::memcpy(
            information.data() + 0x00,
            &touch_resolution, sizeof(touch_resolution));
        std::memcpy(
            information.data() + 0x04,
            &touch_width, sizeof(touch_width));
        std::memcpy(
            information.data() + 0x06,
            &touch_height, sizeof(touch_height));
        information[0x08] = 30;
        information[0x09] = 30;
        information[0x0b] = 1;
        information[0x0c] = 1;
        result = TryWriteGuestBytes(
                     request.integer_arguments[1],
                     information.data(), information.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "dyIhnXq-0SM") {
        constexpr std::uint32_t Zero = 0;
        const auto output = request.integer_arguments[1];
        const auto hit_written =
            TryWriteGuestBytes(output, &Zero, sizeof(Zero));
        const auto set_written =
            TryWriteGuestBytes(
                output + 0x14, &Zero, sizeof(Zero));
        result = hit_written && set_written
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "ZP4e7rlzOUk") {
        std::array<std::uint8_t, 0x40> mount_result{};
        constexpr std::string_view MountPoint{"/savedata0"};
        std::memcpy(
            mount_result.data(), MountPoint.data(),
            MountPoint.size());
        const auto output = request.integer_arguments[1];
        result = TryWriteGuestBytes(
                     output, mount_result.data(),
                     mount_result.size())
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        if (result == 0) {
            std::filesystem::path save_directory;
            {
                const std::lock_guard lock{g_runtime.mutex};
                save_directory = g_runtime.save_data_directory;
            }
            std::error_code error;
            std::filesystem::create_directories(
                save_directory, error);
            if (error) {
                result = OrbisError(UINT32_C(0x809f000b));
            }
        }
        return true;
    }
    if (symbol == "gjRZNnw0JPE") {
        const auto rdx = request.integer_arguments[2];
        const auto rcx = request.integer_arguments[3];
        const auto output =
            rdx != 0 && rdx <= UINT16_MAX ? rcx : rdx;
        constexpr std::uint32_t Resource = 1;
        if (output != 0) {
            (void)TryWriteGuestBytes(
                output, &Resource, sizeof(Resource));
        }
        result = 0;
        return true;
    }
    if (symbol == "uW4vfTwMQVo" ||
        symbol == "sDCBrmc61XU" ||
        symbol == "85zul--eGXs" ||
        symbol == "TywrFKCoLGY") {
        result = 0;
        return true;
    }
    if (symbol == "2JtWUUiYBXs" ||
        symbol == "wRbq6ZjNop4") {
        const auto version =
            static_cast<std::uint32_t>(
                request.integer_arguments[0]);
        if (version != 7 && version != 8 &&
            version != 10 && version != 13) {
            result = 0;
            return true;
        }
        return TryGetAgcRegisterDefaults(
            symbol == "wRbq6ZjNop4", result);
    }
    if (symbol == "23LRUSvYu1M") {
        result = 0;
        return true;
    }
    if (symbol == "f3dg2CSgRKY") {
        return TryCreateAgcShader(request, result);
    }
    if (symbol == "D9sr1xGUriE") {
        return TryAgcCreatePrimState(request, result);
    }
    if (symbol == "HV4j+E0MBHE") {
        return TryAgcCreateInterpolantMapping(request, result);
    }
    if (symbol == "TRO721eVt4g") {
        return TryAgcResetDrawQueue(request, result);
    }
    if (symbol == "qj7QZpgr9Uw") {
        return TryAgcType2Packet(request, result);
    }
    if (symbol == "LtTouSCZjHM") {
        return TryAgcNop(request, result);
    }
    if (symbol == "aJf+j5yntiU") {
        return TryAgcEventWrite(request, result);
    }
    if (symbol == "57labkp+rSQ") {
        return TryAgcAcquireMem(request, result);
    }
    if (symbol == "wr23dPKyWc0") {
        return TryAgcReleaseMem(request, result);
    }
    if (symbol == "VmW0Tdpy420") {
        return TryAgcWaitRegMem(request, result);
    }
    if (symbol == "WmAc2MEj6Io") {
        return TryAgcDmaData(request, result);
    }
    if (symbol == "IxYiarKlXxM") {
        return TryAgcPatchDmaDataAddress(
            request, false, result);
    }
    if (symbol == "cdDRpqcFGbU") {
        return TryAgcPatchDmaDataAddress(
            request, true, result);
    }
    if (symbol == "3KDcnM3lrcU") {
        return TryAgcPatchWaitRegMemAddress(request, result);
    }
    if (symbol == "0fWWK5uG9rQ") {
        result = TryWriteGuestValue(
                     request.integer_arguments[0] + 12u,
                     request.integer_arguments[1])
            ? 0
            : OrbisError(UINT32_C(0x80020101));
        return true;
    }
    if (symbol == "ZvwO9euwYzc") {
        return TryAgcSetRegistersIndirect(
            request, AgcRCxRegsIndirect, result);
    }
    if (symbol == "-HOOCn0JY48") {
        return TryAgcSetRegistersIndirect(
            request, AgcRShRegsIndirect, result);
    }
    if (symbol == "hvUfkUIQcOE") {
        return TryAgcSetRegistersIndirect(
            request, AgcRUcRegsIndirect, result);
    }
    if (symbol == "vcmNN+AAXnY" ||
        symbol == "Qrj4c+61z4A" ||
        symbol == "6lNcCp+fxi4") {
        return TryAgcPatchIndirectAddress(request, result);
    }
    if (symbol == "d-6uF9sZDIU" ||
        symbol == "z2duB-hHQSM" ||
        symbol == "vRoArM9zaIk") {
        return TryAgcPatchIndirectCount(request, result);
    }
    if (symbol == "n2fD4A+pb+g") {
        return TryAgcSetShRegisterRange(request, result);
    }
    if (symbol == "V++UgBtQhn0") {
        return TryAgcGetDataPacketPayload(request, result);
    }
    if (symbol == "MWiElSNE8j8") {
        return TryAgcWaitUntilSafeForRendering(request, result);
    }
    if (symbol == "YUeqkyT7mEQ") {
        return TryAgcSetFlip(request, result);
    }
    if (symbol == "GIIW2J37e70") {
        return TryAgcSetIndexSize(request, result);
    }
    if (symbol == "8N2tmT3jmC8") {
        return TryAgcSetIndexCount(request, result);
    }
    if (symbol == "mljzuGDZRQ4") {
        result = 7u * sizeof(std::uint32_t);
        return true;
    }
    if (symbol == "tSBxhAPyytQ") {
        return TryAgcSetNumInstances(request, result);
    }
    if (symbol == "l4fM9K-Lyks") {
        return TryAgcSetIndexBuffer(request, result);
    }
    if (symbol == "B+aG9DUnTKA") {
        return TryAgcDrawIndexOffset(request, result);
    }
    if (symbol == "k3GhuSNmBLU") {
        return TryAgcCbDispatch(request, result);
    }
    if (symbol == "UZbQjYAwwXM") {
        return TryAgcCbSetShRegistersDirect(request, result);
    }
    if (symbol == "JrtiDtKeS38") {
        return TryAgcAcbResetQueue(request, result);
    }
    if (symbol == "cFazmnXpJOE") {
        return TryAgcAcbEventWrite(request, result);
    }
    if (symbol == "KT-hTp-Ch14") {
        return TryAgcAcbAcquireMem(request, result);
    }
    if (symbol == "htn36gPnBk4") {
        return TryAgcAcbWaitRegMem(request, result);
    }
    if (symbol == "j3EtxFkSIhQ") {
        return TryAgcAcbDispatchIndirect(request, result);
    }
    if (symbol == "eZ4+17OQz4Q" ||
        symbol == "i1jyy49AjXU") {
        return TryAgcWriteData(request, result);
    }
    if (symbol == "Yw0jKSqop+E") {
        return TryAgcDcbDrawIndexAuto(request, result);
    }
    if (symbol == "RmaJwLtc8rY") {
        return TryAgcDcbSetBaseIndirectArgs(request, result);
    }
    if (symbol == "CtB+A9-VxO0") {
        return TryAgcDcbDispatchIndirect(request, result);
    }
    if (symbol == "H7uZqCoNuWk") {
        return TryAgcDcbPopMarker(request, result);
    }
    if (symbol == "vuSXe69VILM") {
        return TryAgcDcbGetLodStats(request, result);
    }
    if (symbol == "gSRnr79F8tQ") {
        return TryAgcDriverSubmitAcb(request, result);
    }
    if (symbol == "UglJIZjGssM") {
        return TryAgcDriverSubmitDcb(request, result);
    }
    if (symbol == "JfEPXVxhFqA") {
        result = 0;
        return true;
    }
    if (symbol == "h9z6+0hEydk") {
        result = 0;
        return true;
    }
    if (symbol == "ekNvsT22rsY") {
        const auto buffer_length = static_cast<std::uint32_t>(
            request.integer_arguments[3]);
        const auto frequency = static_cast<std::uint32_t>(
            request.integer_arguments[4]);
        const auto format = static_cast<std::int32_t>(
            request.integer_arguments[5]);
        const auto base_format =
            static_cast<std::uint32_t>(format) & 0xffu;
        const auto channels = base_format == 0 || base_format == 3
            ? 1u
            : (base_format == 1 || base_format == 4
                   ? 2u
                   : (base_format >= 2 && base_format <= 7 ? 8u : 0u));
        const auto bytes_per_sample =
            base_format >= 3 && base_format <= 5 ||
                    base_format == 7
                ? 4u
                : 2u;
        if (buffer_length == 0 || frequency == 0 ||
            channels == 0) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        GuestAudioOutPort port{
            .buffer_length = buffer_length,
            .frequency = frequency,
            .format = format,
            .channels = channels,
            .bytes_per_sample = bytes_per_sample,
        };
#ifdef __ANDROID__
        (void)OpenGuestAudioOut(port);
#endif
        const std::lock_guard lock{g_runtime.mutex};
        auto handle = g_runtime.next_audio_out_handle++;
        while (handle <= 0 ||
               g_runtime.guest_audio_out_ports.contains(handle)) {
            handle = g_runtime.next_audio_out_handle++;
        }
        g_runtime.guest_audio_out_ports.emplace(
            handle, port);
        result = static_cast<std::uint64_t>(handle);
        return true;
    }
    if (symbol == "b+uAV89IlxE") {
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        const auto flags =
            static_cast<std::uint32_t>(request.integer_arguments[1]);
        const auto volumes = request.integer_arguments[2];
        std::array<std::int32_t, 8> channel_volumes{};
        if (volumes != 0 &&
            !TryReadGuestBytes(
                volumes, channel_volumes.data(),
                sizeof(channel_volumes))) {
            result = OrbisError(UINT32_C(0x80020101));
            return true;
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto found =
            g_runtime.guest_audio_out_ports.find(handle);
        if (found == g_runtime.guest_audio_out_ports.end()) {
            result = OrbisError(UINT32_C(0x80020003));
            return true;
        }
        float volume{};
        for (std::size_t channel = 0;
             channel < channel_volumes.size(); ++channel) {
            if ((flags & (1u << channel)) != 0) {
                volume = std::max(
                    volume,
                    static_cast<float>(channel_volumes[channel]) /
                        32768.0f);
            }
        }
        if (flags != 0) {
            found->second.volume =
                std::clamp(volume, 0.0f, 1.0f);
        }
        result = 0;
        return true;
    }
    if (symbol == "QOQtbeDqsT4") {
        const auto handle = static_cast<std::int32_t>(
            request.integer_arguments[0]);
        GuestAudioOutPort port{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto found =
                g_runtime.guest_audio_out_ports.find(handle);
            if (found == g_runtime.guest_audio_out_ports.end()) {
                result = OrbisError(UINT32_C(0x80020003));
                return true;
            }
            ++found->second.output_count;
            port = found->second;
        }
        const auto duration =
            std::chrono::microseconds(
                std::max<std::uint64_t>(
                    1,
                    static_cast<std::uint64_t>(
                        port.buffer_length) *
                        UINT64_C(1000000) /
                        port.frequency));
#ifdef __ANDROID__
        if (!OutputGuestAudio(
                port, request.integer_arguments[1],
                port.output_count)) {
            std::this_thread::sleep_for(duration);
        }
#else
        std::this_thread::sleep_for(duration);
#endif
        result = 0;
        if (g_runtime.shutdown_requested.load(
                std::memory_order_acquire)) {
            (void)Lsx4::Translation::CompleteCurrentGuestExecution(0);
        }
        return true;
    }
    if (symbol == "6UgtwV+0zb4" ||
        symbol == "OxhIB8LB-PQ" ||
        symbol == "Jmi+9w9u0E4") {
        return TryCreateGuestPthread(request, result);
    }
    if (symbol == "3kg7rT0NQIs" ||
        symbol == "FJrT5LuUBAU" ||
        symbol == "6Z83sYWFlA8") {
        result = request.integer_arguments[0];
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-EXIT",
            "thread exit symbol=%s value=0x%llx stack=0x%llx",
            symbol.c_str(),
            static_cast<unsigned long long>(result),
            static_cast<unsigned long long>(request.guest_stack));
#endif
        (void)Lsx4::Translation::CompleteCurrentGuestExecution(result);
        return true;
    }
    if (symbol == "pO96TwzOm5E") {
        result = Ps5DirectMemorySize;
        return true;
    }
    if (symbol == "aNz11fnnzi4") {
        const auto output_size = request.integer_arguments[0];
        constexpr std::uint64_t AvailableFlexibleMemory =
            UINT64_C(512) * 1024 * 1024;
        result =
            output_size != 0 &&
                    TryWriteGuestU64(
                        output_size, AvailableFlexibleMemory)
                ? 0
                : OrbisError(UINT32_C(0x80020003));
        return true;
    }
    if (symbol == "7oxv3PPCumo") {
        return TryReserveVirtualRange(request, result);
    }
    if (symbol == "IWIBBdTHit4") {
        return TryMapFlexibleMemory(request, result);
    }
    if (symbol == "mL8NDH86iQI") {
        return TryMapFlexibleMemory(request, result);
    }
    if (symbol == "rTXw65xmLIA") {
        return TryAllocateDirectMemory(request, result);
    }
    if (symbol == "C0f7TJcbfac") {
        return TryAvailableDirectMemorySize(request, result);
    }
    if (symbol == "B+vc2AO2Zrc") {
        auto translated = request;
        translated.integer_arguments[0] = 0;
        translated.integer_arguments[1] = Ps5DirectMemorySize;
        translated.integer_arguments[2] =
            request.integer_arguments[0];
        translated.integer_arguments[3] =
            request.integer_arguments[1];
        translated.integer_arguments[4] =
            request.integer_arguments[2];
        translated.integer_arguments[5] =
            request.integer_arguments[3];
        return TryAllocateDirectMemory(translated, result);
    }
    if (symbol == "L-Q3LEjIbgA") {
        return TryMapDirectMemory(request, result);
    }
    if (symbol == "2SKEx6bSq-4") {
        return TryKernelBatchMap(request, result);
    }
    return false;
}

void SetRuntimeStatus(std::string status) {
    const std::lock_guard lock{g_runtime.mutex};
    g_runtime.status = std::move(status);
}

void ResetPs5TranslationSession() noexcept {
    try {
        const auto generation =
            g_translation_session_generation.fetch_add(
                1, std::memory_order_relaxed);
        Lsx4::Translation::ConfigureArtifactStore(
            "", "ps5-session-" + std::to_string(generation),
            generation, false);
    } catch (...) {
    }
}

int RuntimeStrcmp(const char* left, const char* right) {
    if (left == nullptr || right == nullptr) {
        return left == right ? 0 : (left == nullptr ? -1 : 1);
    }
    return std::strcmp(left, right);
}

std::uint64_t InvokeAndroidFallbackHle(
    void*, const Lsx4Ps5HleCall* const call) {
#ifdef __ANDROID__
    if (call != nullptr) {
        const auto symbol = HleSymbol(call->function);
        static std::mutex trace_mutex;
        static std::unordered_map<std::string, std::uint64_t>
            trace_counts;
        std::uint64_t count{};
        {
            const std::lock_guard lock{trace_mutex};
            count = ++trace_counts[symbol];
        }
        if (count <= 2 || (count & (count - 1u)) == 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "hle nid=%s count=%llu thunk=0x%llx "
                "a0=0x%llx a1=0x%llx a2=0x%llx",
                symbol.empty() ? "(unknown)" : symbol.c_str(),
                static_cast<unsigned long long>(count),
                static_cast<unsigned long long>(call->function),
                static_cast<unsigned long long>(
                    call->integer_arguments[0]),
                static_cast<unsigned long long>(
                    call->integer_arguments[1]),
                static_cast<unsigned long long>(
                    call->integer_arguments[2]));
        }
        const bool trace_after_second_equeue =
            Ps5DiagnosticFaultProbeEnabled() &&
            g_runtime.diagnostic_second_event_queue_created.load(
                std::memory_order_acquire);
        static std::atomic<std::uint64_t> post_equeue_trace_count{};
        const bool noisy_worker_call =
            symbol == "JTvBflhYazQ" ||
            symbol == "7uhBFWRAS60" ||
            symbol == "T72hz6ffq08";
        if (trace_after_second_equeue && !noisy_worker_call) {
            const auto trace_index =
                post_equeue_trace_count.fetch_add(
                    1, std::memory_order_relaxed) + 1u;
            if (trace_index <= 768u) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5-POSTEQ",
                    "call=%llu nid=%s thunk=0x%llx "
                    "a0=0x%llx a1=0x%llx a2=0x%llx "
                    "a3=0x%llx a4=0x%llx a5=0x%llx stack=0x%llx",
                    static_cast<unsigned long long>(trace_index),
                    symbol.empty() ? "(unknown)" : symbol.c_str(),
                    static_cast<unsigned long long>(call->function),
                    static_cast<unsigned long long>(
                        call->integer_arguments[0]),
                    static_cast<unsigned long long>(
                        call->integer_arguments[1]),
                    static_cast<unsigned long long>(
                        call->integer_arguments[2]),
                    static_cast<unsigned long long>(
                        call->integer_arguments[3]),
                    static_cast<unsigned long long>(
                        call->integer_arguments[4]),
                    static_cast<unsigned long long>(
                        call->integer_arguments[5]),
                    static_cast<unsigned long long>(
                        call->guest_stack));
                }
            }
        }
#endif
    return 0;
}

}

extern "C" const char* executor_lsx4_ps5_runtime_abi() {
    return "lsx4-ps5-runtime/7";
}

extern "C" const char* executor_lsx4_ps5_runtime_status() {
    static thread_local std::string snapshot;
    const std::lock_guard lock{g_runtime.mutex};
    snapshot = g_runtime.status;
    return snapshot.c_str();
}

extern "C" int executor_lsx4_ps5_runtime_attach_surface(
    void* const native_window) {
#ifdef __ANDROID__
    auto* const window =
        static_cast<ANativeWindow*>(native_window);
    if (window == nullptr) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    ANativeWindow_acquire(window);
    ANativeWindow* previous{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        previous =
            std::exchange(g_runtime.android_window, window);
    }
    Lsx4::Ps5Desktop::ResetVulkanPresenter();
    if (previous != nullptr) {
        ANativeWindow_release(previous);
    }
    TryPresentGameSplashToAndroid();
    return LSX4_PS5_EXECUTE_OK;
#else
    (void)native_window;
    return LSX4_PS5_EXECUTE_BACKEND_NOT_READY;
#endif
}

extern "C" int executor_lsx4_ps5_runtime_detach_surface() {
#ifdef __ANDROID__
    ANativeWindow* previous{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        previous =
            std::exchange(g_runtime.android_window, nullptr);
    }
    Lsx4::Ps5Desktop::ResetVulkanPresenter();
    if (previous != nullptr) {
        ANativeWindow_release(previous);
    }
    return LSX4_PS5_EXECUTE_OK;
#else
    return LSX4_PS5_EXECUTE_BACKEND_NOT_READY;
#endif
}

extern "C" int executor_lsx4_ps5_runtime_initialize_android(
    const char* const root_directory,
    const char* const user_id) {
    static std::mutex initialize_mutex;
    const std::lock_guard initialize_lock{initialize_mutex};
    if (root_directory == nullptr || root_directory[0] == '\0') {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Lsx4Ps5RuntimeConfig config{};
    config.artifact_directory = root_directory;
    config.title_id =
        user_id != nullptr && user_id[0] != '\0'
        ? user_id
        : "android";
    bool already_initialized{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        already_initialized = g_runtime.initialized;
    }
    const auto status = already_initialized
        ? LSX4_PS5_EXECUTE_OK
        : executor_lsx4_ps5_runtime_initialize(&config);
    if (status != LSX4_PS5_EXECUTE_OK) {
        return status;
    }
    Lsx4Ps5RuntimeCallbacks callbacks{};
    callbacks.invoke_hle = &InvokeAndroidFallbackHle;
    return executor_lsx4_ps5_runtime_set_callbacks(&callbacks);
}

extern "C" int executor_lsx4_ps5_runtime_scan_game(
    const char* const game_path) {
    if (game_path == nullptr || game_path[0] == '\0') {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Lsx4Ps5LoadedExecutable executable{};
    Lsx4Ps5GameProbeReport report{};
    const auto status = executor_lsx4_ps5_runtime_load_game(
        game_path, &executable, &report);
    if (status != LSX4_PS5_EXECUTE_OK) {
        return status;
    }
    return executor_lsx4_ps5_runtime_unload_eboot(
        executable.handle);
}

extern "C" int executor_lsx4_ps5_runtime_launch_game_jit(
    const char* const game_path) {
    if (game_path == nullptr || game_path[0] == '\0') {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Lsx4Ps5LoadedExecutable executable{};
    Lsx4Ps5GameProbeReport report{};
    auto status = executor_lsx4_ps5_runtime_load_game(
        game_path, &executable, &report);
    if (status != LSX4_PS5_EXECUTE_OK) {
        return status;
    }
    Lsx4Ps5GuestResult result{};
    status = executor_lsx4_ps5_runtime_launch_eboot(
        executable.handle, nullptr, &result);
    if (status == LSX4_PS5_EXECUTE_OK) {
        status = result.status;
    }
    const std::string launch_detail =
        executor_lsx4_ps5_runtime_status();
    if (status == LSX4_PS5_EXECUTE_OK) {
        SetRuntimeStatus("PS5 Android guest active");
        return status;
    }
    (void)executor_lsx4_ps5_runtime_unload_eboot(
        executable.handle);
    SetRuntimeStatus(
        "PS5 Android launch failed: " + launch_detail);
    return status;
}

extern "C" int executor_lsx4_ps5_runtime_set_pad_button(
    const std::uint32_t button_mask, const int pressed) {
    bool changed{};
    if (pressed != 0) {
        const auto previous = g_ps5_pad_buttons.fetch_or(
            button_mask, std::memory_order_relaxed);
        changed = (previous & button_mask) != button_mask;
    } else {
        const auto previous = g_ps5_pad_buttons.fetch_and(
            ~button_mask, std::memory_order_relaxed);
        changed = (previous & button_mask) != 0u;
    }
    if (changed) {
        g_ps5_pad_timestamp.store(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    std::chrono::steady_clock::now() -
                    g_ps5_process_start).count()),
            std::memory_order_relaxed);
    }
    return LSX4_PS5_EXECUTE_OK;
}

extern "C" int executor_lsx4_ps5_runtime_set_pad_axis(
    const int axis, const int value) {
    if (axis < 0 ||
        axis >= static_cast<int>(g_ps5_pad_axes.size()) ||
        value < 0 || value > 255) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    const auto previous =
        g_ps5_pad_axes[static_cast<std::size_t>(axis)].exchange(
            static_cast<std::uint8_t>(value),
            std::memory_order_relaxed);
    if (previous != static_cast<std::uint8_t>(value)) {
        g_ps5_pad_timestamp.store(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<
                    std::chrono::microseconds>(
                    std::chrono::steady_clock::now() -
                    g_ps5_process_start).count()),
            std::memory_order_relaxed);
#ifdef __ANDROID__
        static std::atomic<std::uint32_t> axis_logs{};
        if (axis_logs.fetch_add(
                1u, std::memory_order_relaxed) < 64u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4-PS5",
                "pad host axis=%d value=%d previous=%u",
                axis, value, previous);
        }
#endif
    }
    return LSX4_PS5_EXECUTE_OK;
}

extern "C" int executor_lsx4_ps5_runtime_hud_stats(
    std::uint64_t* const values, const std::size_t value_count) {
    constexpr std::size_t ValueCount = 21u;
    if (values == nullptr || value_count < ValueCount) {
        return -1;
    }
    const auto cpu = Lsx4::Translation::ReadTranslationCounters();
    std::uint64_t draw_count{};
    std::uint64_t submit_count{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        draw_count = g_runtime.agc_draw_count;
        submit_count = g_runtime.agc_submit_count;
    }
    values[0] = cpu.decoded_regions;
    values[1] = cpu.native_regions;
    values[2] = cpu.semantic_regions;
    values[3] = cpu.unsupported_regions;
    values[4] = cpu.memory_cache_hits;
    values[5] = draw_count;
    values[6] = draw_count;
    values[7] = 0u;
    values[8] = submit_count;
    values[9] = submit_count;
#ifdef __ANDROID__
    values[10] = Lsx4::Ps5Desktop::GetVulkanPresentCount();
#else
    values[10] = 0u;
#endif
    values[11] = cpu.stored_ir_enabled;
    values[12] = cpu.stored_ir_loaded;
    values[13] = cpu.stored_ir_hits;
    values[14] = cpu.stored_native_hits;
    values[15] = cpu.stored_native_captured;
    values[16] = cpu.stored_ir_written;
    values[17] = cpu.stored_native_written;
    values[18] = 0u;
    values[19] = 0u;
    values[20] = 0u;
    return static_cast<int>(ValueCount);
}

extern "C" int executor_lsx4_ps5_runtime_set_managed_optimization(
    const int option, const int enabled) {
    constexpr int TieredJit = 4;
    constexpr int JitTraceCompilation = 5;
    static std::atomic<bool> tiered_jit{false};
    static std::atomic<bool> trace_compilation{false};
    const bool active = enabled != 0;
    switch (option) {
    case 1:
    case 2:
    case 3:
        return LSX4_PS5_EXECUTE_OK;
    case TieredJit:
        tiered_jit.store(active, std::memory_order_release);
        break;
    case JitTraceCompilation:
        trace_compilation.store(active, std::memory_order_release);
        if (active) {
            tiered_jit.store(true, std::memory_order_release);
        }
        break;
    default:
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Executor::Jit::ConfigureTieredJit(
        tiered_jit.load(std::memory_order_acquire),
        trace_compilation.load(std::memory_order_acquire));
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4-PS5",
        "managed JIT option=%d enabled=%d tiered=%d trace=%d",
        option, active ? 1 : 0,
        Executor::Jit::TieredJitEnabled() ? 1 : 0,
        Executor::Jit::JitTraceCompilationEnabled() ? 1 : 0);
#endif
    return LSX4_PS5_EXECUTE_OK;
}

extern "C" int executor_lsx4_ps5_runtime_initialize(
    const Lsx4Ps5RuntimeConfig* const config) {
    if (config == nullptr || config->size < sizeof(Lsx4Ps5RuntimeConfig) ||
        config->abi_version != LSX4_PS5_RUNTIME_ABI_VERSION ||
        config->guest_page_size != LSX4_PS5_GUEST_PAGE_SIZE ||
        config->guest_address_limit <= 0x10000) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    executor_lsx4_ps5_runtime_reset();
    g_ps5_pad_buttons.store(0, std::memory_order_relaxed);
    g_ps5_pad_timestamp.store(0, std::memory_order_relaxed);
    for (std::size_t axis = 0;
         axis < g_ps5_pad_axes.size(); ++axis) {
        g_ps5_pad_axes[axis].store(
            axis < 4 ? std::uint8_t{128} : std::uint8_t{0},
            std::memory_order_relaxed);
    }
    const auto compatibility =
        Lsx4::Ps5Desktop::InspectBackendCompatibility();
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.initialized = true;
        g_runtime.shutdown_requested.store(
            false, std::memory_order_release);
        g_runtime.guest_address_limit = config->guest_address_limit;
        g_runtime.preferred_image_base =
            Ps5ImageBase +
            (g_translation_session_generation.load(
                 std::memory_order_relaxed) %
             Ps5ImageSessionSlots) *
                Ps5ImageSessionStride;
        g_runtime.status = compatibility.ready
            ? "PS5 runtime initialized"
            : "PS5 runtime initialized; CPU compatibility gates remain";
        if (!EnsureHleDataLocked()) {
            g_runtime.initialized = false;
            g_runtime.status =
                "PS5 runtime HLE data initialization failed";
        }
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
        }
    }

    InstallPs5DiagnosticFaultHandlers();
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_set_callbacks(
    const Lsx4Ps5RuntimeCallbacks* const callbacks) {
    if (callbacks == nullptr ||
        callbacks->size < sizeof(Lsx4Ps5RuntimeCallbacks) ||
        callbacks->abi_version != LSX4_PS5_RUNTIME_ABI_VERSION) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    const std::lock_guard lock{g_runtime.mutex};
    g_runtime.callbacks = *callbacks;
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_register_mapping(
    const Lsx4Ps5GuestMapping* const mapping) {
    if (mapping == nullptr ||
        mapping->size < sizeof(Lsx4Ps5GuestMapping)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    const std::lock_guard lock{g_runtime.mutex};
    if (!g_runtime.initialized) {
        return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
    }
    return RegisterMappingLocked(mapping->guest_address, mapping->byte_count,
                                 mapping->protection, mapping->label, false)
        ? 0
        : LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
}

extern "C" int executor_lsx4_ps5_runtime_unregister_mapping(
    const std::uint64_t guest_address, const std::uint64_t byte_count) {
    const std::lock_guard lock{g_runtime.mutex};
    const auto removed =
        g_runtime.mappings.EraseExact(guest_address, byte_count, false);
    return removed != 0 ? 0 : LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
}

extern "C" int executor_lsx4_ps5_runtime_execute(
    const Lsx4Ps5GuestEntry* const entry,
    Lsx4Ps5GuestResult* const result) {
    if (entry == nullptr || result == nullptr ||
        entry->size < sizeof(Lsx4Ps5GuestEntry) ||
        result->size < sizeof(Lsx4Ps5GuestResult) ||
        entry->argument_count > std::size(entry->arguments)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *result = {};

    std::uint64_t execution_thread_handle{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            result->status = LSX4_PS5_EXECUTE_NOT_INITIALIZED;
            return result->status;
        }
        if (!HasAccessLocked(entry->address, 1, LSX4_PS5_GUEST_EXECUTE)) {
            result->status = LSX4_PS5_EXECUTE_UNMAPPED_ENTRY;
            return result->status;
        }
        const auto thread = std::ranges::find(
            g_runtime.threads, entry->fs_base,
            &GuestThreadContext::fs_base);
        if (thread != g_runtime.threads.end()) {
            execution_thread_handle = thread->handle;
        }
    }

#ifndef __ANDROID__
    const auto compatibility =
        Lsx4::Ps5Desktop::InspectBackendCompatibility();
    if (!compatibility.ready) {
        result->status = LSX4_PS5_EXECUTE_BACKEND_NOT_READY;
        return result->status;
    }
#endif

    Lsx4::Translation::EntryRequest request{};
    std::copy_n(entry->arguments, entry->argument_count,
                request.arguments.begin());
    request.argument_count = entry->argument_count;
    request.stack_argument_mode =
        entry->argument_count == 6 &&
                entry->arguments[0] != 0
            ? 2
            : 0;
    request.thread_segment_origin = entry->fs_base;
    request.supplied_stack_base = entry->stack_base;
    request.supplied_stack_bytes = entry->stack_size;
    request.permit_return_sentinel = true;

    InstallPs5DiagnosticFaultHandlers();
    const auto previous_thread_handle = std::exchange(
        g_current_guest_thread_handle, execution_thread_handle);
    const auto previous_fs_base = std::exchange(
        g_current_guest_fs_base, entry->fs_base);
    try {
        result->value =
            Lsx4::Translation::ExecuteGuest(entry->address, request);
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-EXIT",
            "execute returned thread=%llu entry=0x%llx value=0x%llx",
            static_cast<unsigned long long>(execution_thread_handle),
            static_cast<unsigned long long>(entry->address),
            static_cast<unsigned long long>(result->value));
#endif
    } catch (const std::exception& exception) {
        g_current_guest_thread_handle = previous_thread_handle;
        g_current_guest_fs_base = previous_fs_base;
        result->status = LSX4_PS5_EXECUTE_INTERNAL_ERROR;
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4-PS5-EXIT",
            "execute exception thread=%llu entry=0x%llx detail=%s",
            static_cast<unsigned long long>(execution_thread_handle),
            static_cast<unsigned long long>(entry->address),
            exception.what());
        executor_jit_dump_thread_states("ps5-execute-exception");
#endif
        SetRuntimeStatus(
            std::string{"PS5 guest execution exception: "} +
            exception.what());
        return result->status;
    } catch (...) {
        g_current_guest_thread_handle = previous_thread_handle;
        g_current_guest_fs_base = previous_fs_base;
        result->status = LSX4_PS5_EXECUTE_INTERNAL_ERROR;
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4-PS5-EXIT",
            "execute unknown exception thread=%llu entry=0x%llx",
            static_cast<unsigned long long>(execution_thread_handle),
            static_cast<unsigned long long>(entry->address));
        executor_jit_dump_thread_states(
            "ps5-execute-unknown-exception");
#endif
        SetRuntimeStatus("PS5 guest execution raised a host exception");
        return result->status;
    }
    g_current_guest_thread_handle = previous_thread_handle;
    g_current_guest_fs_base = previous_fs_base;
    result->status = LSX4_PS5_EXECUTE_OK;
    return result->status;
}

extern "C" int executor_lsx4_ps5_runtime_load_eboot(
    const char* const path,
    Lsx4Ps5LoadedExecutable* const executable) {
    if (path == nullptr || executable == nullptr ||
        executable->size < sizeof(Lsx4Ps5LoadedExecutable)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *executable = {};
    const std::lock_guard module_load_lock{g_module_load_mutex};
    std::uint64_t handle{};
    Funnel::Ps5Desktop::LoaderOptions loader_options{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
        }
        handle = g_runtime.next_program_handle++;
        if (handle == 0) {
            handle = g_runtime.next_program_handle++;
        }
        loader_options.preferred_image_base =
            g_runtime.preferred_image_base;
    }

    Funnel::Ps5Desktop::LoadedImage image{};
    std::string error;
    auto loader_context = SnapshotLoaderContext(handle, false);
    const auto host = Ps5LoaderHost(&loader_context);
    if (!Funnel::Ps5Desktop::LoadNextGenExecutable(
            path, host, image, error, loader_options)) {
        SetRuntimeStatus("PS5 eboot load failed: " + error);
        return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
    }

    bool accepted{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (g_runtime.initialized) {
            g_runtime.programs.push_back(
                {.handle = handle,
                 .owner_handle = 0,
                 .module = false,
                 .started = false,
                 .start_at_boot = false,
                 .path = path,
                 .image = std::move(image)});
            g_runtime.status = "PS5 eboot mapped and relocated";
            accepted = true;
        }
    }
    if (!accepted) {
        Funnel::Ps5Desktop::UnloadNextGenExecutable(host, image);
        return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
    }

    const std::lock_guard lock{g_runtime.mutex};
    const auto found = std::ranges::find(
        g_runtime.programs, handle, &LoadedProgram::handle);
    if (found == g_runtime.programs.end()) {
        return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
    }
    const auto& loaded = found->image;
    executable->size = sizeof(Lsx4Ps5LoadedExecutable);
    executable->flags =
        (loaded.next_gen ? LSX4_PS5_EXECUTABLE_NEXT_GEN : 0u) |
        (loaded.self_container ? LSX4_PS5_EXECUTABLE_SELF : 0u);
    executable->handle = handle;
    executable->base = loaded.base;
    executable->entry = loaded.entry;
    executable->mapped_size = loaded.mapped_size;
    executable->tls_image = loaded.tls_image;
    executable->tls_file_size = loaded.tls_file_size;
    executable->tls_memory_size = loaded.tls_memory_size;
    executable->tls_alignment = loaded.tls_alignment;
    executable->tls_static_offset = loaded.tls_static_offset;
    executable->tls_module_id = loaded.tls_module_id;
    executable->process_param = loaded.process_param;
    executable->owner_handle = 0;
    executable->initializer_count = static_cast<std::uint32_t>(
        loaded.preinitializer_functions.size() +
        loaded.initializer_functions.size());
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_load_module(
    const std::uint64_t owner_handle, const char* const path,
    Lsx4Ps5LoadedExecutable* const module) {
    if (owner_handle == 0 || path == nullptr || module == nullptr ||
        module->size < sizeof(Lsx4Ps5LoadedExecutable)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *module = {};
    const std::lock_guard module_load_lock{g_module_load_mutex};
    Funnel::Ps5Desktop::LoaderOptions loader_options{
        .tls_module_id = 1,
    };
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto owner = std::ranges::find(
            g_runtime.programs, owner_handle, &LoadedProgram::handle);
        if (!g_runtime.initialized) {
            return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
        }
        if (owner == g_runtime.programs.end() || owner->module) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        loader_options.preferred_image_base =
            g_runtime.preferred_image_base;
        for (const auto& program : g_runtime.programs) {
            if (program.handle != owner_handle &&
                program.owner_handle != owner_handle) {
                continue;
            }
            loader_options.tls_module_id = std::max(
                loader_options.tls_module_id,
                program.image.tls_module_id + 1);
            loader_options.previous_tls_static_offset = std::max(
                loader_options.previous_tls_static_offset,
                program.image.tls_static_offset);
        }
    }

    Funnel::Ps5Desktop::LoadedImage image{};
    std::string error;
    auto loader_context = SnapshotLoaderContext(owner_handle, false);
    const auto host = Ps5LoaderHost(&loader_context);
    if (!Funnel::Ps5Desktop::LoadNextGenExecutable(
            path, host, image, error, loader_options)) {
        SetRuntimeStatus("PS5 module load failed: " + error);
        return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
    }

    std::uint64_t handle{};
    bool accepted{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto owner = std::ranges::find(
            g_runtime.programs, owner_handle, &LoadedProgram::handle);
        if (g_runtime.initialized && owner != g_runtime.programs.end() &&
            !owner->module) {
            if (!InstallLateTlsModuleLocked(
                    owner_handle, image, error)) {
                g_runtime.status =
                    "PS5 late module TLS installation failed: " + error;
            } else {
            handle = g_runtime.next_program_handle++;
            if (handle == 0) {
                handle = g_runtime.next_program_handle++;
            }
            g_runtime.programs.push_back(
                {.handle = handle,
                 .owner_handle = owner_handle,
                 .module = true,
                 .started = false,
                 .start_at_boot =
                     ModuleStartsAtBoot(std::filesystem::path{path}),
                 .path = path,
                 .image = std::move(image)});
            g_runtime.status = "PS5 module mapped and relocated";
            accepted = true;
            }
        }
    }
    if (!accepted) {
        Funnel::Ps5Desktop::UnloadNextGenExecutable(host, image);
        return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
    }
    if (!RebindProgramFamily(owner_handle, error)) {
        Funnel::Ps5Desktop::LoadedImage rolled_back_image{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto found = std::ranges::find(
                g_runtime.programs, handle, &LoadedProgram::handle);
            if (found != g_runtime.programs.end()) {
                RollbackLateTlsModuleLocked(
                    owner_handle, handle, found->image);
                rolled_back_image = std::move(found->image);
                g_runtime.programs.erase(found);
            }
        }
        Funnel::Ps5Desktop::UnloadNextGenExecutable(
            host, rolled_back_image);
        SetRuntimeStatus("PS5 intermodule rebind failed: " + error);
        return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
    }

    const std::lock_guard lock{g_runtime.mutex};
    const auto found = std::ranges::find(
        g_runtime.programs, handle, &LoadedProgram::handle);
    if (found == g_runtime.programs.end()) {
        return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
    }
    const auto& loaded = found->image;
    module->size = sizeof(Lsx4Ps5LoadedExecutable);
    module->flags =
        (loaded.next_gen ? LSX4_PS5_EXECUTABLE_NEXT_GEN : 0u) |
        (loaded.self_container ? LSX4_PS5_EXECUTABLE_SELF : 0u) |
        LSX4_PS5_EXECUTABLE_MODULE;
    module->handle = handle;
    module->base = loaded.base;
    module->entry = loaded.entry;
    module->mapped_size = loaded.mapped_size;
    module->tls_image = loaded.tls_image;
    module->tls_file_size = loaded.tls_file_size;
    module->tls_memory_size = loaded.tls_memory_size;
    module->tls_alignment = loaded.tls_alignment;
    module->tls_static_offset = loaded.tls_static_offset;
    module->tls_module_id = loaded.tls_module_id;
    module->process_param = loaded.process_param;
    module->owner_handle = owner_handle;
    module->initializer_count = static_cast<std::uint32_t>(
        loaded.initializer_functions.size());
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_probe_report(
    const std::uint64_t owner_handle,
    Lsx4Ps5GameProbeReport* const report,
    char* const text, const std::size_t text_capacity) {
    if (owner_handle == 0 || report == nullptr ||
        report->size < sizeof(Lsx4Ps5GameProbeReport) ||
        (text == nullptr && text_capacity != 0)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Lsx4Ps5GameProbeReport generated{};
    std::string generated_text;
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
        }
        if (!FillGameProbeLocked(
                owner_handle, generated, generated_text)) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
    }
    *report = generated;
    if (text != nullptr && text_capacity != 0) {
        const auto copied =
            std::min(text_capacity - 1, generated_text.size());
        std::memcpy(text, generated_text.data(), copied);
        text[copied] = '\0';
    }
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_load_game(
    const char* const game_path,
    Lsx4Ps5LoadedExecutable* const executable,
    Lsx4Ps5GameProbeReport* const report) {
    if (game_path == nullptr || executable == nullptr ||
        executable->size < sizeof(Lsx4Ps5LoadedExecutable) ||
        report == nullptr ||
        report->size < sizeof(Lsx4Ps5GameProbeReport)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }

    std::error_code filesystem_error;
    std::filesystem::path requested{game_path};
    std::filesystem::path eboot_path;
    std::filesystem::path game_directory;
    if (std::filesystem::is_directory(requested, filesystem_error)) {
        game_directory = requested;
        eboot_path = game_directory / "eboot.bin";
    } else {
        eboot_path = requested;
        game_directory = requested.parent_path();
    }
    if (filesystem_error ||
        !std::filesystem::is_regular_file(
            eboot_path, filesystem_error) ||
        filesystem_error) {
        SetRuntimeStatus("PS5 game probe cannot find eboot.bin");
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }

    const auto load_status =
        executor_lsx4_ps5_runtime_load_eboot(
            eboot_path.string().c_str(), executable);
    if (load_status != 0) {
        return load_status;
    }
    const auto owner_handle = executable->handle;
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.app0_directory =
            game_directory.lexically_normal();
        g_runtime.save_data_directory =
            (game_directory / ".lsx4-savedata")
                .lexically_normal();
#ifdef __ANDROID__
        g_runtime.game_splash_active = false;
        g_runtime.game_splash_presented = false;
#endif
        constexpr std::size_t ExpectedLargeTitleFiles =
            64u * 1024u;
        g_runtime.apr_files.reserve(ExpectedLargeTitleFiles);
        g_runtime.apr_file_ids_by_path.reserve(
            ExpectedLargeTitleFiles);
        g_runtime.resolved_guest_paths.reserve(
            ExpectedLargeTitleFiles);
        g_runtime.resolved_guest_file_sizes.reserve(
            ExpectedLargeTitleFiles);
    }
#ifdef __ANDROID__
    TryPresentGameSplashToAndroid();
#endif
    std::unordered_map<std::string, std::uint64_t>
        indexed_guest_file_sizes;
    const auto file_size_manifest =
        game_directory / ".lsx4-file-sizes.bin";
    if (LoadGuestFileSizeIndex(
            file_size_manifest, indexed_guest_file_sizes)) {
        const auto indexed_count =
            indexed_guest_file_sizes.size();
        {
            const std::lock_guard lock{g_runtime.mutex};
            g_runtime.indexed_guest_file_sizes =
                std::move(indexed_guest_file_sizes);
        }
#ifdef __ANDROID__
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4-PS5-APR",
            "file-size-index entries=%zu path=%s",
            indexed_count,
            file_size_manifest.string().c_str());
#endif
    }

    struct ModuleCandidate {
        std::filesystem::path path;
    };
    std::vector<ModuleCandidate> modules;
    const std::array module_directories{
        game_directory / "sce_module",
        game_directory / "sce_modules",
        game_directory / "Media" / "Modules",
        game_directory / "Media" / "Plugins",
    };
    for (const auto& directory : module_directories) {
        filesystem_error.clear();
        if (!std::filesystem::is_directory(
                directory, filesystem_error) ||
            filesystem_error) {
            continue;
        }
        std::filesystem::directory_iterator iterator{
            directory, filesystem_error};
        const std::filesystem::directory_iterator end;
        while (!filesystem_error && iterator != end) {
            const auto& entry = *iterator;
            if (entry.is_regular_file(filesystem_error) &&
                !filesystem_error) {
                const auto extension =
                    LowerAscii(entry.path().extension().string());
                if (extension == ".prx" || extension == ".sprx") {
                    modules.push_back({entry.path()});
                }
            }
            iterator.increment(filesystem_error);
        }
        if (filesystem_error) {
            SetRuntimeStatus(
                "PS5 module directory scan failed: " +
                directory.string());
        }
    }
    std::ranges::sort(
        modules, [](const ModuleCandidate& left,
                    const ModuleCandidate& right) {
            return LowerAscii(left.path.string()) <
                   LowerAscii(right.path.string());
        });
    const auto duplicate = std::ranges::unique(
        modules, [](const ModuleCandidate& left,
                    const ModuleCandidate& right) {
            return LowerAscii(left.path.string()) ==
                   LowerAscii(right.path.string());
        });
    modules.erase(duplicate.begin(), duplicate.end());

    std::uint32_t failed_modules{};
    for (const auto& candidate : modules) {
        const auto file_name =
            LowerAscii(candidate.path.filename().string());
        if (file_name == "libkernel.prx" ||
            file_name == "libkernel_sys.prx") {
            continue;
        }
        Lsx4Ps5LoadedExecutable module{};
        const auto status = executor_lsx4_ps5_runtime_load_module(
            owner_handle, candidate.path.string().c_str(), &module);
        if (status != 0) {
            ++failed_modules;
        }
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto owner = std::ranges::find(
            g_runtime.programs, owner_handle, &LoadedProgram::handle);
        if (owner == g_runtime.programs.end() || owner->module) {
            return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
        }
        owner->module_scan_complete = true;
        owner->discovered_modules =
            static_cast<std::uint32_t>(modules.size());
        owner->failed_modules = failed_modules;
    }

    DumpProgramFamilyForDiagnostics(owner_handle);
    std::array<char, 2048> probe_text{};
    const auto probe_status =
        executor_lsx4_ps5_runtime_probe_report(
            owner_handle, report, probe_text.data(), probe_text.size());
    if (probe_status != 0) {
        return probe_status;
    }
    SetRuntimeStatus(probe_text.data());
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_unload_eboot(
    const std::uint64_t handle) {
    bool unloading_module{};
    bool module_started{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (found == g_runtime.programs.end()) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        unloading_module = found->module;
        module_started = found->started;
    }
    if (unloading_module) {
        if (module_started) {
            Lsx4Ps5GuestResult stop_result{};
            const auto stop_status =
                executor_lsx4_ps5_runtime_stop_module(
                    handle, &stop_result);
            if (stop_status != LSX4_PS5_EXECUTE_OK) {
                return stop_status;
            }
        }

        const std::lock_guard module_load_lock{g_module_load_mutex};
        Funnel::Ps5Desktop::LoadedImage image{};
        {
            const std::lock_guard lock{g_runtime.mutex};
            const auto found = std::ranges::find(
                g_runtime.programs, handle, &LoadedProgram::handle);
            if (found == g_runtime.programs.end() || !found->module) {
                return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
            }
            RollbackLateTlsModuleLocked(
                found->owner_handle, found->handle, found->image);
            image = std::move(found->image);
            g_runtime.programs.erase(found);
        }
        const auto host = Ps5LoaderHost();
        Funnel::Ps5Desktop::UnloadNextGenExecutable(host, image);
        ResetPs5TranslationSession();
        SetRuntimeStatus("PS5 module unloaded");
        return LSX4_PS5_EXECUTE_OK;
    }

    std::vector<std::thread> guest_host_threads;
    std::vector<std::shared_ptr<GuestSemaphore>> guest_semaphores;
    std::vector<std::shared_ptr<GuestConditionVariable>>
        guest_condition_variables;
    std::vector<std::shared_ptr<GuestEventFlag>>
        guest_event_flags;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (found == g_runtime.programs.end()) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        g_runtime.shutdown_requested.store(
            true, std::memory_order_release);
        guest_host_threads =
            std::move(g_runtime.guest_host_threads);
        guest_semaphores.reserve(
            g_runtime.guest_semaphores.size());
        for (const auto& [semaphore_handle, semaphore] :
             g_runtime.guest_semaphores) {
            (void)semaphore_handle;
            guest_semaphores.push_back(semaphore);
        }
        guest_condition_variables.reserve(
            g_runtime.guest_condition_variables.size());
        for (const auto& [address, condition] :
             g_runtime.guest_condition_variables) {
            (void)address;
            guest_condition_variables.push_back(condition);
        }
        guest_event_flags.reserve(
            g_runtime.guest_event_flags.size());
        for (const auto& [event_flag_handle, event_flag] :
             g_runtime.guest_event_flags) {
            (void)event_flag_handle;
            guest_event_flags.push_back(event_flag);
        }
    }
    for (const auto& semaphore : guest_semaphores) {
        semaphore->condition.notify_all();
    }
    for (const auto& condition : guest_condition_variables) {
        condition->condition.notify_all();
    }
    for (const auto& event_flag : guest_event_flags) {
        event_flag->condition.notify_all();
    }
    g_runtime.guest_event_queue_condition.notify_all();
    for (auto& thread : guest_host_threads) {
        if (!thread.joinable()) {
            continue;
        }
        if (thread.get_id() == std::this_thread::get_id()) {
            thread.detach();
        } else {
            thread.join();
        }
    }

    std::vector<Funnel::Ps5Desktop::LoadedImage> images;
    std::vector<GuestThreadContext> threads;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (found == g_runtime.programs.end()) {
            g_runtime.shutdown_requested.store(
                false, std::memory_order_release);
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        std::vector<std::uint64_t> removed_handles;
        for (auto iterator = g_runtime.programs.begin();
             iterator != g_runtime.programs.end();) {
            if (iterator->handle != handle &&
                iterator->owner_handle != handle) {
                ++iterator;
                continue;
            }
            removed_handles.push_back(iterator->handle);
            images.push_back(std::move(iterator->image));
            iterator = g_runtime.programs.erase(iterator);
        }
        for (auto iterator = g_runtime.threads.begin();
             iterator != g_runtime.threads.end();) {
            if (!std::ranges::contains(
                    removed_handles, iterator->program_handle)) {
                ++iterator;
                continue;
            }
            threads.push_back(*iterator);
            const auto mapping_base =
                reinterpret_cast<std::uint64_t>(iterator->allocation);
            (void)g_runtime.mappings.EraseExact(
                mapping_base, iterator->mapping_size, true);
            iterator = g_runtime.threads.erase(iterator);
        }
        g_runtime.guest_mutexes.clear();
        g_runtime.guest_pthread_attributes.clear();
        g_runtime.guest_semaphores.clear();
        g_runtime.guest_condition_variables.clear();
        g_runtime.guest_event_flags.clear();
        g_event_flag_generation.fetch_add(
            1, std::memory_order_acq_rel);
        g_runtime.guest_video_out_ports.clear();
        g_runtime.guest_event_queues.clear();
        g_runtime.apr_files.clear();
        g_runtime.apr_file_ids_by_path.clear();
        g_runtime.apr_submissions.clear();
        g_runtime.apr_completion_writes.clear();
        g_runtime.apr_command_buffers.clear();
        g_runtime.apr_completion_events.clear();
        g_runtime.resolved_guest_paths.clear();
        g_runtime.resolved_guest_file_sizes.clear();
        g_runtime.indexed_guest_file_sizes.clear();
        g_runtime.next_event_flag_handle = 1;
        g_runtime.next_apr_file_id = 1;
        g_runtime.next_apr_submission_id = 1;
        g_runtime.apr_read_count = 0;
        g_runtime.guest_vblank_thread_started = false;
        g_runtime.diagnostic_second_event_queue_created.store(
            false, std::memory_order_release);
        g_runtime.guest_login_event_delivered.store(
            false, std::memory_order_release);
        for (auto& [handle, port] :
             g_runtime.guest_audio_out_ports) {
            (void)handle;
#ifdef __ANDROID__
            CloseGuestAudioOut(port);
#endif
        }
        g_runtime.guest_audio_out_ports.clear();
        g_runtime.next_pthread_attribute_handle = 1;
        g_runtime.next_video_out_handle = 1;
        g_runtime.next_audio_out_handle = 1;
        g_runtime.next_net_pool_handle = 1;
        g_runtime.next_ssl_handle = 1;
        g_runtime.next_http2_handle = 1;
        g_runtime.next_event_queue_handle = 1;
        g_runtime.agc_flip_count = 0;
        g_runtime.agc_dumped_frames = 0;
        g_runtime.agc_dumped_dcbs = 0;
        g_runtime.agc_dumped_draw_states = 0;
        g_runtime.agc_draw_count = 0;
        g_runtime.agc_dumped_textures = 0;
        g_runtime.agc_submit_count = 0;
        g_runtime.agc_last_draw_signature = 0;
        g_runtime.agc_logged_draw_signatures.clear();
        g_runtime.agc_draw_signature_logs = 0;
        g_runtime.agc_cx_registers.clear();
        g_runtime.agc_sh_registers.clear();
        g_runtime.agc_uc_registers.clear();
        g_runtime.agc_index_buffer_address = 0;
        g_runtime.agc_index_buffer_count = 0;
        g_runtime.agc_index_size = 0;
        g_runtime.agc_instance_count = 1;
        g_runtime.agc_dispatch_indirect_args_base = 0;
        g_runtime.agc_invalid_flip_count = 0;
        g_runtime.agc_next_flip = {};
        g_runtime.agc_cpu_next_render = {};
        ++g_runtime.agc_cpu_texture_cache_epoch;
        g_runtime.agc_cpu_capture_active = false;
        g_runtime.agc_cpu_capture_target = 0;
        g_runtime.agc_cpu_pending_capture_target = 0;
        g_runtime.presenter_texture = {};
        g_runtime.agc_cpu_frame.reset();
        g_runtime.agc_cpu_surfaces.clear();
        g_runtime.agc_cpu_working_surfaces.clear();
        g_runtime.agc_cpu_capture_active = false;
        g_runtime.agc_cpu_capture_target = 0;
        g_runtime.agc_cpu_pending_capture_target = 0;
        g_runtime.agc_cpu_quality_command_draws = 0u;
        g_runtime.agc_cpu_best_rendered_draws = 0u;
        g_runtime.agc_cpu_best_coverage = 0u;
        g_runtime.msg_dialog_status.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_status.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_mode.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_user_data.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_dir_name.fill('\0');
    }
    for (const auto& thread : threads) {
        munmap(thread.allocation, thread.mapping_size);
    }
    const auto host = Ps5LoaderHost();
    for (auto& image : images) {
        Funnel::Ps5Desktop::UnloadNextGenExecutable(host, image);
    }
    g_runtime.shutdown_requested.store(
        false, std::memory_order_release);
    ResetPs5TranslationSession();
    SetRuntimeStatus("PS5 program/module unloaded");
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_start_module(
    const std::uint64_t module_handle,
    Lsx4Ps5GuestResult* const result) {
    if (result == nullptr ||
        result->size < sizeof(Lsx4Ps5GuestResult)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *result = {};
    std::uint64_t owner_handle{};
    bool already_started{};
    std::vector<std::uint64_t> initializers;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto module = std::ranges::find(
            g_runtime.programs, module_handle, &LoadedProgram::handle);
        if (module == g_runtime.programs.end() || !module->module) {
            result->status = LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
            return result->status;
        }
        owner_handle = module->owner_handle;
        already_started = module->started;
        initializers = module->image.initializer_functions;
    }
    if (already_started) {
        result->status = LSX4_PS5_EXECUTE_OK;
        return result->status;
    }
    Lsx4Ps5GuestThreadContext thread_context{};
    const bool borrowed_thread_context =
        g_current_guest_fs_base != 0;
    if (borrowed_thread_context) {
        thread_context.handle = g_current_guest_thread_handle;
        thread_context.program_handle = owner_handle;
        thread_context.fs_base = g_current_guest_fs_base;
    } else {
        const auto create_status =
            executor_lsx4_ps5_runtime_create_thread_context(
                owner_handle, &thread_context);
        if (create_status != 0) {
            result->status = create_status;
            SetRuntimeStatus("PS5 module TCB initialization failed");
            return create_status;
        }
    }

    int execute_status = LSX4_PS5_EXECUTE_OK;
    for (const auto initializer : initializers) {
        Lsx4Ps5GuestEntry entry{};
        entry.address = initializer;
        entry.fs_base = thread_context.fs_base;
        Lsx4Ps5GuestResult initializer_result{};
        execute_status =
            executor_lsx4_ps5_runtime_execute(&entry, &initializer_result);
        *result = initializer_result;
        if (execute_status != LSX4_PS5_EXECUTE_OK) {
            break;
        }
    }
    if (!borrowed_thread_context) {
        (void)executor_lsx4_ps5_runtime_destroy_thread_context(
            thread_context.handle);
    }
    if (execute_status != LSX4_PS5_EXECUTE_OK) {
        return execute_status;
    }
    if (initializers.empty()) {
        result->status = LSX4_PS5_EXECUTE_OK;
        result->value = 0;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto module = std::ranges::find(
            g_runtime.programs, module_handle, &LoadedProgram::handle);
        if (module != g_runtime.programs.end() && module->module) {
            module->started = true;
        }
    }
    SetRuntimeStatus("PS5 module initialized");
    return result->status;
}

extern "C" int executor_lsx4_ps5_runtime_stop_module(
    const std::uint64_t module_handle,
    Lsx4Ps5GuestResult* const result) {
    if (result == nullptr ||
        result->size < sizeof(Lsx4Ps5GuestResult)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *result = {};
    std::uint64_t owner_handle{};
    bool started{};
    std::vector<std::uint64_t> finalizers;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto module = std::ranges::find(
            g_runtime.programs, module_handle, &LoadedProgram::handle);
        if (module == g_runtime.programs.end() || !module->module) {
            result->status = LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
            return result->status;
        }
        owner_handle = module->owner_handle;
        started = module->started;
        finalizers = module->image.finalizer_functions;
    }
    if (!started) {
        result->status = LSX4_PS5_EXECUTE_OK;
        return result->status;
    }

    Lsx4Ps5GuestThreadContext thread_context{};
    const auto create_status =
        executor_lsx4_ps5_runtime_create_thread_context(
            owner_handle, &thread_context);
    if (create_status != 0) {
        result->status = create_status;
        SetRuntimeStatus("PS5 module finalizer TCB initialization failed");
        return create_status;
    }
    int execute_status = LSX4_PS5_EXECUTE_OK;
    for (const auto finalizer : finalizers) {
        Lsx4Ps5GuestEntry entry{};
        entry.address = finalizer;
        entry.fs_base = thread_context.fs_base;
        Lsx4Ps5GuestResult finalizer_result{};
        execute_status =
            executor_lsx4_ps5_runtime_execute(&entry, &finalizer_result);
        *result = finalizer_result;
        if (execute_status != LSX4_PS5_EXECUTE_OK) {
            break;
        }
    }
    (void)executor_lsx4_ps5_runtime_destroy_thread_context(
        thread_context.handle);
    if (execute_status != LSX4_PS5_EXECUTE_OK) {
        SetRuntimeStatus("PS5 module finalizer failed");
        return execute_status;
    }
    if (finalizers.empty()) {
        result->status = LSX4_PS5_EXECUTE_OK;
        result->value = 0;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto module = std::ranges::find(
            g_runtime.programs, module_handle, &LoadedProgram::handle);
        if (module != g_runtime.programs.end() && module->module) {
            module->started = false;
        }
    }
    SetRuntimeStatus("PS5 module finalized");
    return result->status;
}

extern "C" int executor_lsx4_ps5_runtime_launch_eboot(
    const std::uint64_t handle, const Lsx4Ps5GuestEntry* const entry,
    Lsx4Ps5GuestResult* const result) {
    if (result == nullptr ||
        result->size < sizeof(Lsx4Ps5GuestResult) ||
        (entry != nullptr &&
         entry->size < sizeof(Lsx4Ps5GuestEntry))) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    *result = {};
    std::vector<std::uint64_t> startup_modules;
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto owner = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (owner == g_runtime.programs.end() || owner->module) {
            result->status = LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
            return result->status;
        }
        if (owner->module_scan_complete) {
            Lsx4Ps5GameProbeReport report{};
            std::string probe_text;
            if (!FillGameProbeLocked(handle, report, probe_text) ||
                (report.flags &
                 LSX4_PS5_GAME_READY_TO_LAUNCH) == 0) {
                result->status = LSX4_PS5_EXECUTE_BACKEND_NOT_READY;
                g_runtime.status = std::move(probe_text);
                return result->status;
            }
        }
        startup_modules = StartupModuleOrderLocked(handle);
    }
    Lsx4Ps5GuestEntry request{};
    std::vector<std::uint64_t> initializers;
    bool initializers_started{};
    if (entry != nullptr) {
        request = *entry;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (found == g_runtime.programs.end()) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        if (found->module) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        request.address = found->image.entry;
        initializers_started = found->started;
        if (!initializers_started) {
            initializers = found->image.preinitializer_functions;
            initializers.insert(
                initializers.end(),
                found->image.initializer_functions.begin(),
                found->image.initializer_functions.end());
            if (EntryDirectlyCallsInitializer(
                    found->image, initializers)) {
#ifdef __ANDROID__
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4-PS5",
                    "eboot entry owns initializer dispatch "
                    "entry=0x%llx initializers=%zu",
                    static_cast<unsigned long long>(
                        found->image.entry),
                    initializers.size());
#endif
                initializers.clear();
            }
        }
    }
    request.size = sizeof(request);
    if (request.argument_count == 0) {
        const std::lock_guard lock{g_runtime.mutex};
        if (!EnsureHleDataLocked()) {
            result->status = LSX4_PS5_EXECUTE_INTERNAL_ERROR;
            g_runtime.status =
                "PS5 process entry parameter allocation failed";
            return result->status;
        }
        const auto hle_data =
            reinterpret_cast<std::uint64_t>(g_runtime.hle_data);
        const auto entry_params =
            hle_data + Ps5ProcessEntryParamsOffset;
        const auto image_name =
            hle_data + Ps5ProgramNameBufferOffset;
        constexpr std::uint32_t ArgumentCount = 1;
        constexpr std::uint32_t Reserved = 0;
        std::memcpy(
            reinterpret_cast<void*>(entry_params),
            &ArgumentCount, sizeof(ArgumentCount));
        std::memcpy(
            reinterpret_cast<void*>(
                entry_params + sizeof(ArgumentCount)),
            &Reserved, sizeof(Reserved));
        std::memcpy(
            reinterpret_cast<void*>(entry_params + 0x08),
            &image_name, sizeof(image_name));
        std::memset(
            reinterpret_cast<void*>(entry_params + 0x10),
            0, 2 * sizeof(std::uint64_t));
        const auto exit_handler = CreateHleStubLocked(
            handle, "__lsx4_ps5_program_exit", 0);
        if (exit_handler == 0) {
            result->status = LSX4_PS5_EXECUTE_INTERNAL_ERROR;
            g_runtime.status =
                "PS5 process exit handler allocation failed";
            return result->status;
        }
        request.argument_count = 6;
        request.arguments[0] = entry_params;
        request.arguments[1] = exit_handler;
    }
    Lsx4Ps5GuestThreadContext automatic_thread{};
    const bool needs_thread_context = request.fs_base == 0;
    if (needs_thread_context) {
        const auto create_status =
            executor_lsx4_ps5_runtime_create_thread_context(
                handle, &automatic_thread);
        if (create_status != 0) {
            *result = {};
            result->status = create_status;
            SetRuntimeStatus("PS5 eboot TLS/TCB initialization failed");
            return create_status;
        }
        request.fs_base = automatic_thread.fs_base;
    }
    const auto previous_startup_thread_handle = std::exchange(
        g_current_guest_thread_handle,
        needs_thread_context ? automatic_thread.handle
                             : g_current_guest_thread_handle);
    const auto previous_startup_fs_base = std::exchange(
        g_current_guest_fs_base, request.fs_base);
    for (const auto module_handle : startup_modules) {
        Lsx4Ps5GuestResult module_result{};
        const auto start_status =
            executor_lsx4_ps5_runtime_start_module(
                module_handle, &module_result);
        if (start_status != LSX4_PS5_EXECUTE_OK) {
            g_current_guest_thread_handle =
                previous_startup_thread_handle;
            g_current_guest_fs_base = previous_startup_fs_base;
            *result = module_result;
            if (needs_thread_context) {
                (void)executor_lsx4_ps5_runtime_destroy_thread_context(
                    automatic_thread.handle);
            }
            return start_status;
        }
    }
    g_current_guest_thread_handle = previous_startup_thread_handle;
    g_current_guest_fs_base = previous_startup_fs_base;
    if (!initializers_started) {
        for (const auto initializer : initializers) {
            Lsx4Ps5GuestEntry initializer_request{};
            initializer_request.address = initializer;
            initializer_request.fs_base = request.fs_base;
            initializer_request.stack_base = request.stack_base;
            initializer_request.stack_size = request.stack_size;
            Lsx4Ps5GuestResult initializer_result{};
            const auto initializer_status =
                executor_lsx4_ps5_runtime_execute(
                    &initializer_request, &initializer_result);
            if (initializer_status != LSX4_PS5_EXECUTE_OK) {
                *result = initializer_result;
                const std::string detail =
                    executor_lsx4_ps5_runtime_status();
                if (needs_thread_context) {
                    (void)executor_lsx4_ps5_runtime_destroy_thread_context(
                        automatic_thread.handle);
                }
                SetRuntimeStatus(
                    "PS5 eboot initializer failed: " + detail);
                return initializer_status;
            }
        }
        const std::lock_guard lock{g_runtime.mutex};
        const auto found = std::ranges::find(
            g_runtime.programs, handle, &LoadedProgram::handle);
        if (found != g_runtime.programs.end() && !found->module) {
            found->started = true;
        }
    }
    const auto execute_status =
        executor_lsx4_ps5_runtime_execute(&request, result);
    if (needs_thread_context) {
        (void)executor_lsx4_ps5_runtime_destroy_thread_context(
            automatic_thread.handle);
    }
    return execute_status;
}

extern "C" int executor_lsx4_ps5_runtime_create_thread_context(
    const std::uint64_t program_handle,
    Lsx4Ps5GuestThreadContext* const thread_context) {
    if (thread_context == nullptr ||
        thread_context->size < sizeof(Lsx4Ps5GuestThreadContext)) {
        return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
    }
    Lsx4Ps5GuestThreadContext created{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!g_runtime.initialized) {
            return LSX4_PS5_EXECUTE_NOT_INITIALIZED;
        }
        if (program_handle == 0 ||
            std::ranges::none_of(
                g_runtime.programs,
                [&](const LoadedProgram& program) {
                    return program.handle == program_handle;
                })) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
        if (!CreateThreadContextLocked(program_handle, created)) {
            return LSX4_PS5_EXECUTE_INTERNAL_ERROR;
        }
    }
    *thread_context = created;
    return 0;
}

extern "C" int executor_lsx4_ps5_runtime_destroy_thread_context(
    const std::uint64_t thread_handle) {
    GuestThreadContext removed{};
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!RemoveThreadContextLocked(thread_handle, removed)) {
            return LSX4_PS5_EXECUTE_INVALID_ARGUMENT;
        }
    }
    munmap(removed.allocation, removed.mapping_size);
    return 0;
}

extern "C" void executor_lsx4_ps5_runtime_reset() {
    bool had_runtime_session{};
    std::vector<std::thread> guest_host_threads;
    std::vector<std::shared_ptr<GuestSemaphore>> guest_semaphores;
    std::vector<std::shared_ptr<GuestConditionVariable>>
        guest_condition_variables;
    std::vector<std::shared_ptr<GuestEventFlag>>
        guest_event_flags;
    {
        const std::lock_guard lock{g_runtime.mutex};
        had_runtime_session =
            g_runtime.initialized || !g_runtime.programs.empty();
        g_runtime.shutdown_requested.store(
            true, std::memory_order_release);
        g_runtime.initialized = false;
        guest_host_threads =
            std::move(g_runtime.guest_host_threads);
        guest_semaphores.reserve(
            g_runtime.guest_semaphores.size());
        for (const auto& [handle, semaphore] :
             g_runtime.guest_semaphores) {
            (void)handle;
            guest_semaphores.push_back(semaphore);
        }
        guest_condition_variables.reserve(
            g_runtime.guest_condition_variables.size());
        for (const auto& [address, condition] :
             g_runtime.guest_condition_variables) {
            (void)address;
            guest_condition_variables.push_back(condition);
        }
        guest_event_flags.reserve(
            g_runtime.guest_event_flags.size());
        for (const auto& [event_flag_handle, event_flag] :
             g_runtime.guest_event_flags) {
            (void)event_flag_handle;
            guest_event_flags.push_back(event_flag);
        }
    }
    for (const auto& semaphore : guest_semaphores) {
        semaphore->condition.notify_all();
    }
    for (const auto& condition : guest_condition_variables) {
        condition->condition.notify_all();
    }
    for (const auto& event_flag : guest_event_flags) {
        event_flag->condition.notify_all();
    }
    g_runtime.guest_event_queue_condition.notify_all();
    for (auto& thread : guest_host_threads) {
        if (!thread.joinable()) {
            continue;
        }
        if (thread.get_id() == std::this_thread::get_id()) {
            thread.detach();
        } else {
            thread.join();
        }
    }
    RestorePs5DiagnosticFaultHandlers();
    std::vector<LoadedProgram> programs;
    std::vector<GuestThreadContext> threads;
    std::vector<OwnedGuestMapping> owned_guest_mappings;
    std::unordered_map<std::uint64_t, LibcHeapAllocation>
        libc_heap_allocations;
    std::vector<GuestStdioFile> stdio_files;
    std::vector<GuestStdioHandlePage> stdio_handle_pages;
    std::vector<GuestKernelFile> kernel_files;
    std::uint8_t* hle_slab{};
    std::uint8_t* hle_data{};
#ifdef __ANDROID__
    ANativeWindow* android_window{};
#endif
    {
        const std::lock_guard lock{g_runtime.mutex};
        g_runtime.callbacks = {};
        programs = std::move(g_runtime.programs);
        threads = std::move(g_runtime.threads);
        owned_guest_mappings =
            std::move(g_runtime.owned_guest_mappings);
        libc_heap_allocations =
            std::move(g_runtime.libc_heap_allocations);
        stdio_files = std::move(g_runtime.stdio_files);
        stdio_handle_pages =
            std::move(g_runtime.stdio_handle_pages);
        kernel_files = std::move(g_runtime.kernel_files);
        g_runtime.app0_directory.clear();
        g_runtime.save_data_directory.clear();
#ifdef __ANDROID__
        g_runtime.game_splash_active = false;
        g_runtime.game_splash_presented = false;
#endif
        g_runtime.agc_register_defaults = 0;
        g_runtime.agc_internal_register_defaults = 0;
        g_runtime.msg_dialog_status.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_status.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_mode.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_user_data.store(
            0, std::memory_order_release);
        g_runtime.save_data_dialog_dir_name.fill('\0');
        g_runtime.guest_mutexes.clear();
        g_runtime.guest_pthread_attributes.clear();
        g_runtime.guest_semaphores.clear();
        g_runtime.guest_condition_variables.clear();
        g_runtime.guest_event_flags.clear();
        g_event_flag_generation.fetch_add(
            1, std::memory_order_acq_rel);
        g_runtime.guest_video_out_ports.clear();
        g_runtime.guest_event_queues.clear();
        g_runtime.apr_files.clear();
        g_runtime.apr_file_ids_by_path.clear();
        g_runtime.apr_submissions.clear();
        g_runtime.apr_completion_writes.clear();
        g_runtime.apr_command_buffers.clear();
        g_runtime.apr_completion_events.clear();
        g_runtime.resolved_guest_paths.clear();
        g_runtime.resolved_guest_file_sizes.clear();
        g_runtime.indexed_guest_file_sizes.clear();
        g_runtime.next_event_flag_handle = 1;
        g_runtime.next_kernel_file_handle = 3;
        g_runtime.next_apr_file_id = 1;
        g_runtime.next_apr_submission_id = 1;
        g_runtime.apr_read_count = 0;
        g_runtime.guest_vblank_thread_started = false;
        g_runtime.diagnostic_second_event_queue_created.store(
            false, std::memory_order_release);
        g_runtime.guest_login_event_delivered.store(
            false, std::memory_order_release);
        for (auto& [handle, port] :
             g_runtime.guest_audio_out_ports) {
            (void)handle;
#ifdef __ANDROID__
            CloseGuestAudioOut(port);
#endif
        }
        g_runtime.guest_audio_out_ports.clear();
        g_runtime.presenter_texture = {};
        g_runtime.agc_cpu_frame.reset();
        g_runtime.agc_cpu_surfaces.clear();
        g_runtime.agc_cpu_working_surfaces.clear();
        g_runtime.agc_cpu_quality_command_draws = 0u;
        g_runtime.agc_cpu_best_rendered_draws = 0u;
        g_runtime.agc_cpu_best_coverage = 0u;
#ifdef __ANDROID__
        android_window =
            std::exchange(g_runtime.android_window, nullptr);
#endif
        g_runtime.direct_allocations.clear();
        g_runtime.next_direct_offset = 0;
        g_runtime.next_direct_map_hint = Ps5DirectMapHint;
        g_runtime.next_flexible_map_hint = Ps5FlexibleMapHint;
        g_runtime.next_libc_heap_hint = Ps5LibcHeapHint;
        hle_slab = std::exchange(g_runtime.hle_slab, nullptr);
        hle_data = std::exchange(g_runtime.hle_data, nullptr);
        g_runtime.next_mutex_object_offset =
            Ps5MutexObjectPoolOffset;
        g_runtime.hle_slab_used = 0;
        ++g_runtime.hle_slab_slot;
        g_runtime.hle_bindings.clear();
        g_hle_binding_generation.fetch_add(
            1, std::memory_order_release);
        g_runtime.next_program_handle = 1;
        g_runtime.next_thread_handle = 1;
        g_runtime.next_pthread_attribute_handle = 1;
        g_runtime.next_semaphore_handle = 1;
        g_runtime.next_video_out_handle = 1;
        g_runtime.next_audio_out_handle = 1;
        g_runtime.next_event_queue_handle = 1;
        g_runtime.agc_flip_count = 0;
        g_runtime.agc_dumped_frames = 0;
        g_runtime.agc_dumped_dcbs = 0;
        g_runtime.agc_dumped_draw_states = 0;
        g_runtime.agc_draw_count = 0;
        g_runtime.agc_dumped_textures = 0;
        g_runtime.agc_submit_count = 0;
        g_runtime.agc_cx_registers.clear();
        g_runtime.agc_sh_registers.clear();
        g_runtime.agc_uc_registers.clear();
        g_runtime.agc_index_buffer_address = 0;
        g_runtime.agc_index_buffer_count = 0;
        g_runtime.agc_index_size = 0;
        g_runtime.agc_instance_count = 1;
        g_runtime.agc_dispatch_indirect_args_base = 0;
        g_runtime.agc_next_flip = {};
        g_runtime.agc_cpu_next_render = {};
        ++g_runtime.agc_cpu_texture_cache_epoch;
        g_runtime.agc_cpu_capture_active = false;
        g_runtime.agc_cpu_capture_target = 0;
        g_runtime.agc_cpu_pending_capture_target = 0;
    }
#ifdef __ANDROID__
    Lsx4::Ps5Desktop::ResetVulkanPresenter();
    if (android_window != nullptr) {
        ANativeWindow_release(android_window);
    }
#endif
    const auto host = Ps5LoaderHost();
    for (auto& program : programs) {
        Funnel::Ps5Desktop::UnloadNextGenExecutable(
            host, program.image);
    }
    for (const auto& thread : threads) {
        munmap(thread.allocation, thread.mapping_size);
    }
    for (const auto& mapping : owned_guest_mappings) {
        munmap(mapping.allocation, mapping.byte_count);
    }
    for (const auto& file : stdio_files) {
        if (file.stream != nullptr) {
            std::fclose(file.stream);
        }
    }
    for (const auto& page : stdio_handle_pages) {
        munmap(page.allocation, page.mapped_size);
    }
    for (const auto& file : kernel_files) {
        if (file.stream != nullptr) {
            std::fclose(file.stream);
        }
    }
    for (const auto& [address, allocation] : libc_heap_allocations) {
        munmap(allocation.allocation, allocation.mapped_size);
    }
    if (hle_slab != nullptr) {
        munmap(hle_slab, Ps5HleSlabSize);
    }
    if (hle_data != nullptr) {
        munmap(hle_data, Ps5HleDataSize);
    }
    if (had_runtime_session) {
        ResetPs5TranslationSession();
    }
    const std::lock_guard lock{g_runtime.mutex};
    g_runtime.mappings.Clear();
    g_runtime.status = "PS5 runtime is not initialized";
}

extern "C" bool ExecutorJitIsReadableGuestRange(
    const std::uint64_t address, const std::size_t size) {
    const std::lock_guard lock{g_runtime.mutex};
    return HasAccessLocked(address, size, LSX4_PS5_GUEST_READ);
}

extern "C" bool ExecutorJitIsExecutableGuestAddress(
    const std::uint64_t address) {
    const std::lock_guard lock{g_runtime.mutex};
    return HasAccessLocked(address, 1, LSX4_PS5_GUEST_EXECUTE);
}

extern "C" bool ExecutorJitReadGuestBytes(
    const std::uint64_t address, void* const destination,
    const std::size_t size) {
    if (destination == nullptr) {
        return false;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!HasAccessLocked(address, size, LSX4_PS5_GUEST_READ)) {
            return false;
        }
    }
    std::memcpy(destination, reinterpret_cast<const void*>(address), size);
    return true;
}

extern "C" bool ExecutorJitReadGuestBytesStable(
    const std::uint64_t address, void* const destination,
    const std::size_t size) {
    return ExecutorJitReadGuestBytes(address, destination, size);
}

extern "C" bool ExecutorJitWriteGuestBytes(
    const std::uint64_t address, const void* const source,
    const std::size_t size) {
    if (source == nullptr) {
        return false;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (!HasAccessLocked(address, size, LSX4_PS5_GUEST_WRITE)) {
            return false;
        }
    }
    std::memcpy(reinterpret_cast<void*>(address), source, size);
    return true;
}

extern "C" void executor_lsx4_android_register_guest_readable_range(
    const void* const base, const std::size_t size, const char* const label) {
    if (base == nullptr || size == 0) {
        return;
    }
    const std::lock_guard lock{g_runtime.mutex};
    (void)RegisterMappingLocked(
        reinterpret_cast<std::uint64_t>(base), size,
        LSX4_PS5_GUEST_READ | LSX4_PS5_GUEST_WRITE, label, true);
}

extern "C" void executor_lsx4_android_note_guest_stack_window(
    const void* const base, const GuestWindowExtent size,
    const char* const label) {
    executor_lsx4_android_register_guest_readable_range(base, size, label);
}

extern "C" int executor_jit_classify_hle_thunk(
    const std::uint64_t thunk) {
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (g_runtime.hle_bindings.contains(thunk)) {
            return 2;
        }
    }
    const auto callbacks = SnapshotCallbacks();
    if (callbacks.classify_hle != nullptr) {
        return callbacks.classify_hle(callbacks.context, thunk);
    }
    if (callbacks.resolve_hle == nullptr) {
        return 0;
    }
    std::uint64_t native_function = 0;
    return callbacks.resolve_hle(callbacks.context, thunk, &native_function);
}

extern "C" int executor_jit_resolve_hle_thunk(
    const std::uint64_t thunk, std::uint64_t* const native_function) {
    if (native_function == nullptr) {
        return 0;
    }
    *native_function = 0;
    struct ResolutionCache {
        std::uint64_t generation{};
        std::unordered_map<std::uint64_t, std::uint64_t> bindings;
    };
    thread_local ResolutionCache cache;
    const auto generation =
        g_hle_binding_generation.load(std::memory_order_acquire);
    if (cache.generation != generation) {
        cache.bindings.clear();
        cache.generation = generation;
    }
    if (const auto cached = cache.bindings.find(thunk);
        cached != cache.bindings.end()) {
        *native_function = cached->second;
        return 2;
    }
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (const auto found = g_runtime.hle_bindings.find(thunk);
            found != g_runtime.hle_bindings.end()) {
            *native_function = found->second.native_function != 0
                ? found->second.native_function
                : thunk;
            cache.bindings.emplace(thunk, *native_function);
            return 2;
        }
    }
    const auto callbacks = SnapshotCallbacks();
    return callbacks.resolve_hle != nullptr
        ? callbacks.resolve_hle(callbacks.context, thunk, native_function)
        : 0;
}

extern "C" int executor_jit_lookup_hle_thunk(
    const std::uint64_t thunk, std::uint64_t* const native_function) {
    if (native_function == nullptr) {
        return 0;
    }
    *native_function = 0;
    {
        const std::lock_guard lock{g_runtime.mutex};
        if (const auto found = g_runtime.hle_bindings.find(thunk);
            found != g_runtime.hle_bindings.end()) {
            if (found->second.native_function == 0) {
                return 0;
            }
            *native_function = found->second.native_function;
            return 2;
        }
    }
    const auto callbacks = SnapshotCallbacks();
    return callbacks.resolve_hle != nullptr
        ? callbacks.resolve_hle(callbacks.context, thunk, native_function)
        : 0;
}

extern "C" int executor_jit_lookup_leaf_hle_thunk(
    const std::uint64_t thunk, std::uint64_t* const native_function) {
    return executor_jit_lookup_hle_thunk(thunk, native_function);
}

std::uint64_t ExecutorJitResolvedLeafHleCallback(
    const std::uint64_t native_function, const std::uint64_t arg0,
    const std::uint64_t arg1, const std::uint64_t arg2,
    const std::uint64_t arg3, const std::uint64_t arg4,
    const std::uint64_t arg5, const std::uint64_t guest_rsp) {
    if (native_function == 0) {
        return 0;
    }
    using NativeLeaf = std::uint64_t (*)(
        std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
        std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
    try {
        return reinterpret_cast<NativeLeaf>(native_function)(
            arg0, arg1, arg2, arg3, arg4, arg5, guest_rsp, 0);
    } catch (...) {
        SetRuntimeStatus("PS5 native leaf HLE raised a host exception");
        return 0;
    }
}

Lsx4::Translation::HleLeafOutcome
ExecutorJitTryLeafHleThunkCallback(
    const std::uint64_t thunk, const std::uint64_t arg0,
    const std::uint64_t arg1, const std::uint64_t arg2,
    const std::uint64_t arg3, const std::uint64_t arg4,
    const std::uint64_t arg5, const std::uint64_t guest_rsp) {
    std::uint64_t native_function = 0;
    if (executor_jit_lookup_leaf_hle_thunk(thunk, &native_function) == 0 ||
        native_function == 0) {
        return {};
    }
    return {
        .value = ExecutorJitResolvedLeafHleCallback(
            native_function, arg0, arg1, arg2, arg3, arg4, arg5, guest_rsp),
        .accepted = 1,
    };
}

extern "C" int executor_jit_hle_thunk_returns_zero(
    const std::uint64_t thunk) {
    const auto callbacks = SnapshotCallbacks();
    return callbacks.hle_returns_zero != nullptr
        ? callbacks.hle_returns_zero(callbacks.context, thunk)
        : 0;
}

extern "C" int executor_jit_hle_fp_result_kind(
    const std::uint64_t native_function) {
    const auto callbacks = SnapshotCallbacks();
    return callbacks.hle_fp_result_kind != nullptr
        ? callbacks.hle_fp_result_kind(callbacks.context, native_function)
        : 0;
}

extern "C" int executor_jit_hle_fp_bridge_selftest() {
    const auto callbacks = SnapshotCallbacks();
    return callbacks.invoke_hle != nullptr ? 1 : 0;
}

extern "C" std::uint64_t executor_jit_ensure_hle_thunk_slab() {
    const std::lock_guard lock{g_runtime.mutex};
    return g_runtime.initialized ? EnsureHleSlabLocked() : 0;
}

extern "C" std::uint64_t executor_jit_hle_thunk_slab_base() {
    const std::lock_guard lock{g_runtime.mutex};
    return reinterpret_cast<std::uint64_t>(g_runtime.hle_slab);
}

extern "C" std::uint64_t executor_jit_hle_thunk_slab_size() {
    const std::lock_guard lock{g_runtime.mutex};
    return g_runtime.hle_slab != nullptr ? Ps5HleSlabSize : 0;
}

extern "C" std::uint64_t executor_jit_get_hle_stub_for_native(
    const char* const name, const std::uint64_t native_function) {
    const std::lock_guard lock{g_runtime.mutex};
    return g_runtime.initialized
        ? CreateHleStubLocked(0, name, native_function)
        : 0;
}

extern "C" std::uint64_t executor_jit_stable_libc_strcmp() {
    return reinterpret_cast<std::uint64_t>(&RuntimeStrcmp);
}

extern "C" int executor_lsx4_android_dispatch_deferred_guest_signal(
    const std::int32_t native_signal, const std::int32_t signal_code,
    const std::int32_t signal_errno, const std::int32_t source_pid,
    const std::uint32_t source_uid, const std::uint64_t fault_address,
    const std::uint64_t guest_rip, const std::int32_t is_write) {
    const auto callbacks = SnapshotCallbacks();
    return callbacks.dispatch_signal != nullptr
        ? callbacks.dispatch_signal(
              callbacks.context, native_signal, signal_code, signal_errno,
              source_pid, source_uid, fault_address, guest_rip, is_write)
        : 0;
}

namespace Lsx4::Translation {

std::uint64_t InvokeRuntimeHle(const HleBridgeRequest& request) {
    const auto callbacks = SnapshotCallbacks();
    std::uint64_t builtin_result{};
    const auto handled =
        TryInvokeBuiltinHle(request, builtin_result);
    Lsx4Ps5HleCall call{
        .function = request.function,
        .guest_stack = request.guest_stack,
    };
    std::ranges::copy(request.integer_arguments, call.integer_arguments);
    std::ranges::copy(request.floating_arguments, call.floating_arguments);
    std::uint64_t callback_result{};
    try {
        if (!handled && callbacks.invoke_hle != nullptr) {
            callback_result =
                callbacks.invoke_hle(callbacks.context, &call);
        }
    } catch (...) {
        SetRuntimeStatus("PS5 HLE callback raised a host exception");
        return 0;
    }
    const auto result =
        handled ? builtin_result : callback_result;
    if (Ps5TraceHleEnabled()) {
        const auto symbol = HleSymbol(request.function);
        std::fprintf(
            stderr,
            "PS5_HLE_RESULT symbol=%s thunk=0x%llx handled=%d "
            "result=0x%llx args=%llx,%llx,%llx,%llx,%llx,%llx\n",
            symbol.empty() ? "<unbound>" : symbol.c_str(),
            static_cast<unsigned long long>(request.function),
            handled ? 1 : 0,
            static_cast<unsigned long long>(result),
            static_cast<unsigned long long>(request.integer_arguments[0]),
            static_cast<unsigned long long>(request.integer_arguments[1]),
            static_cast<unsigned long long>(request.integer_arguments[2]),
            static_cast<unsigned long long>(request.integer_arguments[3]),
            static_cast<unsigned long long>(request.integer_arguments[4]),
            static_cast<unsigned long long>(request.integer_arguments[5]));
        std::fflush(stderr);
    }
#ifdef __ANDROID__
    WatchDreamingForeachState(
        HleSymbol(request.function), result,
        request.guest_stack);
#endif
    return result;
}

std::uintptr_t SelectiveLeafBridgeEntry() noexcept {
    return reinterpret_cast<std::uintptr_t>(
        &ExecutorJitTryLeafHleThunkCallback);
}

std::uintptr_t ResolvedLeafBridgeEntry() noexcept {
    return reinterpret_cast<std::uintptr_t>(
        &ExecutorJitResolvedLeafHleCallback);
}

}
