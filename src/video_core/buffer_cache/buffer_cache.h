// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <boost/container/small_vector.hpp>
#include <optional>
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
}

namespace VideoCore {

using BufferId = Common::SlotId;

static constexpr BufferId NULL_BUFFER_ID{0};

class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
    };

    struct PreparedVertexBuffers {
        boost::container::small_vector<vk::VertexInputAttributeDescription2EXT, 32> attributes;
        boost::container::small_vector<vk::VertexInputBindingDescription2EXT, 32> bindings;
        boost::container::small_vector<vk::Buffer, 32> buffers;
        boost::container::small_vector<vk::DeviceSize, 32> offsets;
        boost::container::small_vector<vk::DeviceSize, 32> sizes;
        boost::container::small_vector<vk::DeviceSize, 32> strides;
        bool dynamic_vertex_input{};
    };

    struct PreparedIndexBuffer {
        vk::Buffer buffer{};
        vk::DeviceSize offset{};
        vk::IndexType type{};
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    /// Resolves host vertex buffers without recording state into a command buffer. Buffer
    /// synchronization may submit and rotate the current command buffer, so all fallible preparation
    /// must finish before the prepared state is emitted next to the draw.
    [[nodiscard]] PreparedVertexBuffers PrepareVertexBuffers(
        const Vulkan::GraphicsPipeline& pipeline,
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
        bool executor_probe_bound = false, u64 executor_probe_sequence = 0);

    /// Records previously prepared vertex input and buffer bindings into the current command buffer.
    void BindPreparedVertexBuffers(const PreparedVertexBuffers& prepared);

    /// Resolves the host index buffer without recording its binding.
    [[nodiscard]] PreparedIndexBuffer PrepareIndexBuffer(
        u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
        bool executor_probe_bound = false, u64 executor_probe_sequence = 0);

    /// Records a previously prepared index-buffer binding into the current command buffer.
    void BindPreparedIndexBuffer(const PreparedIndexBuffer& prepared);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {});

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Exact byte-range variant used for immutable shader identity. The page tracker deliberately
    /// rounds to 4 KiB for mprotect, so querying it for a shader can report a false dependency when
    /// unrelated GPU output shares the same guest page. gpu_modified_ranges retains the original
    /// write intervals and still detects a real GPU-generated shader update.
    [[nodiscard]] bool IsRegionGpuModifiedExact(VAddr addr, size_t size) const {
        return gpu_modified_ranges.Intersects(addr, size);
    }

    /// Shader identity metadata may only survive a GPU readback while the guest backing remains
    /// unchanged. Tracking is opt-in so the normal renderer pays no generation-update cost.
    void EnableGuestReadbackGenerationTracking() noexcept {
        guest_readback_generation_tracking.store(true, std::memory_order_release);
    }

    [[nodiscard]] u64 GetGuestReadbackGeneration() const noexcept {
        return guest_readback_generation.load(std::memory_order_acquire);
    }

#ifdef __ANDROID__
    /// Bounded, read-only fingerprint of the bytes currently stored in the cached Vulkan buffer.
    /// This deliberately does not reconcile the guest CPU backing or clear GPU-dirty tracking.
    void ExecutorProbeGpuBuffer(VAddr device_addr, u32 size, const char* role, u64 sequence);

    /// Compares guest bytes with the exact byte range already bound to a shader descriptor. Unlike
    /// ExecutorProbeGpuBuffer this does not call ObtainBuffer, so the diagnostic cannot hide a
    /// missed upload by synchronizing the cache before it observes it.
    void ExecutorProbeBoundGpuBuffer(VAddr guest_addr, u32 size, vk::Buffer bound_buffer,
                                     u64 bound_offset, const char* role, u64 sequence, u32 slot);
#endif

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector(std::optional<size_t> sampled_device_memory = std::nullopt);

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    template <bool async>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write);

    void NotifyGuestReadback() noexcept {
        if (guest_readback_generation_tracking.load(std::memory_order_relaxed)) {
            guest_readback_generation.fetch_add(1, std::memory_order_release);
        }
    }

#ifdef __ANDROID__
    /// Resolve one precise CPU-read fault together with nearby GPU-dirty cache buffers using one
    /// command-buffer submission/fence. This preserves precise readback semantics while avoiding
    /// a serialized signal + Finish round-trip for every independently cached 4 KiB resource.
    void DownloadBufferMemoryWave(Buffer& primary_buffer, VAddr device_addr, u64 size);
#endif

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 gc_tick = 0;
    std::atomic<bool> guest_readback_generation_tracking{false};
    std::atomic<u64> guest_readback_generation{0};
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    RangeSet gpu_modified_ranges;
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
};

} // namespace VideoCore
