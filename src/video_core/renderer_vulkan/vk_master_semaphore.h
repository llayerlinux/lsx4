// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class MasterSemaphore {
public:
    explicit MasterSemaphore(const Instance& instance_);
    ~MasterSemaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsDeviceLost() const noexcept {
        return device_lost.load(std::memory_order_acquire) ||
               device_terminal.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return !IsDeviceLost() && KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Marks the Vulkan device as terminally lost. The first caller emits the diagnostic; all
    /// subsequent timeline operations become no-ops instead of retrying a dead device.
    void MarkDeviceLost(vk::Result result, const char* where) noexcept;

    /// Stops all schedulers which share this VkDevice. Unlike MarkDeviceLost this also accepts a
    /// bounded-wait timeout: Android drivers can wedge in a fence/timeline/WSI wait without ever
    /// returning VK_ERROR_DEVICE_LOST.
    void MarkTerminal(vk::Result result, const char* where) noexcept;

    [[nodiscard]] static bool IsDeviceTerminal() noexcept {
        return device_terminal.load(std::memory_order_acquire);
    }

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
    std::atomic_bool device_lost{false};
    static std::atomic_bool device_terminal;
    static std::atomic<u32> live_instances;
};

} // namespace Vulkan
