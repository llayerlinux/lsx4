// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/stack_windows.h"
#include "executor/dynamic_translation/process_memory.h"

#include <cstddef>
#include <new>

#if defined(__ANDROID__)
#include <sys/mman.h>
#endif

extern "C" void executor_lsx4_android_note_guest_stack_window(
    const void* base, std::uint64_t size, const char* label);

namespace Lsx4::Translation {
namespace {

constexpr std::size_t PoolBytes = 8u * 1024u * 1024u;
constexpr std::uint64_t ParentClearance = 0x800;

struct ThreadStackPool {
    std::byte* lower{};
    std::byte* upper{};
    std::byte* cursor{};
};

thread_local ThreadStackPool pool{};

std::byte* AllocatePool() noexcept {
#if defined(__ANDROID__)
    if (void* const mapping = MapProcessMemoryBelow4GiB(PoolBytes)) {
        executor_lsx4_android_note_guest_stack_window(mapping, PoolBytes,
                                                       "translation_stack_pool");
        return static_cast<std::byte*>(mapping);
    }
    void* const mapping = mmap(nullptr, PoolBytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping != MAP_FAILED) {
        executor_lsx4_android_note_guest_stack_window(mapping, PoolBytes,
                                                       "translation_stack_pool");
        return static_cast<std::byte*>(mapping);
    }
#endif
    auto* const allocation = new (std::nothrow) std::byte[PoolBytes];
    if (allocation != nullptr) {
        executor_lsx4_android_note_guest_stack_window(allocation, PoolBytes,
                                                       "translation_stack_pool");
    }
    return allocation;
}

}

StackWindow AcquireStackWindow(const LiveStateView parent) noexcept {
    if (pool.lower == nullptr) {
        pool.lower = AllocatePool();
        if (pool.lower == nullptr) {
            return {};
        }
        pool.upper = pool.lower + PoolBytes;
        pool.cursor = pool.upper;
    }
    const auto prior = reinterpret_cast<std::uint64_t>(pool.cursor);
    std::uint64_t selected = prior;
    if (parent) {
        const std::uint64_t parent_stack =
            parent.Read(IntegerRegister::Stack) & ~std::uint64_t{15};
        const std::uint64_t lower = reinterpret_cast<std::uint64_t>(pool.lower);
        if (parent_stack > lower + ParentClearance) {
            const std::uint64_t candidate = parent_stack - ParentClearance;
            if (candidate < selected) {
                selected = candidate;
            }
        }
    }
    pool.cursor = reinterpret_cast<std::byte*>(selected);
    return {reinterpret_cast<std::uint64_t>(pool.lower), selected, prior};
}

void ReleaseStackWindow(const StackWindow& window) noexcept {
    if (window.restore_pointer == 0 || pool.lower == nullptr) {
        return;
    }
    const auto lower = reinterpret_cast<std::uint64_t>(pool.lower);
    const auto upper = reinterpret_cast<std::uint64_t>(pool.upper);
    if (window.restore_pointer >= lower && window.restore_pointer <= upper) {
        pool.cursor = reinterpret_cast<std::byte*>(window.restore_pointer);
    }
}

}
