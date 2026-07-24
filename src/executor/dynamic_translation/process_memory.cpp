// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/process_memory.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#if defined(__linux__)
#include <link.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Lsx4::Translation {
namespace {

bool AddressRangeExists(const std::uint64_t address,
                        const std::size_t byte_count) noexcept {
    return address >= 0x10000 && byte_count != 0 &&
           byte_count - 1 <= std::numeric_limits<std::uint64_t>::max() - address;
}

#if defined(__linux__)
std::pair<std::uintptr_t, std::uintptr_t> OwningImageRange() noexcept {
    static std::once_flag once;
    static std::pair<std::uintptr_t, std::uintptr_t> range{};
    std::call_once(once, [] {
        struct SearchContext {
            std::uintptr_t needle{};
            std::pair<std::uintptr_t, std::uintptr_t> found{};
        } context{reinterpret_cast<std::uintptr_t>(&WriteProcessGuestMemory), {}};
        dl_iterate_phdr(
            [](dl_phdr_info* info, std::size_t, void* opaque) {
                auto& context = *static_cast<SearchContext*>(opaque);
                std::uintptr_t first = std::numeric_limits<std::uintptr_t>::max();
                std::uintptr_t last = 0;
                bool owns = false;
                for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
                    const ElfW(Phdr)& header = info->dlpi_phdr[index];
                    if (header.p_type != PT_LOAD) {
                        continue;
                    }
                    const std::uintptr_t begin = info->dlpi_addr + header.p_vaddr;
                    const std::uintptr_t end = begin + header.p_memsz;
                    owns = owns ||
                           (context.needle >= begin && context.needle < end);
                    first = begin < first ? begin : first;
                    last = end > last ? end : last;
                }
                if (owns && last > first) {
                    context.found = {first, last};
                    return 1;
                }
                return 0;
            },
            &context);
        range = context.found;
    });
    return range;
}
#endif

bool OverlapsOwningImage(const std::uint64_t address,
                         const std::size_t byte_count) noexcept {
#if defined(__linux__)
    const auto [first, last] = OwningImageRange();
    return last > first && address < last && address + byte_count > first;
#else
    (void)address;
    (void)byte_count;
    return false;
#endif
}

}

bool IsAddressRangeRepresentable(const std::uint64_t address,
                                 const std::size_t byte_count) noexcept {
    return byte_count == 0 ||
           byte_count - 1 <= std::numeric_limits<std::uint64_t>::max() - address;
}

bool IsProcessImageMemory(const std::uint64_t address,
                          const std::size_t byte_count) noexcept {
    return AddressRangeExists(address, byte_count) &&
           OverlapsOwningImage(address, byte_count);
}

bool QueryProcessImageBounds(std::uint64_t& first,
                             std::uint64_t& last) noexcept {
#if defined(__linux__)
    const auto bounds = OwningImageRange();
    first = bounds.first;
    last = bounds.second;
    return last > first;
#else
    first = 0;
    last = 0;
    return false;
#endif
}

void* MapProcessMemoryBelow4GiB(const std::size_t byte_count) noexcept {
#if defined(__linux__) && defined(MAP_FIXED_NOREPLACE)
    if (byte_count == 0) {
        return nullptr;
    }
    const long page_query = sysconf(_SC_PAGESIZE);
    if (page_query <= 0) {
        return nullptr;
    }
    const auto page_bytes = static_cast<std::uint64_t>(page_query);
    if (byte_count > std::numeric_limits<std::uint64_t>::max() - (page_bytes - 1)) {
        return nullptr;
    }
    const std::uint64_t mapping_bytes =
        (static_cast<std::uint64_t>(byte_count) + page_bytes - 1) / page_bytes * page_bytes;
    const std::uint64_t address_ceiling =
        std::uint64_t{1} << std::numeric_limits<std::uint32_t>::digits;
    if (mapping_bytes >= address_ceiling) {
        return nullptr;
    }

    std::uint64_t candidate = (address_ceiling - mapping_bytes) / page_bytes * page_bytes;
    while (candidate >= mapping_bytes) {
        void* const requested = reinterpret_cast<void*>(candidate);
        void* const mapping = mmap(requested, mapping_bytes, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (mapping == requested) {
            return mapping;
        }
        if (mapping != MAP_FAILED) {
            munmap(mapping, mapping_bytes);
        }
        candidate -= mapping_bytes;
    }
#else
    (void)byte_count;
#endif
    return nullptr;
}

bool ReadProcessGuestMemory(const std::uint64_t address, void* const destination,
                            const std::size_t byte_count) noexcept {
    if (destination == nullptr || !AddressRangeExists(address, byte_count)) {
        return false;
    }
#if defined(__linux__)
    iovec local{destination, byte_count};
    iovec remote{reinterpret_cast<void*>(address), byte_count};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
           static_cast<ssize_t>(byte_count);
#else
    std::memcpy(destination, reinterpret_cast<const void*>(address), byte_count);
    return true;
#endif
}

bool ReadProcessGuestMemoryWithFaultDispatch(
    const std::uint64_t address, void* const destination,
    const std::size_t byte_count) {
    if (destination == nullptr || !AddressRangeExists(address, byte_count)) {
        return false;
    }
    const auto* const first = reinterpret_cast<const std::byte*>(address);
    auto* const output = static_cast<std::byte*>(destination);
    std::copy_n(first, byte_count, output);
    return true;
}

bool WriteProcessGuestMemory(const std::uint64_t address, const void* const source,
                             const std::size_t byte_count) noexcept {
    if (source == nullptr || !AddressRangeExists(address, byte_count)) {
        return false;
    }
    if (IsProcessImageMemory(address, byte_count)) {
        return false;
    }
#if defined(__linux__)
    iovec local{const_cast<void*>(source), byte_count};
    iovec remote{reinterpret_cast<void*>(address), byte_count};
    return process_vm_writev(getpid(), &local, 1, &remote, 1, 0) ==
           static_cast<ssize_t>(byte_count);
#else
    std::memcpy(reinterpret_cast<void*>(address), source, byte_count);
    return true;
#endif
}

bool WriteProcessGuestMemoryWithFaultDispatch(
    const std::uint64_t address, const void* const source,
    const std::size_t byte_count) {
    if (source == nullptr || !AddressRangeExists(address, byte_count) ||
        IsProcessImageMemory(address, byte_count)) {
        return false;
    }
    auto* const destination = reinterpret_cast<std::byte*>(address);
    const auto* const first = static_cast<const std::byte*>(source);
    std::copy_n(first, byte_count, destination);
    return true;
}

bool ReadProcessGuestScalar(const std::uint64_t address,
                            std::uint64_t& value) noexcept {
    value = 0;
    return ReadProcessGuestMemory(address, &value, sizeof(value));
}

bool WriteProcessGuestScalar(const std::uint64_t address,
                             const std::uint64_t value) noexcept {
    return WriteProcessGuestMemory(address, &value, sizeof(value));
}

}
