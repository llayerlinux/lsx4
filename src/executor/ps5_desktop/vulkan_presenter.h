// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __ANDROID__
struct ANativeWindow;

namespace Lsx4::Ps5Desktop {

// Uploads the software-composited PS5 frame and scales it into the Android
// Vulkan swapchain. Returns false so the caller can use its CPU fallback when
// Vulkan WSI is unavailable or has to be rebuilt.
bool PresentVulkanFrame(ANativeWindow* window,
                        const std::uint8_t* rgba,
                        std::size_t byte_count,
                        std::uint32_t width,
                        std::uint32_t height,
                        std::uint64_t frame_number);

// Surface lifecycle entry points call this before replacing/releasing the
// ANativeWindow. It is safe to call when Vulkan was never initialized.
void ResetVulkanPresenter();

}  // namespace Lsx4::Ps5Desktop
#endif
