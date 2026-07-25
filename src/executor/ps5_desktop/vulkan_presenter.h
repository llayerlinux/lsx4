// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __ANDROID__
struct ANativeWindow;

namespace Lsx4::Ps5Desktop {

bool PresentVulkanFrame(ANativeWindow* window,
                        const std::uint8_t* rgba,
                        std::size_t byte_count,
                        std::uint32_t width,
                        std::uint32_t height,
                        std::uint64_t frame_number);

void ResetVulkanPresenter();

}
#endif
