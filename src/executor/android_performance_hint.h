// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Executor::AndroidPerformanceHint {

void SetEnabled(bool enabled) noexcept;
[[nodiscard]] bool IsEnabled() noexcept;

void ReportFramePresented() noexcept;

} // namespace Executor::AndroidPerformanceHint
