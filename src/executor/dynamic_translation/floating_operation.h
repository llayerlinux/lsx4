// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

enum class FloatingBinaryOperation : std::uint8_t {
    Sum,
    Difference,
    Product,
    Quotient,
    Minimum,
    Maximum,
    AlternatingDifferenceSum,
};

[[nodiscard]] float ApplyFloatingBinary(float left, float right,
                                        FloatingBinaryOperation operation) noexcept;
[[nodiscard]] double ApplyFloatingBinary(double left, double right,
                                          FloatingBinaryOperation operation) noexcept;

}
