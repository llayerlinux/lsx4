// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/packed_lane_ops.h"
#include "executor/dynamic_translation/floating_relation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Executor::Jit::VectorSemantic {
namespace {

constexpr std::size_t kMaximumVectorBytes = 32;

bool ValidWidth(const std::size_t width) noexcept {
    return width != 0 && width <= kMaximumVectorBytes;
}

template <typename T>
T LoadLane(const std::uint8_t* source) noexcept {
    T value{};
    std::memcpy(&value, source, sizeof(value));
    return value;
}

template <typename T>
void StoreLane(std::uint8_t* destination, const T value) noexcept {
    std::memcpy(destination, &value, sizeof(value));
}

template <typename Float>
void ComputeSquareRoots(std::array<std::uint8_t, kMaximumVectorBytes>& output,
                        const std::array<std::uint8_t, kMaximumVectorBytes>& input,
                        const std::size_t total_width) noexcept {
    const std::size_t lane_count = total_width / sizeof(Float);
    for (std::size_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t byte_index = lane * sizeof(Float);
        StoreLane(output.data() + byte_index,
                  std::sqrt(LoadLane<Float>(input.data() + byte_index)));
    }
}

template <typename Float>
void TransformFloatingLanes(
    std::array<std::uint8_t, kMaximumVectorBytes>& output,
    const std::array<std::uint8_t, kMaximumVectorBytes>& left,
    const std::array<std::uint8_t, kMaximumVectorBytes>& right,
    const std::size_t total_width,
    const Lsx4::Translation::FloatingBinaryOperation operation,
    const bool alternating) noexcept {
    const std::size_t count = total_width / sizeof(Float);
    for (std::size_t lane = 0; lane < count; ++lane) {
        const std::size_t cursor = lane * sizeof(Float);
        const auto selected_operation = alternating
            ? ((lane & 1u) == 0
                   ? Lsx4::Translation::FloatingBinaryOperation::Difference
                   : Lsx4::Translation::FloatingBinaryOperation::Sum)
            : operation;
        StoreLane(output.data() + cursor,
                  Lsx4::Translation::ApplyFloatingBinary(
                      LoadLane<Float>(left.data() + cursor),
                      LoadLane<Float>(right.data() + cursor),
                      selected_operation));
    }
}

bool SignedLaneGreater(const std::uint8_t* const left,
                       const std::uint8_t* const right,
                       const std::size_t width) noexcept {
    switch (width) {
    case 1:
        return LoadLane<std::int8_t>(left) > LoadLane<std::int8_t>(right);
    case 2:
        return LoadLane<std::int16_t>(left) > LoadLane<std::int16_t>(right);
    case 4:
        return LoadLane<std::int32_t>(left) > LoadLane<std::int32_t>(right);
    case 8:
        return LoadLane<std::int64_t>(left) > LoadLane<std::int64_t>(right);
    default:
        return false;
    }
}

std::uint64_t LoadUnsignedLane(const std::uint8_t* const source,
                               const std::size_t width) noexcept {
    switch (width) {
    case 1: return LoadLane<std::uint8_t>(source);
    case 2: return LoadLane<std::uint16_t>(source);
    case 4: return LoadLane<std::uint32_t>(source);
    case 8: return LoadLane<std::uint64_t>(source);
    default: return 0;
    }
}

std::int64_t LoadSignedLane(const std::uint8_t* const source,
                           const std::size_t width) noexcept {
    switch (width) {
    case 1: return LoadLane<std::int8_t>(source);
    case 2: return LoadLane<std::int16_t>(source);
    case 4: return LoadLane<std::int32_t>(source);
    case 8: return LoadLane<std::int64_t>(source);
    default: return 0;
    }
}

void StoreUnsignedLane(std::uint8_t* const destination, const std::uint64_t value,
                       const std::size_t width) noexcept {
    switch (width) {
    case 1: StoreLane(destination, static_cast<std::uint8_t>(value)); break;
    case 2: StoreLane(destination, static_cast<std::uint16_t>(value)); break;
    case 4: StoreLane(destination, static_cast<std::uint32_t>(value)); break;
    case 8: StoreLane(destination, value); break;
    default: break;
    }
}

}

MaskedLaneSelection SelectMaskedLanes(const void* const mask,
                                      const std::size_t lane_bytes,
                                      const std::size_t total_width) noexcept {
    MaskedLaneSelection selected{};
    if (mask == nullptr || lane_bytes == 0 || !ValidWidth(total_width) ||
        total_width % lane_bytes != 0) {
        return selected;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(mask);
    std::size_t offset = 0;
    while (offset < total_width) {
        const std::size_t sign_byte = offset + lane_bytes - 1;
        if ((bytes[sign_byte] & 0x80u) != 0) {
            selected.byte_offsets[selected.count++] =
                static_cast<std::uint8_t>(offset);
        }
        offset += lane_bytes;
    }
    return selected;
}

bool ShiftIntegerElements(void* const destination, const void* const source,
                          const std::uint64_t distance,
                          const std::size_t element_width,
                          const std::size_t total_width,
                          const PackedShiftKind kind) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || source == nullptr || !valid_element ||
        total_width > kMaximumVectorBytes || total_width % element_width != 0) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> input{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(input.data(), source, total_width);
    const std::uint64_t lane_bits = element_width * 8u;
    for (std::size_t offset = 0; offset < total_width; offset += element_width) {
        const std::uint64_t unsigned_lane =
            LoadUnsignedLane(input.data() + offset, element_width);
        std::uint64_t shifted = 0;
        if (distance < lane_bits) {
            switch (kind) {
            case PackedShiftKind::LogicalLeft:
                shifted = unsigned_lane << distance;
                break;
            case PackedShiftKind::LogicalRight:
                shifted = unsigned_lane >> distance;
                break;
            case PackedShiftKind::ArithmeticRight:
                shifted = static_cast<std::uint64_t>(
                    LoadSignedLane(input.data() + offset, element_width) >> distance);
                break;
            }
        } else if (kind == PackedShiftKind::ArithmeticRight &&
                   LoadSignedLane(input.data() + offset, element_width) < 0) {
            shifted = UINT64_MAX;
        }
        StoreUnsignedLane(output.data() + offset, shifted, element_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

float CombineHorizontalSingle(const float left, const float right,
                              const bool subtract) noexcept {
#if defined(__aarch64__)
    const float32x2_t left_lane = vdup_n_f32(left);
    const float32x2_t right_lane = vdup_n_f32(right);
    const float32x2_t result = subtract ? vsub_f32(left_lane, right_lane)
                                        : vadd_f32(left_lane, right_lane);
    return vget_lane_f32(result, 0);
#else
    return subtract ? left - right : left + right;
#endif
}

double CombineHorizontalDouble(const double left, const double right,
                               const bool subtract) noexcept {
#if defined(__aarch64__)
    const float64x1_t left_lane = vdup_n_f64(left);
    float64x1_t right_lane = vdup_n_f64(right);
    if (subtract) {
        right_lane = vneg_f64(right_lane);
    }
    return vget_lane_f64(vadd_f64(left_lane, right_lane), 0);
#else
    double adjusted_right = right;
    if (subtract) {
        const auto bits = std::bit_cast<std::uint64_t>(right) ^
                          (std::uint64_t{1} << 63u);
        adjusted_right = std::bit_cast<double>(bits);
    }
    return left + adjusted_right;
#endif
}

std::uint64_t ReadUnsignedLane(const void* const source,
                               const std::size_t lane_bytes) noexcept {
    if (source == nullptr || lane_bytes == 0 || lane_bytes > sizeof(std::uint64_t)) {
        return 0;
    }
    std::uint64_t value{};
    std::memcpy(&value, source, lane_bytes);
    return value;
}

std::int64_t ReadSignedLane(const void* const source,
                            const std::size_t lane_bytes) noexcept {
    const std::size_t bounded = std::min(lane_bytes, sizeof(std::uint64_t));
    if (bounded == 0) {
        return 0;
    }
    const std::uint32_t shift = static_cast<std::uint32_t>(
        (sizeof(std::uint64_t) - bounded) * 8u);
    const auto bits = ReadUnsignedLane(source, bounded);
    return static_cast<std::int64_t>(bits << shift) >> shift;
}

bool WriteLane(void* const destination, const std::uint64_t value,
               const std::size_t lane_bytes) noexcept {
    if (destination == nullptr || lane_bytes == 0 || lane_bytes > sizeof(value)) {
        return false;
    }
    std::memcpy(destination, &value, lane_bytes);
    return true;
}

bool ComposeQwordPair(void* const destination, const void* const lower_source,
                      const std::size_t lower_offset, const void* const upper_source,
                      const std::size_t upper_offset) noexcept {
    if (destination == nullptr || lower_source == nullptr || upper_source == nullptr ||
        lower_offset > 24 || upper_offset > 24) {
        return false;
    }
    std::array<std::byte, 16> pair{};
    std::memcpy(pair.data(), static_cast<const std::byte*>(lower_source) + lower_offset, 8);
    std::memcpy(pair.data() + 8, static_cast<const std::byte*>(upper_source) + upper_offset, 8);
    std::memcpy(destination, pair.data(), pair.size());
    return true;
}

std::uint64_t CollectSignMask(const void* const source, const std::size_t lane_bytes,
                              const std::size_t total_width) noexcept {
    const bool invalid_shape =
        lane_bytes == 0 || lane_bytes > sizeof(std::uint64_t) ||
        total_width > 32 || total_width % lane_bytes != 0;
    if (invalid_shape) {
        return 0;
    }
    if (!source) {
        return 0;
    }
    const auto bytes = static_cast<const std::uint8_t*>(source);
    std::uint64_t mask = 0;
    std::size_t lane = 0;
    for (std::size_t end = lane_bytes; end <= total_width; end += lane_bytes) {
        const auto sign_byte = bytes[end - 1];
        mask |= static_cast<std::uint64_t>(sign_byte >> 7u) << lane;
        ++lane;
    }
    return mask;
}

bool ShuffleBytesByMask(void* const destination, const void* const source,
                        const void* const mask, const std::size_t total_width) noexcept {
    if (destination == nullptr || source == nullptr || mask == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* input = static_cast<const std::uint8_t*>(source);
    const auto* selectors = static_cast<const std::uint8_t*>(mask);
    std::array<std::uint8_t, 32> output{};
    for (std::size_t index = 0; index < total_width; ++index) {
        const std::uint8_t selector = selectors[index];
        if ((selector & 0x80u) == 0) {
            const std::size_t half = (index / 16u) * 16u;
            output[index] = input[half + (selector & 0x0fu)];
        }
    }
    std::memcpy(destination, output.data(), total_width);
    return true;
}

PackedBitTestResult TestPackedBits(const void* const left,
                                   const void* const right,
                                   const std::size_t width) noexcept {
    if (left == nullptr || right == nullptr || !ValidWidth(width)) {
        return {};
    }

    const auto* lhs = static_cast<const std::uint8_t*>(left);
    const auto* rhs = static_cast<const std::uint8_t*>(right);
    bool intersection_nonzero = false;
    bool masked_complement_nonzero = false;
#if defined(__aarch64__)
    if (width == 16 || width == 32) {
        uint8x16_t intersection = vdupq_n_u8(0);
        uint8x16_t masked_complement = vdupq_n_u8(0);
        for (std::size_t offset = 0; offset < width; offset += 16) {
            const uint8x16_t left_lane = vld1q_u8(lhs + offset);
            const uint8x16_t right_lane = vld1q_u8(rhs + offset);
            intersection = vorrq_u8(intersection,
                                    vandq_u8(left_lane, right_lane));
            masked_complement = vorrq_u8(
                masked_complement, vbicq_u8(right_lane, left_lane));
        }
        intersection_nonzero = vmaxvq_u8(intersection) != 0;
        masked_complement_nonzero = vmaxvq_u8(masked_complement) != 0;
    } else
#endif
    {
        for (std::size_t offset = 0; offset < width; ++offset) {
            intersection_nonzero |= (lhs[offset] & rhs[offset]) != 0;
            masked_complement_nonzero |=
                (static_cast<std::uint8_t>(~lhs[offset]) & rhs[offset]) != 0;
        }
    }
    return {!intersection_nonzero, !masked_complement_nonzero};
}

bool SumAbsoluteByteGroups(void* const destination, const void* const left,
                           const void* const right,
                           const std::size_t total_width) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* first = static_cast<const std::uint8_t*>(left);
    const auto* second = static_cast<const std::uint8_t*>(right);
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    constexpr std::size_t kReductionWidth = 8;
    for (std::size_t group = 0; group < total_width / kReductionWidth; ++group) {
        const std::size_t base = group * kReductionWidth;
        std::uint64_t total = 0;
        for (std::size_t member = base; member != base + kReductionWidth; ++member) {
            const int delta = static_cast<int>(first[member]) -
                              static_cast<int>(second[member]);
            total += static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
        }
        StoreLane(output.data() + base, total);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool AlignRight128(void* const destination, const void* const upper,
                   const void* const lower, const std::size_t distance) noexcept {
    if (destination == nullptr || upper == nullptr || lower == nullptr) {
        return false;
    }

    std::array<std::uint8_t, 32> concatenated{};
    std::memcpy(concatenated.data(), lower, 16);
    std::memcpy(concatenated.data() + 16, upper, 16);

    std::array<std::uint8_t, 16> aligned{};
    if (distance < concatenated.size()) {
        const std::size_t available = concatenated.size() - distance;
        std::memcpy(aligned.data(), concatenated.data() + distance,
                    std::min(aligned.size(), available));
    }
    std::memcpy(destination, aligned.data(), aligned.size());
    return true;
}

bool ShuffleImmediate(void* const destination, const void* const left,
                      const void* const right, const std::size_t total_width,
                      const std::uint8_t control,
                      const ImmediateShuffleKind kind) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* first = static_cast<const std::uint8_t*>(left);
    const auto* second = static_cast<const std::uint8_t*>(right);
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    for (std::size_t half = 0; half < total_width; half += 16) {
        const auto selected_lane = [control](const std::size_t lane) {
            return static_cast<std::size_t>((control >> (lane + lane)) & 3u);
        };
        if (kind == ImmediateShuffleKind::Dwords) {
            for (std::uint32_t lane = 0; lane != 4; ++lane) {
                const std::size_t selected = selected_lane(
                    static_cast<std::size_t>(lane));
                std::memcpy(output.data() + half + lane * 4,
                            second + half + selected * 4, 4);
            }
        } else if (kind == ImmediateShuffleKind::LowWords ||
                   kind == ImmediateShuffleKind::HighWords) {
            std::memcpy(output.data() + half, second + half, 16);
            const std::size_t base_word =
                kind == ImmediateShuffleKind::HighWords ? 4u : 0u;
            for (std::uint32_t lane = 0; lane != 4; ++lane) {
                const std::size_t selected = selected_lane(
                    static_cast<std::size_t>(lane));
                std::memcpy(output.data() + half + (base_word + lane) * 2,
                            second + half + (base_word + selected) * 2, 2);
            }
        } else if (kind == ImmediateShuffleKind::SinglePrecisionPairs) {
            for (std::uint32_t lane = 0; lane != 4; ++lane) {
                const std::size_t selected = selected_lane(
                    static_cast<std::size_t>(lane));
                const auto* source = lane < 2 ? first : second;
                std::memcpy(output.data() + half + lane * 4,
                            source + half + selected * 4, 4);
            }
        } else {
            const std::size_t control_pair = half == 0 ? 0u : 2u;
            const std::size_t left_lane = (control >> control_pair) & 1u;
            const std::size_t right_lane = (control >> (control_pair + 1u)) & 1u;
            std::memcpy(output.data() + half, first + half + left_lane * 8, 8);
            std::memcpy(output.data() + half + 8, second + half + right_lane * 8, 8);
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool ShiftBytesInsideHalves(void* const destination, const void* const source,
                            const std::size_t distance,
                            const std::size_t total_width,
                            const bool toward_high_addresses) noexcept {
    if (destination == nullptr || source == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> input{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(input.data(), source, total_width);
    constexpr std::size_t kHalfWidth = 16;
    if (distance < kHalfWidth) {
        const std::size_t copy_width = kHalfWidth - distance;
        for (std::size_t half = 0; half < total_width; half += kHalfWidth) {
            const std::size_t source_bias = toward_high_addresses ? 0 : distance;
            const std::size_t destination_bias = toward_high_addresses ? distance : 0;
            std::copy_n(input.data() + half + source_bias, copy_width,
                        output.data() + half + destination_bias);
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool ApplyBitwise(void* const destination, const void* const left,
                  const void* const right, const std::size_t total_width,
                  const PackedLogicKind kind) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* first = static_cast<const std::uint8_t*>(left);
    const auto* second = static_cast<const std::uint8_t*>(right);
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    for (std::size_t position = 0; position != total_width; ++position) {
        switch (kind) {
        case PackedLogicKind::And:
            output[position] = first[position] & second[position];
            break;
        case PackedLogicKind::AndNot:
            output[position] = static_cast<std::uint8_t>(~first[position]) & second[position];
            break;
        case PackedLogicKind::Or:
            output[position] = first[position] | second[position];
            break;
        case PackedLogicKind::Xor:
            output[position] = first[position] ^ second[position];
            break;
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool PackWithSaturation(void* const destination, const void* const left,
                        const void* const right, const std::size_t total_width,
                        const std::size_t source_element_width,
                        const bool unsigned_destination) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32) ||
        (source_element_width != 2 && source_element_width != 4)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> inputs[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> packed{};
    std::memcpy(inputs[0].data(), left, total_width);
    std::memcpy(inputs[1].data(), right, total_width);

    const std::size_t packed_width = source_element_width / 2;
    const unsigned packed_bits = static_cast<unsigned>(packed_width * 8);
    const std::int64_t lower = unsigned_destination
                                   ? 0
                                   : -(std::int64_t{1} << (packed_bits - 1));
    const std::int64_t upper = unsigned_destination
                                   ? static_cast<std::int64_t>(
                                         (std::uint64_t{1} << packed_bits) - 1)
                                   : (std::int64_t{1} << (packed_bits - 1)) - 1;
    constexpr std::size_t kLaneWidth = 16;
    const std::size_t source_count = kLaneWidth / source_element_width;
    for (std::size_t lane_base = 0; lane_base < total_width;
         lane_base += kLaneWidth) {
        for (std::size_t side = 0; side < std::size(inputs); ++side) {
            for (std::size_t element = 0; element < source_count; ++element) {
                const auto value = LoadSignedLane(
                    inputs[side].data() + lane_base + element * source_element_width,
                    source_element_width);
                const auto narrowed = static_cast<std::uint64_t>(
                    std::clamp(value, lower, upper));
                const std::size_t destination_element =
                    side * source_count + element;
                StoreUnsignedLane(
                    packed.data() + lane_base + destination_element * packed_width,
                    narrowed, packed_width);
            }
        }
    }
    std::memcpy(destination, packed.data(), packed.size());
    return true;
}

bool ApplyFloatingElements(
    void* const destination, const void* const left, const void* const right,
    const std::size_t element_width, const std::size_t total_width,
    const Lsx4::Translation::FloatingBinaryOperation operation) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (element_width != sizeof(float) && element_width != sizeof(double)) ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> operands[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(operands[0].data(), left, total_width);
    std::memcpy(operands[1].data(), right, total_width);
    if (element_width == sizeof(double)) {
        TransformFloatingLanes<double>(output, operands[0], operands[1],
                                       total_width, operation, false);
    } else {
        TransformFloatingLanes<float>(output, operands[0], operands[1],
                                      total_width, operation, false);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool ApplyAlternatingFloatingElements(
    void* const destination, const void* const left, const void* const right,
    const std::size_t element_width, const std::size_t total_width) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (element_width != sizeof(float) && element_width != sizeof(double)) ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> operands[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(operands[0].data(), left, total_width);
    std::memcpy(operands[1].data(), right, total_width);
    if (element_width == sizeof(double)) {
        TransformFloatingLanes<double>(
            output, operands[0], operands[1], total_width,
            Lsx4::Translation::FloatingBinaryOperation::Sum, true);
    } else {
        TransformFloatingLanes<float>(
            output, operands[0], operands[1], total_width,
            Lsx4::Translation::FloatingBinaryOperation::Sum, true);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool AbsoluteIntegerElements(void* const destination, const void* const source,
                             const std::size_t element_width,
                             const std::size_t total_width) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || source == nullptr || !valid_element ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> input{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(input.data(), source, total_width);
    const unsigned bits = static_cast<unsigned>(element_width * 8);
    const std::uint64_t width_mask = bits == 64
                                         ? UINT64_MAX
                                         : (std::uint64_t{1} << bits) - 1;
    const std::uint64_t sign_mask = std::uint64_t{1} << (bits - 1);
    for (std::size_t cursor = 0; cursor < total_width;
         cursor += element_width) {
        const std::uint64_t lane =
            LoadUnsignedLane(input.data() + cursor, element_width);
        const std::uint64_t magnitude = (lane & sign_mask) == 0
                                            ? lane
                                            : (~lane + 1) & width_mask;
        StoreUnsignedLane(output.data() + cursor, magnitude, element_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool DotProductSinglePrecision(void* const destination, const void* const left,
                               const void* const right, const std::uint8_t control,
                               const std::size_t total_width) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* lhs = static_cast<const std::byte*>(left);
    const auto* rhs = static_cast<const std::byte*>(right);
    std::array<std::byte, 32> output{};
    std::size_t group = 0;
    while (group != total_width) {
        float sum = 0.0f;
        std::size_t lane = 0;
        while (lane != 4) {
            if ((control & (0x10u << lane)) == 0) {
                ++lane;
                continue;
            }
            float left_value{};
            float right_value{};
            std::memcpy(&left_value, lhs + group + lane * 4, sizeof(float));
            std::memcpy(&right_value, rhs + group + lane * 4, sizeof(float));
            sum += left_value * right_value;
            ++lane;
        }
        lane = 0;
        while (lane != 4) {
            if ((control & (1u << lane)) != 0) {
                std::memcpy(output.data() + group + lane * 4, &sum, sizeof(sum));
            }
            ++lane;
        }
        group += 16;
    }
    std::memcpy(destination, output.data(), total_width);
    return true;
}

bool ConvertDwordFloatElements(void* const destination, const void* const source,
                               const bool integer_to_float, const bool truncate,
                               const std::size_t total_width,
                               const std::uint32_t mxcsr) noexcept {
    if (destination == nullptr || source == nullptr ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* input = static_cast<const std::byte*>(source);
    std::array<std::byte, 32> output{};
    const std::uint32_t rounding_mode = (mxcsr >> 13u) & 3u;
    for (std::size_t offset = 0; offset < total_width; offset += 4) {
        const std::uint32_t bits =
            static_cast<std::uint32_t>(ReadUnsignedLane(input + offset, 4));
        std::uint32_t converted = 0;
        if (integer_to_float) {
            converted = std::bit_cast<std::uint32_t>(
                static_cast<float>(static_cast<std::int32_t>(bits)));
        } else {
            const float value = std::bit_cast<float>(bits);
            const double rounded = truncate
                ? std::trunc(value)
                : Lsx4::Translation::RoundFloatingValue(value, rounding_mode);
            converted = 0x80000000u;
            const auto minimum =
                static_cast<double>(std::numeric_limits<std::int32_t>::min());
            const auto maximum =
                static_cast<double>(std::numeric_limits<std::int32_t>::max());
            const bool representable =
                std::isfinite(rounded) && rounded >= minimum && rounded <= maximum;
            if (representable) {
                const auto signed_value = static_cast<std::int32_t>(rounded);
                converted = std::bit_cast<std::uint32_t>(signed_value);
            }
        }
        if (!WriteLane(output.data() + offset, converted, 4)) {
            return false;
        }
    }
    std::memcpy(destination, output.data(), total_width);
    return true;
}

bool ConvertFloatingElementWidth(void* const destination,
                                 const void* const source,
                                 const std::size_t destination_width,
                                 const std::size_t source_width,
                                 const bool widen_to_double) noexcept {
    if (destination == nullptr || source == nullptr ||
        (destination_width != 16 && destination_width != 32) ||
        source_width == 0 || source_width > kMaximumVectorBytes) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> input{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(input.data(), source, source_width);
    if (widen_to_double) {
        const std::size_t lanes = destination_width / sizeof(double);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            StoreLane<double>(output.data() + lane * sizeof(double),
                              static_cast<double>(LoadLane<float>(
                                  input.data() + lane * sizeof(float))));
        }
    } else {
        const std::size_t lanes = std::min(destination_width / sizeof(float),
                                           source_width / sizeof(double));
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            StoreLane<float>(output.data() + lane * sizeof(float),
                             static_cast<float>(LoadLane<double>(
                                 input.data() + lane * sizeof(double))));
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool DuplicateDwordParity(void* const destination, const void* const source,
                          const std::size_t width,
                          const bool odd_lanes) noexcept {
    if (destination == nullptr || source == nullptr ||
        (width != 16 && width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> source_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(source_snapshot.data(), source, width);
#if defined(__aarch64__)
    for (std::size_t offset = 0; offset < width; offset += 16) {
        const uint32x4_t input = vreinterpretq_u32_u8(
            vld1q_u8(source_snapshot.data() + offset));
        const uint32x4_t selected = odd_lanes
            ? vuzp2q_u32(input, input)
            : vuzp1q_u32(input, input);
        vst1q_u8(result.data() + offset,
                 vreinterpretq_u8_u32(vzip1q_u32(selected, selected)));
    }
#else
    for (std::size_t base = 0; base < width; base += 16) {
        const std::size_t first = base + (odd_lanes ? 4u : 0u);
        const std::size_t second = base + (odd_lanes ? 12u : 8u);
        const std::uint32_t a = LoadLane<std::uint32_t>(source_snapshot.data() + first);
        const std::uint32_t b = LoadLane<std::uint32_t>(source_snapshot.data() + second);
        StoreLane(result.data() + base, a);
        StoreLane(result.data() + base + 4, a);
        StoreLane(result.data() + base + 8, b);
        StoreLane(result.data() + base + 12, b);
    }
#endif
    std::memcpy(destination, result.data(), result.size());
    return true;
}

bool DuplicateLowQwordPerLane(void* const destination, const void* const source,
                              const std::size_t width,
                              const bool zero_upper) noexcept {
    if (destination == nullptr || source == nullptr ||
        (width != 16 && width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> source_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(source_snapshot.data(), source, width);
#if defined(__aarch64__)
    for (std::size_t base = 0; base < width; base += 16) {
        const uint64x2_t input = vreinterpretq_u64_u8(
            vld1q_u8(source_snapshot.data() + base));
        vst1q_u8(result.data() + base,
                 vreinterpretq_u8_u64(vdupq_laneq_u64(input, 0)));
    }
#else
    for (std::size_t base = 0; base < width; base += 16) {
        const std::uint64_t value =
            LoadLane<std::uint64_t>(source_snapshot.data() + base);
        StoreLane(result.data() + base, value);
        StoreLane(result.data() + base + 8, value);
    }
#endif
    std::memcpy(destination, result.data(), width);
    if (zero_upper && width < result.size()) {
        std::memset(static_cast<std::uint8_t*>(destination) + width, 0,
                    result.size() - width);
    }
    return true;
}

bool AddOrSubtractPacked(void* const destination, const void* const left,
                         const void* const right,
                         const std::size_t element_width,
                         const std::size_t total_width,
                         const bool subtract) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32) ||
        (element_width != 1 && element_width != 2 && element_width != 4 &&
         element_width != 8)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> lhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> rhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(lhs.data(), left, total_width);
    std::memcpy(rhs.data(), right, total_width);
#if defined(__aarch64__)
    for (std::size_t offset = 0; offset < total_width; offset += 16) {
        const uint8x16_t left_bytes = vld1q_u8(lhs.data() + offset);
        const uint8x16_t right_bytes = vld1q_u8(rhs.data() + offset);
        uint8x16_t output{};
        switch (element_width) {
        case 1:
            output = subtract ? vsubq_u8(left_bytes, right_bytes)
                              : vaddq_u8(left_bytes, right_bytes);
            break;
        case 2:
            output = vreinterpretq_u8_u16(subtract
                ? vsubq_u16(vreinterpretq_u16_u8(left_bytes),
                            vreinterpretq_u16_u8(right_bytes))
                : vaddq_u16(vreinterpretq_u16_u8(left_bytes),
                            vreinterpretq_u16_u8(right_bytes)));
            break;
        case 4:
            output = vreinterpretq_u8_u32(subtract
                ? vsubq_u32(vreinterpretq_u32_u8(left_bytes),
                            vreinterpretq_u32_u8(right_bytes))
                : vaddq_u32(vreinterpretq_u32_u8(left_bytes),
                            vreinterpretq_u32_u8(right_bytes)));
            break;
        default:
            output = vreinterpretq_u8_u64(subtract
                ? vsubq_u64(vreinterpretq_u64_u8(left_bytes),
                            vreinterpretq_u64_u8(right_bytes))
                : vaddq_u64(vreinterpretq_u64_u8(left_bytes),
                            vreinterpretq_u64_u8(right_bytes)));
            break;
        }
        vst1q_u8(result.data() + offset, output);
    }
#else
    for (std::size_t offset = 0; offset < total_width;
         offset += element_width) {
        std::uint64_t left_value = 0;
        std::uint64_t right_value = 0;
        std::memcpy(&left_value, lhs.data() + offset, element_width);
        std::memcpy(&right_value, rhs.data() + offset, element_width);
        const std::uint64_t output = subtract ? left_value - right_value
                                              : left_value + right_value;
        std::memcpy(result.data() + offset, &output, element_width);
    }
#endif
    std::memcpy(destination, result.data(), result.size());
    return true;
}

bool AddOrSubtractSaturating(void* const destination, const void* const left,
                             const void* const right,
                             const std::size_t element_width,
                             const std::size_t total_width,
                             const bool subtract,
                             const bool signed_elements) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (element_width != 1 && element_width != 2) ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* lhs = static_cast<const std::uint8_t*>(left);
    const auto* rhs = static_cast<const std::uint8_t*>(right);
    auto* output = static_cast<std::uint8_t*>(destination);
    const std::int64_t signed_limit = std::int64_t{1} << (element_width * 8u - 1u);
    const std::uint64_t unsigned_limit =
        (std::uint64_t{1} << (element_width * 8u)) - 1u;
    for (std::size_t offset = 0; offset < total_width; offset += element_width) {
        std::uint64_t lane{};
        if (signed_elements) {
            const std::int64_t l = ReadSignedLane(lhs + offset, element_width);
            const std::int64_t r = ReadSignedLane(rhs + offset, element_width);
            const std::int64_t raw = subtract ? l - r : l + r;
            lane = static_cast<std::uint64_t>(
                std::clamp(raw, -signed_limit, signed_limit - 1));
        } else {
            const std::uint64_t l = ReadUnsignedLane(lhs + offset, element_width);
            const std::uint64_t r = ReadUnsignedLane(rhs + offset, element_width);
            lane = subtract ? (l >= r ? l - r : 0u)
                            : std::min(unsigned_limit, l + r);
        }
        if (!WriteLane(output + offset, lane, element_width)) {
            return false;
        }
    }
    return true;
}

bool ReplicateElement(void* const destination, const void* const element,
                      const std::size_t element_width,
                      const std::size_t total_width) noexcept {
    const bool supported_element = element_width == 1 || element_width == 2 ||
                                   element_width == 4 || element_width == 8 ||
                                   element_width == 16;
    if (destination == nullptr || element == nullptr || !supported_element ||
        (total_width != 16 && total_width != 32) ||
        element_width > total_width) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(output.data(), element, element_width);
    std::size_t initialized = element_width;
    while (initialized < total_width) {
        const std::size_t copy_count =
            std::min(initialized, total_width - initialized);
        std::memcpy(output.data() + initialized, output.data(), copy_count);
        initialized += copy_count;
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool SquareRootElements(void* const destination, const void* const source,
                        const std::size_t element_width,
                        const std::size_t total_width) noexcept {
    if (destination == nullptr || source == nullptr ||
        (element_width != sizeof(float) && element_width != sizeof(double)) ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> input{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(input.data(), source, total_width);
    if (element_width == sizeof(double)) {
        ComputeSquareRoots<double>(output, input, total_width);
    } else {
        ComputeSquareRoots<float>(output, input, total_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool CompareIntegerElements(void* const destination, const void* const left,
                            const void* const right,
                            const std::size_t element_width,
                            const std::size_t total_width,
                            const bool signed_greater_than) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || left == nullptr || right == nullptr ||
        !valid_element || (total_width != 16 && total_width != 32)) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> lhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> rhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(lhs.data(), left, total_width);
    std::memcpy(rhs.data(), right, total_width);
    for (std::size_t cursor = 0; cursor < total_width; cursor += element_width) {
        const bool selected = signed_greater_than
            ? SignedLaneGreater(lhs.data() + cursor, rhs.data() + cursor,
                                element_width)
            : std::memcmp(lhs.data() + cursor, rhs.data() + cursor,
                          element_width) == 0;
        std::memset(output.data() + cursor, selected ? 0xff : 0x00,
                    element_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool SelectIntegerExtrema(void* const destination, const void* const left,
                          const void* const right, const std::size_t element_width,
                          const std::size_t total_width, const bool signed_comparison,
                          const bool select_maximum) noexcept {
    const bool supported_lane = element_width == 1 || element_width == 2 ||
                                element_width == 4 || element_width == 8;
    if (destination == nullptr || left == nullptr || right == nullptr ||
        !supported_lane || (total_width != 16 && total_width != 32)) {
        return false;
    }
    const auto* left_bytes = static_cast<const std::byte*>(left);
    const auto* right_bytes = static_cast<const std::byte*>(right);
    std::array<std::byte, 32> output{};
    for (std::size_t offset = 0; offset < total_width; offset += element_width) {
        bool choose_right = false;
        if (signed_comparison) {
            const auto lhs = ReadSignedLane(left_bytes + offset, element_width);
            const auto rhs = ReadSignedLane(right_bytes + offset, element_width);
            choose_right = select_maximum ? rhs > lhs : rhs <= lhs;
        } else {
            const auto lhs = ReadUnsignedLane(left_bytes + offset, element_width);
            const auto rhs = ReadUnsignedLane(right_bytes + offset, element_width);
            choose_right = select_maximum ? rhs > lhs : rhs <= lhs;
        }
        std::memcpy(output.data() + offset,
                    (choose_right ? right_bytes : left_bytes) + offset, element_width);
    }
    std::memcpy(destination, output.data(), total_width);
    return true;
}

bool ExtendIntegerElements(void* const destination, const void* const source,
                           const std::size_t source_element_width,
                           const std::size_t destination_element_width,
                           const std::size_t total_width,
                           const bool sign_extend) noexcept {
    const bool source_width_supported = source_element_width == 1 ||
                                        source_element_width == 2 ||
                                        source_element_width == 4;
    if (destination == nullptr || source == nullptr || !source_width_supported ||
        destination_element_width <= source_element_width || destination_element_width > 8 ||
        (total_width != 16 && total_width != 32) ||
        total_width % destination_element_width != 0) {
        return false;
    }
    const auto* input = static_cast<const std::byte*>(source);
    std::array<std::byte, 32> output{};
    const std::size_t lanes = total_width / destination_element_width;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        const std::byte* element = input + lane * source_element_width;
        const std::uint64_t value = sign_extend
            ? static_cast<std::uint64_t>(ReadSignedLane(element, source_element_width))
            : ReadUnsignedLane(element, source_element_width);
        if (!WriteLane(output.data() + lane * destination_element_width,
                       value, destination_element_width)) {
            return false;
        }
    }
    std::memcpy(destination, output.data(), total_width);
    return true;
}

bool InterleaveElementHalves(void* const destination, const void* const left,
                             const void* const right,
                             const std::size_t element_width,
                             const std::size_t total_width,
                             const bool select_upper_half) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || left == nullptr || right == nullptr ||
        !valid_element || (total_width != 16 && total_width != 32)) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> sources[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(sources[0].data(), left, total_width);
    std::memcpy(sources[1].data(), right, total_width);
    constexpr std::size_t vector_half_width = 16;
    constexpr std::size_t selected_span = vector_half_width / 2;
    const std::size_t source_bias = select_upper_half ? selected_span : 0;
    const std::size_t elements_per_source = selected_span / element_width;
    for (std::size_t vector_base = 0; vector_base < total_width;
         vector_base += vector_half_width) {
        for (std::size_t source_element = 0;
             source_element < elements_per_source; ++source_element) {
            for (std::size_t source_index = 0; source_index < 2;
                 ++source_index) {
                const std::size_t destination_element =
                    source_element * 2 + source_index;
                std::memcpy(
                    output.data() + vector_base +
                        destination_element * element_width,
                    sources[source_index].data() + vector_base + source_bias +
                        source_element * element_width,
                    element_width);
            }
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool SelectElementsByImmediate(void* const destination, const void* const left,
                               const void* const right,
                               const std::uint32_t selection_bits,
                               const std::size_t element_width,
                               const std::size_t total_width,
                               const bool restart_bits_per_half) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || left == nullptr || right == nullptr ||
        !valid_element || (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> inputs[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(inputs[0].data(), left, total_width);
    std::memcpy(inputs[1].data(), right, total_width);
    for (std::size_t offset = 0; offset < total_width;
         offset += element_width) {
        const std::size_t selector_index = restart_bits_per_half
            ? (offset % 16) / element_width
            : offset / element_width;
        const std::size_t input_index =
            (selection_bits >> selector_index) & 1u;
        std::memcpy(output.data() + offset, inputs[input_index].data() + offset,
                    element_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool SelectElementsBySign(void* const destination, const void* const left,
                          const void* const right, const void* const selectors,
                          const std::size_t element_width,
                          const std::size_t total_width) noexcept {
    const bool valid_element = element_width == 1 || element_width == 2 ||
                               element_width == 4 || element_width == 8;
    if (destination == nullptr || left == nullptr || right == nullptr ||
        selectors == nullptr || !valid_element ||
        (total_width != 16 && total_width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> inputs[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> mask{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(inputs[0].data(), left, total_width);
    std::memcpy(inputs[1].data(), right, total_width);
    std::memcpy(mask.data(), selectors, total_width);
    for (std::size_t offset = 0; offset < total_width;
         offset += element_width) {
        const std::size_t sign_position = offset + element_width - 1;
        const std::size_t input_index = (mask[sign_position] >> 7) & 1u;
        std::memcpy(output.data() + offset, inputs[input_index].data() + offset,
                    element_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool Replace128BitHalf(void* const destination, const void* const original,
                       const void* const replacement,
                       const std::uint32_t half_index) noexcept {
    if (destination == nullptr || original == nullptr || replacement == nullptr) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::array<std::uint8_t, 16> incoming{};
    std::memcpy(output.data(), original, output.size());
    std::memcpy(incoming.data(), replacement, incoming.size());
    std::memcpy(output.data() + (half_index & 1u) * incoming.size(),
                incoming.data(), incoming.size());
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool Permute128BitHalves(void* const destination, const void* const left,
                         const void* const right,
                         const std::uint32_t selection) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr) {
        return false;
    }
    constexpr std::size_t half_width = 16;
    std::array<std::uint8_t, kMaximumVectorBytes> inputs[2]{};
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::memcpy(inputs[0].data(), left, inputs[0].size());
    std::memcpy(inputs[1].data(), right, inputs[1].size());
    for (std::size_t destination_half = 0; destination_half < 2;
         ++destination_half) {
        const std::uint32_t command =
            (selection >> (destination_half * 4)) & 0xfu;
        if ((command & 8u) != 0) {
            continue;
        }
        const std::size_t input_image = (command >> 1) & 1u;
        const std::size_t input_half = command & 1u;
        std::memcpy(output.data() + destination_half * half_width,
                    inputs[input_image].data() + input_half * half_width,
                    half_width);
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

bool InsertSinglePrecisionElement(void* const destination,
                                  const void* const original,
                                  const void* const source,
                                  const std::uint32_t control,
                                  const bool source_is_memory) noexcept {
    if (destination == nullptr || original == nullptr || source == nullptr) {
        return false;
    }
    constexpr std::size_t element_width = sizeof(float);
    std::array<std::uint8_t, kMaximumVectorBytes> output{};
    std::array<std::uint8_t, 16> incoming{};
    std::memcpy(output.data(), original, 16);
    std::memcpy(incoming.data(), source, incoming.size());
    const std::size_t source_index =
        source_is_memory ? 0 : ((control >> 6) & 3u);
    const std::size_t destination_index = (control >> 4) & 3u;
    std::memcpy(output.data() + destination_index * element_width,
                incoming.data() + source_index * element_width, element_width);
    for (std::size_t index = 0; index < 4; ++index) {
        if ((control & (1u << index)) != 0) {
            std::memset(output.data() + index * element_width, 0, element_width);
        }
    }
    std::memcpy(destination, output.data(), output.size());
    return true;
}

}
