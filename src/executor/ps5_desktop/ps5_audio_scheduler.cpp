// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/ps5_desktop/ps5_audio_scheduler.h"

#include "core/libraries/audio/audio_mix.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <limits>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <aaudio/AAudio.h>
#endif

namespace Lsx4::Ps5Desktop {

#ifdef __ANDROID__
namespace {

using Libraries::AudioOut::AudioBufferFormat;
using Libraries::AudioOut::AudioConversionGuard;
using Libraries::AudioOut::AudioMonotonicNanoseconds;
using Libraries::AudioOut::AudioSampleType;
using Libraries::AudioOut::AudioStreamGuard;
using Libraries::AudioOut::ConvertAudioBufferToFloat;
using Libraries::AudioOut::GetAudioOptimizationStats;
using Libraries::AudioOut::MixAudioBufferInPlace;
using Libraries::AudioOut::ObserveAudioStreamConsume;
using Libraries::AudioOut::ObserveAudioStreamSubmit;
using Libraries::AudioOut::RecordAudioBackendWrite;
using Libraries::AudioOut::RecordAudioGuestCopy;
using Libraries::AudioOut::RecordAudioSchedulerBatch;
using Libraries::AudioOut::RecordAudioSchedulerWakeup;
using Libraries::AudioOut::RecordAudioZeroCopyHandoff;

constexpr std::chrono::microseconds BatchCoalescingWindow{150};
constexpr std::int64_t BackendWriteTimeoutNs = INT64_C(20000000);

struct BackendKey {
    std::uint32_t sample_rate{};
    std::uint32_t output_channels{};
    std::uint32_t buffer_frames{};
    std::uint32_t route_group{};

    bool operator==(const BackendKey&) const = default;
};

struct Ps5AudioBackend {
    BackendKey key{};
    AAudioStream* stream{};

    ~Ps5AudioBackend() {
        if (stream != nullptr) {
            (void)AAudioStream_requestStop(stream);
            (void)AAudioStream_close(stream);
            stream = nullptr;
        }
    }
};

[[nodiscard]] std::uint32_t OutputChannels(
    const Ps5AudioPortConfig& config) noexcept {
    return config.guest_channels == 1u ? 1u : 2u;
}

[[nodiscard]] std::uint32_t RouteGroup(
    const Ps5AudioPortConfig& config) noexcept {
    // The controller/pad speaker is paced independently. Main, BGM, voice and
    // personal ports all target Android's shared device output and may be
    // mixed into one backend write when their cadence matches.
    return config.port_type == 4 ? 4u : 0u;
}

[[nodiscard]] AudioBufferFormat GuestFormat(
    const Ps5AudioPortConfig& config) noexcept {
    const auto base_format =
        static_cast<std::uint32_t>(config.format) & 0xffu;
    const bool is_float =
        (base_format >= 3u && base_format <= 5u) ||
        base_format == 7u;
    return {
        .sample_type = is_float
            ? AudioSampleType::Float32
            : AudioSampleType::S16,
        .channels =
            static_cast<std::uint8_t>(config.guest_channels),
        .standard_8ch =
            config.guest_channels == 8u &&
            (base_format == 6u || base_format == 7u),
    };
}

[[nodiscard]] std::shared_ptr<Ps5AudioBackend> OpenBackend(
    const BackendKey& key) {
    AAudioStreamBuilder* builder{};
    auto status = AAudio_createStreamBuilder(&builder);
    if (status != AAUDIO_OK || builder == nullptr) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=builder error=%s\n",
            AAudio_convertResultToText(status));
        return {};
    }
    AAudioStreamBuilder_setDirection(
        builder, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setPerformanceMode(
        builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(
        builder, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setSampleRate(
        builder, static_cast<std::int32_t>(key.sample_rate));
    AAudioStreamBuilder_setChannelCount(
        builder,
        static_cast<std::int32_t>(key.output_channels));
    AAudioStreamBuilder_setFormat(
        builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setBufferCapacityInFrames(
        builder,
        static_cast<std::int32_t>(
            std::max<std::uint32_t>(
                key.buffer_frames * 4u, 512u)));

    auto backend = std::make_shared<Ps5AudioBackend>();
    backend->key = key;
    status =
        AAudioStreamBuilder_openStream(builder, &backend->stream);
    AAudioStreamBuilder_delete(builder);
    if (status != AAUDIO_OK || backend->stream == nullptr) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=open error=%s\n",
            AAudio_convertResultToText(status));
        return {};
    }
    status = AAudioStream_requestStart(backend->stream);
    if (status != AAUDIO_OK) {
        std::fprintf(
            stderr, "PS5_AUDIO_OPEN stage=start error=%s\n",
            AAudio_convertResultToText(status));
        return {};
    }
    return backend;
}

} // namespace

struct Ps5AudioPort {
    std::mutex mutex;
    std::condition_variable_any output_cv;
    Ps5AudioPortConfig config{};
    std::shared_ptr<Ps5AudioBackend> backend;
    std::array<std::vector<std::uint8_t>, 2> input_buffers;
    std::array<std::vector<float>, 2> normalized_buffers;
    AudioConversionGuard conversion_guard;
    AudioConversionGuard mix_guard;
    AudioStreamGuard stream_guard;
    std::int8_t ready_slot{-1};
    std::int8_t processing_slot{-1};
    std::uint8_t next_write_slot{};
    bool closing{};
    std::uint64_t ready_sequence{};
    std::uint64_t ready_timestamp_ns{};
    std::uint64_t next_output_deadline_ns{};
    float ready_gain{1.0f};

    [[nodiscard]] bool CanAcceptBuffer() const noexcept {
        return ready_slot < 0;
    }

    [[nodiscard]] bool IsIdle() const noexcept {
        return ready_slot < 0 && processing_slot < 0;
    }
};

namespace {

class Ps5AudioScheduler {
public:
    ~Ps5AudioScheduler() {
        Stop();
    }

    [[nodiscard]] Ps5AudioPortHandle OpenPort(
        const Ps5AudioPortConfig& config) {
        if (config.buffer_frames == 0u ||
            config.buffer_frames >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max() /
                    4) ||
            config.sample_rate == 0u ||
            config.sample_rate >
                static_cast<std::uint32_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            (config.guest_channels != 1u &&
             config.guest_channels != 2u &&
             config.guest_channels != 8u) ||
            (config.bytes_per_sample != 2u &&
             config.bytes_per_sample != 4u)) {
            return {};
        }
        const auto sample_count =
            static_cast<std::uint64_t>(config.buffer_frames) *
            config.guest_channels;
        if (sample_count >
            std::numeric_limits<std::size_t>::max() /
                config.bytes_per_sample) {
            return {};
        }
        const auto input_bytes =
            static_cast<std::size_t>(sample_count) *
            config.bytes_per_sample;
        const auto output_channels = OutputChannels(config);
        if (config.buffer_frames >
            std::numeric_limits<std::size_t>::max() /
                output_channels) {
            return {};
        }
        const auto output_samples =
            static_cast<std::size_t>(config.buffer_frames) *
            output_channels;
        const BackendKey key{
            .sample_rate = config.sample_rate,
            .output_channels = output_channels,
            .buffer_frames = config.buffer_frames,
            .route_group = RouteGroup(config),
        };

        std::shared_ptr<Ps5AudioBackend> backend;
        {
            std::scoped_lock backend_lock{backend_mutex_};
            for (auto iterator = backends_.begin();
                 iterator != backends_.end();) {
                auto existing = iterator->lock();
                if (!existing) {
                    iterator = backends_.erase(iterator);
                    continue;
                }
                if (existing->key == key) {
                    backend = std::move(existing);
                    break;
                }
                ++iterator;
            }
            if (!backend) {
                backend = OpenBackend(key);
                if (!backend) {
                    return {};
                }
                backends_.push_back(backend);
            }
        }

        auto port = std::make_shared<Ps5AudioPort>();
        port->config = config;
        port->backend = std::move(backend);
        for (auto& buffer : port->input_buffers) {
            buffer.resize(input_bytes);
        }
        for (auto& buffer : port->normalized_buffers) {
            buffer.resize(output_samples);
        }
        EnsureStarted();
        std::fprintf(
            stderr,
            "PS5_AUDIO_OPEN ok rate=%u frames=%u guest_channels=%u "
            "output_channels=%u format=%d route=%u backend=%p "
            "mode=shared-batch\n",
            config.sample_rate, config.buffer_frames,
            config.guest_channels, output_channels, config.format,
            key.route_group,
            static_cast<void*>(port->backend->stream));
        return port;
    }

    void ClosePort(Ps5AudioPortHandle& port) noexcept {
        if (!port) {
            return;
        }
        {
            std::unique_lock port_lock{port->mutex};
            port->closing = true;
            port->output_cv.notify_all();
            port->output_cv.wait(
                port_lock, [&port] { return port->IsIdle(); });
            port->backend.reset();
        }
        port.reset();
    }

    [[nodiscard]] bool QueueBuffer(
        const Ps5AudioPortHandle& port,
        const std::uint64_t source_address, const float gain,
        const std::uint64_t sequence,
        const Ps5AudioGuestReadCallback read_guest) noexcept {
        if (!port || source_address == 0u ||
            read_guest == nullptr) {
            return false;
        }
        const auto wait_started = AudioMonotonicNanoseconds();
        {
            std::unique_lock port_lock{port->mutex};
            port->output_cv.wait(
                port_lock, [&port] {
                    return port->CanAcceptBuffer() ||
                        port->closing;
                });
            if (port->closing || !port->backend) {
                return false;
            }
            const auto slot = port->next_write_slot;
            auto& input = port->input_buffers[slot];
            if (!read_guest(
                    source_address, input.data(), input.size())) {
                std::fprintf(
                    stderr,
                    "PS5_AUDIO_OUTPUT error=guest-read "
                    "address=0x%llx bytes=%zu\n",
                    static_cast<unsigned long long>(
                        source_address),
                    input.size());
                return false;
            }
            const auto submit_timestamp =
                AudioMonotonicNanoseconds();
            ObserveAudioStreamSubmit(
                port->stream_guard, sequence, submit_timestamp);
            port->ready_slot =
                static_cast<std::int8_t>(slot);
            port->ready_sequence = sequence;
            port->ready_timestamp_ns = submit_timestamp;
            port->ready_gain =
                std::clamp(gain, 0.0f, 1.0f);
            port->next_write_slot ^= 1u;
        }
        RecordAudioGuestCopy(
            port->input_buffers[0].size(),
            AudioMonotonicNanoseconds() - wait_started);
        {
            std::scoped_lock queue_lock{queue_mutex_};
            ready_ports_.push_back(port);
        }
        queue_cv_.notify_one();
        return true;
    }

private:
    struct BatchItem {
        Ps5AudioPortHandle port;
        std::uint8_t slot{};
        std::uint64_t sequence{};
        std::uint64_t submit_timestamp_ns{};
        std::uint64_t deadline_ns{};
        float gain{1.0f};

        [[nodiscard]] const void* Input() const noexcept {
            return port->input_buffers[slot].data();
        }

        [[nodiscard]] float* Output() const noexcept {
            return port->normalized_buffers[slot].data();
        }
    };

    void EnsureStarted() {
        std::scoped_lock worker_lock{worker_mutex_};
        if (worker_.joinable()) {
            return;
        }
        worker_ = std::jthread(
            [this](const std::stop_token stop) {
                WorkerLoop(stop);
            });
    }

    void AcquireBatch(std::vector<BatchItem>& batch) {
        std::deque<Ps5AudioPortHandle> ready;
        {
            std::scoped_lock queue_lock{queue_mutex_};
            ready.swap(ready_ports_);
        }
        batch.clear();
        batch.reserve(ready.size());
        for (auto& port : ready) {
            std::unique_lock port_lock{port->mutex};
            if (port->ready_slot < 0) {
                continue;
            }
            const auto slot =
                static_cast<std::uint8_t>(port->ready_slot);
            batch.push_back({
                .port = port,
                .slot = slot,
                .sequence = port->ready_sequence,
                .submit_timestamp_ns =
                    port->ready_timestamp_ns,
                .deadline_ns =
                    port->next_output_deadline_ns,
                .gain = port->ready_gain,
            });
            port->processing_slot =
                static_cast<std::int8_t>(slot);
            port->ready_slot = -1;
            RecordAudioZeroCopyHandoff();
            port_lock.unlock();
            port->output_cv.notify_all();
        }
    }

    static void PaceBatch(
        const std::span<const BatchItem> batch) {
        std::uint64_t deadline =
            std::numeric_limits<std::uint64_t>::max();
        for (const auto& item : batch) {
            if (item.deadline_ns == 0u) {
                return;
            }
            deadline = std::min(deadline, item.deadline_ns);
        }
        const auto now = AudioMonotonicNanoseconds();
        if (deadline > now) {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(deadline - now));
        }
    }

    [[nodiscard]] static std::uint32_t ConvertAndMix(
        const std::span<BatchItem> batch) {
        if (batch.empty()) {
            return 0u;
        }
        auto& destination = batch.front();
        const auto frames =
            destination.port->config.buffer_frames;
        const auto output_channels =
            destination.port->backend->key.output_channels;
        const auto output_samples =
            static_cast<std::size_t>(frames) *
            output_channels;
        (void)ConvertAudioBufferToFloat(
            destination.Input(), frames,
            GuestFormat(destination.port->config),
            output_channels, destination.gain,
            std::span<float>{
                destination.Output(), output_samples},
            destination.port->conversion_guard);

        const AudioBufferFormat normalized_format{
            .sample_type = AudioSampleType::Float32,
            .channels =
                static_cast<std::uint8_t>(output_channels),
            .standard_8ch = false,
        };
        std::uint32_t mixed_ports{};
        for (auto& source : batch.subspan(1)) {
            (void)ConvertAudioBufferToFloat(
                source.Input(), frames,
                GuestFormat(source.port->config),
                output_channels, source.gain,
                std::span<float>{
                    source.Output(), output_samples},
                source.port->conversion_guard);
            (void)MixAudioBufferInPlace(
                destination.Output(), source.Output(), frames,
                normalized_format, 1.0f,
                destination.port->mix_guard);
            ++mixed_ports;
        }
        return mixed_ports;
    }

    [[nodiscard]] static std::int32_t OutputBatch(
        const std::span<BatchItem> batch,
        std::uint64_t& write_finished_ns) {
        if (batch.empty() ||
            !batch.front().port->backend ||
            batch.front().port->backend->stream == nullptr) {
            write_finished_ns = AudioMonotonicNanoseconds();
            return AAUDIO_ERROR_INVALID_STATE;
        }
        auto& backend = *batch.front().port->backend;
        const auto frames =
            batch.front().port->config.buffer_frames;
        const auto xrun_before = std::max<std::int32_t>(
            AAudioStream_getXRunCount(backend.stream), 0);
        const auto write_started = AudioMonotonicNanoseconds();
        const auto written = AAudioStream_write(
            backend.stream, batch.front().Output(),
            static_cast<std::int32_t>(frames),
            BackendWriteTimeoutNs);
        write_finished_ns = AudioMonotonicNanoseconds();
        const auto xrun_after = std::max<std::int32_t>(
            AAudioStream_getXRunCount(backend.stream), 0);
        RecordAudioBackendWrite(
            write_finished_ns - write_started, written, frames,
            static_cast<std::uint32_t>(
                std::max(xrun_after - xrun_before, 0)));
        return written;
    }

    static void CompleteBatch(
        const std::span<BatchItem> batch,
        const std::uint64_t scheduled_at_ns,
        const std::uint64_t completed_at_ns) {
        for (auto& item : batch) {
            auto& port = *item.port;
            const auto period_ns =
                UINT64_C(1000000000) *
                port.config.buffer_frames /
                std::max<std::uint32_t>(
                    port.config.sample_rate, 1u);
            std::unique_lock port_lock{port.mutex};
            (void)ObserveAudioStreamConsume(
                port.stream_guard, item.sequence,
                item.submit_timestamp_ns, completed_at_ns,
                period_ns);
            port.processing_slot = -1;
            if (item.deadline_ns == 0u ||
                scheduled_at_ns >
                    item.deadline_ns + period_ns * 4u) {
                port.next_output_deadline_ns =
                    scheduled_at_ns + period_ns;
            } else {
                port.next_output_deadline_ns =
                    item.deadline_ns + period_ns;
            }
            port_lock.unlock();
            port.output_cv.notify_all();
        }
    }

    void WorkerLoop(const std::stop_token stop) {
        std::fprintf(
            stderr,
            "PS5_AUDIO_SCHEDULER started mode=shared-batch "
            "slots=2\n");
        std::vector<BatchItem> batch;
        while (!stop.stop_requested()) {
            {
                std::unique_lock queue_lock{queue_mutex_};
                queue_cv_.wait(
                    queue_lock, stop, [this] {
                        return !ready_ports_.empty();
                    });
            }
            if (stop.stop_requested()) {
                break;
            }
            // Give concurrent guest audio ports a bounded opportunity to join
            // this cadence without adding a full audio-period of latency.
            std::this_thread::sleep_for(BatchCoalescingWindow);
            AcquireBatch(batch);
            if (batch.empty()) {
                continue;
            }
            RecordAudioSchedulerWakeup();
            std::ranges::sort(
                batch, {}, [](const BatchItem& item) {
                    return item.port->backend.get();
                });
            for (std::size_t first = 0u;
                 first < batch.size();) {
                auto* const backend =
                    batch[first].port->backend.get();
                auto last = first + 1u;
                while (last < batch.size() &&
                       batch[last].port->backend.get() ==
                           backend) {
                    ++last;
                }
                const std::span<BatchItem> group{
                    batch.data() + first, last - first};
                const auto batch_started =
                    std::chrono::steady_clock::now();
                PaceBatch(group);
                const auto scheduled_at =
                    AudioMonotonicNanoseconds();
                const auto mixed_ports = ConvertAndMix(group);
                std::uint64_t completed_at{};
                const auto written =
                    OutputBatch(group, completed_at);
                CompleteBatch(
                    group, scheduled_at, completed_at);
                const auto scheduling_ns =
                    static_cast<std::uint64_t>(
                        std::chrono::duration_cast<
                            std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            batch_started)
                            .count());
                RecordAudioSchedulerBatch(
                    static_cast<std::uint32_t>(group.size()),
                    mixed_ports, scheduling_ns);

                const auto count =
                    output_batches_.fetch_add(
                        1u, std::memory_order_relaxed) +
                    1u;
                if (count <= 8u ||
                    (count % 200u) == 0u ||
                    written < 0) {
                    const auto stats =
                        GetAudioOptimizationStats();
                    std::fprintf(
                        stderr,
                        "PS5_AUDIO_BATCH count=%llu ports=%zu "
                        "mixed=%u frames=%d/%u rate=%u "
                        "channels=%u status=%s scheduler_batches=%llu "
                        "scheduler_ports=%llu simd_frames=%llu "
                        "scalar_frames=%llu underruns=%llu xruns=%llu\n",
                        static_cast<unsigned long long>(count),
                        group.size(), mixed_ports,
                        written >= 0 ? written : 0,
                        group.front().port->config.buffer_frames,
                        group.front().port->config.sample_rate,
                        group.front()
                            .port->backend->key.output_channels,
                        written >= 0
                            ? "ok"
                            : AAudio_convertResultToText(
                                  static_cast<aaudio_result_t>(
                                      written)),
                        static_cast<unsigned long long>(
                            stats.scheduler_batches),
                        static_cast<unsigned long long>(
                            stats.scheduler_ports),
                        static_cast<unsigned long long>(
                            stats.simd_frames),
                        static_cast<unsigned long long>(
                            stats.scalar_frames),
                        static_cast<unsigned long long>(
                            stats.scheduler_underruns),
                        static_cast<unsigned long long>(
                            stats.xruns));
                }
                first = last;
            }
        }
    }

    void Stop() noexcept {
        std::jthread worker;
        {
            std::scoped_lock worker_lock{worker_mutex_};
            if (!worker_.joinable()) {
                return;
            }
            worker_.request_stop();
            queue_cv_.notify_all();
            worker = std::move(worker_);
        }
        worker.join();
    }

    std::mutex worker_mutex_;
    std::jthread worker_;
    std::mutex queue_mutex_;
    std::condition_variable_any queue_cv_;
    std::deque<Ps5AudioPortHandle> ready_ports_;
    std::mutex backend_mutex_;
    std::vector<std::weak_ptr<Ps5AudioBackend>> backends_;
    std::atomic<std::uint64_t> output_batches_{};
};

Ps5AudioScheduler& AudioScheduler() {
    static Ps5AudioScheduler scheduler;
    return scheduler;
}

} // namespace

Ps5AudioPortHandle OpenPs5AudioPort(
    const Ps5AudioPortConfig& config) {
    return AudioScheduler().OpenPort(config);
}

void ClosePs5AudioPort(Ps5AudioPortHandle& port) noexcept {
    AudioScheduler().ClosePort(port);
}

bool QueuePs5AudioBuffer(
    const Ps5AudioPortHandle& port,
    const std::uint64_t source_address, const float gain,
    const std::uint64_t sequence,
    const Ps5AudioGuestReadCallback read_guest) noexcept {
    return AudioScheduler().QueueBuffer(
        port, source_address, gain, sequence, read_guest);
}

#else

struct Ps5AudioPort {};

Ps5AudioPortHandle OpenPs5AudioPort(
    const Ps5AudioPortConfig&) {
    return {};
}

void ClosePs5AudioPort(Ps5AudioPortHandle& port) noexcept {
    port.reset();
}

bool QueuePs5AudioBuffer(
    const Ps5AudioPortHandle&, const std::uint64_t,
    const float, const std::uint64_t,
    const Ps5AudioGuestReadCallback) noexcept {
    return false;
}

#endif

} // namespace Lsx4::Ps5Desktop
