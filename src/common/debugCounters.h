#ifndef KYTY_COMMON_DEBUG_COUNTERS_H_
#define KYTY_COMMON_DEBUG_COUNTERS_H_

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Common::DebugCounters {

static_assert(std::atomic<uint64_t>::is_always_lock_free &&
              std::atomic<uint8_t>::is_always_lock_free);

// Collect events only while the panel is visible or a frame profiler capture runs. Relaxed,
// lock-free atomics keep updates cheap and safe in signal handlers. Rates come from comparing two
// snapshots.
enum class Counter {
	// GPU work.
	Draws,
	DrawInstances,
	IndirectDraws,
	Dispatches,
	QueueSubmits,
	GpuWaits, // CPU blocked on a GPU timeline semaphore.
	GpuWaitNs,
	GpuFaults, // Guest writes to GPU-tracked memory.
	ShadersCompiled,
	ShaderCompileNs,
	PipelinesCreated,
	PipelineCreateNs,
	// Caches.
	TextureUploadBytes,
	TextureDownloadBytes,
	TextureEvictions,
	BufferUploadBytes,
	BufferDownloadBytes,
	BufferEvictions,
	// Presentation.
	FlipsRequested,
	FlipsRejected, // Guest flip queue was full.
	LateVblanks,   // The present thread fell behind the vblank rate.
	SwapchainRecreations,
	DroppedPresents,
	// Emulator.
	EmulatedInstructions, // Host CPU lacked an instruction the guest used.
	UnresolvedImportCalls,
	GuestFileReadBytes,
	GuestFileWriteBytes,
	AudioUnderruns,
	Count
};

enum class Peak { ShaderCompileNs, PipelineCreateNs, Count };

namespace Detail {
// Who collects: the panel, the frame profiler.
inline constexpr uint8_t     PANEL    = 1u;
inline constexpr uint8_t     PROFILER = 2u;
inline std::atomic<uint8_t>  g_collectors {0};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t> g_peaks[static_cast<int>(Peak::Count)] {};
} // namespace Detail

// Whether the panel is shown.
[[nodiscard]] inline bool Enabled() noexcept {
	return (Detail::g_collectors.load(std::memory_order_relaxed) & Detail::PANEL) != 0;
}

inline void SetEnabled(bool enabled) noexcept {
	if (enabled) {
		Detail::g_collectors.fetch_or(Detail::PANEL, std::memory_order_relaxed);
	} else {
		Detail::g_collectors.fetch_and(static_cast<uint8_t>(~Detail::PANEL),
		                               std::memory_order_relaxed);
	}
}

// Whether events are counted: for the panel or a frame profiler capture.
[[nodiscard]] inline bool Collecting() noexcept {
	return Detail::g_collectors.load(std::memory_order_relaxed) != 0;
}

// The frame profiler reads these counters during a capture, without showing the panel.
inline void SetProfilerCollecting(bool collecting) noexcept {
	if (collecting) {
		Detail::g_collectors.fetch_or(Detail::PROFILER, std::memory_order_relaxed);
	} else {
		Detail::g_collectors.fetch_and(static_cast<uint8_t>(~Detail::PROFILER),
		                               std::memory_order_relaxed);
	}
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (!Collecting()) {
		return;
	}
	Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
}

class ScopedTimer final {
public:
	ScopedTimer(Counter count, Counter duration, Peak peak = Peak::Count)
	    : m_count(count), m_duration(duration), m_peak(peak),
	      m_start(Collecting() ? std::chrono::steady_clock::now()
	                        : std::chrono::steady_clock::time_point {}) {}
	~ScopedTimer() {
		if (m_start == std::chrono::steady_clock::time_point {} || !Collecting()) {
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count());
		Add(m_count);
		Add(m_duration, elapsed);
		if (m_peak != Peak::Count) {
			auto& peak     = Detail::g_peaks[static_cast<int>(m_peak)];
			auto  previous = peak.load(std::memory_order_relaxed);
			while (previous < elapsed &&
			       !peak.compare_exchange_weak(previous, elapsed, std::memory_order_relaxed)) {
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_count;
	Counter                               m_duration;
	Peak                                  m_peak;
	std::chrono::steady_clock::time_point m_start;
};

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	return Detail::g_counters[static_cast<int>(counter)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
