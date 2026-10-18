#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/frameProfiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		if (m_lock.test_and_set(std::memory_order_acquire)) {
			WaitContended(thread);
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	void WaitContended(uint32_t thread) noexcept {
		const uint64_t t0    = Common::FrameProfiler::Start();
		uint32_t       spins = 0;
		do {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			// The holder may be in a protection syscall: after a short spin, give up the core
			// instead of starving it.
			if (spins++ < 64) {
#if defined(__x86_64__) || defined(_M_X64)
				_mm_pause();
#elif defined(_M_ARM64)
				__yield();
#elif defined(__aarch64__)
				asm volatile("yield");
#endif
			} else {
				std::this_thread::yield();
			}
		} while (m_lock.test_and_set(std::memory_order_acquire));
		if (t0 != 0) {
			Common::FrameProfiler::AddSince(
			    Common::FrameProfiler::CurrentRole() == Common::FrameProfiler::Role::Gpu
			        ? Common::FrameProfiler::Counter::RegionLockWaitGpuTime
			        : Common::FrameProfiler::Counter::RegionLockWaitGuestTime,
			    t0);
		}
	}

	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// Counts the transitions that can make a finished synchronization incomplete. While both still
// hold the values read before a range was synchronized, synchronizing it again changes nothing:
// no page of it became CPU-dirty, and none of its GPU-dirty pages was handed back.
struct TrackerEpochs {
	std::atomic<uint64_t> cpu_dirtied {0};
	std::atomic<uint64_t> gpu_cleaned {0};
	// One bit per region, set while the region may hold CPU-dirty pages (MayHaveCpuDirty), so
	// syncs of large ranges find those regions a word at a time.
	std::atomic<uint64_t>* cpu_any_words = nullptr;
};

static_assert(std::atomic<uint64_t>::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr, TrackerEpochs* epochs = nullptr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr), m_epochs(epochs) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	// Readable without the lock: false once no page of the region is CPU-dirty, so syncs of
	// large clean ranges skip the region. A write racing the check lands after the sync.
	[[nodiscard]] bool MayHaveCpuDirty() const noexcept {
		return m_cpu_any.load(std::memory_order_acquire);
	}
	// Readable without the lock, like MayHaveCpuDirty. Only the GPU thread makes pages
	// GPU-dirty, so on that thread a false answer holds until it marks some itself.
	[[nodiscard]] bool MayHaveGpuDirty() const noexcept {
		return m_gpu_any.load(std::memory_order_acquire);
	}
	// Readable without the lock: every page is GPU-dirty, so a written sync of the whole region
	// has nothing to upload or mark. A racing hand-back of a page orders after that sync.
	[[nodiscard]] bool IsGpuModifiedThroughout() const noexcept {
		return m_gpu_full.load(std::memory_order_acquire);
	}
	// Requires the lock.
	[[nodiscard]] bool IsGpuModifiedThroughout(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return m_gpu_dirty.AllInRange(start, end);
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().AnyInRange(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.AnyInRange(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (m_cpu_dirty.AnyInRange(start, end)) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto&      bits    = GetBits<source>();
		const bool changes = enable ? !bits.AllInRange(start, end) : bits.AnyInRange(start, end);
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateCpuSummary();
			UpdateProtection<!enable, false>();
		} else {
			UpdateGpuSummary();
			UpdateProtection<enable, true>();
		}
		if (changes) {
			NoteTransition<source, enable>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				UpdateCpuSummary();
				UpdateProtection<true, false>();
			} else {
				UpdateGpuSummary();
				UpdateProtection<false, true>();
			}
			if (mask.Any()) {
				NoteTransition<source, false>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

private:
	void UpdateCpuSummary() noexcept {
		const bool any = m_cpu_dirty.Any();
		if (m_cpu_any.exchange(any, std::memory_order_acq_rel) == any || m_epochs == nullptr ||
		    m_epochs->cpu_any_words == nullptr) {
			return;
		}
		const auto index = m_cpu_addr / TRACKER_REGION_SIZE;
		auto&      word  = m_epochs->cpu_any_words[index / 64u];
		const auto bit   = uint64_t {1} << (index % 64u);
		if (any) {
			word.fetch_or(bit, std::memory_order_release);
		} else {
			word.fetch_and(~bit, std::memory_order_release);
		}
	}
	void UpdateGpuSummary() noexcept {
		m_gpu_any.store(m_gpu_dirty.Any(), std::memory_order_release);
		m_gpu_full.store(m_gpu_dirty.AllInRange(0, TRACKER_REGION_PAGES), std::memory_order_release);
	}

	// After the bits changed, so a synchronization that reads the new count sees them.
	template <DirtySource source, bool enable>
	void NoteTransition() noexcept {
		if (m_epochs == nullptr) {
			return;
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			m_epochs->cpu_dirtied.fetch_add(1, std::memory_order_acq_rel);
		} else if constexpr (source == DirtySource::Gpu && !enable) {
			m_epochs->gpu_cleaned.fetch_add(1, std::memory_order_acq_rel);
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager&      m_page_manager;
	uint64_t          m_cpu_addr = 0;
	TrackerEpochs*    m_epochs   = nullptr;
	RegionBits        m_cpu_dirty;
	RegionBits        m_gpu_dirty;
	std::atomic<bool> m_cpu_any {true};
	std::atomic<bool> m_gpu_any {false};
	std::atomic<bool> m_gpu_full {false};
	RegionBits        m_writable;
	RegionBits        m_readable;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
