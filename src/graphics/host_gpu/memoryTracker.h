#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// Queue after changing ownership; a concurrent drain must not consume an event
	// before the dirty pages it describes become visible.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	// Waits for the holds queued into the set it takes (HoldBdaSync). A thread must not take
	// while it holds.
	[[nodiscard]] RangeSet TakeBdaSync();
	[[nodiscard]] bool     HasBdaSync() const noexcept {
		return m_bda_sync_nonempty.load(std::memory_order_acquire);
	}
	// Counts the takes of a non-empty set.
	[[nodiscard]] uint64_t BdaSyncGeneration() const noexcept {
		return m_bda_sync_generation.load(std::memory_order_relaxed);
	}
	// Keeps TakeBdaSync from returning the set it was queued into until the holder drops it.
	class BdaSyncHold {
	public:
		BdaSyncHold() = default;
		explicit BdaSyncHold(std::atomic<uint32_t>& in_flight): m_in_flight(&in_flight) {}
		BdaSyncHold(BdaSyncHold&& other) noexcept
		    : m_in_flight(std::exchange(other.m_in_flight, nullptr)) {}
		BdaSyncHold& operator=(BdaSyncHold&&)      = delete;
		BdaSyncHold(const BdaSyncHold&)            = delete;
		BdaSyncHold& operator=(const BdaSyncHold&) = delete;
		~BdaSyncHold() {
			if (m_in_flight != nullptr) {
				m_in_flight->fetch_sub(1, std::memory_order_release);
			}
		}

	private:
		std::atomic<uint32_t>* m_in_flight = nullptr;
	};
	// QueueBdaSync before the ownership change, returning a hold to drop once it is made: the
	// take that gets the range then sees its pages changed whole, so a CPU write through pages
	// turned writable cannot slip between the take and a queue made after the change. Holders
	// do not wait for each other (faults on many threads); only a take waits, for the holds of
	// the set it takes.
	[[nodiscard]] BdaSyncHold HoldBdaSync(uint64_t vaddr, uint64_t size);
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		InvalidateRegion(vaddr, size, std::forward<Flush>(on_flush),
		                 [](uint64_t, uint64_t) noexcept { return 0; });
	}
	// As above. on_cpu_dirty(vaddr, size) runs for each region part before its pages may turn
	// CPU-dirty and writable, and what it returns is held until they have (a hold, so that an
	// upload taking the note sees the change whole). It runs before the region lock is taken.
	template <typename Flush, typename CpuDirty>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush,
	                      CpuDirty&& on_cpu_dirty) noexcept {
		(void)Invalidate<false>(vaddr, size, 0, on_flush, on_cpu_dirty);
	}
	// The run-ahead candidates of a write fault, in bytes: up_bytes from the next page on,
	// down_bytes ending at the faulting page.
	struct AheadCandidates {
		uint64_t up_bytes   = 0;
		uint64_t down_bytes = 0;
	};
	// A guest write fault at vaddr: InvalidateRegion over its byte that also notes the page in the
	// write history. Returns the pages run-ahead may make CPU-dirty (InvalidateAhead) on either
	// side: when CPU-dirty pages run directly below the page, the window above it, and when they
	// run directly above it, the window below it. Each is as long as its run, AHEAD_MIN_PAGES to
	// AHEAD_MAX_PAGES, cut to its eligible prefix walking away from the page
	// (RegionManager::AheadEligible); otherwise 0.
	template <typename Flush, typename CpuDirty>
	[[nodiscard]] AheadCandidates InvalidateFault(uint64_t vaddr, uint64_t history_epoch,
	                                              Flush&& on_flush, CpuDirty&& on_cpu_dirty) noexcept {
		return Invalidate<true>(vaddr, 1, history_epoch, on_flush, on_cpu_dirty);
	}
	// Run-ahead after InvalidateFault: makes the eligible prefix of [vaddr, vaddr + size), walked
	// in direction (from its last page for Down) and checked again under the lock since the state
	// may have moved, CPU-dirty and writable. It never flushes: the prefix ends before a GPU-dirty
	// page. on_cpu_dirty is as for InvalidateRegion. The range is whole pages within one region.
	// Returns the bytes made CPU-dirty, from vaddr on (Up) or ending at vaddr + size (Down).
	template <typename CpuDirty>
	uint64_t InvalidateAhead(uint64_t vaddr, uint64_t size, AheadDirection direction,
	                         uint64_t history_epoch, CpuDirty&& on_cpu_dirty) noexcept {
		static_assert(std::is_invocable_v<CpuDirty&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		ValidateRange(vaddr, size);
		if (size == 0 || vaddr % TRACKER_PAGE_SIZE != 0 || size % TRACKER_PAGE_SIZE != 0 ||
		    vaddr / TRACKER_REGION_SIZE != (vaddr + size - 1) / TRACKER_REGION_SIZE) {
			EXIT("invalid run-ahead range\n");
		}
		auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return 0;
		}
		const auto pages = size / TRACKER_PAGE_SIZE;
		const bool down  = direction == AheadDirection::Down;
		// Released after the lock, once the pages are CPU-dirty, as in InvalidateRegion. It notes
		// the whole range pending even when the check below marks less: an upload of the pending
		// set copies only CPU-dirty pages.
		[[maybe_unused]] const auto hold = on_cpu_dirty(vaddr, size);
		std::unique_lock            lock(manager->lock);
		// A downward window is walked from its last page.
		const auto top      = down ? vaddr + size - TRACKER_PAGE_SIZE : vaddr;
		const auto eligible = manager->AheadEligible(top, pages, history_epoch, direction);
		if (eligible == 0) {
			return 0;
		}
		const auto bytes = eligible * TRACKER_PAGE_SIZE;
		const auto first = down ? vaddr + size - bytes : vaddr;
		manager->ChangeState<DirtySource::Cpu, true>(first, bytes);
		lock.unlock();
		// As InvalidateRegion: queued after the change.
		QueueBdaSync(first, bytes);
		return bytes;
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
	}

	// Uploads the CPU-dirty pages of a range; a written range becomes GPU-dirty. The range
	// functor sees each CPU-dirty page run, then upload_func runs while the regions stay locked.
	// Neither runs when the range provably has nothing to upload and nothing to mark.
	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		ValidateRange(vaddr, size);
		// Read before walking: a transition during the walk must invalidate what it records.
		const SyncEpochs epochs {m_epochs.cpu_dirtied.load(std::memory_order_acquire),
		                         m_epochs.gpu_cleaned.load(std::memory_order_acquire)};
		if (IsSyncRedundant(vaddr, size, is_written, epochs)) {
			return;
		}
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		if (!is_written) {
			CreateRegions(vaddr, size);
			// Only regions that may hold CPU-dirty pages; a write racing the unlocked check
			// lands after this sync.
			ForEachMarkedRegion(m_cpu_any_words.get(), vaddr, size,
			                    [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				                    if (!manager->MayHaveCpuDirty()) {
					                    return;
				                    }
				                    std::scoped_lock lock(manager->lock);
				                    if (manager->IsModified<DirtySource::Cpu>(offset, bytes)) {
					                    manager->ForEachModifiedRange<DirtySource::Cpu, true>(
					                        manager->GetCpuAddr() + offset, bytes, range_func);
				                    }
			                    });
			upload_func();
			s_upload_owner = previous_upload_owner;
			NoteSynced(vaddr, size, is_written, epochs);
			return;
		}
		auto& locked = s_locked_regions;
		locked.clear();
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			// A GPU-dirty page is never CPU-dirty: nothing to upload, nothing to mark.
			if (bytes == TRACKER_REGION_SIZE && manager->IsGpuModifiedThroughout()) {
				return;
			}
			manager->lock.lock();
			if (manager->IsGpuModifiedThroughout(offset, bytes)) {
				manager->lock.unlock();
				return;
			}
			if (manager->IsModified<DirtySource::Cpu>(offset, bytes)) {
				manager->ForEachModifiedRange<DirtySource::Cpu, true>(
				    manager->GetCpuAddr() + offset, bytes, range_func);
			}
			locked.emplace_back(manager, offset, bytes);
		});
		upload_func();
		for (const auto& [manager, offset, bytes]: locked) {
			manager->template ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes);
			manager->lock.unlock();
		}
		locked.clear();
		s_upload_owner = previous_upload_owner;
		NoteSynced(vaddr, size, is_written, epochs);
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static constexpr size_t REGION_WORDS = REGION_COUNT / 64u;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;
	// Regions a written sync holds locked across its upload.
	inline static thread_local std::vector<std::tuple<RegionManager*, uint64_t, uint64_t>>
	    s_locked_regions;

	struct SyncEpochs {
		uint64_t cpu_dirtied = 0;
		uint64_t gpu_cleaned = 0;
	};
	// A range synchronized at some epochs: clean of CPU writes while cpu_dirtied is unchanged,
	// and, if it was written, GPU-dirty throughout while gpu_cleaned is unchanged. Values are
	// stored plus one so that zero marks an unused or read-only field.
	struct SyncRecord {
		uint64_t vaddr       = 0;
		uint64_t size        = 0;
		uint64_t cpu_dirtied = 0;
		uint64_t gpu_cleaned = 0;
	};
	// The same ranges are bound draw after draw. Two ways per set, so the stream of one-off
	// small ranges does not keep evicting the large ones that are costly to walk.
	static constexpr size_t SYNC_RECORD_SETS = 512;
	using SyncRecordSet                      = std::array<SyncRecord, 2>;

	[[nodiscard]] static size_t SyncRecordSetIndex(uint64_t vaddr, uint64_t size) noexcept {
		const auto hash = (vaddr ^ (size * 0x9e3779b97f4a7c15ull)) * 0xff51afd7ed558ccdull;
		return static_cast<size_t>(hash >> 55u) % SYNC_RECORD_SETS;
	}
	[[nodiscard]] bool IsSyncRedundant(uint64_t vaddr, uint64_t size, bool is_written,
	                                   const SyncEpochs& epochs) noexcept;
	void NoteSynced(uint64_t vaddr, uint64_t size, bool is_written,
	                const SyncEpochs& epochs) noexcept;

	// InvalidateRegion; with note_history, InvalidateFault for a range within one page.
	template <bool note_history, typename Flush, typename CpuDirty>
	AheadCandidates Invalidate(uint64_t vaddr, uint64_t size, uint64_t history_epoch,
	                           Flush& on_flush, CpuDirty& on_cpu_dirty) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		static_assert(std::is_invocable_v<CpuDirty&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();

		AheadCandidates ahead;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				[[maybe_unused]] const auto hold =
				    on_cpu_dirty(manager->GetCpuAddr() + offset, bytes);
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				const auto address = manager->GetCpuAddr() + offset;
				manager->ChangeState<DirtySource::Cpu, true>(address, bytes);
				if constexpr (note_history) {
					manager->NoteCpuWritten(address, history_epoch);
					ahead = FindAheadCandidates(*manager, address, history_epoch);
				}
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
		QueueBdaSync(vaddr, size);
		return ahead;
	}
	// Requires the region's lock. The run-ahead candidates around the page holding vaddr.
	static AheadCandidates FindAheadCandidates(RegionManager& manager, uint64_t vaddr,
	                                           uint64_t history_epoch) {
		const auto      page = vaddr - vaddr % TRACKER_PAGE_SIZE;
		AheadCandidates ahead;
		const auto      below = manager.CpuDirtyRunBelow(vaddr, AHEAD_MAX_PAGES);
		if (below != 0 && page + TRACKER_PAGE_SIZE != manager.GetCpuAddr() + TRACKER_REGION_SIZE) {
			const auto window = std::clamp(below, AHEAD_MIN_PAGES, AHEAD_MAX_PAGES);
			ahead.up_bytes    = manager.AheadEligible(page + TRACKER_PAGE_SIZE, window,
			                                          history_epoch, AheadDirection::Up) *
			                 TRACKER_PAGE_SIZE;
		}
		const auto above = manager.CpuDirtyRunAbove(vaddr, AHEAD_MAX_PAGES);
		if (above != 0 && page != manager.GetCpuAddr()) {
			const auto window = std::clamp(above, AHEAD_MIN_PAGES, AHEAD_MAX_PAGES);
			ahead.down_bytes  = manager.AheadEligible(page - TRACKER_PAGE_SIZE, window,
			                                          history_epoch, AheadDirection::Down) *
			                   TRACKER_PAGE_SIZE;
		}
		return ahead;
	}

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	// Calls func(index, mask) for each word of a per-region bitset that covers the range, with
	// mask selecting the range's regions in that word.
	template <typename Func>
	static void ForEachRegionWord(uint64_t vaddr, uint64_t size, Func&& func) {
		if (size == 0) {
			return;
		}
		const auto first = vaddr / TRACKER_REGION_SIZE;
		const auto last  = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto word = first / 64u; word <= last / 64u; word++) {
			const auto low  = word == first / 64u ? first % 64u : 0u;
			const auto high = word == last / 64u ? last % 64u : 63u;
			const auto mask = (~uint64_t {0} >> (63u - high)) & (~uint64_t {0} << low);
			func(word, mask);
		}
	}
	void CreateRegions(uint64_t vaddr, uint64_t size) {
		ForEachRegionWord(vaddr, size, [&](uint64_t word, uint64_t mask) {
			for (auto missing = ~m_region_words[word].load(std::memory_order_acquire) & mask;
			     missing != 0; missing &= missing - 1) {
				(void)GetOrCreateRegion(word * 64u +
				                        static_cast<uint64_t>(std::countr_zero(missing)));
			}
		});
	}
	// Like Iterate<false>, over the regions whose bit is set in words.
	template <typename Func>
	void ForEachMarkedRegion(const std::atomic<uint64_t>* words, uint64_t vaddr, uint64_t size,
	                         Func&& func) {
		const auto end = vaddr + size;
		ForEachRegionWord(vaddr, size, [&](uint64_t word, uint64_t mask) {
			for (auto marked = words[word].load(std::memory_order_acquire) & mask; marked != 0;
			     marked &= marked - 1) {
				const auto index   = word * 64u + static_cast<uint64_t>(std::countr_zero(marked));
				auto*      manager = m_regions[index].load(std::memory_order_acquire);
				if (manager == nullptr) {
					continue;
				}
				const auto region = index * TRACKER_REGION_SIZE;
				const auto begin  = std::max(vaddr, region);
				const auto stop   = std::min(end, region + TRACKER_REGION_SIZE);
				func(manager, begin - region, stop - begin);
			}
		});
	}

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	// Per region, one bit each: the region exists; it may hold CPU-dirty pages.
	std::unique_ptr<std::atomic<uint64_t>[]>       m_region_words;
	std::unique_ptr<std::atomic<uint64_t>[]>       m_cpu_any_words;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	std::mutex                                     m_bda_sync_mutex;
	RangeSet                                       m_bda_sync;
	// Bumped under the mutex as a take swaps the set out. A hold counts itself under the mutex,
	// in the slot of the generation its range was queued into, so the take that gets the range
	// waits for it (and for older holds still in that slot, which only waits longer).
	std::atomic<uint64_t>                m_bda_sync_generation {0};
	std::array<std::atomic<uint32_t>, 2> m_bda_sync_in_flight {};
	// Stored under the mutex, so a take with nothing queued is one load.
	std::atomic<bool> m_bda_sync_nonempty {false};
	PageManager&                                   m_page_manager;
	TrackerEpochs                                  m_epochs;
	// Uploads may come from any thread; the records belong to whichever holds this.
	std::atomic_flag                               m_sync_records_lock = ATOMIC_FLAG_INIT;
	std::array<SyncRecordSet, SYNC_RECORD_SETS>    m_sync_records {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
