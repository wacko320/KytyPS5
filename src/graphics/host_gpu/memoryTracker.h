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
	[[nodiscard]] RangeSet TakeBdaSync();
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
		QueueBdaSync(vaddr, size);
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
	PageManager&                                   m_page_manager;
	TrackerEpochs                                  m_epochs;
	// Uploads may come from any thread; the records belong to whichever holds this.
	std::atomic_flag                               m_sync_records_lock = ATOMIC_FLAG_INIT;
	std::array<SyncRecordSet, SYNC_RECORD_SETS>    m_sync_records {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
