#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/frameProfiler.h"

#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

namespace FrameProfiler = Common::FrameProfiler;

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions              = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
	m_region_words         = std::make_unique<std::atomic<uint64_t>[]>(REGION_WORDS);
	m_cpu_any_words        = std::make_unique<std::atomic<uint64_t>[]>(REGION_WORDS);
	m_epochs.cpu_any_words = m_cpu_any_words.get();
}

MemoryTracker::~MemoryTracker() = default;

bool MemoryTracker::IsSyncRedundant(uint64_t vaddr, uint64_t size, bool is_written,
                                    const SyncEpochs& epochs) noexcept {
	// Another thread consulting the records just means this sync walks the regions.
	if (m_sync_records_lock.test_and_set(std::memory_order_acquire)) {
		return false;
	}
	bool redundant = false;
	for (const auto& record: m_sync_records[SyncRecordSetIndex(vaddr, size)]) {
		// A range still GPU-dirty throughout holds no CPU-dirty page either.
		redundant |= record.vaddr == vaddr && record.size == size &&
		             (record.gpu_cleaned == epochs.gpu_cleaned + 1 ||
		              (!is_written && record.cpu_dirtied == epochs.cpu_dirtied + 1));
	}
	m_sync_records_lock.clear(std::memory_order_release);
	return redundant;
}

void MemoryTracker::NoteSynced(uint64_t vaddr, uint64_t size, bool is_written,
                               const SyncEpochs& epochs) noexcept {
	if (m_sync_records_lock.test_and_set(std::memory_order_acquire)) {
		return;
	}
	auto& set = m_sync_records[SyncRecordSetIndex(vaddr, size)];
	// The same range again, else the smaller of the two.
	auto* slot = &set[set[1].vaddr == vaddr && set[1].size == size ? 1 : 0];
	if (!(slot->vaddr == vaddr && slot->size == size) && set[1].size < set[0].size) {
		slot = &set[1];
	}
	*slot = {vaddr, size, epochs.cpu_dirtied + 1, is_written ? epochs.gpu_cleaned + 1 : 0};
	m_sync_records_lock.clear(std::memory_order_release);
}

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager =
	    std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE, &m_epochs);
	auto* ptr = manager.get();
	m_region_storage.push_back(std::move(manager));
	// A new region starts CPU-dirty throughout.
	const auto bit = uint64_t {1} << (index % 64u);
	m_cpu_any_words[index / 64u].fetch_or(bit, std::memory_order_release);
	m_regions[index].store(ptr, std::memory_order_release);
	m_region_words[index / 64u].fetch_or(bit, std::memory_order_release);
	m_epochs.cpu_dirtied.fetch_add(1, std::memory_order_acq_rel);
	return ptr;
}

void MemoryTracker::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
	m_bda_sync_nonempty.store(true, std::memory_order_release);
}

MemoryTracker::BdaSyncHold MemoryTracker::HoldBdaSync(uint64_t vaddr, uint64_t size) {
	const auto      begin      = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto      end        = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	const uint64_t  profile_t0 = FrameProfiler::Start();
	std::lock_guard lock(m_bda_sync_mutex);
	if (profile_t0 != 0) {
		FrameProfiler::AddSince(FrameProfiler::CurrentRole() == FrameProfiler::Role::Gpu
		                            ? FrameProfiler::Counter::PendingLockWaitGpuTime
		                            : FrameProfiler::Counter::PendingLockWaitGuestTime,
		                        profile_t0);
		FrameProfiler::Add(FrameProfiler::Counter::PendingHolds);
	}
	// Counted under the mutex, in the slot of the take that gets this range.
	auto& in_flight =
	    m_bda_sync_in_flight[m_bda_sync_generation.load(std::memory_order_relaxed) & 1];
	in_flight.fetch_add(1, std::memory_order_relaxed);
	BdaSyncHold hold(in_flight);
	m_bda_sync.Add(begin, end - begin);
	m_bda_sync_nonempty.store(true, std::memory_order_release);
	return hold;
}

RangeSet MemoryTracker::TakeBdaSync() {
	if (!m_bda_sync_nonempty.load(std::memory_order_acquire)) {
		return {};
	}
	RangeSet taken;
	uint64_t generation = 0;
	{
		std::lock_guard lock(m_bda_sync_mutex);
		taken = std::exchange(m_bda_sync, RangeSet {});
		m_bda_sync_nonempty.store(false, std::memory_order_relaxed);
		generation = m_bda_sync_generation.fetch_add(1, std::memory_order_relaxed);
	}
	// The holds queued into the taken set counted themselves in this generation's slot under
	// the mutex; they are short (a page-state change), so spin.
	const uint64_t wait_t0 = FrameProfiler::Start();
	for (uint32_t spins = 0;
	     m_bda_sync_in_flight[generation & 1].load(std::memory_order_acquire) != 0; spins++) {
		if (spins < 64) {
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
	}
	if (wait_t0 != 0) {
		const uint64_t ticks = FrameProfiler::Ticks() - wait_t0;
		FrameProfiler::Add(FrameProfiler::Counter::EpochUploadWaitTime, ticks);
		FrameProfiler::Add(FrameProfiler::Counter::EpochUploadWaits);
		FrameProfiler::Max(FrameProfiler::Counter::EpochUploadLongestWait, ticks);
	}
	return taken;
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		// As racy as the locked answer: a CPU write may land right after either.
		if (!manager->MayHaveCpuDirty()) {
			return false;
		}
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		if (!manager->MayHaveGpuDirty()) {
			return false;
		}
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
	QueueBdaSync(vaddr, size);
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UntrackMemory(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	std::vector<RegionManager*> managers;
	managers.reserve((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) /
	                 TRACKER_REGION_SIZE);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
		managers.push_back(manager);
	});

	std::vector<std::unique_lock<TrackingSpinLock>> locks;
	locks.reserve(managers.size());
	for (auto* manager: managers) {
		locks.emplace_back(manager->lock);
	}
	if (Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	    })) {
		EXIT("cannot untrack GPU-dirty memory\n");
	}
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
	locks.clear();
	QueueBdaSync(vaddr, size);
}

} // namespace Libs::Graphics
