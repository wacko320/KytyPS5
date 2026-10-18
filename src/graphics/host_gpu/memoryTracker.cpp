#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

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
}

RangeSet MemoryTracker::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, RangeSet {});
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
