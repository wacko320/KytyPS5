#include "graphics/host_gpu/renderer/cache/coherenceEpoch.h"

#include "common/frameProfiler.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include <atomic>

namespace Libs::Graphics {

namespace FrameProfiler = Common::FrameProfiler;

CoherenceEpoch::Hold CoherenceEpoch::HoldPending(uint64_t vaddr, uint64_t size) {
	return m_cache.m_memory_tracker.HoldBdaSync(vaddr, size);
}

void CoherenceEpoch::NotePending(uint64_t vaddr, uint64_t size) {
	m_cache.m_memory_tracker.QueueBdaSync(vaddr, size);
}

bool CoherenceEpoch::HasPending() const noexcept {
	return m_cache.m_memory_tracker.HasBdaSync();
}

uint64_t CoherenceEpoch::Generation() const noexcept {
	return m_cache.m_memory_tracker.BdaSyncGeneration();
}

void CoherenceEpoch::Acquire() {
	// After a WAIT_REG_MEM the caller just read a guest value that a guest thread stored after
	// noting its writes pending: that read must not move past the load below.
	std::atomic_thread_fence(std::memory_order_acquire);
	m_epoch_id++;
	if (FrameProfiler::Active()) {
		FrameProfiler::RecordEvent(FrameProfiler::Event::EpochAcquire, m_epoch_id, 0);
		FrameProfiler::Add(FrameProfiler::Counter::EpochAcquires);
	}
	UploadPending();
}

void CoherenceEpoch::UploadPending() {
	if (!HasPending()) {
		return;
	}
	// Waits for the holds of the set it takes.
	const auto pending = m_cache.m_memory_tracker.TakeBdaSync();
	m_cache.UploadCpuPending(pending);
}

void CoherenceEpoch::SyncRangeNow(uint64_t vaddr, uint64_t size) {
	m_cache.SynchronizeBuffersInRange(vaddr, size);
}

} // namespace Libs::Graphics
