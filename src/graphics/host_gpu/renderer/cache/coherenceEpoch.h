#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COHERENCEEPOCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COHERENCEEPOCH_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/memoryTracker.h"

namespace Libs::Graphics {

class BufferCache;

// CPU writes reach the cached buffers once per epoch instead of once per binding. An epoch begins
// at each command-buffer slice and after each WAIT_REG_MEM (or rewind) that passes. A CPU write the
// GPU may legally see was made before one of those, or by the command processor itself, which
// uploads its writes as it makes them (SyncRangeNow). A write made after the acquire and not
// followed by one races the GPU work on hardware too. Within an epoch a read binding is therefore a
// plain translation of the guest address.
//
// The pending set is the memory tracker's BDA sync queue (MemoryTracker::QueueBdaSync): an acquire
// takes all of it and uploads it in one batch, and PrepareBda uploads what was queued since
// (UploadPending).
class CoherenceEpoch {
public:
	using Hold = MemoryTracker::BdaSyncHold;

	explicit CoherenceEpoch(BufferCache& cache): m_cache(cache) {}
	KYTY_CLASS_NO_COPY(CoherenceEpoch);

	// Any thread: ranges whose buffers may lack CPU writes (write faults, invalidations, mappings).
	void NotePending(uint64_t vaddr, uint64_t size);
	// Any thread: NotePending, returning a hold (MemoryTracker::HoldBdaSync). Pages made writable
	// before the hold is dropped are seen by the upload that takes the note as pending and
	// CPU-dirty. A thread must not upload while it holds.
	[[nodiscard]] Hold HoldPending(uint64_t vaddr, uint64_t size);
	// GPU thread: begins an epoch. Every pending range goes up to the buffers over it in one batch.
	void Acquire();
	// GPU thread: uploads what is pending now, within the epoch, in one batch.
	void UploadPending();
	[[nodiscard]] bool HasPending() const noexcept;
	// GPU thread: uploads the range to the buffers over it now, after the GPU work recorded so far.
	void SyncRangeNow(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t EpochId() const noexcept { return m_epoch_id; }
	// Any thread: counts the takes of a non-empty pending set (MemoryTracker::BdaSyncGeneration).
	[[nodiscard]] uint64_t Generation() const noexcept;

private:
	BufferCache& m_cache;
	uint64_t     m_epoch_id = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COHERENCEEPOCH_H_
