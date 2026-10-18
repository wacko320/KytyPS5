#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_TICKSLOTRING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_TICKSLOTRING_H_

#include "common/frameProfiler.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;

// The slots of a ring of GPU-written records, in segments that remember the tick that last used
// them: a slot is handed out again only once that tick has completed (a wrap within the open
// command buffer submits it and waits), so no barrier orders a slot's next writer after its
// earlier readers.
class TickSlotRing {
public:
	// The most slots of a segment; a ring has at least four segments.
	static constexpr uint32_t SegmentSlots = 1024;

	// `slots`: a power of two, at least 4.
	explicit TickSlotRing(uint32_t slots);

	// The next slot, after waiting for its segment's earlier tick (counted in `waits` and the
	// profiler's `counter`, the wait at `site`).
	[[nodiscard]] uint32_t Next(CommandScheduler& scheduler, uint64_t& waits,
	                            Common::FrameProfiler::Counter  counter,
	                            Common::FrameProfiler::WaitSite site);
	// The slot is read by a command recorded at `tick`.
	void Touch(uint32_t slot, uint64_t tick);
	// The most slots taken at once without one coming round to a segment taken before.
	[[nodiscard]] uint32_t Capacity() const noexcept { return m_slots - m_segment_slots; }
	[[nodiscard]] uint32_t Slots() const noexcept { return m_slots; }
	// Slots handed out so far.
	[[nodiscard]] uint64_t Taken() const noexcept { return m_taken; }

private:
	std::vector<uint64_t> m_ticks;
	uint32_t              m_slots;
	uint32_t              m_segment_slots;
	uint32_t              m_next  = 0;
	uint64_t              m_taken = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_TICKSLOTRING_H_ */
