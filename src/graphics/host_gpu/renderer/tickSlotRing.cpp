#include "graphics/host_gpu/renderer/tickSlotRing.h"

#include "common/assert.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

TickSlotRing::TickSlotRing(uint32_t slots)
    : m_slots(slots), m_segment_slots(std::min(SegmentSlots, slots / 4u)) {
	EXIT_IF(slots < 4 || (slots & (slots - 1u)) != 0);
	m_ticks.resize(slots / m_segment_slots, 0);
}

uint32_t TickSlotRing::Next(CommandScheduler& scheduler, uint64_t& waits,
                            Common::FrameProfiler::Counter  counter,
                            Common::FrameProfiler::WaitSite site) {
	const auto slot = m_next;
	m_next          = (m_next + 1u) & (m_slots - 1u);
	m_taken++;
	auto& tick = m_ticks[slot / m_segment_slots];
	// Each use of a segment notes the open tick, so its first slot finds the last one of the
	// previous lap. Rings are sized to hold several frames' slots: expected never to wait.
	if (slot % m_segment_slots == 0 && tick != 0 && !scheduler.IsFree(tick)) {
		waits++;
		Common::FrameProfiler::Add(counter);
		scheduler.Wait(tick, site);
	}
	tick = scheduler.CurrentTick();
	return slot;
}

void TickSlotRing::Touch(uint32_t slot, uint64_t tick) {
	auto& last = m_ticks[slot / m_segment_slots];
	last       = std::max(last, tick);
}

} // namespace Libs::Graphics
