#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/tickSlotRing.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// Where vkCmdDispatchIndirect finds a conversion's workgroup counts and the shader its thread
// counts.
struct IndirectThreadDispatchArguments {
	vk::Buffer        buffer          = nullptr;
	uint64_t          offset          = 0;
	vk::DeviceAddress threads_address = 0;
};

// Thread counts to convert, and the workgroup size.
struct IndirectThreadDispatchSource {
	vk::Buffer              buffer = nullptr;
	uint64_t                offset = 0;
	std::array<uint32_t, 3> group_size {};
};

// Turns guest indirect dispatch arguments given in threads (USE_THREAD_DIMENSIONS) into
// workgroup counts with a GPU pass, so the arguments may be produced by earlier GPU work. Each
// slot also keeps the x, y, z thread counts, which the dispatched shader reads by device address
// to disable the lanes of partial workgroups.
//
// The slots are a ring whose segments remember the tick that last used them: a slot is reused
// only once that tick has completed (a wrap within the open command buffer submits it and
// waits), so no barrier orders a slot's next writer after its earlier readers. Conversions are
// recorded one at a time (Convert, with its barriers) or as a batch for the indirect dispatches
// of one guest synchronisation window (ConvertBatch: one barrier after all).
class IndirectThreadDispatch {
public:
	static constexpr uint32_t DefaultSlots = 64 * 1024;

	// `slots` in the ring (a power of two; tests use small rings).
	IndirectThreadDispatch(GraphicContext& graphics, CommandScheduler& scheduler,
	                       uint32_t slots = DefaultSlots);
	~IndirectThreadDispatch();
	KYTY_CLASS_NO_COPY(IndirectThreadDispatch);

	using Arguments = IndirectThreadDispatchArguments;
	using Source    = IndirectThreadDispatchSource;

	// Records the conversion of the thread counts at source.buffer + source.offset into the
	// current command buffer, after every earlier shader and transfer write, and returns where
	// vkCmdDispatchIndirect finds the workgroup counts and the shader the thread counts.
	[[nodiscard]] Arguments Convert(const Source& source);

	// Records the conversions of all sources back to back, then one barrier making them visible
	// to indirect and shader reads; `order_before`: a barrier first orders them after the compute
	// writes recorded before. `converted` takes each source's arguments.
	void ConvertBatch(std::span<const Source> sources, bool order_before,
	                  std::span<Arguments> converted);

	// The dispatch reading the thread counts at threads_address (and its workgroup counts) is
	// recorded now: their slot stays in use until the open tick completes.
	void NoteRead(vk::DeviceAddress threads_address);

	// The most sources one ConvertBatch takes: the ring less one segment.
	[[nodiscard]] uint32_t BatchCapacity() const noexcept { return m_ring.Capacity(); }

	// Waits for a slot segment's earlier tick (tests).
	[[nodiscard]] uint64_t RingWaitCount() const noexcept { return m_ring_waits; }

private:
	static constexpr uint32_t SlotBytes         = 32;
	static constexpr uint32_t ThreadsSlotOffset = 16;

	// The next slot of the ring, after waiting for its segment's earlier tick.
	[[nodiscard]] uint32_t NextSlot();
	void RecordConversion(vk::CommandBuffer command, const Source& source, uint32_t slot);
	[[nodiscard]] Arguments SlotArguments(uint32_t slot) const;

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	Buffer                  m_groups;
	TickSlotRing            m_ring;
	uint64_t                m_ring_waits      = 0;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
	std::vector<uint32_t>   m_batch_slots;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_ */
