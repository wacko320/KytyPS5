#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDRAWARGS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDRAWARGS_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/tickSlotRing.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;
struct DrawInstanceCapture;

// Where vkCmdDraw*Indirect finds a draw's arguments.
struct IndirectDrawCommand {
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
};

// Copies guest DRAW_INDIRECT / DRAW_INDEX_INDIRECT arguments into slots for vkCmdDraw*Indirect
// with a GPU pass, so the arguments may be produced by earlier GPU work and the CPU never reads
// them: an indexed draw's index count is clamped to INDEX_BUFFER_SIZE there, as the CPU path
// clamps it. A slot can also keep the draw's instance count for a later direct draw that reads
// the NUM_INSTANCES state the indirect draw leaves (ReadInstances): the slots are in device-local
// memory the host can read when the device has it (CanCapture).
class IndirectDrawArgs {
public:
	static constexpr uint32_t DefaultSlots = 4096;
	// Dwords of a slot: VkDrawIndexedIndirectCommand, the larger command.
	static constexpr uint32_t SlotDwords = 8;

	// `slots`: a power of two (tests use small rings).
	IndirectDrawArgs(GraphicContext& graphics, CommandScheduler& scheduler,
	                 uint32_t slots = DefaultSlots);
	~IndirectDrawArgs();
	KYTY_CLASS_NO_COPY(IndirectDrawArgs);

	// Whether the host can read converted arguments (instance captures).
	[[nodiscard]] bool CanCapture() const noexcept { return !m_slots.Mapped().empty(); }

	// Records the copy of the `dwords` argument dwords at buffer + offset into a slot, the first
	// clamped to `index_limit` (0: not clamped) and, with `drop_offsets`, the vertex and instance
	// offsets zeroed (the last two dwords), and a barrier making the slot visible to
	// indirect draws, into the current command buffer, whose rendering instance has ended. The
	// writes that produced the arguments are ordered before by the guest's synchronisation, as
	// for indirect dispatches. `capture` (CanCapture only): the slot also keeps the instance count
	// (dword 1) for ReadInstances.
	[[nodiscard]] IndirectDrawCommand Convert(vk::Buffer buffer, uint64_t offset, uint32_t dwords,
	                                          uint32_t index_limit, bool drop_offsets,
	                                          DrawInstanceCapture* capture);
	// The draw reading `command` is recorded now: a slot stays in use until the open tick
	// completes, and a read of guest memory is ordered before later shader writes
	// (CommandScheduler::OrderIndirectArgsReads).
	void NoteRead(const IndirectDrawCommand& command);
	// The instance count a capture kept, once the GPU wrote it (waits for its tick).
	[[nodiscard]] uint32_t ReadInstances(const DrawInstanceCapture& capture);

	// Lifetime totals (tests): waits for a slot segment's earlier tick, conversions, and
	// instance counts read.
	[[nodiscard]] uint64_t RingWaitCount() const noexcept { return m_ring_waits; }
	[[nodiscard]] uint64_t ConversionCount() const noexcept { return m_ring.Taken(); }
	[[nodiscard]] uint64_t InstanceReadCount() const noexcept { return m_instance_reads; }

private:
	static constexpr uint32_t SlotBytes = SlotDwords * sizeof(uint32_t);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	Buffer                  m_slots;
	TickSlotRing            m_ring;
	uint64_t                m_ring_waits      = 0;
	uint64_t                m_instance_reads  = 0;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTDRAWARGS_H_ */
