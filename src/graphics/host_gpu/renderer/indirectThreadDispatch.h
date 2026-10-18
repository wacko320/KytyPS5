#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// Turns guest indirect dispatch arguments given in threads (USE_THREAD_DIMENSIONS) into
// workgroup counts with a GPU pass, so the arguments may be produced by earlier GPU work. Each
// slot also keeps the x, y, z thread counts, which the dispatched shader reads by device address
// to disable the lanes of partial workgroups.
class IndirectThreadDispatch {
public:
	IndirectThreadDispatch(GraphicContext& graphics, CommandScheduler& scheduler);
	~IndirectThreadDispatch();
	KYTY_CLASS_NO_COPY(IndirectThreadDispatch);

	struct Arguments {
		vk::Buffer        buffer          = nullptr;
		uint64_t          offset          = 0;
		vk::DeviceAddress threads_address = 0;
	};

	// Records the conversion of the thread counts at threads_buffer + threads_offset and returns
	// where vkCmdDispatchIndirect finds the workgroup counts and the shader finds the thread
	// counts.
	[[nodiscard]] Arguments Convert(vk::CommandBuffer command, vk::Buffer threads_buffer,
	                                uint64_t                       threads_offset,
	                                const std::array<uint32_t, 3>& group_size);

private:
	static constexpr uint32_t Slots             = 4096;
	static constexpr uint32_t SlotBytes         = 32;
	static constexpr uint32_t ThreadsSlotOffset = 16;

	uint32_t NextSlot(vk::CommandBuffer command);

	GraphicContext&         m_graphics;
	Buffer                  m_groups;
	uint32_t                m_next_slot       = 0;
	vk::DescriptorSetLayout m_desc_layout     = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_INDIRECTTHREADDISPATCH_H_ */
