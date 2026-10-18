#include "graphics/host_gpu/renderer/indirectDrawArgs.h"

#include "common/assert.h"
#include "common/frameProfiler.h"
#include "gpu_tiler_shaders/draw_args_convert_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace Libs::Graphics {

namespace {

struct PushConstants {
	uint32_t args_index;
	uint32_t converted_index;
	uint32_t dwords;
	uint32_t index_limit;
	uint32_t offsets_from;
};

} // namespace

IndirectDrawArgs::IndirectDrawArgs(GraphicContext& graphics, CommandScheduler& scheduler,
                                   uint32_t slots)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_slots(Buffer::HostReadable {}, graphics, scheduler, 0, AllFlags,
              static_cast<uint64_t>(slots) * SlotBytes),
      m_ring(slots) {
	SetVulkanObjectNameF(m_graphics.device, m_slots.Handle(), "Indirect Draw Arguments");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_desc_layout),
	    "create indirect-draw-arguments descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create indirect-draw-arguments pipeline layout");

	const auto module = CompileSPV(DRAW_ARGS_CONVERT_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create indirect-draw-arguments pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Indirect Draw Arguments");
}

IndirectDrawArgs::~IndirectDrawArgs() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
}

IndirectDrawCommand IndirectDrawArgs::Convert(vk::Buffer buffer, uint64_t offset, uint32_t dwords,
                                              uint32_t index_limit, bool drop_offsets,
                                              DrawInstanceCapture* capture) {
	EXIT_IF(buffer == nullptr || (offset & 3u) != 0 || dwords < 4 || dwords > SlotDwords ||
	        (capture != nullptr && !CanCapture()));
	const auto slot = m_ring.Next(m_scheduler, m_ring_waits,
	                              Common::FrameProfiler::Counter::DrawIndirectRingWaits,
	                              Common::FrameProfiler::WaitSite::DrawArgsRingReuse);
	// A window around the arguments, not the whole buffer: a cache buffer can be larger than
	// maxStorageBufferRange.
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto window      = offset / alignment * alignment;
	const auto args_offset = offset - window;
	const vk::DescriptorBufferInfo infos[] {
	    {buffer, window, args_offset + dwords * sizeof(uint32_t)},
	    {m_slots.Handle(), 0, m_slots.Size()},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const PushConstants constants {static_cast<uint32_t>(args_offset / 4u), slot * SlotDwords,
	                               dwords, index_limit,
	                               drop_offsets ? (dwords == 5 ? 3u : 2u) : 0u};
	const auto command = m_scheduler.Current().Handle();
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch(1, 1, 1);
	// The slot before the indirect draw reading it, and the read of the arguments before any
	// later shader write to them.
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &after, 0, nullptr, 0,
	                        nullptr);
	Common::FrameProfiler::Add(Common::FrameProfiler::Counter::DrawIndirectConversions);
	if (capture != nullptr) {
		// The host reads the slot once its tick completes.
		m_scheduler.RequestHostBarrier();
		*capture = {m_ring.Taken(), m_scheduler.CurrentTick(), slot};
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::DrawIndirectInstanceCaptures);
	}
	return {m_slots.Handle(), static_cast<uint64_t>(slot) * SlotBytes};
}

void IndirectDrawArgs::NoteRead(const IndirectDrawCommand& command) {
	if (command.buffer != m_slots.Handle()) {
		// The draw reads guest memory itself. A conversion's read is ordered before later work
		// by its barrier (the indirect stage is the first of both pipelines).
		m_scheduler.NoteIndirectArgsRead();
		return;
	}
	m_ring.Touch(static_cast<uint32_t>(command.offset / SlotBytes), m_scheduler.CurrentTick());
}

uint32_t IndirectDrawArgs::ReadInstances(const DrawInstanceCapture& capture) {
	// A capture is replaced by the next indirect draw of its processor long before a lap of the
	// ring could take its slot again.
	EXIT_IF(capture.tick == 0 || !CanCapture() || m_ring.Taken() - capture.taken >= m_ring.Slots());
	if (!m_scheduler.IsFree(capture.tick)) {
		m_scheduler.Wait(capture.tick, Common::FrameProfiler::WaitSite::DrawInstancesRead);
	}
	m_instance_reads++;
	uint32_t instances = 0;
	std::memcpy(&instances,
	            m_slots.Mapped().data() + static_cast<uint64_t>(capture.slot) * SlotBytes + 4u,
	            sizeof(instances));
	return instances;
}

} // namespace Libs::Graphics
