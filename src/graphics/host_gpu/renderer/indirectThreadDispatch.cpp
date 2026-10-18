#include "graphics/host_gpu/renderer/indirectThreadDispatch.h"

#include "common/assert.h"
#include "common/frameProfiler.h"
#include "gpu_tiler_shaders/dispatch_threads_to_groups_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

struct PushConstants {
	std::array<uint32_t, 3> group_size;
	uint32_t                threads_index;
	uint32_t                groups_index;
};

} // namespace

IndirectThreadDispatch::IndirectThreadDispatch(GraphicContext&   graphics,
                                               CommandScheduler& scheduler, uint32_t slots)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_groups(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
               AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
               static_cast<uint64_t>(slots) * SlotBytes),
      m_ring(slots) {
	SetVulkanObjectNameF(m_graphics.device, m_groups.Handle(), "Indirect Thread Dispatch Groups");

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
	    "create indirect-thread-dispatch descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(PushConstants)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                            &m_pipeline_layout),
	                     "create indirect-thread-dispatch pipeline layout");

	const auto module = CompileSPV(DISPATCH_THREADS_TO_GROUPS_SPV, m_graphics.device);
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
	RequireVulkanSuccess(result, "create indirect-thread-dispatch pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Indirect Thread Dispatch");
}

IndirectThreadDispatch::~IndirectThreadDispatch() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_desc_layout, nullptr);
}

uint32_t IndirectThreadDispatch::NextSlot() {
	return m_ring.Next(m_scheduler, m_ring_waits,
	                   Common::FrameProfiler::Counter::ThreadCountRingWaits,
	                   Common::FrameProfiler::WaitSite::ThreadCountRingReuse);
}

void IndirectThreadDispatch::NoteRead(vk::DeviceAddress threads_address) {
	const auto groups = m_groups.BufferDeviceAddress();
	EXIT_IF(threads_address < groups + ThreadsSlotOffset ||
	        threads_address >= groups + m_groups.Size());
	const auto offset = threads_address - groups - ThreadsSlotOffset;
	m_ring.Touch(static_cast<uint32_t>(offset / SlotBytes), m_scheduler.CurrentTick());
}

void IndirectThreadDispatch::RecordConversion(vk::CommandBuffer command, const Source& source,
                                              uint32_t slot) {
	EXIT_IF((source.offset & 3u) != 0);
	// A window around the three counts, not the whole buffer: a cache buffer can be larger than
	// maxStorageBufferRange.
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto window         = source.offset / alignment * alignment;
	const auto threads_offset = source.offset - window;
	const vk::DescriptorBufferInfo infos[] {
	    {source.buffer, window, threads_offset + 3 * sizeof(uint32_t)},
	    {m_groups.Handle(), 0, m_groups.Size()},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const PushConstants constants {source.group_size, static_cast<uint32_t>(threads_offset / 4u),
	                               slot * (SlotBytes / 4u)};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch(1, 1, 1);
}

IndirectThreadDispatch::Arguments IndirectThreadDispatch::SlotArguments(uint32_t slot) const {
	const auto offset = static_cast<uint64_t>(slot) * SlotBytes;
	return {m_groups.Handle(), offset, m_groups.BufferDeviceAddress() + offset + ThreadsSlotOffset};
}

namespace {

// The conversions' writes before the indirect dispatches and the shaders reading the counts.
void ConvertedBarrier(vk::CommandBuffer command) {
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eDrawIndirect |
	                            vk::PipelineStageFlagBits::eComputeShader,
	                        {}, 1, &after, 0, nullptr, 0, nullptr);
}

} // namespace

IndirectThreadDispatch::Arguments IndirectThreadDispatch::Convert(const Source& source) {
	const auto slot    = NextSlot();
	const auto command = m_scheduler.Current().Handle();
	// The thread counts may come from earlier shader or transfer writes.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                            vk::PipelineStageFlagBits::eComputeShader |
	                            vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                        0, nullptr);
	RecordConversion(command, source, slot);
	ConvertedBarrier(command);
	return SlotArguments(slot);
}

void IndirectThreadDispatch::ConvertBatch(std::span<const Source> sources, bool order_before,
                                          std::span<Arguments> converted) {
	EXIT_IF(sources.empty() || converted.size() != sources.size() ||
	        sources.size() > BatchCapacity());
	// All slots first: a wait for a segment submits the open command buffer. Within the
	// capacity no slot comes round to a segment this batch took before.
	m_batch_slots.clear();
	for (size_t i = 0; i < sources.size(); i++) {
		m_batch_slots.push_back(NextSlot());
	}
	// The conversions are recorded at the tick that is open now, also into the slots taken
	// before a wait moved it on (an entry no dispatch takes is not noted again).
	const auto tick = m_scheduler.CurrentTick();
	for (const auto slot: m_batch_slots) {
		m_ring.Touch(slot, tick);
	}
	const auto command = m_scheduler.Current().Handle();
	if (order_before) {
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
		before.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0,
		                        nullptr, 0, nullptr);
	}
	for (size_t i = 0; i < sources.size(); i++) {
		RecordConversion(command, sources[i], m_batch_slots[i]);
		converted[i] = SlotArguments(m_batch_slots[i]);
	}
	ConvertedBarrier(command);
}

} // namespace Libs::Graphics
