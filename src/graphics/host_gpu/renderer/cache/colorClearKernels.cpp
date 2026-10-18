#include "graphics/host_gpu/renderer/cache/colorClearKernels.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/color_clear_fill_spv.h"
#include "gpu_tiler_shaders/color_clear_spv.h"
#include "gpu_tiler_shaders/color_clear_store_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

enum CheckMode : uint32_t { ModeCheck = 0, ModeFinalize = 1, ModeExpand = 2 };

struct CheckConstants {
	uint32_t                mode;
	uint32_t                first_word;
	uint32_t                slice_words;
	uint32_t                slice_count;
	uint32_t                code_count;
	uint32_t                mask_index;
	uint32_t                predicate_index;
	uint32_t                pad;
	std::array<uint32_t, 2> codes;
	std::array<uint32_t, 2> groups;
};

struct StoreConstants {
	std::array<float, 4>    color;
	std::array<uint32_t, 2> extent;
};

struct FillConstants {
	uint32_t                first_word;
	uint32_t                word_count;
	std::array<uint32_t, 2> pattern;
};

constexpr uint32_t GroupSize = 256;
// Each invocation of a check or expansion handles at least this many dwords of its slice.
constexpr uint32_t WordsPerInvocation = 8;
constexpr uint32_t MaxGroupsPerSlice  = 64;

[[nodiscard]] uint64_t MaskBytes(uint32_t slices) {
	return Common::AlignUp(static_cast<uint64_t>(slices) * sizeof(uint32_t), 16);
}

[[nodiscard]] uint32_t SliceGroups(uint64_t slice_bytes) {
	const auto words = slice_bytes / sizeof(uint32_t);
	const auto per_group = static_cast<uint64_t>(GroupSize) * WordsPerInvocation;
	return static_cast<uint32_t>(std::clamp<uint64_t>((words + per_group - 1) / per_group, 1,
	                                                  MaxGroupsPerSlice));
}

vk::Pipeline CreatePipeline(GraphicContext& graphics, std::span<const uint32_t> code,
                            vk::PipelineLayout layout, const char* name) {
	const auto module = CompileSPV(code, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = layout;
	vk::Pipeline pipeline = nullptr;
	const auto   result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create color-clear pipeline");
	SetVulkanObjectNameF(graphics.device, pipeline, "{}", name);
	return pipeline;
}

void CreateLayouts(GraphicContext& graphics, std::span<const vk::DescriptorSetLayoutBinding> bindings,
                   uint32_t push_bytes, vk::DescriptorSetLayout& set_layout,
                   vk::PipelineLayout& pipeline_layout) {
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                               &set_layout),
	                     "create color-clear descriptor layout");
	const vk::PushConstantRange   push_range {vk::ShaderStageFlagBits::eCompute, 0, push_bytes};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout),
	    "create color-clear pipeline layout");
}

} // namespace

ColorClearKernels::ColorClearKernels(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_results(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eIndirectBuffer |
                    (graphics.conditional_rendering_enabled
                         ? vk::BufferUsageFlagBits::eConditionalRenderingEXT
                         : vk::BufferUsageFlags {}),
                ResultsBytes) {
	SetVulkanObjectNameF(m_graphics.device, m_results.Handle(), "Color Clear Predicates");
	const vk::DescriptorSetLayoutBinding check_bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	CreateLayouts(m_graphics, check_bindings, sizeof(CheckConstants), m_check_layout,
	              m_check_pipeline_layout);
	m_check = CreatePipeline(m_graphics, COLOR_CLEAR_SPV, m_check_pipeline_layout,
	                         "Color Clear Check");
	const vk::DescriptorSetLayoutBinding store_bindings[] {
	    {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	CreateLayouts(m_graphics, store_bindings, sizeof(StoreConstants), m_store_layout,
	              m_store_pipeline_layout);
	m_store = CreatePipeline(m_graphics, COLOR_CLEAR_STORE_SPV, m_store_pipeline_layout,
	                         "Color Clear Store");
	const vk::DescriptorSetLayoutBinding fill_bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	CreateLayouts(m_graphics, fill_bindings, sizeof(FillConstants), m_fill_layout,
	              m_fill_pipeline_layout);
	m_fill = CreatePipeline(m_graphics, COLOR_CLEAR_FILL_SPV, m_fill_pipeline_layout,
	                        "Color Clear Fill");
}

ColorClearKernels::~ColorClearKernels() {
	m_graphics.device.destroyPipeline(m_fill, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fill_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fill_layout, nullptr);
	m_graphics.device.destroyPipeline(m_store, nullptr);
	m_graphics.device.destroyPipelineLayout(m_store_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_store_layout, nullptr);
	m_graphics.device.destroyPipeline(m_check, nullptr);
	m_graphics.device.destroyPipelineLayout(m_check_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_check_layout, nullptr);
}

bool ColorClearKernels::Fits(uint32_t slices, uint32_t codes) noexcept {
	return codes != 0 && codes <= MaxCodes && slices != 0 &&
	       MaskBytes(slices) + static_cast<uint64_t>(slices) * codes * RecordBytes <= ResultsBytes;
}

void ColorClearKernels::BindCheck(vk::CommandBuffer command, vk::Buffer metadata, uint64_t window,
                                  uint64_t range) {
	const vk::DescriptorBufferInfo infos[] {
	    {metadata, window, range},
	    {m_results.Handle(), 0, m_results.Size()},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_check);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_check_pipeline_layout, 0,
	                             writes);
}

ColorClearKernels::Check ColorClearKernels::RecordCheck(vk::CommandBuffer command,
                                                        vk::Buffer metadata, uint64_t offset,
                                                        uint64_t slice_bytes, uint32_t slices,
                                                        std::span<const uint8_t> codes,
                                                        vk::Extent2D store_groups) {
	const auto code_count = static_cast<uint32_t>(codes.size());
	EXIT_IF(!Fits(slices, code_count) || offset % sizeof(uint32_t) != 0 ||
	        slice_bytes % sizeof(uint32_t) != 0 || slice_bytes == 0);
	const auto bytes = MaskBytes(slices) + static_cast<uint64_t>(slices) * code_count * RecordBytes;
	if (m_next + bytes > ResultsBytes) {
		m_next = 0;
	}
	Check check {.results        = m_results.Handle(),
	             .mask_offset    = m_next,
	             .records_offset = m_next + MaskBytes(slices),
	             .codes          = code_count,
	             .slices         = slices};
	m_next += bytes;

	// The metadata's writers, and the readers of the ring's earlier records (predicates,
	// indirect counts and expansions), come first.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead |
	                       vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer |
	                            vk::PipelineStageFlagBits::eComputeShader,
	                        {}, 1, &before, 0, nullptr, 0, nullptr);
	command.fillBuffer(m_results.Handle(), check.mask_offset, MaskBytes(slices), 0);
	vk::MemoryBarrier cleared {};
	cleared.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	cleared.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &cleared, 0, nullptr,
	                        0, nullptr);

	// A window around the slices, not the whole buffer: a cache buffer can be larger than
	// maxStorageBufferRange.
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto window = offset / alignment * alignment;
	BindCheck(command, metadata, window, offset - window + slice_bytes * slices);
	CheckConstants constants {};
	constants.mode            = ModeCheck;
	constants.first_word      = static_cast<uint32_t>((offset - window) / sizeof(uint32_t));
	constants.slice_words     = static_cast<uint32_t>(slice_bytes / sizeof(uint32_t));
	constants.slice_count     = slices;
	constants.code_count      = code_count;
	constants.mask_index      = static_cast<uint32_t>(check.mask_offset / sizeof(uint32_t));
	constants.predicate_index = static_cast<uint32_t>(check.records_offset / sizeof(uint32_t));
	for (uint32_t k = 0; k < code_count; k++) {
		constants.codes[k / 4] |= static_cast<uint32_t>(codes[k]) << (8 * (k % 4));
	}
	constants.groups = {store_groups.width, store_groups.height};
	command.pushConstants(m_check_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch(SliceGroups(slice_bytes), slices, 1);

	vk::MemoryBarrier checked {};
	checked.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	checked.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &checked, 0, nullptr,
	                        0, nullptr);
	constants.mode = ModeFinalize;
	command.pushConstants(m_check_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch((slices * code_count + GroupSize - 1) / GroupSize, 1, 1);

	vk::MemoryBarrier finalized {};
	finalized.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	finalized.dstAccessMask = vk::AccessFlagBits::eShaderRead |
	                          vk::AccessFlagBits::eIndirectCommandRead;
	auto stages = vk::PipelineStageFlagBits::eComputeShader |
	              vk::PipelineStageFlagBits::eDrawIndirect;
	if (m_graphics.conditional_rendering_enabled) {
		finalized.dstAccessMask |= vk::AccessFlagBits::eConditionalRenderingReadEXT;
		stages |= vk::PipelineStageFlagBits::eConditionalRenderingEXT;
	}
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, stages, {}, 1, &finalized,
	                        0, nullptr, 0, nullptr);
	return check;
}

void ColorClearKernels::RecordExpand(vk::CommandBuffer command, const Check& check,
                                     vk::Buffer metadata, uint64_t offset, uint64_t slice_bytes) {
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto window = offset / alignment * alignment;
	BindCheck(command, metadata, window, offset - window + slice_bytes * check.slices);
	CheckConstants constants {};
	constants.mode            = ModeExpand;
	constants.first_word      = static_cast<uint32_t>((offset - window) / sizeof(uint32_t));
	constants.slice_words     = static_cast<uint32_t>(slice_bytes / sizeof(uint32_t));
	constants.slice_count     = check.slices;
	constants.code_count      = check.codes;
	constants.mask_index      = static_cast<uint32_t>(check.mask_offset / sizeof(uint32_t));
	constants.predicate_index = static_cast<uint32_t>(check.records_offset / sizeof(uint32_t));
	command.pushConstants(m_check_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatch(SliceGroups(slice_bytes), check.slices, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                        nullptr);
}

void ColorClearKernels::RecordStoreClear(vk::CommandBuffer command, vk::ImageView view,
                                         vk::Extent2D extent, const std::array<float, 4>& color,
                                         const Check& check, uint32_t slice, uint32_t code) {
	const vk::DescriptorImageInfo image_info {nullptr, view, vk::ImageLayout::eGeneral};
	vk::WriteDescriptorSet        write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageImage;
	write.pImageInfo      = &image_info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_store);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_store_pipeline_layout, 0, 1,
	                             &write);
	const StoreConstants constants {color, {extent.width, extent.height}};
	command.pushConstants(m_store_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatchIndirect(check.results, check.RecordOffset(slice, code));
}

void ColorClearKernels::RecordBufferFill(vk::CommandBuffer command, vk::Buffer buffer,
                                         uint64_t offset, uint64_t slice_bytes,
                                         const std::array<uint32_t, 2>& pattern,
                                         const Check& check, uint32_t slice, uint32_t code) {
	EXIT_IF(offset % sizeof(uint32_t) != 0 || slice_bytes % sizeof(uint32_t) != 0 ||
	        slice_bytes / sizeof(uint32_t) > UINT32_MAX);
	// The buffer's earlier writers come first.
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0, nullptr,
	                        0, nullptr);
	const auto alignment = std::max<uint64_t>(
	    m_graphics.physical_device_properties.limits.minStorageBufferOffsetAlignment, 4);
	const auto window = offset / alignment * alignment;
	const vk::DescriptorBufferInfo info {buffer, window, offset - window + slice_bytes};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fill);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_fill_pipeline_layout, 0, 1,
	                             &write);
	const FillConstants constants {static_cast<uint32_t>((offset - window) / sizeof(uint32_t)),
	                               static_cast<uint32_t>(slice_bytes / sizeof(uint32_t)), pattern};
	command.pushConstants(m_fill_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(constants), &constants);
	command.dispatchIndirect(check.results, check.RecordOffset(slice, code));
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                        nullptr);
}

} // namespace Libs::Graphics
