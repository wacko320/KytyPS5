#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

constexpr size_t BitmapBytes = BufferCache::CACHING_NUMPAGES / 8;
static_assert(BufferCache::BDA_WRITTEN_BITMAP_WORD * sizeof(uint32_t) == BitmapBytes);
static_assert(BufferCache::FAULT_BUFFER_SIZE == 2 * BitmapBytes);
// Each download area holds the list of faulting pages, then the list of pages untracked stores
// wrote; each list is a count, then the pages (fault_buffer_process.comp). A list that fills
// keeps the pages it has no room for in the bitmap, for the next pass.
constexpr size_t MaxPageFaults    = 1024;
constexpr size_t MaxWrittenPages  = 16384;
constexpr size_t FaultListBytes   = MaxPageFaults * sizeof(uint64_t);
constexpr size_t WrittenListBytes = MaxWrittenPages * sizeof(uint64_t);
constexpr size_t PageFaultAreaSize = FaultListBytes + WrittenListBytes;

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BufferCache::FAULT_BUFFER_SIZE),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * PageFaultAreaSize) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

void FaultManager::ProcessFaultBuffer() {
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		m_scheduler.Wait(wait_tick, Common::FrameProfiler::WaitSite::FaultBufferReuse);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	// Only the counts: the pass writes the entries it counts.
	std::memset(mapped, 0, sizeof(uint64_t));
	std::memset(mapped + FaultListBytes, 0, sizeof(uint64_t));
	m_download_buffer.Flush(offset, PageFaultAreaSize);
	// The written pages are reported only after a command that may write them, or after a pass
	// that had no room for all of them.
	const bool written_pass = m_written_pass_due;
	m_written_pass_due      = false;

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, BitmapBytes},
	    {m_download_buffer.Handle(), offset, FaultListBytes},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0, writes);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	if (written_pass) {
		// The same compaction of the written-page bitmap into its own list.
		const vk::DescriptorBufferInfo written_infos[] {
		    {m_fault_buffer.Handle(), BitmapBytes, BitmapBytes},
		    {m_download_buffer.Handle(), offset + FaultListBytes, WrittenListBytes},
		};
		for (uint32_t index = 0; index < writes.size(); ++index) {
			writes[index].pBufferInfo = &written_infos[index];
		}
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
		                             m_fault_process_pipeline_layout, 0, writes);
		command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	}
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	const auto area = m_current_area;
	auto report = written_pass ? m_buffer_cache.BeginUntrackedReport() : BufferCache::UntrackedReport {};
	m_scheduler.DeferOperation([this, mapped, offset, area, written_pass,
	                            report = std::move(report)]() mutable {
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		const auto  count =
		    std::min<uint64_t>(static_cast<uint32_t>(faults[0]), MaxPageFaults - 1u);
		for (uint64_t index = 1; index <= count; ++index) {
			// --shader-loop-limit reports tripped loops as faults on the never-mapped low pages
			// (word = 1 + shader hash % 1024, bit = loop header id % 32).
			// The first 32 pages hold no watchdog words: a fault there is a near-null guest pointer,
			// which has nothing to cache either.
			if (faults[index] < 32ull * (1 + 1024) * BufferCache::CACHING_PAGESIZE) {
				const auto page = faults[index] / BufferCache::CACHING_PAGESIZE;
				if (page < 32) {
					continue;
				}
				LOGF_COLOR(Log::Color::BrightRed,
				           "Shader loop watchdog tripped: shader hash %% 1024 = 0x%03" PRIx64
				           ", loop id %" PRIu64 " (mod 32)\n",
				           page / 32 - 1, page % 32);
				continue;
			}
			const auto address = BufferCache::GuestAddress(faults[index]);
			fault_ranges.Add(address, BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", address);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
		});
		if (written_pass) {
			const auto* written = std::bit_cast<const uint64_t*>(mapped + FaultListBytes);
			const auto  reported = static_cast<uint32_t>(written[0]);
			// A list that filled left the rest in the bitmap, for the next pass.
			const bool complete = reported < MaxWrittenPages;
			if (!complete) {
				m_written_pass_due = true;
			}
			RangeSet written_ranges;
			for (uint64_t index = 1; index <= std::min<uint64_t>(reported, MaxWrittenPages - 1u);
			     ++index) {
				written_ranges.Add(BufferCache::GuestAddress(written[index]),
				                   BufferCache::CACHING_PAGESIZE);
			}
			m_buffer_cache.NoteUntrackedWritten(written_ranges, std::move(report), complete);
		}
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

} // namespace Libs::Graphics
