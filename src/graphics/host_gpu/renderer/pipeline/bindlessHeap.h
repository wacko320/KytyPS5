#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/BindlessImages.h"

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class StreamBuffer;

// Sampled images whose T# only the GPU knows (see BindlessImages.h). Shaders record the T#s
// they miss; the heap validates them after the submission and serves them from the next
// dispatch on, so a new texture reads as the null image for one dispatch.
class BindlessHeap {
public:
	using ImageClass = ShaderRecompiler::IR::Bindless::ImageClass;
	using Descriptor = std::array<uint32_t, 8>;

	struct Entry {
		Descriptor descriptor {};
		// A sampler entry has class SamplerClass or CompareSamplerClass.
		ImageClass image_class = ImageClass::Dim2D;
		uint32_t   slot        = 0; // in the class's array; slot 0 is the null image
	};

	BindlessHeap(GraphicContext& graphics, CommandScheduler& scheduler);
	~BindlessHeap();
	KYTY_CLASS_NO_COPY(BindlessHeap);

	[[nodiscard]] vk::DescriptorSetLayout Layout() const noexcept { return m_layout; }
	[[nodiscard]] std::span<const Entry>  Entries() const noexcept { return m_entries; }

	// Picks the current command buffer's miss region (may wait for an earlier command buffer);
	// called while a dispatch or draw is prepared, before its commands are recorded.
	void Reserve();
	// Writes and binds set 1 for one dispatch or draw. images[class] holds the class's views in
	// slot order, starting with the null image; classes the shaders do not use are empty.
	// samplers and compare_samplers likewise start with a default sampler, or are empty.
	void Commit(vk::CommandBuffer command, vk::PipelineLayout layout,
	            vk::PipelineBindPoint bind_point, StreamBuffer& upload,
	            const std::array<std::vector<vk::DescriptorImageInfo>,
	                             ShaderRecompiler::IR::Bindless::ClassCount>& images,
	            std::span<const vk::DescriptorImageInfo>                        samplers,
	            std::span<const vk::DescriptorImageInfo> compare_samplers);
	// Makes the misses visible to the host once the command buffer completes.
	void Finish();

private:
	struct PendingPool {
		vk::DescriptorPool pool;
		uint64_t           tick;
	};

	[[nodiscard]] vk::DescriptorSet AllocateSet();
	[[nodiscard]] std::vector<uint32_t> BuildTable() const;
	void AddMisses(std::span<const uint32_t> records);
	void AddMiss(const Descriptor& descriptor, uint32_t image_class);

	GraphicContext&                  m_graphics;
	CommandScheduler&                m_scheduler;
	vk::DescriptorSetLayout          m_layout       = nullptr;
	vk::DescriptorPool               m_current_pool = nullptr;
	std::deque<PendingPool>          m_pending_pools;
	Buffer                           m_misses;
	std::vector<uint64_t>            m_miss_ticks;
	uint32_t                         m_next_miss_region = 0;
	uint64_t                         m_last_miss_offset = 0;
	uint64_t                         m_region_tick      = 0; // the command buffer the region serves
	std::vector<Entry>               m_entries;
	std::map<std::pair<Descriptor, uint32_t>, uint32_t> m_known;
	std::set<std::pair<Descriptor, uint32_t>>           m_rejected;
	std::array<uint32_t, ShaderRecompiler::IR::Bindless::ClassCount + 2u> m_class_counts {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BINDLESSHEAP_H_
