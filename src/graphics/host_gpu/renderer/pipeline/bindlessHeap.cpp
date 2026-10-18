#include "graphics/host_gpu/renderer/pipeline/bindlessHeap.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <cinttypes>
#include <cstring>

namespace Libs::Graphics {

namespace {

namespace Bindless = ShaderRecompiler::IR::Bindless;

constexpr uint32_t PoolSets    = 32;
constexpr uint32_t MissRegions = 64;
constexpr uint64_t MissBytes   = (1u + Bindless::MaxMisses * Bindless::MissDwords) * sizeof(uint32_t);
constexpr uint64_t MissStride  = (MissBytes + 255u) & ~uint64_t {255u};

// Why a T# cannot be a heap entry of the class, or nullptr when it can.
const char* RejectReason(const BindlessHeap::Descriptor& words, Bindless::ImageClass image_class) {
	ShaderTextureResource descriptor;
	std::memcpy(descriptor.fields, words.data(), sizeof(descriptor.fields));
	if (descriptor.IsNull()) {
		return "null";
	}
	const auto type = descriptor.Type();
	bool       type_ok = false;
	switch (image_class) {
		case Bindless::ImageClass::Dim1D:
		case Bindless::ImageClass::Dim1DArray:
			type_ok = type == Prospero::ImageType::kColor1D ||
			          type == Prospero::ImageType::kColor1DArray;
			break;
		case Bindless::ImageClass::Dim2D:
		case Bindless::ImageClass::Dim2DArray:
		case Bindless::ImageClass::Compare2D:
		case Bindless::ImageClass::Compare2DArray:
			type_ok = type == Prospero::ImageType::kColor2D ||
			          type == Prospero::ImageType::kColor2DArray;
			break;
		case Bindless::ImageClass::Dim3D: type_ok = type == Prospero::ImageType::kColor3D; break;
		default: break;
	}
	if (!type_ok) {
		return "image type does not match the instruction";
	}
	const auto format = descriptor.Format();
	if (Prospero::SampledTextureNumericClass(format) != Prospero::TextureNumericClass::Float) {
		return "format does not sample as float";
	}
	if (Prospero::IsFmaskTextureFormat(format) ||
	    TextureGetSurfaceFormatInfo(format).conversion_format != Prospero::BufferFormat::kInvalid) {
		return "format needs shader conversion";
	}
	return nullptr;
}

} // namespace

BindlessHeap::BindlessHeap(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_misses(graphics, scheduler, MemoryUsage::Download, 0, AllFlags, MissStride * MissRegions),
      m_miss_ticks(MissRegions, 0) {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	std::vector<vk::DescriptorBindingFlags>     flags;
	for (const auto binding: {Bindless::TableBinding, Bindless::MissBinding}) {
		bindings.push_back({binding, vk::DescriptorType::eStorageBuffer, 1,
		                    vk::ShaderStageFlagBits::eAll, nullptr});
		flags.emplace_back();
	}
	for (uint32_t image_class = 0; image_class < Bindless::ClassCount; image_class++) {
		bindings.push_back({Bindless::FirstImageBinding + image_class,
		                    vk::DescriptorType::eSampledImage, Bindless::HeapSlots,
		                    vk::ShaderStageFlagBits::eAll, nullptr});
		flags.emplace_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
	}
	for (const auto binding: {Bindless::SamplerBinding, Bindless::CompareSamplerBinding}) {
		bindings.push_back({binding, vk::DescriptorType::eSampler, Bindless::HeapSlots,
		                    vk::ShaderStageFlagBits::eAll, nullptr});
		flags.emplace_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
	}
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(flags.size());
	flags_info.pBindingFlags = flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout),
	                     "create bindless heap descriptor layout");
	m_graphics.bindless_heap_layout = m_layout;
}

BindlessHeap::~BindlessHeap() {
	for (const auto& pending: m_pending_pools) {
		m_scheduler.Wait(pending.tick, Common::FrameProfiler::WaitSite::BindlessHeapTeardown);
		m_graphics.device.destroyDescriptorPool(pending.pool, nullptr);
	}
	if (m_current_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	}
	m_graphics.bindless_heap_layout = nullptr;
	m_graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

vk::DescriptorSet BindlessHeap::AllocateSet() {
	const auto TryAllocate = [&](vk::DescriptorSet& set) {
		if (m_current_pool == nullptr) {
			return false;
		}
		vk::DescriptorSetAllocateInfo allocate {};
		allocate.descriptorPool     = m_current_pool;
		allocate.descriptorSetCount = 1;
		allocate.pSetLayouts        = &m_layout;
		return m_graphics.device.allocateDescriptorSets(&allocate, &set) == vk::Result::eSuccess;
	};
	vk::DescriptorSet set = nullptr;
	if (TryAllocate(set)) {
		return set;
	}
	if (m_current_pool != nullptr) {
		m_pending_pools.push_back({m_current_pool, m_scheduler.CurrentTick()});
		m_current_pool = nullptr;
	}
	if (!m_pending_pools.empty() && m_scheduler.IsFree(m_pending_pools.front().tick)) {
		m_current_pool = m_pending_pools.front().pool;
		m_pending_pools.pop_front();
		RequireVulkanSuccess(m_graphics.device.resetDescriptorPool(m_current_pool, {}),
		                     "reset bindless heap descriptor pool");
	} else {
		const std::array sizes {
		    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 2 * PoolSets},
		    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage,
		                            PoolSets * Bindless::ClassCount * Bindless::HeapSlots},
		    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 2 * PoolSets * Bindless::HeapSlots},
		};
		vk::DescriptorPoolCreateInfo create {};
		create.maxSets       = PoolSets;
		create.poolSizeCount = static_cast<uint32_t>(sizes.size());
		create.pPoolSizes    = sizes.data();
		RequireVulkanSuccess(
		    m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool),
		    "create bindless heap descriptor pool");
	}
	EXIT_IF(!TryAllocate(set));
	return set;
}

std::vector<uint32_t> BindlessHeap::BuildTable() const {
	uint32_t capacity = 8;
	while (capacity < m_entries.size() * 2u) {
		capacity *= 2u;
	}
	for (;;) {
		std::vector<uint32_t> table(1u + capacity * Bindless::EntryDwords, 0u);
		table[0]      = capacity;
		bool overflow = false;
		for (const auto& entry: m_entries) {
			const auto image_class = static_cast<uint32_t>(entry.image_class);
			const auto hash        = Bindless::Hash(entry.descriptor, image_class);
			bool       placed      = false;
			for (uint32_t probe = 0; probe < Bindless::MaxProbes; probe++) {
				const auto base = 1u + ((hash + probe) & (capacity - 1u)) * Bindless::EntryDwords;
				if (table[base + 10u] != 0u) {
					// Lookups stop at the first matching hash, so a key whose hash equals an
					// earlier entry's stays a miss.
					if (table[base] == hash) {
						placed = true;
						break;
					}
					continue;
				}
				table[base] = hash;
				std::copy(entry.descriptor.begin(), entry.descriptor.end(), table.begin() + base + 1u);
				table[base + 9u]  = image_class;
				table[base + 10u] = entry.slot + 1u;
				placed            = true;
				break;
			}
			if (!placed) {
				overflow = true;
				break;
			}
		}
		if (!overflow) {
			return table;
		}
		capacity *= 2u;
	}
}

void BindlessHeap::Reserve() {
	const auto tick = m_scheduler.CurrentTick();
	if (m_region_tick == tick) {
		return;
	}
	// One miss region per command buffer: its dispatches and draws append to the same records
	// (the count is atomic), which the host reads once the command buffer completes. A region
	// still pending belongs to an earlier, submitted command buffer.
	const auto region  = m_next_miss_region;
	m_next_miss_region = (m_next_miss_region + 1u) % MissRegions;
	if (m_miss_ticks[region] != 0) {
		m_scheduler.Wait(m_miss_ticks[region], Common::FrameProfiler::WaitSite::BindlessMissRegion);
		m_scheduler.PopPendingOperations();
	}
	const auto miss_offset = region * MissStride;
	std::memset(m_misses.Mapped().data() + miss_offset, 0, sizeof(uint32_t));
	m_misses.Flush(miss_offset, sizeof(uint32_t));
	auto* records        = m_misses.Mapped().data() + miss_offset;
	m_miss_ticks[region] = tick;
	m_region_tick        = tick;
	m_last_miss_offset   = miss_offset;
	m_scheduler.DeferOperation([this, records, miss_offset, region] {
		m_misses.Invalidate(miss_offset, MissBytes);
		const auto* words = reinterpret_cast<const uint32_t*>(records);
		const auto  count = std::min(words[0], Bindless::MaxMisses);
		AddMisses({words + 1, count * Bindless::MissDwords});
		m_miss_ticks[region] = 0;
	});
}

void BindlessHeap::Commit(
    vk::CommandBuffer command, vk::PipelineLayout layout, vk::PipelineBindPoint bind_point,
    StreamBuffer& upload,
    const std::array<std::vector<vk::DescriptorImageInfo>, Bindless::ClassCount>& images,
    std::span<const vk::DescriptorImageInfo> samplers,
    std::span<const vk::DescriptorImageInfo> compare_samplers) {
	// Normally reserved while the draw or dispatch was prepared (that may wait); a command
	// buffer restarted since then reserves here.
	Reserve();
	const auto set   = AllocateSet();
	const auto table = BuildTable();
	const auto table_offset =
	    upload.Copy(table.data(), table.size() * sizeof(uint32_t), m_graphics.StorageMinAlignment());

	const auto miss_offset = m_last_miss_offset;

	const vk::DescriptorBufferInfo table_info {upload.Handle(), table_offset,
	                                           table.size() * sizeof(uint32_t)};
	const vk::DescriptorBufferInfo miss_info {m_misses.Handle(), miss_offset, MissBytes};
	std::vector<vk::WriteDescriptorSet> writes;
	for (const auto& [binding, info]:
	     {std::pair {Bindless::TableBinding, &table_info},
	      std::pair {Bindless::MissBinding, &miss_info}}) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = set;
		write.dstBinding      = binding;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eStorageBuffer;
		write.pBufferInfo     = info;
		writes.push_back(write);
	}
	for (uint32_t image_class = 0; image_class < Bindless::ClassCount; image_class++) {
		if (images[image_class].empty()) {
			continue;
		}
		EXIT_IF(images[image_class].size() > Bindless::HeapSlots);
		vk::WriteDescriptorSet write {};
		write.dstSet          = set;
		write.dstBinding      = Bindless::FirstImageBinding + image_class;
		write.descriptorCount = static_cast<uint32_t>(images[image_class].size());
		write.descriptorType  = vk::DescriptorType::eSampledImage;
		write.pImageInfo      = images[image_class].data();
		writes.push_back(write);
	}
	for (const auto& [binding, list]:
	     {std::pair {Bindless::SamplerBinding, samplers},
	      std::pair {Bindless::CompareSamplerBinding, compare_samplers}}) {
		if (list.empty()) {
			continue;
		}
		EXIT_IF(list.size() > Bindless::HeapSlots);
		vk::WriteDescriptorSet write {};
		write.dstSet          = set;
		write.dstBinding      = binding;
		write.descriptorCount = static_cast<uint32_t>(list.size());
		write.descriptorType  = vk::DescriptorType::eSampler;
		write.pImageInfo      = list.data();
		writes.push_back(write);
	}
	m_graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
	                                       nullptr);
	command.bindDescriptorSets(bind_point, layout, Bindless::DescriptorSet, 1, &set, 0, nullptr);
}

void BindlessHeap::Finish() {
	// The miss records become host-visible at the end of the command buffer: a barrier here
	// would be invalid inside a draw's render pass.
	m_scheduler.RequestHostBarrier();
}

void BindlessHeap::AddMisses(std::span<const uint32_t> records) {
	for (size_t offset = 0; offset + Bindless::MissDwords <= records.size();
	     offset += Bindless::MissDwords) {
		Descriptor descriptor {};
		std::copy_n(records.begin() + static_cast<ptrdiff_t>(offset), descriptor.size(),
		            descriptor.begin());
		AddMiss(descriptor, records[offset + 8u]);
	}
}

void BindlessHeap::AddMiss(const Descriptor& descriptor, uint32_t image_class) {
	const auto key = std::pair {descriptor, image_class};
	if (image_class > Bindless::CompareSamplerClass || m_known.contains(key) || m_rejected.contains(key)) {
		return;
	}
	const auto  type   = static_cast<ImageClass>(image_class);
	const char* reason = nullptr;
	if (image_class == Bindless::SamplerClass || image_class == Bindless::CompareSamplerClass) {
		reason = std::any_of(descriptor.begin() + 4, descriptor.end(), [](uint32_t word) {
			         return word != 0u;
		         })
		             ? "sampler key is not zero-padded"
		             : nullptr;
	} else {
		reason = RejectReason(descriptor, type);
	}
	if (reason != nullptr) {
		m_rejected.insert(key);
		LOGF("Bindless heap: rejected T# %08x %08x %08x %08x %08x %08x %08x %08x class %u: %s\n",
		     descriptor[0], descriptor[1], descriptor[2], descriptor[3], descriptor[4],
		     descriptor[5], descriptor[6], descriptor[7], image_class, reason);
		return;
	}
	if (m_class_counts[image_class] + 1u >= Bindless::HeapSlots) {
		m_rejected.insert(key);
		LOGF("Bindless heap: class %u is full\n", image_class);
		return;
	}
	const auto slot = ++m_class_counts[image_class];
	m_known.emplace(key, static_cast<uint32_t>(m_entries.size()));
	m_entries.push_back({descriptor, type, slot});
	if (m_entries.size() <= 256u || (m_entries.size() & (m_entries.size() - 1u)) == 0u) {
		LOGF("Bindless heap: %zu entries (class %u slot %u)\n", m_entries.size(), image_class,
		     slot);
	}
}

} // namespace Libs::Graphics
