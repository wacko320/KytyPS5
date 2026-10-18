#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/frameProfiler.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/indirectThreadDispatch.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

void RenderExecutor::UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
                                            const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// Reads and transforms of metadata do not establish a clear state.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				(void)cache.ClearMeta(descriptor.Base48());
			}
		}
	}
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.program->shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::BindSharedMemory(ShaderComputeInputInfo& input, PreparedBindings& bindings,
                                      uint64_t indirect_args) {
	if (input.stage.program->bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	// When the dispatch's size is unknown on the CPU, at least this much is bound.
	constexpr uint64_t UnsizedBytes = 32ull << 20u;
	auto&      cache = m_context.GetBufferCache();
	const auto range_limit =
	    m_context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	bool sized = true;
	if (indirect_args != 0) {
		// Arguments the GPU wrote would be read back (a drain), and thread dimensions are
		// converted on the GPU: neither is read here.
		if (input.dispatch_thread_dimensions ||
		    cache.IsRegionGpuModified(indirect_args, sizeof(vk::DispatchIndirectCommand))) {
			sized = false;
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::ComputeLdsIndirectUnsized);
		} else {
			cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
			std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
			            sizeof(input.workgroup_counts));
		}
	}
	uint64_t size = sized ? sizeof(uint32_t) : std::min<uint64_t>(UnsizedBytes, range_limit);
	if (sized && std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > range_limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > range_limit / count);
			size *= count;
		}
	}
	// LDS has no contents to preserve: a larger buffer replaces the old one, which work recorded
	// before may still use.
	if (m_lds_storage == nullptr || m_lds_storage->Size() < size) {
		const auto capacity = std::max<uint64_t>(
		    {size, m_lds_storage != nullptr ? m_lds_storage->Size() * 2u : 0u, 1ull << 20u});
		if (m_lds_storage != nullptr) {
			m_context.GetCommandScheduler().DeferOperation(
			    [old = std::move(m_lds_storage)]() mutable { old.reset(); });
		}
		m_lds_storage = std::make_unique<Buffer>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		    capacity);
		m_lds_unordered = false;
	}
	if (!sized) {
		size = std::min<uint64_t>(m_lds_storage->Size(), range_limit);
	}
	bindings.shared_memory = {m_lds_storage->Handle(), 0, size};
}

void RenderExecutor::OrderSharedMemory(CommandRecorder command, const PreparedBindings& bindings) {
	if (bindings.shared_memory.buffer == nullptr || !m_lds_unordered ||
	    m_context.GetCommandScheduler().GlobalBarrierCount() != m_lds_global_barriers) {
		return;
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &barrier, 0, nullptr,
	                        0, nullptr);
	m_lds_unordered = false;
	m_lds_barriers++;
	Common::FrameProfiler::Add(Common::FrameProfiler::Counter::ComputeLdsBarriers);
}

void RenderExecutor::NoteSharedMemoryDispatch(const PreparedBindings& bindings) {
	if (bindings.shared_memory.buffer == nullptr) {
		return;
	}
	m_lds_unordered       = true;
	m_lds_global_barriers = m_context.GetCommandScheduler().GlobalBarrierCount();
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute,
                                    ThreadDispatchBatch* batch) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	m_context.GetCommandScheduler().FeedIdleGpu();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	const auto compute_program =
	    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	// A clear that replaces the dispatch is ordered by the guest's synchronisation before it too.
	m_context.GetCommandScheduler().BeforeGuestDispatch();
	if (resources.specialization_reads.empty()) {
		// Cached metadata addresses can be reused as ordinary buffers; preserve their guest writes.
		UpdateComputeMetaClears(input_info, buffer);
		if (TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
		                          thread_group_z, mode)) {
			ResetBindings();
			return;
		}
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	if (program.info.uses_dma) {
		m_context.PrefetchBda(resources.dma_prefetch);
	}
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	} else if (program.info.loop_watchdog) {
		m_context.RequestFaultProcessing();
	}
	RebindImages(bindings);
	BindSharedMemory(input_info, bindings);
	// Acquiring the batch's arguments can merge cache buffers; finalize shader bindings afterward.
	if (batch != nullptr && !batch->requests.empty()) {
		RecordThreadBatch(*batch);
	}
	RebindBuffers(bindings);

	auto              vk_buffer        = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	CommitBindlessHeap(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	                   std::span {&descriptor_stage, 1u});
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	// Before the dispatch is recorded: a readback in between must not take it for unwritten.
	if (program.has_address_writes) {
		m_context.GetBufferCache().NoteUntrackedWrites();
	}
	// No barrier of its own: the guest's synchronisation orders dispatches (CommandScheduler::
	// RequestGuestBarrier), as the hardware overlaps the dispatches between its syncs.
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.BeforeGuestDispatch();
	if (HasShaderBufferWrites(input_info.stage)) {
		scheduler.OrderIndirectArgsReads(vk::PipelineStageFlagBits::eComputeShader);
	}
	OrderSharedMemory(vk_buffer, bindings);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	scheduler.CountGuestDispatch();
	NoteSharedMemoryDispatch(bindings);
	if (use_thread_dimensions) {
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::ComputeThreadDimensionDispatches);
	}
	if (bindings.uses_bindless) {
		m_context.GetBindlessHeap().Finish();
	}
	m_context.GetBufferCache().CommitWrittenRanges();
	ResetBindings();
}


void RenderExecutor::RecordThreadBatch(ThreadDispatchBatch& batch) {
	using Common::FrameProfiler::Counter;
	auto& cache     = m_context.GetBufferCache();
	auto& scheduler = m_context.GetCommandScheduler();
	// At a window's head the batch follows the guest barrier the dispatch takes; later in a
	// window it follows the compute writes recorded before it, as the conversion of a single
	// dispatch does. That one is defensive: arguments a dispatch of the same window wrote are a
	// guest race on the hardware.
	if (scheduler.GuestBarrierPending()) {
		scheduler.GlobalBarrier(CommandScheduler::GlobalBarrierSite::ThreadBatch);
	}
	const bool order_before = scheduler.UnorderedComputeReads();
	// Acquiring a range can create or join cache buffers, which may move the ones translated
	// before: translate all, then take the final buffers.
	m_thread_batch.clear();
	m_thread_ranges.clear();
	// Requests past what one batch's slots may hold are left to convert alone (a mismatch).
	const auto capacity = IndirectThreads().BatchCapacity();
	for (const auto& request: batch.requests) {
		if (m_thread_batch.size() == capacity) {
			break;
		}
		// A prediction: arguments the dispatch will not read may be anywhere.
		if (m_context.IsMapped(request.args_addr, sizeof(vk::DispatchIndirectCommand))) {
			(void)cache.TranslateRead(request.args_addr, sizeof(vk::DispatchIndirectCommand));
			m_thread_batch.push_back({request, {}});
			m_thread_ranges.emplace_back(request.args_addr, sizeof(vk::DispatchIndirectCommand));
		}
	}
	if (m_thread_batch.empty()) {
		return;
	}
	m_thread_sources.clear();
	for (const auto& entry: m_thread_batch) {
		const auto [args_buffer, args_offset] = cache.TranslateRead(
		    entry.request.args_addr, sizeof(vk::DispatchIndirectCommand));
		EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
		m_thread_sources.push_back({args_buffer->Handle(), args_offset, entry.request.group_size});
	}
	// From here on, a possible change to the arguments' bytes is noted against the batch.
	cache.WatchUnchanged(m_thread_ranges);
	m_thread_converted.resize(m_thread_sources.size());
	IndirectThreads().ConvertBatch(m_thread_sources, order_before, m_thread_converted);
	scheduler.NoteComputeReadBarrier();
	for (size_t i = 0; i < m_thread_batch.size(); i++) {
		m_thread_batch[i].converted = m_thread_converted[i];
	}
	m_thread_batch_next = 0;
	batch.serial        = ++m_thread_batch_serial;
	m_thread_stats.batches++;
	Common::FrameProfiler::Add(Counter::ComputeIndirectThreadBatches);
	Common::FrameProfiler::Add(Counter::ComputeIndirectThreadBatchEntries, m_thread_batch.size());
}

const IndirectThreadDispatchArguments* RenderExecutor::FindBatchedThreads(
    const ThreadDispatchBatch* batch, uint64_t args_addr, const std::array<uint32_t, 3>& group_size,
    uint64_t change_serial) {
	using Common::FrameProfiler::Counter;
	const auto fallback = [this](Counter reason, size_t stat) {
		m_thread_stats.fallbacks[stat]++;
		Common::FrameProfiler::Add(Counter::ComputeIndirectThreadFallback);
		Common::FrameProfiler::Add(reason);
		return nullptr;
	};
	if (batch == nullptr || batch->serial == 0 || batch->serial != m_thread_batch_serial) {
		return fallback(Counter::ComputeIndirectThreadFallbackNotReached, 0);
	}
	// Dispatches take the entries in order; a dispatch the scan did not predict leaves them.
	const auto matches = [&](const BatchedThreads& entry) {
		return entry.request.args_addr == args_addr && entry.request.group_size == group_size;
	};
	auto index = m_thread_batch_next;
	while (index < m_thread_batch.size() && !matches(m_thread_batch[index])) {
		index++;
	}
	if (index == m_thread_batch.size()) {
		return fallback(Counter::ComputeIndirectThreadFallbackMismatch, 1);
	}
	m_thread_batch_next = index + 1;
	// The conversion read the bytes when the batch was recorded. An upload or join since, or a
	// GPU write recorded before this dispatch began, may have changed them; this dispatch's own
	// written bindings run after the conversion, as on the hardware its packet reads them first.
	const auto changes = m_context.GetBufferCache().UnchangedRangeChanges(index);
	if (changes.other_write) {
		return fallback(Counter::ComputeIndirectThreadFallbackUpload, 2);
	}
	if (changes.first_gpu_write != 0 && changes.first_gpu_write <= change_serial) {
		return fallback(Counter::ComputeIndirectThreadFallbackWritten, 3);
	}
	m_thread_stats.batched++;
	Common::FrameProfiler::Add(Counter::ComputeIndirectThreadBatched);
	return &m_thread_batch[index].converted;
}

IndirectDrawArgs& RenderExecutor::IndirectDraws() {
	if (m_indirect_draws == nullptr) {
		m_indirect_draws = std::make_unique<IndirectDrawArgs>(m_context.GetGraphics(),
		                                                      m_context.GetCommandScheduler());
	}
	return *m_indirect_draws;
}

uint32_t RenderExecutor::ReadCapturedInstances(const DrawInstanceCapture& capture) {
	Common::LockGuard lock(m_context.GetMutex());
	Common::FrameProfiler::Add(Common::FrameProfiler::Counter::DrawIndirectLazyInstanceReads);
	return IndirectDraws().ReadInstances(capture);
}

IndirectThreadDispatch& RenderExecutor::IndirectThreads() {
	if (m_indirect_threads == nullptr) {
		m_indirect_threads = std::make_unique<IndirectThreadDispatch>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler());
	}
	return *m_indirect_threads;
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute,
                                      ThreadDispatchBatch* batch) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0);
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	m_context.GetCommandScheduler().PopPendingOperations();
	m_context.GetCommandScheduler().FeedIdleGpu();
	const auto change_serial = m_context.GetBufferCache().ChangeSerial();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	// Thread counts only the GPU may know: the shader reads them through a device address.
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.dispatch_thread_indirect   = use_thread_dimensions;
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	const auto& program = *input_info.stage.program;
	if (program.info.uses_dma) {
		m_context.PrefetchBda(input_info.stage.resources->dma_prefetch);
	}
	PreparedBindings* descriptor_stage = &bindings;
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	} else if (program.info.loop_watchdog) {
		m_context.RequestFaultProcessing();
	}
	BindSharedMemory(input_info, bindings, args_addr);
	RebindImages(bindings);
	// Thread-dimension arguments are converted on the GPU: earlier dispatches of this submission
	// may still be computing them. Record the conversion before binding this pipeline, in the
	// batch of this guest synchronisation window when one covers it. The batch acquires ranges
	// (which can join cache buffers) and may submit: before this dispatch's arguments are taken.
	if (batch != nullptr && !batch->requests.empty()) {
		RecordThreadBatch(*batch);
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	// Thread-dimension arguments are converted on the GPU: earlier dispatches of this submission
	// may still be computing them. Record the conversion before binding this pipeline.
	auto indirect_buffer = args_buffer->Handle();
	auto indirect_offset = static_cast<uint64_t>(args_offset);
	vk::DeviceAddress threads_address = 0;
	if (use_thread_dimensions) {
		const std::array<uint32_t, 3> group_size {cs_regs.cs_regs.num_thread_x,
		                                          cs_regs.cs_regs.num_thread_y,
		                                          cs_regs.cs_regs.num_thread_z};
		const auto* batched = FindBatchedThreads(batch, args_addr, group_size, change_serial);
		const auto  converted =
		    batched != nullptr
		        ? *batched
		        : IndirectThreads().Convert({indirect_buffer, indirect_offset, group_size});
		threads_address = converted.threads_address;
		indirect_buffer = converted.buffer;
		indirect_offset = converted.offset;
		if (const auto dword = program.bindings.dispatch_thread_dword;
		    dword != ShaderRecompiler::IR::PushData::NoStart) {
			bindings.shader_data[dword]      = static_cast<uint32_t>(converted.threads_address);
			bindings.shader_data[dword + 1u] = static_cast<uint32_t>(converted.threads_address >> 32u);
		}
	}
	RebindBuffers(bindings);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	CommitBindlessHeap(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	                   std::span {&descriptor_stage, 1u});
	const auto vk_buffer = buffer.Handle();
	// Before the dispatch is recorded: a readback in between must not take it for unwritten.
	if (program.has_address_writes) {
		m_context.GetBufferCache().NoteUntrackedWrites();
	}
	// No barrier of its own, for the arguments either: GPU work that produced them is ordered
	// by the guest's synchronisation before this packet (the barrier includes indirect reads),
	// uploads and joins by their own barriers, and conversions by theirs.
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.BeforeGuestDispatch();
	if (HasShaderBufferWrites(input_info.stage)) {
		scheduler.OrderIndirectArgsReads(vk::PipelineStageFlagBits::eComputeShader);
	}
	OrderSharedMemory(vk_buffer, bindings);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatchIndirect(indirect_buffer, indirect_offset);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	scheduler.CountGuestDispatch();
	NoteSharedMemoryDispatch(bindings);
	if (use_thread_dimensions) {
		// A batched slot was taken at the batch's tick; this dispatch may be in a later one.
		IndirectThreads().NoteRead(threads_address);
	}
	if (Common::FrameProfiler::Active()) {
		using Common::FrameProfiler::Counter;
		Common::FrameProfiler::Add(Counter::ComputeIndirectDispatches);
		if (use_thread_dimensions) {
			Common::FrameProfiler::Add(Counter::ComputeThreadDimensionDispatches);
		}
		if (m_context.GetBufferCache().HasGpuDirtyBytes(args_addr,
		                                                sizeof(vk::DispatchIndirectCommand))) {
			Common::FrameProfiler::Add(Counter::ComputeIndirectArgsGpuWritten);
		}
	}
	if (bindings.uses_bindless) {
		m_context.GetBindlessHeap().Finish();
	}
	m_context.GetBufferCache().CommitWrittenRanges();
	ResetBindings();
}

} // namespace Libs::Graphics
