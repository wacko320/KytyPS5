#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/frameProfiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/srtPageWatch.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessHeap.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <chrono>
#include <memory>
#include <span>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	BindlessHeap&       GetBindlessHeap() { return m_bindless_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// Any thread: a read of these bytes faulted on a page protected because the GPU wrote to it,
	// but none of them is GPU-written, so guest memory already holds their value.
	[[nodiscard]] bool CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
	                                     uint64_t size) noexcept;
	// GPU thread only (false elsewhere): no byte of the range is GPU-written, under a GPU-written
	// image, or waiting for a download or other deferred write to publish it.
	[[nodiscard]] bool IsGpuSettled(uint64_t vaddr, uint64_t size);
	// GPU thread only (false elsewhere): buffer bytes the GPU wrote and the CPU has not read
	// back, with no GPU-written image over them.
	[[nodiscard]] bool IsGpuWrittenUnread(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// GPU thread: see BufferCache::TryWriteGpuOrdered. False when the caller must write guest
	// memory itself (after PrepareGpuThreadWrite).
	bool TryWriteGpuOrdered(uint64_t vaddr, const void* data, uint64_t size);
	// Before the GPU thread stores to guest memory directly: invalidates the range if a write
	// would fault, so it does not take a fault per page.
	void PrepareGpuThreadWrite(uint64_t vaddr, uint64_t size);
	// GPU thread: the guest backing's view of the range when a store to it can bypass page
	// protection (BufferCache::WriteThroughView), images over it taking it as a CPU write. A
	// store through the guest view would fault on a watched page, then the upload after it would
	// watch the page again: once per store, for labels written over and over.
	[[nodiscard]] uint8_t* BeginWriteThrough(uint64_t vaddr, uint64_t size);
	// GPU thread: a command-processor store to guest memory, made by writer(dst), where dst is
	// where the range's bytes are to be stored. Every such store goes through here: the buffers
	// over the range take it in command order, so GPU work recorded after it reads it without a
	// per-binding upload (CoherenceEpoch).
	template <typename Writer>
	void GpuThreadGuestWrite(uint64_t vaddr, uint64_t size, Writer&& writer) {
		Common::FrameProfiler::Rec(Common::FrameProfiler::Event::GpuGuestWrite, vaddr, size);
		// Uploads recorded before the store take the bytes before it.
		m_buffer_cache.WaitUploadsFrom(vaddr, size);
		if (auto* through = BeginWriteThrough(vaddr, size); through != nullptr) {
			std::forward<Writer>(writer)(static_cast<void*>(through));
			SrtPageWatch::Guest().Unsettle(vaddr, size);
			m_buffer_cache.WriteThrough(vaddr, through, size);
			return;
		}
		PrepareGpuThreadWrite(vaddr, size);
		std::forward<Writer>(writer)(reinterpret_cast<void*>(vaddr));
		SrtPageWatch::Guest().Unsettle(vaddr, size);
		m_buffer_cache.Epoch().SyncRangeNow(vaddr, size);
	}
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// Calls func(begin, end) for each mapped sub-range of [vaddr, vaddr + size).
	template <typename Func>
	void ForEachMappedRange(uint64_t vaddr, uint64_t size, Func&& func) const {
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_mapped_ranges.ForEachInRange(vaddr, size, func);
	}
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	// Gives the pages of host-computed scalar DMA addresses cache buffers before a dispatch or
	// draw, so the BDA table maps them and first reads of fresh guest memory do not return zero.
	void               PrefetchBda(std::span<const uint64_t> addresses);
	void               PrepareBda();
	// A shader that reports through the fault buffer (the loop watchdog) was recorded.
	void               RequestFaultProcessing() { m_fault_process_pending = true; }
	void               RunGarbageCollector();
	void               ProfileLevels();

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	BindlessHeap              m_bindless_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;
	std::chrono::steady_clock::time_point m_levels_sampled {};

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
