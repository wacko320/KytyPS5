#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/frameProfiler.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_bindless_heap(graphics, m_command_scheduler),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Buffers can outlive an unmap of some of their pages; synchronization skips those.
	m_buffer_cache.SetMappedParts(
	    [this](uint64_t vaddr, uint64_t size, BufferCache::MappedParts& out) {
		    if (IsMapped(vaddr, size)) {
			    return true;
		    }
		    out.clear();
		    ForEachMappedRange(vaddr, size,
		                       [&](uint64_t begin, uint64_t end) { out.emplace_back(begin, end); });
		    return false;
	    });
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish(Common::FrameProfiler::WaitSite::GpuShutdown);
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (Common::FrameProfiler::Active()) {
		using Common::FrameProfiler::Counter;
		Common::FrameProfiler::Add(access == PageFaultAccess::Write ? Counter::GuestWriteFaultsRender
		                                                            : Counter::GuestReadFaultsRender,
		                           Common::FrameProfiler::CurrentRole());
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		const Common::FrameProfiler::ScopedCause cause(
		    Common::FrameProfiler::ReadbackCause::FaultRead);
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::CanServeCleanRead(uint64_t fault_vaddr, uint64_t vaddr,
                                      uint64_t size) noexcept {
	return IsMapped(vaddr, size) && m_page_manager.IsReadWatched(fault_vaddr) &&
	       m_buffer_cache.IsCleanForConcurrentRead(vaddr, size) &&
	       !m_texture_cache.IsRegionGpuModified(vaddr, size);
}

void RenderContext::PrepareGpuThreadWrite(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t PageSize = 4096;
	for (auto page = vaddr & ~(PageSize - 1); page < vaddr + size; page += PageSize) {
		if (m_page_manager.IsWriteWatched(page)) {
			(void)InvalidateMemory(vaddr, size);
			return;
		}
	}
}

uint8_t* RenderContext::BeginWriteThrough(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size) || m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return nullptr;
	}
	auto* through = m_buffer_cache.WriteThroughView(vaddr, size);
	if (through == nullptr) {
		return nullptr;
	}
	// As PrepareGpuThreadWrite: an image over a watched page re-reads guest memory.
	constexpr uint64_t PageSize = 4096;
	for (auto page = vaddr & ~(PageSize - 1); page < vaddr + size; page += PageSize) {
		if (m_page_manager.IsWriteWatched(page)) {
			m_texture_cache.InvalidateMemory(vaddr, size);
			break;
		}
	}
	return through;
}

bool RenderContext::TryWriteGpuOrdered(uint64_t vaddr, const void* data, uint64_t size) {
	return IsMapped(vaddr, size) && !m_texture_cache.IsRegionGpuModified(vaddr, size) &&
	       m_buffer_cache.TryWriteGpuOrdered(vaddr, data, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	// A buffer that outlived an unmap of this range missed the CPU writes since (uploads skip
	// unmapped pages, and pages left CPU-dirty do not fault): the next acquire uploads them.
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Another lane's work, or a deferred write to
		// guest memory (registered by the buffer and texture download publishes), may still use
		// the range. The drain (a full GPU idle) is skipped for ranges no buffer, image, download
		// or pending guest write involves: games unmap dozens of such pages a frame.
		if (m_command_scheduler.Active() && (m_buffer_cache.MayBeInUseByGpu(vaddr, size) ||
		                                     m_texture_cache.IsRegionRegistered(vaddr, size))) {
			const auto tick = m_command_scheduler.CurrentTick();
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::UnmapDrains);
			m_command_scheduler.Finish(Common::FrameProfiler::WaitSite::UnmapDrain);
			m_command_scheduler.WaitPriorityOperations(tick,
			                                           Common::FrameProfiler::WaitSite::UnmapDrain);
		}
		{
			// Labels the readback of an invalidation inside an unmap.
			const Common::FrameProfiler::ScopedCause cause(
			    Common::FrameProfiler::ReadbackCause::Unmap);
			m_buffer_cache.InvalidateMemory(vaddr, size);
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrefetchBda(std::span<const uint64_t> addresses) {
	uint64_t previous = 0;
	for (const auto address: addresses) {
		const auto page = address & ~(BufferCache::CACHING_PAGESIZE - 1);
		if (page != 0 && page != previous && IsMapped(page, BufferCache::CACHING_PAGESIZE)) {
			(void)m_buffer_cache.FindBuffer(page, BufferCache::CACHING_PAGESIZE);
		}
		previous = page;
	}
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Buffers already hold every CPU write the GPU may see: the epoch's acquire uploaded them
	// (CoherenceEpoch), and new buffers upload when created. Shader stores through BDA are not
	// tracked, though: a page the CPU wrote since the acquire would be uploaded whole over them
	// at the next one. Synchronize what was queued since the acquire before them.
	m_buffer_cache.EnsureBdaZeroEntry();
	{
		const Common::FrameProfiler::ScopedUploadSource source(
		    Common::FrameProfiler::UploadSource::Bda);
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	}
	m_fault_process_pending = true;
}

// Samples the mapping, buffer and texture sizes for a running frame-profiler capture, at most
// every 250 ms.
void RenderContext::ProfileLevels() {
	using Common::FrameProfiler::Level;
	using Common::FrameProfiler::SetLevelValue;
	const auto now = std::chrono::steady_clock::now();
	if (now - m_levels_sampled < std::chrono::milliseconds(250)) {
		return;
	}
	m_levels_sampled = now;
	uint64_t mapped  = 0;
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_mapped_ranges.ForEach([&](uint64_t begin, uint64_t end) { mapped += end - begin; });
	}
	SetLevelValue(Level::MappedBytes, mapped);
	SetLevelValue(Level::BufferBytes, m_buffer_cache.BufferBytes());
	SetLevelValue(Level::BufferCount, m_buffer_cache.BufferCount());
	SetLevelValue(Level::TextureBytes, m_texture_cache.ImageBytes());
	SetLevelValue(Level::TexturePeakBytes, m_texture_cache.ImagePeakBytes());
	if (m_graphics.CanReportMemoryUsage()) {
		SetLevelValue(Level::DeviceUsageBytes, m_graphics.GetDeviceMemoryUsage());
	}
	SetLevelValue(Level::DeviceBudgetBytes, m_graphics.GetTotalMemoryBudget());
}

void RenderContext::RunGarbageCollector() {
	if (Common::FrameProfiler::Active()) {
		ProfileLevels();
	}
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
