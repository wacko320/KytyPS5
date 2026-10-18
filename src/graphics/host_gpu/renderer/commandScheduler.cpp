#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/frameProfiler.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cstdio>
#include <optional>

namespace Libs::Graphics {

using Common::FrameProfiler::WaitSite;

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics, bool present)
    : m_master(graphics), m_context(context), m_graphics(graphics), m_timestamps(graphics, present),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1, WaitSite::SchedulerShutdown);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait(WaitSite site, std::source_location location) {
	const auto tick = Submit();
	m_master.Wait(tick, site, location);
	BeginNext();
}

void CommandScheduler::Finish(WaitSite site, std::source_location location) {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1, site, location);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick, WaitSite site, std::source_location location) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick, site, location);
		BeginNext();
	} else {
		m_master.Wait(tick, site, location);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Runs before every draw and dispatch: query the GPU only when the next operation is not
	// known complete, and not more often than RefreshInterval. Each query is an ioctl.
	constexpr auto RefreshInterval = std::chrono::microseconds(250);
	bool           refreshed       = false;
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			if (!m_master.IsFree(m_pending_operations.front().tick)) {
				const auto now = std::chrono::steady_clock::now();
				if (refreshed || now - m_last_refresh < RefreshInterval) {
					return;
				}
				m_last_refresh = now;
				m_master.Refresh();
				refreshed = true;
				if (!m_master.IsFree(m_pending_operations.front().tick)) {
					return;
				}
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick, WaitSite::PendingOperations);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick, WaitSite::PriorityFence);
		Common::FrameProfiler::Rec(Common::FrameProfiler::Event::Fence, operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick, WaitSite site,
                                              std::source_location location) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	const auto done = [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	};
	if (done()) {
		return;
	}
	const Common::FrameProfiler::ScopedWait profile(true, site, location, tick);
	m_operation_available.wait(lock, done);
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	BeginRecording();
	return m_command;
}

void CommandScheduler::BeginRecording() {
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	// Reads the timestamps of command buffers already known complete: never a wait.
	m_timestamps.Poll(m_master.KnownGpuTick());
	m_timestamps.Begin(m_command.m_buffer);
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	struct ProfiledSubmit {
		explicit ProfiledSubmit(uint64_t tick): t0(Common::FrameProfiler::Start()) {
			Common::FrameProfiler::RecIf(t0, Common::FrameProfiler::Event::VkSubmit, tick);
		}
		~ProfiledSubmit() {
			Common::FrameProfiler::RecIf(t0, Common::FrameProfiler::Event::VkSubmitEnd);
		}
		const uint64_t t0;
	} const profile(CurrentTick());
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	const bool pre_submit = m_in_pre_submit;
	if (m_pre_submit && !m_in_pre_submit && m_command.m_registers != nullptr) {
		bool open = false;
		{
			std::lock_guard lock(m_operation_mutex);
			open = m_operation_state == OperationState::Open;
		}
		if (open) {
			// The hook may record and wrap stream buffers, which submits through here again.
			m_in_pre_submit = true;
			m_pre_submit();
			m_in_pre_submit = false;
		}
	}

	if (m_host_barrier_requested) {
		m_host_barrier_requested = false;
		m_command.EndRendering();
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
		m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                                   vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0,
		                                   nullptr, 0, nullptr);
	}
	// Timestamps are written outside a render pass.
	m_command.EndRendering();
	const auto timing = m_timestamps.End(m_command.m_buffer);
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		const uint64_t submit_t0 = Common::FrameProfiler::Start();
		result                   = graphics.queue.submit(1, &submit_info, nullptr);
		if (submit_t0 != 0) {
			const auto ticks = Common::FrameProfiler::Ticks() - submit_t0;
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::VkSubmitTime, ticks);
			Common::FrameProfiler::Max(Common::FrameProfiler::Counter::VkSubmitLongest, ticks);
		}
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_submitted.store(tick, std::memory_order_release);
	m_timestamps.Submitted(timing, tick);
	if (profile.t0 != 0) {
		using Common::FrameProfiler::Counter;
		// SubmitsCommand is the rest of QueueSubmits (the capture reads the debug counters).
		if (m_command.m_registers == nullptr || pre_submit) {
			Common::FrameProfiler::Add(m_command.m_registers == nullptr ? Counter::SubmitsOther
			                                                            : Counter::SubmitsPreSubmit);
		}
	}
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
