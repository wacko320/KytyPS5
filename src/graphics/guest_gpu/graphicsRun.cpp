#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/frameProfiler.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/command_processor/pm4Lookahead.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

struct DrawIndirectArgs {
	uint32_t vertex_count_per_instance;
	uint32_t instance_count;
	uint32_t start_vertex_location;
	uint32_t start_instance_location;
};

struct DrawIndexedIndirectArgs {
	uint32_t index_count_per_instance;
	uint32_t instance_count;
	uint32_t start_index_location;
	uint32_t base_vertex_location;
	uint32_t start_instance_location;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances                    = 1;
	m_num_instances_capture            = {};
	m_num_instances_unknown            = false;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::CopyRegisterState(const CommandProcessor& other) {
	m_ctx                  = other.m_ctx;
	m_context_state_pushed = other.m_context_state_pushed;
	if (m_context_state_pushed) {
		m_saved_ctx = other.m_saved_ctx;
	}
	m_ucfg                             = other.m_ucfg;
	m_sh_ctx                           = other.m_sh_ctx;
	m_user_data_marker                 = other.m_user_data_marker;
	m_index_type_and_size              = other.m_index_type_and_size;
	m_index_buffer_size                = other.m_index_buffer_size;
	m_index_base_addr                  = other.m_index_base_addr;
	m_draw_indirect_args_base_addr     = other.m_draw_indirect_args_base_addr;
	m_dispatch_indirect_args_base_addr = other.m_dispatch_indirect_args_base_addr;
	m_num_instances                    = other.m_num_instances;
	m_num_instances_capture            = other.m_num_instances_capture;
	m_num_instances_unknown            = other.m_num_instances_unknown;
	m_predicate_skip                   = other.m_predicate_skip;
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish(Common::FrameProfiler::WaitSite::Pm4BufferWait);
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4(SuspendReason::Ce);
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4(SuspendReason::DeDiff);
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4(SuspendReason::Rewind);
		return;
	}
	// The guest marked the commands after the rewind valid; what it wrote before that is ordered
	// before them.
	m_renderer.GetBufferCache().Epoch().Acquire();
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	const auto bytes = static_cast<size_t>(dw_num) * 4;
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst), bytes, [&](void* to) {
		memcpy(to, m_const_ram + offset / 4, bytes);
	});
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(*addr, ref, mask, func)) {
		// A wait that never completes leaves the game stuck with no other trace; report it once
		// after two seconds (a retried wait re-executes this packet).
		struct StalledWait {
			const void*                           addr = nullptr;
			uint64_t                              ref  = 0;
			std::chrono::steady_clock::time_point since;
			bool                                  reported = false;
		};
		thread_local StalledWait stalled;
		const auto               now = std::chrono::steady_clock::now();
		if (stalled.addr != addr || stalled.ref != static_cast<uint64_t>(ref)) {
			stalled = {addr, static_cast<uint64_t>(ref), now, false};
		} else if (!stalled.reported && now - stalled.since > std::chrono::seconds(2)) {
			stalled.reported = true;
			LOGF("WaitRegMem stalled >2s: cp=%p addr=%p value=0x%" PRIx64 " mask=0x%" PRIx64
			     " func=%" PRIu32 " ref=0x%" PRIx64 "\n",
			     static_cast<const void*>(this), static_cast<const void*>(addr),
			     static_cast<uint64_t>(*addr), static_cast<uint64_t>(mask), func,
			     static_cast<uint64_t>(ref));
		}
		SuspendPm4(SuspendReason::WaitRegMem, reinterpret_cast<uint64_t>(addr), func);
		Common::FrameProfiler::Rec(Common::FrameProfiler::Event::SuspendWait,
		                           static_cast<uint64_t>(ref), static_cast<uint64_t>(mask));
		return;
	}
	if (m_suspended_wait != 0) {
		if (m_suspended_wait == reinterpret_cast<uint64_t>(addr)) {
			Common::FrameProfiler::Rec(Common::FrameProfiler::Event::WaitPass, m_suspended_wait,
			                           static_cast<uint64_t>(*addr));
		}
		m_suspended_wait = 0;
	}
	// A CPU write before the one this wait observed is ordered before the commands after it, also
	// when the wait passes on arrival (a resumed wait begins a new slice as well).
	m_renderer.GetBufferCache().Epoch().Acquire();
	// So is the GPU work before it: labels are written when their packet is processed, before
	// the work they follow completes on the host GPU.
	RequestGuestBarrier(GuestSync::WaitRegMem);
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	const auto bytes = static_cast<uint64_t>(write_one_address ? 1u : dw_num) * sizeof(uint32_t);
	// Repeated writes to one address leave the last.
	const auto* data = write_one_address ? src + (dw_num - 1u) : src;
	const auto profile_label = [&](bool gpu_ordered) {
		if (Common::FrameProfiler::Active()) {
			uint64_t value = 0;
			std::memcpy(&value, data, std::min<uint64_t>(bytes, sizeof(value)));
			Common::FrameProfiler::RecordEvent(
			    Common::FrameProfiler::Event::LabelWrite,
			    reinterpret_cast<uint64_t>(dst) | (gpu_ordered ? 1ull << 63u : 0u), value);
		}
	};
	if (m_renderer.TryWriteGpuOrdered(reinterpret_cast<uint64_t>(dst), data, bytes)) {
		profile_label(true);
		return;
	}
	profile_label(false);
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst), bytes, [&](void* to) {
		auto* words = static_cast<uint32_t*>(to);
		if (write_one_address) {
			for (uint32_t i = 0; i < dw_num; i++) {
				words[0] = src[i];
			}
		} else {
			memcpy(words, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
		}
	});
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(dst_address, num_bytes,
	                               [&](void* to) { std::memcpy(to, &value, num_bytes); });
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::WriteLodStats(void* dst, uint32_t num_bytes) {
	// No statistics are gathered: the buffer reads as zero, with its first dword (the ready
	// label) set. Hack?
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst), num_bytes, [&](void* to) {
		memset(to, 0, num_bytes);
		if (num_bytes >= sizeof(uint32_t)) {
			const uint32_t label = 1;
			memcpy(to, &label, sizeof(label));
		}
	});
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	// The command processor writes memory: arguments scanned ahead may change.
	EndThreadWindow();
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	if (Common::FrameProfiler::Active()) {
		Common::FrameProfiler::RecordEvent(Common::FrameProfiler::Event::Enqueue,
		                                   submission.queue_id,
		                                   static_cast<uint64_t>(submission.type));
		if (submission.type == SubmissionType::Graphics && submission.queue_id == 0) {
			Common::FrameProfiler::NoteRenderThread();
		}
	}
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// GPU thread: a wait's time and count, and the longest single wait.
static void ProfileGpuWait(uint64_t t0, Common::FrameProfiler::Counter time,
                           Common::FrameProfiler::Counter count) {
	if (t0 == 0) {
		return;
	}
	const uint64_t ticks = Common::FrameProfiler::Ticks() - t0;
	Common::FrameProfiler::Add(time, ticks);
	Common::FrameProfiler::Add(count);
	Common::FrameProfiler::Max(Common::FrameProfiler::Counter::GpuLongestWait, ticks);
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;
	g_gpu_state  = gpu;
	Common::FrameProfiler::MarkGpuThread();

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				const uint64_t idle_t0 = Common::FrameProfiler::Start();
				Common::FrameProfiler::RecIf(idle_t0, Common::FrameProfiler::Event::IdleBegin);
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
				Common::FrameProfiler::RecIf(idle_t0, Common::FrameProfiler::Event::IdleEnd);
				ProfileGpuWait(idle_t0, Common::FrameProfiler::Counter::GpuIdleTime,
				               Common::FrameProfiler::Counter::GpuIdleWaits);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const uint64_t block_t0 = Common::FrameProfiler::Start();
					Common::FrameProfiler::RecIf(block_t0, Common::FrameProfiler::Event::BlockedBegin);
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					Common::FrameProfiler::RecIf(block_t0, Common::FrameProfiler::Event::BlockedEnd);
					ProfileGpuWait(block_t0, Common::FrameProfiler::Counter::GpuBlockedTime,
					               Common::FrameProfiler::Counter::GpuBlockedWaits);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}

		// A processing slice: one GPU-thread command or one Process call.
		const uint64_t proc_t0  = Common::FrameProfiler::Start();
		uint64_t       proc_cpu = 0;
		if (proc_t0 != 0) {
			Common::FrameProfiler::RecordEvent(Common::FrameProfiler::Event::ProcessBegin,
			                                   command ? 99u : submission.queue_id,
			                                   command || submission.started ? 0u : 1u);
			proc_cpu = Common::FrameProfiler::ThreadCpuNs();
		}
		const auto end_slice = [proc_t0, proc_cpu](bool complete) {
			if (proc_t0 == 0) {
				return;
			}
			Common::FrameProfiler::RecordEvent(Common::FrameProfiler::Event::ProcessEnd,
			                                   complete ? 1u : 0u, 0);
			Common::FrameProfiler::AddSince(Common::FrameProfiler::Counter::GpuProcessTime, proc_t0);
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::GpuProcessCpuTime,
			                           Common::FrameProfiler::ThreadCpuNs() - proc_cpu);
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::GpuProcessSlices);
		};

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			command();
			end_slice(true);

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const bool complete = gpu->Process(submission);
		end_slice(complete);

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
		// A submission does not overlap the work submitted before it.
		cp.RequestGuestBarrier(CommandProcessor::GuestSync::Submission);
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				cp.BufferFlush();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				cp.BufferFlush();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			cp.EmitGlobalBarrier();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	if (!m_lookahead_checked) {
		m_lookahead_checked = true;
		if (m_renderer.GetPipelineCache().PrefetchEnabled()) {
			m_lookahead = std::make_unique<Pm4Lookahead>(m_renderer, m_interrupt_event_id);
		}
	}
	// Other processors and the guest ran since the last slice.
	if (m_lookahead != nullptr) {
		m_lookahead->Reset();
	}
	EndThreadWindow();
	// Every command-buffer slice begins an epoch: CPU writes made before it reach the buffers
	// once here, so its read bindings only translate guest addresses (CoherenceEpoch).
	m_renderer.GetBufferCache().Epoch().Acquire();
	ProcessPm4(execution);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4(SuspendReason reason, uint64_t address, uint32_t func) {
	EXIT_IF(g_current_execution == nullptr);
	Common::FrameProfiler::Rec(Common::FrameProfiler::Event::Suspend, address,
	                           static_cast<uint64_t>(reason) | (uint64_t {func} << 8u));
	if (reason == SuspendReason::WaitRegMem) {
		m_suspended_wait = address;
	}
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		// Readbacks are labelled with the packet that asked for them (frame profiler).
		const Common::FrameProfiler::ScopedPacket profile_packet(Common::FrameProfiler::PacketFetch);
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if ((packet_header >> 30u) == 0u) {
			const auto packet_dw = ((packet_header >> 16u) & 0x3fffu) + 2u;
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		if (m_lookahead != nullptr) {
			const auto kind = Pm4Lookahead::Classify(packet);
			if (kind == Pm4Lookahead::PacketKind::Draw ||
			    kind == Pm4Lookahead::PacketKind::Dispatch) {
				Common::FrameProfiler::t_packet = Common::FrameProfiler::PacketLookahead;
				m_lookahead->OnDraw(*this, cursor.commands, cursor.offset_dw);
			}
		}
		Common::FrameProfiler::t_packet = static_cast<uint8_t>(opcode);
		m_packet_commands = cursor.commands;
		m_packet_offset   = cursor.offset_dw;
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		m_packet_commands = {};
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			const auto now = std::chrono::steady_clock::now();
			if (execution.m_stall_packet != packet) {
				execution.m_stall_packet   = packet;
				execution.m_stall_since    = now;
				execution.m_stall_reported = false;
			} else if (!execution.m_stall_reported &&
			           now - execution.m_stall_since > std::chrono::seconds(2)) {
				execution.m_stall_reported = true;
				LOGF("PM4 execution %p suspended >2s on packet %p: op=0x%02" PRIx32
				     " r=0x%02" PRIx32 " len=%" PRIu32 " dw=[%08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "]\n",
				     static_cast<const void*>(&execution), static_cast<const void*>(packet),
				     opcode, KYTY_PM4_R(packet_header), KYTY_PM4_LEN(packet_header),
				     packet[0], packet[1], packet[2], packet[3], packet[4]);
			}
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances         = num_instances;
	m_num_instances_capture = {};
	m_num_instances_unknown = false;
}

void CommandProcessor::SetIndirectInstances(uint32_t instance_count) {
	m_num_instances         = instance_count;
	m_num_instances_capture = {};
	m_num_instances_unknown = false;
}

uint32_t CommandProcessor::NumInstances() {
	// InstanceCountMayBeRead found no reader before the state is set again.
	EXIT_IF(m_num_instances_unknown);
	if (m_num_instances_capture.tick != 0) {
		m_num_instances = m_renderer.GetRenderExecutor().ReadCapturedInstances(
		    m_num_instances_capture);
		m_num_instances_capture = {};
	}
	return m_num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4(SuspendReason::Predication, reinterpret_cast<uint64_t>(address));
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	if (GraphicsRunDebugDumpEnabled() && (args.base_vertex != 0 || args.first_instance != 0)) {
		LOGF("\t draw indexed offsets: base_vertex = %" PRId32 ", first_instance = %" PRIu32 "\n",
		     args.base_vertex, args.first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	DrawIndex({.index_count = index_count, .index_addr = index_addr});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
}

bool CommandProcessor::InstanceCountMayBeRead() const {
	if (m_packet_commands.empty()) {
		// Not executing a packet of a stream (tests): nothing to scan.
		return true;
	}
	constexpr uint32_t MaxScanWords = 4096;
	const auto         total        = static_cast<uint32_t>(m_packet_commands.size());
	auto     scan   = m_packet_offset + Pm4::PacketSizeDw(m_packet_commands[m_packet_offset]);
	uint32_t budget = MaxScanWords;
	// Mirrors ProcessPm4's framing, as the lookahead does.
	while (scan < total && budget != 0) {
		const auto* packet    = m_packet_commands.data() + scan;
		const auto  remaining = total - scan;
		const auto  header    = packet[0];
		uint32_t    length    = 1;
		const bool  one_dword = header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u;
		if (!one_dword) {
			if (remaining < 2) {
				return true;
			}
			length = (header >> 30u) == 0u   ? ((header >> 16u) & 0x3fffu) + 2u
			         : (header >> 30u) == 3u ? Pm4::PacketSizeDw(header)
			                                 : 0u;
			if (length == 0 || length > remaining) {
				return true;
			}
		}
		const bool skipped = one_dword || (header >> 30u) == 0u ||
		                     ((header & 1u) != 0 && m_predicate_skip);
		if (!skipped) {
			const auto exact = header & ~1u;
			switch ((exact >> 8u) & 0xffu) {
				case Pm4::IT_NUM_INSTANCES:
					if (exact == 0xc0002f00u) {
						return false;
					}
					return true;
				case Pm4::IT_DRAW_INDIRECT:
				case Pm4::IT_DRAW_INDEX_INDIRECT:
					// Either path sets the state.
					if (exact == 0xc0032400u || exact == 0xc0032500u) {
						return false;
					}
					return true;
				case Pm4::IT_DRAW_INDEX_2:
				case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
					// The long form takes its own instance count when it is not 0.
					if (exact == 0xc0073a00u && packet[7] != 0) {
						break;
					}
					return true;
				default:
					switch (Pm4Lookahead::Classify(packet)) {
						case Pm4Lookahead::PacketKind::Register:
						case Pm4Lookahead::PacketKind::Transparent:
						case Pm4Lookahead::PacketKind::Dispatch: break;
						case Pm4Lookahead::PacketKind::Draw:
						case Pm4Lookahead::PacketKind::Barrier: return true;
					}
					break;
			}
		}
		scan += length;
		budget -= std::min(budget, length);
	}
	return true;
}

CommandProcessor::IndirectArgsRoute CommandProcessor::DrawIndirectGpu(uint64_t args_addr,
                                                                      bool indexed, bool last) {
	using Common::FrameProfiler::Counter;
	Common::FrameProfiler::Add(Counter::DrawIndirectDraws);
	const uint64_t size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	// A CPU read of arguments the GPU may have written waits for that work to finish. Arguments
	// outside mapped guest memory no GPU work wrote.
	if (!m_renderer.IsMapped(args_addr, size) ||
	    !m_renderer.GetBufferCache().IsRegionGpuModified(args_addr, size)) {
		return IndirectArgsRoute::Cpu;
	}
	Common::FrameProfiler::Add(Counter::DrawIndirectArgsGpuModified);
	const auto refuse = [](Counter reason) {
		Common::FrameProfiler::Add(reason);
		Common::FrameProfiler::Add(Counter::DrawIndirectCpuArgs);
		return IndirectArgsRoute::Refused;
	};
	if (m_renderer.GetTextureCache().IsRegionGpuModified(args_addr, size)) {
		// A GPU-written image holds them, not the buffer the draw would read.
		return refuse(Counter::DrawIndirectBlockedImage);
	}
	if ((args_addr & 3u) != 0) {
		return refuse(Counter::DrawIndirectBlockedAlignment);
	}
	auto&       executor = m_renderer.GetRenderExecutor();
	const bool  capture  = last && InstanceCountMayBeRead();
	GpuArgsDraw draw;
	if (indexed) {
		uint64_t index_size = 0;
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
		if (m_index_buffer_size == 0) {
			// No bound range for the draw's first index and count.
			return refuse(Counter::DrawIndirectBlockedIndexSize);
		}
		if (!m_renderer.IsMapped(m_index_base_addr, m_index_buffer_size * index_size)) {
			return refuse(Counter::DrawIndirectBlockedIndexRange);
		}
		draw = executor.DrawIndex(
		    m_submit_id, CurrentBuffer(),
		    {.index_addr          = reinterpret_cast<const void*>(m_index_base_addr),
		     .index_type_and_size = m_index_type_and_size,
		     .offset_source       = DrawOffsetSource::IndirectArgs,
		     .gpu_args            = {args_addr, capture},
		     .index_buffer_size   = m_index_buffer_size});
	} else {
		draw = executor.DrawAuto(m_submit_id, CurrentBuffer(),
		                         {.offset_source = DrawOffsetSource::IndirectArgs,
		                          .gpu_args      = {args_addr, capture}});
	}
	if (!draw.recorded) {
		// The renderer counted why.
		Common::FrameProfiler::Add(Counter::DrawIndirectCpuArgs);
		return IndirectArgsRoute::Refused;
	}
	Common::FrameProfiler::Add(Counter::DrawIndirectGpuArgs);
	EXIT_IF(capture != (draw.instances.tick != 0));
	m_num_instances_capture = draw.instances;
	m_num_instances_unknown = !capture;
	return IndirectArgsRoute::Gpu;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	// A count GPU work wrote: its read waits for that work. The draws then read their arguments
	// on the CPU as well (vkCmdDrawIndexedIndirectCount would take both on the GPU).
	const auto count_vaddr = reinterpret_cast<uint64_t>(count_addr);
	const bool gpu_count =
	    count_addr != nullptr && m_renderer.IsMapped(count_vaddr, sizeof(uint32_t)) &&
	    m_renderer.GetBufferCache().IsRegionGpuModified(count_vaddr, sizeof(uint32_t));
	// Arguments patched into user SGPRs are needed on the CPU.
	const bool sgpr_destination = registers.vertex_offset != Pm4::SH_NOP ||
	                              registers.instance_offset != Pm4::SH_NOP ||
	                              registers.index_offset != Pm4::SH_NOP;
	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}

	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	const auto args_size = indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	uint64_t index_size = 0;
	if (indexed) {
		switch (m_index_type_and_size) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
		}
	}

	// Each draw sets the NUM_INSTANCES state the next one replaces: only the last one's may be
	// read.
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = m_draw_indirect_args_base_addr + data_offset +
		                       static_cast<uint64_t>(i) * stride_in_bytes;

		// GPU-written arguments read on the CPU (frame profiler: DrawIndirectCpuArgsZero).
		bool cpu_reads_gpu_args = false;
		if (gpu_count || sgpr_destination) {
			using Common::FrameProfiler::Counter;
			Common::FrameProfiler::Add(Counter::DrawIndirectDraws);
			const uint64_t size =
			    indexed ? sizeof(DrawIndexedIndirectArgs) : sizeof(DrawIndirectArgs);
			if (m_renderer.IsMapped(args_addr, size) &&
			    m_renderer.GetBufferCache().IsRegionGpuModified(args_addr, size)) {
				Common::FrameProfiler::Add(Counter::DrawIndirectArgsGpuModified);
				Common::FrameProfiler::Add(gpu_count ? Counter::DrawIndirectBlockedGpuCount
				                                     : Counter::DrawIndirectBlockedSgprDestination);
				Common::FrameProfiler::Add(Counter::DrawIndirectCpuArgs);
				cpu_reads_gpu_args = true;
			}
		} else if (const auto route = DrawIndirectGpu(args_addr, indexed, i + 1 == draw_count);
		           route == IndirectArgsRoute::Gpu) {
			continue;
		} else {
			cpu_reads_gpu_args = route == IndirectArgsRoute::Refused;
		}

		if (!indexed) {
			auto* args = reinterpret_cast<const DrawIndirectArgs*>(args_addr);
			if (cpu_reads_gpu_args && args->vertex_count_per_instance == 0) {
				Common::FrameProfiler::Add(Common::FrameProfiler::Counter::DrawIndirectCpuArgsZero);
			}
			PatchIndirectDrawOffsets(*this, registers, args->start_vertex_location,
			                         args->start_instance_location, 0);
			SetIndirectInstances(args->instance_count);
			DrawIndexAuto({.vertex_count   = args->vertex_count_per_instance,
			               .instance_count = args->instance_count,
			               .first_vertex   = args->start_vertex_location,
			               .first_instance = args->start_instance_location,
			               .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		auto* args = reinterpret_cast<const DrawIndexedIndirectArgs*>(args_addr);
		if (cpu_reads_gpu_args && args->index_count_per_instance == 0) {
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::DrawIndirectCpuArgsZero);
		}
		PatchIndirectDrawOffsets(*this, registers, args->base_vertex_location,
		                         args->start_instance_location, args->start_index_location);

		auto* index_addr = reinterpret_cast<const void*>(
		    m_index_base_addr + static_cast<uint64_t>(args->start_index_location) * index_size);

		const uint32_t index_count =
		    (m_index_buffer_size != 0
		         ? std::min(args->index_count_per_instance,
		                    m_index_buffer_size -
		                        std::min(args->start_index_location, m_index_buffer_size))
		         : args->index_count_per_instance);
		if (GraphicsRunDebugDumpEnabled() && index_count != args->index_count_per_instance) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args->index_count_per_instance, index_count);
			}
		}

		SetIndirectInstances(args->instance_count);
		DrawIndex({.index_count    = index_count,
		           .index_addr     = index_addr,
		           .instance_count = args->instance_count,
		           .base_vertex    = static_cast<int32_t>(args->base_vertex_location),
		           .first_instance = args->start_instance_location,
		           .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));

	uint32_t frame_num = 0;

	{
		frame_num = m_renderer.GetGpu().GetFrameNum();
		if (GraphicsRunDebugDumpEnabled()) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
				const auto& cs = m_sh_ctx.GetCs().cs_regs;
				const auto& oa = m_ucfg.GetGdsOaCounter(m_ucfg.GetGdsOaState().GetIndex());
				LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
				     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
				     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
				     "\n",
				     frame_num, m_submit_id, thread_group_x, thread_group_y, thread_group_z,
				     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
				     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
				     cs.data_addr, m_ucfg.GetGdsOaState().GetIndex(),
				     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
				     oa.GetSpaceAvailable());
			}
		}

		auto batch = PlanThreadBatch(false);
		m_renderer.GetRenderExecutor().DispatchDirect(m_submit_id, CurrentBuffer(), thread_group_x,
		                                              thread_group_y, thread_group_z, mode,
		                                              IsAsyncComputeQueue(), &batch);
		NoteThreadBatch(batch);
	}
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	m_sh_ctx.SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	const bool threads = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	auto       batch   = PlanThreadBatch(threads);
	m_renderer.GetRenderExecutor().DispatchIndirect(m_submit_id, CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue(), &batch);
	NoteThreadBatch(batch);
}

ThreadDispatchBatch CommandProcessor::PlanThreadBatch(bool indirect_threads) {
	ThreadDispatchBatch batch;
	if (m_packet_commands.empty()) {
		// Not executing a packet of a stream (tests): no window to scan.
		return batch;
	}
	auto&      window = m_thread_window;
	const auto offset = m_packet_offset;
	if (window.valid && window.commands == m_packet_commands.data() && offset >= window.begin &&
	    offset < window.end) {
		batch.serial = window.serial;
		return batch;
	}
	// A dispatch heads a window when a guest barrier precedes it: the batch then shares that
	// barrier. Later in a window, only an indirect thread-dimension dispatch starts a batch.
	auto& scheduler = GetScheduler();
	if (!indirect_threads && !scheduler.GuestBarrierPending() && scheduler.UnorderedCompute()) {
		return batch;
	}
	if (m_thread_scan == nullptr) {
		m_thread_scan = std::make_unique<Pm4ThreadDispatchScan>(m_renderer, m_interrupt_event_id);
	}
	const auto t0  = Common::FrameProfiler::Start();
	const auto end = m_thread_scan->Scan(*this, m_packet_commands, offset, m_thread_requests);
	if (t0 != 0) {
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::ComputeIndirectThreadScans);
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::ComputeIndirectThreadScanTime,
		                           Common::FrameProfiler::Ticks() - t0);
	}
	window         = {m_packet_commands.data(), offset, end, 0, true};
	batch.requests = m_thread_requests;
	return batch;
}

void CommandProcessor::NoteThreadBatch(const ThreadDispatchBatch& batch) {
	if (batch.requests.empty()) {
		return;
	}
	// A dispatch that returned before recording the batch leaves the window to the next one.
	m_thread_window.serial = batch.serial;
	m_thread_window.valid  = batch.serial != 0;
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = NumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(m_submit_id, CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	auto& command = CurrentBuffer();
	Common::FrameProfiler::Rec(Common::FrameProfiler::Event::EopWrite,
	                           reinterpret_cast<uint64_t>(dst_gpu_addr), static_cast<uint64_t>(value));

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				TriggerEopEventAtEndOfPipe(interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	auto write32 = [&] {
		auto* dst  = static_cast<uint32_t*>(dst_gpu_addr);
		auto  data = static_cast<uint32_t>(value);
		EXIT_IF(dst == nullptr);
		(void)command.Handle();
		EndThreadWindow();
		m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst), sizeof(data),
		                               [&](void* to) { std::memcpy(to, &data, sizeof(data)); });

		if (with_interrupt) {
			TriggerEopEventAtEndOfPipe(interrupt_context_id);
		}
	};

	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) {
					auto* dst = static_cast<uint32_t*>(dst_gpu_addr);
					SynchronizeGpu();
					EndThreadWindow();
					m_renderer.GpuThreadGuestWrite(
					    reinterpret_cast<uint64_t>(dst), uint64_t {value >> 16u} * sizeof(uint32_t),
					    [&](void* to) {
						    Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
						                  static_cast<uint32_t*>(to), value & 0xffffu, value >> 16u);
					    });
					if (with_interrupt) {
						m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
					}
					return;
				}
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				write32();
				return;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06) {
					switch (cache_action) {
						case 0x00:
						case 0x38: write32(); return;
						default: break;
					}
				}
			} else {
				if (event_write_source == 0x04) {
					value = Sync::ReadReferenceClock();
				}
				auto write64 = [&] {
					auto* dst = static_cast<uint64_t*>(dst_gpu_addr);
					EXIT_IF(dst == nullptr);
					(void)command.Handle();
					EndThreadWindow();
					m_renderer.GpuThreadGuestWrite(
					    reinterpret_cast<uint64_t>(dst), sizeof(value),
					    [&](void* to) { std::memcpy(to, &value, sizeof(value)); });

					if (with_interrupt) {
						TriggerEopEventAtEndOfPipe(interrupt_context_id);
					}
				};

				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) {
									write64();
									return;
								}
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) &&
								     event_index == 0x05) ||
								    (event_index == 0x00)) {
									write64();
									return;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) {
									write64();
									return;
								}
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) {
									write64();
									return;
								}
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) {
							write64();
							return;
						}
						break;
					default: break;
				}
			}
			break;
		default: break;
	}

	EXIT("unknown event type\n");
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());
	GetScheduler().GlobalBarrier();
}

void CommandProcessor::RequestGuestBarrier(GuestSync sync) {
	using Common::FrameProfiler::Counter;
	static constexpr Counter Counters[] = {
	    Counter::GuestSyncPacketsPartialFlush, Counter::GuestSyncPacketsOther,
	    Counter::GuestSyncPacketsReleaseMem,   Counter::GuestSyncPacketsAcquireMem,
	    Counter::GuestSyncPacketsWaitRegMem,   Counter::GuestSyncSubmissions};
	static_assert(std::size(Counters) == static_cast<size_t>(GuestSync::Submission) + 1);
	Common::FrameProfiler::Add(Counters[static_cast<uint32_t>(sync)]);
	Common::FrameProfiler::Rec(Common::FrameProfiler::Event::GuestSync,
	                           static_cast<uint64_t>(sync));
	GetScheduler().RequestGuestBarrier();
	EndThreadWindow();
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    });
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010:
			RequestGuestBarrier(GuestSync::PartialFlush);
			EmitGlobalBarrier();
			break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			RequestGuestBarrier(GuestSync::CacheEvent);
			EmitGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			RequestGuestBarrier(GuestSync::CacheEvent);
			EmitGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			constexpr uint32_t db_count     = 16u;
			constexpr uint64_t results_size = uint64_t {db_count} * 2u * sizeof(uint64_t);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			EndThreadWindow();
			m_renderer.GpuThreadGuestWrite(event_address, results_size, [&](void* to) {
				auto* results = static_cast<volatile uint64_t*>(to);
				for (uint32_t db = 0; db < db_count; db++) {
					results[db * 2u] = value;
				}
			});
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	auto& command = CurrentBuffer();
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().Flush();
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	EXIT_IF(dst_gpu_addr == nullptr);
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value),
	                               [&](void* to) { std::memcpy(to, &value, sizeof(value)); });
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().Flush();
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	auto& command = CurrentBuffer();

	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	EXIT_IF(dst_gpu_addr == nullptr);
	EndThreadWindow();
	m_renderer.GpuThreadGuestWrite(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value),
	                               [&](void* to) { std::memcpy(to, &value, sizeof(value)); });
	auto request = Sync::PrepareVideoOutFlip(command, m_flip.handle, m_flip.index, m_flip.flip_mode,
	                                         m_flip.flip_arg);
	(void)command.Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, request] {
		    renderer.GetVideoOut().CompleteFlip(request);
		    renderer.TriggerInterrupt(event_id, 0);
	    });
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	struct ProcessorScope {
		explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
		~ProcessorScope() { g_current_processor = nullptr; }
	};
	ProcessorScope processor_scope(*this);

	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().Flush();
}

void CommandProcessor::SynchronizeGpu() {
	GetScheduler().Finish(Common::FrameProfiler::WaitSite::Pm4SynchronizeGpu);
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

} // namespace Libs::Graphics
