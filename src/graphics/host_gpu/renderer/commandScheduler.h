#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <source_location>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	using WaitSite = Common::FrameProfiler::WaitSite;

	// `present`: the presentation scheduler (its GPU time is counted apart).
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool present = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	// Each wait takes the frame profiler's site; unlabelled callers are listed by location.
	void           FlushAndWait(WaitSite             site     = WaitSite::Other,
	                            std::source_location location = std::source_location::current());
	void           Finish(WaitSite             site     = WaitSite::Other,
	                      std::source_location location = std::source_location::current());
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick, WaitSite site = WaitSite::Other,
	                               std::source_location location = std::source_location::current());
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(
	                         uint64_t tick, WaitSite site = WaitSite::Other,
	                         std::source_location location = std::source_location::current());
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// Runs just before each submit of a command buffer with command registers, while it still
	// records and operations are still accepted. The hook may submit (a stream-buffer wrap):
	// that nested submit does not run it again.
	void SetPreSubmitHook(std::function<void()> hook) { m_pre_submit = std::move(hook); }
	// Any thread: whether the command buffer of the tick has been handed to the queue.
	[[nodiscard]] bool IsSubmitted(uint64_t tick) const noexcept {
		return m_submitted.load(std::memory_order_acquire) >= tick;
	}
	// The current command buffer ends with a memory barrier to host reads (for shader writes the
	// host reads once the submission completes, such as bindless miss records). A barrier after
	// a draw would sit inside its render pass.
	void                      RequestHostBarrier() noexcept { m_host_barrier_requested = true; }
	// Guest-ordered barriers. A guest dispatch carries no barrier of its own: as on the hardware,
	// dispatches overlap until the guest synchronises them. Each guest synchronisation packet
	// (partial flush, cache event, release, acquire, label wait, submission start) requests one
	// global barrier, recorded before the next dispatch, or before the next draw while dispatches
	// recorded since the last global barrier are unordered. The request survives submits and
	// feeds: barriers order the command buffers of the one queue in submission order.
	// Emulator-internal GPU work (uploads, downloads, joins, copies, conversions, image
	// transitions) carries its own barriers. Recording thread only.
	void RequestGuestBarrier() noexcept {
		m_guest_barrier_requested = true;
		m_guest_syncs++;
	}
	[[nodiscard]] bool GuestBarrierPending() const noexcept { return m_guest_barrier_requested; }
	// Records the global barrier now: every earlier write before every later access. The site
	// (trace only) tells a packet's own barrier from a request turned into one.
	enum class GlobalBarrierSite : uint8_t { Now, Dispatch, Draw, ThreadBatch };
	void GlobalBarrier(GlobalBarrierSite site = GlobalBarrierSite::Now);
	// Before recording a guest dispatch command (after its preparation, and before a clear that
	// replaces it), and before a guest draw prepares anything.
	void BeforeGuestDispatch();
	void BeforeGuestDraw();
	// Whether a guest dispatch was recorded since the last global barrier, or since the last
	// barrier that made compute writes visible to compute reads (NoteComputeReadBarrier).
	[[nodiscard]] bool UnorderedCompute() const noexcept {
		return m_guest_dispatches != m_dispatches_at_global_barrier;
	}
	[[nodiscard]] bool UnorderedComputeReads() const noexcept {
		return m_guest_dispatches != m_dispatches_at_compute_barrier;
	}
	void NoteComputeReadBarrier() noexcept { m_dispatches_at_compute_barrier = m_guest_dispatches; }
	// Indirect draws reading guest memory as their arguments (IndirectDrawArgs). The hardware's
	// command processor reads them before the work of any later packet runs, so a later guest
	// shader write to that memory must wait for the reads even without a guest synchronisation:
	// OrderIndirectArgsReads records that dependency, before guest work with shader writes at
	// `stages`, when such a read was recorded since the last one or the last global barrier. Ends
	// the rendering instance then.
	void NoteIndirectArgsRead() noexcept { m_indirect_args_reads_unordered = true; }
	void OrderIndirectArgsReads(vk::PipelineStageFlags stages);
	// Dependencies OrderIndirectArgsReads recorded (tests; recording thread only).
	[[nodiscard]] uint64_t IndirectArgsReadBarrierCount() const noexcept {
		return m_indirect_args_read_barriers;
	}
	// Follow each guest draw and dispatch command at once. CountGuestDispatch also keeps the
	// compute-ordering census (frame profiler and tests).
	void CountGuestDraw() noexcept { m_open_work++; }
	void CountGuestDispatch();
	// Lifetime totals (tests): global barriers, and guest dispatches with a global barrier since
	// the previous one.
	[[nodiscard]] uint64_t GlobalBarrierCount() const noexcept { return m_global_barriers; }
	[[nodiscard]] uint64_t OrderedDispatchCount() const noexcept { return m_ordered_dispatches; }
	[[nodiscard]] uint64_t GuestDispatchCount() const noexcept { return m_guest_dispatches; }
	// Called before a guest draw or dispatch prepares anything (where a readback may submit):
	// submits the open command buffer without a wait when it holds FeedMinWork guest draws and
	// dispatches and every submitted one has completed, so the host GPU runs them while the
	// rest of the stream records instead of sitting idle until the next wait.
	void FeedIdleGpu();
	// Submits made by FeedIdleGpu (tests; recording thread only).
	[[nodiscard]] uint64_t FeedSubmitCount() const noexcept { return m_feed_submits; }
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	// GPU execution time of this scheduler's command buffers (labelled regions: Current()).
	[[nodiscard]] GpuTimestamps&   Timestamps() noexcept { return m_timestamps; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	friend struct CommandSchedulerTestAccess;

	// GTA V's first mission records about 1,660 dispatches into its largest command buffer over
	// 14 ms while every submitted one has completed.
	static constexpr uint32_t FeedMinWork = 64;

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void BeginRecording();
	void PriorityOperationsThread(std::stop_token stop);
	// Priority thread: notes the time when the last submitted command buffer is seen complete
	// (the frame profiler's GpuQueueEmptyTime).
	void NoteQueueEmpty();
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	// The presentation scheduler: it neither feeds the GPU nor watches its submits.
	const bool                   m_present;
	GraphicContext&              m_graphics;
	GpuTimestamps                m_timestamps;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	// Wakes the priority thread alone: an operation, a submit, or the stop.
	std::condition_variable      m_priority_wake;
	bool                         m_guest_barrier_requested       = false;
	bool                         m_barrier_since_dispatch        = false;
	bool                         m_indirect_args_reads_unordered = false;
	uint64_t                     m_indirect_args_read_barriers   = 0;
	uint64_t                     m_guest_syncs                   = 0;
	uint64_t                     m_guest_syncs_at_dispatch       = 0;
	uint64_t                     m_global_barriers               = 0;
	uint64_t                     m_ordered_dispatches            = 0;
	uint64_t                     m_guest_dispatches              = 0;
	uint64_t                     m_dispatches_at_global_barrier  = 0;
	uint64_t                     m_dispatches_at_compute_barrier = 0;
	// Guest draws and dispatches recorded into the open command buffer.
	uint32_t                     m_open_work     = 0;
	uint32_t                     m_feed_min_work = FeedMinWork;
	uint64_t                     m_feed_submits  = 0;
	// Written by the priority thread while the frame profiler is active: the last submitted tick
	// it saw complete, and when (profiler ticks).
	std::atomic<uint64_t>        m_empty_tick {0};
	std::atomic<uint64_t>        m_empty_since {0};
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
	bool                         m_host_barrier_requested = false;
	std::function<void()>        m_pre_submit;
	std::chrono::steady_clock::time_point m_last_refresh {};
	bool                         m_in_pre_submit = false;
	std::atomic<uint64_t>        m_submitted {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
