#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> draw_commands,
	            std::span<const uint32_t> constant_commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;

	// GPU thread: the command processor stored `size` bytes (`bytes`, as written) at `vaddr`. A
	// queue whose suspended WAIT_REG_MEM this satisfies runs at the running queue's next packet
	// boundary.
	static void NoteGuestWrite(uint64_t vaddr, uint64_t size, const void* bytes);
	// GPU thread: the command processor is about to store to guest memory
	// (CommandProcessor::FetchAhead).
	static void BeforeCommandProcessorStore();
	// GPU thread, before a readback's wait for `tick` (QueuePark): true when the dispatch parks
	// instead (QueuePark::Pending), the caller then returns without waiting. Only inside the
	// QueuePark scopes and while another queue can run.
	[[nodiscard]] static bool ParkInsteadOfWait(uint64_t tick);
	// GPU thread, before a materialization reads GPU-written memory back: true when the test hook
	// below parks the dispatch here (once per packet), whether or not the read would wait.
	[[nodiscard]] static bool ForcedParkBeforeReadback();

	// Queue-scheduling events since start (GPU thread; tests read them through SendCommandSync).
	struct SchedulingStats {
		uint64_t release_switches = 0; // a released queue ran next
		uint64_t yields           = 0; // a slice ended at a packet boundary for another queue
		uint64_t parks            = 0;
		uint64_t park_retries     = 0; // parked again on the same packet
		uint64_t parked_slices    = 0; // slices of other queues while one was parked
		uint64_t ready_resumes    = 0; // a parked queue resumed once its tick completed
		uint64_t idle_resumes     = 0; // a parked queue resumed with nothing else to run
	};
	[[nodiscard]] const SchedulingStats& GetSchedulingStats() const { return m_stats; }
	// Tests: park every dispatch packet whose materialization reads GPU-written memory back, once,
	// while another queue can run.
	void ForceParkForTesting(bool enabled) { m_force_park = enabled; }

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	static constexpr uint32_t NoQueue              = QueueCount;

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
	};

	// A WAIT_REG_MEM a queue suspended on (address 0: none).
	struct SuspendedWait {
		uint64_t address = 0;
		uint32_t size    = 0;
		uint32_t func    = 0;
		uint64_t ref     = 0;
		uint64_t mask    = 0;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	// GPU thread: queue scheduling (all below run on it).
	void              NoteSuspendedWait(const SuspendedWait& wait);
	void              ReleaseWaits(uint64_t vaddr, uint64_t size, const void* bytes);
	// A queue other than the running one may run now, or is parked: m_queue_mutex held.
	[[nodiscard]] bool OtherQueueRunnable() const;
	[[nodiscard]] bool ParkedReady() const;
	// At a packet boundary: another queue should run now.
	[[nodiscard]] bool ShouldYield();
	void               Park(uint64_t tick, const uint32_t* packet);
	void               UpdateParkedTick();
	// Under m_queue_mutex: the next queue to run, or -1.
	[[nodiscard]] int  SelectQueue();
	[[nodiscard]] int  SelectNextQueue();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	// GPU thread: queue scheduling. A queue released by a label it waits for, a queue parked on a
	// host-GPU tick, and the queue processing now.
	std::array<SuspendedWait, QueueCount> m_waits {};
	uint32_t                              m_wait_count = 0;
	std::deque<uint32_t>                  m_released;
	std::array<uint64_t, QueueCount>      m_park_tick {};
	std::array<uint64_t, QueueCount>      m_park_since {};
	std::array<const uint32_t*, QueueCount> m_park_packet {};
	uint32_t                              m_park_count     = 0;
	uint64_t                              m_min_park_tick  = 0;
	std::chrono::steady_clock::time_point m_park_refresh;
	uint32_t                              m_current_queue  = NoQueue;
	// A released queue waits for the running one; the running slice ended at a yield or park.
	bool                                  m_yield_requested = false;
	bool                                  m_slice_yielded   = false;
	// The packet ProcessPm4 executes, for the forced park (tests).
	const uint32_t*                       m_current_packet  = nullptr;
	const uint32_t*                       m_forced_packet   = nullptr;
	bool                                  m_force_park      = false;
	SchedulingStats                       m_stats;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
