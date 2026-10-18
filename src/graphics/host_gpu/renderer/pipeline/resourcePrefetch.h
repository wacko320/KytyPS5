#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCEPREFETCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCEPREFETCH_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class SrtPageWatch;

// Materializes shader resources for draws the command processor has decoded but not executed
// yet, on worker threads, so the GPU thread does not evaluate SRT and descriptor IR itself.
//
// A worker evaluates a copy of the plan against a guess of the draw's user data and records
// every guest-memory read it makes. The GPU thread uses the result only when the draw reaches
// it with the same plan, user data and shader base, and when every recorded read, replayed in
// order through the draw's own runtime, returns what the worker saw. The evaluation depends on
// nothing else, so an accepted result is the one the serial path computes; anything else falls
// back to it.
class ResourcePrefetcher {
public:
	using ResourcePlan           = ShaderRecompiler::IR::ResourcePlan;
	using ResourceSnapshot       = ShaderRecompiler::IR::ResourceSnapshot;
	using ResourceSpecialization = ShaderRecompiler::IR::ResourceSpecialization;
	using SrtRuntime             = ShaderRecompiler::IR::SrtRuntime;

	struct MemoryOps {
		// Worker threads: a readable view of the 4 KiB guest page at `page`, or null. The bytes
		// may be stale, since the GPU thread validates them, but the view must not fault.
		const void* (*view_page)(uint64_t page) = nullptr;
		// Optional: pages that need no replay. Both or neither.
		SrtPageWatch* watch = nullptr;
		// GPU thread: guest memory holds the final bytes of the SrtPageWatch page at `page`.
		bool (*page_settled)(uint64_t page) = nullptr;
		// Replay the results accepted on settled pages too, and count those that differ
		// (Stats::replay_mismatches). This never changes which result is used.
		bool replay_check = false;
	};

	struct Stats {
		uint64_t submitted = 0;
		// A draw's evaluation used a worker result.
		uint64_t hits = 0;
		// No speculation matched the draw's plan (not predicted, or mispredicted).
		uint64_t unmatched = 0;
		// Speculated with other user data or shader base.
		uint64_t mismatched = 0;
		// The worker had not started yet; the GPU thread evaluated it instead.
		uint64_t cancelled = 0;
		// The worker could not evaluate it (unreadable memory or failed materialization).
		uint64_t failed = 0;
		// A replayed read returned other bytes.
		uint64_t stale = 0;
		// Plans the prefetcher stopped speculating for after repeated stale reads.
		uint64_t throttled = 0;
		// Hits accepted on settled pages, without a replay.
		uint64_t unreplayed = 0;
		// MemoryOps::replay_check: unreplayed hits whose replay differed.
		uint64_t replay_mismatches = 0;
		// Hand-offs to the workers, and the evaluations they queued.
		uint64_t flushes      = 0;
		uint64_t flushed_jobs = 0;
		// Job records allocated (a retired record is reused).
		uint64_t jobs = 0;
	};

	static constexpr uint64_t PageSize = 4096;

	ResourcePrefetcher(uint32_t worker_count, MemoryOps ops);
	~ResourcePrefetcher();
	ResourcePrefetcher(const ResourcePrefetcher&)            = delete;
	ResourcePrefetcher& operator=(const ResourcePrefetcher&) = delete;

	[[nodiscard]] uint32_t WorkerCount() const { return static_cast<uint32_t>(m_workers.size()); }

	// The rest runs on the GPU thread.

	// Plans that read linked call sources are left to the serial path.
	[[nodiscard]] static bool CanSpeculate(const ResourcePlan& plan);

	// Queues an evaluation for draw `draw` (draw numbers only increase). `key` identifies the
	// plan; `replicas` holds one independent copy of it per worker. A result is used only for a
	// runtime with the same user data, shader base and workgroup counts.
	void Submit(uint64_t draw, const void* key, std::span<const ResourcePlan* const> replicas,
	            std::span<const uint32_t> user_data, uint64_t shader_base,
	            std::span<const uint32_t> workgroup_counts = {});
	// Until Flush, submissions wait on the GPU thread and reach the workers together, with one
	// lock and one wakeup, also when a batch fills up or a draw looks for one of them.
	void BeginBatch() { m_batching = true; }
	void Flush();
	// The command processor reached draw `draw`: speculation for earlier draws is dropped.
	void BeginDraw(uint64_t draw);
	void Clear();
	// Clear, and wait until no worker evaluates a plan any more: before the plans or their
	// replicas change.
	void ClearAndWait();

	// Materializes `key` for the current draw from a worker result. False leaves the outputs
	// untouched; the caller then evaluates the plan itself.
	bool TryMaterialize(const void* key, const SrtRuntime& runtime, ResourceSnapshot& snapshot,
	                    ResourceSpecialization& specialization);

	[[nodiscard]] const Stats& GetStats() const { return m_stats; }
	// No handed-over job of the window is queued or running (tests wait for this).
	[[nodiscard]] bool Quiescent() const;
	// GPU thread: see MemoryOps::replay_check.
	void SetReplayCheck(bool enabled) { m_ops.replay_check = enabled; }

private:
	enum class State : uint8_t { Queued, Running, Done, Cancelled };

	struct Read {
		uint64_t address = 0;
		uint32_t offset  = 0;
		uint32_t count   = 0;
		bool     checked = false;
		bool     ok      = false;
	};

	struct Job {
		// generation << 8 | State. The generation tells a worker that a queued pointer is stale.
		std::atomic<uint64_t> state {0};
		uint64_t              generation = 0;
		uint64_t              draw       = 0;
		const void*           key        = nullptr;
		bool                  consumed   = false;
		std::vector<const ResourcePlan*> replicas;
		std::vector<uint32_t>            user_data;
		uint64_t                         shader_base = 0;
		std::vector<uint32_t>            workgroup_counts;
		// The SrtPageWatch generation and settle count at the submit.
		uint64_t                         watch_generation = 0;
		uint64_t                         watch_settles    = 0;
		// Written by the worker before it publishes Done.
		bool                   ok = false;
		ResourceSnapshot       snapshot;
		ResourceSpecialization specialization;
		std::vector<Read>      reads;
		std::vector<uint32_t>  values;
		// Every read succeeded.
		bool                   reads_ok = true;
		// The SrtPageWatch pages the reads touched.
		std::vector<uint64_t>  pages;
	};

	struct QueueItem {
		Job*     job        = nullptr;
		uint64_t generation = 0;
	};

	struct Health {
		int32_t  score   = 0;
		uint32_t skipped = 0;
	};

	struct WorkerContext;

	static uint64_t Pack(uint64_t generation, State state) {
		return (generation << 8u) | static_cast<uint64_t>(state);
	}
	static State StateOf(uint64_t word) { return static_cast<State>(word & 0xffu); }

	// Queues the waiting submissions that were not retired meanwhile.
	void Publish();
	void WorkerLoop(uint32_t index);
	void Evaluate(WorkerContext& context, Job& job);
	bool Validate(const Job& job, const SrtRuntime& runtime);
	[[nodiscard]] bool IsSettled(const Job& job) const;
	void               Settle(const Job& job);
	Job* Allocate();
	void Retire(Job* job);
	static bool ReadDirect(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadChecked(void* userdata, uint64_t address, std::span<uint32_t> values);

	MemoryOps m_ops;

	// GPU thread.
	std::deque<Job*>                           m_window;
	std::vector<std::unique_ptr<Job>>          m_jobs;
	std::vector<Job*>                          m_free;
	std::vector<Job*>                          m_retired;
	uint64_t                                   m_current_draw = 0;
	std::unordered_map<const void*, Health>    m_health;
	Stats                                      m_stats;
	struct ReadGroup {
		uint64_t page  = 0;
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint32_t offset = 0;
		uint8_t  state = 0;
	};
	std::vector<ReadGroup> m_groups;
	std::vector<int32_t>   m_group_of;
	std::vector<uint32_t>  m_group_values;
	std::vector<uint32_t>  m_scratch;
	// Pages found unsettled at m_unsettled_generation, not checked again before it moves.
	std::vector<uint64_t>  m_unsettled;
	uint64_t               m_unsettled_generation = 0;
	// Submitted, not queued yet (BeginBatch).
	std::vector<QueueItem> m_pending;
	bool                   m_batching = false;

	// Shared with the workers.
	std::mutex              m_queue_mutex;
	std::condition_variable m_queue_ready;
	std::deque<QueueItem>   m_queue;
	std::atomic<uint32_t>   m_queued {0};
	std::atomic<uint32_t>   m_sleeping {0};
	std::atomic<bool>       m_stop {false};
	std::vector<std::thread> m_workers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RESOURCEPREFETCH_H_
