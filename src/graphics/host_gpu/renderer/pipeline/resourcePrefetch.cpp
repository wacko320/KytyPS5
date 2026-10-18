#include "graphics/host_gpu/renderer/pipeline/resourcePrefetch.h"

#include "graphics/guest_gpu/queuePark.h"

#include "graphics/host_gpu/renderer/cache/srtPageWatch.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

void CpuRelax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#else
	std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// A draw's preparation takes tens of microseconds on the GPU thread, so a worker that just ran
// out of jobs usually gets the next one within this window. Sleeping instead costs a wakeup of
// similar length on every draw.
constexpr auto WorkerSpin = std::chrono::microseconds(50);

// Plans whose speculation keeps reading stale bytes (their inputs are written by the GPU or by
// packets just before the draw) are retried only occasionally.
constexpr int32_t  HealthMax      = 8;
constexpr int32_t  HealthPenalty  = 4;
constexpr int32_t  HealthThrottle = -16;
constexpr int32_t  HealthMin      = -64;
constexpr uint32_t ThrottleRetry  = 16;

constexpr uint8_t GroupUnchecked = 0;
constexpr uint8_t GroupClean     = 1;
constexpr uint8_t GroupDirty     = 2;
constexpr size_t  MaxGroups      = 32;

constexpr size_t MaxUnsettled = 64;

// A batch this long is handed over before it ends, so the workers start on long scans.
constexpr size_t FlushCap = 32;

} // namespace

struct ResourcePrefetcher::WorkerContext {
	static constexpr size_t CacheSize = 8;

	const ResourcePrefetcher* owner   = nullptr;
	uint32_t                  index   = 0;
	Job*                      job     = nullptr;
	bool                      aborted = false;
	std::array<uint64_t, CacheSize>       pages {};
	std::array<const uint8_t*, CacheSize> views {};
	uint32_t                              cached = 0;
	uint32_t                              next   = 0;
	std::vector<uint32_t>                 bindless_pcs;

	void Reset() {
		cached = 0;
		next   = 0;
	}

	const uint8_t* View(uint64_t page) {
		for (uint32_t i = 0; i < cached; i++) {
			if (pages[i] == page) {
				return views[i];
			}
		}
		if (owner->m_ops.watch != nullptr) {
			const auto watched = page & ~(SrtPageWatch::PageSize - 1u);
			if (std::ranges::find(job->pages, watched) == job->pages.end()) {
				job->pages.push_back(watched);
			}
		}
		const auto* view = static_cast<const uint8_t*>(owner->m_ops.view_page(page));
		const auto  slot = cached < CacheSize ? cached++ : next++ % CacheSize;
		pages[slot]      = page;
		views[slot]      = view;
		return view;
	}

	bool Read(uint64_t address, std::span<uint32_t> values) {
		const auto size = values.size_bytes();
		if (size == 0 || UINT64_MAX - address < size) {
			return false;
		}
		auto*    out     = reinterpret_cast<uint8_t*>(values.data());
		uint64_t current = address;
		uint64_t left    = size;
		while (left != 0) {
			const auto page   = current & ~(PageSize - 1u);
			const auto offset = current - page;
			const auto bytes  = std::min<uint64_t>(left, PageSize - offset);
			const auto* view  = View(page);
			if (view == nullptr) {
				return false;
			}
			std::memcpy(out, view + offset, bytes);
			out += bytes;
			current += bytes;
			left -= bytes;
		}
		return true;
	}

	void Record(uint64_t address, std::span<const uint32_t> values, bool checked, bool ok) {
		auto& job = *this->job;
		job.reads.push_back({.address = address,
		                     .offset  = static_cast<uint32_t>(job.values.size()),
		                     .count   = static_cast<uint32_t>(values.size()),
		                     .checked = checked,
		                     .ok      = ok});
		if (ok) {
			job.values.insert(job.values.end(), values.begin(), values.end());
		} else {
			job.reads_ok = false;
		}
	}
};

ResourcePrefetcher::ResourcePrefetcher(uint32_t worker_count, MemoryOps ops): m_ops(ops) {
	if (m_ops.view_page == nullptr) {
		return;
	}
	m_workers.reserve(worker_count);
	for (uint32_t i = 0; i < worker_count; i++) {
		m_workers.emplace_back([this, i]() { WorkerLoop(i); });
	}
}

ResourcePrefetcher::~ResourcePrefetcher() {
	{
		std::lock_guard lock(m_queue_mutex);
		m_stop.store(true, std::memory_order_relaxed);
	}
	m_queue_ready.notify_all();
	for (auto& worker: m_workers) {
		worker.join();
	}
}

bool ResourcePrefetcher::CanSpeculate(const ResourcePlan& plan) {
	// Capturing plans (capture_specialization_reads) are speculated too. MaterializeResources
	// wraps the worker's readers in the same SrtReadCapture and clean walker as the GPU thread's,
	// so the captured ranges are what those readers returned, while the worker's own recording
	// (Job::reads) stays outside the snapshot. A result is used only when every recorded read,
	// strict ones included, gives the same outcome and words through the draw's readers (or its
	// pages stayed settled, which no GPU-written byte is), so the serial evaluation makes the
	// same reads in the same order: the same specialization_reads and clean-walker results.
	// Plans with linked call sources read them through the source's own refresh, which a worker
	// cannot replay, and their captured ranges start with that refresh's reads.
	return plan.resource_tracking_complete && plan.srt_plan_complete && plan.source_reads.empty();
}

void ResourcePrefetcher::WorkerLoop(uint32_t index) {
	WorkerContext context {.owner = this, .index = index};
	// One item per lock: an item a worker holds but has not started is out of the others' reach.
	const auto pop = [this]() {
		const auto item = m_queue.front();
		m_queue.pop_front();
		m_queued.fetch_sub(1, std::memory_order_relaxed);
		return item;
	};
	for (;;) {
		QueueItem item {};
		bool      found = false;
		const auto spin_end = std::chrono::steady_clock::now() + WorkerSpin;
		for (uint32_t spin = 0;; spin++) {
			if (m_queued.load(std::memory_order_acquire) != 0) {
				// The GPU thread queues under this lock: a spinning worker never makes it wait.
				std::unique_lock lock(m_queue_mutex, std::try_to_lock);
				if (lock.owns_lock() && !m_queue.empty()) {
					item  = pop();
					found = true;
					break;
				}
			}
			if (m_stop.load(std::memory_order_relaxed)) {
				return;
			}
			if ((spin & 63u) == 63u && std::chrono::steady_clock::now() >= spin_end) {
				break;
			}
			CpuRelax();
		}
		if (!found) {
			std::unique_lock lock(m_queue_mutex);
			m_sleeping.fetch_add(1, std::memory_order_seq_cst);
			m_queue_ready.wait(lock, [this]() {
				return m_stop.load(std::memory_order_relaxed) || !m_queue.empty();
			});
			m_sleeping.fetch_sub(1, std::memory_order_relaxed);
			if (m_stop.load(std::memory_order_relaxed)) {
				return;
			}
			item = pop();
		}
		auto expected = Pack(item.generation, State::Queued);
		if (!item.job->state.compare_exchange_strong(expected, Pack(item.generation, State::Running),
		                                             std::memory_order_acq_rel)) {
			continue;
		}
		Evaluate(context, *item.job);
		item.job->state.store(Pack(item.generation, State::Done), std::memory_order_release);
	}
}

void ResourcePrefetcher::Evaluate(WorkerContext& context, Job& job) {
	job.reads.clear();
	job.values.clear();
	job.pages.clear();
	job.reads_ok    = true;
	context.job     = &job;
	context.aborted = false;
	context.Reset();
	const SrtRuntime runtime {
	    .user_data                  = job.user_data,
	    .shader_base                = job.shader_base,
	    .read_memory                = ReadDirect,
	    .userdata                   = &context,
	    .read_specialization_memory = ReadChecked,
	    .workgroup_counts           = job.workgroup_counts,
	};
	const ShaderRecompiler::IR::SpecializationFailureReports quiet(false);
	// As the serial path: with a fallback list, which also bounds indirect image candidates.
	// A failed result is discarded, so the list's contents do not matter.
	context.bindless_pcs.clear();
	job.ok = ShaderRecompiler::IR::MaterializeResources(*job.replicas[context.index], runtime,
	                                                    job.snapshot, job.specialization,
	                                                    &context.bindless_pcs) &&
	         !context.aborted;
	context.job = nullptr;
}

// The serial walker loads these words directly and cannot fail, so an unreadable page means the
// worker cannot reproduce it.
bool ResourcePrefetcher::ReadDirect(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& context = *static_cast<WorkerContext*>(userdata);
	if (!context.Read(address, values)) {
		context.aborted = true;
		return false;
	}
	context.Record(address, values, false, true);
	return true;
}

// The serial reader refuses GPU-written bytes. The worker cannot see GPU-thread state, so it
// reads them anyway and the replay checks the outcome.
bool ResourcePrefetcher::ReadChecked(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto&      context = *static_cast<WorkerContext*>(userdata);
	const bool ok      = !values.empty() && context.Read(address, values);
	context.Record(address, values, true, ok);
	return ok;
}

ResourcePrefetcher::Job* ResourcePrefetcher::Allocate() {
	std::erase_if(m_retired, [this](Job* job) {
		if (StateOf(job->state.load(std::memory_order_acquire)) == State::Running) {
			return false;
		}
		m_free.push_back(job);
		return true;
	});
	if (!m_free.empty()) {
		auto* job = m_free.back();
		m_free.pop_back();
		return job;
	}
	m_jobs.push_back(std::make_unique<Job>());
	m_stats.jobs++;
	return m_jobs.back().get();
}

void ResourcePrefetcher::Retire(Job* job) {
	auto word = job->state.load(std::memory_order_acquire);
	if (StateOf(word) == State::Queued) {
		job->state.compare_exchange_strong(word, Pack(job->generation, State::Cancelled),
		                                   std::memory_order_acq_rel);
	}
	if (StateOf(job->state.load(std::memory_order_acquire)) == State::Running) {
		m_retired.push_back(job);
	} else {
		m_free.push_back(job);
	}
}

void ResourcePrefetcher::Submit(uint64_t draw, const void* key,
                                std::span<const ResourcePlan* const> replicas,
                                std::span<const uint32_t> user_data, uint64_t shader_base,
                                std::span<const uint32_t> workgroup_counts) {
	if (m_workers.empty() || replicas.size() < m_workers.size()) {
		return;
	}
	auto& health = m_health[key];
	if (health.score <= HealthThrottle && ++health.skipped % ThrottleRetry != 0) {
		m_stats.throttled++;
		return;
	}
	auto* job = Allocate();
	job->generation++;
	job->draw     = draw;
	job->key      = key;
	job->consumed = false;
	job->replicas.assign(replicas.begin(), replicas.end());
	job->user_data.assign(user_data.begin(), user_data.end());
	job->shader_base = shader_base;
	job->workgroup_counts.assign(workgroup_counts.begin(), workgroup_counts.end());
	if (m_ops.watch != nullptr) {
		job->watch_generation = m_ops.watch->Generation();
		job->watch_settles    = m_ops.watch->SettleCount();
	}
	job->state.store(Pack(job->generation, State::Queued), std::memory_order_relaxed);
	m_window.push_back(job);
	m_pending.push_back({job, job->generation});
	m_stats.submitted++;
	if (!m_batching || m_pending.size() >= FlushCap) {
		Publish();
	}
}

void ResourcePrefetcher::Flush() {
	m_batching = false;
	Publish();
}

void ResourcePrefetcher::Publish() {
	// Retired while it waited (its record may already hold a later submission).
	std::erase_if(m_pending, [](const QueueItem& item) {
		return item.job->state.load(std::memory_order_relaxed) !=
		       Pack(item.generation, State::Queued);
	});
	if (m_pending.empty()) {
		return;
	}
	const auto queued = static_cast<uint32_t>(m_pending.size());
	{
		std::lock_guard lock(m_queue_mutex);
		m_queue.insert(m_queue.end(), m_pending.begin(), m_pending.end());
		m_queued.fetch_add(queued, std::memory_order_release);
	}
	m_pending.clear();
	m_stats.flushes++;
	m_stats.flushed_jobs += queued;
	const auto sleeping = m_sleeping.load(std::memory_order_seq_cst);
	if (sleeping == 0) {
		return;
	}
	if (queued >= sleeping) {
		m_queue_ready.notify_all();
	} else {
		for (uint32_t i = 0; i < queued; i++) {
			m_queue_ready.notify_one();
		}
	}
}

void ResourcePrefetcher::BeginDraw(uint64_t draw) {
	m_current_draw = draw;
	while (!m_window.empty() && m_window.front()->draw < draw) {
		Retire(m_window.front());
		m_window.pop_front();
	}
}

void ResourcePrefetcher::Clear() {
	for (auto* job: m_window) {
		Retire(job);
	}
	m_window.clear();
	// Every waiting submission is in the window or already retired.
	m_pending.clear();
}

void ResourcePrefetcher::ClearAndWait() {
	Clear();
	for (uint32_t spin = 0;; spin++) {
		std::erase_if(m_retired, [this](Job* job) {
			if (StateOf(job->state.load(std::memory_order_acquire)) == State::Running) {
				return false;
			}
			m_free.push_back(job);
			return true;
		});
		if (m_retired.empty()) {
			return;
		}
		if (spin < 4096) {
			CpuRelax();
		} else {
			std::this_thread::yield();
		}
	}
}

bool ResourcePrefetcher::TryMaterialize(const void* key, const SrtRuntime& runtime,
                                        ResourceSnapshot&       snapshot,
                                        ResourceSpecialization& specialization) {
	// A batch still waiting holds this draw's speculation or later ones: hand it over first.
	Publish();
	Job* job = nullptr;
	for (auto* candidate: m_window) {
		if (candidate->draw != m_current_draw) {
			break;
		}
		if (candidate->key == key && !candidate->consumed) {
			job = candidate;
			break;
		}
	}
	if (job == nullptr) {
		m_stats.unmatched++;
		return false;
	}
	job->consumed = true;
	// The worker read through a strict reader; without one the serial evaluation differs.
	if (job->shader_base != runtime.shader_base || runtime.read_specialization_memory == nullptr ||
	    !std::ranges::equal(job->user_data, runtime.user_data) ||
	    !std::ranges::equal(job->workgroup_counts, runtime.workgroup_counts)) {
		m_stats.mismatched++;
		return false;
	}
	auto word = job->state.load(std::memory_order_acquire);
	if (StateOf(word) == State::Queued &&
	    job->state.compare_exchange_strong(word, Pack(job->generation, State::Cancelled),
	                                       std::memory_order_acq_rel)) {
		m_stats.cancelled++;
		return false;
	}
	for (uint32_t spin = 0; StateOf(word = job->state.load(std::memory_order_acquire)) ==
	                        State::Running;
	     spin++) {
		if (spin < 4096) {
			CpuRelax();
		} else {
			std::this_thread::yield();
		}
	}
	auto& health = m_health[key];
	if (StateOf(word) != State::Done || !job->ok) {
		m_stats.failed++;
		health.score = std::max(health.score - HealthPenalty, HealthMin);
		return false;
	}
	if (IsSettled(*job)) {
		if (m_ops.replay_check && !Validate(*job, runtime)) {
			m_stats.replay_mismatches++;
		}
		m_stats.unreplayed++;
	} else if (!Validate(*job, runtime)) {
		// A dispatch that parked failed the replay on purpose: the job is not stale.
		if (QueuePark::Pending()) {
			return false;
		}
		m_stats.stale++;
		health.score = std::max(health.score - HealthPenalty, HealthMin);
		return false;
	} else {
		Settle(*job);
	}
	snapshot       = job->snapshot;
	specialization = job->specialization;
	health.score   = std::min(health.score + 1, HealthMax);
	m_stats.hits++;
	return true;
}

bool ResourcePrefetcher::IsSettled(const Job& job) const {
	// A failed read's outcome depends on more than the bytes: it is replayed.
	if (m_ops.watch == nullptr || !job.reads_ok ||
	    job.watch_generation != m_ops.watch->Generation()) {
		return false;
	}
	return std::ranges::all_of(job.pages, [this, &job](uint64_t page) {
		return m_ops.watch->IsSettledBy(page, job.watch_settles);
	});
}

// After a replay accepted the job: its pages that hold their final bytes settle.
void ResourcePrefetcher::Settle(const Job& job) {
	if (m_ops.watch == nullptr) {
		return;
	}
	if (const auto generation = m_ops.watch->Generation(); generation != m_unsettled_generation) {
		m_unsettled.clear();
		m_unsettled_generation = generation;
	}
	for (const auto page: job.pages) {
		if (m_ops.watch->IsSettled(page) ||
		    std::ranges::find(m_unsettled, page) != m_unsettled.end()) {
			continue;
		}
		if (m_ops.page_settled(page)) {
			m_ops.watch->Settle(page);
		} else if (m_unsettled.size() < MaxUnsettled) {
			m_unsettled.push_back(page);
		}
	}
}

bool ResourcePrefetcher::Quiescent() const {
	return std::ranges::none_of(m_window, [this](const Job* job) {
		const auto state = StateOf(job->state.load(std::memory_order_acquire));
		return state == State::Running ||
		       (state == State::Queued &&
		        std::ranges::none_of(m_pending, [job](const QueueItem& item) {
			        return item.job == job;
		        }));
	});
}

namespace {

// Most recorded reads are one word: compare without a library call.
bool SameWords(std::span<const uint32_t> a, std::span<const uint32_t> b) {
	if (a.size() != b.size()) {
		return false;
	}
	if (a.size() == 1u) {
		return a[0] == b[0];
	}
	return std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

} // namespace

// Replays the worker's reads in their original order with the draw's own readers. Reads of one
// page are first tried as one covering checked read: bytes that are clean now stay clean while
// the GPU thread only reads, so each read the cover spans would succeed on its own too, and an
// ordinary read of clean bytes returns the same words as the checked one.
bool ResourcePrefetcher::Validate(const Job& job, const SrtRuntime& runtime) {
	// The worker read GPU-written bytes from guest memory, where they are stale; replaying them
	// would read them back. The serial path leaves data-only ones to a GPU copy instead.
	if (runtime.gpu_written != nullptr) {
		uint64_t clean_page = UINT64_MAX;
		for (const auto& read: job.reads) {
			const auto bytes = uint64_t {std::max(read.count, 1u)} * sizeof(uint32_t);
			const auto page  = read.address & ~(PageSize - 1u);
			const bool one_page = ((read.address + bytes - 1u) & ~(PageSize - 1u)) == page;
			if (one_page && page == clean_page) {
				continue;
			}
			if (one_page && !runtime.gpu_written(page, PageSize)) {
				clean_page = page;
				continue;
			}
			if (runtime.gpu_written(read.address, bytes)) {
				return false;
			}
		}
	}
	m_groups.clear();
	m_group_of.assign(job.reads.size(), -1);
	uint32_t group_words = 0;
	for (size_t i = 0; i < job.reads.size(); i++) {
		const auto& read = job.reads[i];
		const auto  end  = read.address + uint64_t {read.count} * sizeof(uint32_t);
		if (!read.ok || read.count == 0u || (read.address & 3u) != 0 ||
		    (read.address & ~(PageSize - 1u)) != ((end - 1u) & ~(PageSize - 1u))) {
			continue;
		}
		const auto page  = read.address & ~(PageSize - 1u);
		auto       group = std::ranges::find(m_groups, page, &ReadGroup::page);
		if (group == m_groups.end()) {
			if (m_groups.size() == MaxGroups) {
				continue;
			}
			m_groups.push_back({.page = page, .begin = read.address, .end = end});
			group = m_groups.end() - 1;
		}
		group->begin = std::min(group->begin, read.address);
		group->end   = std::max(group->end, end);
		m_group_of[i] = static_cast<int32_t>(group - m_groups.begin());
	}
	for (auto& group: m_groups) {
		group.offset = group_words;
		group_words += static_cast<uint32_t>((group.end - group.begin) / sizeof(uint32_t));
	}
	m_group_values.resize(group_words);

	for (size_t i = 0; i < job.reads.size(); i++) {
		const auto& read = job.reads[i];
		// Nearly all reads are the walker's single-word loads.
		if (!read.checked && read.count == 1 && runtime.read_memory == nullptr) {
			uint32_t word = 0;
			std::memcpy(&word, reinterpret_cast<const void*>(read.address), sizeof(word));
			if (word != job.values[read.offset]) {
				return false;
			}
			continue;
		}
		const auto recorded = std::span<const uint32_t>(job.values).subspan(
		    read.ok ? read.offset : 0u, read.ok ? read.count : 0u);
		if (m_group_of[i] >= 0) {
			auto&      group = m_groups[m_group_of[i]];
			const auto words =
			    std::span(m_group_values)
			        .subspan(group.offset, (group.end - group.begin) / sizeof(uint32_t));
			if (group.state == GroupUnchecked) {
				group.state = runtime.read_specialization_memory(runtime.userdata, group.begin, words)
				                  ? GroupClean
				                  : GroupDirty;
			}
			if (group.state == GroupClean) {
				const auto first = (read.address - group.begin) / sizeof(uint32_t);
				if (!SameWords(words.subspan(first, read.count), recorded)) {
					return false;
				}
				continue;
			}
		}
		if (m_scratch.size() < read.count) {
			m_scratch.resize(read.count);
		}
		const auto current = std::span(m_scratch).first(read.count);
		if (!read.checked) {
			if (runtime.read_memory != nullptr) {
				if (!runtime.read_memory(runtime.userdata, read.address, current)) {
					return false;
				}
			} else {
				std::memcpy(current.data(), reinterpret_cast<const void*>(read.address),
				            current.size_bytes());
			}
			if (!SameWords(current, recorded)) {
				return false;
			}
			continue;
		}
		const bool ok = runtime.read_specialization_memory != nullptr && !current.empty() &&
		                runtime.read_specialization_memory(runtime.userdata, read.address, current);
		if (ok != read.ok || (ok && !SameWords(current, recorded))) {
			return false;
		}
	}
	return true;
}

} // namespace Libs::Graphics
