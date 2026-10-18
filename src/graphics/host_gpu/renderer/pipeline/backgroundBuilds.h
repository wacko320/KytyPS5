#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BACKGROUNDBUILDS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BACKGROUNDBUILDS_H_

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

// Lowers the calling thread's scheduling priority, for work nothing waits on yet.
inline void LowerBackgroundBuildPriority(int nice = 5) {
#if defined(__linux__)
	// Linux applies nice values per thread.
	setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), nice);
#else
	(void)nice;
#endif
}

// Builds that a later request is expected to need, started ahead on a few worker threads.
// Each build has a key that identifies its result exactly; the requester takes the result by
// key: a finished build is handed over, a running one is waited for, and a queued one is taken
// off the queue and built on the requester's thread. Results nobody takes are disposed of when
// the queue is full (oldest first) and at shutdown. Thread-safe.
template <typename Result>
class KeyedBuildQueue {
public:
	using Build   = std::function<Result()>;
	using Dispose = std::function<void(Result&)>;

	// At most `max_threads` workers (started on demand, at nice value `nice` on Linux) and
	// `max_entries` builds queued, running or finished but not taken.
	KeyedBuildQueue(uint32_t max_threads, uint32_t max_entries, Dispose dispose, int nice = 5)
	    : m_max_threads(std::max(max_threads, 1u)), m_max_entries(std::max(max_entries, 1u)),
	      m_nice(nice), m_dispose(std::move(dispose)) {}
	~KeyedBuildQueue() { Shutdown(); }
	KeyedBuildQueue(const KeyedBuildQueue&)            = delete;
	KeyedBuildQueue& operator=(const KeyedBuildQueue&) = delete;

	// False when the key is already known, the queue is full of unfinished builds, or the queue
	// shut down: the build is not run.
	bool Submit(uint64_t key, Build build) {
		std::unique_lock lock(m_mutex);
		if (m_stopped || Find(key) != m_entries.end()) return false;
		if (m_entries.size() >= m_max_entries && !EvictOldestFinished(lock)) return false;
		m_entries.push_back({.key = key, .build = std::move(build)});
		m_queue.push_back(key);
		if (m_threads.size() < m_max_threads && m_idle == 0) {
			m_threads.emplace_back([this] { Work(); });
		}
		m_work.notify_one();
		return true;
	}

	[[nodiscard]] bool Contains(uint64_t key) const {
		std::lock_guard lock(m_mutex);
		return Find(key) != m_entries.end();
	}

	// The build's result, waiting for it or building it here when needed; nullopt when no build
	// has this key.
	std::optional<Result> Take(uint64_t key) { return Take(key, true); }

	// As Take, but a build no worker started yet is dropped instead (nullopt): for builds the
	// requester has a cheaper way to make itself.
	std::optional<Result> TakeStarted(uint64_t key) { return Take(key, false); }

	// Until no build is queued or running.
	void WaitIdle() {
		std::unique_lock lock(m_mutex);
		m_done.wait(lock, [&] {
			return m_stopped || std::ranges::none_of(m_entries, [](const Entry& entry) {
				       return entry.state != State::Finished;
			       });
		});
	}

	// Stops the workers after their current build, drops queued builds and disposes of results
	// nobody took. Later submissions are refused. Any thread.
	void Shutdown() {
		std::vector<std::thread> threads;
		{
			std::lock_guard lock(m_mutex);
			m_stopped = true;
			m_queue.clear();
			threads.swap(m_threads);
		}
		m_work.notify_all();
		for (auto& thread: threads) thread.join();
		std::lock_guard lock(m_mutex);
		// A requester waiting in Take (another thread) still takes its entry.
		std::erase_if(m_entries, [&](Entry& entry) {
			if (entry.claimed) return false;
			if (entry.state == State::Finished && entry.result && m_dispose) {
				m_dispose(*entry.result);
				m_stats.disposed++;
			}
			return true;
		});
	}

	struct Stats {
		uint64_t built    = 0; // by a worker
		uint64_t taken    = 0; // handed over finished or after a wait
		uint64_t stolen   = 0; // still queued when taken: built by the requester
		uint64_t disposed = 0; // finished but never taken
		uint64_t dropped  = 0; // still queued when TakeStarted asked for it
	};
	[[nodiscard]] Stats GetStats() const {
		std::lock_guard lock(m_mutex);
		return m_stats;
	}

private:
	std::optional<Result> Take(uint64_t key, bool build_queued) {
		std::unique_lock lock(m_mutex);
		auto entry = Find(key);
		if (entry == m_entries.end()) return std::nullopt;
		if (entry->state == State::Queued) {
			std::erase(m_queue, key);
			auto build = std::move(entry->build);
			m_entries.erase(entry);
			if (!build_queued) {
				m_stats.dropped++;
				return std::nullopt;
			}
			m_stats.stolen++;
			lock.unlock();
			return build();
		}
		entry->claimed = true; // not evicted while this waits
		m_done.wait(lock, [&] { return entry->state == State::Finished; });
		auto result = std::move(entry->result);
		m_entries.erase(entry);
		m_stats.taken++;
		return result;
	}

	enum class State { Queued, Running, Finished };
	struct Entry {
		uint64_t              key = 0;
		Build                 build;
		State                 state   = State::Queued;
		bool                  claimed = false;
		std::optional<Result> result;
	};

	typename std::list<Entry>::iterator Find(uint64_t key) {
		return std::ranges::find(m_entries, key, &Entry::key);
	}
	typename std::list<Entry>::const_iterator Find(uint64_t key) const {
		return std::ranges::find(m_entries, key, &Entry::key);
	}

	bool EvictOldestFinished(std::unique_lock<std::mutex>&) {
		const auto oldest = std::ranges::find_if(m_entries, [](const Entry& entry) {
			return entry.state == State::Finished && !entry.claimed;
		});
		if (oldest == m_entries.end()) return false;
		if (oldest->result && m_dispose) m_dispose(*oldest->result);
		m_entries.erase(oldest);
		m_stats.disposed++;
		return true;
	}

	void Work() {
		LowerBackgroundBuildPriority(m_nice);
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_idle++;
			m_work.wait(lock, [&] { return m_stopped || !m_queue.empty(); });
			m_idle--;
			if (m_stopped) return;
			const auto key = m_queue.front();
			m_queue.pop_front();
			auto entry = Find(key);
			entry->state = State::Running;
			auto build   = std::move(entry->build);
			lock.unlock();
			auto result = build();
			lock.lock();
			// Entries are erased only once finished or while queued: this one is still there.
			entry->result = std::move(result);
			entry->state  = State::Finished;
			m_stats.built++;
			m_done.notify_all();
		}
	}

	const uint32_t           m_max_threads;
	const uint32_t           m_max_entries;
	const int                m_nice;
	Dispose                  m_dispose;
	mutable std::mutex       m_mutex;
	std::condition_variable  m_work;
	std::condition_variable  m_done;
	std::list<Entry>         m_entries;
	std::deque<uint64_t>     m_queue;
	std::vector<std::thread> m_threads;
	uint32_t                 m_idle    = 0;
	bool                     m_stopped = false;
	Stats                    m_stats;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BACKGROUNDBUILDS_H_
