#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// Labelled spans of a command buffer whose GPU time is measured (frameProfiler.h counters
// GpuRegionTime*, in this order).
enum class GpuRegion : uint32_t { Compactor, Check, FlatCopy, DeltaUpload, Count };

// GPU execution time from timestamp queries: one query pool per command buffer, written at its
// top and bottom and around labelled regions, read once its tick is known complete. Always on
// when the device has timestamps on the queue (timestampComputeAndGraphics and nonzero valid
// bits); otherwise every call is a no-op and the statistics stay 0. Pools are reset on the host
// after their results are read (hostQueryReset), or, without it, in the command buffer at Begin.
//
// Busy time runs from a command buffer's top-of-pipe timestamp to its bottom one: a command
// buffer that blocks on an event or a wait stage counts as busy while it waits (an upper bound).
//
// Results are read by the recording thread when it begins a command buffer, for the ticks the
// GPU is already known to have completed: never a wait, and no completion operation that the
// scheduler's priority waits would have to wait for. Single recording thread.
class GpuTimestamps {
public:
	static constexpr uint32_t RegionCount = static_cast<uint32_t>(GpuRegion::Count);
	// Region spans recorded per command buffer; later ones are counted as unmeasured.
	static constexpr uint32_t MaxRegionSpans = 511;

	struct Stats {
		uint64_t command_buffers = 0; // timed at completion
		uint64_t untimed         = 0; // no pool was free
		uint64_t busy_ns         = 0; // union of the command buffers' intervals
		uint64_t unmeasured      = 0; // region spans past MaxRegionSpans
		std::array<uint64_t, RegionCount> region_ns {};
	};

	// A completed command buffer: GPU timestamps of its top and bottom, in nanoseconds.
	struct Interval {
		uint64_t tick     = 0;
		uint64_t begin_ns = 0;
		uint64_t end_ns   = 0;
	};

	// `present`: the presentation scheduler's, whose busy time is counted apart (both schedulers
	// submit to one queue, and their command buffers are not merged into one union).
	GpuTimestamps(GraphicContext& graphics, bool present);
	~GpuTimestamps();
	KYTY_CLASS_NO_COPY(GpuTimestamps);

	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }

	// The recording thread, in command-buffer order: Begin after vkBeginCommandBuffer, End before
	// vkEndCommandBuffer, outside a render pass.
	void Begin(vk::CommandBuffer command);
	void BeginRegion(vk::CommandBuffer command, GpuRegion region);
	void EndRegion(vk::CommandBuffer command, GpuRegion region);
	// Ends the command buffer's timing; pass the token to Submitted with its tick (0: untimed).
	[[nodiscard]] uint32_t End(vk::CommandBuffer command);
	void                   Submitted(uint32_t token, uint64_t tick);
	// Reads the command buffers of ticks up to `completed_tick`, in tick order: adds their time to
	// the statistics and the frame profiler, and releases their pools. Begin does it too.
	void Poll(uint64_t completed_tick);

	[[nodiscard]] Stats GetStats() const;
	// The most recent completed command buffers, oldest first.
	[[nodiscard]] std::vector<Interval> RecentIntervals() const;

private:
	static constexpr uint32_t QueryCount  = 2 + 2 * MaxRegionSpans;
	static constexpr uint32_t MaxPools    = 256;
	static constexpr uint32_t RecentCount = 16;

	struct Pool {
		uint32_t                               token = 0;
		vk::QueryPool                          pool  = nullptr;
		// Completed region spans: span k uses queries 2 + 2k and 3 + 2k.
		uint32_t                               spans       = 0;
		std::array<GpuRegion, MaxRegionSpans>  regions     {};
		bool                                   region_open = false;
	};

	struct InFlight {
		uint64_t tick  = 0;
		uint32_t token = 0;
	};

	void Complete(const InFlight& command_buffer);

	GraphicContext& m_graphics;
	const bool      m_present;
	bool            m_enabled    = false;
	bool            m_host_reset = false;
	double          m_period_ns  = 1.0;
	uint64_t        m_valid_mask = 0;
	// Recording thread only: the pools (a pool's index + 1 is its token), the free ones, the
	// submitted ones in tick order, the recording command buffer's pool (null: untimed) and the
	// busy-time union.
	std::vector<std::unique_ptr<Pool>> m_pools;
	std::vector<uint32_t>              m_free;
	std::deque<InFlight>               m_in_flight;
	Pool*                              m_current     = nullptr;
	uint64_t                           m_last_end_ns = 0;
	mutable std::mutex                 m_recent_mutex;
	std::vector<Interval>              m_recent;

	std::atomic<uint64_t> m_command_buffers {0};
	std::atomic<uint64_t> m_untimed {0};
	std::atomic<uint64_t> m_busy_ns {0};
	std::atomic<uint64_t> m_unmeasured {0};
	std::array<std::atomic<uint64_t>, RegionCount> m_region_ns {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
