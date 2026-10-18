#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include "common/assert.h"
#include "common/frameProfiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>

namespace Libs::Graphics {

GpuTimestamps::GpuTimestamps(GraphicContext& graphics, bool present)
    : m_graphics(graphics), m_present(present) {
	const auto& limits = graphics.physical_device_properties.limits;
	if (graphics.physical_device == nullptr || graphics.device == nullptr ||
	    !limits.timestampComputeAndGraphics || limits.timestampPeriod <= 0.0f) {
		return;
	}
	uint32_t family_count = 0;
	graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
	if (graphics.queue_family >= families.size()) {
		return;
	}
	const auto valid_bits = families[graphics.queue_family].timestampValidBits;
	if (valid_bits == 0) {
		return;
	}
	m_valid_mask = valid_bits >= 64 ? ~uint64_t {0} : (uint64_t {1} << valid_bits) - 1u;
	m_period_ns  = limits.timestampPeriod;
	m_enabled    = true;
	m_host_reset = graphics.host_query_reset_enabled;
	m_recent.reserve(RecentCount);
}

GpuTimestamps::~GpuTimestamps() {
	for (const auto& pool: m_pools) {
		m_graphics.device.destroyQueryPool(pool->pool, nullptr);
	}
}

void GpuTimestamps::Begin(CommandRecorder command) {
	EXIT_IF(m_current != nullptr);
	if (!m_enabled) {
		return;
	}
	if (!m_free.empty()) {
		m_current = m_pools[m_free.back() - 1].get();
		m_free.pop_back();
	} else if (m_pools.size() < MaxPools) {
		vk::QueryPoolCreateInfo create {};
		create.queryType   = vk::QueryType::eTimestamp;
		create.queryCount  = QueryCount;
		vk::QueryPool pool = nullptr;
		if (m_graphics.device.createQueryPool(&create, nullptr, &pool) == vk::Result::eSuccess) {
			if (m_host_reset) {
				m_graphics.device.resetQueryPool(pool, 0, QueryCount);
			}
			m_pools.push_back(std::make_unique<Pool>());
			m_current        = m_pools.back().get();
			m_current->token = static_cast<uint32_t>(m_pools.size());
			m_current->pool  = pool;
		}
	}
	if (m_current == nullptr) {
		m_untimed.fetch_add(1, std::memory_order_relaxed);
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::GpuCommandBuffersUntimed);
		return;
	}
	m_current->spans       = 0;
	m_current->region_open = false;
	if (!m_host_reset) {
		command.resetQueryPool(m_current->pool, 0, QueryCount);
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, m_current->pool, 0);
}

void GpuTimestamps::BeginRegion(CommandRecorder command, GpuRegion region) {
	if (m_current == nullptr) {
		return;
	}
	if (m_current->region_open || m_current->spans == MaxRegionSpans) {
		m_unmeasured.fetch_add(1, std::memory_order_relaxed);
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::GpuRegionSpansUnmeasured);
		return;
	}
	// After the work recorded before it: the span is the region's own work.
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_current->pool,
	                       2 + 2 * m_current->spans);
	m_current->regions[m_current->spans] = region;
	m_current->region_open               = true;
}

void GpuTimestamps::EndRegion(CommandRecorder command, GpuRegion region) {
	if (m_current == nullptr || !m_current->region_open ||
	    m_current->regions[m_current->spans] != region) {
		return;
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_current->pool,
	                       3 + 2 * m_current->spans);
	m_current->spans++;
	m_current->region_open = false;
}

uint32_t GpuTimestamps::End(CommandRecorder command) {
	if (m_current == nullptr) {
		return 0;
	}
	command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_current->pool, 1);
	const auto token = m_current->token;
	m_current        = nullptr;
	return token;
}

void GpuTimestamps::Submitted(uint32_t token, uint64_t tick) {
	if (token != 0) {
		m_in_flight.push_back({tick, token});
	}
}

void GpuTimestamps::Poll(uint64_t completed_tick) {
	while (!m_in_flight.empty() && m_in_flight.front().tick <= completed_tick) {
		Complete(m_in_flight.front());
		m_in_flight.pop_front();
	}
}

void GpuTimestamps::Complete(const InFlight& command_buffer) {
	auto&                            pool  = *m_pools[command_buffer.token - 1];
	std::array<uint64_t, QueryCount> values {};
	const auto                       count = 2 + 2 * pool.spans;
	const auto result = m_graphics.device.getQueryPoolResults(
	    pool.pool, 0, count, count * sizeof(uint64_t), values.data(), sizeof(uint64_t),
	    vk::QueryResultFlagBits::e64);
	if (m_host_reset) {
		m_graphics.device.resetQueryPool(pool.pool, 0, count);
	}
	m_free.push_back(command_buffer.token);
	const auto ns = [&](uint32_t query) {
		const auto ticks = static_cast<double>(values[query] & m_valid_mask);
		return static_cast<uint64_t>(ticks * m_period_ns);
	};
	const auto begin = ns(0);
	const auto end   = ns(1);
	if (result != vk::Result::eSuccess || end < begin) {
		m_untimed.fetch_add(1, std::memory_order_relaxed);
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::GpuCommandBuffersUntimed);
		return;
	}
	// Command buffers can overlap on the GPU: busy time is their union, in tick order.
	const auto from    = std::max(begin, m_last_end_ns);
	const auto busy_ns = end > from ? end - from : 0;
	m_last_end_ns      = std::max(m_last_end_ns, end);
	{
		std::lock_guard lock(m_recent_mutex);
		if (m_recent.size() == RecentCount) {
			m_recent.erase(m_recent.begin());
		}
		m_recent.push_back({command_buffer.tick, begin, end});
	}
	using Common::FrameProfiler::Counter;
	static_assert(static_cast<uint32_t>(Counter::GpuRegionTimeDeltaUpload) ==
	              static_cast<uint32_t>(Counter::GpuRegionTimeCompactor) + RegionCount - 1);
	m_command_buffers.fetch_add(1, std::memory_order_relaxed);
	m_busy_ns.fetch_add(busy_ns, std::memory_order_relaxed);
	Common::FrameProfiler::Add(Counter::GpuCommandBuffersTimed);
	Common::FrameProfiler::Add(m_present ? Counter::GpuPresentBusyTime : Counter::GpuBusyTime,
	                           busy_ns);
	for (uint32_t span = 0; span < pool.spans; span++) {
		const auto span_begin = ns(2 + 2 * span);
		const auto span_end   = ns(3 + 2 * span);
		const auto region     = static_cast<uint32_t>(pool.regions[span]);
		const auto span_ns    = span_end >= span_begin ? span_end - span_begin : 0;
		m_region_ns[region].fetch_add(span_ns, std::memory_order_relaxed);
		Common::FrameProfiler::Add(
		    static_cast<Counter>(static_cast<uint32_t>(Counter::GpuRegionTimeCompactor) + region),
		    span_ns);
	}
}

GpuTimestamps::Stats GpuTimestamps::GetStats() const {
	Stats stats;
	stats.command_buffers = m_command_buffers.load(std::memory_order_relaxed);
	stats.untimed         = m_untimed.load(std::memory_order_relaxed);
	stats.busy_ns         = m_busy_ns.load(std::memory_order_relaxed);
	stats.unmeasured      = m_unmeasured.load(std::memory_order_relaxed);
	for (uint32_t region = 0; region < RegionCount; region++) {
		stats.region_ns[region] = m_region_ns[region].load(std::memory_order_relaxed);
	}
	return stats;
}

std::vector<GpuTimestamps::Interval> GpuTimestamps::RecentIntervals() const {
	std::lock_guard lock(m_recent_mutex);
	return m_recent;
}

} // namespace Libs::Graphics
