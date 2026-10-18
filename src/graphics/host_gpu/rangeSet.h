#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>

namespace Libs::Graphics {

class RangeSet final {
public:
	void Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		uint64_t begin = address;
		uint64_t last  = end;
		while (it != m_ranges.end() && it->first <= last) {
			begin = std::min(begin, it->first);
			last  = std::max(last, it->second);
			it    = m_ranges.erase(it);
		}
		m_ranges.emplace(begin, last);
	}

	void Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
	}

	void Clear() { m_ranges.clear(); }

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
};

// Disjoint byte ranges, each with a value (a tick) a later Assign over it replaces.
class RangeTickMap final {
public:
	void Assign(uint64_t begin, uint64_t end, uint64_t tick) {
		if (begin >= end) {
			return;
		}
		auto it = m_ranges.lower_bound(begin);
		if (it != m_ranges.begin() && std::prev(it)->second.end > begin) {
			it = std::prev(it);
		}
		while (it != m_ranges.end() && it->first < end) {
			const auto first = it->first;
			const auto entry = it->second;
			it               = m_ranges.erase(it);
			if (first < begin) {
				m_ranges.emplace(first, Entry {begin, entry.tick});
			}
			if (entry.end > end) {
				m_ranges.emplace(end, Entry {entry.end, entry.tick});
				break;
			}
		}
		m_ranges.emplace(begin, Entry {end, tick});
	}

	// The largest value over the ranges intersecting [begin, end); 0 for none.
	[[nodiscard]] uint64_t Max(uint64_t begin, uint64_t end) const {
		uint64_t tick = 0;
		ForEachInRange(begin, end, [&](uint64_t, uint64_t, uint64_t value) {
			tick = std::max(tick, value);
		});
		return tick;
	}

	// Drops the ranges whose value is at most `tick`: a range then reads as 0.
	void EraseUpTo(uint64_t tick) {
		std::erase_if(m_ranges, [tick](const auto& range) { return range.second.tick <= tick; });
	}

	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }

	template <typename Func>
	void ForEachInRange(uint64_t begin, uint64_t end, Func&& func) const {
		auto it = m_ranges.upper_bound(begin);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto first = std::max(begin, it->first);
			const auto last  = std::min(end, it->second.end);
			if (first < last) {
				func(first, last, it->second.tick);
			}
		}
	}

private:
	struct Entry {
		uint64_t end  = 0;
		uint64_t tick = 0;
	};
	std::map<uint64_t, Entry> m_ranges;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
