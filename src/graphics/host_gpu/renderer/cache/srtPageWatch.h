#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SRTPAGEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SRTPAGEWATCH_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

// The 16 KiB guest pages whose bytes a prefetched resource evaluation may use without replaying
// its reads (ResourcePrefetcher).
//
// A page is settled once the GPU thread, after a replay that read it, found that guest memory
// holds its final bytes: nothing over it is GPU-written, downloading or waiting to be published,
// no GPU-written image covers it, and no other guest address shows its backing.
// Whatever may change a page's bytes from outside the guest's CPU threads, or make the GPU own
// them, unsettles it first: command-processor stores, write bindings, downloads, GPU-written
// images, mappings. Pages are watched by guest address: an alias of the same backing would not be
// unsettled with it, so such pages never settle.
//
// Each settle takes the next settle count. An evaluation submitted at settle count C and at the
// current generation, whose pages are all still settled with counts up to C, read what a replay
// would read now: its pages were settled throughout, and a change since would have unsettled
// them. The generation moves at each coherence epoch (CoherenceEpoch::Acquire): CPU writes before
// it are not reported, and race the GPU work on hardware too.
class SrtPageWatch {
public:
	static constexpr uint32_t PageBits = 14;
	static constexpr uint64_t PageSize = uint64_t {1} << PageBits;

	SrtPageWatch() = default;
	~SrtPageWatch() {
		for (auto& leaf: m_leaves) {
			delete leaf.load(std::memory_order_relaxed);
		}
	}
	SrtPageWatch(const SrtPageWatch&)            = delete;
	SrtPageWatch& operator=(const SrtPageWatch&) = delete;

	// The one the caches report guest memory changes to.
	static SrtPageWatch& Guest() {
		static SrtPageWatch watch;
		return watch;
	}

	// Any thread.
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}
	void Advance() noexcept { m_generation.fetch_add(1, std::memory_order_acq_rel); }

	// The range's bytes may change, or stop being final in guest memory. GPU thread, or a range
	// the GPU thread cannot settle meanwhile on bytes other than its final ones (a new mapping):
	// Settle checks and sets without a lock.
	void Unsettle(uint64_t vaddr, uint64_t size) noexcept {
		if (size == 0 || vaddr >= AddressLimit) {
			return;
		}
		const auto first = vaddr >> PageBits;
		const auto last  = (std::min(AddressLimit - vaddr, size) - 1u + vaddr) >> PageBits;
		for (auto page = first; page <= last;) {
			const auto leaf_end = ((page >> LeafBits) + 1u) << LeafBits;
			const auto end      = std::min(last + 1u, leaf_end);
			auto*      leaf     = m_leaves[page >> LeafBits].load(std::memory_order_acquire);
			for (; leaf != nullptr && page < end;) {
				const auto bit   = page & (WordBits - 1u);
				const auto count = std::min<uint64_t>(WordBits - bit, end - page);
				const auto mask =
				    (count == WordBits ? ~uint64_t {0} : ((uint64_t {1} << count) - 1u)) << bit;
				auto& word = leaf->words[(page & (LeafPages - 1u)) / WordBits];
				if ((word.load(std::memory_order_relaxed) & mask) != 0) {
					word.fetch_and(~mask, std::memory_order_acq_rel);
				}
				page += count;
			}
			page = end;
		}
	}

	// GPU thread: the settles so far.
	[[nodiscard]] uint64_t SettleCount() const noexcept { return m_settles; }
	// GPU thread: the 16 KiB page at `page` (aligned) is settled.
	[[nodiscard]] bool IsSettled(uint64_t page) const noexcept {
		return IsSettledBy(page, UINT64_MAX);
	}
	// GPU thread: the page is settled, and has been since settle count `count` or earlier.
	[[nodiscard]] bool IsSettledBy(uint64_t page, uint64_t count) const noexcept {
		if (page >= AddressLimit) {
			return false;
		}
		const auto  index = page >> PageBits;
		const auto* leaf  = m_leaves[index >> LeafBits].load(std::memory_order_acquire);
		if (leaf == nullptr) {
			return false;
		}
		const auto local = index & (LeafPages - 1u);
		const auto word  = leaf->words[local / WordBits].load(std::memory_order_acquire);
		return ((word >> (local & (WordBits - 1u))) & 1u) != 0 && leaf->stamps[local] <= count;
	}
	// GPU thread: the caller found the page's bytes final in guest memory.
	void Settle(uint64_t page) {
		if (page >= AddressLimit) {
			return;
		}
		const auto index = page >> PageBits;
		auto&      slot  = m_leaves[index >> LeafBits];
		auto*      leaf  = slot.load(std::memory_order_acquire);
		if (leaf == nullptr) {
			leaf = new Leaf;
			slot.store(leaf, std::memory_order_release);
		}
		const auto local = index & (LeafPages - 1u);
		const auto bit   = uint64_t {1} << (local & (WordBits - 1u));
		auto&      word  = leaf->words[local / WordBits];
		if ((word.load(std::memory_order_relaxed) & bit) == 0) {
			leaf->stamps[local] = ++m_settles;
			word.fetch_or(bit, std::memory_order_acq_rel);
		}
	}

private:
	// 48-bit addresses: 2^14 leaves of 2^20 pages (16 GiB) each, made on first use. A page's
	// settle count is written before its bit is set and read only while the bit is set, so the
	// counts are left uninitialized: only the pages that settle touch their memory.
	static constexpr uint64_t AddressLimit = uint64_t {1} << 48u;
	static constexpr uint32_t LeafBits     = 20;
	static constexpr uint64_t LeafPages    = uint64_t {1} << LeafBits;
	static constexpr uint64_t WordBits     = 64;
	static constexpr size_t   LeafCount    = size_t {1} << (48u - PageBits - LeafBits);

	struct Leaf {
		std::array<std::atomic<uint64_t>, LeafPages / WordBits> words {};
		std::unique_ptr<uint64_t[]> stamps = std::make_unique_for_overwrite<uint64_t[]>(LeafPages);
	};

	std::array<std::atomic<Leaf*>, LeafCount> m_leaves {};
	std::atomic<uint64_t>                     m_generation {1};
	uint64_t                                  m_settles = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_SRTPAGEWATCH_H_
