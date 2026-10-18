#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>
#include <vector>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// Whether the page holding vaddr is read-protected for tracking. Unlocked: a hint that can
	// be stale by the time the caller acts on it.
	[[nodiscard]] bool IsReadWatched(uint64_t vaddr) const noexcept;
	// Whether a write to the page holding vaddr would fault for tracking. Unlocked hint.
	[[nodiscard]] bool IsWriteWatched(uint64_t vaddr) const noexcept;
	// Lazy release, for a hand-back of pages the GPU writes again soon: while set on this
	// thread, removed watchers leave the pages' protection as it is, noted, so the next watcher
	// to ask for it again needs no protection change. A fault on such a page applies what its
	// watchers ask (ApplyDeferred) and the access is retried. Returns the previous setting.
	static bool SetLazyRelease(bool lazy) noexcept;
	// Any thread, from a fault: whether the page's protection was left and is now applied.
	[[nodiscard]] bool ApplyDeferred(uint64_t vaddr);

	// Deferred write protection, for an upload walk that copies its pages after protecting them:
	// while a sink is set on this thread, the ranges its new write watches would protect are
	// appended to it instead, and the caller has them applied later (ApplyProtection), before
	// it copies their bytes. A CPU write in between lands before the copy. Any other change to
	// the watchers of such a page applies its protection at once (a watch that needs its faults,
	// such as an image's or a BVH's, never starts on a writable page). Returns the previous sink.
	static std::vector<GuestRange>* SetDeferredWriteWatches(std::vector<GuestRange>* sink) noexcept;
	// Any thread: protects the range's pages a deferred walk left, as their watchers now ask.
	void ApplyProtection(uint64_t vaddr, uint64_t size);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
