// Measures the tracker work behind BufferCache::SynchronizeBuffer and the ObtainBuffer queries
// for repeated binds of the same ranges, the pattern that dominates per-draw setup. Page
// protection is stubbed out: steady-state rebinds change no protection, and the numbers should
// show tracker bookkeeping rather than mprotect.

#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t, uint64_t, Common::VirtualMemory::Mode) {
	return true;
}

} // namespace Libs::LibKernel::Memory

namespace {

using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::TRACKER_REGION_SIZE;

constexpr uint64_t MiB  = 1024ull * 1024ull;
constexpr uint64_t Base = 0x100000000ull;

uint64_t g_sink = 0;

void Sync(MemoryTracker& tracker, uint64_t vaddr, uint64_t size, bool is_written) {
	tracker.ForEachUploadRange(
	    vaddr, size, is_written, [](uint64_t, uint64_t bytes) noexcept { g_sink += bytes; },
	    []() noexcept {});
}

double Measure(const char* name, uint64_t iterations, const std::function<void()>& body) {
	body();
	const auto start = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < iterations; i++) {
		body();
	}
	const auto elapsed = std::chrono::steady_clock::now() - start;
	const auto ns =
	    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) /
	    static_cast<double>(iterations);
	std::printf("%-58s %10.1f ns/op\n", name, ns);
	return ns;
}

} // namespace

int main() {
	PageManager   page_manager;
	MemoryTracker tracker(page_manager);

	// Distinct ranges so the cases do not share state: a small constant-like binding, a
	// mid-size one, and the huge whole-buffer RW bindings seen in Rift Apart.
	const uint64_t small_w   = Base;                       // 64 KiB written
	const uint64_t small_r   = Base + 8 * MiB;             // 64 KiB read
	const uint64_t mid_w     = Base + 16 * MiB;            // 25 MiB written
	const uint64_t huge_w    = Base + 64 * MiB;            // 232 MiB written
	const uint64_t huge_r    = Base + 320 * MiB;           // 232 MiB read
	const uint64_t giant_w   = Base + 1024 * MiB;          // 3 GiB written
	const uint64_t tiny_r    = Base + 12 * MiB;            // 16 KiB read, ObtainBuffer queries
	const uint64_t other     = Base + 10 * MiB;            // CPU writes elsewhere
	const uint64_t other_gpu = Base + 11 * MiB;            // readback elsewhere

	Measure("written sync, 64 KiB", 200000, [&] { Sync(tracker, small_w, 64 * 1024, true); });
	Measure("read sync, 64 KiB (clean)", 200000, [&] { Sync(tracker, small_r, 64 * 1024, false); });
	Measure("written sync, 25 MiB", 20000, [&] { Sync(tracker, mid_w, 25 * MiB, true); });
	Measure("written sync, 232 MiB", 5000, [&] { Sync(tracker, huge_w, 232 * MiB, true); });
	Measure("read sync, 232 MiB (clean)", 5000, [&] { Sync(tracker, huge_r, 232 * MiB, false); });
	Measure("written sync, 3 GiB", 1000, [&] { Sync(tracker, giant_w, 3072 * MiB, true); });
	Measure("GPU+CPU modified queries, 16 KiB (clean)", 200000, [&] {
		g_sink += tracker.IsRegionGpuModified(tiny_r, 16 * 1024) ? 1 : 0;
		g_sink += tracker.IsRegionCpuModified(tiny_r, 16 * 1024) ? 1 : 0;
	});

	// Churn: another page is CPU-written (as a write fault would) between binds, or a readback
	// elsewhere clears GPU ownership. Shows the cost when the global state keeps moving.
	Sync(tracker, other, TRACKER_PAGE_SIZE, false);
	Measure("(churn alone: CPU write elsewhere + its sync)", 200000, [&] {
		tracker.InvalidateRegion(other, TRACKER_PAGE_SIZE, [] {});
		Sync(tracker, other, TRACKER_PAGE_SIZE, false);
	});
	Measure("(churn alone: readback elsewhere + its sync)", 200000, [&] {
		Sync(tracker, other_gpu, TRACKER_PAGE_SIZE, true);
		tracker.UnmarkRegionAsGpuModified(other_gpu, TRACKER_PAGE_SIZE);
	});
	Measure("written sync, 232 MiB, CPU write elsewhere each bind", 5000, [&] {
		tracker.InvalidateRegion(other, TRACKER_PAGE_SIZE, [] {});
		Sync(tracker, other, TRACKER_PAGE_SIZE, false);
		Sync(tracker, huge_w, 232 * MiB, true);
	});
	Measure("read sync, 64 KiB, CPU write elsewhere each bind", 200000, [&] {
		tracker.InvalidateRegion(other, TRACKER_PAGE_SIZE, [] {});
		Sync(tracker, other, TRACKER_PAGE_SIZE, false);
		Sync(tracker, small_r, 64 * 1024, false);
	});
	Measure("written sync, 232 MiB, readback elsewhere each bind", 5000, [&] {
		Sync(tracker, other_gpu, TRACKER_PAGE_SIZE, true);
		tracker.UnmarkRegionAsGpuModified(other_gpu, TRACKER_PAGE_SIZE);
		Sync(tracker, huge_w, 232 * MiB, true);
	});
	Measure("written sync, 64 KiB, readback elsewhere each bind", 200000, [&] {
		Sync(tracker, other_gpu, TRACKER_PAGE_SIZE, true);
		tracker.UnmarkRegionAsGpuModified(other_gpu, TRACKER_PAGE_SIZE);
		Sync(tracker, small_w, 64 * 1024, true);
	});

	// BufferCache::ObtainBuffer records every writable binding in its exact GPU-dirty set.
	RangeSet dirty;
	for (uint64_t i = 0; i < 256; i++) {
		dirty.Add(Base + i * 2 * MiB, MiB);
	}
	Measure("RangeSet::Add of an already-dirty range", 200000,
	        [&] { dirty.Add(Base + 100 * 2 * MiB + 4096, 64 * 1024); });
	Measure("RangeSet::Contains of an already-dirty range", 200000,
	        [&] { g_sink += dirty.Contains(Base + 100 * 2 * MiB + 4096, 64 * 1024) ? 1 : 0; });

	std::printf("(sink %llu)\n", static_cast<unsigned long long>(g_sink));
	// The page manager refuses to die with pages still watched; the benchmark leaves them so.
	std::fflush(stdout);
	std::_Exit(0);
}
