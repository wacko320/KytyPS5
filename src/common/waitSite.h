#ifndef EMULATOR_SRC_COMMON_WAITSITE_H_
#define EMULATOR_SRC_COMMON_WAITSITE_H_

// The frame profiler's wait sites, apart from frameProfiler.h so that the wait APIs can take one
// without pulling in the profiler.

#include <cstdint>

namespace Common::FrameProfiler {

// Who blocks on the GPU: one site per caller of MasterSemaphore::Wait and of the
// CommandScheduler waits (Wait, Finish, FinishAll, FlushAndWait, WaitPriorityOperations).
// Callers that pass none are Other; trace.txt and summary.txt list their source locations.
#define KYTY_FRAME_PROFILER_WAIT_SITES(X)                                                          \
	X(Other, "unlabelled caller")                                                                  \
	X(SchedulerShutdown, "scheduler shutdown")                                                     \
	X(PendingOperations, "deferred operation, its tick's priority operations")                     \
	X(PriorityFence, "priority-operations thread, its next tick (background)")                     \
	X(GpuShutdown, "guest GPU shutdown")                                                           \
	X(UnmapDrain, "unmap of a range the GPU may use")                                              \
	X(Pm4BufferWait, "PM4 buffer wait")                                                            \
	X(Pm4SynchronizeGpu, "PM4 GPU synchronization")                                                \
	X(FaultBufferReuse, "fault buffer area reuse")                                                 \
	X(StreamBufferReuse, "stream buffer range reuse")                                              \
	X(WrittenBackPage, "written-back page hand-back")                                              \
	X(ReadbackPendingPublish, "GPU-thread readback, another download's publish")                   \
	X(ReadbackGpuThread, "GPU-thread readback, its download")                                      \
	X(ReadbackGuestThread, "guest-thread readback, its download")                                  \
	X(PromoteToShared, "buffer promotion to shared memory")                                        \
	X(GarbageCollector, "buffer garbage collection")                                               \
	X(DescriptorHeapTeardown, "descriptor heap teardown")                                          \
	X(BindlessHeapTeardown, "bindless heap teardown")                                              \
	X(BindlessMissRegion, "bindless miss-record region reuse")                                     \
	X(FramePoolTeardown, "presentation frame pool teardown")                                       \
	X(PresentFrameReuse, "presentation frame reuse")                                               \
	X(FrameDump, "--dump-frames readback")                                                         \
	X(ThreadCountRingReuse, "thread-count ring segment reuse")                                     \
	X(DrawArgsRingReuse, "draw-argument ring segment reuse")                                       \
	X(DrawInstancesRead, "indirect draw's instance count, read for a later draw")

#define KYTY_FP_WAIT_SITE_ENUM(name, description) name,
enum class WaitSite : uint8_t { KYTY_FRAME_PROFILER_WAIT_SITES(KYTY_FP_WAIT_SITE_ENUM) Count };
#undef KYTY_FP_WAIT_SITE_ENUM

constexpr uint32_t WaitSiteCount = static_cast<uint32_t>(WaitSite::Count);

// A thread that waits by design and blocks nobody (the priority-operations thread on its next
// tick): kept out of the per-role wait counters and out of the share of wait time.
[[nodiscard]] constexpr bool IsBackgroundSite(WaitSite site) {
	return site == WaitSite::PriorityFence;
}

} // namespace Common::FrameProfiler

#endif // EMULATOR_SRC_COMMON_WAITSITE_H_
