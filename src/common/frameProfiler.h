#ifndef EMULATOR_SRC_COMMON_FRAMEPROFILER_H_
#define EMULATOR_SRC_COMMON_FRAMEPROFILER_H_

// Frame profiler: counters, phase timers and an event trace over a window of frames, armed by a
// hotkey or --profile-frames. When no capture runs, every site costs one relaxed load of
// g_active: no timestamps, no thread ids, no stores.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <source_location>
#include <span>
#include <vector>

#include "common/waitSite.h"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace Common::FrameProfiler {

constexpr uint32_t DefaultFrameCount = 100;

// Trace event ids. The numbers are part of trace.txt and stay stable; retired ids are not reused.
enum class Event : uint16_t {
	Frame              = 1,  // a: frame number
	Enqueue            = 2,  // a: queue, b: submission type
	ProcessBegin       = 3,  // a: queue (99: GPU-thread commands), b: 1 on a submission's first slice
	ProcessEnd         = 4,  // a: 1 when the submission completed
	IdleBegin          = 5,  // GPU thread waits for work
	IdleEnd            = 6,
	BlockedBegin       = 7,  // GPU thread polls blocked queues
	BlockedEnd         = 8,
	VkSubmit           = 9,  // a: tick
	Fence              = 10, // a: tick, seen by the priority-operations thread
	Interrupt          = 11, // a: event id
	FlipRequest        = 12, // a: request id
	FlipReady          = 13, // a: request id
	Vblank             = 14,
	PresentBegin       = 15, // a: group
	PresentEnd         = 16, // a: group
	GuestReadbackBegin = 17, // a: vaddr, b: size
	GuestReadbackEnd   = 18,
	GpuReadbackBegin   = 19, // a: vaddr, b: size
	GpuReadbackEnd     = 20,
	Fault              = 21, // a: access, b: vaddr
	Suspend            = 22, // a: polled address (0: none), b: SuspendReason | func << 8
	EqueueWaitBegin    = 23, // a: queue, b: timeout (all ones: none)
	EqueueWaitEnd      = 24, // a: result, b: first event's ident
	FlipWaitBegin      = 25,
	FlipWaitEnd        = 26,
	SlotWaitBegin      = 27,
	SlotWaitEnd        = 28,
	VkSubmitEnd        = 29,
	WriteBind          = 36, // a: vaddr, b: size
	WriteShader        = 37, // a: shader hash of the program being bound, b: 1 for a texel buffer
	AcquireMem         = 38, // a: PM4 header
	EpochAcquire       = 40, // a: epoch id
	EpochUpload        = 41, // a: bytes, b: copies of a CPU-pending upload
	GpuGuestWrite      = 43, // a: vaddr, b: size (labels, EOP values, DMA data, ...)
	EopWrite           = 44, // a: vaddr, b: value
	// a: WaitSite | unlabelled location << 16 (trace.txt lists the locations), b: tick
	HostWaitBegin      = 45,
	HostWaitEnd        = 46,
	ReadbackContext    = 47, // a: context code (ContextCode), b: shader hash
	PriorityWaitBegin  = 48, // a: as HostWaitBegin, b: tick
	PriorityWaitEnd    = 49,
	GlobalBarrier      = 50, // a: CommandScheduler::GlobalBarrierSite
	GuestSync          = 51, // a: CommandProcessor::GuestSync
	GuestDispatch      = 52, // a: barriers since the previous guest dispatch, b: guest syncs since
	QueueEmpty         = 53, // a: last submitted tick, seen complete (priority thread)
	Feed               = 54, // a: guest draws and dispatches of the command buffer FeedIdleGpu submits
	WaitPass           = 55, // a: address, b: value: a WAIT_REG_MEM that suspended passed
	SuspendWait        = 56, // a: reference, b: mask: follows a WaitRegMem Suspend
	LabelWrite         = 57, // a: vaddr (bit 63: GPU-ordered, visible at completion), b: first qword (WRITE_DATA)
	QueueSwitch        = 58, // a: queue released or ready, b: 1 for a parked queue's tick
	QueuePark          = 59, // a: queue, b: tick it waits for
};

[[nodiscard]] const char* EventName(Event id);

// Binding phases on the GPU thread.
enum class Phase : uint32_t {
	Materialize,
	PrepareBindings,
	StageBindings,
	VertexIndex,
	Commit,
	Count
};

[[nodiscard]] const char* PhaseName(Phase phase);

// Thread roles: the thread that enqueues graphics work on queue 0, other guest threads, the GPU
// thread. Per-role counters (R in the list below) are consecutive in this order.
enum class Role : uint32_t { Render, Other, Gpu };

enum class Unit : uint8_t { Count, Ticks, Nanoseconds, Bytes };
enum class Kind : uint8_t { Sum, Max };

// X(name, unit, kind, description); R declares the Render, Other and Gpu variants of a counter.
// SubmitsCommand, VkSubmitCalls and ComputeDispatches have no sites of their own: a capture fills
// them from the performance panel's counters (Common::DebugCounters), which it reads per frame.
#define KYTY_FRAME_PROFILER_COUNTERS(X, R)                                                         \
	X(GpuProcessTime, Ticks, Sum, "GPU thread: processing wall time")                              \
	X(GpuProcessCpuTime, Nanoseconds, Sum, "GPU thread: processing CPU time")                          \
	X(GpuProcessSlices, Count, Sum, "GPU thread: processing slices")                               \
	X(GpuIdleTime, Ticks, Sum, "GPU thread: idle, waiting for work")                                   \
	X(GpuIdleWaits, Count, Sum, "GPU thread: idle waits")                                          \
	X(GpuBlockedTime, Ticks, Sum, "GPU thread: blocked-queue polls")                                   \
	X(GpuBlockedWaits, Count, Sum, "GPU thread: blocked-queue polls, count")                       \
	X(GpuLongestWait, Ticks, Max, "GPU thread: longest single wait")                               \
	X(GpuProcessTimeGraphics, Ticks, Sum, "GPU thread: processing wall time, graphics queue")        \
	X(GpuProcessTimeCompute, Ticks, Sum, "GPU thread: processing wall time, compute queues")         \
	X(QueueReleaseSwitches, Count, Sum, "queue scheduling: switches to a queue a label released")    \
	X(QueueParks, Count, Sum, "queue scheduling: dispatches parked on a host-GPU tick")             \
	X(QueueParkRetries, Count, Sum, "queue scheduling: parked again on the same packet")             \
	X(QueueParkHiddenTime, Ticks, Sum, "queue scheduling: parked time other queues ran in")         \
	X(SubmitsCommand, Count, Sum, "host submits: command buffers (QueueSubmits less the others)") \
	X(SubmitsPreSubmit, Count, Sum, "host submits: from a pre-submit hook")                        \
	X(SubmitsOther, Count, Sum, "host submits: without command registers")                         \
	X(EpochAcquires, Count, Sum, "coherence: epoch acquires")                                      \
	X(WriteThroughs, Count, Sum, "coherence: write-throughs")                                      \
	X(UploadCallsAcquire, Count, Sum, "upload (acquire): calls")                                   \
	X(UploadNonEmptyAcquire, Count, Sum, "upload (acquire): calls with copies")                    \
	X(UploadBytesAcquire, Bytes, Sum, "upload (acquire): bytes")                                   \
	X(UploadCopiesAcquire, Count, Sum, "upload (acquire): copies")                                 \
	X(UploadCallsCreate, Count, Sum, "upload (buffer creation): calls")                            \
	X(UploadNonEmptyCreate, Count, Sum, "upload (buffer creation): calls with copies")             \
	X(UploadBytesCreate, Bytes, Sum, "upload (buffer creation): bytes")                            \
	X(UploadCopiesCreate, Count, Sum, "upload (buffer creation): copies")                          \
	X(UploadCallsBda, Count, Sum, "upload (BDA): calls")                                           \
	X(UploadNonEmptyBda, Count, Sum, "upload (BDA): calls with copies")                            \
	X(UploadBytesBda, Bytes, Sum, "upload (BDA): bytes")                                           \
	X(UploadCopiesBda, Count, Sum, "upload (BDA): copies")                                         \
	X(UploadTime, Ticks, Sum, "upload: total time")                                                    \
	X(UploadWalkTime, Ticks, Sum, "upload: page walk")                                                 \
	X(UploadCopyTime, Ticks, Sum, "upload: staging copy")                                              \
	X(UploadTemporaryStaging, Count, Sum, "upload: temporary staging buffers")                     \
	X(PendingHolds, Count, Sum, "coherence: pending holds")                                        \
	X(PendingLockWaitGuestTime, Ticks, Sum, "coherence: pending-lock wait, guest threads")             \
	X(PendingLockWaitGpuTime, Ticks, Sum, "coherence: pending-lock wait, GPU thread")                  \
	X(EpochUploadWaits, Count, Sum, "coherence: upload waits for in-flight holds")                 \
	X(EpochUploadWaitTime, Ticks, Sum, "coherence: upload wait for in-flight holds")                   \
	X(EpochUploadLongestWait, Ticks, Max, "coherence: longest upload wait for holds")              \
	X(RegionLockWaitGpuTime, Ticks, Sum, "region lock: contended wait, GPU thread")                    \
	X(RegionLockWaitGuestTime, Ticks, Sum, "region lock: contended wait, other threads")               \
	X(Materializations, Count, Sum, "materialize: calls")                                          \
	X(MaterializeSrtReads, Count, Sum, "materialize: SRT reads planned")                           \
	X(PrefetchTries, Count, Sum, "materialize: prefetch tries")                                    \
	X(PrefetchHits, Count, Sum, "materialize: prefetch hits")                                      \
	X(PrefetchTryTime, Ticks, Sum, "materialize: prefetch tries, time")                                \
	X(PrefetchHitTime, Ticks, Sum, "materialize: prefetch hits, time")                                 \
	X(PlainMaterializations, Count, Sum, "materialize: without prefetch")                          \
	X(PlainMaterializeTime, Ticks, Sum, "materialize: without prefetch, time")                         \
	X(SettleChecks, Count, Sum, "materialize: page settle checks")                                 \
	X(PrecompileSubmitted, Count, Sum, "precompile: new programs handed to workers")               \
	X(PrecompileHits, Count, Sum, "precompile: new programs taken from a worker")                  \
	X(PrecompileWaitTime, Ticks, Sum, "precompile: GPU thread waiting for a worker")                   \
	X(PrecompileMisses, Count, Sum, "precompile: new programs compiled on the GPU thread")         \
	X(PrecompilePipelineHits, Count, Sum, "precompile: graphics pipelines taken from a worker")    \
	X(PrecompileWasted, Count, Sum, "precompile: worker programs dropped unused")                  \
	X(SettleCheckTime, Ticks, Sum, "materialize: page settle checks, time")                            \
	X(WritebackRecords, Count, Sum, "writeback: record calls")                                     \
	X(WritebackBytes, Bytes, Sum, "writeback: bytes copied")                                       \
	X(WritebackRanges, Count, Sum, "writeback: ranges copied")                                     \
	X(WritebackPages, Count, Sum, "writeback: pages completed")                                    \
	X(WritebackPagesNew, Count, Sum, "writeback: pages completed, new")                            \
	R(WritebackHits, Count, Sum, "readback: written-back page hits")                               \
	R(WritebackHitTime, Ticks, Sum, "readback: written-back page hits, time")                          \
	R(WritebackMissUnsubmitted, Count, Sum, "readback: written-back miss, unsubmitted")            \
	R(WritebackMissChanged, Count, Sum, "readback: written-back miss, changed")                    \
	R(WritebackMissChangedTime, Ticks, Sum, "readback: written-back miss, changed, time")          \
	R(ReadMemoryCalls, Count, Sum, "readback: ReadMemory calls")                                   \
	R(ReadMemoryTime, Ticks, Sum, "readback: ReadMemory time")                                         \
	R(DownloadCalls, Count, Sum, "download: calls")                                                \
	R(DownloadTime, Ticks, Sum, "download: time")                                                      \
	R(DownloadQueueTime, Ticks, Sum, "download: until the GPU thread ran it")                          \
	R(DownloadFirstCommandTime, Ticks, Sum, "download: until the first command returned")              \
	R(DownloadWaitTime, Ticks, Sum, "download: wait for the copy")                                     \
	R(DownloadSecondCommandTime, Ticks, Sum, "download: second command")                               \
	R(DownloadCopied, Count, Sum, "download: outcome copied")                                      \
	R(DownloadFast, Count, Sum, "download: outcome fast path")                                     \
	R(DownloadNoCopy, Count, Sum, "download: outcome no copy")                                     \
	R(DownloadUnregistered, Count, Sum, "download: outcome unregistered")                                 \
	R(DownloadPendingPublish, Count, Sum, "download: waited for a publish")                       \
	R(HostWaits, Count, Sum, "host GPU: blocking timeline waits")                                \
	R(HostWaitTime, Ticks, Sum, "host GPU: blocking timeline wait time")                           \
	R(PriorityWaits, Count, Sum, "host GPU: blocking waits for priority operations")             \
	R(PriorityWaitTime, Ticks, Sum, "host GPU: wait time for priority operations")                \
	X(GpuReadbackMaterializeTime, Ticks, Sum, "readback (GPU thread): during materialization")    \
	X(GpuReadbackBindingTime, Ticks, Sum, "readback (GPU thread): other binding phases")           \
	X(GpuReadbackPacketTime, Ticks, Sum, "readback (GPU thread): PM4 packet, outside binding")     \
	X(GpuReadbackCommandTime, Ticks, Sum, "readback (GPU thread): outside PM4 packets")            \
	X(ReadbackDirect, Count, Sum, "direct readback: read from the buffer's mapping")               \
	X(ReadbackDirectTime, Ticks, Sum, "direct readback: time")                                     \
	X(ReadbackDirectBytes, Bytes, Sum, "direct readback: GPU-dirty bytes copied")                  \
	X(ReadbackDirectBlockedWriter, Count, Sum, "direct readback: writer open or running")          \
	X(ReadbackDirectBlockedUntracked, Count, Sum, "direct readback: untracked write running")      \
	X(ReadbackDirectBlockedPending, Count, Sum, "direct readback: publish pending on the pages")   \
	X(ReadbackDirectBlockedUnmapped, Count, Sum, "direct readback: buffer not host-readable")      \
	X(ReadbackDirectBlockedSize, Count, Sum, "direct readback: request over a page")               \
	X(ReadbackDirectBlockedTime, Ticks, Sum, "direct readback: not eligible, download time")       \
	X(ReadbackDirectWaitedWriter, Count, Sum, "direct readback: waited for the writer first")      \
	X(ReadbackDirectWaitedUntracked, Count, Sum, "direct readback: waited for untracked writes")   \
	X(UntrackedGateSkips, Count, Sum, "direct readback: untracked writes running out of reach")    \
	X(UntrackedGateHazards, Count, Sum, "direct readback: skipped untracked writes hit its pages") \
	X(MaterializeReadbacks, Count, Sum, "materialize readback: readbacks")                         \
	X(MaterializeReadbackTimeRemovable, Ticks, Sum, "materialize readback: only read V#s")         \
	X(MaterializeReadbackTimeBufferRead, Ticks, Sum, "materialize readback: a read V#")            \
	X(MaterializeReadbackTimeBufferWritten, Ticks, Sum, "materialize readback: a written V#")      \
	X(MaterializeReadbackTimeTexture, Ticks, Sum, "materialize readback: a T#")                    \
	X(MaterializeReadbackTimeSampler, Ticks, Sum, "materialize readback: an S#")                   \
	X(MaterializeReadbackTimeCondition, Ticks, Sum, "materialize readback: a block condition")     \
	X(MaterializeReadbackTimeAddress, Ticks, Sum, "materialize readback: a prefetch address")      \
	X(MaterializeReadbackTimeTablePointer, Ticks, Sum, "materialize readback: a read's address")   \
	X(MaterializeReadbackTimeImageKey, Ticks, Sum, "materialize readback: an image key or mask")   \
	X(MaterializeReadbackTimeOther, Ticks, Sum, "materialize readback: another CPU consumer")      \
	X(MaterializeReadbackTimeUnplanned, Ticks, Sum, "materialize readback: no planned dword")      \
	X(MaterializeReadbackTimeCopyable, Ticks, Sum, "materialize readback: GPU-copyable only")      \
	X(GpuCommandBuffersTimed, Count, Sum, "GPU time: command buffers timed at completion")         \
	X(GpuCommandBuffersUntimed, Count, Sum, "GPU time: command buffers not timed")                 \
	X(GpuBusyTime, Nanoseconds, Sum, "GPU time: busy (union of command buffers), at completion")   \
	X(GpuPresentBusyTime, Nanoseconds, Sum, "GPU time: presentation command buffers")              \
	X(GpuRegionTimeCompactor, Nanoseconds, Sum, "GPU time: write-bitmap compactor")                \
	X(GpuRegionTimeCheck, Nanoseconds, Sum, "GPU time: check kernels")                             \
	X(GpuRegionTimeFlatCopy, Nanoseconds, Sum, "GPU time: flat-buffer dword copies")               \
	X(GpuRegionTimeDeltaUpload, Nanoseconds, Sum, "GPU time: delta uploads")                       \
	X(GpuRegionSpansUnmeasured, Count, Sum, "GPU time: region spans not timed")                  \
	R(GuestWriteFaults, Count, Sum, "faults: guest write faults on mapped pages")                  \
	R(GuestReadFaults, Count, Sum, "faults: guest read faults on mapped pages")                    \
	X(RunAheadUpPages, Count, Sum, "faults: run-ahead pages made writable above a fault")         \
	X(RunAheadDownPages, Count, Sum, "faults: run-ahead pages made writable below a fault")       \
	X(RunAheadMarks, Count, Sum, "faults: run-ahead windows marked")                              \
	R(PageProtectCalls, Count, Sum, "page protection: protection changes of write tracking")       \
	R(PageProtectTime, Ticks, Sum, "page protection: protection-change time of write tracking")      \
	R(TrackingMprotectCalls, Count, Sum, "page protection: mprotect calls of write tracking")       \
	R(TrackingMprotectTime, Ticks, Sum, "page protection: mprotect time of write tracking")         \
	R(TrackingWriteProtectCalls, Count, Sum, "page protection: userfaultfd write-protect calls")    \
	R(TrackingWriteProtectTime, Ticks, Sum, "page protection: userfaultfd write-protect time")      \
	X(UploadWalkProtectCalls, Count, Sum, "upload: page walk, mprotect calls")                    \
	X(UploadWalkProtectTime, Ticks, Sum, "upload: page walk, mprotect time")                           \
	X(UploadHelperJobs, Count, Sum, "upload: walks the upload helper protected and copied")        \
	X(UploadHelperWaitTime, Ticks, Sum, "upload: waits for the upload helper (submits, stores)")   \
	X(GpuMapCalls, Count, Sum, "mappings: GPU-range maps")                                         \
	X(GpuMapBytes, Bytes, Sum, "mappings: GPU-range maps, bytes")                                  \
	X(GpuUnmapCalls, Count, Sum, "mappings: GPU-range unmaps")                                     \
	X(GpuUnmapBytes, Bytes, Sum, "mappings: GPU-range unmaps, bytes")                              \
	X(GpuProtectAddBytes, Bytes, Sum, "mappings: mprotect adding GPU access, bytes")               \
	X(GpuProtectRemoveBytes, Bytes, Sum, "mappings: mprotect removing GPU access, bytes")          \
	X(UnmapDrains, Count, Sum, "unmap: full GPU drains")                                           \
	X(BufferCreations, Count, Sum, "buffers: created")                                             \
	X(BufferCreatedBytes, Bytes, Sum, "buffers: created, bytes")                                   \
	X(BufferLargestCreated, Bytes, Max, "buffers: largest created")                                \
	X(BufferJoins, Count, Sum, "buffers: joined into a new buffer")                                \
	X(BufferJoinedBytes, Bytes, Sum, "buffers: joined into a new buffer, bytes copied")            \
	X(VkSubmitCalls, Count, Sum, "host submits: vkQueueSubmit calls (debug counter QueueSubmits)") \
	X(VkSubmitTime, Ticks, Sum, "host submits: vkQueueSubmit time")                                    \
	X(VkSubmitLongest, Ticks, Max, "host submits: longest vkQueueSubmit")                          \
	X(GpuQueueEmptyTime, Ticks, Sum, "host submits: queue empty before a submit")                      \
	X(FeedSubmits, Count, Sum, "host submits: feeding an idle GPU (FeedIdleGpu)")                  \
	X(ComputeDispatches, Count, Sum, "compute: guest dispatches recorded (debug counter Dispatches)") \
	X(ComputeDispatchesAfterBarrier, Count, Sum, "compute: dispatches after any barrier")          \
	X(ComputeDispatchesAfterGuestSync, Count, Sum, "compute: dispatches after a guest sync")       \
	X(ComputeDispatchesAfterGlobalBarrier, Count, Sum, "compute: after barrier, a global one")     \
	X(ComputeDispatchesAfterOtherBarrier, Count, Sum, "compute: after barrier, image/upload/join") \
	X(ComputeDispatchesAfterThreadBarrier, Count, Sum, "compute: after barrier, thread-dim only")  \
	X(ComputeIndirectDispatches, Count, Sum, "compute: indirect dispatches")                       \
	X(ComputeIndirectArgsGpuWritten, Count, Sum, "compute: indirect, GPU-written arguments")       \
	X(ComputeThreadDimensionDispatches, Count, Sum, "compute: dispatches with thread dimensions")  \
	X(ComputeIndirectThreadScans, Count, Sum, "compute: window scans for thread-dim batches")      \
	X(ComputeIndirectThreadScanTime, Ticks, Sum, "compute: window scans, time")                    \
	X(ComputeIndirectThreadBatches, Count, Sum, "compute: thread-dimension conversion batches")    \
	X(ComputeIndirectThreadBatchEntries, Count, Sum, "compute: conversions in batches")            \
	X(ComputeIndirectThreadBatched, Count, Sum, "compute: indirect thread dims, batch converted")  \
	X(ComputeIndirectThreadFallback, Count, Sum, "compute: indirect thread dims, converted alone") \
	X(ComputeIndirectThreadFallbackNotReached, Count, Sum, "compute: alone, no batch reached it")  \
	X(ComputeIndirectThreadFallbackMismatch, Count, Sum, "compute: alone, not in the batch")       \
	X(ComputeIndirectThreadFallbackUpload, Count, Sum, "compute: alone, args uploaded or joined")  \
	X(ComputeIndirectThreadFallbackWritten, Count, Sum, "compute: alone, args GPU-written since")  \
	X(ThreadCountRingWaits, Count, Sum, "compute: thread-count ring segment waits")                \
	X(ComputeLdsBarriers, Count, Sum, "compute: barriers between LDS-storage dispatches")          \
	X(ComputeLdsIndirectUnsized, Count, Sum, "compute: LDS storage bound whole (GPU-sized)")      \
	X(GlobalBarriers, Count, Sum, "barriers: global barriers (guest packets, requests)")           \
	X(GuestSyncPacketsPartialFlush, Count, Sum, "guest sync: partial flushes")                     \
	X(GuestSyncPacketsReleaseMem, Count, Sum, "guest sync: releases (RELEASE_MEM, EOP, EOS)")      \
	X(GuestSyncPacketsAcquireMem, Count, Sum, "guest sync: acquires")                              \
	X(GuestSyncPacketsWaitRegMem, Count, Sum, "guest sync: passed label waits")                    \
	X(GuestSyncPacketsOther, Count, Sum, "guest sync: cache flush events")                         \
	X(GuestSyncSubmissions, Count, Sum, "guest sync: submission starts")                           \
	X(DrawStorageWriteBarriers, Count, Sum, "draw: barriers after storage-writing draws")          \
	X(DrawIndirectDraws, Count, Sum, "draw indirect: draws (each MULTI entry counts)")             \
	X(DrawIndirectArgsGpuModified, Count, Sum, "draw indirect: arguments GPU-modified")            \
	X(DrawIndirectGpuArgs, Count, Sum, "draw indirect: arguments read on the GPU")                 \
	X(DrawIndirectCpuArgs, Count, Sum, "draw indirect: GPU-modified arguments read on the CPU")    \
	X(DrawIndirectCpuArgsZero, Count, Sum, "draw indirect: CPU-read GPU-modified, zero count")     \
	X(DrawIndirectBlockedImage, Count, Sum, "draw indirect: CPU, arguments under a GPU image")     \
	X(DrawIndirectBlockedAlignment, Count, Sum, "draw indirect: CPU, arguments not 4-aligned")     \
	X(DrawIndirectBlockedIndexSize, Count, Sum, "draw indirect: CPU, INDEX_BUFFER_SIZE 0")         \
	X(DrawIndirectBlockedIndexRange, Count, Sum, "draw indirect: CPU, index range unmapped")       \
	X(DrawIndirectBlocked8Bit, Count, Sum, "draw indirect: CPU, 8-bit indices")                    \
	X(DrawIndirectBlockedGpuCount, Count, Sum, "draw indirect: CPU, MULTI count GPU-modified")     \
	X(DrawIndirectBlockedSgprDestination, Count, Sum, "draw indirect: CPU, offsets into SGPRs")   \
	X(DrawIndirectBlockedSpecial, Count, Sum, "draw indirect: CPU, metadata/copy/resolve draw")    \
	X(DrawIndirectBlockedTopology, Count, Sum, "draw indirect: CPU, rect/quad/patch topology")     \
	X(DrawIndirectBlockedRestart, Count, Sum, "draw indirect: CPU, non-native restart index")      \
	X(DrawIndirectBlockedMesh, Count, Sum, "draw indirect: CPU, mesh-shader route")                \
	X(DrawIndirectBlockedTarget, Count, Sum, "draw indirect: CPU, target not GPU-owned or clear")  \
	X(DrawIndirectBlockedShaderWrites, Count, Sum, "draw indirect: CPU, shader stores")            \
	X(DrawIndirectBlockedNoDraw, Count, Sum, "draw indirect: CPU, no draw and count needed")       \
	X(DrawIndirectBlockedCapture, Count, Sum, "draw indirect: CPU, count needed, no host read")    \
	X(DrawIndirectBlockedFirstInstance, Count, Sum, "draw indirect: CPU, no firstInstance")        \
	X(DrawIndirectConversions, Count, Sum, "draw indirect: argument conversions")                  \
	X(DrawIndirectRenderBreaks, Count, Sum, "draw indirect: rendering ended for a conversion")     \
	X(DrawIndirectInstanceCaptures, Count, Sum, "draw indirect: instance counts kept on the GPU")  \
	X(DrawIndirectLazyInstanceReads, Count, Sum, "draw indirect: kept instance counts read")       \
	X(DrawIndirectRingWaits, Count, Sum, "draw indirect: argument ring segment waits")             \
	X(DrawIndirectReadBarriers, Count, Sum, "draw indirect: argument reads before shader writes")  \
	X(ColorClearLookups, Count, Sum, "fast clear: lookups of DCC/CMask images")                    \
	X(ColorClearChecks, Count, Sum, "fast clear: lookups that check metadata slices")              \
	X(ColorClearGpuModified, Count, Sum, "fast clear: checks of GPU-modified metadata")            \
	X(ColorClearReadbacks, Count, Sum, "fast clear: metadata readbacks for a CPU check")           \
	X(ColorClearCpuSlicesCleared, Count, Sum, "fast clear: slices cleared by a CPU check")         \
	X(ColorClearGpuChecks, Count, Sum, "fast clear: checks recorded on the GPU")                   \
	X(ColorClearGpuSlices, Count, Sum, "fast clear: slices checked on the GPU")                    \
	X(ColorClearGpuUnchanged, Count, Sum, "fast clear: GPU-modified metadata unchanged, skipped")  \
	X(ColorClearStorageClears, Count, Sum, "fast clear: GPU checks cleared through storage")       \
	X(ColorClearNoGpuRoute, Count, Sum, "fast clear: GPU-modified checks without a GPU route")     \
	X(ColorClearGuestOwned, Count, Sum, "fast clear: no GPU route as the image is guest-owned")   \
	X(ColorClearBufferFills, Count, Sum, "fast clear: GPU checks cleared into the guest buffer")   \
	X(SurfaceMetaQueries, Count, Sum, "surface metadata: HTile clear-state queries and updates")   \
	X(RenderTargetResolves, Count, Sum, "render targets: resolved from their registers")         \
	X(RenderTargetReuses, Count, Sum, "render targets: previous resolve reused (same registers)")  \
	R(ColorMetadataReadbacks, Count, Sum, "readback: other readbacks of DCC/CMask metadata")        \
	R(ColorMetadataReadbackTime, Ticks, Sum, "readback: other readbacks of DCC/CMask metadata, time")

#define KYTY_FP_ENUM(name, unit, kind, description) name,
#define KYTY_FP_ENUM_ROLE(name, unit, kind, description) name##Render, name##Other, name##Gpu,
enum class Counter : uint32_t {
	KYTY_FRAME_PROFILER_COUNTERS(KYTY_FP_ENUM, KYTY_FP_ENUM_ROLE) Count
};
#undef KYTY_FP_ENUM
#undef KYTY_FP_ENUM_ROLE

constexpr uint32_t CounterCount = static_cast<uint32_t>(Counter::Count);

struct CounterInfo {
	const char* name;
	const char* description;
	Unit        unit;
	Kind        kind;
};

[[nodiscard]] const CounterInfo& GetCounterInfo(Counter counter);

// Absolute values sampled while a capture runs (statistics kept elsewhere); the summary reports
// their change over the window.
enum class Gauge : uint32_t {
	PrefetchSubmitted,
	PrefetchHits,
	PrefetchUnmatched,
	PrefetchMismatched,
	PrefetchCancelled,
	PrefetchFailed,
	PrefetchStale,
	PrefetchThrottled,
	PrefetchUnreplayed,
	PrefetchReplayMismatches,
	PrefetchFlushes,
	PrefetchFlushedJobs,
	SrtWatchGeneration,
	SrtWatchSettles,
	Count
};

constexpr uint32_t GaugeCount = static_cast<uint32_t>(Gauge::Count);

[[nodiscard]] const char* GaugeName(Gauge gauge);

// Absolute values sampled while a capture runs (sizes kept elsewhere); the summary reports the
// first, last and largest sample of the window.
enum class Level : uint32_t {
	MappedBytes,      // RenderContext's mapped set (every GPU-range mapping)
	BufferBytes,      // BufferCache buffers
	BufferCount,
	TextureBytes,     // TextureCache images
	TexturePeakBytes, // since start
	DeviceUsageBytes, // VK_EXT_memory_budget usage
	DeviceBudgetBytes,
	Count
};

constexpr uint32_t LevelCount = static_cast<uint32_t>(Level::Count);

[[nodiscard]] const char* LevelName(Level level);

inline std::atomic<bool>     g_active {false};
inline std::atomic<uint64_t> g_counters[CounterCount] {};
inline std::atomic<uint64_t> g_phases[static_cast<uint32_t>(Phase::Count)] {};

[[nodiscard]] inline bool Active() {
	return g_active.load(std::memory_order_relaxed);
}

// Timestamp in ticks: the TSC on x86-64, steady-clock nanoseconds elsewhere. Only call it while
// active; the capture measures the tick rate.
[[nodiscard]] inline uint64_t Ticks() {
#if defined(__x86_64__) || defined(_M_X64)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

// Write-tracking mprotect calls and ticks of this thread while active, so a scope can take the
// share of its time spent in them.
inline thread_local uint64_t t_protect_calls = 0;
inline thread_local uint64_t t_protect_ticks = 0;

// A timestamp for a later AddSince, or 0 when inactive.
[[nodiscard]] inline uint64_t Start() {
	return Active() ? Ticks() : 0;
}

// Unchecked: for sites that have seen Active() or a nonzero Start().
void RecordEvent(Event id, uint64_t a, uint64_t b);
void RaiseMax(Counter counter, uint64_t value);
[[nodiscard]] Role     CurrentRole();
[[nodiscard]] uint64_t ThreadCpuNs();
void                   NoteRenderThread();
void                   SetGaugeValue(Gauge gauge, uint64_t value);
void                   SetLevelValue(Level level, uint64_t value);

inline void Rec(Event id, uint64_t a = 0, uint64_t b = 0) {
	if (Active()) {
		RecordEvent(id, a, b);
	}
}

// Records when t0 (from Start) is nonzero, without loading the flag again.
inline void RecIf(uint64_t t0, Event id, uint64_t a = 0, uint64_t b = 0) {
	if (t0 != 0) {
		RecordEvent(id, a, b);
	}
}

inline void Add(Counter counter, uint64_t value = 1) {
	if (Active()) {
		g_counters[static_cast<uint32_t>(counter)].fetch_add(value, std::memory_order_relaxed);
	}
}

// For per-role counters: pass the Render variant.
inline void Add(Counter render_counter, Role role, uint64_t value = 1) {
	Add(static_cast<Counter>(static_cast<uint32_t>(render_counter) + static_cast<uint32_t>(role)),
	    value);
}

inline void Max(Counter counter, uint64_t value) {
	if (Active()) {
		RaiseMax(counter, value);
	}
}

// Adds the ticks since t0 (from Start); nothing when t0 is 0.
inline void AddSince(Counter counter, uint64_t t0) {
	if (t0 != 0) {
		g_counters[static_cast<uint32_t>(counter)].fetch_add(Ticks() - t0,
		                                                    std::memory_order_relaxed);
	}
}

constexpr uint8_t NoPhase = 0xff;

// The binding phase the current thread is in (NoPhase: none); kept while active.
inline thread_local uint8_t t_phase = NoPhase;

// Times a phase until destruction, Stop or a Switch to another phase.
class ScopedPhase final {
public:
	explicit ScopedPhase(Phase phase): m_phase(phase), m_t0(Start()) { Enter(); }
	// For a site that has loaded Active() already.
	ScopedPhase(Phase phase, bool active): m_phase(phase), m_t0(active ? Ticks() : 0) {
		Enter();
	}
	~ScopedPhase() { Stop(); }
	ScopedPhase(const ScopedPhase&)            = delete;
	ScopedPhase& operator=(const ScopedPhase&) = delete;

	void Stop() {
		if (m_t0 != 0) {
			g_phases[static_cast<uint32_t>(m_phase)].fetch_add(Ticks() - m_t0,
			                                                   std::memory_order_relaxed);
			m_t0    = 0;
			t_phase = m_previous;
		}
	}
	void Switch(Phase phase) {
		Stop();
		m_phase = phase;
		m_t0    = Start();
		Enter();
	}

private:
	void Enter() {
		if (m_t0 != 0) {
			m_previous = t_phase;
			t_phase    = static_cast<uint8_t>(m_phase);
		}
	}

	Phase    m_phase;
	uint64_t m_t0;
	uint8_t  m_previous = NoPhase;
};

// The GPU thread marks itself once; roles and per-thread counters use it.
void MarkGpuThread();

// Who asked for a CPU-pending upload. The upload counters come in groups of four per source, in
// this order.
enum class UploadSource : uint32_t { Acquire, Create, Bda };

void SetUploadSource(UploadSource source);

// The counter of the current thread's upload source: pass the Acquire variant.
[[nodiscard]] Counter UploadCounter(Counter acquire_counter);

// Sets this thread's upload source for its lifetime (only while active).
class ScopedUploadSource final {
public:
	explicit ScopedUploadSource(UploadSource source): m_set(Active()) {
		if (m_set) {
			SetUploadSource(source);
		}
	}
	~ScopedUploadSource() {
		if (m_set) {
			SetUploadSource(UploadSource::Acquire);
		}
	}
	ScopedUploadSource(const ScopedUploadSource&)            = delete;
	ScopedUploadSource& operator=(const ScopedUploadSource&) = delete;

private:
	bool m_set;
};

// Shader hash of the program the GPU thread last looked up (only stored while active), for
// WriteShader events.
void                   SetCurrentShader(uint64_t hash);
[[nodiscard]] uint64_t CurrentShader();

// What the current thread is doing, for labelling its readbacks: the PM4 opcode it runs
// (NoPacket: none), why a readback was asked for, and the shader being materialized. Only kept
// while active.
enum class ReadbackCause : uint8_t { None, FaultRead, FaultWrite, Invalidate, Metadata, Unmap };
constexpr uint8_t           NoPacket        = 0xff;
constexpr uint8_t           PacketFetch     = 0xfe;
constexpr uint8_t           PacketLookahead = 0xfd;
inline thread_local uint8_t       t_packet             = NoPacket;
inline thread_local ReadbackCause t_readback_cause     = ReadbackCause::None;
inline thread_local uint64_t      t_materialize_shader = 0;

// a for ReadbackContext: opcode | phase << 8 | cause << 16 | role << 24.
[[nodiscard]] uint64_t ContextCode();

// Sets the PM4 packet label for a scope. Always, not only while active: a capture that starts
// inside the scope must not leave the label behind (a thread-local store costs no more than
// loading the flag).
class ScopedPacket final {
public:
	explicit ScopedPacket(uint8_t packet): m_previous(t_packet) { t_packet = packet; }
	~ScopedPacket() { t_packet = m_previous; }
	ScopedPacket(const ScopedPacket&)            = delete;
	ScopedPacket& operator=(const ScopedPacket&) = delete;

private:
	const uint8_t m_previous;
};

// Sets the readback cause for a scope unless an outer scope has set one (only while active): the
// outer cause wins, so the invalidation inside an unmap is labelled Unmap.
class ScopedCause final {
public:
	explicit ScopedCause(ReadbackCause cause)
	    : m_previous(t_readback_cause), m_set(Active() && t_readback_cause == ReadbackCause::None) {
		if (m_set) {
			t_readback_cause = cause;
		}
	}
	~ScopedCause() {
		if (m_set) {
			t_readback_cause = m_previous;
		}
	}
	ScopedCause(const ScopedCause&)            = delete;
	ScopedCause& operator=(const ScopedCause&) = delete;

private:
	const ReadbackCause m_previous;
	const bool          m_set;
};

[[nodiscard]] const char* WaitSiteName(WaitSite site);

// Records a blocking wait from construction to destruction: per site and role, and, for Other,
// per source location. A wait inside another wait on the same thread counts once in the role
// counters (as part of the outer wait) and with its own time, excluded from the outer one's, in
// the site table.
class ScopedWait final {
public:
	ScopedWait(bool priority, WaitSite site, const std::source_location& location, uint64_t tick)
	    : m_priority(priority), m_site(site), m_t0(Start()) {
		if (m_t0 != 0) {
			Begin(location, tick);
		}
	}
	~ScopedWait() {
		if (m_t0 != 0) {
			End();
		}
	}
	ScopedWait(const ScopedWait&)            = delete;
	ScopedWait& operator=(const ScopedWait&) = delete;

private:
	void Begin(const std::source_location& location, uint64_t tick);
	void End();

	const bool     m_priority;
	const WaitSite m_site;
	uint32_t       m_location = 0;
	uint32_t       m_depth    = 0;
	const uint64_t m_t0;
};

// The CPU consumers of a dword materialization read: what keeps the GPU thread waiting when the
// dword is GPU-written. Bits; a readback has the union of the dwords it covers.
enum class Consumer : uint16_t {
	BufferRead    = 1u << 0u, // a V# the shader only reads
	BufferWritten = 1u << 1u, // a V# the shader writes
	Texture       = 1u << 2u, // a T#
	Sampler       = 1u << 3u, // an S#
	Condition     = 1u << 4u, // a block condition
	Address       = 1u << 5u, // a DMA prefetch address
	TablePointer  = 1u << 6u, // the address of another SRT read
	ImageKey      = 1u << 7u, // an indirect image's key count or selector mask
	// A fill value, a clean (strict) slot, a uniform-buffer or LoadBufferU32 read, a dword read
	// more than once, a read the walker does not evaluate as one dword, a slot whose flat offset
	// is not its own, or a source no resource uses.
	Other         = 1u << 8u,
	// No planned dword explains the readback: it was asked for by a read whose address the
	// attribution cannot evaluate or that the plan does not list, or no planned dword lies on the
	// pages it waited for.
	Unplanned     = 1u << 9u,
	// Only dwords no CPU-evaluated value needs (GPU-copyable slots): the CPU read them anyway.
	Copyable      = 1u << 10u,
};
constexpr uint32_t ConsumerCount = 11;

[[nodiscard]] constexpr uint16_t operator|(Consumer a, Consumer b) {
	return static_cast<uint16_t>(static_cast<uint16_t>(a) | static_cast<uint16_t>(b));
}
[[nodiscard]] constexpr uint16_t Bit(Consumer consumer) {
	return static_cast<uint16_t>(consumer);
}
[[nodiscard]] const char* ConsumerName(uint32_t bit_index);

// A guest range [begin, end).
struct AddressRange {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

// A readback during a materialization: the dword read that asked for it, the pages whose
// GPU-written bytes it waited for, its time and labels.
struct MaterializeReadback {
	uint64_t                  address = 0;
	std::vector<AddressRange> waited;
	uint64_t                  ticks   = 0;
	uint64_t                  context = 0;
	uint64_t                  shader  = 0;
};

// A planned dword read of the materialized shader: its guest address and Consumer bits.
struct SlotRead {
	uint64_t address   = 0;
	uint16_t consumers = 0;
};

// The union of the consumers of every planned dword on the pages the readback waited for (a
// clean page in its download window would not have needed the wait); Unplanned when the dword
// that asked for it is none of them; Copyable when they have no CPU consumer.
[[nodiscard]] uint16_t ReadbackConsumers(const MaterializeReadback& readback,
                                         std::span<const SlotRead>  reads);
// Whether G3 (read V#s decoded on the GPU) removes the readback: every consumer is a read V#.
[[nodiscard]] constexpr bool RemovableReadback(uint16_t consumers) {
	return consumers == Bit(Consumer::BufferRead);
}

// Collects the readbacks of one materialization on this thread (only while active).
class ScopedMaterialize final {
public:
	explicit ScopedMaterialize(uint64_t shader)
	    : m_previous_shader(t_materialize_shader), m_set(Active()) {
		if (m_set) {
			Begin(shader);
		}
	}
	~ScopedMaterialize() {
		if (m_set) {
			End();
		}
	}
	ScopedMaterialize(const ScopedMaterialize&)            = delete;
	ScopedMaterialize& operator=(const ScopedMaterialize&) = delete;

	// Ends the collection and returns its readbacks.
	[[nodiscard]] std::vector<MaterializeReadback> Take();

private:
	void Begin(uint64_t shader);
	void End();

	const uint64_t m_previous_shader;
	const bool     m_set;
};

// A ReadMemory call has ended: adds it to the "readbacks by cause and shader" table under the
// current context, and to the current materialization's readbacks. `waited` are the pages whose
// GPU-written bytes it waited for.
void NoteReadback(uint64_t address, std::span<const AddressRange> waited, uint64_t ticks);
// Attributes a materialization readback to its consumers (counters and the readback table).
void NoteMaterializeReadback(const MaterializeReadback& readback, uint16_t consumers);

// Capture control. Arm starts a capture of `count` frames at the first frame boundary whose
// number is at least `start_frame` (0: the next boundary). OnFrame runs once per presented frame;
// it returns true once, when a capture has finished and its output is written.
void                               Arm(uint64_t start_frame, uint32_t count);
[[nodiscard]] bool                 Armed();
[[nodiscard]] bool                 OnFrame(uint64_t frame_number);
[[nodiscard]] bool                 OnFrameAt(uint64_t frame_number, uint64_t now_ns);
[[nodiscard]] std::filesystem::path LastOutputDirectory();

// Output goes to <root>/kyty-profile-YYYYMMDD-HHMMSS/; the root defaults to the working directory.
void SetOutputRoot(const std::filesystem::path& root);
// Ring capacity in events, a power of two; only before the first capture allocates the ring.
[[nodiscard]] bool SetRingCapacity(uint64_t events);

// Inspection, for tests.
[[nodiscard]] bool     RingAllocated();
[[nodiscard]] uint64_t RecordedEvents();
[[nodiscard]] uint64_t CounterValue(Counter counter);
[[nodiscard]] double   LastFrameAverage(Counter counter);
[[nodiscard]] double   LastPhaseAverageTicks(Phase phase);
struct WaitTotals {
	uint64_t waits = 0;
	uint64_t ticks = 0;
};
// Over the last capture: the waits of one site and role (priority: WaitPriorityOperations).
[[nodiscard]] WaitTotals LastWaitSite(bool priority, WaitSite site, Role role);

} // namespace Common::FrameProfiler

#endif // EMULATOR_SRC_COMMON_FRAMEPROFILER_H_
