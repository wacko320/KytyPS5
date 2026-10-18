#include "common/frameProfiler.h"

#include "common/debugCounters.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <source_location>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace FrameProfiler = Common::FrameProfiler;

constexpr uint64_t RingCapacity = 1024;
constexpr uint64_t FrameNs      = 16'000'000;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FrameProfilerTests: failed: %s\n", message);
		std::abort();
	}
}

std::string ReadFile(const std::filesystem::path& path) {
	std::ifstream     file(path);
	std::stringstream text;
	text << file.rdbuf();
	return text.str();
}

// Event lines of a trace (tsc tid id a b), in file order.
struct TraceEvent {
	uint64_t tsc;
	uint32_t tid;
	uint32_t id;
	uint64_t a;
	uint64_t b;
};

std::vector<TraceEvent> ReadTrace(const std::filesystem::path& path) {
	std::vector<TraceEvent> events;
	std::ifstream           file(path);
	for (std::string line; std::getline(file, line);) {
		if (line.empty() || line[0] == '#') {
			continue;
		}
		std::istringstream fields(line);
		TraceEvent         e {};
		fields >> e.tsc >> e.tid >> e.id >> e.a >> e.b;
		Check(!fields.fail(), "trace line has five numeric fields");
		events.push_back(e);
	}
	return events;
}

// Runs `count` profiled frames from frame `first`, calling body(i) inside frame i.
template <typename Body>
std::filesystem::path Capture(uint64_t first, uint32_t count, Body&& body) {
	FrameProfiler::Arm(first, count);
	uint64_t now   = 1'000'000'000;
	uint64_t frame = first;
	Check(!FrameProfiler::OnFrameAt(frame, now), "capture starts without finishing");
	Check(FrameProfiler::Active(), "capture is active after its first boundary");
	for (uint32_t i = 0; i < count; i++) {
		body(i);
		now += FrameNs;
		const bool finished = FrameProfiler::OnFrameAt(++frame, now);
		Check(finished == (i + 1 == count), "capture finishes after its last frame only");
	}
	Check(!FrameProfiler::Active(), "capture is inactive after it finished");
	return FrameProfiler::LastOutputDirectory();
}

void TestInactiveSitesRecordNothing() {
	Check(!FrameProfiler::Active(), "profiler starts inactive");
	Check(!FrameProfiler::RingAllocated(), "ring is not allocated before the first capture");
	FrameProfiler::Rec(FrameProfiler::Event::Vblank, 1, 2);
	FrameProfiler::Add(FrameProfiler::Counter::EpochAcquires, 5);
	FrameProfiler::Add(FrameProfiler::Counter::ReadMemoryCallsRender, FrameProfiler::Role::Gpu);
	FrameProfiler::Max(FrameProfiler::Counter::GpuLongestWait, 7);
	FrameProfiler::AddSince(FrameProfiler::Counter::UploadTime, FrameProfiler::Start());
	{
		FrameProfiler::ScopedPhase phase(FrameProfiler::Phase::Commit);
		phase.Switch(FrameProfiler::Phase::Materialize);
	}
	Check(FrameProfiler::Start() == 0, "Start is 0 when inactive");
	Check(!FrameProfiler::RingAllocated(), "inactive sites do not allocate the ring");
	Check(FrameProfiler::RecordedEvents() == 0, "inactive Rec records nothing");
	for (uint32_t c = 0; c < FrameProfiler::CounterCount; c++) {
		Check(FrameProfiler::CounterValue(static_cast<FrameProfiler::Counter>(c)) == 0,
		      "inactive sites leave every counter at 0");
	}
	// Frames pass without a capture armed.
	for (uint64_t frame = 1; frame < 10; frame++) {
		Check(!FrameProfiler::OnFrameAt(frame, frame * FrameNs), "no capture without arming");
	}
	Check(!FrameProfiler::Active(), "frames do not activate an unarmed profiler");
}

void TestCaptureAverages(const std::filesystem::path& root) {
	constexpr uint32_t Frames = 4;
	// Not yet: the capture starts at frame 100.
	FrameProfiler::Arm(100, Frames);
	Check(!FrameProfiler::OnFrameAt(99, 0), "a frame before the start does not capture");
	Check(!FrameProfiler::Active(), "inactive before the start frame");

	const auto dir = Capture(100, Frames, [](uint32_t i) {
		// Frame i: i + 1 acquires, 2 writebacks of 1 KiB each, a GPU-thread readback.
		FrameProfiler::Add(FrameProfiler::Counter::EpochAcquires, i + 1);
		FrameProfiler::Add(FrameProfiler::Counter::WritebackBytes, 1024);
		FrameProfiler::Add(FrameProfiler::Counter::WritebackBytes, 1024);
		FrameProfiler::Add(FrameProfiler::Counter::ReadMemoryCallsRender, FrameProfiler::Role::Gpu);
		FrameProfiler::Max(FrameProfiler::Counter::GpuLongestWait, 10 * (i + 1));
		FrameProfiler::Rec(FrameProfiler::Event::Vblank, i);
		FrameProfiler::ScopedPhase phase(FrameProfiler::Phase::Commit);
	});
	Check(!dir.empty() && std::filesystem::is_directory(dir), "capture wrote its directory");
	Check(dir.parent_path() == root, "capture directory is under the output root");
	Check(dir.filename().string().starts_with("kyty-profile-"), "capture directory name");

	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::EpochAcquires) == 2.5,
	      "per-frame average of a counter: (1 + 2 + 3 + 4) / 4");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::WritebackBytes) == 2048.0,
	      "per-frame average of bytes");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::ReadMemoryCallsGpu) == 1.0,
	      "per-role counter lands in the role's slot");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::ReadMemoryCallsRender) == 0.0,
	      "per-role counter leaves the other roles");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::GpuLongestWait) == 40.0,
	      "a max counter reports the largest frame value");
	Check(FrameProfiler::LastPhaseAverageTicks(FrameProfiler::Phase::Commit) > 0.0,
	      "phase timer accumulated");

	const auto summary = ReadFile(dir / "summary.txt");
	Check(summary.find("frames:    4 (frame 101 to 104)") != std::string::npos,
	      "summary names the frames");
	Check(summary.find("mean 16.00  median 16.00  p90 16.00") != std::string::npos,
	      "summary frame-time statistics");
	Check(summary.find("EpochAcquires") != std::string::npos &&
	          summary.find("2.500") != std::string::npos,
	      "summary lists the counter average");
	Check(summary.find("coherence: epoch acquires") != std::string::npos,
	      "summary uses readable names");
	Check(summary.find("WritebackBytes") != std::string::npos &&
	          summary.find("2.000 KiB") != std::string::npos,
	      "summary shows bytes in KiB");

	const auto frames = ReadFile(dir / "frames.txt");
	Check(frames.find("\n101 16.000") != std::string::npos &&
	          frames.find("\n104 16.000") != std::string::npos,
	      "frames.txt has a row per frame");

	const auto trace_text = ReadFile(dir / "trace.txt");
	Check(trace_text.starts_with("# base "), "trace starts with the base line");
	Check(trace_text.find("tsc_hz") != std::string::npos, "trace header has the tick rate");
	Check(trace_text.find(" 14=Vblank") != std::string::npos, "trace header lists event ids");
	const auto events = ReadTrace(dir / "trace.txt");
	uint32_t   frame_events  = 0;
	uint32_t   vblank_events = 0;
	for (const auto& e: events) {
		frame_events += e.id == static_cast<uint32_t>(FrameProfiler::Event::Frame) ? 1 : 0;
		vblank_events += e.id == static_cast<uint32_t>(FrameProfiler::Event::Vblank) ? 1 : 0;
	}
	Check(frame_events == Frames + 1, "a frame event at every boundary of the window");
	Check(vblank_events == Frames, "events recorded during the capture");

	// After the capture, sites are inert again.
	FrameProfiler::Add(FrameProfiler::Counter::EpochAcquires);
	Check(FrameProfiler::CounterValue(FrameProfiler::Counter::EpochAcquires) == 0,
	      "counters stay at 0 after the capture");
}

void TestRingWrap() {
	// Recorded: the opening frame event, vblanks 0..1323, the closing frame event (1326).
	constexpr uint64_t Vblanks = RingCapacity + 300;
	const auto         record  = [](uint32_t) {
		for (uint64_t i = 0; i < Vblanks; i++) {
			FrameProfiler::Rec(FrameProfiler::Event::Vblank, i);
		}
	};
	const auto dir    = Capture(1000, 1, record);
	const auto events = ReadTrace(dir / "trace.txt");
	Check(events.size() == RingCapacity, "a wrapped ring keeps its capacity in events");
	// Oldest first: the newest 1024 of 1326 start at recording index 302, vblank 301.
	const uint64_t first_kept = Vblanks + 2 - RingCapacity - 1;
	for (size_t i = 0; i + 1 < events.size(); i++) {
		Check(events[i].id == static_cast<uint32_t>(FrameProfiler::Event::Vblank),
		      "kept events are the newest vblanks");
		Check(events[i].a == first_kept + i, "kept events are in recording order");
	}
	Check(events.back().id == static_cast<uint32_t>(FrameProfiler::Event::Frame),
	      "the closing frame event is last");
	for (size_t i = 1; i < events.size(); i++) {
		Check(events[i].tsc >= events[i - 1].tsc, "timestamps rise through the wrap");
	}
	const auto summary = ReadFile(dir / "summary.txt");
	Check(summary.find("1326 events recorded, 1024 kept") != std::string::npos,
	      "summary reports recorded and kept events");
}

// Stands in for a labelled wait API: callers that pass no site are Other, by location.
uint32_t Block(FrameProfiler::WaitSite site = FrameProfiler::WaitSite::Other,
               std::source_location location = std::source_location::current()) {
	const FrameProfiler::ScopedWait wait(false, site, location, 7);
	std::this_thread::sleep_for(std::chrono::microseconds(200));
	return location.line();
}

void TestWaitSites() {
	using FrameProfiler::Role;
	using FrameProfiler::WaitSite;
	uint32_t   unlabelled_line = 0;
	const auto dir             = Capture(2000, 2, [&](uint32_t i) {
		FrameProfiler::NoteRenderThread();
		// Two site-labelled waits per frame on the render thread, one priority wait.
		Block(WaitSite::UnmapDrain);
		Block(WaitSite::UnmapDrain);
		{
			const FrameProfiler::ScopedWait wait(true, WaitSite::ReadbackGpuThread,
			                                     std::source_location::current(), 3);
		}
		// Nested: the inner wait's time is its own, not the outer one's.
		{
			const FrameProfiler::ScopedWait outer(false, WaitSite::GarbageCollector,
			                                      std::source_location::current(), 1);
			std::this_thread::sleep_for(std::chrono::microseconds(100));
			Block(WaitSite::PromoteToShared);
			Block(WaitSite::PromoteToShared);
		}
		// Role split: the same site on another guest thread and on the GPU thread.
		std::thread([] { Block(WaitSite::StreamBufferReuse); }).join();
		std::thread([] {
			FrameProfiler::MarkGpuThread();
			Block(WaitSite::StreamBufferReuse);
			Block(WaitSite::StreamBufferReuse);
		}).join();
		unlabelled_line = Block();
		// A background wait: listed, but not in the role counters or the share.
		Block(WaitSite::PriorityFence);
		(void)i;
	});

	const auto unmap = FrameProfiler::LastWaitSite(false, WaitSite::UnmapDrain, Role::Render);
	Check(unmap.waits == 4 && unmap.ticks > 0, "site aggregation: waits per site and role");
	Check(FrameProfiler::LastWaitSite(false, WaitSite::UnmapDrain, Role::Other).waits == 0 &&
	          FrameProfiler::LastWaitSite(true, WaitSite::UnmapDrain, Role::Render).waits == 0,
	      "site aggregation keeps roles and kinds apart");
	Check(FrameProfiler::LastWaitSite(true, WaitSite::ReadbackGpuThread, Role::Render).waits == 2,
	      "priority waits aggregate on their own");

	const auto outer = FrameProfiler::LastWaitSite(false, WaitSite::GarbageCollector, Role::Render);
	const auto inner = FrameProfiler::LastWaitSite(false, WaitSite::PromoteToShared, Role::Render);
	Check(outer.waits == 2 && inner.waits == 4, "nested waits each count at their own site");
	Check(inner.ticks > outer.ticks,
	      "a nested wait's time is not also charged to the wait around it (4 x 200 us against "
	      "2 x 100 us of its own)");
	// The role counters count the outer wait once, with the inner ones inside it.
	const auto host_waits = FrameProfiler::LastFrameAverage(FrameProfiler::Counter::HostWaitsRender);
	Check(host_waits == 4.0,
	      "role counter: 2 unmap waits, 1 outer wait and 1 unlabelled per frame, not the background "
	      "wait");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::PriorityWaitsRender) == 1.0,
	      "role counter: priority waits");

	Check(FrameProfiler::LastWaitSite(false, WaitSite::StreamBufferReuse, Role::Other).waits == 2 &&
	          FrameProfiler::LastWaitSite(false, WaitSite::StreamBufferReuse, Role::Gpu).waits == 4 &&
	          FrameProfiler::LastWaitSite(false, WaitSite::StreamBufferReuse, Role::Render).waits ==
	              0,
	      "role split: other guest threads and the GPU thread");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::HostWaitsGpu) == 2.0 &&
	          FrameProfiler::LastFrameAverage(FrameProfiler::Counter::HostWaitsOther) == 1.0,
	      "role split in the per-role counters");

	Check(FrameProfiler::LastWaitSite(false, WaitSite::PriorityFence, Role::Render).waits == 2,
	      "a background site is aggregated like any other");
	const auto other = FrameProfiler::LastWaitSite(false, WaitSite::Other, Role::Render);
	Check(other.waits == 2, "a caller that passes no site is Other");
	const auto summary = ReadFile(dir / "summary.txt");
	const auto location =
	    std::string("FrameProfilerTests.cpp:") + std::to_string(unlabelled_line) + " ";
	Check(summary.find("unlabelled callers:") != std::string::npos &&
	          summary.find(location) != std::string::npos,
	      "the summary lists an unlabelled caller by its source location");
	const auto table = summary.find("Waits by site and role");
	Check(table != std::string::npos, "the summary has the wait-site table");
	const auto promote = summary.find("PromoteToShared", table);
	const auto garbage = summary.find("GarbageCollector", table);
	Check(promote != std::string::npos && garbage != std::string::npos && promote < garbage,
	      "the wait-site table is ranked by time");
	const auto fence = summary.find("PriorityFence", table);
	Check(fence != std::string::npos && summary.rfind("     bg", fence) > summary.rfind('\n', fence),
	      "a background site is listed without a share");
	const auto trace = ReadFile(dir / "trace.txt");
	Check(trace.find("# wait-sites 0=Other 1=SchedulerShutdown") != std::string::npos &&
	          trace.find("# wait-location ") != std::string::npos,
	      "the trace header names the sites and the unlabelled locations");
	uint32_t begins = 0;
	for (const auto& e: ReadTrace(dir / "trace.txt")) {
		if (e.id == static_cast<uint32_t>(FrameProfiler::Event::HostWaitBegin) &&
		    (e.a & 0xffffu) == static_cast<uint32_t>(WaitSite::UnmapDrain)) {
			begins++;
			Check(e.b == 7, "HostWaitBegin carries the tick");
		}
	}
	Check(begins == 4, "a HostWaitBegin event per wait");
}

void TestReadbackConsumers() {
	using FrameProfiler::Bit;
	using FrameProfiler::Consumer;
	using FrameProfiler::MaterializeReadback;
	using FrameProfiler::SlotRead;
	// Dwords of one materialization: two read V# dwords and a table pointer on page 0x1000, a T#
	// dword on page 0x3000.
	const std::vector<SlotRead> reads {
	    {0x1000, Bit(Consumer::BufferRead)},
	    {0x1004, Bit(Consumer::BufferRead)},
	    {0x1ffc, Consumer::TablePointer | Consumer::BufferRead},
	    {0x3010, Bit(Consumer::Texture)},
	    {0x4000, 0},
	};
	const auto readback = [](uint64_t address, std::vector<FrameProfiler::AddressRange> waited) {
		return MaterializeReadback {.address = address, .waited = std::move(waited)};
	};
	Check(FrameProfiler::ReadbackConsumers(readback(0x1004, {{0x1000, 0x2000}}), reads) ==
	          (Consumer::BufferRead | Consumer::TablePointer),
	      "a readback has the union of every dword on the pages it waited for, not the first "
	      "one's");
	const auto narrow = FrameProfiler::ReadbackConsumers(readback(0x1000, {{0x1000, 0x1008}}), reads);
	Check(narrow == Bit(Consumer::BufferRead) && FrameProfiler::RemovableReadback(narrow),
	      "a readback of read-V# dwords only is removable");
	Check(FrameProfiler::ReadbackConsumers(readback(0x1000, {{0x1000, 0x2000}, {0x3000, 0x4000}}),
	                                       reads) ==
	          (Bit(Consumer::BufferRead) | Bit(Consumer::TablePointer) | Bit(Consumer::Texture)),
	      "a readback that waited for several pages covers the dwords of each");
	// A download window spans 0x0-0x4000, but only the page of 0x1000 was GPU-written: the T#
	// dword on the clean page 0x3000 did not cause the wait.
	Check(FrameProfiler::ReadbackConsumers(readback(0x1000, {{0x1000, 0x1008}}), reads) ==
	          Bit(Consumer::BufferRead),
	      "a clean page in the download window does not contribute");
	Check(FrameProfiler::ReadbackConsumers(readback(0x2000, {{0x2000, 0x3000}}), reads) ==
	          Bit(Consumer::Unplanned),
	      "a readback no planned dword explains is Unplanned");
	Check(FrameProfiler::ReadbackConsumers(readback(0x4000, {{0x4000, 0x5000}}), reads) ==
	          Bit(Consumer::Copyable),
	      "a readback of dwords without a CPU consumer is Copyable");
	Check(FrameProfiler::ReadbackConsumers(readback(0x1800, {{0x1000, 0x2000}}), reads) ==
	          ((Consumer::BufferRead | Consumer::TablePointer) | Bit(Consumer::Unplanned)),
	      "a readback asked for by an unplanned dword keeps Unplanned beside the planned ones");

	// Through the profiler: two readbacks of one materialization, attributed afterwards.
	constexpr uint64_t Shader = 0x1111222233334444ull;
	const auto dir = Capture(3000, 1, [&](uint32_t) {
		{
			FrameProfiler::ScopedMaterialize materialize(Shader);
			const std::vector<FrameProfiler::AddressRange> first {{0x1000, 0x1008}};
			const std::vector<FrameProfiler::AddressRange> second {{0x1000, 0x2000}};
			FrameProfiler::NoteReadback(0x1000, first, 1000);
			FrameProfiler::NoteReadback(0x1004, second, 3000);
			const auto readbacks = materialize.Take();
			Check(readbacks.size() == 2 && readbacks[1].ticks == 3000 &&
			          readbacks[0].shader == Shader,
			      "a materialization collects its readbacks");
			for (const auto& readback: readbacks) {
				FrameProfiler::NoteMaterializeReadback(
				    readback, FrameProfiler::ReadbackConsumers(readback, reads));
			}
		}
		const std::vector<FrameProfiler::AddressRange> outside {{0x5000, 0x6000}};
		FrameProfiler::NoteReadback(0x5000, outside, 500);
	});
	using FrameProfiler::Counter;
	Check(FrameProfiler::LastFrameAverage(Counter::MaterializeReadbacks) == 2.0,
	      "materialization readbacks counted");
	Check(FrameProfiler::LastFrameAverage(Counter::MaterializeReadbackTimeRemovable) == 1000.0,
	      "removable time: only the readback whose consumers are all read V#s");
	Check(FrameProfiler::LastFrameAverage(Counter::MaterializeReadbackTimeBufferRead) == 4000.0 &&
	          FrameProfiler::LastFrameAverage(Counter::MaterializeReadbackTimeTablePointer) ==
	              3000.0 &&
	          FrameProfiler::LastFrameAverage(Counter::MaterializeReadbackTimeTexture) == 0.0,
	      "per-class time: every readback whose set has the class");
	const auto summary = ReadFile(dir / "summary.txt");
	const auto table   = summary.find("Readbacks by cause and shader");
	Check(table != std::string::npos, "the summary has the readback table");
	const auto row = summary.find("0x1111222233334444", table);
	Check(row != std::string::npos &&
	          summary.find("buffer-read+table-pointer", row) < summary.find('\n', row),
	      "the readback table lists a shader's consumer set");
	Check(summary.find("0x0000000000000000", table) != std::string::npos,
	      "readbacks outside a materialization are listed too");
}

// Levels report their first, last and largest sample of the window, not a per-frame sum, and a
// sample before the capture does not count.
void TestLevels() {
	using FrameProfiler::Level;
	constexpr uint64_t MiB = 1024 * 1024;
	FrameProfiler::SetLevelValue(Level::BufferBytes, 100 * MiB);
	const auto dir = Capture(300, 3, [](uint32_t i) {
		constexpr uint64_t samples[3] = {2, 7, 5};
		FrameProfiler::SetLevelValue(Level::BufferBytes, samples[i] * MiB);
		FrameProfiler::SetLevelValue(Level::BufferCount, 10 + i);
	});
	const auto summary = ReadFile(dir / "summary.txt");
	const auto table   = summary.find("Levels (first, last, largest sample");
	Check(table != std::string::npos, "the summary has the level table");
	const auto line = [&](const char* name) {
		const auto at = summary.find(name, table);
		Check(at != std::string::npos, "the level table lists the level");
		return summary.substr(at, summary.find('\n', at) - at);
	};
	Check(line("buffers ").find("2.0          5.0          7.0") != std::string::npos,
	      "a byte level shows first, last and largest in MiB");
	Check(line("buffer count").find("10.0         12.0         12.0") != std::string::npos,
	      "a count level shows its samples as they are");
	Check(line("textures ").find(" -") != std::string::npos, "an unsampled level shows a dash");
}

// A capture counts the performance panel's events without showing the panel, and fills the
// counters that count the same events from them.
void TestDebugCounters() {
	namespace DebugCounters = Common::DebugCounters;
	using DebugCounters::Counter;
	Check(!DebugCounters::Collecting(), "debug counters are off without the panel or a capture");
	DebugCounters::Add(Counter::Dispatches, 100);
	Check(DebugCounters::Get(Counter::Dispatches) == 0, "debug counters ignore events while off");
	const auto dir = Capture(400, 2, [](uint32_t i) {
		Check(DebugCounters::Collecting() && !DebugCounters::Enabled(),
		      "a capture collects debug counters without showing the panel");
		DebugCounters::Add(Counter::Dispatches, 3 + i);
		DebugCounters::Add(Counter::QueueSubmits, 5);
		FrameProfiler::Add(FrameProfiler::Counter::SubmitsOther, 1);
		FrameProfiler::Add(FrameProfiler::Counter::SubmitsPreSubmit, 1);
		DebugCounters::Add(Counter::Draws, 7);
	});
	Check(!DebugCounters::Collecting(), "debug counters are off after the capture");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::ComputeDispatches) == 3.5,
	      "ComputeDispatches comes from the Dispatches debug counter");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::VkSubmitCalls) == 5.0,
	      "VkSubmitCalls comes from the QueueSubmits debug counter");
	Check(FrameProfiler::LastFrameAverage(FrameProfiler::Counter::SubmitsCommand) == 3.0,
	      "SubmitsCommand is QueueSubmits less the other submits");
	const auto summary = ReadFile(dir / "summary.txt");
	const auto table   = summary.find("Debug counters (per frame");
	Check(table != std::string::npos, "the summary has the debug counter table");
	const auto at = summary.find("Draws ", table);
	Check(at != std::string::npos &&
	          summary.substr(at, summary.find('\n', at) - at).find("7.000") != std::string::npos,
	      "the debug counter table shows per-frame values");

	// The panel's state is its own: a capture leaves it shown.
	DebugCounters::SetEnabled(true);
	(void)Capture(500, 1, [](uint32_t) {});
	Check(DebugCounters::Enabled() && DebugCounters::Collecting(),
	      "a capture leaves the shown panel collecting");
	DebugCounters::SetEnabled(false);
	Check(!DebugCounters::Collecting(), "hiding the panel stops collection");
}

} // namespace

int main() {
	const auto root =
	    std::filesystem::temp_directory_path() /
	    ("kyty_frame_profiler_tests_" +
	     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::filesystem::create_directories(root);
	FrameProfiler::SetOutputRoot(root);
	Check(FrameProfiler::SetRingCapacity(RingCapacity), "ring capacity set before allocation");
	Check(!FrameProfiler::SetRingCapacity(1000), "ring capacity must be a power of two");

	TestInactiveSitesRecordNothing();
	TestCaptureAverages(root);
	Check(!FrameProfiler::SetRingCapacity(2048), "ring capacity is fixed once allocated");
	TestRingWrap();
	TestWaitSites();
	TestReadbackConsumers();
	TestLevels();
	TestDebugCounters();

	std::filesystem::remove_all(root);
	std::printf("FrameProfilerTests: ok\n");
	return 0;
}
