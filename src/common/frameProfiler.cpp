#include "common/frameProfiler.h"

#include "common/debugCounters.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#else
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace Common::FrameProfiler {

namespace {

struct EventRecord {
	uint64_t tsc;
	uint32_t tid;
	uint16_t id;
	uint16_t pad;
	uint64_t a;
	uint64_t b;
};

constexpr uint64_t DefaultRingCapacity = uint64_t {1} << 22;

#define KYTY_FP_INFO(name, unit, kind, description)                                                \
	CounterInfo {#name, description, Unit::unit, Kind::kind},
#define KYTY_FP_INFO_ROLE(name, unit, kind, description)                                           \
	CounterInfo {#name "Render", description " (render thread)", Unit::unit, Kind::kind},          \
	    CounterInfo {#name "Other", description " (other threads)", Unit::unit, Kind::kind},       \
	    CounterInfo {#name "Gpu", description " (GPU thread)", Unit::unit, Kind::kind},
constexpr CounterInfo CounterInfos[] = {
    KYTY_FRAME_PROFILER_COUNTERS(KYTY_FP_INFO, KYTY_FP_INFO_ROLE)};
#undef KYTY_FP_INFO
#undef KYTY_FP_INFO_ROLE
static_assert(std::size(CounterInfos) == CounterCount);

constexpr uint32_t PhaseCount = static_cast<uint32_t>(Phase::Count);

constexpr const char* PhaseNames[PhaseCount] = {"materialize", "prepare-bindings", "stage-bindings",
                                                "vertex+index", "commit"};

constexpr const char* GaugeNames[GaugeCount] = {
    "prefetch submitted",     "prefetch hits",      "prefetch unmatched",
    "prefetch mismatched",    "prefetch cancelled", "prefetch failed",
    "prefetch stale",         "prefetch throttled", "prefetch unreplayed hits",
    "prefetch replay mismatches", "prefetch flushes", "prefetch flushed jobs",
    "SRT watch generation moves", "SRT watch settles"};

struct LevelInfo {
	const char* name;
	bool        bytes;
};

constexpr LevelInfo LevelInfos[LevelCount] = {
    {"mapped (GPU range)", true}, {"buffers", true},       {"buffer count", false},
    {"textures", true},           {"textures, peak since start", true},
    {"device usage", true},       {"device budget", true}};

struct EventNameEntry {
	Event       id;
	const char* name;
};

constexpr EventNameEntry EventNames[] = {
    {Event::Frame, "Frame"},
    {Event::Enqueue, "Enqueue"},
    {Event::ProcessBegin, "ProcB"},
    {Event::ProcessEnd, "ProcE"},
    {Event::IdleBegin, "IdleB"},
    {Event::IdleEnd, "IdleE"},
    {Event::BlockedBegin, "BlockB"},
    {Event::BlockedEnd, "BlockE"},
    {Event::VkSubmit, "VkSubmit"},
    {Event::Fence, "Fence"},
    {Event::Interrupt, "Irq"},
    {Event::FlipRequest, "FlipReq"},
    {Event::FlipReady, "FlipReady"},
    {Event::Vblank, "Vblank"},
    {Event::PresentBegin, "PresentB"},
    {Event::PresentEnd, "PresentE"},
    {Event::GuestReadbackBegin, "GuestRbB"},
    {Event::GuestReadbackEnd, "GuestRbE"},
    {Event::GpuReadbackBegin, "GpuRbB"},
    {Event::GpuReadbackEnd, "GpuRbE"},
    {Event::Fault, "Fault"},
    {Event::Suspend, "Suspend"},
    {Event::EqueueWaitBegin, "EqWaitB"},
    {Event::EqueueWaitEnd, "EqWaitE"},
    {Event::FlipWaitBegin, "FlipWaitB"},
    {Event::FlipWaitEnd, "FlipWaitE"},
    {Event::SlotWaitBegin, "SlotWaitB"},
    {Event::SlotWaitEnd, "SlotWaitE"},
    {Event::VkSubmitEnd, "VkSubmitE"},
    {Event::WriteBind, "WriteBind"},
    {Event::WriteShader, "WriteShader"},
    {Event::AcquireMem, "AcquireMem"},
    {Event::EpochAcquire, "EpochAcquire"},
    {Event::EpochUpload, "EpochUpload"},
    {Event::GpuGuestWrite, "GpuGuestWrite"},
    {Event::EopWrite, "EopWrite"},
    {Event::HostWaitBegin, "HostWaitB"},
    {Event::HostWaitEnd, "HostWaitE"},
    {Event::ReadbackContext, "RbContext"},
    {Event::PriorityWaitBegin, "PrioWaitB"},
    {Event::PriorityWaitEnd, "PrioWaitE"},
    {Event::GlobalBarrier, "GlobalBarrier"},
    {Event::GuestSync, "GuestSync"},
    {Event::GuestDispatch, "GuestDispatch"},
    {Event::QueueEmpty, "QueueEmpty"},
    {Event::Feed, "Feed"},
};

#define KYTY_FP_WAIT_SITE_INFO(name, description) {#name, description},
struct WaitSiteInfo {
	const char* name;
	const char* description;
};
constexpr WaitSiteInfo WaitSiteInfos[] = {KYTY_FRAME_PROFILER_WAIT_SITES(KYTY_FP_WAIT_SITE_INFO)};
#undef KYTY_FP_WAIT_SITE_INFO
static_assert(std::size(WaitSiteInfos) == WaitSiteCount);

constexpr const char* ConsumerNames[ConsumerCount] = {
    "buffer-read",   "buffer-written", "texture", "sampler", "condition", "address",
    "table-pointer", "image-key",      "other",   "unplanned", "copyable"};

constexpr uint32_t RoleCount = 3;
constexpr const char* RoleNames[RoleCount] = {"render", "other", "gpu"};

// Blocking waits by kind (host timeline, priority operations), site and role, over the capture.
struct WaitCell {
	std::atomic<uint64_t> waits {0};
	std::atomic<uint64_t> ticks {0};
};
WaitCell   g_waits[2][WaitSiteCount][RoleCount];
WaitTotals g_last_waits[2][WaitSiteCount][RoleCount];

// Source locations of unlabelled (WaitSite::Other) waits, interned on first use; id 0 is "past
// the table".
constexpr uint32_t MaxWaitLocations = 64;
struct WaitLocation {
	const char* file     = nullptr;
	uint32_t    line     = 0;
	const char* function = nullptr;
};
std::mutex            g_location_mutex;
WaitLocation          g_locations[MaxWaitLocations];
std::atomic<uint32_t> g_location_count {1};
WaitCell              g_location_waits[2][MaxWaitLocations];

// Waits nested on one thread: each level's inner waits, so a level counts only its own time.
constexpr uint32_t    MaxWaitDepth = 8;
thread_local uint32_t t_wait_depth = 0;
thread_local uint64_t t_wait_inner[MaxWaitDepth] {};

// ReadMemory calls by context code and shader, over the capture.
struct ReadbackKey {
	uint64_t context = 0;
	uint64_t shader  = 0;
	bool     operator==(const ReadbackKey&) const = default;
};
struct ReadbackKeyHash {
	size_t operator()(const ReadbackKey& key) const {
		return std::hash<uint64_t> {}(key.context * 0x9e3779b97f4a7c15ull ^ key.shader);
	}
};
struct ReadbackRow {
	uint64_t calls           = 0;
	uint64_t ticks           = 0;
	uint64_t attributed      = 0;
	uint64_t removable_ticks = 0;
	uint16_t consumers       = 0;
};
std::mutex                                                   g_readback_mutex;
std::unordered_map<ReadbackKey, ReadbackRow, ReadbackKeyHash> g_readbacks;

// The current materialization's readbacks on this thread (ScopedMaterialize).
thread_local bool                             t_collect_readbacks = false;
thread_local std::vector<MaterializeReadback> t_materialize_readbacks;

// The performance panel's counters (Common::DebugCounters), which a capture turns on and reads.
using DebugCounter = DebugCounters::Counter;
constexpr uint32_t DebugCounterCount = static_cast<uint32_t>(DebugCounter::Count);

struct DebugCounterInfo {
	const char* name;
	Unit        unit;
};

constexpr DebugCounterInfo DebugCounterInfos[DebugCounterCount] = {
    {"Draws", Unit::Count},
    {"DrawInstances", Unit::Count},
    {"IndirectDraws", Unit::Count},
    {"Dispatches", Unit::Count},
    {"QueueSubmits", Unit::Count},
    {"GpuWaits", Unit::Count},
    {"GpuWaitTime", Unit::Nanoseconds},
    {"GpuFaults", Unit::Count},
    {"ShadersCompiled", Unit::Count},
    {"ShaderCompileTime", Unit::Nanoseconds},
    {"PipelinesCreated", Unit::Count},
    {"PipelineCreateTime", Unit::Nanoseconds},
    {"TextureUploadBytes", Unit::Bytes},
    {"TextureDownloadBytes", Unit::Bytes},
    {"TextureEvictions", Unit::Count},
    {"BufferUploadBytes", Unit::Bytes},
    {"BufferDownloadBytes", Unit::Bytes},
    {"BufferEvictions", Unit::Count},
    {"FlipsRequested", Unit::Count},
    {"FlipsRejected", Unit::Count},
    {"LateVblanks", Unit::Count},
    {"SwapchainRecreations", Unit::Count},
    {"DroppedPresents", Unit::Count},
    {"EmulatedInstructions", Unit::Count},
    {"UnresolvedImportCalls", Unit::Count},
    {"GuestFileReadBytes", Unit::Bytes},
    {"GuestFileWriteBytes", Unit::Bytes},
    {"AudioUnderruns", Unit::Count},
};
static_assert(static_cast<uint32_t>(DebugCounter::AudioUnderruns) + 1 == DebugCounterCount);

// Our counters that count the same event as a debug counter: a capture copies the debug
// counter's per-frame change into them.
struct SameEvent {
	Counter      counter;
	DebugCounter debug;
};
constexpr SameEvent SameEvents[] = {
    {Counter::VkSubmitCalls, DebugCounter::QueueSubmits},
    {Counter::ComputeDispatches, DebugCounter::Dispatches},
};

struct FrameRecord {
	uint64_t number   = 0;
	double   ms       = 0.0;
	uint64_t counters[CounterCount] {};
	uint64_t phases[PhaseCount] {};
	uint64_t debug[DebugCounterCount] {};
};

enum class State { Idle, Armed, Capturing };

std::atomic<State>        g_state {State::Idle};
std::atomic<EventRecord*> g_ring {nullptr};
uint64_t                  g_ring_capacity = DefaultRingCapacity;
std::atomic<uint64_t>     g_head {0};
std::atomic<uint32_t>     g_render_tid {0};
std::atomic<uint64_t>     g_gauge_first[GaugeCount] {};
std::atomic<uint64_t>     g_gauge_last[GaugeCount] {};
std::atomic<bool>         g_gauge_seen[GaugeCount] {};
std::atomic<uint64_t>     g_level_first[LevelCount] {};
std::atomic<uint64_t>     g_level_last[LevelCount] {};
std::atomic<uint64_t>     g_level_max[LevelCount] {};
std::atomic<bool>         g_level_seen[LevelCount] {};
thread_local bool         t_gpu_thread     = false;
thread_local UploadSource t_upload_source  = UploadSource::Acquire;
thread_local uint64_t     t_current_shader = 0;

// Capture control, under g_mutex.
std::mutex               g_mutex;
uint64_t                 g_start_frame    = 0;
uint32_t                 g_count          = DefaultFrameCount;
uint64_t                 g_first_frame    = 0;
uint64_t                 g_last_frame_ns  = 0;
uint64_t                 g_calib_ticks    = 0;
uint64_t                 g_calib_ns       = 0;
std::vector<FrameRecord> g_frames;
uint64_t                 g_debug_base[DebugCounterCount] {};
std::filesystem::path    g_output_root;
std::filesystem::path    g_last_output;
double                   g_last_average[CounterCount] {};
double                   g_last_phase_average[PhaseCount] {};

uint64_t SteadyNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint32_t ThreadId() {
#if defined(_WIN32)
	static thread_local const uint32_t tid = static_cast<uint32_t>(GetCurrentThreadId());
#elif defined(__linux__)
	static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
#elif defined(__APPLE__)
	static thread_local const uint32_t tid = [] {
		uint64_t id = 0;
		pthread_threadid_np(nullptr, &id);
		return static_cast<uint32_t>(id);
	}();
#else
	static thread_local const uint32_t tid = 0;
#endif
	return tid;
}

uint64_t ModuleBase() {
#if defined(_WIN32)
	return reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
#else
	Dl_info info {};
	if (dladdr(reinterpret_cast<void*>(&ModuleBase), &info) != 0) {
		return reinterpret_cast<uint64_t>(info.dli_fbase);
	}
	return 0;
#endif
}

std::string Timestamp() {
	const std::time_t now = std::time(nullptr);
	std::tm           local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[32] = {};
	std::strftime(text, sizeof(text), "%Y%m%d-%H%M%S", &local);
	return text;
}

double TicksToMs(double ticks, double tick_hz) {
	return tick_hz > 0.0 ? ticks * 1000.0 / tick_hz : 0.0;
}

double ToDisplay(const CounterInfo& info, double value, double tick_hz) {
	switch (info.unit) {
		case Unit::Ticks: return TicksToMs(value, tick_hz);
		case Unit::Nanoseconds: return value / 1e6;
		case Unit::Bytes: return value / 1024.0;
		case Unit::Count: return value;
	}
	return value;
}

const char* UnitLabel(Unit unit) {
	switch (unit) {
		case Unit::Ticks:
		case Unit::Nanoseconds: return "ms";
		case Unit::Bytes: return "KiB";
		case Unit::Count: return "";
	}
	return "";
}

double Percentile(std::vector<double> values, double fraction) {
	if (values.empty()) {
		return 0.0;
	}
	std::sort(values.begin(), values.end());
	const auto index = std::min(values.size() - 1,
	                            static_cast<size_t>(fraction * static_cast<double>(values.size())));
	return values[index];
}

double Median(std::vector<double> values) {
	if (values.empty()) {
		return 0.0;
	}
	std::sort(values.begin(), values.end());
	const auto n = values.size();
	return n % 2 == 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

// Per-frame value of a counter over the window: the mean for sums, the largest for maxima.
double WindowValue(Counter counter) {
	const auto index = static_cast<uint32_t>(counter);
	if (g_frames.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	double max = 0.0;
	for (const auto& frame: g_frames) {
		sum += static_cast<double>(frame.counters[index]);
		max = std::max(max, static_cast<double>(frame.counters[index]));
	}
	return CounterInfos[index].kind == Kind::Max ? max
	                                             : sum / static_cast<double>(g_frames.size());
}

double PhaseAverage(uint32_t phase) {
	if (g_frames.empty()) {
		return 0.0;
	}
	double sum = 0.0;
	for (const auto& frame: g_frames) {
		sum += static_cast<double>(frame.phases[phase]);
	}
	return sum / static_cast<double>(g_frames.size());
}

constexpr const char* CauseNames[] = {"-", "fault-read", "fault-write", "invalidate", "metadata",
                                       "unmap"};

// Where a readback ran, from its context code: the binding phase, else the PM4 packet.
std::string ReadbackWhere(uint64_t code) {
	const auto packet = static_cast<uint8_t>(code);
	const auto phase  = static_cast<uint8_t>(code >> 8u);
	if (phase < PhaseCount) {
		return PhaseNames[phase];
	}
	switch (packet) {
		case NoPacket: return "command";
		case PacketFetch: return "pm4 fetch";
		case PacketLookahead: return "pm4 lookahead";
		default: {
			char text[16] = {};
			std::snprintf(text, sizeof(text), "pm4 0x%02x", packet);
			return text;
		}
	}
}

std::string ConsumerList(uint16_t consumers) {
	std::string text;
	for (uint32_t bit = 0; bit < ConsumerCount; bit++) {
		if ((consumers & (1u << bit)) != 0) {
			text += text.empty() ? "" : "+";
			text += ConsumerNames[bit];
		}
	}
	return text.empty() ? "-" : text;
}

// Ranked by time: every site and role that waited, then the unlabelled callers by location.
void WriteWaitTable(FILE* f, double n, double tick_hz) {
	struct Row {
		uint32_t kind;
		uint32_t site;
		uint32_t role;
		uint64_t waits;
		uint64_t ticks;
	};
	std::vector<Row> rows;
	uint64_t         total = 0;
	for (uint32_t kind = 0; kind < 2; kind++) {
		for (uint32_t site = 0; site < WaitSiteCount; site++) {
			for (uint32_t role = 0; role < RoleCount; role++) {
				const auto& cell  = g_waits[kind][site][role];
				const auto  waits = cell.waits.load(std::memory_order_relaxed);
				const auto  ticks = cell.ticks.load(std::memory_order_relaxed);
				if (waits != 0) {
					rows.push_back({kind, site, role, waits, ticks});
					total += IsBackgroundSite(static_cast<WaitSite>(site)) ? 0 : ticks;
				}
			}
		}
	}
	std::sort(rows.begin(), rows.end(),
	          [](const Row& a, const Row& b) { return a.ticks > b.ticks; });
	std::fprintf(f, "\nWaits by site and role (per frame; a nested wait counts only its own time; "
	                "background sites are not in the share)\n");
	std::fprintf(f, "  %8s %8s %7s  %-8s %-6s  %-22s %s\n", "ms", "waits", "share", "kind",
	             "role", "site", "description");
	for (const auto& row: rows) {
		const auto& info = WaitSiteInfos[row.site];
		char        share[16] = "     bg";
		if (!IsBackgroundSite(static_cast<WaitSite>(row.site))) {
			std::snprintf(share, sizeof(share), "%6.1f%%",
			              total != 0 ? 100.0 * static_cast<double>(row.ticks) /
			                               static_cast<double>(total)
			                         : 0.0);
		}
		std::fprintf(f, "  %8.3f %8.2f %s  %-8s %-6s  %-22s %s\n",
		             TicksToMs(static_cast<double>(row.ticks), tick_hz) / n,
		             static_cast<double>(row.waits) / n, share,
		             row.kind == 0 ? "timeline" : "priority", RoleNames[row.role], info.name,
		             info.description);
	}
	const auto locations =
	    std::min(g_location_count.load(std::memory_order_acquire), MaxWaitLocations);
	bool       header    = false;
	for (uint32_t kind = 0; kind < 2; kind++) {
		for (uint32_t id = 0; id < locations; id++) {
			const auto& cell  = g_location_waits[kind][id];
			const auto  waits = cell.waits.load(std::memory_order_relaxed);
			if (waits == 0) {
				continue;
			}
			if (!header) {
				std::fprintf(f, "  unlabelled callers:\n");
				header = true;
			}
			const auto& location = g_locations[id];
			std::fprintf(f, "  %8.3f %8.2f %7s  %-8s %s:%u %s\n",
			             TicksToMs(static_cast<double>(cell.ticks.load(std::memory_order_relaxed)),
			                       tick_hz) /
			                 n,
			             static_cast<double>(waits) / n, "", kind == 0 ? "timeline" : "priority",
			             id == 0 ? "(more locations)" : location.file, location.line,
			             id == 0 ? "" : location.function);
		}
	}
}

// Ranked by time; consumers and the removable time for materialization readbacks.
void WriteReadbackTable(FILE* f, double n, double tick_hz) {
	std::vector<std::pair<ReadbackKey, ReadbackRow>> rows;
	{
		std::lock_guard lock(g_readback_mutex);
		rows.assign(g_readbacks.begin(), g_readbacks.end());
	}
	std::sort(rows.begin(), rows.end(),
	          [](const auto& a, const auto& b) { return a.second.ticks > b.second.ticks; });
	constexpr size_t MaxRows = 60;
	std::fprintf(f,
	             "\nReadbacks by cause and shader (per frame; ReadMemory calls; top %zu of %zu)\n",
	             std::min(MaxRows, rows.size()), rows.size());
	std::fprintf(f, "  %8s %8s  %-6s %-11s %-16s %-18s %9s  %s\n", "ms", "calls", "role", "cause",
	             "where", "shader", "removable", "consumers");
	for (size_t i = 0; i < rows.size() && i < MaxRows; i++) {
		const auto& [key, row] = rows[i];
		const auto role  = static_cast<uint8_t>(key.context >> 24u);
		const auto cause = static_cast<uint8_t>(key.context >> 16u);
		std::fprintf(f, "  %8.3f %8.2f  %-6s %-11s %-16s 0x%016llx %9.3f  %s\n",
		             TicksToMs(static_cast<double>(row.ticks), tick_hz) / n,
		             static_cast<double>(row.calls) / n, role < RoleCount ? RoleNames[role] : "?",
		             cause < std::size(CauseNames) ? CauseNames[cause] : "?",
		             ReadbackWhere(key.context).c_str(),
		             static_cast<unsigned long long>(key.shader),
		             TicksToMs(static_cast<double>(row.removable_ticks), tick_hz) / n,
		             row.attributed != 0 ? ConsumerList(row.consumers).c_str() : "");
	}
}

void WriteSummary(const std::filesystem::path& path, double tick_hz, uint64_t recorded) {
	FILE* f = std::fopen(path.string().c_str(), "w");
	if (f == nullptr) {
		return;
	}
	const double n = static_cast<double>(g_frames.size());
	std::fprintf(f, "Kyty frame profiler capture\n");
	std::fprintf(f, "frames:    %zu (frame %llu to %llu)\n", g_frames.size(),
	             static_cast<unsigned long long>(g_frames.empty() ? 0 : g_frames.front().number),
	             static_cast<unsigned long long>(g_frames.empty() ? 0 : g_frames.back().number));
	std::fprintf(f, "tick rate: %.6f GHz (measured over the capture)\n", tick_hz / 1e9);
	std::fprintf(f, "trace:     %llu events recorded, %llu kept (ring capacity %llu)\n",
	             static_cast<unsigned long long>(recorded),
	             static_cast<unsigned long long>(std::min(recorded, g_ring_capacity)),
	             static_cast<unsigned long long>(g_ring_capacity));

	std::vector<double> times;
	times.reserve(g_frames.size());
	double total = 0.0;
	for (const auto& frame: g_frames) {
		times.push_back(frame.ms);
		total += frame.ms;
	}
	const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
	std::fprintf(f, "\nFrame time (ms)\n");
	std::fprintf(f, "  mean %.2f  median %.2f  p90 %.2f  min %.2f  max %.2f\n",
	             n > 0 ? total / n : 0.0, Median(times), Percentile(times, 0.9),
	             times.empty() ? 0.0 : *min_it, times.empty() ? 0.0 : *max_it);

	auto ms = [&](Counter counter) { return TicksToMs(WindowValue(counter), tick_hz); };
	std::fprintf(f, "\nGPU thread (ms per frame)\n");
	std::fprintf(f, "  processing wall %.2f  processing CPU %.2f  slices %.1f\n",
	             ms(Counter::GpuProcessTime), WindowValue(Counter::GpuProcessCpuTime) / 1e6,
	             WindowValue(Counter::GpuProcessSlices));
	std::fprintf(f, "  idle %.2f (waits %.1f)  blocked polls %.2f (%.1f)  longest wait %.2f\n",
	             ms(Counter::GpuIdleTime), WindowValue(Counter::GpuIdleWaits),
	             ms(Counter::GpuBlockedTime), WindowValue(Counter::GpuBlockedWaits),
	             ms(Counter::GpuLongestWait));

	// Read one command buffer late, at the next one's begin. Busy runs from a command buffer's
	// top-of-pipe timestamp to its bottom one, so a command buffer that blocks on an event counts
	// as busy while it waits: an upper bound.
	std::fprintf(f, "\nGPU execution (ms per frame, timestamps read at completion; busy is an "
	                "upper bound)\n");
	std::fprintf(f,
	             "  busy %.2f (command buffers %.1f, untimed %.1f)  present %.2f\n"
	             "  compactor %.3f  check %.3f  flat copy %.3f  delta upload %.3f  (spans not "
	             "timed %.1f)\n",
	             WindowValue(Counter::GpuBusyTime) / 1e6,
	             WindowValue(Counter::GpuCommandBuffersTimed),
	             WindowValue(Counter::GpuCommandBuffersUntimed),
	             WindowValue(Counter::GpuPresentBusyTime) / 1e6,
	             WindowValue(Counter::GpuRegionTimeCompactor) / 1e6,
	             WindowValue(Counter::GpuRegionTimeCheck) / 1e6,
	             WindowValue(Counter::GpuRegionTimeFlatCopy) / 1e6,
	             WindowValue(Counter::GpuRegionTimeDeltaUpload) / 1e6,
	             WindowValue(Counter::GpuRegionSpansUnmeasured));

	std::fprintf(f, "\nPhases (ms per frame, GPU thread)\n");
	double phase_total = 0.0;
	for (uint32_t p = 0; p < PhaseCount; p++) {
		const double value = TicksToMs(PhaseAverage(p), tick_hz);
		phase_total += value;
		std::fprintf(f, "  %-20s %8.3f\n", PhaseNames[p], value);
	}
	std::fprintf(f, "  %-20s %8.3f\n", "total", phase_total);

	if (n > 0) {
		WriteWaitTable(f, n, tick_hz);
		WriteReadbackTable(f, n, tick_hz);
	}

	std::fprintf(f, "\nCounters (per frame; time in ms, bytes in KiB; [max] is the largest value "
	                "in any frame)\n");
	for (uint32_t c = 0; c < CounterCount; c++) {
		const auto& info  = CounterInfos[c];
		const auto  value = ToDisplay(info, WindowValue(static_cast<Counter>(c)), tick_hz);
		std::fprintf(f, "  %-38s %12.3f %-3s %s%s\n", info.name, value, UnitLabel(info.unit),
		             info.description, info.kind == Kind::Max ? " [max]" : "");
	}

	std::fprintf(f, "\nDebug counters (per frame; the performance panel's, read at frame "
	                "boundaries; time in ms, bytes in KiB)\n");
	for (uint32_t c = 0; c < DebugCounterCount; c++) {
		const auto& info = DebugCounterInfos[c];
		double      sum  = 0.0;
		for (const auto& frame: g_frames) {
			sum += static_cast<double>(frame.debug[c]);
		}
		const CounterInfo display {info.name, "", info.unit, Kind::Sum};
		std::fprintf(f, "  %-38s %12.3f %s\n", info.name,
		             ToDisplay(display, n > 0 ? sum / n : 0.0, tick_hz), UnitLabel(info.unit));
	}

	std::fprintf(f, "\nGauges (change per frame)\n");
	for (uint32_t g = 0; g < GaugeCount; g++) {
		double delta = 0.0;
		if (g_gauge_seen[g].load(std::memory_order_relaxed) && n > 0) {
			delta = static_cast<double>(g_gauge_last[g].load(std::memory_order_relaxed) -
			                            g_gauge_first[g].load(std::memory_order_relaxed)) /
			        n;
		}
		std::fprintf(f, "  %-32s %12.3f\n", GaugeNames[g], delta);
	}

	std::fprintf(f, "\nLevels (first, last, largest sample; bytes in MiB)\n");
	for (uint32_t l = 0; l < LevelCount; l++) {
		if (!g_level_seen[l].load(std::memory_order_relaxed)) {
			std::fprintf(f, "  %-32s %12s\n", LevelInfos[l].name, "-");
			continue;
		}
		const double scale = LevelInfos[l].bytes ? 1.0 / (1024.0 * 1024.0) : 1.0;
		std::fprintf(f, "  %-32s %12.1f %12.1f %12.1f\n", LevelInfos[l].name,
		             static_cast<double>(g_level_first[l].load(std::memory_order_relaxed)) * scale,
		             static_cast<double>(g_level_last[l].load(std::memory_order_relaxed)) * scale,
		             static_cast<double>(g_level_max[l].load(std::memory_order_relaxed)) * scale);
	}
	std::fclose(f);
}

void WriteFrames(const std::filesystem::path& path, double tick_hz) {
	FILE* f = std::fopen(path.string().c_str(), "w");
	if (f == nullptr) {
		return;
	}
	std::fprintf(f, "# frame frame_ms");
	for (const auto* name: PhaseNames) {
		std::fprintf(f, " %s", name);
	}
	std::fprintf(f, " phase_total gpu_process gpu_cpu gpu_idle gpu_blocked (all ms)\n");
	auto ms = [&](const FrameRecord& frame, Counter counter) {
		return TicksToMs(static_cast<double>(frame.counters[static_cast<uint32_t>(counter)]),
		                 tick_hz);
	};
	for (const auto& frame: g_frames) {
		std::fprintf(f, "%llu %.3f", static_cast<unsigned long long>(frame.number), frame.ms);
		double total = 0.0;
		for (uint32_t p = 0; p < PhaseCount; p++) {
			const double value = TicksToMs(static_cast<double>(frame.phases[p]), tick_hz);
			total += value;
			std::fprintf(f, " %.3f", value);
		}
		std::fprintf(
		    f, " %.3f %.3f %.3f %.3f %.3f\n", total, ms(frame, Counter::GpuProcessTime),
		    static_cast<double>(frame.counters[static_cast<uint32_t>(Counter::GpuProcessCpuTime)]) /
		        1e6,
		    ms(frame, Counter::GpuIdleTime), ms(frame, Counter::GpuBlockedTime));
	}
	std::fclose(f);
}

void WriteTrace(const std::filesystem::path& path, double tick_hz, uint64_t recorded) {
	FILE* f = std::fopen(path.string().c_str(), "w");
	if (f == nullptr) {
		return;
	}
	// The first line stays "# base": the symbolizing scripts read it there.
	std::fprintf(f, "# base %llu\n", static_cast<unsigned long long>(ModuleBase()));
	std::fprintf(f, "# kyty frame profiler trace: tsc tid id a b; tsc_hz %.0f\n", tick_hz);
	std::fprintf(f, "# ids");
	for (const auto& entry: EventNames) {
		std::fprintf(f, " %u=%s", static_cast<unsigned>(entry.id), entry.name);
	}
	std::fprintf(f, "\n");
	std::fprintf(f, "# wait-sites");
	for (uint32_t site = 0; site < WaitSiteCount; site++) {
		std::fprintf(f, " %u=%s", site, WaitSiteInfos[site].name);
	}
	std::fprintf(f, "\n");
	{
		std::lock_guard lock(g_location_mutex);
		const auto count =
		    std::min(g_location_count.load(std::memory_order_relaxed), MaxWaitLocations);
		for (uint32_t id = 1; id < count; id++) {
			std::fprintf(f, "# wait-location %u %s:%u %s\n", id, g_locations[id].file,
			             g_locations[id].line, g_locations[id].function);
		}
	}
#if defined(__linux__)
	if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
		char line[512];
		while (std::fgets(line, sizeof(line), maps) != nullptr) {
			std::fprintf(f, "# map %s", line);
		}
		std::fclose(maps);
	}
#endif
	const auto* ring  = g_ring.load(std::memory_order_acquire);
	const auto  kept  = std::min(recorded, g_ring_capacity);
	const auto  first = recorded - kept;
	for (uint64_t i = first; ring != nullptr && i < recorded; i++) {
		const auto& e = ring[i & (g_ring_capacity - 1)];
		std::fprintf(f, "%llu %u %u %llu %llu\n", static_cast<unsigned long long>(e.tsc), e.tid,
		             static_cast<unsigned>(e.id), static_cast<unsigned long long>(e.a),
		             static_cast<unsigned long long>(e.b));
	}
	std::fclose(f);
}

std::filesystem::path MakeOutputDirectory() {
	std::error_code ec;
	auto            root = g_output_root.empty() ? std::filesystem::current_path(ec) : g_output_root;
	const auto      base = "kyty-profile-" + Timestamp();
	auto            dir  = root / base;
	for (int i = 2; std::filesystem::exists(dir, ec); i++) {
		dir = root / (base + "-" + std::to_string(i));
	}
	std::filesystem::create_directories(dir, ec);
	return dir;
}

void ResetCaptureState() {
	for (auto& counter: g_counters) {
		counter.store(0, std::memory_order_relaxed);
	}
	for (auto& phase: g_phases) {
		phase.store(0, std::memory_order_relaxed);
	}
	for (auto& seen: g_gauge_seen) {
		seen.store(false, std::memory_order_relaxed);
	}
	for (uint32_t l = 0; l < LevelCount; l++) {
		g_level_seen[l].store(false, std::memory_order_relaxed);
		g_level_max[l].store(0, std::memory_order_relaxed);
	}
	for (auto& kind: g_waits) {
		for (auto& site: kind) {
			for (auto& cell: site) {
				cell.waits.store(0, std::memory_order_relaxed);
				cell.ticks.store(0, std::memory_order_relaxed);
			}
		}
	}
	for (auto& kind: g_location_waits) {
		for (auto& cell: kind) {
			cell.waits.store(0, std::memory_order_relaxed);
			cell.ticks.store(0, std::memory_order_relaxed);
		}
	}
	{
		std::lock_guard lock(g_readback_mutex);
		g_readbacks.clear();
	}
	g_head.store(0, std::memory_order_relaxed);
	g_frames.clear();
}

void BeginCapture(uint64_t frame_number, uint64_t now_ns) {
	if (g_ring.load(std::memory_order_acquire) == nullptr) {
		g_ring.store(new EventRecord[g_ring_capacity], std::memory_order_release);
	}
	ResetCaptureState();
	g_frames.reserve(g_count);
	g_first_frame   = frame_number;
	g_last_frame_ns = now_ns;
	g_calib_ns      = SteadyNs();
	g_calib_ticks   = Ticks();
	// Counted for the panel too while it is shown; it keeps its own state.
	DebugCounters::SetProfilerCollecting(true);
	for (uint32_t c = 0; c < DebugCounterCount; c++) {
		g_debug_base[c] = DebugCounters::Get(static_cast<DebugCounter>(c));
	}
	g_state.store(State::Capturing, std::memory_order_relaxed);
	g_active.store(true, std::memory_order_release);
	RecordEvent(Event::Frame, frame_number, 0);
}

// Closes the frame that ends at this boundary.
void EndFrame(uint64_t frame_number, uint64_t now_ns) {
	FrameRecord frame;
	frame.number = frame_number;
	frame.ms     = static_cast<double>(now_ns - g_last_frame_ns) / 1e6;
	for (uint32_t c = 0; c < CounterCount; c++) {
		frame.counters[c] = g_counters[c].exchange(0, std::memory_order_relaxed);
	}
	for (uint32_t p = 0; p < PhaseCount; p++) {
		frame.phases[p] = g_phases[p].exchange(0, std::memory_order_relaxed);
	}
	// Read after our counters: a submit adds QueueSubmits before its SubmitsOther or
	// SubmitsPreSubmit, so the frame's QueueSubmits covers them.
	for (uint32_t c = 0; c < DebugCounterCount; c++) {
		const auto value = DebugCounters::Get(static_cast<DebugCounter>(c));
		frame.debug[c]   = value - g_debug_base[c];
		g_debug_base[c]  = value;
	}
	for (const auto& same: SameEvents) {
		frame.counters[static_cast<uint32_t>(same.counter)] =
		    frame.debug[static_cast<uint32_t>(same.debug)];
	}
	const auto submits = frame.debug[static_cast<uint32_t>(DebugCounter::QueueSubmits)];
	const auto others  = frame.counters[static_cast<uint32_t>(Counter::SubmitsPreSubmit)] +
	                    frame.counters[static_cast<uint32_t>(Counter::SubmitsOther)];
	frame.counters[static_cast<uint32_t>(Counter::SubmitsCommand)] =
	    submits - std::min(others, submits);
	g_frames.push_back(frame);
	g_last_frame_ns = now_ns;
	RecordEvent(Event::Frame, frame_number, 0);
}

void FinishCapture() {
	g_active.store(false, std::memory_order_release);
	DebugCounters::SetProfilerCollecting(false);
	g_state.store(State::Idle, std::memory_order_relaxed);
	const auto   end_ticks = Ticks();
	const auto   end_ns    = SteadyNs();
	const double tick_hz =
	    end_ns > g_calib_ns ? static_cast<double>(end_ticks - g_calib_ticks) * 1e9 /
	                              static_cast<double>(end_ns - g_calib_ns)
	                        : 0.0;
	const auto recorded = g_head.load(std::memory_order_acquire);

	for (uint32_t c = 0; c < CounterCount; c++) {
		g_last_average[c] = WindowValue(static_cast<Counter>(c));
	}
	for (uint32_t p = 0; p < PhaseCount; p++) {
		g_last_phase_average[p] = PhaseAverage(p);
	}
	for (uint32_t kind = 0; kind < 2; kind++) {
		for (uint32_t site = 0; site < WaitSiteCount; site++) {
			for (uint32_t role = 0; role < RoleCount; role++) {
				const auto& cell = g_waits[kind][site][role];
				g_last_waits[kind][site][role] = {cell.waits.load(std::memory_order_relaxed),
				                                  cell.ticks.load(std::memory_order_relaxed)};
			}
		}
	}

	const auto dir = MakeOutputDirectory();
	WriteSummary(dir / "summary.txt", tick_hz, recorded);
	WriteFrames(dir / "frames.txt", tick_hz);
	WriteTrace(dir / "trace.txt", tick_hz, recorded);
	g_last_output = dir;
}

} // namespace

const char* EventName(Event id) {
	for (const auto& entry: EventNames) {
		if (entry.id == id) {
			return entry.name;
		}
	}
	return "?";
}

const char* PhaseName(Phase phase) {
	return PhaseNames[static_cast<uint32_t>(phase)];
}

const CounterInfo& GetCounterInfo(Counter counter) {
	return CounterInfos[static_cast<uint32_t>(counter)];
}

const char* GaugeName(Gauge gauge) {
	return GaugeNames[static_cast<uint32_t>(gauge)];
}

const char* LevelName(Level level) {
	return LevelInfos[static_cast<uint32_t>(level)].name;
}

void RecordEvent(Event id, uint64_t a, uint64_t b) {
	auto* ring = g_ring.load(std::memory_order_acquire);
	if (ring == nullptr) {
		return;
	}
	const auto i = g_head.fetch_add(1, std::memory_order_relaxed);
	auto&      e = ring[i & (g_ring_capacity - 1)];
	e.tsc        = Ticks();
	e.tid        = ThreadId();
	e.id         = static_cast<uint16_t>(id);
	e.pad        = 0;
	e.a          = a;
	e.b          = b;
}

void RaiseMax(Counter counter, uint64_t value) {
	auto& slot = g_counters[static_cast<uint32_t>(counter)];
	auto  cur  = slot.load(std::memory_order_relaxed);
	while (value > cur && !slot.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
	}
}

Role CurrentRole() {
	if (t_gpu_thread) {
		return Role::Gpu;
	}
	return ThreadId() == g_render_tid.load(std::memory_order_relaxed) ? Role::Render : Role::Other;
}

uint64_t ThreadCpuNs() {
#if defined(_WIN32)
	FILETIME creation {};
	FILETIME exit {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) == 0) {
		return 0;
	}
	const auto to_u64 = [](const FILETIME& t) {
		return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime;
	};
	return (to_u64(kernel) + to_u64(user)) * 100;
#else
	timespec ts {};
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
#endif
}

void NoteRenderThread() {
	g_render_tid.store(ThreadId(), std::memory_order_relaxed);
}

void SetGaugeValue(Gauge gauge, uint64_t value) {
	const auto index = static_cast<uint32_t>(gauge);
	if (!g_gauge_seen[index].load(std::memory_order_relaxed)) {
		g_gauge_first[index].store(value, std::memory_order_relaxed);
		g_gauge_seen[index].store(true, std::memory_order_relaxed);
	}
	g_gauge_last[index].store(value, std::memory_order_relaxed);
}

void SetLevelValue(Level level, uint64_t value) {
	const auto index = static_cast<uint32_t>(level);
	if (!g_level_seen[index].load(std::memory_order_relaxed)) {
		g_level_first[index].store(value, std::memory_order_relaxed);
		g_level_seen[index].store(true, std::memory_order_relaxed);
	}
	g_level_last[index].store(value, std::memory_order_relaxed);
	auto max = g_level_max[index].load(std::memory_order_relaxed);
	while (value > max &&
	       !g_level_max[index].compare_exchange_weak(max, value, std::memory_order_relaxed)) {
	}
}

void MarkGpuThread() {
	t_gpu_thread = true;
}

void SetUploadSource(UploadSource source) {
	t_upload_source = source;
}

Counter UploadCounter(Counter acquire_counter) {
	static_assert(static_cast<uint32_t>(Counter::UploadCallsCreate) ==
	              static_cast<uint32_t>(Counter::UploadCallsAcquire) + 4);
	static_assert(static_cast<uint32_t>(Counter::UploadCallsBda) ==
	              static_cast<uint32_t>(Counter::UploadCallsAcquire) + 8);
	return static_cast<Counter>(static_cast<uint32_t>(acquire_counter) +
	                            4 * static_cast<uint32_t>(t_upload_source));
}

uint64_t ContextCode() {
	return t_packet | static_cast<uint64_t>(t_phase) << 8u |
	       static_cast<uint64_t>(t_readback_cause) << 16u |
	       static_cast<uint64_t>(CurrentRole()) << 24u;
}

const char* WaitSiteName(WaitSite site) {
	return WaitSiteInfos[static_cast<uint32_t>(site)].name;
}

const char* ConsumerName(uint32_t bit_index) {
	return bit_index < ConsumerCount ? ConsumerNames[bit_index] : "?";
}

void ScopedWait::Begin(const std::source_location& location, uint64_t tick) {
	if (m_site == WaitSite::Other) {
		std::lock_guard lock(g_location_mutex);
		const auto      count = g_location_count.load(std::memory_order_relaxed);
		for (uint32_t id = 1; id < count; id++) {
			if (g_locations[id].line == location.line() &&
			    std::strcmp(g_locations[id].file, location.file_name()) == 0) {
				m_location = id;
				break;
			}
		}
		if (m_location == 0 && count < MaxWaitLocations) {
			g_locations[count] = {location.file_name(), location.line(), location.function_name()};
			g_location_count.store(count + 1, std::memory_order_release);
			m_location = count;
		}
	}
	m_depth = t_wait_depth++;
	if (m_depth < MaxWaitDepth) {
		t_wait_inner[m_depth] = 0;
	}
	RecordEvent(m_priority ? Event::PriorityWaitBegin : Event::HostWaitBegin,
	            static_cast<uint64_t>(m_site) | static_cast<uint64_t>(m_location) << 16u, tick);
}

void ScopedWait::End() {
	const auto total = Ticks() - m_t0;
	const auto inner = m_depth < MaxWaitDepth ? t_wait_inner[m_depth] : 0;
	const auto own   = total - std::min(inner, total);
	t_wait_depth--;
	if (m_depth > 0 && m_depth - 1 < MaxWaitDepth) {
		t_wait_inner[m_depth - 1] += total;
	}
	RecordEvent(m_priority ? Event::PriorityWaitEnd : Event::HostWaitEnd, 0, 0);
	if (!Active()) {
		return;
	}
	const auto role = CurrentRole();
	const auto kind = m_priority ? 1u : 0u;
	auto&      cell = g_waits[kind][static_cast<uint32_t>(m_site)][static_cast<uint32_t>(role)];
	cell.waits.fetch_add(1, std::memory_order_relaxed);
	cell.ticks.fetch_add(own, std::memory_order_relaxed);
	if (m_site == WaitSite::Other) {
		auto& location = g_location_waits[kind][m_location];
		location.waits.fetch_add(1, std::memory_order_relaxed);
		location.ticks.fetch_add(own, std::memory_order_relaxed);
	}
	if (m_depth == 0 && !IsBackgroundSite(m_site)) {
		Add(m_priority ? Counter::PriorityWaitsRender : Counter::HostWaitsRender, role);
		Add(m_priority ? Counter::PriorityWaitTimeRender : Counter::HostWaitTimeRender, role,
		    total);
	}
}

uint16_t ReadbackConsumers(const MaterializeReadback& readback, std::span<const SlotRead> reads) {
	const auto waited = [&](uint64_t address) {
		return std::ranges::any_of(readback.waited, [address](const AddressRange& range) {
			return address < range.end && address + sizeof(uint32_t) > range.begin;
		});
	};
	uint16_t consumers = 0;
	bool     asked     = false;
	for (const auto& read: reads) {
		if (waited(read.address)) {
			consumers |= read.consumers;
		}
		asked = asked || (readback.address >= read.address &&
		                  readback.address < read.address + sizeof(uint32_t));
	}
	if (!asked) {
		return static_cast<uint16_t>(consumers | Bit(Consumer::Unplanned));
	}
	return consumers == 0 ? Bit(Consumer::Copyable) : consumers;
}

void ScopedMaterialize::Begin(uint64_t shader) {
	t_materialize_shader = shader;
	t_collect_readbacks  = true;
	t_materialize_readbacks.clear();
}

void ScopedMaterialize::End() {
	t_materialize_shader = m_previous_shader;
	t_collect_readbacks  = false;
}

std::vector<MaterializeReadback> ScopedMaterialize::Take() {
	if (!m_set) {
		return {};
	}
	t_collect_readbacks = false;
	return std::move(t_materialize_readbacks);
}

void NoteReadback(uint64_t address, std::span<const AddressRange> waited, uint64_t ticks) {
	const auto context = ContextCode();
	const auto shader  = t_materialize_shader != 0 ? t_materialize_shader : CurrentShader();
	{
		std::lock_guard lock(g_readback_mutex);
		auto&           row = g_readbacks[{context, shader}];
		row.calls++;
		row.ticks += ticks;
	}
	if (t_collect_readbacks) {
		t_materialize_readbacks.push_back({
		    .address = address,
		    .waited  = {waited.begin(), waited.end()},
		    .ticks   = ticks,
		    .context = context,
		    .shader  = shader,
		});
	}
}

void NoteMaterializeReadback(const MaterializeReadback& readback, uint16_t consumers) {
	constexpr auto FirstClass = static_cast<uint32_t>(Counter::MaterializeReadbackTimeBufferRead);
	static_assert(static_cast<uint32_t>(Counter::MaterializeReadbackTimeCopyable) ==
	              FirstClass + ConsumerCount - 1);
	const bool removable = RemovableReadback(consumers);
	Add(Counter::MaterializeReadbacks);
	if (removable) {
		Add(Counter::MaterializeReadbackTimeRemovable, readback.ticks);
	}
	for (uint32_t bit = 0; bit < ConsumerCount; bit++) {
		if ((consumers & (1u << bit)) != 0) {
			Add(static_cast<Counter>(FirstClass + bit), readback.ticks);
		}
	}
	if (!Active()) {
		return;
	}
	std::lock_guard lock(g_readback_mutex);
	auto&           row = g_readbacks[{readback.context, readback.shader}];
	row.attributed++;
	row.consumers |= consumers;
	row.removable_ticks += removable ? readback.ticks : 0;
}

void SetCurrentShader(uint64_t hash) {
	if (Active()) {
		t_current_shader = hash;
	}
}

uint64_t CurrentShader() {
	return t_current_shader;
}

void Arm(uint64_t start_frame, uint32_t count) {
	std::lock_guard lock(g_mutex);
	if (g_state.load(std::memory_order_relaxed) == State::Capturing) {
		return;
	}
	g_start_frame = start_frame;
	g_count       = count == 0 ? DefaultFrameCount : count;
	g_state.store(State::Armed, std::memory_order_relaxed);
}

bool Armed() {
	return g_state.load(std::memory_order_relaxed) == State::Armed;
}

bool OnFrame(uint64_t frame_number) {
	if (g_state.load(std::memory_order_relaxed) == State::Idle) {
		return false;
	}
	return OnFrameAt(frame_number, SteadyNs());
}

bool OnFrameAt(uint64_t frame_number, uint64_t now_ns) {
	if (g_state.load(std::memory_order_relaxed) == State::Idle) {
		return false;
	}
	std::lock_guard lock(g_mutex);
	switch (g_state.load(std::memory_order_relaxed)) {
		case State::Idle: return false;
		case State::Armed:
			if (g_start_frame == 0 || frame_number >= g_start_frame) {
				BeginCapture(frame_number, now_ns);
			}
			return false;
		case State::Capturing:
			EndFrame(frame_number, now_ns);
			if (g_frames.size() < g_count) {
				return false;
			}
			FinishCapture();
			return true;
	}
	return false;
}

std::filesystem::path LastOutputDirectory() {
	std::lock_guard lock(g_mutex);
	return g_last_output;
}

void SetOutputRoot(const std::filesystem::path& root) {
	std::lock_guard lock(g_mutex);
	g_output_root = root;
}

bool SetRingCapacity(uint64_t events) {
	std::lock_guard lock(g_mutex);
	if (g_ring.load(std::memory_order_acquire) != nullptr || events == 0 ||
	    (events & (events - 1)) != 0) {
		return false;
	}
	g_ring_capacity = events;
	return true;
}

bool RingAllocated() {
	return g_ring.load(std::memory_order_acquire) != nullptr;
}

uint64_t RecordedEvents() {
	return g_head.load(std::memory_order_acquire);
}

uint64_t CounterValue(Counter counter) {
	return g_counters[static_cast<uint32_t>(counter)].load(std::memory_order_relaxed);
}

double LastFrameAverage(Counter counter) {
	std::lock_guard lock(g_mutex);
	return g_last_average[static_cast<uint32_t>(counter)];
}

double LastPhaseAverageTicks(Phase phase) {
	std::lock_guard lock(g_mutex);
	return g_last_phase_average[static_cast<uint32_t>(phase)];
}

WaitTotals LastWaitSite(bool priority, WaitSite site, Role role) {
	std::lock_guard lock(g_mutex);
	return g_last_waits[priority ? 1 : 0][static_cast<uint32_t>(site)][static_cast<uint32_t>(role)];
}

} // namespace Common::FrameProfiler
