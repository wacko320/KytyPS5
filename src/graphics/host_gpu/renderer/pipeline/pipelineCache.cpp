#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/queuePark.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/cache/srtPageWatch.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/resourcePrefetch.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

// Preparation for a predicted draw (`speculative`): what the real preparation stops the emulator
// on fails instead.
#define KYTY_PRECOMPILE_REQUIRE(speculative, x)                                                    \
	do {                                                                                           \
		if (x) {                                                                                   \
			if (speculative) return {};                                                            \
			EXIT_NOT_IMPLEMENTED(x);                                                               \
		}                                                                                          \
	} while (0)

// What GetGraphicsPrograms compiles: each stage's program, and the push-data cursor the first
// stage starts from.
struct GraphicsStages {
	bool                        tess_active = false;
	std::array<ShaderParams, 3> vertex;
	ShaderParams                pixel;
	uint32_t                    push_data_cursor = 0;
};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

std::optional<vk::PolygonMode> ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front,
                                                  bool cull_back, bool speculative) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	KYTY_PRECOMPILE_REQUIRE(speculative, mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		if (speculative) {
			return std::nullopt;
		}
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default:
			if (speculative) {
				return std::nullopt;
			}
			EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

// Materialization reads descriptors a word at a time, and each checked read costs locks and an
// image lookup. Within one materialization on the GPU thread nothing makes memory GPU-dirty, so
// each 4 KiB page is checked once and later words come straight from its backing view.
struct CheckedReadPages {
	static constexpr size_t          Ways   = 8;
	bool                             active = false;
	std::array<uint64_t, Ways>       tags {};  // page | 1, 0 = empty
	std::array<const uint8_t*, Ways> views {}; // nullptr: not clean as a whole page
	size_t                           next = 0;
};
thread_local CheckedReadPages t_checked_pages;

struct CheckedReadScope {
	CheckedReadScope() {
		t_checked_pages.active = true;
		t_checked_pages.tags.fill(0);
		t_checked_pages.next = 0;
	}
	~CheckedReadScope() { t_checked_pages.active = false; }
	KYTY_CLASS_NO_COPY(CheckedReadScope);
};

// The page's backing view if the whole page is GPU-clean, else nullptr; checked once per scope.
const uint8_t* CheckedPageView(CheckedReadPages& pages, uint64_t page) {
	for (size_t i = 0; i < CheckedReadPages::Ways; i++) {
		if (pages.tags[i] == (page | 1u)) {
			return pages.views[i];
		}
	}
	const auto* view  = Libs::LibKernel::Memory::TryViewGpuCleanBacking(page, 4096);
	const auto  slot  = pages.next++ % CheckedReadPages::Ways;
	pages.tags[slot]  = page | 1u;
	pages.views[slot] = view;
	return view;
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// A dispatch that parked gives up its evaluation (QueuePark).
	if (values.empty() || QueuePark::Pending()) {
		return false;
	}
	const auto bytes = values.size_bytes();
	auto&      pages = t_checked_pages;
	const auto page  = address & ~uint64_t {4095};
	if (pages.active && address + bytes <= page + 4096) {
		const auto* view = CheckedPageView(pages, page);
		if (view != nullptr) {
			std::memcpy(values.data(), view + (address - page), bytes);
			return true;
		}
	}
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), bytes);
}

// Ordinary (unchecked) SRT reads try the checked reader first: on the GPU thread it reads back only
// the GPU-dirty bytes it overlaps, where a plain load through the guest pointer faults on a page
// protected for a GPU write and reads back the whole page. Otherwise they are plain loads. An
// unmapped address (Wolverine reads through a null pointer) reads as zeros, as BDA loads of missing
// pages do, instead of crashing the emulator.
bool ReadOrdinaryGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	if (ReadShaderGuestMemory(userdata, address, values)) {
		return true;
	}
	if (QueuePark::Pending()) {
		return false;
	}
	uint64_t mapped = 0;
	while (mapped < values.size_bytes()) {
		const auto bytes = Libs::LibKernel::Memory::MappingBytesFrom(address + mapped);
		if (bytes == 0) {
			break;
		}
		mapped += bytes;
	}
	if (mapped < values.size_bytes()) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Warning: shader resource read from unmapped address 0x{:016x} reads as zero\n",
			    address));
		}
		std::fill(values.begin(), values.end(), 0u);
		return true;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// The compiled SRT walker loads table levels through the guest pointer. It may, inside a checked
// scope on the GPU thread, where every page of the level is mapped and GPU-clean as a whole
// (the view ReadShaderGuestMemory would read): no GPU-dirty byte to read back, no GPU-written
// image, nothing unmapped (Wolverine's null tables). Otherwise the walk takes
// ReadOrdinaryGuestMemory, which also leaves data-only slots over GPU-written bytes to a GPU copy.
bool CanLoadGuestTableRaw(void*, uint64_t address, uint64_t size) {
	auto& pages = t_checked_pages;
	if (!pages.active || size == 0 || address > UINT64_MAX - size) {
		return false;
	}
	for (auto page = address & ~uint64_t {4095}; page < address + size; page += 4096) {
		if (CheckedPageView(pages, page) == nullptr) {
			return false;
		}
	}
	return true;
}

// Prefetch workers read the backing store: guest pages the GPU thread protects would fault.
const void* ViewGuestPage(uint64_t page) {
	Libs::LibKernel::Memory::GuestBackingView view {};
	return Libs::LibKernel::Memory::GetGuestBackingView(page, ResourcePrefetcher::PageSize, &view)
	           ? view.pointer
	           : nullptr;
}

// GPU thread: whether a prefetch replay that read the page may settle it (SrtPageWatch).
bool IsSettledGuestPage(uint64_t page) {
	const uint64_t profile_t0 = Common::FrameProfiler::Start();
	const bool settled = Libs::LibKernel::Memory::IsBackingSettled(page, SrtPageWatch::PageSize);
	if (profile_t0 != 0) {
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::SettleChecks);
		Common::FrameProfiler::AddSince(Common::FrameProfiler::Counter::SettleCheckTime,
		                                profile_t0);
	}
	return settled;
}

// Samples the prefetch and SRT page-watch statistics for a running frame-profiler capture.
void ProfilePrefetchStats(const ResourcePrefetcher* prefetcher) {
	using Common::FrameProfiler::Gauge;
	using Common::FrameProfiler::SetGaugeValue;
	SetGaugeValue(Gauge::SrtWatchGeneration, SrtPageWatch::Guest().Generation());
	SetGaugeValue(Gauge::SrtWatchSettles, SrtPageWatch::Guest().SettleCount());
	if (prefetcher == nullptr) {
		return;
	}
	const auto& stats = prefetcher->GetStats();
	SetGaugeValue(Gauge::PrefetchSubmitted, stats.submitted);
	SetGaugeValue(Gauge::PrefetchHits, stats.hits);
	SetGaugeValue(Gauge::PrefetchUnmatched, stats.unmatched);
	SetGaugeValue(Gauge::PrefetchMismatched, stats.mismatched);
	SetGaugeValue(Gauge::PrefetchCancelled, stats.cancelled);
	SetGaugeValue(Gauge::PrefetchFailed, stats.failed);
	SetGaugeValue(Gauge::PrefetchStale, stats.stale);
	SetGaugeValue(Gauge::PrefetchThrottled, stats.throttled);
	SetGaugeValue(Gauge::PrefetchUnreplayed, stats.unreplayed);
	SetGaugeValue(Gauge::PrefetchReplayMismatches, stats.replay_mismatches);
	SetGaugeValue(Gauge::PrefetchFlushes, stats.flushes);
	SetGaugeValue(Gauge::PrefetchFlushedJobs, stats.flushed_jobs);
}

// A result accepted on settled pages whose replay differs means a missed SrtPageWatch report
// (MemoryOps::replay_check, debug builds): logged, sparsely.
void ReportReplayMismatches(const ResourcePrefetcher& prefetcher, uint64_t shader_hash) {
	static uint64_t reported   = 0;
	const auto      mismatches = prefetcher.GetStats().replay_mismatches;
	if (mismatches == reported) {
		return;
	}
	reported = mismatches;
	if (mismatches <= 64 || (mismatches & (mismatches - 1)) == 0) {
		LOGF("PipelineCache: a prefetched evaluation accepted on settled pages differs from its "
		     "replay (%" PRIu64 " so far), shader 0x%016" PRIx64 "\n",
		     mismatches, shader_hash);
	}
}

// Profiler only: the dwords a materialization read through its runtime, ordinary and strict, so
// the attribution can evaluate each slot's address from the values the materialization used,
// without reading guest memory again (that could fault and read back on its own).
struct MaterializeReadLog {
	ShaderRecompiler::IR::SrtRuntime           inner;
	std::vector<std::pair<uint64_t, uint32_t>> ordinary;
	std::vector<std::pair<uint64_t, uint32_t>> strict;

	static void Note(std::vector<std::pair<uint64_t, uint32_t>>& log, uint64_t address,
	                 std::span<const uint32_t> values) {
		for (size_t i = 0; i < values.size(); i++) {
			log.emplace_back(address + i * sizeof(uint32_t), values[i]);
		}
	}
	static bool OrdinaryRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto& log = *static_cast<MaterializeReadLog*>(userdata);
		if (!log.inner.read_memory(log.inner.userdata, address, values)) {
			return false;
		}
		Note(log.ordinary, address, values);
		return true;
	}
	static bool StrictRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
		auto& log = *static_cast<MaterializeReadLog*>(userdata);
		if (!log.inner.read_specialization_memory(log.inner.userdata, address, values)) {
			return false;
		}
		Note(log.strict, address, values);
		return true;
	}
	// The logged runtime: the same readers, recording what they return.
	ShaderRecompiler::IR::SrtRuntime Logged() {
		auto runtime     = inner;
		runtime.userdata = this;
		if (inner.read_memory != nullptr) {
			runtime.read_memory = OrdinaryRead;
		}
		if (inner.read_specialization_memory != nullptr) {
			runtime.read_specialization_memory = StrictRead;
		}
		return runtime;
	}

	// Before Replaying: by address, the first read of a dword first.
	void Sort() {
		const auto by_address = [](const auto& a, const auto& b) { return a.first < b.first; };
		std::ranges::stable_sort(ordinary, by_address);
		std::ranges::stable_sort(strict, by_address);
	}
	static bool Replay(const std::vector<std::pair<uint64_t, uint32_t>>& log, uint64_t address,
	                   std::span<uint32_t> values) {
		for (size_t i = 0; i < values.size(); i++) {
			const auto dword = address + i * sizeof(uint32_t);
			const auto found = std::ranges::lower_bound(log, dword, {},
			                                            &std::pair<uint64_t, uint32_t>::first);
			if (found == log.end() || found->first != dword) {
				return false;
			}
			values[i] = found->second;
		}
		return true;
	}
	// A runtime that serves only the logged dwords (sorted): any other read fails.
	ShaderRecompiler::IR::SrtRuntime Replaying() const {
		auto runtime        = inner;
		runtime.userdata    = const_cast<MaterializeReadLog*>(this);
		runtime.gpu_written = nullptr;
		runtime.read_memory = +[](void* userdata, uint64_t address, std::span<uint32_t> values) {
			return Replay(static_cast<const MaterializeReadLog*>(userdata)->ordinary, address,
			              values);
		};
		runtime.read_specialization_memory =
		    +[](void* userdata, uint64_t address, std::span<uint32_t> values) {
			    return Replay(static_cast<const MaterializeReadLog*>(userdata)->strict, address,
			                  values);
		    };
		return runtime;
	}
};

// Profiler only: each readback during a materialization gets the union of the CPU consumers of
// the planned dwords on the pages it waited for (a drain stays while any of them needs the CPU):
// the flat slots and the one-dword reads nested in what the CPU evaluates, uniform-buffer reads
// included. A read whose address does not evaluate from the logged dwords is left out.
void AttributeMaterializeReadbacks(
    const ShaderRecompiler::IR::ResourcePlan&                   plan,
    MaterializeReadLog&                                         log,
    std::span<const Common::FrameProfiler::MaterializeReadback> readbacks) {
	using Common::FrameProfiler::Bit;
	using Common::FrameProfiler::Consumer;
	if (readbacks.empty()) {
		return;
	}
	if (plan.srt_read_consumers.size() != plan.srt_reads.size()) {
		for (const auto& readback: readbacks) {
			Common::FrameProfiler::NoteMaterializeReadback(readback, Bit(Consumer::Unplanned));
		}
		return;
	}
	std::vector<Common::FrameProfiler::SlotRead> reads;
	reads.reserve(plan.srt_reads.size());
	log.Sort();
	const auto                      replaying     = log.Replaying();
	const auto                      clean_runtime = ShaderRecompiler::IR::CleanRuntime(replaying);
	ShaderRecompiler::IR::SrtWalker clean(plan, clean_runtime);
	ShaderRecompiler::IR::SrtWalker walker(plan, replaying, &clean);
	for (uint32_t slot = 0; slot < plan.srt_reads.size(); slot++) {
		uint64_t address = 0;
		if (walker.EvaluateLoadAddress(plan.srt_reads[slot].value, address)) {
			reads.push_back({address, plan.srt_read_consumers[slot]});
		}
	}
	for (const auto& [read, consumers]: plan.srt_nested_reads) {
		uint64_t address = 0;
		if (walker.EvaluateLoadAddress(read, address)) {
			reads.push_back({address, consumers});
		}
	}
	for (const auto& readback: readbacks) {
		Common::FrameProfiler::NoteMaterializeReadback(
		    readback, Common::FrameProfiler::ReadbackConsumers(readback, reads));
	}
}

// The draw-prep prefetcher's worker count.
uint32_t DrawPrepThreads() {
	// Two keep up with a draw every ~20 us; more mostly compete with the guest's threads.
	return std::clamp(std::thread::hardware_concurrency() / 6u, 1u, 3u);
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

bool CanLoadSrtTableRaw(uint64_t address, uint64_t size) {
	CheckedReadScope scope;
	return CanLoadGuestTableRaw(nullptr, address, size);
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		// SourceEntry::plan_generation it was compiled for.
		uint32_t                                    plan_generation = 0;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		// The linked code's key: the leaf callee's words, or a hash of a table-linked program.
		std::span<const uint32_t> FunctionCode() const {
			return call_source ? std::span<const uint32_t>(call_source->linked->key)
			                   : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// Materializes the plan. When an indirect image table cannot be specialized for this
		// runtime, its image is added to bindless_pcs and false is returned: the program must be
		// translated again (with CompileOptions::bindless_image_pcs) and its plan extracted again.
		// A prefetch worker's result is used when one was accepted (ResourcePrefetcher).
		bool Materialize(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                 ResourcePrefetcher* prefetcher) {
			if (!Common::FrameProfiler::Active()) {
				return MaterializeEntry(runtime, prefetcher);
			}
			// Profiling: the same readers, logged, so its readbacks can be attributed afterwards.
			Common::FrameProfiler::ScopedMaterialize profile_readbacks(resource_plan.shader_hash);
			MaterializeReadLog                       log {.inner = runtime};
			const bool materialized = MaterializeEntry(log.Logged(), prefetcher);
			AttributeMaterializeReadbacks(resource_plan, log, profile_readbacks.Take());
			return materialized;
		}

		bool MaterializeEntry(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                      ResourcePrefetcher* prefetcher) {
			using Common::FrameProfiler::Counter;
			const bool profiling = Common::FrameProfiler::Active();
			const Common::FrameProfiler::ScopedPhase profile_phase(
			    Common::FrameProfiler::Phase::Materialize, profiling);
			if (profiling) {
				Common::FrameProfiler::Add(Counter::Materializations);
				Common::FrameProfiler::Add(Counter::MaterializeSrtReads,
				                           resource_plan.srt_reads.size());
				ProfilePrefetchStats(prefetcher);
			}
			const CheckedReadScope checked_reads;
			if (prefetcher != nullptr && !replica_views.empty()) {
				const uint64_t try_t0 = profiling ? Common::FrameProfiler::Ticks() : 0;
				const bool     hit =
				    prefetcher->TryMaterialize(this, runtime, resources, specialization);
				if (try_t0 != 0) {
					const uint64_t ticks = Common::FrameProfiler::Ticks() - try_t0;
					Common::FrameProfiler::Add(Counter::PrefetchTries);
					Common::FrameProfiler::Add(Counter::PrefetchTryTime, ticks);
					if (hit) {
						Common::FrameProfiler::Add(Counter::PrefetchHits);
						Common::FrameProfiler::Add(Counter::PrefetchHitTime, ticks);
					}
				}
				ReportReplayMismatches(*prefetcher, resource_plan.shader_hash);
				// A parked dispatch's result is discarded: it evaluates again.
				if (hit || QueuePark::Pending()) {
					return true;
				}
			}
			const uint64_t plain_t0 = profiling ? Common::FrameProfiler::Ticks() : 0;
			std::vector<uint32_t> pcs;
			// A parked attempt fails every read after the park and evaluates again: its failures
			// are not reported.
			ShaderRecompiler::IR::DeferredSpecializationFailureReports reports;
			const bool materialized = ShaderRecompiler::IR::MaterializeResources(
			    resource_plan, runtime, resources, specialization, &pcs);
			if (plain_t0 != 0) {
				Common::FrameProfiler::Add(Counter::PlainMaterializations);
				Common::FrameProfiler::AddSince(Counter::PlainMaterializeTime, plain_t0);
			}
			if (QueuePark::Pending()) {
				return true;
			}
			reports.Publish();
			if (materialized) {
				return true;
			}
			bool added = false;
			for (const auto pc: pcs) {
				if (std::ranges::find(bindless_pcs, pc) == bindless_pcs.end()) {
					bindless_pcs.push_back(pc);
					added = true;
					std::printf("Shader 0x%016" PRIx64 ": indirect image table at pc 0x%08x falls "
					            "back to the bindless heap\n",
					            resource_plan.shader_hash, pc);
				}
			}
			EXIT_IF(!added);
			++plan_generation;
			return false;
		}

		template <bool HasCalls>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor, ResourcePrefetcher* prefetcher) {
			if (!Materialize(runtime, prefetcher)) {
				return nullptr;
			}
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.plan_generation != plan_generation ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			return nullptr;
		}

		// A table-linked compute kernel's last dispatch, to link and build it ahead when a table
		// it shares changes (BuildSiblingsAhead).
		struct ComputeSeed {
			std::vector<uint32_t>  user_data;
			uint64_t               shader_base = 0;
			uint64_t               hash        = 0;
			ShaderComputeInputInfo input;
		};

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		std::unique_ptr<ComputeSeed>                    seed;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		// Tier-2 bindless fallback: first-use pcs of images whose tables could not be specialized
		// (sticky for the entry), and how often the plan changed because of them.
		std::vector<uint32_t>                       bindless_pcs;
		uint32_t                                    plan_generation = 0;
		// Where the last draw's permutation left the push-data cursor, for the stage a
		// precompile compiles after this one (UINT32_MAX: not known).
		uint32_t                                    push_end = UINT32_MAX;
		// The permutation the last draw ran, for the pipeline a precompile builds with it.
		size_t                                      last_permutation = SIZE_MAX;
		// One independent copy of the plan per prefetch worker: evaluation keeps scratch in
		// the plan. Empty when the plan is not speculated.
		std::vector<std::unique_ptr<ShaderRecompiler::IR::ResourcePlan>> replicas;
		std::vector<const ShaderRecompiler::IR::ResourcePlan*>           replica_views;
		// Plans of table-linked programs the call source linked before, by linked-code key: a
		// table that switches between target sets (one per frame, say) relinks to code it has
		// translated before, and gets that code's plan back instead of a new translation.
		struct LinkedPlan {
			std::vector<uint32_t>                                            key;
			uint32_t                                                         plan_generation = 0;
			ShaderRecompiler::IR::ResourcePlan                               plan;
			std::vector<std::unique_ptr<ShaderRecompiler::IR::ResourcePlan>> replicas;
		};
		static constexpr size_t     MaxLinkedPlans = 4;
		std::vector<LinkedPlan>     linked_plans;
	};

	// How a stage's MaterializeResources input follows from its registers, learned from the
	// last draw that ran the shader at an address.
	enum class PrefetchSlot : uint32_t { Pixel, Vertex, Compute, Count };
	struct Prediction {
		SourceEntry*            entry        = nullptr;
		uint64_t                shader_base  = 0;
		uint32_t                prefix_count = 0;
		std::array<uint32_t, 8> prefix {};
		// A GS back half receives its user-data address in s0:s1.
		bool prefix_is_gs_address = false;
	};

	// A table-linked entry relinked from `previous_key` to its current code: keeps the plan of
	// the code it leaves, and takes back the plan of the code it returns to. True when the
	// entry's plan now belongs to its current code (no translation needed).
	bool SwapLinkedPlan(SourceEntry& entry, std::vector<uint32_t> previous_key) {
		const auto key   = entry.FunctionCode();
		auto&      plans = entry.linked_plans;
		const auto found = std::ranges::find_if(plans, [&](const SourceEntry::LinkedPlan& plan) {
			return plan.plan_generation == entry.plan_generation && std::ranges::equal(plan.key, key);
		});
		if (found == plans.end() && previous_key.empty()) {
			return false;
		}
		// Prefetch workers may still evaluate the replicas that move.
		if (prefetcher != nullptr && !entry.replicas.empty()) {
			prefetcher->ClearAndWait();
		}
		entry.replica_views.clear();
		SourceEntry::LinkedPlan left {.key             = std::move(previous_key),
		                              .plan_generation = entry.plan_generation,
		                              .plan            = std::move(entry.resource_plan),
		                              .replicas        = std::move(entry.replicas)};
		entry.replicas.clear();
		const bool restored = found != plans.end();
		if (restored) {
			entry.resource_plan = std::move(found->plan);
			entry.replicas      = std::move(found->replicas);
			plans.erase(found);
			entry.resource_plan.source_reads = entry.call_source->reads;
			for (const auto& replica: entry.replicas) {
				replica->source_reads = entry.call_source->reads;
				entry.replica_views.push_back(replica.get());
			}
		}
		if (!left.key.empty()) {
			if (plans.size() >= SourceEntry::MaxLinkedPlans) {
				plans.erase(plans.begin());
			}
			plans.push_back(std::move(left));
		}
		return restored;
	}

	// After the entry's plan was (re)extracted from `program`: one copy per worker. Workers may
	// still be evaluating the old copies, so the prefetcher drops and waits for their jobs.
	void ResetReplicas(SourceEntry& entry, const ShaderRecompiler::IR::Program& program) {
		if (prefetcher == nullptr) {
			return;
		}
		if (!entry.replicas.empty()) {
			prefetcher->ClearAndWait();
			entry.replica_views.clear();
			entry.replicas.clear();
		}
		if (!ResourcePrefetcher::CanSpeculate(entry.resource_plan)) {
			return;
		}
		for (uint32_t i = 0; i < prefetcher->WorkerCount(); i++) {
			entry.replicas.push_back(std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
			    ShaderRecompiler::IR::ExtractResourcePlan(program)));
			entry.replica_views.push_back(entry.replicas.back().get());
		}
	}

	void Learn(PrefetchSlot slot, uint64_t shader_address, const ShaderParams& params,
	           uint32_t sgpr_count, uint64_t gs_user_data_address) {
		auto*      entry     = last_materialized;
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		if (prefetcher == nullptr || entry == nullptr || entry->replica_views.empty() ||
		    sgpr_count > user_data.size() ||
		    user_data.size() - sgpr_count > Prediction {}.prefix.size()) {
			return;
		}
		auto& prediction        = predictions[static_cast<uint32_t>(slot)][shader_address];
		prediction.entry        = entry;
		prediction.shader_base  = params.Base();
		prediction.prefix_count = static_cast<uint32_t>(user_data.size()) - sgpr_count;
		std::copy_n(user_data.begin(), prediction.prefix_count, prediction.prefix.begin());
		prediction.prefix_is_gs_address =
		    prediction.prefix_count >= 2 && gs_user_data_address != 0 &&
		    user_data[0] == static_cast<uint32_t>(gs_user_data_address) &&
		    user_data[1] == static_cast<uint32_t>(gs_user_data_address >> 32u);
	}

	void Speculate(uint64_t draw, PrefetchSlot slot, uint64_t shader_address,
	               const HW::UserSgprInfo& sgprs, uint32_t sgpr_count,
	               uint64_t gs_user_data_address, std::span<const uint32_t> workgroup_counts) {
		const auto& table = predictions[static_cast<uint32_t>(slot)];
		const auto  found = table.find(shader_address);
		if (found == table.end() || sgpr_count > HW::UserSgprInfo::SGPRS_MAX) {
			return;
		}
		const auto& prediction = found->second;
		std::array<uint32_t, 8 + HW::UserSgprInfo::SGPRS_MAX> user_data {};
		std::copy_n(prediction.prefix.begin(), prediction.prefix_count, user_data.begin());
		if (prediction.prefix_is_gs_address) {
			user_data[0] = static_cast<uint32_t>(gs_user_data_address);
			user_data[1] = static_cast<uint32_t>(gs_user_data_address >> 32u);
		}
		std::copy_n(sgprs.value, sgpr_count, user_data.begin() + prediction.prefix_count);
		prefetcher->Submit(draw, prediction.entry, prediction.entry->replica_views,
		                   std::span(user_data).first(prediction.prefix_count + sgpr_count),
		                   prediction.shader_base, workgroup_counts);
	}

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;
	// Linked kernels built ahead after one relink; the queue bounds the workers.
	static constexpr uint32_t MaxSiblingBuilds = 4;

	ShaderRecompiler::CompileOptions BaseOptions(ShaderType stage, uint64_t hash,
	                                             std::span<const uint32_t> user_data,
	                                             const char*               label) const {
		ShaderRecompiler::CompileOptions options;
		options.stage               = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash         = hash;
		options.loop_watchdog_limit = Config::ShaderLoopLimit();
		options.untracked_writes    = true;
		options.bindless_images     = true;
		options.table_calls         = true;
		options.user_data           = user_data;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		return options;
	}

	void RecordComputeSeed(SourceEntry& entry, const ShaderParams& params,
	                       const ShaderComputeInputInfo& input_info) {
		if (!entry.call_source || !entry.call_source->table) return;
		if (!entry.seed) entry.seed = std::make_unique<SourceEntry::ComputeSeed>();
		auto& seed = *entry.seed;
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		seed.user_data.assign(user_data.begin(), user_data.end());
		seed.shader_base = params.Base();
		seed.hash        = params.hash;
		seed.input       = input_info;
		seed.input.stage = {};
	}

	// `changed` just relinked because call tables changed. Linked compute kernels that read their
	// targets from one of those tables (two passes of one ray-tracing frame) are likely to relink
	// at their next dispatch against the same new callees: link them now from their last
	// dispatch's user data, translate them here (guest reads stay on the GPU thread) and build
	// their pipelines on a worker. Their own dispatch still links, translates and checks the
	// result: it adopts the prebuilt pipeline only for the same ComputePipelineBuildKey
	// (PipelineCache::GetComputePipeline), waiting for it if needed.
	void BuildSiblingsAhead(const SourceEntry& changed) {
		if (!submit_build) return;
		uint32_t started = 0;
		for (auto& [key, sibling]: programs) {
			if (&sibling == &changed || key.stage != ShaderType::Compute || !sibling.seed ||
			    !sibling.call_source || !sibling.call_source->table ||
			    !ShaderRecompiler::SharesChangedCallTable(*changed.call_source,
			                                              *sibling.call_source))
				continue;
			if (started++ == MaxSiblingBuilds) break;
			BuildAhead(sibling, changed.resource_plan.shader_hash);
		}
	}

	void BuildAhead(const SourceEntry& sibling, uint64_t trigger_hash) {
		const auto& seed   = *sibling.seed;
		auto        source = ShaderRecompiler::CloneTableSource(*sibling.call_source);
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = seed.user_data,
		    .shader_base                = seed.shader_base,
		    .read_memory                = ReadOrdinaryGuestMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .gpu_written                = Libs::LibKernel::Memory::IsGpuWrittenUnread,
		};
		runtime.workgroup_counts = seed.input.workgroup_counts;
		ShaderRecompiler::RefreshShaderSource(*source, runtime);
		if (sibling.call_source->linked && source->linked->key == sibling.call_source->linked->key)
			return; // its tables did not change for it
		auto                 input = seed.input;
		ShaderStageInputInfo stage_input {};
		stage_input.compute = &input;
		auto options = BaseOptions(ShaderType::Compute, seed.hash, seed.user_data,
		                           "ShaderRecompiler CS");
		options.input_info         = stage_input;
		options.wave_size          = input.wave_size;
		options.bindless_image_pcs = sibling.bindless_pcs;
		ShaderRecompiler::IR::ResourceSnapshot       snapshot;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::TranslateResult            translated;
		// As Get: tables that cannot be specialized move to the bindless heap.
		for (uint32_t attempt = 0;; attempt++) {
			translated = ShaderRecompiler::TranslateProgram(source->linked->decoded, options);
			translated.program.source_reads = source->reads;
			const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			std::vector<uint32_t> pcs;
			bool                  materialized = false;
			{
				const CheckedReadScope checked_reads;
				const ShaderRecompiler::IR::SpecializationFailureReports quiet(false);
				materialized = ShaderRecompiler::IR::MaterializeResources(
				    plan, runtime, snapshot, specialization, &pcs);
			}
			if (materialized) break;
			bool added = false;
			for (const auto pc: pcs) {
				if (std::ranges::find(options.bindless_image_pcs, pc) ==
				    options.bindless_image_pcs.end()) {
					options.bindless_image_pcs.push_back(pc);
					added = true;
				}
			}
			if (!added || attempt == 3) return;
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               std::move(specialization), 0);
		auto program = std::make_shared<const ShaderRecompiler::IR::CompiledShaderInfo>(
		    std::move(result.program).TakeCompiledInfo());
		const auto key = ComputePipelineBuildKey(*program, result.spirv);
		if (submit_build(key, std::make_shared<const std::vector<uint32_t>>(std::move(result.spirv)),
		                 std::move(program))) {
			std::printf("Shader 0x%016" PRIx64 ": a shared call table changed; building linked "
			            "kernel 0x%016" PRIx64 " ahead (key 0x%016" PRIx64 ")\n",
			            trigger_hash, seed.hash, key);
		}
	}

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code,
	                               bool                      record_build_key = false) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		Permutation permutation {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
		if (record_build_key) {
			build_keys[permutation.handle.id] =
			    ComputePipelineBuildKey(permutation.program, result.spirv);
		}
		return permutation;
	}

	// The options Get translates a program for a stage with.
	template <typename InputInfo>
	ShaderRecompiler::CompileOptions StageOptions(ShaderType stage, const ShaderParams& params,
	                                              InputInfo& input_info,
	                                              const char*& stage_name) const {
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto options       = BaseOptions(stage, params.hash, user_data, label);
		options.back_code  = params.back_code;
		options.input_info = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		last_materialized    = nullptr;
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory                = ReadOrdinaryGuestMemory,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .gpu_written                = Libs::LibKernel::Memory::IsGpuWrittenUnread,
		    .raw_table_reads            = CanLoadGuestTableRaw,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
			if (input_info.dispatch_thread_indirect) {
				runtime.indirect_args_group_size = input_info.threads_num;
			}
		}
		bool source_changed = false;
		uint32_t plan_generation = 0;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			plan_generation = cached.plan_generation;
			Permutation* permutation = nullptr;
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				RecordComputeSeed(cached, params, input_info);
			}
			if (cached.call_source) {
				std::vector<uint32_t> previous_key;
				if (cached.call_source->linked) {
					const auto key = cached.FunctionCode();
					previous_key.assign(key.begin(), key.end());
				}
				source_changed = cached.RefreshCallSource(runtime);
				if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
					if (source_changed) BuildSiblingsAhead(cached);
				}
				// Code linked before: its plan comes back, and no translation is needed.
				if (source_changed && SwapLinkedPlan(cached, std::move(previous_key))) {
					source_changed = false;
				}
				if (!source_changed) {
					// A cached plan's evaluation writes only outputs the next one rewrites: a
					// dispatch may give up in it and run again (QueuePark).
					const QueuePark::RestartableScope restartable;
					permutation = cached.FindPermutation<true>(runtime, push_data_cursor,
					                                           prefetcher.get());
				}
			} else {
				const QueuePark::RestartableScope restartable;
				permutation =
				    cached.FindPermutation<false>(runtime, push_data_cursor, prefetcher.get());
			}
			if (QueuePark::Pending()) {
				// The dispatch returns before recording, and runs again.
				return {};
			}
			if (permutation != nullptr) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				last_materialized = &entry->second;
				entry->second.last_permutation =
				    static_cast<size_t>(permutation - entry->second.permutations.data());
				return permutation->handle;
			}
		}

		const char* stage_name = nullptr;
		auto        options    = StageOptions(stage, params, input_info, stage_name);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		// A new program a worker translated ahead (TakePrecompiled): its plan stands in for the
		// translation, and its permutation for the compile when the draw materializes the same
		// specialization.
		std::optional<PrecompiledStage> precompiled;
		if (entry == programs.end()) {
			precompiled = TakePrecompiled(lookup_key);
		}
		const bool adopt_plan = precompiled && precompiled->translated;
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end() && !adopt_plan) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == programs.end() ? (adopt_plan ? nullptr : &source)
		                                        : entry->second.call_source.get();
		bool translated_here = false;
		const auto translate = [&]() {
			++translations;
			translated_here = true;
			if (entry != programs.end()) options.bindless_image_pcs = entry->second.bindless_pcs;
			ShaderRecompiler::TranslateResult result;
			if (current != nullptr) {
				if (current->call) {
					if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
					result = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
					result.program.source_reads = current->reads;
				} else {
					result = ShaderRecompiler::TranslateProgram(current->decoded, options);
				}
			} else {
				result = ShaderRecompiler::TranslateProgram(params.code, options);
			}
			return result;
		};
		ShaderRecompiler::TranslateResult translated;
		if (!adopt_plan) {
			translated = translate();
		}
		// Tables that cannot be specialized move to the bindless heap; each retry adds new pcs.
		const auto materialize = [&]() {
			while (!entry->second.Materialize(runtime, prefetcher.get())) {
				translated = translate();
				entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				ResetReplicas(entry->second, translated.program);
			}
		};
		if (entry == programs.end() && adopt_plan) {
			entry = programs.try_emplace(lookup_key, std::move(precompiled->plan), nullptr).first;
			AdoptReplicas(entry->second, *precompiled);
			NotePrecompileKnown(stage, params.Base());
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				RecordComputeSeed(entry->second, params, input_info);
			}
			materialize();
		} else if (entry == programs.end()) {
			auto retained = source.call ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			ResetReplicas(entry->second, translated.program);
			NotePrecompileKnown(stage, params.Base());
			current = entry->second.call_source.get();
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				RecordComputeSeed(entry->second, params, input_info);
			}
			materialize();
		} else if (source_changed || plan_generation != entry->second.plan_generation) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			ResetReplicas(entry->second, translated.program);
			materialize();
			if (const auto* permutation = entry->second.FindPermutation<true>(
			        runtime, push_data_cursor, prefetcher.get())) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				last_materialized = &entry->second;
				entry->second.last_permutation =
				    static_cast<size_t>(permutation - entry->second.permutations.data());
				return permutation->handle;
			}
		}
		// The worker's permutation, when the draw materialized what the worker compiled for:
		// the same plan (no retranslation here), specialization and push-data cursor.
		if (precompiled && precompiled->compiled && !translated_here && !QueuePark::Pending() &&
		    entry->second.plan_generation == 0 &&
		    precompiled->push_cursor == push_data_cursor &&
		    precompiled->specialization == entry->second.specialization) {
			entry->second.permutations.push_back(AdoptPermutation(*precompiled));
			precompile_stats.hits++;
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::PrecompileHits);
		} else {
			if (precompiled) {
				DisposePrecompiledStage(*precompiled);
			}
			if (!translated_here) {
				translated = translate();
			}
			const bool table_linked_compute = stage == ShaderType::Compute &&
			                                  entry->second.call_source &&
			                                  entry->second.call_source->table.has_value();
			entry->second.permutations.push_back(CompilePermutation(
			    stage_name, options, std::move(translated), entry->second.specialization,
			    push_data_cursor, entry->second.FunctionCode(), table_linked_compute));
			precompile_stats.misses++;
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::PrecompileMisses);
		}
		entry->second.permutations.back().plan_generation = entry->second.plan_generation;
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		last_materialized = &entry->second;
		entry->second.last_permutation = entry->second.permutations.size() - 1;

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	// Programs compiled ahead (PipelineCache::PrecompileDraw/PrecompileDispatch). The lookahead
	// predicts a draw's registers; when a stage's ProgramKey is not cached yet, a worker prepares
	// what Get would for a new entry: the source, the translation and its resource plan, and,
	// when it can materialize the plan from the backing store, the permutation for that
	// specialization and push-data cursor, with its shader module (and, for compute, its
	// pipeline). Get takes it only for the exact ProgramKey, and the permutation only for the
	// specialization and cursor the draw materializes itself; anything else is compiled as before
	// and the worker's work is dropped.
	struct PrecompiledStage {
		ProgramKey key;
		// `plan` holds the translation's resource plan (a linked source is left to Get).
		bool                                                             translated = false;
		ShaderRecompiler::IR::ResourcePlan                               plan;
		std::vector<std::unique_ptr<ShaderRecompiler::IR::ResourcePlan>> replicas;
		// `program` and `module` hold the permutation compiled for `specialization` at
		// `push_cursor`.
		bool                                         compiled    = false;
		uint32_t                                     push_cursor = 0;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		vk::ShaderModule                             module = nullptr;
		std::optional<Pipeline>                      compute_pipeline;
	};
	using PrecompileInput =
	    std::variant<ShaderVertexInputInfo, ShaderPixelInputInfo, ShaderComputeInputInfo>;
	struct PrecompileRequest {
		ShaderType      stage = ShaderType::Unknown;
		ProgramKey      key;
		ShaderParams    params;
		PrecompileInput input;
		// The push-data cursor the stage starts at; nullopt: where the previous request's
		// program leaves it.
		std::optional<uint32_t> push_cursor;
	};
	// A program the draw already has: the permutation it ran last.
	struct PrecompileKnownProgram {
		uint64_t                                 id     = 0;
		vk::ShaderModule                         module = nullptr;
		ShaderRecompiler::IR::CompiledShaderInfo program;
	};
	// The draw's graphics pipeline, built after the bundle's stages from the predicted targets
	// and state. Each stage is a known program, or the bundle's stage with that key.
	struct PrecompilePipelineRequest {
		std::unique_ptr<HW::Context>            context;
		Prospero::PrimitiveType                 primitive_type = Prospero::PrimitiveType::kNone;
		vk::PrimitiveTopology                   topology       = vk::PrimitiveTopology::ePointList;
		uint32_t                                color_output_mask = 0;
		PipelineDepthTarget                     depth;
		ShaderVertexInputInfo                   vertex_info;
		std::optional<ShaderPixelInputInfo>     pixel_info;
		ProgramKey                              vertex_key;
		ProgramKey                              pixel_key;
		std::optional<PrecompileKnownProgram>   known_vertex;
		std::optional<PrecompileKnownProgram>   known_pixel;
	};
	// One worker job: the new stages of one draw, in the order Get takes them, and the draw's
	// pipeline keyed by ProvisionalProgramId for the bundle's programs.
	struct PrecompileBundle {
		std::vector<PrecompiledStage>                            stages;
		std::optional<std::pair<GraphicsPipelineKey, Pipeline>> pipeline;
	};

	// The id a graphics pipeline built ahead uses for a bundle's program until a draw adopts it
	// (ShaderProgram ids count up from 1 and never reach the top bit).
	static uint64_t ProvisionalProgramId(const ProgramKey& key) {
		return PrecompileKeyHash(key) | (uint64_t {1} << 63u);
	}

	static constexpr uint32_t MaxPrecompileBundles = 64;
	static constexpr size_t   MaxPrecompiledReady  = 64;

	// The worker count: compiles are long and each holds a large working set, and the GPU thread
	// and the guest's threads come first.
	static uint32_t PrecompileThreads() {
		return std::clamp(std::thread::hardware_concurrency() / 6u, 1u, 2u);
	}

	template <typename InputInfo>
	static ProgramKey MakeProgramKey(ShaderType stage, const ShaderParams& params,
	                                 const InputInfo& input_info) {
		ProgramKey key;
		key.stage           = stage;
		key.hash            = params.hash;
		key.user_data_count = params.user_data_count;
		key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, key.static_state);
		return key;
	}

	static uint64_t PrecompileKeyHash(const ProgramKey& key) {
		const std::array<uint64_t, 4> head {static_cast<uint64_t>(key.stage), key.hash,
		                                    key.user_data_count, key.code_size};
		return XXH3_64bits_withSeed(key.static_state.data(),
		                            key.static_state.size() * sizeof(uint32_t),
		                            XXH3_64bits(head.data(), sizeof(head)));
	}

	// Worker threads: guest words from the backing store, without faults or readbacks. Stale
	// words only make the worker's specialization differ from the draw's.
	static bool ReadPrecompileMemory(void*, uint64_t address, std::span<uint32_t> values) {
		if (values.empty()) {
			return false;
		}
		auto*      out   = reinterpret_cast<uint8_t*>(values.data());
		const auto bytes = values.size_bytes();
		for (uint64_t done = 0; done < bytes;) {
			const auto at   = address + done;
			const auto page = at & ~(ResourcePrefetcher::PageSize - 1);
			const auto* view = static_cast<const uint8_t*>(ViewGuestPage(page));
			if (view == nullptr) {
				return false;
			}
			const auto count = std::min(bytes - done, page + ResourcePrefetcher::PageSize - at);
			std::memcpy(out + done, view + (at - page), count);
			done += count;
		}
		return true;
	}

	void DestroyPipeline(Pipeline& pipeline) const {
		graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
		graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	}

	// A stage nobody takes: its module and pipeline go. Any thread.
	void DisposePrecompiledStage(PrecompiledStage& stage) {
		if (stage.module != nullptr) {
			graphics.device.destroyShaderModule(stage.module, nullptr);
			stage.module = nullptr;
		}
		if (stage.compute_pipeline) {
			DestroyPipeline(*stage.compute_pipeline);
			stage.compute_pipeline.reset();
		}
		precompile_wasted.fetch_add(1, std::memory_order_relaxed);
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::PrecompileWasted);
	}

	// Worker thread: Get's new-entry path for one stage, from the predicted inputs.
	template <typename InputInfo>
	void BuildPrecompiledStage(PrecompileRequest& request, InputInfo& input_info,
	                           std::optional<uint32_t> cursor, uint32_t worker_count,
	                           vk::PipelineCache driver_cache, PrecompiledStage& out) const {
		const char* stage_name = nullptr;
		auto        options = StageOptions(request.stage, request.params, input_info, stage_name);
		// Dumps belong to the draw's own compile.
		options.dump_ir    = false;
		options.early_dump = false;
		auto source = ShaderRecompiler::PrepareShaderSource(request.params.code, options);
		if (source.call) {
			return; // a linked source reads its call targets on the GPU thread
		}
		auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
		out.plan        = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (worker_count != 0 && ResourcePrefetcher::CanSpeculate(out.plan)) {
			for (uint32_t i = 0; i < worker_count; i++) {
				out.replicas.push_back(std::make_unique<ShaderRecompiler::IR::ResourcePlan>(
				    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)));
			}
		}
		out.translated = true;
		if (!cursor) {
			return;
		}
		const auto user_data =
		    std::span(request.params.user_data).first(request.params.user_data_count);
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = user_data,
		    .shader_base                = request.params.Base(),
		    .read_memory                = ReadPrecompileMemory,
		    .read_specialization_memory = ReadPrecompileMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts      = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
			if (input_info.dispatch_thread_indirect) {
				runtime.indirect_args_group_size = input_info.threads_num;
			}
		}
		ShaderRecompiler::IR::ResourceSnapshot       snapshot;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<uint32_t>                        pcs;
		{
			const ShaderRecompiler::IR::SpecializationFailureReports quiet(false);
			if (!ShaderRecompiler::IR::MaterializeResources(out.plan, runtime, snapshot,
			                                                specialization, &pcs)) {
				return; // the draw moves tables to the bindless heap and translates again
			}
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, *cursor);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			return; // the draw's own compile reports it
		}
		const auto module = CompileSPV(result.spirv, graphics.device);
		if (module == nullptr) {
			return;
		}
		out.program        = std::move(result.program).TakeCompiledInfo();
		out.specialization = std::move(specialization);
		out.push_cursor    = *cursor;
		out.module         = module;
		out.compiled       = true;
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// What GetComputePipeline creates for the permutation: it depends on the program
			// and its module only.
			const ShaderRecompiler::IR::ResourceSnapshot resources;
			ShaderComputeInputInfo                       input;
			input.stage = {.program = &out.program, .resources = &resources};
			if (CanCreatePipeline(graphics, input)) {
				Pipeline pipeline;
				CreatePipelineInternal(graphics, pipeline, input, module, driver_cache);
				out.compute_pipeline = pipeline;
			}
		}
	}

	// Worker thread: the bundle's stages in order, each starting where the previous one's
	// program leaves the push-data cursor.
	PrecompileBundle BuildPrecompileBundle(std::vector<PrecompileRequest>&            requests,
	                                       std::optional<PrecompilePipelineRequest>& pipeline,
	                                       uint32_t worker_count, vk::PipelineCache driver_cache) {
		PrecompileBundle        bundle;
		std::optional<uint32_t> cursor;
		for (auto& request: requests) {
			if (request.push_cursor) {
				cursor = request.push_cursor;
			}
			auto& out = bundle.stages.emplace_back();
			out.key   = request.key;
			std::visit(
			    [&](auto& input) {
				    BuildPrecompiledStage(request, input, cursor, worker_count, driver_cache, out);
			    },
			    request.input);
			if (out.compiled) {
				out.program.bindings.AdvancePushData(*cursor);
			} else {
				cursor.reset();
			}
		}
		if (pipeline) {
			BuildPrecompiledPipeline(*pipeline, bundle, driver_cache);
		}
		return bundle;
	}

	// Worker thread: what GetGraphicsPipeline would create for the predicted draw, with the
	// attachments RenderExecutor::PrepareDrawRenderState would resolve from its registers.
	void BuildPrecompiledPipeline(PrecompilePipelineRequest& request, PrecompileBundle& bundle,
	                              vk::PipelineCache driver_cache) const {
		struct Stage {
			uint64_t                                        id      = 0;
			vk::ShaderModule                                module  = nullptr;
			const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr;
		};
		const auto resolve = [&](const ProgramKey& key,
		                         const std::optional<PrecompileKnownProgram>& known) {
			if (known) {
				return Stage {known->id, known->module, &known->program};
			}
			for (const auto& stage: bundle.stages) {
				if (stage.compiled && stage.key == key) {
					return Stage {ProvisionalProgramId(key), stage.module, &stage.program};
				}
			}
			return Stage {};
		};
		const auto vertex = resolve(request.vertex_key, request.known_vertex);
		const bool pixel_active = request.pixel_info.has_value();
		const auto pixel = pixel_active ? resolve(request.pixel_key, request.known_pixel) : Stage {};
		if (vertex.program == nullptr || (pixel_active && pixel.program == nullptr)) {
			return;
		}
		const auto& ctx = *request.context;
		// The targets the pixel program writes (PrepareDrawRenderState's MRT mask), resolved as
		// ResolveRenderColorTarget does.
		uint32_t mrt_mask = 0;
		if (pixel_active) {
			for (const auto& output: pixel.program->info.outputs) {
				if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
				    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
					mrt_mask |= 1u << output.location;
				}
			}
		}
		mrt_mask &= request.color_output_mask;
		std::array<PipelineColorTarget, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
		uint32_t                                                      color_count = 0;
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) == 0) {
				continue;
			}
			const auto& rt = ctx.GetRenderTarget(slot);
			if (rt.base.addr == 0 || render_target_mask_slot(ctx.GetRenderTargetMask(), slot) == 0) {
				continue;
			}
			const auto format = TextureTryGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                                    rt.info.channel_order);
			const auto samples = render_sample_count(rt.attrib.num_fragments);
			if (!format || samples == 0 || rt.attrib.num_samples != rt.attrib.num_fragments) {
				return;
			}
			colors[color_count++] = {.slot           = slot,
			                         .format         = format->format,
			                         .samples        = samples,
			                         .export_mapping = format->export_mapping,
			                         .guest_format   = format->guest_format};
		}
		const ShaderRecompiler::IR::ResourceSnapshot resources;
		auto vertex_info  = request.vertex_info;
		vertex_info.stage = {.program = vertex.program, .resources = &resources};
		std::optional<ShaderPixelInputInfo> pixel_info;
		if (pixel_active) {
			pixel_info        = *request.pixel_info;
			pixel_info->stage = {.program = pixel.program, .resources = &resources};
		}
		GraphicsPipelineKey key {};
		key.vertex_shader_ids[0] = vertex.id;
		key.ps_shader_id         = pixel.id;
		const auto* ps_input     = pixel_info ? &*pixel_info : nullptr;
		if (!PipelineCache::BuildGraphicsPipelineKey(
		        graphics, std::span(colors).first(color_count), request.depth, vertex_info, ctx,
		        request.primitive_type, ps_input, request.topology, false, key, true) ||
		    !CanCreatePipeline(graphics, key.rendering, key.vertex_input,
		                       std::span(&vertex_info, 1), ps_input, key.static_params)) {
			return;
		}
		GraphicsPrograms programs;
		programs.vertex[0] = {.id = vertex.id, .module = vertex.module};
		programs.pixel     = {.id = pixel.id, .module = pixel.module};
		Pipeline pipeline;
		CreatePipelineInternal(graphics, pipeline, key.rendering, key.vertex_input,
		                       std::span(&vertex_info, 1), ps_input, programs, key.static_params,
		                       driver_cache);
		bundle.pipeline.emplace(key, pipeline);
	}

	// GPU thread: hands the stages that are new to a worker. False when the workers are full.
	bool SubmitPrecompile(std::vector<PrecompileRequest>           requests,
	                      std::optional<PrecompilePipelineRequest> pipeline) {
		std::erase_if(requests, [&](const PrecompileRequest& request) {
			const auto hash = PrecompileKeyHash(request.key);
			return programs.contains(request.key) || precompile_bundles.contains(hash) ||
			       precompile_ready.contains(hash);
		});
		if (requests.empty() || precompiles == nullptr) {
			return true;
		}
		// The pipeline needs each stage known or in this bundle.
		const auto in_bundle = [&](const ProgramKey& key) {
			return std::ranges::any_of(requests,
			                           [&](const PrecompileRequest& request) { return request.key == key; });
		};
		if (pipeline && ((!pipeline->known_vertex && !in_bundle(pipeline->vertex_key)) ||
		                 (pipeline->pixel_info && !pipeline->known_pixel &&
		                  !in_bundle(pipeline->pixel_key)))) {
			pipeline.reset();
		}
		// Bundles the queue dropped unfinished or evicted unclaimed leave their stages here.
		if (precompile_bundles.size() > 4u * MaxPrecompileBundles) {
			std::erase_if(precompile_bundles,
			              [&](const auto& item) { return !precompiles->Contains(item.second); });
		}
		// The first new stage starts at a known cursor, or none is compiled ahead.
		const auto bundle = ++next_precompile_bundle;
		std::vector<uint64_t> hashes;
		for (const auto& request: requests) {
			hashes.push_back(PrecompileKeyHash(request.key));
		}
		const auto worker_count = prefetcher != nullptr ? prefetcher->WorkerCount() : 0u;
		struct Job {
			std::vector<PrecompileRequest>           requests;
			std::optional<PrecompilePipelineRequest> pipeline;
		};
		auto       shared = std::make_shared<Job>(Job {std::move(requests), std::move(pipeline)});
		const bool queued = precompiles->Submit(
		    bundle, [this, shared, worker_count, driver_cache = precompile_driver_cache]() {
			    return BuildPrecompileBundle(shared->requests, shared->pipeline, worker_count,
			                                 driver_cache);
		    });
		if (!queued) {
			return false;
		}
		for (const auto hash: hashes) {
			precompile_bundles[hash] = bundle;
		}
		precompile_stats.submitted += hashes.size();
		Common::FrameProfiler::Add(Common::FrameProfiler::Counter::PrecompileSubmitted,
		                           hashes.size());
		return true;
	}

	// GPU thread: the worker's stage for a ProgramKey Get has no entry for, waiting for a
	// running build; nullopt when none was started or it was for another key.
	std::optional<PrecompiledStage> TakePrecompiled(const ProgramKey& key) {
		if (precompiles == nullptr) {
			return std::nullopt;
		}
		const auto hash = PrecompileKeyHash(key);
		if (const auto bundle = precompile_bundles.find(hash); bundle != precompile_bundles.end()) {
			const auto     id  = bundle->second;
			const uint64_t t0  = Common::FrameProfiler::Active() ? Common::FrameProfiler::Ticks() : 0;
			auto           got = precompiles->TakeStarted(id);
			if (t0 != 0) {
				Common::FrameProfiler::AddSince(Common::FrameProfiler::Counter::PrecompileWaitTime,
				                                t0);
			}
			std::erase_if(precompile_bundles, [&](const auto& item) { return item.second == id; });
			if (got && got->pipeline) {
				auto& [pipeline_key, pipeline] = *got->pipeline;
				if (precompiled_graphics.contains(pipeline_key)) {
					DestroyPipeline(pipeline);
				} else {
					precompiled_graphics_order.push_back(pipeline_key);
					precompiled_graphics.emplace(pipeline_key, pipeline);
					while (precompiled_graphics_order.size() > MaxPrecompiledReady) {
						const auto old = precompiled_graphics.find(precompiled_graphics_order.front());
						precompiled_graphics_order.pop_front();
						if (old != precompiled_graphics.end()) {
							DestroyPipeline(old->second);
							precompiled_graphics.erase(old);
							precompile_wasted.fetch_add(1, std::memory_order_relaxed);
							Common::FrameProfiler::Add(
							    Common::FrameProfiler::Counter::PrecompileWasted);
						}
					}
				}
			}
			if (got) {
				for (auto& stage: got->stages) {
					const auto stage_hash = PrecompileKeyHash(stage.key);
					if (precompile_ready.contains(stage_hash)) {
						DisposePrecompiledStage(stage);
						continue;
					}
					precompile_ready_order.push_back(stage_hash);
					precompile_ready.emplace(stage_hash, std::move(stage));
				}
			}
		}
		const auto ready = precompile_ready.find(hash);
		if (ready == precompile_ready.end()) {
			return std::nullopt;
		}
		auto stage = std::move(ready->second);
		precompile_ready.erase(ready);
		std::erase(precompile_ready_order, hash);
		TrimPrecompiledReady();
		if (stage.key != key) {
			DisposePrecompiledStage(stage);
			return std::nullopt;
		}
		return stage;
	}

	// Stages of taken bundles that no draw asked for yet: the oldest go beyond the bound.
	void TrimPrecompiledReady() {
		while (precompile_ready_order.size() > MaxPrecompiledReady) {
			const auto hash = precompile_ready_order.front();
			precompile_ready_order.pop_front();
			if (const auto ready = precompile_ready.find(hash); ready != precompile_ready.end()) {
				DisposePrecompiledStage(ready->second);
				precompile_ready.erase(ready);
			}
		}
	}

	void AdoptReplicas(SourceEntry& entry, PrecompiledStage& stage) {
		if (prefetcher == nullptr || stage.replicas.size() != prefetcher->WorkerCount()) {
			return;
		}
		entry.replicas = std::move(stage.replicas);
		for (const auto& replica: entry.replicas) {
			entry.replica_views.push_back(replica.get());
		}
	}

	Permutation AdoptPermutation(PrecompiledStage& stage) {
		Permutation permutation {
		    .specialization = std::move(stage.specialization),
		    .program        = std::move(stage.program),
		    .handle         = {.id = ++next_shader_id, .module = stage.module},
		};
		stage.module = nullptr;
		provisional_ids.emplace(permutation.handle.id, ProvisionalProgramId(stage.key));
		if (stage.compute_pipeline) {
			precompiled_pipelines.emplace(permutation.handle.id, *stage.compute_pipeline);
			stage.compute_pipeline.reset();
		}
		return permutation;
	}

	// Shader addresses whose programs a lookahead prediction looked up (or a draw created)
	// since the shader map last changed: a predicted draw whose stages are all here costs no
	// further lookup.
	bool PrecompileKnown(ShaderType stage, uint64_t address) {
		RefreshPrecompileKnown();
		return precompile_known[stage == ShaderType::Pixel ? 0 : stage == ShaderType::Compute ? 2 : 1]
		    .contains(address);
	}
	void NotePrecompileKnown(ShaderType stage, uint64_t address) {
		RefreshPrecompileKnown();
		precompile_known[stage == ShaderType::Pixel ? 0 : stage == ShaderType::Compute ? 2 : 1]
		    .insert(address);
	}
	void RefreshPrecompileKnown() {
		const auto generation = ShaderMapGeneration();
		if (generation != precompile_known_generation) {
			precompile_known_generation = generation;
			for (auto& known: precompile_known) {
				known.clear();
			}
		}
	}

	// Before the device and the driver cache go (PipelineCache's destructor).
	void StopPrecompiles() {
		if (precompiles != nullptr) {
			precompiles->Shutdown();
		}
		for (auto& [hash, stage]: precompile_ready) {
			(void)hash;
			DisposePrecompiledStage(stage);
		}
		precompile_ready.clear();
		precompile_ready_order.clear();
		precompile_bundles.clear();
		for (auto& [id, pipeline]: precompiled_pipelines) {
			(void)id;
			DestroyPipeline(pipeline);
		}
		precompiled_pipelines.clear();
		for (auto& [key, pipeline]: precompiled_graphics) {
			(void)key;
			DestroyPipeline(pipeline);
		}
		precompiled_graphics.clear();
		precompiled_graphics_order.clear();
	}

	explicit ProgramCache(GraphicContext& graphics): graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		prefetcher = std::make_unique<ResourcePrefetcher>(
		    DrawPrepThreads(),
		    ResourcePrefetcher::MemoryOps {.view_page    = ViewGuestPage,
		                                   .watch        = &SrtPageWatch::Guest(),
		                                   .page_settled = IsSettledGuestPage,
		                                   .replay_check = KYTY_BUILD == KYTY_BUILD_DEBUG});
	}
	~ProgramCache() {
		StopPrecompiles();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	uint64_t                                                    next_shader_id = 0;
	// The entry the last Get materialized, for Learn.
	SourceEntry* last_materialized = nullptr;
	// BuildSiblingsAhead's hand-off to PipelineCache's background builds; false when refused.
	std::function<bool(uint64_t key, std::shared_ptr<const std::vector<uint32_t>> spirv,
	                   std::shared_ptr<const ShaderRecompiler::IR::CompiledShaderInfo> program)>
	    submit_build;
	// ComputePipelineBuildKey of table-linked compute programs, by ShaderProgram::id, until
	// GetComputePipeline takes it.
	std::unordered_map<uint64_t, uint64_t> build_keys;
	std::array<std::unordered_map<uint64_t, Prediction>,
	           static_cast<size_t>(PrefetchSlot::Count)>
	    predictions;
	uint64_t next_prefetch_draw = 0;
	// Declared after the plans above: its workers evaluate them, so it stops first.
	std::unique_ptr<ResourcePrefetcher> prefetcher;
	uint64_t                            translations = 0;
	// Precompile state, GPU thread: the bundle each submitted stage is in (by
	// PrecompileKeyHash), the stages of taken bundles no draw took yet, and the pipelines of
	// adopted compute permutations until GetComputePipeline takes them (by ShaderProgram::id).
	std::unordered_map<uint64_t, uint64_t>           precompile_bundles;
	std::unordered_map<uint64_t, PrecompiledStage>   precompile_ready;
	std::deque<uint64_t>                             precompile_ready_order;
	std::unordered_map<uint64_t, Pipeline>           precompiled_pipelines;
	// Graphics pipelines built ahead, by their key with ProvisionalProgramId for programs the
	// bundle compiled; adopted permutations map their ShaderProgram id to that id.
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> precompiled_graphics;
	std::deque<GraphicsPipelineKey>                  precompiled_graphics_order;
	std::unordered_map<uint64_t, uint64_t>           provisional_ids;
	std::array<std::unordered_set<uint64_t>, 3>      precompile_known;
	uint64_t                                         precompile_known_generation = 0;
	uint64_t                                         next_precompile_bundle      = 0;
	PrecompileStats                                  precompile_stats;
	std::atomic<uint64_t>                            precompile_wasted {0};
	vk::PipelineCache                                precompile_driver_cache = nullptr;
	// Declared after everything its workers use, so it stops first.
	std::unique_ptr<KeyedBuildQueue<PrecompileBundle>> precompiles;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	// Two workers: a relink rarely has more than one sibling, and each build holds a large
	// compiler working set. Finished pipelines nobody takes are destroyed.
	m_background_builds = std::make_unique<KeyedBuildQueue<Pipeline>>(
	    2u, 2u * ProgramCache::MaxSiblingBuilds, [this](Pipeline& pipeline) {
		    m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
		    m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
		    m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	    });
	// Precompiles: as many workers as PrecompileThreads, below the sibling builds' priority (their
	// requester is already waiting). Bundles nobody takes give their modules back.
	m_program_cache->precompile_driver_cache = m_driver_cache;
	m_program_cache->precompiles = std::make_unique<KeyedBuildQueue<ProgramCache::PrecompileBundle>>(
	    ProgramCache::PrecompileThreads(), ProgramCache::MaxPrecompileBundles,
	    [this](ProgramCache::PrecompileBundle& bundle) {
		    for (auto& stage: bundle.stages) {
			    m_program_cache->DisposePrecompiledStage(stage);
		    }
		    if (bundle.pipeline) {
			    m_program_cache->DestroyPipeline(bundle.pipeline->second);
		    }
	    },
	    10);
	m_program_cache->submit_build =
	    [this](uint64_t key, std::shared_ptr<const std::vector<uint32_t>> spirv,
	           std::shared_ptr<const ShaderRecompiler::IR::CompiledShaderInfo> program) {
		    return m_background_builds->Submit(key, [this, spirv, program]() {
			    Pipeline   pipeline;
			    const auto module = CompileSPV(*spirv, m_graphics.device);
			    EXIT_IF(module == nullptr);
			    const ShaderRecompiler::IR::ResourceSnapshot resources;
			    ShaderComputeInputInfo                       input;
			    input.stage = {.program = program.get(), .resources = &resources};
			    {
				    Common::DebugCounters::ScopedTimer timer(
				        Common::DebugCounters::Counter::PipelinesCreated,
				        Common::DebugCounters::Counter::PipelineCreateNs,
				        Common::DebugCounters::Peak::PipelineCreateNs);
				    CreatePipelineInternal(m_graphics, pipeline, input, module, m_driver_cache);
			    }
			    m_driver_cache_dirty.store(true, std::memory_order_relaxed);
			    m_graphics.device.destroyShaderModule(module, nullptr);
			    return pipeline;
		    });
	    };
}

PipelineCache::~PipelineCache() {
	// Workers use the device and the driver cache: they stop before either goes.
	m_background_builds->Shutdown();
	m_program_cache->StopPrecompiles();
	if (const auto stats = GetPrecompileStats(); stats.submitted != 0) {
		PipelineCacheLog("Precompiles: {} programs handed to workers; {} programs and {} graphics "
		                 "pipelines taken from them, {} programs compiled on the GPU thread, {} "
		                 "results dropped",
		                 stats.submitted, stats.hits, stats.pipeline_hits, stats.misses,
		                 stats.wasted);
	}
	StopDriverCacheSaves();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	const auto now = std::chrono::steady_clock::now();
	if (now < m_next_driver_cache_save) {
		return;
	}
	m_next_driver_cache_save = now + DriverCacheSaveInterval;
	{
		std::lock_guard lock(m_save_state_mutex);
		if (m_save_stop) {
			return;
		}
		m_save_requested = true;
		if (!m_save_thread.joinable()) {
			m_save_thread = std::thread([this] {
				LowerBackgroundBuildPriority();
				std::unique_lock state(m_save_state_mutex);
				for (;;) {
					m_save_wake.wait(state, [this] { return m_save_requested || m_save_stop; });
					if (m_save_stop) {
						return;
					}
					m_save_requested = false;
					state.unlock();
					Save();
					state.lock();
				}
			});
		}
	}
	m_save_wake.notify_one();
}

void PipelineCache::StopDriverCacheSaves() {
	{
		std::lock_guard lock(m_save_state_mutex);
		m_save_stop = true;
	}
	m_save_wake.notify_one();
	if (m_save_thread.joinable()) {
		m_save_thread.join();
	}
}

void PipelineCache::Save() {
	std::lock_guard save(m_save_mutex);
	// A pipeline added while this save reads the cache sets the flag again for the next one.
	if (m_driver_cache == nullptr || !m_driver_cache_dirty.exchange(false)) {
		return;
	}
	if (!WriteDriverCache()) {
		m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	}
}

bool PipelineCache::WriteDriverCache() {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return false;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return false;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return false;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	return true;
}

// The stage programs and inputs a draw compiles with (GetGraphicsPrograms). Speculative, for a
// predicted draw: TryPrepareProgram, and false where the real draw stops the emulator.
static bool PrepareGraphicsStages(GraphicContext& graphics, const HW::VertexShaderInfo& vertex_regs,
                           const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
                           const HW::Context& context, const HW::UserConfig& user_config,
                           std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
                           bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
                           ShaderPixelInputInfo& pixel_info, GraphicsStages& stages,
                           bool speculative) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	stages.tess_active     = tess_active;
	auto& vertex_params    = stages.vertex;
	if (tess_active) {
		if (speculative) {
			return false;
		}
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else if (speculative) {
		const auto params = TryPrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
		if (!params) {
			return false;
		}
		vertex_params[0] = *params;
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		KYTY_PRECOMPILE_REQUIRE(speculative, sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		KYTY_PRECOMPILE_REQUIRE(speculative, Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		KYTY_PRECOMPILE_REQUIRE(speculative, !graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = graphics.subgroup_size;
		const auto& limits      = graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			if (speculative) {
				return false;
			}
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	auto& pixel_params = stages.pixel;
	if (pixel_active) {
		if (speculative) {
			const auto params = TryPrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
			if (!params) {
				return false;
			}
			pixel_params = *params;
		} else {
			pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		}
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				if (speculative) {
					return false;
				}
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	stages.push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id);
	return true;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	GraphicsStages stages;
	(void)PrepareGraphicsStages(m_graphics, vertex_regs, pixel_regs, sh, context, user_config,
	                            target_export_mapping, pixel_active, vertex_info, pixel_info,
	                            stages, false);
	const bool tess_active        = stages.tess_active;
	const auto& vertex_params     = stages.vertex;
	const auto& pixel_params      = stages.pixel;
	uint32_t    push_data_cursor  = stages.push_data_cursor;

	GraphicsPrograms  result;
	auto&             programs = *m_program_cache;
	using Slot                 = ProgramCache::PrefetchSlot;
	if (pixel_active) {
		result.pixel = programs.Get(pixel_params, pixel_info, push_data_cursor);
		if (programs.last_materialized != nullptr) {
			programs.last_materialized->push_end = push_data_cursor;
		}
		programs.Learn(Slot::Pixel, pixel_regs.ps_regs.data_addr, pixel_params,
		               pixel_regs.ps_regs.rsrc2.user_sgpr, 0);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = programs.Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	if (!tess_active) {
		programs.Learn(Slot::Vertex, vertex_regs.es_regs.data_addr, vertex_params[0],
		               vertex_regs.gs_regs.rsrc2.user_sgpr, vertex_regs.gs_regs.user_data_addr);
	}
	return result;
}

// The program and inputs a dispatch compiles with (GetComputeProgram); speculative as
// PrepareGraphicsStages.
static std::optional<ShaderParams> PrepareComputeStage(GraphicContext&              graphics,
                                                       const HW::ComputeShaderInfo& regs,
                                                       const HW::ShaderRegisters&   sh,
                                                       ShaderComputeInputInfo&      input_info,
                                                       bool                         speculative) {
	input_info.host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
	auto params = speculative ? TryPrepareProgram(regs, sh, input_info)
	                          : std::optional {PrepareProgram(regs, sh, input_info)};
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	return params;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	const auto params = *PrepareComputeStage(m_graphics, regs, sh, input_info, false);
	uint32_t          push_data_cursor = 0;
	auto              program = m_program_cache->Get(params, input_info, push_data_cursor);
	if (QueuePark::Pending()) {
		return program;
	}
	m_program_cache->Learn(ProgramCache::PrefetchSlot::Compute, regs.cs_regs.data_addr, params,
	                       regs.cs_regs.user_sgpr, 0);
	return program;
}

void PipelineCache::PrecompileDraw(const HW::Context& context, const HW::Shader& shaders,
                                   const HW::UserConfig& user_config) {
	auto& programs = *m_program_cache;
	if (programs.precompiles == nullptr) {
		return;
	}
	const auto& ps          = shaders.GetPs();
	const auto& vs          = shaders.GetVs();
	const auto  ps_address  = ps.ps_regs.data_addr;
	const auto  vs_address  = vs.es_regs.data_addr;
	const auto  note_known  = [&] {
		  if (ps_address != 0) programs.NotePrecompileKnown(ShaderType::Pixel, ps_address);
		  if (vs_address != 0) programs.NotePrecompileKnown(ShaderType::Vertex, vs_address);
	};
	if ((ps_address == 0 || programs.PrecompileKnown(ShaderType::Pixel, ps_address)) &&
	    (vs_address == 0 || programs.PrecompileKnown(ShaderType::Vertex, vs_address))) {
		return;
	}
	DrawPixelInputs                      pixel_inputs;
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
	GraphicsStages                       stages;
	if (vs_address == 0 || !DrawPixelStageInputs(context, shaders, true, pixel_inputs) ||
	    !PrepareGraphicsStages(m_graphics, vs, ps, context.GetShaderRegisters(), context,
	                           user_config, pixel_inputs.target_export_mapping,
	                           pixel_inputs.ps_active, vertex_info, pixel_info, stages, true)) {
		note_known();
		return;
	}
	using Request = ProgramCache::PrecompileRequest;
	using Known   = ProgramCache::PrecompileKnownProgram;
	const auto known_program = [&](const ProgramCache::ProgramKey& key) -> std::optional<Known> {
		const auto entry = programs.programs.find(key);
		if (entry == programs.programs.end() ||
		    entry->second.last_permutation >= entry->second.permutations.size()) {
			return std::nullopt;
		}
		const auto& permutation = entry->second.permutations[entry->second.last_permutation];
		return Known {.id      = permutation.handle.id,
		              .module  = permutation.handle.module,
		              .program = permutation.program};
	};
	std::vector<Request>    requests;
	std::optional<uint32_t> vertex_cursor = stages.push_data_cursor;
	ProgramCache::ProgramKey pixel_key;
	if (pixel_inputs.ps_active) {
		pixel_key = ProgramCache::MakeProgramKey(ShaderType::Pixel, stages.pixel, pixel_info);
		if (const auto entry = programs.programs.find(pixel_key); entry != programs.programs.end()) {
			vertex_cursor = entry->second.push_end != UINT32_MAX
			                    ? std::optional {entry->second.push_end}
			                    : std::nullopt;
		} else {
			requests.push_back({.stage       = ShaderType::Pixel,
			                    .key         = pixel_key,
			                    .params      = stages.pixel,
			                    .input       = pixel_info,
			                    .push_cursor = stages.push_data_cursor});
			vertex_cursor.reset(); // after the pixel program the worker compiles
		}
	}
	const auto vertex_stage = vertex_info[0].logical_stage;
	const auto vertex_key = ProgramCache::MakeProgramKey(vertex_stage, stages.vertex[0], vertex_info[0]);
	if (!programs.programs.contains(vertex_key)) {
		requests.push_back({.stage       = vertex_stage,
		                    .key         = vertex_key,
		                    .params      = stages.vertex[0],
		                    .input       = vertex_info[0],
		                    .push_cursor = vertex_cursor});
	}
	// The draw's pipeline too, when its state predicts one: a primitive restart that depends on
	// the index size is not predicted.
	std::optional<ProgramCache::PrecompilePipelineRequest> pipeline;
	const auto primitive_type = user_config.GetPrimType();
	const bool restart_unknown =
	    (user_config.GetPrimitiveResetControl() & 1u) != 0 &&
	    (primitive_type == Prospero::PrimitiveType::kLineStrip ||
	     primitive_type == Prospero::PrimitiveType::kTriFan ||
	     primitive_type == Prospero::PrimitiveType::kTriStrip);
	vk::PrimitiveTopology topology {};
	if (!requests.empty() && !restart_unknown && GetDrawTopology(user_config, topology)) {
		if (const auto depth = PredictPipelineDepthTarget(m_graphics, context)) {
			pipeline.emplace();
			pipeline->context           = std::make_unique<HW::Context>(context);
			pipeline->primitive_type    = primitive_type;
			pipeline->topology          = topology;
			pipeline->color_output_mask = pixel_inputs.color_output_mask;
			pipeline->depth             = *depth;
			pipeline->vertex_info       = vertex_info[0];
			pipeline->vertex_key        = vertex_key;
			pipeline->known_vertex      = known_program(vertex_key);
			if (pixel_inputs.ps_active) {
				pipeline->pixel_info  = pixel_info;
				pipeline->pixel_key   = pixel_key;
				pipeline->known_pixel = known_program(pixel_key);
			}
		}
	}
	if (programs.SubmitPrecompile(std::move(requests), std::move(pipeline))) {
		note_known();
	}
}

void PipelineCache::PrecompileDispatch(const HW::Shader& shaders, const HW::ShaderRegisters& sh,
                                       uint32_t mode, bool async_compute,
                                       const std::array<uint32_t, 3>& workgroup_counts,
                                       std::optional<uint64_t>        indirect_args_address) {
	auto&       programs = *m_program_cache;
	const auto& cs       = shaders.GetCs();
	const auto  address  = cs.cs_regs.data_addr;
	if (programs.precompiles == nullptr || address == 0 ||
	    programs.PrecompileKnown(ShaderType::Compute, address)) {
		return;
	}
	// As RenderExecutor::DispatchDirect and DispatchIndirect fill them in.
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool threads = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = threads;
	if (indirect_args_address) {
		input_info.indirect_args_address    = *indirect_args_address;
		input_info.dispatch_thread_indirect = threads;
	} else {
		std::ranges::copy(workgroup_counts, input_info.workgroup_counts);
	}
	const auto params = PrepareComputeStage(m_graphics, cs, sh, input_info, true);
	if (!params) {
		programs.NotePrecompileKnown(ShaderType::Compute, address);
		return;
	}
	std::vector<ProgramCache::PrecompileRequest> requests;
	requests.push_back({.stage       = ShaderType::Compute,
	                    .key         = ProgramCache::MakeProgramKey(ShaderType::Compute, *params,
	                                                                input_info),
	                    .params      = *params,
	                    .input       = input_info,
	                    .push_cursor = 0u});
	if (programs.SubmitPrecompile(std::move(requests), std::nullopt)) {
		programs.NotePrecompileKnown(ShaderType::Compute, address);
	}
}

void PipelineCache::WaitForPrecompiles() {
	if (m_program_cache->precompiles != nullptr) {
		m_program_cache->precompiles->WaitIdle();
	}
}

PipelineCache::PrecompileStats PipelineCache::GetPrecompileStats() const {
	auto stats   = m_program_cache->precompile_stats;
	stats.wasted = m_program_cache->precompile_wasted.load(std::memory_order_relaxed);
	return stats;
}

bool PipelineCache::PrefetchEnabled() const {
	return m_program_cache->prefetcher != nullptr;
}

uint64_t PipelineCache::PrefetchDraw(const HW::Shader& shaders, const HW::UserConfig& user_config) {
	auto&      programs = *m_program_cache;
	const auto draw     = ++programs.next_prefetch_draw;
	if (programs.prefetcher == nullptr ||
	    user_config.GetPrimType() == Prospero::PrimitiveType::kPatch) {
		return draw;
	}
	using Slot     = ProgramCache::PrefetchSlot;
	const auto& ps = shaders.GetPs();
	const auto& vs = shaders.GetVs();
	// In the order the draw materializes them.
	if (ps.ps_regs.data_addr != 0) {
		programs.Speculate(draw, Slot::Pixel, ps.ps_regs.data_addr, ps.ps_user_sgpr,
		                   ps.ps_regs.rsrc2.user_sgpr, 0, {});
	}
	if (vs.es_regs.data_addr != 0) {
		programs.Speculate(draw, Slot::Vertex, vs.es_regs.data_addr, vs.gs_user_sgpr,
		                   vs.gs_regs.rsrc2.user_sgpr, vs.gs_regs.user_data_addr, {});
	}
	return draw;
}

uint64_t PipelineCache::PrefetchDispatch(const HW::Shader&              shaders,
                                         const std::array<uint32_t, 3>& workgroup_counts) {
	auto&       programs = *m_program_cache;
	const auto  draw     = ++programs.next_prefetch_draw;
	const auto& cs       = shaders.GetCs();
	if (programs.prefetcher != nullptr && cs.cs_regs.data_addr != 0) {
		programs.Speculate(draw, ProgramCache::PrefetchSlot::Compute, cs.cs_regs.data_addr,
		                   cs.cs_user_sgpr, cs.cs_regs.user_sgpr, 0, workgroup_counts);
	}
	return draw;
}

void PipelineCache::BeginPrefetchedDraw(uint64_t draw) {
	if (m_program_cache->prefetcher != nullptr) {
		m_program_cache->prefetcher->BeginDraw(draw);
	}
}

void PipelineCache::ClearPrefetch() {
	if (m_program_cache->prefetcher != nullptr) {
		m_program_cache->prefetcher->Clear();
	}
}

void PipelineCache::BeginPrefetchBatch() {
	if (m_program_cache->prefetcher != nullptr) {
		m_program_cache->prefetcher->BeginBatch();
	}
}

void PipelineCache::FlushPrefetch() {
	if (m_program_cache->prefetcher != nullptr) {
		m_program_cache->prefetcher->Flush();
	}
}

void PipelineCache::SetPrefetchReplayCheck(bool enabled) {
	if (m_program_cache->prefetcher != nullptr) {
		m_program_cache->prefetcher->SetReplayCheck(enabled);
	}
}

ResourcePrefetcher::Stats PipelineCache::GetPrefetchStats() const {
	return m_program_cache->prefetcher != nullptr ? m_program_cache->prefetcher->GetStats()
	                                              : ResourcePrefetcher::Stats {};
}

uint64_t PipelineCache::TranslationCount() const {
	return m_program_cache->translations;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

// GetGraphicsPipeline's key, but the program ids, from the draw's targets and state.
// Speculative (a predicted draw): false where the real draw stops the emulator.
bool PipelineCache::BuildGraphicsPipelineKey(
    GraphicContext& graphics, std::span<const PipelineColorTarget> colors,
    const PipelineDepthTarget& depth, const ShaderVertexInputInfo& vs_input_info,
    const HW::Context& ctx, Prospero::PrimitiveType primitive_type,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, GraphicsPipelineKey& key, bool speculative) {
	if (colors.size() > RENDER_COLOR_ATTACHMENTS_MAX) {
		if (speculative) {
			return false;
		}
		EXIT("too many color attachments\n");
	}
	const bool             ps_active   = ps_input_info != nullptr;
	const auto             color_count = static_cast<uint32_t>(colors.size());
	const HW::ModeControl& mc          = ctx.GetModeControl();
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(colors[i].format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].slot));
		if (colors[i].guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].samples;
		} else if (attachment_samples != colors[i].samples) {
			if (speculative) {
				return false;
			}
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!speculative && !warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth = depth.format != vk::Format::eUndefined;
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.samples;
		} else if (attachment_samples != depth.samples) {
			if (speculative) {
				return false;
			}
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		if (speculative &&
		    !static_cast<bool>(
		        graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		        vulkan_sample_count(attachment_samples))) {
			return false;
		}
		EXIT_IF(!static_cast<bool>(
		    graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	if (speculative && (attachment_samples == 0 ||
	                    vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {})) {
		return false;
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (!speculative && ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !graphics.sample_rate_shading_enabled) {
		if (speculative) {
			return false;
		}
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(primitive_type);
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	const auto polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back, speculative);
	if (!polygon_mode) {
		return false;
	}
	static_params.polygon_mode = *polygon_mode;

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		if (speculative &&
		    (vs_input_info.buffers_num < 0 ||
		     vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		     vs_input_info.resources_num < 0 ||
		     vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX)) {
			return false;
		}
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			if (speculative && (binding < 0 || binding >= vs_input_info.buffers_num)) {
				return false;
			}
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	return true;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	std::array<PipelineColorTarget, RENDER_COLOR_ATTACHMENTS_MAX> targets {};
	for (size_t i = 0; i < colors.size(); i++) {
		EXIT_IF(!colors[i].image_id);
		targets[i] = {.slot           = colors[i].target_slot,
		              .format         = colors[i].desc.view_info.format,
		              .samples        = colors[i].desc.info.samples,
		              .export_mapping = colors[i].export_mapping,
		              .guest_format   = colors[i].desc.info.guest_format};
	}
	const PipelineDepthTarget depth_target {
	    .format            = depth.image_id ? depth.desc.view_info.format : vk::Format::eUndefined,
	    .samples           = depth.desc.info.samples,
	    .depth_test_enable = depth.depth_test_enable};
	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id = ps_id;
	(void)BuildGraphicsPipelineKey(m_graphics, std::span(targets).first(colors.size()),
	                               depth_target, vs_input_info, command.GetRegisters(),
	                               command.GetUserConfig().GetPrimType(), ps_input_info, topology,
	                               primitive_restart_enable, key, false);

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}
	// A pipeline built ahead for this draw (ProgramCache precompiles): its key names the
	// programs a precompile compiled by ProvisionalProgramId.
	if (auto& built_ahead = m_program_cache->precompiled_graphics; !built_ahead.empty()) {
		auto        content     = key;
		const auto& provisional = m_program_cache->provisional_ids;
		const auto  map_id      = [&](uint64_t& id) {
			if (const auto found = provisional.find(id); found != provisional.end()) {
				id = found->second;
			}
		};
		for (auto& id: content.vertex_shader_ids) {
			map_id(id);
		}
		map_id(content.ps_shader_id);
		if (const auto found = built_ahead.find(content); found != built_ahead.end()) {
			auto cached = std::make_unique<Pipeline>(found->second);
			built_ahead.erase(found);
			std::erase(m_program_cache->precompiled_graphics_order, content);
			m_program_cache->precompile_stats.pipeline_hits++;
			Common::FrameProfiler::Add(Common::FrameProfiler::Counter::PrecompilePipelineHits);
			auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
			EXIT_IF(!inserted);
			CheckpointDriverCache();
			return *iter->second;
		}
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, key.rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, key.static_params, m_driver_cache);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	auto& build_keys = m_program_cache->build_keys;
	std::optional<Pipeline> prebuilt;
	// A precompiled permutation's pipeline, built from the same module and program.
	auto& precompiled = m_program_cache->precompiled_pipelines;
	if (const auto found = precompiled.find(compute_program.id); found != precompiled.end()) {
		prebuilt = found->second;
		precompiled.erase(found);
	} else if (const auto key = build_keys.find(compute_program.id); key != build_keys.end()) {
		prebuilt = m_background_builds->Take(key->second);
		if (prebuilt) {
			std::printf("Compute program %" PRIu64 ": using the pipeline built ahead (key 0x%016" PRIx64
			            ")\n",
			            compute_program.id, key->second);
		}
		build_keys.erase(key);
	}
	if (prebuilt) {
		*cached = *prebuilt;
	} else {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
