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
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
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
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
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
	if (values.empty()) {
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
				if (hit) {
					return true;
				}
			}
			const uint64_t plain_t0 = profiling ? Common::FrameProfiler::Ticks() : 0;
			std::vector<uint32_t> pcs;
			const bool materialized = ShaderRecompiler::IR::MaterializeResources(
			    resource_plan, runtime, resources, specialization, &pcs);
			if (plain_t0 != 0) {
				Common::FrameProfiler::Add(Counter::PlainMaterializations);
				Common::FrameProfiler::AddSince(Counter::PlainMaterializeTime, plain_t0);
			}
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
		// One independent copy of the plan per prefetch worker: evaluation keeps scratch in
		// the plan. Empty when the plan is not speculated.
		std::vector<std::unique_ptr<ShaderRecompiler::IR::ResourcePlan>> replicas;
		std::vector<const ShaderRecompiler::IR::ResourcePlan*>           replica_views;
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
				source_changed = cached.RefreshCallSource(runtime);
				if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
					if (source_changed) BuildSiblingsAhead(cached);
				}
				if (!source_changed)
					permutation = cached.FindPermutation<true>(runtime, push_data_cursor,
					                                           prefetcher.get());
			} else {
				permutation =
				    cached.FindPermutation<false>(runtime, push_data_cursor, prefetcher.get());
			}
			if (permutation != nullptr) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				last_materialized = &entry->second;
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
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
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		const auto translate = [&]() {
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
		auto translated = translate();
		// Tables that cannot be specialized move to the bindless heap; each retry adds new pcs.
		const auto materialize = [&]() {
			while (!entry->second.Materialize(runtime, prefetcher.get())) {
				translated = translate();
				entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
				ResetReplicas(entry->second, translated.program);
			}
		};
		if (entry == programs.end()) {
			auto retained = source.call ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			ResetReplicas(entry->second, translated.program);
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
				return permutation->handle;
			}
		}
		const bool table_linked_compute = stage == ShaderType::Compute &&
		                                  entry->second.call_source &&
		                                  entry->second.call_source->table.has_value();
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor,
		    entry->second.FunctionCode(), table_linked_compute));
		entry->second.permutations.back().plan_generation = entry->second.plan_generation;
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		last_materialized = &entry->second;

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
	// Declared last: its workers evaluate the plans above, so it stops first.
	std::unique_ptr<ResourcePrefetcher> prefetcher;
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

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
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
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
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
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id);
	GraphicsPrograms  result;
	auto&             programs = *m_program_cache;
	using Slot                 = ProgramCache::PrefetchSlot;
	if (pixel_active) {
		result.pixel = programs.Get(pixel_params, pixel_info, push_data_cursor);
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

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	auto              program = m_program_cache->Get(params, input_info, push_data_cursor);
	m_program_cache->Learn(ProgramCache::PrefetchSlot::Compute, regs.cs_regs.data_addr, params,
	                       regs.cs_regs.user_sgpr, 0);
	return program;
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

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
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
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
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
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
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
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
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
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
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
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache);
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
	if (const auto key = build_keys.find(compute_program.id); key != build_keys.end()) {
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
