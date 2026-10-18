#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/backgroundBuilds.h"
#include "graphics/host_gpu/renderer/pipeline/resourcePrefetch.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 116);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

// A color attachment as a graphics pipeline sees it (PipelineDepthTarget: the depth one).
struct PipelineColorTarget {
	uint32_t                        slot    = 0;
	vk::Format                      format  = vk::Format::eUndefined;
	uint32_t                        samples = 0;
	Prospero::ColorComponentMapping export_mapping;
	Prospero::BufferFormat          guest_format = Prospero::BufferFormat::kInvalid;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

// The owning renderer serializes access. Save may run on any thread: the periodic driver cache
// save runs on a thread of its own while the GPU thread goes on.
class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Writes the driver cache if a pipeline was added since the last save. Any thread; saves
	// do not overlap.
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);

	// Resource materialization ahead of the command processor (see ResourcePrefetcher). The
	// command processor numbers the draws it decodes ahead of execution and reports the one it
	// executes next. GPU thread only.
	// PrefetchDraw and PrefetchDispatch return the number they gave the draw (never 0).
	[[nodiscard]] bool PrefetchEnabled() const;
	uint64_t PrefetchDraw(const HW::Shader& shaders, const HW::UserConfig& user_config);
	// `workgroup_counts`: what the dispatch's materialization will see (zero when indirect).
	uint64_t PrefetchDispatch(const HW::Shader& shaders,
	                          const std::array<uint32_t, 3>& workgroup_counts);
	// Draw 0 matches no speculation.
	void BeginPrefetchedDraw(uint64_t draw);
	void ClearPrefetch();
	// Speculation between these reaches the workers together (ResourcePrefetcher::BeginBatch).
	void BeginPrefetchBatch();
	void FlushPrefetch();
	[[nodiscard]] ResourcePrefetcher::Stats GetPrefetchStats() const;
	// Programs compiled ahead: for a draw or dispatch the lookahead predicted, the stages whose
	// programs are not cached yet are translated and compiled on low-priority workers (and a
	// compute stage's pipeline built). The draw takes a worker's program only for its exact
	// ProgramKey, specialization and push-data cursor, waiting for it when it is being built;
	// a misprediction is only wasted work. GPU thread.
	void PrecompileDraw(const HW::Context& context, const HW::Shader& shaders,
	                    const HW::UserConfig& user_config);
	// `indirect_args_address`: set for an indirect dispatch, else `workgroup_counts` are the
	// direct dispatch's.
	void PrecompileDispatch(const HW::Shader& shaders, const HW::ShaderRegisters& sh, uint32_t mode,
	                        bool async_compute, const std::array<uint32_t, 3>& workgroup_counts,
	                        std::optional<uint64_t> indirect_args_address);
	struct PrecompileStats {
		uint64_t submitted = 0; // stages handed to the workers
		uint64_t hits      = 0; // new permutations taken from a worker
		uint64_t misses    = 0; // new permutations compiled on the GPU thread
		uint64_t wasted    = 0; // worker stages dropped (unused, or only their plan used)
		uint64_t pipeline_hits = 0; // graphics pipelines taken from a worker
	};
	[[nodiscard]] PrecompileStats GetPrecompileStats() const;
	// Tests: until no precompile is queued or running.
	void WaitForPrecompiles();
	// Stops the workers that build ahead (sibling builds, precompiles) after their current build,
	// and logs the precompile statistics. Any thread: at exit, where the cache itself is not
	// destroyed, so no worker still uses the device when it goes.
	void StopBackgroundBuilds();
	// Guest programs translated to IR so far: a new shader, a relinked call, a bindless retry.
	[[nodiscard]] uint64_t TranslationCount() const;
	// See ResourcePrefetcher::MemoryOps::replay_check: on from the start in debug builds.
	void SetPrefetchReplayCheck(bool enabled);

private:
	struct ProgramCache;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	static bool BuildGraphicsPipelineKey(GraphicContext&                     graphics,
	                                     std::span<const PipelineColorTarget> colors,
	                                     const PipelineDepthTarget&          depth,
	                                     const ShaderVertexInputInfo&        vs_input_info,
	                                     const HW::Context&                  ctx,
	                                     Prospero::PrimitiveType             primitive_type,
	                                     const ShaderPixelInputInfo*         ps_input_info,
	                                     vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                     GraphicsPipelineKey& key, bool speculative);

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	vk::PipelineCache             m_driver_cache = nullptr;
	// Compute pipelines built ahead on worker threads: a linked kernel that shares a call table
	// with one that just relinked (ProgramCache::BuildSiblingsAhead). Keyed by
	// ComputePipelineBuildKey.
	std::unique_ptr<KeyedBuildQueue<Pipeline>> m_background_builds;
	std::filesystem::path         m_driver_cache_path;
	// GPU thread only (CheckpointDriverCache).
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	// Set by every pipeline insert into the driver cache, on the GPU thread or a background
	// build's worker; cleared by the Save that writes it.
	std::atomic<bool>             m_driver_cache_dirty {false};
	// One Save at a time: the periodic one on m_save_thread, and the ones at exit.
	std::mutex                    m_save_mutex;
	// The periodic save runs on its own thread, so vkGetPipelineCacheData and the file write
	// stay off the GPU thread and out of the render lock.
	std::mutex                    m_save_state_mutex;
	std::condition_variable       m_save_wake;
	bool                          m_save_requested = false;
	bool                          m_save_stop      = false;
	std::thread                   m_save_thread;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;

	void InitializeDriverCache();
	void CheckpointDriverCache();
	void StopDriverCacheSaves();
	// Save under m_save_mutex: false when the cache could not be read or written.
	bool WriteDriverCache();
};

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// GPU thread: whether one materialization's compiled SRT walker may load [address, address + size)
// through the guest pointer (the raw_table_reads check, in a fresh checked-read scope).
[[nodiscard]] bool CanLoadSrtTableRaw(uint64_t address, uint64_t size);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache);
// Whether CreatePipelineInternal creates these without stopping the emulator: for pipelines built
// ahead from predicted state.
[[nodiscard]] bool CanCreatePipeline(GraphicContext& graphics, const PipelineRenderingState& rendering,
                                     const PipelineVertexInputState&        vertex_input,
                                     std::span<const ShaderVertexInputInfo> vertex_info,
                                     const ShaderPixelInputInfo*            ps_input_info,
                                     const PipelineStaticParameters&        static_params);
[[nodiscard]] bool CanCreatePipeline(GraphicContext& graphics, const ShaderComputeInputInfo& input_info);
// Identifies what CreatePipelineInternal builds for a compute program from this SPIR-V: equal keys
// create equal pipelines.
uint64_t ComputePipelineBuildKey(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 std::span<const uint32_t>                       spirv);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
