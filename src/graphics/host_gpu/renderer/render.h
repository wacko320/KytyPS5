#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/indirectDrawArgs.h"
#include "graphics/host_gpu/renderer/indirectThreadDispatch.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawGpuArgsSource;
struct DrawIndexBufferSource;
struct DrawRenderState;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// An indirect draw's arguments the GPU reads itself (vkCmdDraw*Indirect), so the CPU never
// reads memory GPU work wrote: `address` is their guest address, and `capture_instances` asks to
// keep the instance count for a later draw that reads the NUM_INSTANCES state. address 0: the
// draw's counts are the CPU values.
struct DrawGpuArgs {
	uint64_t address           = 0;
	bool     capture_instances = false;
};

// An instance count kept in an indirect-argument slot (IndirectDrawArgs::ReadInstances): the
// slot, the ring's count of slots taken when it was taken, and the tick that writes it (0: none).
struct DrawInstanceCapture {
	uint64_t taken = 0;
	uint64_t tick  = 0;
	uint32_t slot  = 0;
};

// What a draw with GPU arguments did. Not recorded: nothing was, and the caller reads the
// arguments for the CPU path. A draw with CPU counts always counts as recorded.
struct GpuArgsDraw {
	bool                recorded = true;
	DrawInstanceCapture instances;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With GPU arguments, index_addr is the index base and index_buffer_size (INDEX_BUFFER_SIZE)
	// bounds both the bound range and the index count; the counts and offsets above are unused.
	DrawGpuArgs      gpu_args;
	uint32_t         index_buffer_size          = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	// With GPU arguments, the counts and offsets above are unused.
	DrawGpuArgs      gpu_args;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;
	[[nodiscard]] bool IsRendering() const noexcept { return m_rendering; }

	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

// A guest indirect dispatch with thread dimensions (USE_THREAD_DIMENSIONS) that a scan ahead of
// the command processor found in the guest synchronisation window of a dispatch.
struct ThreadDispatchRequest {
	uint64_t                args_addr = 0;
	std::array<uint32_t, 3> group_size {};
};

// What a guest dispatch hands the renderer about batched thread-dimension conversions:
// `requests`, a batch to record before the dispatch, and `serial`, the batch that covers the
// dispatch (0: none). Recording a batch sets `serial` to it.
struct ThreadDispatchBatch {
	std::span<const ThreadDispatchRequest> requests;
	uint64_t                               serial = 0;
};

class IndirectThreadDispatch;
class TextureResolveCache;
class Buffer;
class SamplerFrontTable;

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context);
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute,
	                    ThreadDispatchBatch* batch = nullptr);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute, ThreadDispatchBatch* batch = nullptr);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	// Binds descriptor set 1, the bindless heap, for stages that use it (after CommitBindings).
	void CommitBindlessHeap(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                        const PipelineCache::Pipeline&     pipeline,
	                        std::span<PreparedBindings* const> bindings);
	// The instance count an indirect draw with GPU arguments kept (GpuArgsDraw::instances).
	[[nodiscard]] uint32_t ReadCapturedInstances(const DrawInstanceCapture& capture);

	IndirectDrawArgs& IndirectDraws();

private:
	[[nodiscard]] IndirectThreadDispatch& IndirectThreads();
	// Guest LDS larger than the device's shared memory lives in a device buffer of the executor's
	// own (grown on demand). An indirect dispatch whose arguments the GPU wrote, or that gives
	// thread dimensions, is not sized on the CPU: it binds the whole buffer.
	void BindSharedMemory(ShaderComputeInputInfo& input, PreparedBindings& bindings,
	                      uint64_t indirect_args = 0);
	// Dispatches carry no barrier of their own, but every LDS-storage dispatch uses the same
	// bytes: one after another LDS dispatch with no global barrier between waits for it.
	void OrderSharedMemory(vk::CommandBuffer command, const PreparedBindings& bindings);
	void NoteSharedMemoryDispatch(const PreparedBindings& bindings);
	// Records batch.requests' conversions (IndirectThreadDispatch::ConvertBatch), after the
	// guest barrier the dispatch would take, and sets batch.serial.
	void RecordThreadBatch(ThreadDispatchBatch& batch);
	// The batched conversion of the arguments at args_addr, or nullptr (and the profiler counts
	// why): none covers this dispatch, the batch has no such entry, or their bytes may have
	// changed since it read them (`change_serial`: BufferCache::ChangeSerial when this dispatch
	// began; its own written bindings come after the conversion).
	[[nodiscard]] const IndirectThreadDispatchArguments*
	FindBatchedThreads(const ThreadDispatchBatch* batch, uint64_t args_addr,
	                   const std::array<uint32_t, 3>& group_size, uint64_t change_serial);
	GpuArgsDraw DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	GpuArgsDraw DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);
	// Whether a draw with GPU arguments must take the CPU path (the frame profiler counts why):
	// from the state its render state is prepared from, then from that render state.
	[[nodiscard]] bool RefuseGpuArgs(const CommandBuffer& buffer, const DrawGpuArgs& gpu_args);
	[[nodiscard]] bool RefuseGpuArgs(const DrawRenderState& state, const DrawGpuArgs& gpu_args);
	// Records the conversion of a draw's GPU arguments when it needs one (an indexed draw's
	// clamp, an instance capture, offsets a native fetch takes from user SGPRs), and returns
	// where its indirect draw command reads them.
	[[nodiscard]] IndirectDrawCommand PrepareGpuArgs(const DrawGpuArgs& gpu_args, bool indexed,
	                                                 uint32_t index_limit, bool drop_offsets,
	                                                 DrawInstanceCapture& capture);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	void PrepareBindlessImages(PreparedBindings& prepared);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, DrawGpuArgsSource* gpu_args);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	void                     UpdateComputeMetaClears(const ShaderComputeInputInfo& input,
	                                                 const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::unique_ptr<IndirectThreadDispatch> m_indirect_threads;
	std::unique_ptr<IndirectDrawArgs>       m_indirect_draws;
	std::unique_ptr<Buffer>                 m_lds_storage;
	// An LDS-storage dispatch was recorded at global barrier count m_lds_global_barriers and no
	// LDS barrier followed it.
	bool                                    m_lds_unordered       = false;
	uint64_t                                m_lds_global_barriers = 0;
	uint64_t                                m_lds_barriers        = 0;
	// The last thread-dimension batch recorded: its requests, their conversions, and the next
	// entry a dispatch is expected to take.
	struct BatchedThreads {
		ThreadDispatchRequest           request;
		IndirectThreadDispatchArguments converted;
	};
	std::vector<BatchedThreads>                  m_thread_batch;
	uint64_t                                     m_thread_batch_serial = 0;
	size_t                                       m_thread_batch_next   = 0;
	std::vector<IndirectThreadDispatchSource>    m_thread_sources;
	std::vector<IndirectThreadDispatchArguments> m_thread_converted;
	std::vector<std::pair<uint64_t, uint64_t>>   m_thread_ranges;
	// Lifetime totals (tests): batches, batched dispatches, and fallbacks by reason (not reached,
	// mismatch, upload, written).
	struct ThreadBatchStats {
		uint64_t                batches = 0;
		uint64_t                batched = 0;
		std::array<uint64_t, 4> fallbacks {};
	} m_thread_stats;
	std::unique_ptr<TextureResolveCache>    m_texture_resolves;
	std::unique_ptr<SamplerFrontTable>      m_samplers;
	// Texture resolves the resolve cache could not answer; tests check that repeats hit.
	uint64_t m_full_texture_resolves = 0;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
