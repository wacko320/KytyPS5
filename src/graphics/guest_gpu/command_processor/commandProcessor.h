#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class Pm4Lookahead;
class Pm4ThreadDispatchScan;

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

// Why a PM4 execution suspended (frame profiler: Event::Suspend).
enum class SuspendReason : uint8_t {
	Other,
	WaitRegMem,
	Ce,
	DeDiff,
	Rewind,
	Predication,
	Yield, // a wait this queue released, or a parked queue's tick, lets another queue run
	Park,  // a dispatch's readback would block on the host GPU while another queue can run
};

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
		// Fetched ahead (CommandProcessor::FetchAhead): `commands` is a copy of the dwords at
		// `live` in guest memory. With `continued`, the cursor below holds the rest of the
		// buffer, still in guest memory.
		std::shared_ptr<const uint32_t[]> fetched;
		const uint32_t*                   live      = nullptr;
		bool                              continued = false;
	};
	// Dwords FetchAhead copied while a packet ran: the stack takes them once the packet returns.
	struct FetchedAhead {
		size_t                      cursor = 0;
		std::shared_ptr<uint32_t[]> words;
		uint32_t                    size         = 0;
		uint32_t                    continuation = 0; // dwords taken from the continuation
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::vector<FetchedAhead> m_fetched_ahead;
	// A rewind passed: the buffer from the REWIND packet on, fetched again (`refetched`, null for
	// none), which ends at `refetched_end` in guest memory.
	std::shared_ptr<uint32_t[]> m_refetched;
	uint32_t                    m_refetched_size = 0;
	const uint32_t*             m_refetched_live = nullptr;
	const uint32_t*             m_refetched_end  = nullptr;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
	// The packet this execution last suspended on, to report waits that never complete.
	const uint32_t*                       m_stall_packet   = nullptr;
	std::chrono::steady_clock::time_point m_stall_since;
	bool                                  m_stall_reported = false;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id);
	~CommandProcessor();

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);
	// The register and draw state packets set, for a lookahead processor.
	void CopyRegisterState(const CommandProcessor& other);

	void            BufferInit();
	void            BufferFlush();
	void            BufferWait();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	// The NUM_INSTANCES state for a draw that takes it.
	[[nodiscard]] uint32_t NumInstances();
	// The NUM_INSTANCES state an indirect draw with CPU-read arguments leaves.
	void SetIndirectInstances(uint32_t instance_count);
	// Whether a draw may take its instance count from the NUM_INSTANCES state the indirect draw
	// packet executing now leaves: true unless a NUM_INSTANCES packet or another indirect draw
	// sets it first in the rest of the stream, before any draw or any packet the scan does not
	// follow.
	[[nodiscard]] bool InstanceCountMayBeRead() const;
	// How an indirect draw took its arguments.
	enum class IndirectArgsRoute {
		Cpu,     // not GPU-modified: the CPU reads them cheaply
		Refused, // GPU-modified, but the draw needs them on the CPU
		Gpu,     // the draw is recorded with the GPU reading them
	};
	// Records the indirect draw at args_addr with its arguments read by the GPU, when they are
	// GPU-modified and the draw allows it. `last`: the packet's last draw, whose instance count
	// a later draw may read (InstanceCountMayBeRead, scanned only for a GPU-route draw).
	[[nodiscard]] IndirectArgsRoute DrawIndirectGpu(uint64_t args_addr, bool indexed, bool last);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
	                  uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier();
	// Guest synchronisation packets: each orders the next dispatch, or the next draw after
	// unordered dispatches, after all earlier work (CommandScheduler::RequestGuestBarrier).
	// Counted for the compute-ordering census.
	enum class GuestSync {
		PartialFlush,
		CacheEvent,
		ReleaseMem,
		AcquireMem,
		WaitRegMem,
		Submission
	};
	void RequestGuestBarrier(GuestSync sync);
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	// Whether the REWIND packet whose payload is at `payload` is valid. One fetched ahead
	// (FetchAhead) as valid stays valid; otherwise the buffer is fetched again from guest memory
	// from the packet on, and once the guest has validated the rewind, the execution reads that
	// copy: the guest validates a rewind once the commands after it are written, and may reuse
	// their memory once it sees the command processor's later stores.
	[[nodiscard]] bool RewindValid(const uint32_t* payload);
	void               WaitForRewind(bool valid);
	// Before a store of the packet executing now becomes visible to the guest (labels and
	// other command-processor writes): the hardware command processor has fetched the dwords
	// after the packet by then, so the guest may reuse their memory once it sees the store.
	// Copies the rest of each command buffer of the execution (up to FetchAheadDwords), which
	// the execution then reads instead of guest memory. GPU thread.
	static void FetchAhead();
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void WriteLodStats(void* dst, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	// How far ahead of a store FetchAhead copies a command buffer (4 KiB).
	static constexpr uint32_t FetchAheadDwords = 1024;
	void ProcessPm4(Pm4Execution& execution);
	// After a packet returns: the stack reads what FetchAhead and RewindValid copied while it
	// ran.
	static void ApplyFetchedAhead(Pm4Execution& execution);
	void SuspendPm4(SuspendReason reason = SuspendReason::Other, uint64_t address = 0,
	                uint32_t func = 0);
	// The thread-dimension batch for the dispatch packet executing now: a new one when the
	// dispatch heads a guest synchronisation window or is an indirect thread-dimension dispatch
	// the current batch does not cover (ThreadDispatchBatch::requests), else the covering one.
	[[nodiscard]] ThreadDispatchBatch PlanThreadBatch(bool indirect_threads);
	void                              NoteThreadBatch(const ThreadDispatchBatch& batch);
	// The command processor writes memory: the batch's window ends.
	void EndThreadWindow() noexcept { m_thread_window.valid = false; }
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	// One whose arguments the GPU reads leaves the count on the GPU: in m_num_instances_capture,
	// read when a later draw needs it (NumInstances), or nowhere when the stream sets it again
	// before any draw reads it (m_num_instances_unknown).
	uint32_t            m_num_instances = 1;
	DrawInstanceCapture m_num_instances_capture;
	bool                m_num_instances_unknown = false;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
	// The address of the WAIT_REG_MEM this processor last suspended on (frame profiler).
	uint64_t  m_suspended_wait              = 0;

	// Decodes draws ahead of execution for the pipeline cache's prefetch workers.
	std::unique_ptr<Pm4Lookahead> m_lookahead;
	bool                          m_lookahead_checked = false;
	// The packet a handler is executing, within its command buffer (empty outside ProcessPm4),
	// and its address in guest memory when the buffer was fetched ahead (null otherwise).
	std::span<const uint32_t> m_packet_commands;
	uint32_t                  m_packet_offset = 0;
	const uint32_t*           m_packet_live   = nullptr;
	// The span of the stream the last thread-dimension window scan covered, and its batch.
	struct ThreadWindow {
		const uint32_t* commands = nullptr;
		uint32_t        begin    = 0;
		uint32_t        end      = 0;
		uint64_t        serial   = 0;
		bool            valid    = false;
	};
	ThreadWindow                           m_thread_window;
	std::unique_ptr<Pm4ThreadDispatchScan> m_thread_scan;
	std::vector<ThreadDispatchRequest>     m_thread_requests;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
