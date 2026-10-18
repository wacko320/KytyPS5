#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H

#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class CommandProcessor;
class RenderContext;
struct ThreadDispatchRequest;

// Decodes the PM4 stream ahead of the command processor, so the resource materialization of
// upcoming draws and dispatches runs on the pipeline cache's prefetch workers while the GPU
// thread executes earlier ones.
//
// A shadow processor applies the register writes between draws, indirect register loads with
// the pairs guest memory holds now. A packet whose effect on the registers it does not reproduce
// (branches, waits) ends the lookahead until execution passes it. The shadow state only predicts inputs: the pipeline cache checks
// each speculated result against the real draw before using it.
class Pm4Lookahead {
public:
	enum class PacketKind { Barrier, Register, Transparent, Draw, Dispatch };

	Pm4Lookahead(RenderContext& renderer, int interrupt_event_id);
	~Pm4Lookahead();
	Pm4Lookahead(const Pm4Lookahead&)            = delete;
	Pm4Lookahead& operator=(const Pm4Lookahead&) = delete;

	// How the lookahead treats a type-3 packet; `packet` points at its header.
	[[nodiscard]] static PacketKind Classify(const uint32_t* packet);

	// The processor is about to execute the draw or dispatch packet at `offset` in `commands`.
	void OnDraw(const CommandProcessor& processor, std::span<const uint32_t> commands,
	            uint32_t offset);
	void Reset();

private:
	struct PendingDraw {
		uint32_t offset = 0;
		uint64_t draw   = 0;
	};

	void Scan();
	bool ApplyIndirectRegisters(const uint32_t* packet);

	RenderContext&                    m_renderer;
	std::unique_ptr<CommandProcessor> m_shadow;
	std::span<const uint32_t>         m_commands;
	uint32_t                          m_scan    = 0;
	bool                              m_blocked = false;
	std::deque<PendingDraw>           m_window;
	std::vector<uint32_t>             m_indirect_pairs;
};

// Finds the indirect dispatches with thread dimensions (USE_THREAD_DIMENSIONS) in the rest of a
// guest synchronisation window, so that their conversions to workgroup counts can be recorded
// together, with one barrier, ahead of the window's dispatches (IndirectThreadDispatch).
//
// From a dispatch about to execute, a shadow processor applies the register writes up to the
// first packet that synchronises (a partial flush or cache event, a release, an acquire, a label
// wait), that the command processor writes memory with (WRITE_DATA, DMA_DATA, COPY_DATA), that
// the lookahead cannot follow (PacketKind::Barrier), or a draw. On the hardware the command
// processor reads indirect arguments when it reaches their packet, while earlier dispatches may
// still run: a guest that reads GPU-produced arguments synchronises before them, so arguments
// are final at the window's start. Command-processor writes end the window, so arguments it
// writes stay exact. As with the lookahead, the result only predicts: each dispatch checks its
// own arguments against the batch.
class Pm4ThreadDispatchScan {
public:
	Pm4ThreadDispatchScan(RenderContext& renderer, int interrupt_event_id);
	~Pm4ThreadDispatchScan();
	Pm4ThreadDispatchScan(const Pm4ThreadDispatchScan&)            = delete;
	Pm4ThreadDispatchScan& operator=(const Pm4ThreadDispatchScan&) = delete;

	// Scans from the dispatch packet at `offset`, which `processor` is executing, and returns the
	// offset of the packet the scan stopped at: the dispatches before it are in `dispatches`.
	[[nodiscard]] uint32_t Scan(const CommandProcessor&           processor,
	                            std::span<const uint32_t>         commands, uint32_t offset,
	                            std::vector<ThreadDispatchRequest>& dispatches);

private:
	static constexpr uint32_t MaxScanWords  = 32768;
	static constexpr size_t   MaxDispatches = 512;

	std::unique_ptr<CommandProcessor> m_shadow;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H
