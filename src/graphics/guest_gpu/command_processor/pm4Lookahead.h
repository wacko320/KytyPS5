#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H

#include <cstdint>
#include <deque>
#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandProcessor;
class RenderContext;

// Decodes the PM4 stream ahead of the command processor, so the resource materialization of
// upcoming draws and dispatches runs on the pipeline cache's prefetch workers while the GPU
// thread executes earlier ones.
//
// A shadow processor applies the register writes between draws. A packet whose effect on the
// registers it does not reproduce (indirect register loads, branches, waits) ends the lookahead
// until execution passes it. The shadow state only predicts inputs: the pipeline cache checks
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

	RenderContext&                    m_renderer;
	std::unique_ptr<CommandProcessor> m_shadow;
	std::span<const uint32_t>         m_commands;
	uint32_t                          m_scan    = 0;
	bool                              m_blocked = false;
	std::deque<PendingDraw>           m_window;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PM4_LOOKAHEAD_H
