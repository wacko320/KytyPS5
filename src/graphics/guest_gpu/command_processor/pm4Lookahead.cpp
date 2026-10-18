#include "graphics/guest_gpu/command_processor/pm4Lookahead.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

namespace {

// Far enough ahead that each draw's evaluation finishes on a worker while the GPU thread
// executes the draws before it.
constexpr size_t   MaxWindow    = 8;
constexpr uint32_t MaxScanWords = 4096;

} // namespace

Pm4Lookahead::Pm4Lookahead(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer),
      m_shadow(std::make_unique<CommandProcessor>(renderer, interrupt_event_id)) {}

Pm4Lookahead::~Pm4Lookahead() = default;

// Only packets whose handler accepts the exact header are decoded: the length of anything else
// is not known to be the one execution uses.
Pm4Lookahead::PacketKind Pm4Lookahead::Classify(const uint32_t* packet) {
	const auto header = packet[0] & ~1u;
	switch ((header >> 8u) & 0xffu) {
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_CLEAR_STATE: return PacketKind::Register;
		case Pm4::IT_NOP:
			switch (KYTY_PM4_R(header)) {
				case Pm4::R_ZERO:
					// Marker payloads carry flips.
					return (packet[1] & 0xffff0000u) == 0x68750000u ? PacketKind::Barrier
					                                                : PacketKind::Transparent;
				case Pm4::R_PUSH_MARKER:
				case Pm4::R_POP_MARKER: return PacketKind::Transparent;
				case Pm4::R_RELEASE_MEM:
					return header == 0xc0061060u ? PacketKind::Transparent : PacketKind::Barrier;
				case Pm4::R_CONTEXT_STATE: return PacketKind::Register;
				default: return PacketKind::Barrier;
			}
		// AgcDcbAcquireMem emits the 8-dword form (byte range in 256-byte units).
		case Pm4::IT_ACQUIRE_MEM:
			return header == 0xc0055800u || header == 0xc0065800u ? PacketKind::Transparent
			                                                      : PacketKind::Barrier;
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_WRITE_DATA: return PacketKind::Transparent;
		case Pm4::IT_EVENT_WRITE_EOP:
			return header == 0xc0044700u ? PacketKind::Transparent : PacketKind::Barrier;
		case Pm4::IT_EVENT_WRITE_EOS:
			return header == 0xc0034802u ? PacketKind::Transparent : PacketKind::Barrier;
		case Pm4::IT_DMA_DATA:
			return header == KYTY_PM4(7, Pm4::IT_DMA_DATA, 0u) ? PacketKind::Transparent
			                                                   : PacketKind::Barrier;
		case Pm4::IT_PFP_SYNC_ME:
			return header == 0xc0004200u ? PacketKind::Transparent : PacketKind::Barrier;
		case Pm4::IT_INCREMENT_DE_COUNTER:
			return header == 0xc0008500u ? PacketKind::Transparent : PacketKind::Barrier;
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DISPATCH_DRAW_PREAMBLE:
			return header == 0xc0073a00u || header == 0xc0042700u ? PacketKind::Draw
			                                                      : PacketKind::Barrier;
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
			return header == 0xc0033500u ? PacketKind::Draw : PacketKind::Barrier;
		case Pm4::IT_DRAW_INDEX_AUTO:
			return header == 0xc0012d00u ? PacketKind::Draw : PacketKind::Barrier;
		// Indirect draws write their arguments into user SGPRs when they execute, so their user
		// data cannot be predicted from the registers.
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: return PacketKind::Barrier;
		case Pm4::IT_DISPATCH_DIRECT:
			return KYTY_PM4_LEN(header) == 5u ? PacketKind::Dispatch : PacketKind::Barrier;
		case Pm4::IT_DISPATCH_INDIRECT:
			return header == 0xc0011600u || header == 0xc0021600u ? PacketKind::Dispatch
			                                                      : PacketKind::Barrier;
		default: return PacketKind::Barrier;
	}
}

void Pm4Lookahead::Reset() {
	m_window.clear();
	m_commands = {};
	m_blocked  = false;
	m_renderer.GetPipelineCache().ClearPrefetch();
}

void Pm4Lookahead::OnDraw(const CommandProcessor& processor, std::span<const uint32_t> commands,
                          uint32_t offset) {
	auto& cache     = m_renderer.GetPipelineCache();
	bool  continued = false;
	if (m_commands.data() == commands.data() && m_commands.size() == commands.size()) {
		while (!m_window.empty() && m_window.front().offset < offset) {
			m_window.pop_front();
		}
		continued = !m_window.empty() && m_window.front().offset == offset;
	}
	if (!continued) {
		// Start over from the executing processor's registers, at this packet.
		cache.ClearPrefetch();
		m_window.clear();
		m_commands = commands;
		m_scan     = offset;
		m_blocked  = false;
		m_shadow->CopyRegisterState(processor);
	}
	Scan();
	uint64_t draw = 0;
	if (!m_window.empty() && m_window.front().offset == offset) {
		draw = m_window.front().draw;
		m_window.pop_front();
	}
	cache.BeginPrefetchedDraw(draw);
}

// Mirrors CommandProcessor::ProcessPm4's framing for the packets it decodes.
void Pm4Lookahead::Scan() {
	auto&      cache  = m_renderer.GetPipelineCache();
	auto&      shadow = *m_shadow;
	const auto total  = static_cast<uint32_t>(m_commands.size());
	uint32_t   budget = MaxScanWords;
	// The speculated draws reach the prefetch workers together, when the scan stops.
	cache.BeginPrefetchBatch();
	while (!m_blocked && m_window.size() <= MaxWindow && m_scan < total && budget != 0) {
		const auto* packet    = m_commands.data() + m_scan;
		const auto  remaining = total - m_scan;
		const auto  header    = packet[0];
		if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
			m_scan++;
			budget--;
			continue;
		}
		if (remaining < 2) {
			m_blocked = true;
			break;
		}
		uint32_t length = 0;
		if ((header >> 30u) == 0u) {
			length = ((header >> 16u) & 0x3fffu) + 2u;
		} else if ((header >> 30u) == 3u) {
			length = Pm4::PacketSizeDw(header);
		}
		if (length == 0 || length > remaining) {
			m_blocked = true;
			break;
		}
		if ((header >> 30u) == 0u || ((header & 1u) != 0 && shadow.ShouldSkipPredicatedPackets())) {
			m_scan += length;
			budget -= std::min(budget, length);
			continue;
		}
		switch (Classify(packet)) {
			case PacketKind::Register: {
				const auto handler = g_cp_op_func[(header >> 8u) & 0xffu];
				const auto used    = handler(shadow, header & ~1u, packet + 1, remaining, total) + 1u;
				if (used > remaining) {
					m_blocked = true;
					continue;
				}
				length = used;
				break;
			}
			case PacketKind::Transparent: break;
			case PacketKind::Draw:
				m_window.push_back({m_scan, cache.PrefetchDraw(shadow.GetShCtx(), shadow.GetUcfg())});
				break;
			case PacketKind::Dispatch: {
				// The dispatch initiator's wave size is written just before the program is read.
				const auto opcode = (header >> 8u) & 0xffu;
				const bool direct = opcode == Pm4::IT_DISPATCH_DIRECT;
				const auto mode   = direct                            ? packet[4]
				                    : (header & ~1u) == 0xc0021600u ? packet[3]
				                                                    : packet[2];
				shadow.GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
				// The workgroup counts the dispatch's materialization sees: an indirect
				// dispatch's are only known on the GPU (zero), a direct one's are the packet's,
				// rounded up to whole workgroups for thread dimensions.
				const auto& cs        = shadow.GetShCtx().GetCs().cs_regs;
				const bool  threads   = (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
				const auto  group_count = [&](uint32_t axis, uint32_t group_size) {
					const auto count = packet[1 + axis];
					const auto size  = std::max(group_size, 1u);
					return threads ? count / size + (count % size != 0u) : count;
				};
				const auto groups = direct ? std::array<uint32_t, 3> {group_count(0, cs.num_thread_x),
				                                                      group_count(1, cs.num_thread_y),
				                                                      group_count(2, cs.num_thread_z)}
				                           : std::array<uint32_t, 3> {};
				m_window.push_back({m_scan, cache.PrefetchDispatch(shadow.GetShCtx(), groups)});
				break;
			}
			case PacketKind::Barrier: m_blocked = true; continue;
		}
		m_scan += length;
		budget -= std::min(budget, length);
	}
	cache.FlushPrefetch();
}

} // namespace Libs::Graphics
