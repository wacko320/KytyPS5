#include "graphics/guest_gpu/command_processor/pm4Lookahead.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <cstring>

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

// An indirect register load reads its (offset, value) pairs from guest memory when it executes.
// The shadow applies them as they are now, from the backing store (no page protection, so no
// readback): like the other register writes, they only predict the draw's inputs.
bool Pm4Lookahead::ApplyIndirectRegisters(const uint32_t* packet) {
	const auto header = packet[0] & ~1u;
	const auto opcode = (header >> 8u) & 0xffu;
	if ((opcode != Pm4::IT_SET_CONTEXT_REG_INDIRECT && opcode != Pm4::IT_SET_SH_REG_INDIRECT &&
	     opcode != Pm4::IT_SET_UCONFIG_REG_INDIRECT) ||
	    KYTY_PM4_LEN(header) != 5u) {
		return false;
	}
	const uint64_t address = (static_cast<uint64_t>(packet[1]) & 0xfffffffcu) |
	                         (static_cast<uint64_t>(packet[2]) << 32u);
	const uint32_t count   = packet[4] & 0x3fffu;
	if (count != 0) {
		const auto bytes = uint64_t {count} * 2u * sizeof(uint32_t);
		LibKernel::Memory::GuestBackingView view {};
		if (address == 0 || !m_renderer.IsMapped(address, bytes) ||
		    !LibKernel::Memory::GetGuestBackingView(address, bytes, &view)) {
			return false;
		}
		m_indirect_pairs.resize(count * 2u);
		std::memcpy(m_indirect_pairs.data(), view.pointer, bytes);
		if (!IndirectRegistersKnown(opcode, m_indirect_pairs)) {
			return false;
		}
	}
	// The packet with its address pointing at the copy.
	const auto                    local = reinterpret_cast<uint64_t>(m_indirect_pairs.data());
	const std::array<uint32_t, 5> copy {
	    packet[0], (packet[1] & 3u) | static_cast<uint32_t>(local & 0xfffffffcu),
	    static_cast<uint32_t>(local >> 32u), packet[3], packet[4]};
	g_cp_op_func[opcode](*m_shadow, header, copy.data() + 1, 5u, 5u);
	return true;
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
			case PacketKind::Barrier:
				if (ApplyIndirectRegisters(packet)) {
					break;
				}
				m_blocked = true;
				continue;
		}
		m_scan += length;
		budget -= std::min(budget, length);
	}
	cache.FlushPrefetch();
}

Pm4ThreadDispatchScan::Pm4ThreadDispatchScan(RenderContext& renderer, int interrupt_event_id)
    : m_shadow(std::make_unique<CommandProcessor>(renderer, interrupt_event_id)) {}

Pm4ThreadDispatchScan::~Pm4ThreadDispatchScan() = default;

// Mirrors CommandProcessor::ProcessPm4's framing, as Pm4Lookahead::Scan does.
uint32_t Pm4ThreadDispatchScan::Scan(const CommandProcessor& processor,
                                     std::span<const uint32_t> commands, uint32_t offset,
                                     std::vector<ThreadDispatchRequest>& dispatches) {
	auto& shadow = *m_shadow;
	shadow.CopyRegisterState(processor);
	dispatches.clear();
	const auto total  = static_cast<uint32_t>(commands.size());
	uint32_t   scan   = offset;
	uint32_t   budget = MaxScanWords;
	while (scan < total && budget != 0 && dispatches.size() < MaxDispatches) {
		const auto* packet    = commands.data() + scan;
		const auto  remaining = total - scan;
		const auto  header    = packet[0];
		if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
			scan++;
			budget--;
			continue;
		}
		if (remaining < 2) {
			break;
		}
		uint32_t length = 0;
		if ((header >> 30u) == 0u) {
			length = ((header >> 16u) & 0x3fffu) + 2u;
		} else if ((header >> 30u) == 3u) {
			length = Pm4::PacketSizeDw(header);
		}
		if (length == 0 || length > remaining) {
			break;
		}
		if ((header >> 30u) == 0u || ((header & 1u) != 0 && shadow.ShouldSkipPredicatedPackets())) {
			scan += length;
			budget -= std::min(budget, length);
			continue;
		}
		const auto opcode = (header >> 8u) & 0xffu;
		bool       stop   = false;
		switch (Pm4Lookahead::Classify(packet)) {
			case Pm4Lookahead::PacketKind::Register: {
				const auto handler = g_cp_op_func[opcode];
				const auto used =
				    handler(shadow, header & ~1u, packet + 1, remaining, total) + 1u;
				if (used > remaining) {
					stop = true;
					break;
				}
				length = used;
				break;
			}
			case Pm4Lookahead::PacketKind::Transparent:
				// Markers, PFP_SYNC_ME and DE counters pass; the rest (acquires, releases, events,
				// memory writes) synchronise or write memory.
				stop = opcode != Pm4::IT_PFP_SYNC_ME && opcode != Pm4::IT_INCREMENT_DE_COUNTER &&
				       !(opcode == Pm4::IT_NOP && KYTY_PM4_R(header) != Pm4::R_RELEASE_MEM);
				break;
			case Pm4Lookahead::PacketKind::Dispatch: {
				if (opcode != Pm4::IT_DISPATCH_INDIRECT) {
					break;
				}
				const bool long_form = (header & ~1u) == 0xc0021600u;
				const auto mode      = long_form ? packet[3] : packet[2];
				if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) == 0) {
					break;
				}
				const auto base = shadow.GetDispatchIndirectArgsBaseAddress();
				const auto args = long_form ? packet[1] | (static_cast<uint64_t>(packet[2]) << 32u)
				                            : (base == 0 ? 0 : base + packet[1]);
				if (args == 0 || (args & 3u) != 0) {
					stop = true;
					break;
				}
				const auto& cs = shadow.GetShCtx().GetCs().cs_regs;
				dispatches.push_back({args, {cs.num_thread_x, cs.num_thread_y, cs.num_thread_z}});
				break;
			}
			case Pm4Lookahead::PacketKind::Draw:
			case Pm4Lookahead::PacketKind::Barrier: stop = true; break;
		}
		if (stop) {
			break;
		}
		scan += length;
		budget -= std::min(budget, length);
	}
	return scan;
}

} // namespace Libs::Graphics
