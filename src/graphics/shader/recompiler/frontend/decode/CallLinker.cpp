#include "graphics/shader/recompiler/frontend/decode/CallLinker.h"

#include <algorithm>
#include <cstring>
#include <array>
#include <fmt/format.h>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Decoder {
namespace {

constexpr uint32_t NoCode = UINT32_MAX;
// Callees are small; this bounds a runaway decode of a pointer that is not code.
constexpr uint32_t MaxCalleeWords = 20u * 1024u;
constexpr uint64_t MaxTableBytes  = 64ull << 20u;

uint32_t SgprCode(const Operand& operand) {
	switch (operand.kind) {
		case OperandKind::Sgpr: return operand.reg;
		case OperandKind::VccLo: return 106u;
		case OperandKind::VccHi: return 107u;
		default: return NoCode;
	}
}

uint32_t ScalarWriteWidth(const Instruction& inst, uint32_t wave_size) {
	if (inst.family == Family::SMEM) {
		return std::max(1u, inst.data_dwords);
	}
	if (inst.opcode == Opcode::V_READLANE_B32 || inst.opcode == Opcode::V_READFIRSTLANE_B32) {
		return 1u;
	}
	if (inst.family != Family::SOP1 && inst.family != Family::SOP2 &&
	    inst.family != Family::SOPK) {
		return wave_size / 32u; // lane masks
	}
	const auto name = magic_enum::enum_name(inst.opcode);
	return name.ends_with("_B64") || name.ends_with("_I64") || name.ends_with("_U64") ? 2u : 1u;
}

bool Writes(const Instruction& inst, uint32_t code, uint32_t wave_size) {
	for (const auto* dst: {&inst.dst, &inst.dst2}) {
		const auto base = SgprCode(*dst);
		if (base != NoCode && code >= base && code < base + ScalarWriteWidth(inst, wave_size)) {
			return true;
		}
	}
	return false;
}

bool WrittenAnywhere(const Program& program, uint32_t code, uint32_t wave_size) {
	return std::ranges::any_of(program.instructions, [&](const Instruction& inst) {
		return Writes(inst, code, wave_size);
	});
}

class Resolver {
public:
	Resolver(const Program& program, uint32_t wave_size)
	    : m_program(program), m_wave_size(wave_size) {
		m_labels.resize(program.code.size() + 1u);
		for (const auto& inst: program.instructions) {
			if (IsDirectBranch(inst.opcode) && inst.branch_target / 4u < m_labels.size()) {
				m_labels[inst.branch_target / 4u] = true;
			}
		}
	}

	// The last instruction before `index` in the same basic block that writes `code`. nullopt
	// when the block has none, and `entry` is set when no block boundary was crossed either, so
	// the value can be the dispatch's initial one.
	std::optional<uint32_t> Def(uint32_t index, uint32_t code, bool* entry = nullptr) const {
		if (entry != nullptr) {
			*entry = false;
		}
		for (uint32_t i = index; i > 0; i--) {
			const auto& inst = m_program.instructions[i - 1u];
			if (Writes(inst, code, m_wave_size)) {
				return i - 1u;
			}
			if (m_labels[inst.pc / 4u]) {
				return std::nullopt;
			}
		}
		if (entry != nullptr) {
			*entry = true;
		}
		return std::nullopt;
	}

	// Follows S_MOV copies back to the SMEM load that produced `code`.
	struct Source {
		uint32_t load  = 0;
		uint32_t dword = 0;
	};
	std::optional<Source> LoadSource(uint32_t index, uint32_t code) const {
		for (uint32_t depth = 0; depth < 16u; depth++) {
			const auto def = Def(index, code);
			if (!def) {
				return std::nullopt;
			}
			const auto& inst = m_program.instructions[*def];
			const auto  dst  = SgprCode(inst.dst);
			if (inst.opcode == Opcode::S_MOV_B32 || inst.opcode == Opcode::S_MOV_B64) {
				const auto src = SgprCode(inst.src0);
				if (src == NoCode) {
					return std::nullopt;
				}
				code  = src + (code - dst);
				index = *def;
				continue;
			}
			if (inst.family == Family::SMEM) {
				return Source {.load = *def, .dword = code - dst};
			}
			return std::nullopt;
		}
		return std::nullopt;
	}

	// Whether the SGPR still holds its dispatch-initial value at instruction `index`: no write to
	// it can reach there. Writes come only after the use, and no branch from them jumps back to
	// or before it (the game reuses s8 as scratch after its last call through it). Callees are
	// assumed to preserve it, as the caller's own copies after each call show.
	bool HoldsInitialValue(uint32_t index, uint32_t code) const {
		const auto use_pc      = m_program.instructions[index].pc;
		uint32_t   first_write = UINT32_MAX;
		for (const auto& inst: m_program.instructions) {
			if (Writes(inst, code, m_wave_size)) {
				first_write = std::min(first_write, inst.pc);
			}
		}
		if (first_write == UINT32_MAX) {
			return true;
		}
		if (first_write <= use_pc) {
			return false;
		}
		return std::ranges::none_of(m_program.instructions, [&](const Instruction& inst) {
			return (inst.opcode == Opcode::S_SETPC_B64 && inst.pc >= first_write) ||
			       (IsDirectBranch(inst.opcode) && inst.pc >= first_write &&
			        inst.branch_target <= use_pc);
		});
	}

	// Follows S_MOV copies (within the block) back to an SGPR that still holds its
	// dispatch-initial value there: the user SGPR's.
	std::optional<uint32_t> EntrySource(uint32_t index, uint32_t code) const {
		for (uint32_t depth = 0; depth < 16u; depth++) {
			const auto def = Def(index, code);
			if (!def) {
				if (HoldsInitialValue(index, code)) {
					return code;
				}
				return std::nullopt;
			}
			const auto& inst = m_program.instructions[*def];
			if (inst.opcode != Opcode::S_MOV_B32 && inst.opcode != Opcode::S_MOV_B64) {
				return std::nullopt;
			}
			const auto src = SgprCode(inst.src0);
			if (src == NoCode) {
				return std::nullopt;
			}
			code  = src + (code - SgprCode(inst.dst));
			index = *def;
		}
		return std::nullopt;
	}

	const Program& program() const { return m_program; }
	bool           WrittenAnywhere(uint32_t code) const {
        return Decoder::WrittenAnywhere(m_program, code, m_wave_size);
	}

private:
	const Program&    m_program;
	uint32_t          m_wave_size = 64;
	std::vector<bool> m_labels;
};

bool IsBufferLoad(Opcode opcode) {
	switch (opcode) {
		case Opcode::S_BUFFER_LOAD_DWORD:
		case Opcode::S_BUFFER_LOAD_DWORDX2:
		case Opcode::S_BUFFER_LOAD_DWORDX4:
		case Opcode::S_BUFFER_LOAD_DWORDX8:
		case Opcode::S_BUFFER_LOAD_DWORDX16: return true;
		default: return false;
	}
}

std::string ResolveSite(const Resolver& resolver, uint32_t index, CallSite& site) {
	const auto& program = resolver.program();
	const auto& call    = program.instructions[index];
	site.pc             = call.pc;
	site.return_sgpr    = SgprCode(call.dst);
	site.pointer_sgpr   = SgprCode(call.src0);
	if (site.return_sgpr == NoCode || site.pointer_sgpr == NoCode) {
		return "call registers are not SGPR pairs";
	}
	const auto user_low  = resolver.EntrySource(index, site.pointer_sgpr);
	const auto user_high = resolver.EntrySource(index, site.pointer_sgpr + 1u);
	if (user_low && user_high && *user_high == *user_low + 1u) {
		site.pointer_in_user_data = true;
		site.pointer_user_sgpr    = *user_low;
		return {};
	}
	const auto low  = resolver.LoadSource(index, site.pointer_sgpr);
	const auto high = resolver.LoadSource(index, site.pointer_sgpr + 1u);
	if (!low || !high || low->load != high->load || high->dword != low->dword + 1u) {
		return "code pointer is not one 64-bit value from a scalar buffer load";
	}
	const auto& load = program.instructions[low->load];
	if (!IsBufferLoad(load.opcode)) {
		return "code pointer is not loaded from a buffer";
	}
	site.pointer_offset = load.offset + low->dword * 4u;

	switch (load.src1.kind) {
		case OperandKind::Null: site.record_stride = 0; break;
		case OperandKind::LiteralConstant:
		case OperandKind::IntegerInlineConstant:
			site.record_stride = 0;
			site.pointer_offset += load.src1.value;
			break;
		default: {
			const auto offset = SgprCode(load.src1);
			const auto def    = offset != NoCode ? resolver.Def(low->load, offset) : std::nullopt;
			const auto* shift = def ? &program.instructions[*def] : nullptr;
			if (shift == nullptr || shift->opcode != Opcode::S_LSHL_B32 ||
			    shift->src1.kind != OperandKind::IntegerInlineConstant || shift->src1.value > 16u) {
				return "table offset is not an index shifted by a constant";
			}
			site.record_stride = 1u << shift->src1.value;
			break;
		}
	}

	const auto descriptor = SgprCode(load.src0);
	if (descriptor == NoCode) {
		return "table descriptor is not in SGPRs";
	}
	std::optional<uint32_t> descriptor_load;
	bool                    all_entry = true;
	for (uint32_t dword = 0; dword < 4u; dword++) {
		bool       entry = false;
		const auto def   = resolver.Def(low->load, descriptor + dword, &entry);
		all_entry &= !def && entry && !resolver.WrittenAnywhere(descriptor + dword);
		if (def && (!descriptor_load || *descriptor_load == *def)) {
			descriptor_load = def;
		} else if (def || !all_entry) {
			descriptor_load.reset();
			all_entry = false;
			break;
		}
	}
	if (all_entry) {
		site.descriptor_in_user_data = true;
		site.descriptor_sgpr         = descriptor;
		return {};
	}
	if (!descriptor_load) {
		return "table descriptor has no single source";
	}
	const auto& desc = program.instructions[*descriptor_load];
	if (desc.opcode != Opcode::S_LOAD_DWORDX4 || SgprCode(desc.dst) != descriptor ||
	    desc.src1.kind != OperandKind::Null) {
		return "table descriptor is not an S_LOAD_DWORDX4 at a constant offset";
	}
	site.srt_sgpr          = SgprCode(desc.src0);
	site.descriptor_offset = desc.offset;
	if (site.srt_sgpr == NoCode || resolver.WrittenAnywhere(site.srt_sgpr) ||
	    resolver.WrittenAnywhere(site.srt_sgpr + 1u)) {
		return "table descriptor address is not a user SGPR pair";
	}
	return {};
}

constexpr uint32_t Sopp(uint32_t opcode, int32_t simm) {
	return 0xbf800000u | (opcode << 16u) | (static_cast<uint32_t>(simm) & 0xffffu);
}

constexpr uint32_t SoppNop     = 0x00u;
constexpr uint32_t SoppBranch  = 0x02u;
constexpr uint32_t SoppScc0    = 0x04u;
constexpr uint32_t SoppScc1    = 0x05u;
constexpr uint32_t SopcCmpEqU32 = 0x06u;

uint32_t SopcLiteral(uint32_t opcode, uint32_t sgpr) {
	return 0xbf000000u | (opcode << 16u) | (255u << 8u) | sgpr;
}

// Branches are patched once every label is placed. A target beyond a SOPP's reach goes to the
// decoder as an override; the linked code is only ever decoded, never run.
class Emitter {
public:
	explicit Emitter(std::vector<uint32_t>& out): m_out(out) {}

	uint32_t Here() const { return static_cast<uint32_t>(m_out.size()); }

	void Branch(uint32_t opcode, uint32_t target_word) {
		m_out.push_back(Sopp(opcode, 0));
		BranchAt(Here() - 1u, target_word);
	}
	// Retargets the branch at word, keeping its encoding.
	void BranchAt(uint32_t word, uint32_t target_word) { m_fixups.push_back({word, target_word}); }

	std::vector<PcOverride> Patch() {
		std::vector<PcOverride> overrides;
		for (const auto& fixup: m_fixups) {
			const auto delta = static_cast<int64_t>(fixup.target) - (fixup.word + 1);
			const bool reaches = delta >= INT16_MIN && delta <= INT16_MAX;
			m_out[fixup.word] = (m_out[fixup.word] & 0xffff0000u) |
			                    (reaches ? static_cast<uint32_t>(delta) & 0xffffu : 0u);
			if (!reaches) {
				overrides.push_back({.pc = fixup.word * 4u, .value = fixup.target * 4u});
			}
		}
		return overrides;
	}

private:
	struct Fixup {
		uint32_t word;
		uint32_t target;
	};
	std::vector<uint32_t>& m_out;
	std::vector<Fixup>     m_fixups;
};

// Reads a callee through its returning S_SETPC_B64: the first one on the return pair that no
// earlier branch jumps past.
std::string ReadCallee(uint64_t address, uint32_t return_sgpr, const ReadGuestWords& read,
                       std::vector<uint32_t>& words, uint32_t& return_word) {
	constexpr uint32_t Chunk = 256u;
	words.clear();
	uint32_t max_target = 0;
	for (uint32_t index = 0;;) {
		// Every instruction is at most three words with its literal.
		while (words.size() < index + 3u) {
			if (words.size() >= MaxCalleeWords) {
				return fmt::format("callee 0x{:x} has no return", address);
			}
			const auto begin = words.size();
			words.resize(begin + Chunk);
			if (!read(address + begin * 4u, std::span(words).subspan(begin))) {
				return fmt::format("callee 0x{:x} is not readable", address);
			}
		}
		Instruction inst;
		DecodeInstruction(words, index, inst);
		if (inst.opcode == Opcode::UNSUPPORTED) {
			return fmt::format("callee 0x{:x} has an unsupported instruction at +0x{:x}", address,
			                   inst.pc);
		}
		if (IsDirectBranch(inst.opcode)) {
			max_target = std::max(max_target, inst.branch_target);
		}
		if (inst.opcode == Opcode::S_SWAPPC_B64 || inst.opcode == Opcode::S_GETPC_B64) {
			return fmt::format("callee 0x{:x} uses {} at +0x{:x}", address,
			                   magic_enum::enum_name(inst.opcode), inst.pc);
		}
		if (inst.opcode == Opcode::S_SETPC_B64) {
			if (SgprCode(inst.src0) != return_sgpr || inst.pc < max_target) {
				return fmt::format("callee 0x{:x} jumps through S_SETPC_B64 at +0x{:x}", address,
				                   inst.pc);
			}
			return_word = index;
			words.resize(index + 1u);
			return {};
		}
		index += inst.word_count;
	}
}

} // namespace

bool MayContainCalls(std::span<const uint32_t> code) {
	return std::ranges::any_of(code, [](uint32_t word) {
		// SOP1 S_SWAPPC_B64 with any destination but NULL (125).
		return (word & 0xff80ff00u) == 0xbe802100u && ((word >> 16u) & 0x7fu) != 125u;
	});
}

CallAnalysis AnalyzeCalls(std::span<const uint32_t> code, uint32_t wave_size) {
	CallAnalysis result;
	Program      program;
	DecodeProgram(code, program);
	result.program_words = program.instructions.empty()
	                           ? 0u
	                           : (program.instructions.back().pc / 4u) +
	                                 program.instructions.back().word_count;
	const Resolver resolver(program, wave_size);
	for (uint32_t index = 0; index < program.instructions.size(); index++) {
		const auto& inst = program.instructions[index];
		if (inst.opcode != Opcode::S_SWAPPC_B64) {
			continue;
		}
		CallSite site;
		if (auto failure = ResolveSite(resolver, index, site); !failure.empty()) {
			result.failure = fmt::format("S_SWAPPC_B64 at pc 0x{:08x}: {}", inst.pc, failure);
			return result;
		}
		result.sites.push_back(site);
	}
	if (!result.sites.empty() &&
	    std::ranges::any_of(program.instructions, [](const Instruction& inst) {
		    return inst.opcode == Opcode::S_SETPC_B64;
	    })) {
		result.failure = "calls are linked only in programs without S_SETPC_B64";
	}
	return result;
}

bool ReadCallTargets(const CallSite& site, std::span<const uint32_t> user_data,
                     const ReadGuestWords& read, std::vector<uint64_t>& targets,
                     std::string& failure) {
	targets.clear();
	if (site.pointer_in_user_data) {
		if (site.pointer_user_sgpr + 2u > user_data.size()) {
			failure = "code pointer is outside the user SGPRs";
			return false;
		}
		const auto target = user_data[site.pointer_user_sgpr] |
		                    (static_cast<uint64_t>(user_data[site.pointer_user_sgpr + 1u]) << 32u);
		if (target != 0u) {
			targets.push_back(target);
		}
		return true;
	}
	std::array<uint32_t, 4> descriptor {};
	if (site.descriptor_in_user_data) {
		if (site.descriptor_sgpr + 4u > user_data.size()) {
			failure = "table descriptor is outside the user SGPRs";
			return false;
		}
		std::copy_n(user_data.begin() + site.descriptor_sgpr, 4, descriptor.begin());
	} else {
		if (site.srt_sgpr + 2u > user_data.size()) {
			failure = "table descriptor address is outside the user SGPRs";
			return false;
		}
		const auto srt = user_data[site.srt_sgpr] |
		                 (static_cast<uint64_t>(user_data[site.srt_sgpr + 1u]) << 32u);
		if (!read(srt + site.descriptor_offset, descriptor)) {
			failure = fmt::format("table descriptor at 0x{:x} is not readable",
			                      srt + site.descriptor_offset);
			return false;
		}
	}
	const uint64_t base    = descriptor[0] | (static_cast<uint64_t>(descriptor[1] & 0xffffu) << 32u);
	const uint32_t stride  = (descriptor[1] >> 16u) & 0x3fffu;
	const uint64_t records = descriptor[2];
	const uint64_t bytes   = stride != 0u ? records * stride : records;
	const uint64_t end =
	    site.record_stride == 0u ? site.pointer_offset + 8ull : std::min(bytes, MaxTableBytes);
	if (base == 0u || end > bytes || end < site.pointer_offset + 8ull) {
		failure = fmt::format("table 0x{:x} of 0x{:x} bytes has no code pointer", base, bytes);
		return false;
	}
	// A table can be megabytes: reuse the buffer (a fresh one costs a map, zeroing and an unmap)
	// and the targets of recent tables whose bytes are unchanged.
	struct ScannedTable {
		uint64_t              base = 0;
		uint64_t              end  = 0;
		std::vector<uint32_t> words;
		std::vector<uint64_t> targets;
	};
	constexpr size_t                                 RecentTables = 4;
	thread_local std::array<ScannedTable, RecentTables> recent;
	thread_local size_t                                 next_recent = 0;
	thread_local std::vector<uint32_t>                  copy;
	const auto      words = static_cast<size_t>((end + 3u) / 4u);
	if (copy.size() < words) {
		copy.resize(words);
	}
	if (!read(base, std::span(copy.data(), words))) {
		failure = fmt::format("table 0x{:x} of 0x{:x} bytes is not readable", base, end);
		return false;
	}
	const uint32_t* table = copy.data();
	for (const auto& scanned: recent) {
		if (scanned.base == base && scanned.end == end &&
		    std::memcmp(scanned.words.data(), table, words * sizeof(uint32_t)) == 0) {
			targets = scanned.targets;
			return true;
		}
	}
	// Few distinct callees fill many records: a small open-addressed set skips the known ones.
	constexpr size_t      SeenSlots = 4096;
	std::array<uint64_t, SeenSlots> seen {};
	const uint64_t step = site.record_stride != 0u ? site.record_stride : end;
	for (uint64_t offset = site.pointer_offset; offset + 8u <= end; offset += step) {
		const auto word    = offset / 4u;
		const auto pointer = table[word] | (static_cast<uint64_t>(table[word + 1u]) << 32u);
		if (pointer == 0u) {
			continue;
		}
		auto slot = static_cast<size_t>((pointer * 0x9e3779b97f4a7c15ull) >> 52u) % SeenSlots;
		while (seen[slot] != 0u && seen[slot] != pointer) {
			slot = (slot + 1u) % SeenSlots;
		}
		if (seen[slot] == pointer) {
			continue;
		}
		if (targets.size() + 1u >= SeenSlots / 2u) {
			failure = "table holds too many distinct code pointers";
			return false;
		}
		seen[slot] = pointer;
		targets.push_back(pointer);
	}
	std::ranges::sort(targets);
	auto& scanned   = recent[next_recent];
	next_recent     = (next_recent + 1u) % RecentTables;
	scanned.base    = base;
	scanned.end     = end;
	scanned.words.assign(table, table + words);
	scanned.targets = targets;
	if (std::ranges::any_of(targets, [](uint64_t pointer) { return (pointer & 3u) != 0u; })) {
		failure = "table holds an unaligned code pointer";
		return false;
	}
	return true;
}

LinkedProgram LinkCalls(std::span<const uint32_t> code, const CallAnalysis& analysis,
                        std::span<const std::vector<uint64_t>> targets,
                        const ReadGuestWords& read) {
	LinkedProgram result;
	if (analysis.sites.size() != targets.size() || analysis.program_words > code.size() ||
	    !std::ranges::is_sorted(analysis.sites, {}, &CallSite::pc)) {
		result.failure = "call targets do not match the analysis";
		return result;
	}
	struct Callee {
		std::vector<uint32_t> words;
		uint32_t              return_word = 0;
	};
	std::vector<std::vector<Callee>> callees(targets.size());
	std::vector<uint32_t>            inserted(targets.size());
	for (size_t site_index = 0; site_index < targets.size(); site_index++) {
		// Six words of compares per target.
		inserted[site_index] = static_cast<uint32_t>(targets[site_index].size()) * 6u;
		for (const auto target: targets[site_index]) {
			auto& callee = callees[site_index].emplace_back();
			if (auto failure = ReadCallee(target, analysis.sites[site_index].return_sgpr, read,
			                              callee.words, callee.return_word);
			    !failure.empty()) {
				result.failure = std::move(failure);
				return result;
			}
			inserted[site_index] += static_cast<uint32_t>(callee.words.size());
		}
	}
	// Each call is followed by a ladder of tests, each guarding its callee, so every branch is
	// forward and only skips one callee: the structurizer turns backward gotos into loops, and
	// branches over other callees into route variables through all of them.
	// Sites are compared with the original word, not the moved one: each site before it shifts
	// it once.
	const auto Moved = [&](uint32_t word) {
		auto moved = word;
		for (size_t site_index = 0; site_index < targets.size(); site_index++) {
			if (analysis.sites[site_index].pc / 4u < word) {
				moved += inserted[site_index];
			}
		}
		return moved;
	};

	Program program;
	DecodeProgram(code.first(analysis.program_words), program);
	auto&   out = result.code;
	Emitter emit(out);
	size_t  next_site = 0;
	for (const auto& inst: program.instructions) {
		const auto word = inst.pc / 4u;
		if (next_site < analysis.sites.size() && analysis.sites[next_site].pc == inst.pc) {
			const auto& site        = analysis.sites[next_site];
			const auto  return_word = Moved(word + 1u);
			out.push_back(Sopp(SoppNop, 0));
			// A pointer outside the targets falls through every test and skips the call.
			for (size_t callee_index = 0; callee_index < targets[next_site].size(); callee_index++) {
				const auto  target = targets[next_site][callee_index];
				const auto& callee = callees[next_site][callee_index];
				const auto  skip   = emit.Here() + 6u + static_cast<uint32_t>(callee.words.size());
				out.push_back(SopcLiteral(SopcCmpEqU32, site.pointer_sgpr));
				out.push_back(static_cast<uint32_t>(target));
				emit.Branch(SoppScc0, skip);
				out.push_back(SopcLiteral(SopcCmpEqU32, site.pointer_sgpr + 1u));
				out.push_back(static_cast<uint32_t>(target >> 32u));
				emit.Branch(SoppScc0, skip);
				const auto start = emit.Here();
				out.insert(out.end(), callee.words.begin(), callee.words.end());
				out[start + callee.return_word] = Sopp(SoppBranch, 0);
				emit.BranchAt(start + callee.return_word, return_word);
			}
			if (emit.Here() != return_word) {
				result.failure = fmt::format(
				    "linked call layout is inconsistent after the call at pc 0x{:x}: word {} "
				    "expected {}{}",
				    inst.pc, emit.Here(), return_word, [&] {
					    std::string sites;
					    for (size_t i = 0; i < analysis.sites.size(); i++) {
						    sites += fmt::format(" [pc 0x{:x} targets {} words {}]",
						                         analysis.sites[i].pc, targets[i].size(), inserted[i]);
					    }
					    return sites;
				    }());
				return result;
			}
			next_site++;
			continue;
		}
		const auto at = emit.Here();
		if (at != Moved(word)) {
			result.failure = fmt::format(
			    "linked call layout is inconsistent at pc 0x{:x}: word {} expected {} (the code "
			    "has words the decoder skipped)",
			    inst.pc, at, Moved(word));
			return result;
		}
		out.insert(out.end(), code.begin() + word, code.begin() + word + inst.word_count);
		if (IsDirectBranch(inst.opcode)) {
			emit.BranchAt(at, Moved(inst.branch_target / 4u));
		} else if (inst.opcode == Opcode::S_GETPC_B64 && at != word) {
			// The PC must still name the guest code: data follows it there.
			result.overrides.push_back({.pc = at * 4u, .value = inst.pc, .getpc = true});
		}
	}
	// The program's trailing data follows unchanged.
	out.insert(out.end(), code.begin() + analysis.program_words, code.end());
	auto branches = emit.Patch();
	result.overrides.insert(result.overrides.end(), branches.begin(), branches.end());
	std::ranges::sort(result.overrides, {}, &PcOverride::pc);
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
