#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_CALLLINKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_CALLLINKER_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

// A call through S_SWAPPC_B64 whose target is a 64-bit code pointer read from a table:
//   s_load_dwordx4        s[D:D+3], s[S:S+1], descriptor_offset   (or the V# in user SGPRs)
//   s_buffer_load_dwordx* s[P:...],  s[D:D+3], index << log2(record_stride)
//   s_swappc_b64          s[R:R+1],  s[P':P'+1]                   (P' copied from the load)
// Rift Apart's ray-tracing kernels call hit shaders this way, through a waterfall over the
// hit group index. Alternatively the pointer is a user SGPR pair, copied by S_MOVs (Wolverine
// passes a function pointer in s[8:9]).
struct CallSite {
	uint32_t pc                = 0;
	uint32_t return_sgpr       = 0; // destination pair; the callee returns through it
	uint32_t pointer_sgpr      = 0; // source pair
	bool     descriptor_in_user_data = false;
	uint32_t descriptor_sgpr   = 0; // user SGPRs holding the V#, when descriptor_in_user_data
	uint32_t srt_sgpr          = 0; // user SGPR pair holding the table's V# address otherwise
	uint32_t descriptor_offset = 0;
	uint32_t record_stride     = 0; // bytes between records; 0 for a constant offset
	uint32_t pointer_offset    = 0; // byte offset of the code pointer from a record's start
	bool     pointer_in_user_data = false;
	uint32_t pointer_user_sgpr    = 0; // user SGPR pair holding the pointer, when set
};

struct CallAnalysis {
	std::vector<CallSite> sites;
	std::string           failure;       // a call exists but its target table is not understood
	uint32_t              program_words = 0; // up to and including the final S_ENDPGM
};

using ReadGuestWords = std::function<bool(uint64_t address, std::span<uint32_t> words)>;

// Cheap prefilter: false when no word of the code can be an S_SWAPPC_B64 with a destination.
bool MayContainCalls(std::span<const uint32_t> code);

// wave_size sizes the lane masks VALU compares write to SGPRs.
CallAnalysis AnalyzeCalls(std::span<const uint32_t> code, uint32_t wave_size);

// Reads the distinct non-null code pointers the site's table holds for this dispatch. table_base,
// when given, receives the table's address (0 for a pointer passed in user SGPRs).
bool ReadCallTargets(const CallSite& site, std::span<const uint32_t> user_data,
                     const ReadGuestWords& read, std::vector<uint64_t>& targets,
                     std::string& failure, uint64_t* table_base = nullptr);

struct LinkedProgram {
	std::vector<uint32_t>   code;
	std::vector<PcOverride> overrides; // for DecodeProgram
	std::string             failure;
};

// Stitches the callees into one program: each call is followed by a copy of each callee behind
// a compare of the pointer with its target, and the callee's returning S_SETPC_B64 becomes a
// branch to the instruction after the call. The chain clobbers SCC, and a pointer outside the targets skips
// the call. targets[i] belongs to analysis.sites[i].
LinkedProgram LinkCalls(std::span<const uint32_t> code, const CallAnalysis& analysis,
                        std::span<const std::vector<uint64_t>> targets,
                        const ReadGuestWords& read);

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_CALLLINKER_H_ */
