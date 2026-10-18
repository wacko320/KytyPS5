// shader_dump_tool [options] <code.bin> [user_sgprs] [group_size] [spirv [out.spv]]
//
// Translates a raw compute shader dump (as written by --graphics-debug-dump) and prints what
// resource tracking made of it; with "spirv" it also compiles and validates the SPIR-V. Not a
// test: it is an offline oracle for recompiler changes over a set of real shaders.
//
// Options:
//   --early-dump          print the native IR before the optimisation passes
//   --scratch-dwords N    per-lane scratch size of the dispatch
//   --wave32              compile for a wave32 guest
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "spirv-tools/libspirv.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <magic_enum.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	using namespace Libs::Graphics;
	bool                     early_dump     = false;
	bool                     wave32         = false;
	uint32_t                 scratch_dwords = 0;
	std::vector<const char*> args;
	for (int i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--early-dump") == 0) {
			early_dump = true;
		} else if (std::strcmp(argv[i], "--wave32") == 0) {
			wave32 = true;
		} else if (std::strcmp(argv[i], "--scratch-dwords") == 0 && i + 1 < argc) {
			scratch_dwords = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
		} else {
			args.push_back(argv[i]);
		}
	}
	if (args.empty()) {
		std::fprintf(stderr,
		             "usage: %s [--early-dump] [--wave32] [--scratch-dwords N] <code.bin> "
		             "[user_sgprs] [group_size] [spirv [out.spv]]\n",
		             argv[0]);
		return 2;
	}
	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::LogDirection::Console;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();

	std::ifstream file(args[0], std::ios::binary);
	if (!file) {
		std::fprintf(stderr, "cannot open %s\n", args[0]);
		return 2;
	}
	std::vector<char>     bytes {std::istreambuf_iterator<char>(file), {}};
	std::vector<uint32_t> code((bytes.size() + 3) / 4);
	std::memcpy(code.data(), bytes.data(), bytes.size());
	std::vector<uint32_t> user_data(args.size() > 1 ? std::strtoul(args[1], nullptr, 0) : 16u);

	// Workgroup id X follows the user SGPRs; one thread-id VGPR.
	ShaderComputeInputInfo compute;
	compute.threads_num[0]      = args.size() > 2 ? std::strtoul(args[2], nullptr, 0) : 64u;
	compute.threads_num[1]      = 1;
	compute.threads_num[2]      = 1;
	compute.group_id[0]         = true;
	compute.thread_ids_num      = 1;
	compute.workgroup_register  = static_cast<int>(user_data.size());
	compute.scratch_size_dwords = scratch_dwords;
	compute.wave_size           = wave32 ? 32u : 64u;
	ShaderRecompiler::CompileOptions options;
	options.input_info.compute = &compute;
	options.stage              = ShaderType::Compute;
	options.wave_size          = compute.wave_size;
	options.dump_ir            = true;
	options.early_dump         = early_dump;
	options.user_data          = user_data;
	// As the emulator compiles them (see PipelineCache).
	options.untracked_writes   = true;
	options.bindless_images    = true;
	options.table_calls        = true;
	// Calls link against the dump's (zero) user data and no guest memory: a pointer passed in
	// user SGPRs has no target, so the call is skipped and only the caller is translated.
	auto source = ShaderRecompiler::PrepareShaderSource(code, options);
	const ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .read_specialization_memory = +[](void*, uint64_t, std::span<uint32_t>) { return false; },
	};
	auto result = ShaderRecompiler::TranslateProgram(
	    source.call ? ShaderRecompiler::RefreshShaderSource(source, runtime) : source.decoded,
	    options);
	const auto& program        = result.program;
	std::printf("uses_dma=%d srt_reads=%zu buffers=%zu images=%zu\n", program.info.uses_dma,
	            program.srt_reads.size(), program.info.buffers.size(), program.info.images.size());
	for (size_t index = 0; index < program.memory_info.size(); index++) {
		const auto& memory = program.memory_info[index];
		std::printf("  memory[%zu] kind=%s resource=%u offset=%u bits=%u dwords=%u planning=%d\n",
		            index, std::string(magic_enum::enum_name(memory.kind)).c_str(), memory.resource,
		            memory.offset, memory.data_bits, memory.data_dwords, memory.planning_only);
	}
	if (args.size() > 3 && std::strcmp(args[3], "spirv") == 0) {
		// Raw, unformatted buffers and float 2D images unless the shader says otherwise.
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		specialization.buffers.resize(program.info.buffers.size());
		specialization.images.resize(program.info.images.size());
		for (size_t index = 0; index < specialization.images.size(); index++) {
			const auto& image      = program.info.images[index];
			auto&       specific   = specialization.images[index];
			specific.numeric_class = image.numeric_class != Prospero::TextureNumericClass::Unsupported
			                             ? image.numeric_class
			                             : Prospero::TextureNumericClass::Float;
			specific.dimension = image.dimension != ShaderRecompiler::Decoder::ImageDimension::Unknown
			                         ? image.dimension
			                         : ShaderRecompiler::Decoder::ImageDimension::Dim2D;
			specific.cube                       = image.cube;
			specific.indirect_root              = image.indirect_root;
			specific.indirect_mapping_offset    = image.indirect_mapping_offset;
			specific.indirect_search_iterations = image.indirect_search_iterations;
		}
		for (size_t index = 0; index < specialization.buffers.size(); index++) {
			specialization.buffers[index].packed_stride = program.info.buffers[index].packed_stride;
		}
		specialization.samplers.resize(program.info.samplers.size());
		for (size_t index = 0; index < specialization.samplers.size(); index++) {
			const auto& sampler = program.info.samplers[index];
			auto&       specific = specialization.samplers[index];
			specific.indirect_root              = sampler.indirect_root;
			specific.indirect_mapping_offset    = sampler.indirect_mapping_offset;
			specific.indirect_search_iterations = sampler.indirect_search_iterations;
		}
		const auto compiled =
		    ShaderRecompiler::CompileProgram(std::move(result), options, specialization);
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
		std::string          messages;
		tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&,
		                             const char* message) { messages += message + std::string("\n"); });
		spvtools::ValidatorOptions validator;
		validator.SetScalarBlockLayout(true);
		const bool valid = tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator);
		std::printf("spirv words=%zu valid=%d\n%s", compiled.spirv.size(), valid, messages.c_str());
		if (args.size() > 4) {
			std::ofstream out(args[4], std::ios::binary);
			out.write(reinterpret_cast<const char*>(compiled.spirv.data()),
			          static_cast<std::streamsize>(compiled.spirv.size() * 4));
		}
	}
	return 0;
}
