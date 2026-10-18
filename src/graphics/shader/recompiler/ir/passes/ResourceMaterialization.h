#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

// The number of image descriptors an indirect table with `count` candidates (root included)
// binds. Tables the guest grows while streaming (a material table gaining entries) would
// otherwise change the descriptor array's size, and so the SPIR-V and pipeline layout, with every
// new candidate. Up to 16 candidates bind exactly; above, counts round up to a multiple of a
// quarter of their power of two (at least 16): at most a quarter of the slots are padding, and a
// table gaining a few entries keeps its pipeline. Padding slots hold null descriptors that no key
// selects.
[[nodiscard]] constexpr uint32_t IndirectImageArrayBucket(uint32_t count) {
	if (count <= 16u) return count;
	const auto step = std::max<uint32_t>(16u, std::bit_floor(count) / 4u);
	return (count + step - 1u) / step * step;
}

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// The result depends only on the plan, the runtime and the words its readers return.
// With bindless_fallback_pcs, a failure caused by a sampled image's indirect table (a candidate
// that cannot be bound, incompatible candidates, more than MaxIndirectImageCandidates, an
// unreadable table, or captured table reads next to address writes) also appends the root image's
// first-use pc; recompiling with those pcs in CompileOptions::bindless_image_pcs samples such
// images through the bindless heap.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization,
                          std::vector<uint32_t>* bindless_fallback_pcs = nullptr);

// Enables or silences this thread's specialization-failure reports while in scope. Speculative
// evaluations read bytes that may be stale and discard failed results.
class SpecializationFailureReports {
public:
	explicit SpecializationFailureReports(bool enabled);
	~SpecializationFailureReports();
	SpecializationFailureReports(const SpecializationFailureReports&)            = delete;
	SpecializationFailureReports& operator=(const SpecializationFailureReports&) = delete;

private:
	bool m_previous;
};

// Holds this thread's specialization-failure reports while in scope instead of printing them;
// Publish prints them. For an evaluation that learns only afterwards whether its result is
// discarded: a dispatch that parks (QueuePark) fails every read after the park, and evaluates
// again.
class DeferredSpecializationFailureReports {
public:
	DeferredSpecializationFailureReports();
	~DeferredSpecializationFailureReports();
	DeferredSpecializationFailureReports(const DeferredSpecializationFailureReports&) = delete;
	DeferredSpecializationFailureReports& operator=(const DeferredSpecializationFailureReports&) =
	    delete;

	[[nodiscard]] const std::vector<std::string>& Reports() const { return m_reports; }
	void                                          Publish();

private:
	std::vector<std::string>* m_previous;
	std::vector<std::string>  m_reports;
};

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
