#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSIMAGES_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSIMAGES_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <array>
#include <cstdint>
#include <optional>

// ABI shared by the SPIR-V emitter and the renderer for sampled images whose descriptors only
// the GPU knows (bindless: a T# read from a table or a pointer chosen at runtime).
//
// Descriptor set 1 holds the heap:
//   binding 0: storage buffer, the lookup table
//              dword 0 = capacity (a power of two), then capacity entries of EntryDwords:
//              the key's Hash, the T#'s 8 dwords, the class, and the slot + 1 (0 marks an empty
//              entry). A lookup takes the first of MaxProbes entries from Hash whose hash
//              matches and verifies its key, so the host never places an entry behind another
//              with the same hash.
//   binding 1: storage buffer, misses
//              dword 0 = count (atomic), then MaxMisses records of MissDwords: T# and class
//   binding 2 + class: array of HeapSlots sampled images of that class's dimension; the
//              Compare classes view depth images for depth-comparison samples
//   SamplerBinding: array of HeapSlots samplers, keyed by the S# padded with zeros to 8 dwords
//              under SamplerClass; they never compare depth
//   CompareSamplerBinding: the same under CompareSamplerClass, comparing as the S# says, for
//              depth-comparison samples
// Slot 0 of every array is a null image or a default sampler, which misses read.
namespace Libs::Graphics::ShaderRecompiler::IR::Bindless {

inline constexpr uint32_t DescriptorSet  = 1;
inline constexpr uint32_t TableBinding   = 0;
inline constexpr uint32_t MissBinding    = 1;
inline constexpr uint32_t FirstImageBinding = 2;
inline constexpr uint32_t HeapSlots      = 1024;
inline constexpr uint32_t EntryDwords    = 11;
inline constexpr uint32_t MissDwords     = 9;
inline constexpr uint32_t MaxMisses      = 128;
inline constexpr uint32_t MaxProbes      = 8;

enum class ImageClass : uint32_t {
	Dim1D,
	Dim1DArray,
	Dim2D,
	Dim2DArray,
	Dim3D,
	Compare2D,
	Compare2DArray,
	Count
};

inline constexpr uint32_t ClassCount            = static_cast<uint32_t>(ImageClass::Count);
inline constexpr uint32_t SamplerClass          = ClassCount;
inline constexpr uint32_t CompareSamplerClass   = ClassCount + 1u;
inline constexpr uint32_t SamplerBinding        = FirstImageBinding + ClassCount;
inline constexpr uint32_t CompareSamplerBinding = SamplerBinding + 1u;

[[nodiscard]] constexpr bool IsCompare(ImageClass image_class) {
	return image_class == ImageClass::Compare2D || image_class == ImageClass::Compare2DArray;
}

[[nodiscard]] constexpr std::optional<ImageClass> ClassFor(Decoder::ImageDimension dimension,
                                                           bool depth_compare) {
	if (depth_compare) {
		switch (dimension) {
			case Decoder::ImageDimension::Dim2D: return ImageClass::Compare2D;
			case Decoder::ImageDimension::Dim2DArray: return ImageClass::Compare2DArray;
			default: return std::nullopt;
		}
	}
	switch (dimension) {
		case Decoder::ImageDimension::Dim1D: return ImageClass::Dim1D;
		case Decoder::ImageDimension::Dim1DArray: return ImageClass::Dim1DArray;
		case Decoder::ImageDimension::Dim2D: return ImageClass::Dim2D;
		case Decoder::ImageDimension::Dim2DArray: return ImageClass::Dim2DArray;
		case Decoder::ImageDimension::Dim3D: return ImageClass::Dim3D;
		default: return std::nullopt;
	}
}

[[nodiscard]] constexpr Decoder::ImageDimension DimensionFor(ImageClass image_class) {
	switch (image_class) {
		case ImageClass::Dim1D: return Decoder::ImageDimension::Dim1D;
		case ImageClass::Dim1DArray: return Decoder::ImageDimension::Dim1DArray;
		case ImageClass::Dim2D:
		case ImageClass::Compare2D: return Decoder::ImageDimension::Dim2D;
		case ImageClass::Dim2DArray:
		case ImageClass::Compare2DArray: return Decoder::ImageDimension::Dim2DArray;
		case ImageClass::Dim3D: return Decoder::ImageDimension::Dim3D;
		default: return Decoder::ImageDimension::Unknown;
	}
}

// The emitter computes the same function in SPIR-V.
[[nodiscard]] constexpr uint32_t Hash(const std::array<uint32_t, 8>& descriptor,
                                      uint32_t image_class) {
	uint32_t hash = 0x811c9dc5u ^ image_class;
	for (const auto dword: descriptor) {
		hash = (hash ^ dword) * 0x01000193u;
	}
	return hash ^ (hash >> 15u);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::Bindless

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSIMAGES_H_ */
