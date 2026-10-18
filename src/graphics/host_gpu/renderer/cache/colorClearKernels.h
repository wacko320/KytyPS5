#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARKERNELS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARKERNELS_H_

#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <span>

namespace Libs::Graphics {

class CommandScheduler;
struct GraphicContext;

// The GPU side of the colour fast-clear check (TextureCache::MaterializeColorClear) for metadata
// the GPU wrote: the slices are checked against the candidate clear codes in the buffer that
// holds them, and the result is a predicate record per slice and code, which conditional
// rendering (or, without it, an indirect storage clear) consumes. Nothing is read back.
class ColorClearKernels {
public:
	// DecodeColorClear accepts at most five codes; the push constants have room for eight.
	static constexpr uint32_t MaxCodes = 8;

	ColorClearKernels(GraphicContext& graphics, CommandScheduler& scheduler);
	~ColorClearKernels();
	KYTY_CLASS_NO_COPY(ColorClearKernels);

	// Where a recorded check left its results.
	struct Check {
		vk::Buffer results        = nullptr;
		uint64_t   mask_offset    = 0;
		uint64_t   records_offset = 0;
		uint32_t   codes          = 0;
		uint32_t   slices         = 0;

		// The 16-byte record of slice and candidate code index: its first word is non-zero iff
		// every byte of the slice equals the code, and the first three are the storage clear's
		// workgroup counts (VkDispatchIndirectCommand).
		[[nodiscard]] uint64_t RecordOffset(uint32_t slice, uint32_t code) const noexcept {
			return records_offset + (static_cast<uint64_t>(slice) * codes + code) * RecordBytes;
		}
	};

	// Whether a check of this many slices and codes fits the results ring.
	[[nodiscard]] static bool Fits(uint32_t slices, uint32_t codes) noexcept;

	// Records the check of `slices` consecutive slices of slice_bytes (a multiple of 4) at
	// metadata + offset (a multiple of 4) against the codes, after every earlier write. The
	// predicate records carry store_groups as the storage clear's group counts.
	[[nodiscard]] Check RecordCheck(CommandRecorder command, vk::Buffer metadata, uint64_t offset,
	                                uint64_t slice_bytes, uint32_t slices,
	                                std::span<const uint8_t> codes, vk::Extent2D store_groups);
	// Records the expansion of the check's matched slices to 0xff bytes, as the hardware leaves
	// decompressed metadata, and a barrier before later readers of the metadata.
	void RecordExpand(CommandRecorder command, const Check& check, vk::Buffer metadata,
	                  uint64_t offset, uint64_t slice_bytes);
	// The fallback without conditional rendering: a storage clear of the view (one 2D layer in
	// GENERAL layout) to the colour, dispatched with the record's group counts.
	void RecordStoreClear(CommandRecorder command, vk::ImageView view, vk::Extent2D extent,
	                      const std::array<float, 4>& color, const Check& check, uint32_t slice,
	                      uint32_t code);
	// For an image whose guest memory the GPU wrote as a buffer: a fill of the slice's
	// slice_bytes at buffer + offset (multiples of 4) with the clear texel (pattern words
	// alternating), dispatched with the record's group counts, and a barrier before later
	// readers and writers of the buffer.
	void RecordBufferFill(CommandRecorder command, vk::Buffer buffer, uint64_t offset,
	                      uint64_t slice_bytes, const std::array<uint32_t, 2>& pattern,
	                      const Check& check, uint32_t slice, uint32_t code);

	static constexpr vk::Extent2D StoreGroupSize {8, 8};

private:
	static constexpr uint64_t RecordBytes  = 16;
	static constexpr uint64_t ResultsBytes = 1024 * 1024;

	void BindCheck(CommandRecorder command, vk::Buffer metadata, uint64_t window,
	               uint64_t range);

	GraphicContext&         m_graphics;
	Buffer                  m_results;
	uint64_t                m_next                  = 0;
	vk::DescriptorSetLayout m_check_layout          = nullptr;
	vk::PipelineLayout      m_check_pipeline_layout = nullptr;
	vk::Pipeline            m_check                 = nullptr;
	vk::DescriptorSetLayout m_store_layout          = nullptr;
	vk::PipelineLayout      m_store_pipeline_layout = nullptr;
	vk::Pipeline            m_store                 = nullptr;
	vk::DescriptorSetLayout m_fill_layout           = nullptr;
	vk::PipelineLayout      m_fill_pipeline_layout  = nullptr;
	vk::Pipeline            m_fill                  = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COLORCLEARKERNELS_H_
