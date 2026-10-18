#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEDUMP_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEDUMP_H_

// --dump-frames: the image the presenter blits to the swapchain, read back from that source image
// (never from the desktop) and written as PNG. A measurement tool like --profile-frames, for
// comparing what two runs presented at the same frame.

#include "common/common.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Copies an image into host memory as RGB8: a blit to RGBA8 converts any blittable colour format
// (and BGRA order), then a buffer copy makes it host-readable. One readback in flight at a time.
class FrameReadback final {
public:
	explicit FrameReadback(GraphicContext& graphics): m_graphics(graphics) {}
	~FrameReadback();
	KYTY_CLASS_NO_COPY(FrameReadback);

	// Records the copy. `source` must be in TransferSrcOptimal with its writes made visible to
	// transfer reads, as it is after the presenter's blit to the swapchain.
	void Record(CommandRecorder command, const VulkanImage& source);
	// RGB8 rows, top to bottom, once the recording's submission has completed.
	[[nodiscard]] std::vector<uint8_t> ReadRgb() const;
	[[nodiscard]] uint32_t             Width() const noexcept { return m_extent.width; }
	[[nodiscard]] uint32_t             Height() const noexcept { return m_extent.height; }

private:
	void Allocate(vk::Extent2D extent);
	void Release();

	GraphicContext& m_graphics;
	vk::Extent2D    m_extent {};
	VulkanImage     m_image;
	vk::Buffer      m_buffer     = nullptr;
	VmaAllocation   m_allocation = nullptr;
	void*           m_mapped     = nullptr;
};

[[nodiscard]] bool WriteRgbPng(const std::filesystem::path& path, uint32_t width, uint32_t height,
                               std::span<const uint8_t> rgb);

// The frames --dump-frames asks for, and where their PNGs go: kyty-frames-<timestamp>/ in the
// working directory, one frame-<number>.png per frame. Frame numbers are those of --frame-times
// and the frame profiler.
class FrameDump final {
public:
	[[nodiscard]] static bool Wants(uint64_t frame_number);
	// Writes the readback of `frame_number`, whose recording has completed. The wait and the
	// encoding run on the presenting thread, so dumped frames take far longer to present (about
	// 600 ms at 3840x2160) and the game paces differently while it is dumped: their frame times
	// are not normal timing, and a dump window must not overlap a --profile-frames window.
	static void Write(uint64_t frame_number, const FrameReadback& readback);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMEDUMP_H_
