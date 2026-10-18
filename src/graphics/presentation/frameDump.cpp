#include "graphics/presentation/frameDump.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"

#include <cstring>
#include <ctime>
#include <string>
#include <system_error>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIWDEF static inline
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

namespace Libs::Graphics {

constexpr vk::Format ReadbackFormat = vk::Format::eR8G8B8A8Unorm;

FrameReadback::~FrameReadback() {
	Release();
}

void FrameReadback::Release() {
	if (m_image.image != nullptr) {
		m_graphics.DeleteImage(m_image);
	}
	if (m_buffer != nullptr) {
		vmaDestroyBuffer(m_graphics.allocator, m_buffer, m_allocation);
		m_buffer     = nullptr;
		m_allocation = nullptr;
		m_mapped     = nullptr;
	}
	m_extent = vk::Extent2D {};
}

void FrameReadback::Allocate(vk::Extent2D extent) {
	if (m_extent == extent && m_buffer != nullptr) {
		return;
	}
	Release();

	vk::ImageCreateInfo create {};
	create.imageType     = vk::ImageType::e2D;
	create.extent        = vk::Extent3D {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = ReadbackFormat;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.samples       = vk::SampleCountFlagBits::e1;
	if (!m_graphics.CreateImage(create, m_image)) {
		EXIT("frame dump: cannot allocate a %ux%u readback image\n", extent.width, extent.height);
	}

	vk::BufferCreateInfo buffer {};
	buffer.size  = vk::DeviceSize {extent.width} * extent.height * 4u;
	buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocation {};
	allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	VmaAllocationInfo info {};
	VkBuffer          native = VK_NULL_HANDLE;
	if (vmaCreateBuffer(m_graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer),
	                    &allocation, &native, &m_allocation, &info) != VK_SUCCESS ||
	    info.pMappedData == nullptr) {
		EXIT("frame dump: cannot allocate a %ux%u readback buffer\n", extent.width, extent.height);
	}
	m_buffer = native;
	m_mapped = info.pMappedData;
	m_extent = extent;
}

void FrameReadback::Record(CommandRecorder command, const VulkanImage& source) {
	const vk::Extent2D extent {source.extent.width, source.extent.height};
	Allocate(extent);

	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageMemoryBarrier2 to_destination {};
	to_destination.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	to_destination.dstStageMask        = vk::PipelineStageFlagBits2::eBlit;
	to_destination.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	to_destination.oldLayout           = vk::ImageLayout::eUndefined;
	to_destination.newLayout           = vk::ImageLayout::eTransferDstOptimal;
	to_destination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_destination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_destination.image               = m_image.image;
	to_destination.subresourceRange    = range;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &to_destination;
	command.pipelineBarrier2(dependency);

	vk::ImageBlit blit {};
	blit.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(extent.width),
                                       static_cast<int32_t>(extent.height), 1};
	blit.dstOffsets[1]  = blit.srcOffsets[1];
	command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal, m_image.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &blit, vk::Filter::eNearest);

	auto to_source          = to_destination;
	to_source.srcStageMask  = vk::PipelineStageFlagBits2::eBlit;
	to_source.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	to_source.dstStageMask  = vk::PipelineStageFlagBits2::eCopy;
	to_source.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	to_source.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_source.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
	dependency.pImageMemoryBarriers = &to_source;
	command.pipelineBarrier2(dependency);

	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImageToBuffer(m_image.image, vk::ImageLayout::eTransferSrcOptimal, m_buffer, 1,
	                          &copy);

	vk::BufferMemoryBarrier2 to_host {};
	to_host.srcStageMask        = vk::PipelineStageFlagBits2::eCopy;
	to_host.srcAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	to_host.dstStageMask        = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask       = vk::AccessFlagBits2::eHostRead;
	to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_host.buffer              = m_buffer;
	to_host.size                = VK_WHOLE_SIZE;
	vk::DependencyInfo host_dependency {};
	host_dependency.bufferMemoryBarrierCount = 1;
	host_dependency.pBufferMemoryBarriers    = &to_host;
	command.pipelineBarrier2(host_dependency);
}

std::vector<uint8_t> FrameReadback::ReadRgb() const {
	EXIT_IF(m_mapped == nullptr);
	const auto pixels = static_cast<size_t>(m_extent.width) * m_extent.height;
	EXIT_IF(vmaInvalidateAllocation(m_graphics.allocator, m_allocation, 0, VK_WHOLE_SIZE) !=
	        VK_SUCCESS);
	const auto*          rgba = static_cast<const uint8_t*>(m_mapped);
	std::vector<uint8_t> rgb(pixels * 3u);
	for (size_t pixel = 0; pixel < pixels; pixel++) {
		std::memcpy(&rgb[pixel * 3u], &rgba[pixel * 4u], 3);
	}
	return rgb;
}

bool WriteRgbPng(const std::filesystem::path& path, uint32_t width, uint32_t height,
                 std::span<const uint8_t> rgb) {
	if (width == 0 || height == 0 || rgb.size() != size_t {width} * height * 3u) {
		return false;
	}
	std::vector<uint8_t> png;
	// Fast rather than small: the frames being dumped wait for the encoder.
	stbi_write_png_compression_level = 1;
	const auto append = [](void* context, void* data, int size) {
		auto*       out   = static_cast<std::vector<uint8_t>*>(context);
		const auto* bytes = static_cast<const uint8_t*>(data);
		out->insert(out->end(), bytes, bytes + size);
	};
	if (stbi_write_png_to_func(append, &png, static_cast<int>(width), static_cast<int>(height), 3,
	                           rgb.data(), static_cast<int>(width * 3u)) == 0) {
		return false;
	}
	Common::File file;
	if (!file.Create(path)) {
		return false;
	}
	uint32_t written = 0;
	file.Write(png.data(), static_cast<uint32_t>(png.size()), &written);
	file.Close();
	return written == png.size();
}

namespace {

std::filesystem::path OutputDirectory() {
	static const std::filesystem::path directory = [] {
		const std::time_t now = std::time(nullptr);
		std::tm           local {};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		localtime_s(&local, &now);
#else
		localtime_r(&now, &local);
#endif
		char stamp[32] = {};
		std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &local);
		std::error_code ec;
		const auto      root = std::filesystem::current_path(ec);
		const auto      base = std::string("kyty-frames-") + stamp;
		auto            dir  = root / base;
		for (int i = 2; std::filesystem::exists(dir, ec); i++) {
			dir = root / (base + "-" + std::to_string(i));
		}
		std::filesystem::create_directories(dir, ec);
		Log::WriteToConsoleAndLog(fmt::format("Frame dump: writing to {}\n",
		                                      Common::PathToString(dir)));
		return dir;
	}();
	return directory;
}

} // namespace

bool FrameDump::Wants(uint64_t frame_number) {
	const auto start = Config::DumpFramesStart();
	return start != 0 && frame_number >= start && frame_number - start < Config::DumpFramesCount();
}

void FrameDump::Write(uint64_t frame_number, const FrameReadback& readback) {
	const auto path = OutputDirectory() / fmt::format("frame-{}.png", frame_number);
	if (!WriteRgbPng(path, readback.Width(), readback.Height(), readback.ReadRgb())) {
		// On the console as well: a lost frame would otherwise only show up in the comparison.
		Log::WriteToConsoleAndLog(
		    fmt::format("Frame dump: cannot write {}\n", Common::PathToString(path)));
	}
}

} // namespace Libs::Graphics
