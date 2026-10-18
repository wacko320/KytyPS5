#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/colorClearKernels.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <bitset>
#include <array>
#include <atomic>
#include <map>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	// How long a FindImage answer holds: while no resolve epoch move since epoch touched the
	// desc's guest range, and before the tick at which an overlapping image the lookup spared as
	// recently used may be freed. epoch 0: never.
	struct ResolveStamp {
		uint64_t epoch       = 0;
		uint64_t expiry_tick = UINT64_MAX;
	};

	// stamp receives how long this answer holds (see ReuseImage).
	[[nodiscard]] ImageId FindImage(ImageDesc& desc, bool exact_format = false,
	                                ResolveStamp* stamp = nullptr);
	// Repeats FindImage's answer id for the desc it adjusted, without the lookup, while the
	// answer's stamp holds, and advances the stamp past epoch moves elsewhere. False once it
	// lapsed: the caller asks FindImage again.
	[[nodiscard]] bool ReuseImage(ImageId id, const ImageDesc& desc, uint32_t metadata_base_layer,
	                              ResolveStamp& stamp);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	// Whether the GPU owns the image's contents: it wrote them last and no guest write or buffer
	// write is pending over them (Image::SafeToDownload), so a GPU write that changes nothing
	// leaves its state as it was.
	[[nodiscard]] bool IsGpuOwned(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	// Whether no image overlaps each of the two run-ahead windows (true for an empty one).
	struct AheadImageFree {
		bool up   = true;
		bool down = true;
	};
	// InvalidateMemory, under the same lock answering whether no image overlaps the pages of
	// [up_address, up_address + up_size) and of [down_address, down_address + down_size).
	[[nodiscard]] AheadImageFree InvalidateMemory(uint64_t address, uint64_t size,
	                                              uint64_t up_address, uint64_t up_size,
	                                              uint64_t down_address, uint64_t down_size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	// Any thread: whether a GPU-modified colour image the CPU has not overwritten covers part of
	// the range.
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	// Bytes of the registered images, now and at most since start (any thread, relaxed).
	[[nodiscard]] uint64_t ImageBytes() const noexcept {
		return m_image_bytes.load(std::memory_order_relaxed);
	}
	[[nodiscard]] uint64_t ImagePeakBytes() const noexcept {
		return m_image_peak_bytes.load(std::memory_order_relaxed);
	}
	void ProcessDownloadImages();
	void RunGarbageCollector();

	// Routes MaterializeColorClear took, since start (always on; the frame profiler counts the
	// same per frame).
	struct ColorClearStats {
		uint64_t gpu_checks         = 0; // checks recorded on the GPU
		uint64_t gpu_unchanged      = 0; // GPU-modified metadata no write changed: nothing recorded
		uint64_t storage_clears     = 0; // GPU checks cleared through storage (no predication)
		uint64_t buffer_fills       = 0; // GPU checks cleared into the guest memory's buffer
		uint64_t readbacks          = 0; // GPU-modified metadata read back for a CPU check
		uint64_t cpu_slices_cleared = 0; // slices a CPU check cleared
	};
	[[nodiscard]] ColorClearStats GetColorClearStats() const noexcept;

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
		// The candidate was kept only as recently used: from this tick a lookup frees it.
		uint64_t free_tick = UINT64_MAX;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image, bool storage);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MoveResolveEpoch(GuestRange range);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	// The clear codes DecodeColorClear accepts for the desc, with their colours: a GPU check
	// needs to test only these.
	struct ColorClearCandidates {
		uint32_t                                                     count = 0;
		std::array<uint8_t, ColorClearKernels::MaxCodes>             codes {};
		std::array<vk::ClearColorValue, ColorClearKernels::MaxCodes> colors {};

		[[nodiscard]] bool operator==(const ColorClearCandidates& other) const noexcept;
	};
	[[nodiscard]] static ColorClearCandidates FindColorClearCandidates(const ImageDesc& desc);
	[[nodiscard]] static bool DecodeColorClearCode(const ImageDesc& desc, uint8_t code,
	                                               vk::ClearColorValue& clear);
	// MaterializeColorClear for GPU-modified metadata: records the check, the predicated clears
	// (or buffer fills) and the expansion, or nothing when no write changed the slices since an
	// equal check. False when the device or the image has no GPU route, or the image is
	// guest-owned (the caller reads the metadata back).
	[[nodiscard]] bool MaterializeColorClearOnGpu(ImageId id, const ImageDesc& desc,
	                                              uint64_t slice_size, uint32_t first,
	                                              uint32_t image_first, uint32_t count);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	ColorClearKernels                                 m_color_clear;
	// GPU checks of colour metadata whose outcome stands while no write moves the range's
	// generation (BufferCache::WriteGeneration): per metadata address, the last few checks by
	// slice size and candidates, with the slices they checked (bit per slice; slices from 64 on
	// are always checked again). A range is watched while it has a record. Guarded by m_lock.
	struct ColorClearRecord {
		uint64_t             size       = 0;
		uint64_t             slice_size = 0;
		uint64_t             generation = 0;
		uint64_t             slices     = 0;
		ColorClearCandidates candidates;
	};
	static constexpr size_t COLOR_CLEAR_RECORDS   = 4;
	static constexpr size_t COLOR_CLEAR_ADDRESSES = 1024;
	std::unordered_map<uint64_t, std::vector<ColorClearRecord>> m_color_clear_records;
	struct {
		std::atomic<uint64_t> gpu_checks {0};
		std::atomic<uint64_t> gpu_unchanged {0};
		std::atomic<uint64_t> storage_clears {0};
		std::atomic<uint64_t> buffer_fills {0};
		std::atomic<uint64_t> readbacks {0};
		std::atomic<uint64_t> cpu_slices_cleared {0};
	} m_color_clear_stats;
	uint64_t                                          m_total_used_memory  = 0;
	std::atomic<uint64_t>                             m_image_bytes {0};
	std::atomic<uint64_t>                             m_image_peak_bytes {0};
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	mutable uint32_t m_image_query_epoch      = 0;
	// Recent FindImagesInRegion answers. The page index changes only when an image registers or
	// unregisters, and every draw asks about the same textures again.
	struct RegionQuery {
		uint64_t                address      = 0;
		uint64_t                size         = 0;
		uint64_t                generation   = 0;
		bool                    page_overlap = false;
		uint32_t                count        = 0;
		std::array<ImageId, 16> ids {};
	};
	static constexpr size_t                             REGION_QUERY_COUNT = 64;
	mutable std::array<RegionQuery, REGION_QUERY_COUNT> m_region_queries {};
	// Starts past the zero of unused queries.
	uint64_t m_registration_generation = 1;
	// Moves whenever FindImage could answer an unchanged desc differently: an image registers or
	// unregisters, or an image changes its stencil association, stencil range or subresources, or
	// metadata presence. Dirty state is not part of it; binding refreshes images on its own. The
	// log keeps the image range of each recent move, so answers for other ranges outlive it.
	static constexpr size_t                  RESOLVE_LOG_SIZE = 256;
	uint64_t                                 m_resolve_epoch  = 1;
	std::array<GuestRange, RESOLVE_LOG_SIZE> m_resolve_log {};
	bool             m_readback_linear_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
