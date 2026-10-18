#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	// The entry after the last page points at a zeroed page, which shader reads of missing or
	// out-of-range pages use instead of branching around the load.
	static constexpr uint64_t BDA_ZERO_ENTRY = CACHING_NUMPAGES;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    (CACHING_NUMPAGES + 1u) * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// ReadMemory from a guest thread: the GPU thread records the download, submits it and goes
	// on; this thread waits for it, then a second short command hands back the pages no GPU
	// write or pending publish has touched since.
	void                   ReadMemoryFromGuestThread(uint64_t vaddr, uint64_t size, bool is_write);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// Called once the command that writes the ranges bound since the last call is recorded: they
	// are copied back (hot pages only) at the end of the submit that holds the write, not of one
	// a stream-buffer wrap made between the binding and the command.
	void CommitWrittenRanges();
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	// Writes BDA_ZERO_ENTRY once; called while recording, before shaders read the table.
	void EnsureBdaZeroEntry();
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Any thread: none of these bytes is GPU-written or on its way back from the GPU (a pending
	// guest write), so guest memory holds their value.
	[[nodiscard]] bool IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// GPU thread: a command-processor write into pages the GPU owns, recorded after the GPU work
	// before it instead of draining the GPU to merge it into guest memory. False (nothing
	// recorded) unless every tracker page of the range is GPU-dirty.
	bool TryWriteGpuOrdered(uint64_t vaddr, const void* data, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBda(const RangeSet& mapped);
	void               QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	// parts(vaddr, size, out) returns true when the whole range is mapped; otherwise it lists the
	// mapped sub-ranges in `out` as [begin, end) pairs, in address order.
	using MappedParts = std::vector<std::pair<uint64_t, uint64_t>>;
	void SetMappedParts(std::function<bool(uint64_t vaddr, uint64_t size, MappedParts& out)> parts) {
		m_mapped_parts = std::move(parts);
	}
	// Deferred operations that write guest memory register their range here first and release
	// it after the write, so an unmap can tell whether one is still on its way. Any thread.
	void               NotePendingGuestWrite(uint64_t vaddr, uint64_t size);
	void               DonePendingGuestWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasPendingGuestWrite(uint64_t vaddr, uint64_t size) const;
	// Whether in-flight GPU work or a pending write to guest memory may involve the range: a
	// registered buffer or a pending guest write (downloads included). GPU thread.
	[[nodiscard]] bool MayBeInUseByGpu(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();
	// GPU thread: the registered buffers.
	[[nodiscard]] uint64_t BufferBytes() const noexcept { return m_buffer_bytes; }
	[[nodiscard]] size_t   BufferCount() const noexcept { return m_buffers.size(); }

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Calls func(begin, end) for each mapped run of [start, end), once for the whole range when
	// it is all mapped.
	template <typename Func>
	void ForEachMappedPart(uint64_t start, uint64_t end, Func&& func);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Records a copy of buffer ranges to the guest backing in the current command buffer. `async`:
	// published once the submission completes (a priority operation); else this waits for it and
	// publishes before returning. The caller has noted the copies as pending guest writes
	// (NotePendingGuestWrite); the publish releases them.
	void RecordDownload(Buffer& buffer, std::vector<vk::BufferCopy>&& copies, uint64_t total_size,
	                    bool async);

	// Write-back of GPU output the CPU reads. Once a page has cost a readback, writable bindings
	// (up to MaxWritebackRange) touching it are copied to the guest backing at the end of their
	// submit. A later CPU fault on such a page only waits for that submit, on the faulting
	// thread: no GPU-thread round trip and no flush of the recording command buffer. The pages
	// after it whose write-backs that wait also covered go back with it.
	// NoteWrittenRange runs as a range is bound for writing; CommitWrittenRanges (public) once
	// the command is recorded; RecordWritebacks just before each submit.
	void               NoteWrittenRange(uint64_t vaddr, uint64_t size);
	void               RecordWritebacks();
	void               ApplyResolvedWritebacks();
	[[nodiscard]] bool TryResolveWrittenBack(uint64_t vaddr, bool is_write);
	// Start of the first 16 KiB page at or after vaddr's with a pending guest write, or
	// UINT64_MAX.
	[[nodiscard]] uint64_t FirstPendingGuestWrite(uint64_t vaddr) const;
	// The latest submit tick that records a pending guest write to a 16 KiB page of the range, or
	// 0. Publishes complete in tick order: once it and its priority operations are done, so are
	// the range's pending writes noted before.
	[[nodiscard]] uint64_t PendingWriteTick(uint64_t vaddr, uint64_t size) const;
	// Moves the copies' pending guest writes, noted before RecordDownload, to the current tick
	// (a staging wrap in between submitted the tick they were noted at).
	void                   RetickPendingGuestWrites(uint64_t buffer_address,
	                                                std::span<const vk::BufferCopy> copies);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	// Pending deferred guest writes per 16 KiB page: how many, and the latest tick recording one.
	struct PendingWrites {
		uint32_t count = 0;
		uint64_t tick  = 0;
	};
	mutable std::mutex                                m_pending_writes_mutex;
	std::map<uint64_t, PendingWrites>                 m_pending_writes;
	FaultManager                                      m_fault_manager;
	std::function<bool(uint64_t, uint64_t, MappedParts&)> m_mapped_parts;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Buffer                                            m_bda_zero_page;
	bool                                              m_bda_zero_entry_written = false;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	// Written on the GPU thread under m_dirty_ranges_mutex; other threads read them under it.
	mutable std::shared_mutex                         m_dirty_ranges_mutex;
	RangeSet                                          m_gpu_modified_ranges;
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	// Bytes of the registered buffers (m_total_used_memory becomes the device's usage).
	uint64_t                                          m_buffer_bytes       = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	// GPU thread only. Pages that cost a readback, with the submit tick at which a CPU access
	// last used them (the readback, or a fault resolved by a write-back). Unused ones decay
	// (RecordWritebacks).
	std::map<uint64_t, uint64_t>                      m_writeback_hot_pages;
	uint64_t                                          m_writeback_sweep_tick = 0;
	struct RangeHash {
		size_t operator()(const std::pair<uint64_t, uint64_t>& range) const noexcept {
			return std::hash<uint64_t> {}(range.first * 0x9e3779b97f4a7c15ull ^ range.second);
		}
	};
	// Writable ranges bound (NoteWrittenRange) whose command is not recorded yet: a submit
	// between (a stream-buffer wrap) publishes no write-back for their pages.
	std::vector<std::pair<uint64_t, uint64_t>>        m_written_pending;
	// Writable ranges CommitWrittenRanges already added to m_writeback_wanted this submit.
	std::unordered_set<std::pair<uint64_t, uint64_t>, RangeHash> m_writeback_noted;
	uint64_t                                          m_writeback_noted_tick = 0;
	RangeSet                                          m_writeback_wanted;
	// Pages whose GPU-dirty bytes all reach the backing when the tick completes, and pages fault
	// threads handed back whose GPU-dirty ranges the GPU thread has yet to drop.
	std::mutex                                        m_writeback_mutex;
	std::map<uint64_t, uint64_t>                      m_writeback_pages;
	// m_writeback_pages.size(), stored under the mutex once its holder's tracker updates are
	// done, so the common case of no pending write-back skips the mutex.
	std::atomic<size_t>                               m_writeback_page_count {0};
	// Handed-back runs of pages: the first is the one a CPU access used.
	struct ResolvedRun {
		uint64_t page;
		uint64_t pages;
	};
	std::vector<ResolvedRun>                          m_writeback_resolved;
	std::atomic<bool>                                 m_writeback_resolved_any {false};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
