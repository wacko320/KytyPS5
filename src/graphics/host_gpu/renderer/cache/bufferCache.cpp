#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/srtPageWatch.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/frameProfiler.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace FrameProfiler = Common::FrameProfiler;
using FrameProfiler::WaitSite;

namespace {

// A guest-thread readback for the frame profiler: its span, time per step and outcome by thread
// role. Inert when the profiler is off.
struct ProfiledDownload {
	ProfiledDownload(uint64_t vaddr, uint64_t size): t0(FrameProfiler::Start()) {
		FrameProfiler::RecIf(t0, FrameProfiler::Event::GuestReadbackBegin, vaddr, size);
		FrameProfiler::RecIf(t0, FrameProfiler::Event::ReadbackContext,
		                     FrameProfiler::ContextCode(), FrameProfiler::CurrentShader());
	}
	~ProfiledDownload() {
		if (t0 == 0) {
			return;
		}
		using FrameProfiler::Counter;
		FrameProfiler::RecordEvent(FrameProfiler::Event::GuestReadbackEnd, 0, 0);
		const auto role = FrameProfiler::CurrentRole();
		FrameProfiler::Add(Counter::DownloadCallsRender, role);
		FrameProfiler::Add(Counter::DownloadTimeRender, role, FrameProfiler::Ticks() - t0);
		FrameProfiler::Add(outcome, role);
		FrameProfiler::Add(Counter::DownloadQueueTimeRender, role, queued != 0 ? queued - t0 : 0);
		FrameProfiler::Add(Counter::DownloadFirstCommandTimeRender, role,
		                   first_done != 0 ? first_done - t0 : 0);
		FrameProfiler::Add(Counter::DownloadWaitTimeRender, role, wait);
		FrameProfiler::Add(Counter::DownloadSecondCommandTimeRender, role,
		                   second_end != 0 ? second_end - second_begin : 0);
	}
	ProfiledDownload(const ProfiledDownload&)            = delete;
	ProfiledDownload& operator=(const ProfiledDownload&) = delete;

	[[nodiscard]] uint64_t Now() const { return t0 != 0 ? FrameProfiler::Ticks() : 0; }

	const uint64_t         t0;
	FrameProfiler::Counter outcome      = FrameProfiler::Counter::DownloadNoCopyRender;
	uint64_t               queued       = 0;
	uint64_t               first_done   = 0;
	uint64_t               wait         = 0;
	uint64_t               second_begin = 0;
	uint64_t               second_end   = 0;
};

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

// Host mappings of device memory are write-combined (uncached for reads): a plain memcpy is about
// 40 times slower than streaming loads of whole lines. Both pointers and the size are multiples
// of DeviceReadAlignment.
constexpr uint64_t DeviceReadAlignment = 32;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define KYTY_DEVICE_READ_TARGET(features) __attribute__((target(features)))
#else
#define KYTY_DEVICE_READ_TARGET(features)
#endif

KYTY_DEVICE_READ_TARGET("avx2")
void ReadDeviceMemoryAvx2(uint8_t* destination, const uint8_t* source, uint64_t size) {
	auto*       to   = reinterpret_cast<__m256i*>(destination);
	const auto* from = reinterpret_cast<const __m256i*>(source);
	for (uint64_t i = 0; i < size / sizeof(__m256i); i++) {
		_mm256_store_si256(to + i, _mm256_stream_load_si256(from + i));
	}
}

KYTY_DEVICE_READ_TARGET("sse4.1")
void ReadDeviceMemorySse41(uint8_t* destination, const uint8_t* source, uint64_t size) {
	auto*       to   = reinterpret_cast<__m128i*>(destination);
	const auto* from = reinterpret_cast<const __m128i*>(source);
	for (uint64_t i = 0; i < size / sizeof(__m128i); i++) {
		_mm_store_si128(to + i, _mm_stream_load_si128(const_cast<__m128i*>(from + i)));
	}
}

#undef KYTY_DEVICE_READ_TARGET
#endif

// Streaming loads are not ordered with the earlier load that saw the writer complete: one fence
// before the reads that depend on it.
void FenceDeviceReads() {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_mfence();
#else
	std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

void ReadDeviceMemory(uint8_t* destination, const uint8_t* source, uint64_t size) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
	static const bool avx2  = __builtin_cpu_supports("avx2");
	static const bool sse41 = __builtin_cpu_supports("sse4.1");
	if (avx2) {
		ReadDeviceMemoryAvx2(destination, source, size);
		return;
	}
	if (sse41) {
		ReadDeviceMemorySse41(destination, source, size);
		return;
	}
#endif
#endif
	std::memcpy(destination, source, size);
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		m_buffer_bytes += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		m_memory_tracker.QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_buffer_bytes -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::EnsureBdaZeroEntry() {
	if (m_bda_zero_entry_written) {
		return;
	}
	m_bda_zero_entry_written = true;
	m_bda_zero_page.Fill(0, m_bda_zero_page.Size(), 0);
	const auto address = m_bda_zero_page.BufferDeviceAddress();
	WriteDataBuffer(m_bda_pagetable_buffer, BDA_ZERO_ENTRY * sizeof(vk::DeviceAddress), &address,
	                sizeof(address));
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	// A fault thread handing written-back pages back (TryResolveWrittenBack) either unmarks them
	// before the scan, which then skips them, or after it. Only this thread adds write-back
	// pages: with none, there is no hand-back to order against.
	std::unique_lock writeback_lock(m_writeback_mutex, std::defer_lock);
	if (m_writeback_page_count.load(std::memory_order_acquire) != 0) {
		writeback_lock.lock();
	}
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    const auto first = copies.size();
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    // Pending before the bytes stop being GPU-dirty: a concurrent clean read
		    // (IsCleanForConcurrentRead) sees one or the other until the publish lands.
		    for (auto index = first; index < copies.size(); index++) {
			    NotePendingGuestWrite(buffer_address + copies[index].srcOffset, copies[index].size);
		    }
		    std::unique_lock lock(m_dirty_ranges_mutex);
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	// Not held while recording: a staging wrap there can submit, and the submit records
	// write-backs.
	if (writeback_lock.owns_lock()) {
		writeback_lock.unlock();
	}
	RecordDownload(buffer, std::move(copies), total_size, async);
	return true;
}

void BufferCache::RecordDownload(Buffer& buffer, std::vector<vk::BufferCopy>&& copies,
                                 uint64_t total_size, bool async) {
	// Here, not in DownloadBufferMemory: the end-of-submit write-backs record downloads too.
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	const auto buffer_address = buffer.CpuAddress();
	const auto noted_tick     = m_scheduler.CurrentTick();
	auto [mapped, offset]     = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	if (m_scheduler.CurrentTick() != noted_tick) {
		RetickPendingGuestWrites(buffer_address, copies);
	}
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			// The guest may have unmapped part of the range since the download was recorded.
			(void)Libs::LibKernel::Memory::WriteMappedBacking(
			    buffer_address + copy.srcOffset, mapped + (copy.dstOffset - offset), copy.size);
			DonePendingGuestWrite(buffer_address + copy.srcOffset, copy.size);
		}
	};
	if (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick, WaitSite::ReadbackGpuThread);
		m_scheduler.WaitPriorityOperations(tick, WaitSite::ReadbackGpuThread);
		publish();
	}
}

namespace {

constexpr uint64_t WritebackPageBits = 12;
constexpr uint64_t WritebackPageSize = uint64_t {1} << WritebackPageBits;
static_assert(WritebackPageSize == TRACKER_PAGE_SIZE);
// Larger bindings (often a whole buffer, mostly not written) copy back only their hot pages.
constexpr uint64_t MaxWritebackRange = 1024 * 1024;
// Per submit; what does not fit is left for a fault to download.
constexpr uint64_t MaxWritebackBytes = 16 * MiB;
// Pages after a fault page a resolved write-back hands back with it (256 KiB beyond the fault
// page).
constexpr uint64_t MaxResolveRunPages = 64;

} // namespace

void BufferCache::NoteWrittenRange(uint64_t vaddr, uint64_t size) {
	const auto first = vaddr >> WritebackPageBits;
	const auto last  = (vaddr + size - 1) >> WritebackPageBits;
	// Only this thread adds pages, and a fault thread publishes the count only after its tracker
	// update and its hand-back: with none left there is nothing to erase.
	if (m_writeback_page_count.load(std::memory_order_acquire) != 0) {
		// Before the tracker marks the pages GPU-dirty again: a fault thread handing a page back
		// holds the mutex until its tracker update is done.
		std::lock_guard lock(m_writeback_mutex);
		m_writeback_pages.erase(m_writeback_pages.lower_bound(first),
		                        m_writeback_pages.upper_bound(last));
		m_writeback_page_count.store(m_writeback_pages.size(), std::memory_order_release);
	}
	// After the erase: a hand-back of these pages queued before it drops their GPU-dirty ranges
	// now, not after the caller adds the new write's, and none can follow it.
	ApplyResolvedWritebacks();
	// Wanted only once the write is recorded: a stream-buffer wrap before that submits, and a
	// write-back recorded then would copy the bytes from before the write.
	m_written_pending.emplace_back(vaddr, size);
}

void BufferCache::CommitWrittenRanges() {
	if (m_written_pending.empty()) {
		return;
	}
	// Recorded now, in the command buffer open at this tick (a wrap since the binding submitted
	// the one it was noted in).
	for (const auto [vaddr, size]: m_written_pending) {
		NoteOpenWrite(vaddr, size);
		NoteUnchangedWrite(vaddr, size, true);
	}
	if (m_writeback_hot_pages.empty()) {
		m_written_pending.clear();
		return;
	}
	// The same binding recurs draw after draw: its range stays wanted until this submit's
	// write-backs are recorded.
	if (const auto tick = m_scheduler.CurrentTick(); tick != m_writeback_noted_tick) {
		m_writeback_noted.clear();
		m_writeback_noted_tick = tick;
	}
	for (const auto [vaddr, size]: m_written_pending) {
		const auto first     = vaddr >> WritebackPageBits;
		const auto last      = (vaddr + size - 1) >> WritebackPageBits;
		const auto hot_begin = m_writeback_hot_pages.lower_bound(first);
		if (hot_begin == m_writeback_hot_pages.end() || hot_begin->first > last) {
			continue;
		}
		if (!m_writeback_noted.insert({vaddr, size}).second) {
			continue;
		}
		if (size <= MaxWritebackRange) {
			m_writeback_wanted.Add(vaddr, size);
			continue;
		}
		// One range per run of consecutive hot pages.
		const auto hot_end = m_writeback_hot_pages.upper_bound(last);
		const auto end     = vaddr + size;
		for (auto hot = hot_begin; hot != hot_end;) {
			const auto run_first = hot->first;
			auto       run_last  = run_first;
			for (++hot; hot != hot_end && hot->first == run_last + 1; ++hot) {
				run_last = hot->first;
			}
			const auto run_begin = std::max(vaddr, run_first << WritebackPageBits);
			const auto run_end   = std::min(end, (run_last + 1) << WritebackPageBits);
			m_writeback_wanted.Add(run_begin, run_end - run_begin);
		}
	}
	m_written_pending.clear();
}

void BufferCache::RecordWritebacks() {
	ApplyResolvedWritebacks();
	// A page stays hot while CPU accesses use its write-backs. Pages hot from a one-off readback
	// otherwise accumulate, and every writable binding over them is copied back each submit. A
	// page dropped too early costs one readback and turns hot again.
	constexpr uint64_t HotPageSweepInterval = 64;
	constexpr uint64_t HotPageLifetime      = 256;
	if (const auto tick = m_scheduler.CurrentTick();
	    tick >= m_writeback_sweep_tick + HotPageSweepInterval) {
		m_writeback_sweep_tick = tick;
		std::erase_if(m_writeback_hot_pages, [&](const auto& hot) {
			return hot.second + HotPageLifetime < tick;
		});
	}
	if (m_writeback_wanted.Empty()) {
		return;
	}
	FrameProfiler::Add(FrameProfiler::Counter::WritebackRecords);
	std::vector<std::pair<uint64_t, uint64_t>> wanted;
	m_writeback_wanted.ForEach([&](uint64_t begin, uint64_t end) { wanted.emplace_back(begin, end); });
	m_writeback_wanted.Clear();
	RangeSet copied;
	uint64_t budget = MaxWritebackBytes;
	for (auto [address, end]: wanted) {
		while (address < end && budget != 0) {
			const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
			if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
				address = Common::AlignDown(address, CACHING_PAGESIZE) + CACHING_PAGESIZE;
				continue;
			}
			auto&      buffer = m_slot_buffers[*owner];
			const auto stop   = std::min(end, buffer.CpuAddress() + buffer.Size());
			std::vector<vk::BufferCopy> copies;
			uint64_t                    total_size = 0;
			m_gpu_modified_ranges.ForEachInRange(
			    address, stop - address, [&](uint64_t begin, uint64_t last) {
				    const auto bytes = std::min(last - begin, budget);
				    if (bytes == 0) {
					    return;
				    }
				    copies.emplace_back(buffer.Offset(begin), total_size, bytes);
				    total_size += Common::AlignUp(bytes, 64);
				    budget -= bytes;
				    copied.Add(begin, bytes);
			    });
			if (!copies.empty()) {
				for (const auto& copy: copies) {
					NotePendingGuestWrite(buffer.CpuAddress() + copy.srcOffset, copy.size);
				}
				RecordDownload(buffer, std::move(copies), total_size, true);
			}
			address = stop;
		}
	}
	// A page bound for a write whose command is not recorded yet (this submit is a wrap within
	// it) gets no entry: the copy holds the bytes from before that write. The submit holding the
	// write publishes it.
	const auto pending_write = [&](uint64_t page) {
		const auto page_begin = page << WritebackPageBits;
		const auto page_end   = page_begin + WritebackPageSize;
		return std::any_of(m_written_pending.begin(), m_written_pending.end(), [&](const auto& range) {
			return range.first < page_end && page_begin < range.first + range.second;
		});
	};
	// A copy recorded before a stream-buffer wrap went in an earlier submit; this one is last.
	const auto      tick = m_scheduler.CurrentTick();
	std::lock_guard lock(m_writeback_mutex);
	copied.ForEach([&](uint64_t begin, uint64_t end) {
		for (auto page = begin >> WritebackPageBits; page <= (end - 1) >> WritebackPageBits;
		     page++) {
			if (pending_write(page)) {
				continue;
			}
			bool complete = true;
			m_gpu_modified_ranges.ForEachInRange(
			    page << WritebackPageBits, WritebackPageSize,
			    [&](uint64_t start, uint64_t last) { complete &= copied.Contains(start, last - start); });
			if (complete) {
				if (FrameProfiler::Active()) {
					FrameProfiler::Add(FrameProfiler::Counter::WritebackPages);
					if (!m_writeback_pages.contains(page)) {
						FrameProfiler::Add(FrameProfiler::Counter::WritebackPagesNew);
					}
				}
				m_writeback_pages[page] = tick;
			}
		}
	});
	m_writeback_page_count.store(m_writeback_pages.size(), std::memory_order_release);
	if (FrameProfiler::Active()) {
		copied.ForEach([](uint64_t begin, uint64_t end) {
			FrameProfiler::Add(FrameProfiler::Counter::WritebackBytes, end - begin);
			FrameProfiler::Add(FrameProfiler::Counter::WritebackRanges);
		});
	}
}

void BufferCache::ApplyResolvedWritebacks() {
	if (!m_writeback_resolved_any.load(std::memory_order_acquire)) {
		return;
	}
	std::vector<ResolvedRun> runs;
	{
		std::lock_guard lock(m_writeback_mutex);
		runs.swap(m_writeback_resolved);
		m_writeback_resolved_any.store(false, std::memory_order_relaxed);
	}
	const auto       tick = m_scheduler.CurrentTick();
	std::unique_lock lock(m_dirty_ranges_mutex);
	for (const auto [page, pages]: runs) {
		m_gpu_modified_ranges.Subtract(page << WritebackPageBits, pages * WritebackPageSize);
		// A CPU access used the run's write-backs: its pages stay hot. The pages after the first
		// no longer fault, so this is their only refresh. A page that is not hot stays so.
		for (auto hot = m_writeback_hot_pages.lower_bound(page);
		     hot != m_writeback_hot_pages.end() && hot->first < page + pages; ++hot) {
			hot->second = tick;
		}
	}
}

bool BufferCache::TryResolveWrittenBack(uint64_t vaddr, bool is_write) {
	const auto page = vaddr >> WritebackPageBits;
	uint64_t   tick = 0;
	if (m_writeback_page_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	const uint64_t profile_t0 = FrameProfiler::Start();
	// Outcome and time by thread role, for the frame profiler.
	const auto profile = [profile_t0](FrameProfiler::Counter count, FrameProfiler::Counter time) {
		if (profile_t0 != 0) {
			const auto role = FrameProfiler::CurrentRole();
			FrameProfiler::Add(count, role);
			FrameProfiler::Add(time, role, FrameProfiler::Ticks() - profile_t0);
		}
	};
	{
		std::lock_guard lock(m_writeback_mutex);
		const auto      entry = m_writeback_pages.find(page);
		if (entry == m_writeback_pages.end()) {
			return false;
		}
		tick = entry->second;
	}
	if (!m_scheduler.IsSubmitted(tick)) {
		if (profile_t0 != 0) {
			FrameProfiler::Add(FrameProfiler::Counter::WritebackMissUnsubmittedRender,
			                   FrameProfiler::CurrentRole());
		}
		return false;
	}
	m_scheduler.GetMasterSemaphore().Wait(tick, WaitSite::WrittenBackPage);
	m_scheduler.WaitPriorityOperations(tick, WaitSite::WrittenBackPage);
	std::lock_guard lock(m_writeback_mutex);
	const auto      entry = m_writeback_pages.find(page);
	if (entry == m_writeback_pages.end() || entry->second != tick) {
		profile(FrameProfiler::Counter::WritebackMissChangedRender,
		        FrameProfiler::Counter::WritebackMissChangedTimeRender);
		return false;
	}
	// A publish recorded after the write-back (a download) lands later: the normal path waits
	// for it. A download notes its publishes under m_writeback_mutex, so none is in between.
	// Lock order: m_writeback_mutex, then m_pending_writes_mutex.
	const auto pending = FirstPendingGuestWrite(page << WritebackPageBits);
	if (pending < (page + 1) << WritebackPageBits) {
		return false;
	}
	// The pages after it whose write-backs the wait covered (their priority operations are
	// drained too) hold current guest bytes: they go back with it, one protection change
	// instead of a fault each. A write recorded later erases a page's entry under the mutex
	// before marking the page again.
	auto     run_end = std::next(entry);
	uint64_t run     = 0;
	while (run < MaxResolveRunPages && run_end != m_writeback_pages.end() &&
	       run_end->first == page + 1 + run && run_end->second <= tick) {
		const auto address = run_end->first << WritebackPageBits;
		if (address + WritebackPageSize > pending ||
		    !m_memory_tracker.IsRegionGpuModified(address, WritebackPageSize)) {
			break;
		}
		++run_end;
		++run;
	}
	m_writeback_pages.erase(entry, run_end);
	m_memory_tracker.UnmarkRegionAsGpuModified(page << WritebackPageBits,
	                                           (run + 1) * WritebackPageSize);
	if (is_write) {
		MarkCpuWritten(vaddr, 1);
	}
	// The GPU thread drops the pages' GPU-dirty ranges before it next records a write. Queued
	// before the count is published: a writer that sees the new count sees the hand-back too.
	m_writeback_resolved.push_back({page, run + 1});
	m_writeback_resolved_any.store(true, std::memory_order_release);
	m_writeback_page_count.store(m_writeback_pages.size(), std::memory_order_release);
	profile(FrameProfiler::Counter::WritebackHitsRender,
	        FrameProfiler::Counter::WritebackHitTimeRender);
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      // A read of several dwords may run past the end of the page.
      m_bda_zero_page(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
                      AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                      CACHING_PAGESIZE + 256u),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_scheduler.SetPreSubmitHook([this] { RecordWritebacks(); });
	m_host_readable_buffers = Buffer::HostReadableSupported(graphics);
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	m_scheduler.SetPreSubmitHook({});
	ApplyResolvedWritebacks();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	// Pending before the pages turn writable, and held until they have: an acquire in between
	// would otherwise miss a write made through them (noted after), or find them still clean
	// and drop the range (noted before, unheld). A flush marks its pages in MarkCpuWritten.
	m_memory_tracker.InvalidateRegion(
	    vaddr, size, [this, vaddr, size] { ReadMemory(vaddr, size, true); },
	    [this](uint64_t address, uint64_t bytes) { return HoldCpuWritten(address, bytes); });
}

MemoryTracker::AheadCandidates BufferCache::InvalidateFault(uint64_t vaddr) {
	if (!GuestRange {vaddr, 1}.Valid()) {
		EXIT("BufferCache: invalid write-fault address\n");
	}
	// As InvalidateMemory.
	return m_memory_tracker.InvalidateFault(
	    vaddr, m_epoch.Generation() >> WRITE_HISTORY_SHIFT,
	    [this, vaddr] {
		    const FrameProfiler::ScopedCause cause(FrameProfiler::ReadbackCause::FaultWrite);
		    ReadMemory(vaddr, 1, true);
	    },
	    [this](uint64_t address, uint64_t bytes) { return HoldCpuWritten(address, bytes); });
}

uint64_t BufferCache::InvalidateAhead(uint64_t vaddr, uint64_t size, AheadDirection direction) {
	return m_memory_tracker.InvalidateAhead(
	    vaddr, size, direction, m_epoch.Generation() >> WRITE_HISTORY_SHIFT,
	    [this](uint64_t address, uint64_t bytes) { return HoldCpuWritten(address, bytes); });
}

CoherenceEpoch::Hold BufferCache::HoldCpuWritten(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	return m_epoch.HoldPending(begin, end - begin);
}

void BufferCache::MarkCpuWritten(uint64_t vaddr, uint64_t size) {
	const auto hold = HoldCpuWritten(vaddr, size);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	// Labels the readback by its context and, on the GPU thread, by what the thread was doing.
	struct ProfiledReadMemory {
		explicit ProfiledReadMemory(uint64_t vaddr, uint64_t size)
		    : t0(FrameProfiler::Start()), address(vaddr) {
			if (t0 == 0) {
				return;
			}
			waited.push_back({Common::AlignDown(vaddr, CACHING_PAGESIZE),
			                  Common::AlignUp(vaddr + size, CACHING_PAGESIZE)});
			FrameProfiler::RecIf(t0, FrameProfiler::Event::GpuReadbackBegin, vaddr, size);
			FrameProfiler::RecIf(t0, FrameProfiler::Event::ReadbackContext,
			                     FrameProfiler::ContextCode(),
			                     FrameProfiler::t_materialize_shader != 0
			                         ? FrameProfiler::t_materialize_shader
			                         : FrameProfiler::CurrentShader());
		}
		~ProfiledReadMemory() {
			if (t0 == 0) {
				return;
			}
			using FrameProfiler::Counter;
			FrameProfiler::RecordEvent(FrameProfiler::Event::GpuReadbackEnd, 0, 0);
			const auto role  = FrameProfiler::CurrentRole();
			const auto ticks = FrameProfiler::Ticks() - t0;
			FrameProfiler::Add(Counter::ReadMemoryCallsRender, role);
			FrameProfiler::Add(Counter::ReadMemoryTimeRender, role, ticks);
			if (color_metadata) {
				FrameProfiler::Add(Counter::ColorMetadataReadbacksRender, role);
				FrameProfiler::Add(Counter::ColorMetadataReadbackTimeRender, role, ticks);
			}
			if (role == FrameProfiler::Role::Gpu) {
				const auto phase = FrameProfiler::t_phase;
				const auto where =
				    phase == static_cast<uint8_t>(FrameProfiler::Phase::Materialize)
				        ? Counter::GpuReadbackMaterializeTime
				    : phase != FrameProfiler::NoPhase ? Counter::GpuReadbackBindingTime
				    : FrameProfiler::t_packet != FrameProfiler::NoPacket
				        ? Counter::GpuReadbackPacketTime
				        : Counter::GpuReadbackCommandTime;
				FrameProfiler::Add(where, ticks);
			}
			if (direct != Counter::Count) {
				FrameProfiler::Add(direct);
				if (direct == Counter::ReadbackDirect) {
					FrameProfiler::Add(Counter::ReadbackDirectTime, ticks);
					FrameProfiler::Add(Counter::ReadbackDirectBytes, direct_bytes);
				} else {
					// The whole download's time (unmaps and invalidations included, as Size): what
					// was not eligible, not time lost to the check.
					FrameProfiler::Add(Counter::ReadbackDirectBlockedTime, ticks);
				}
			}
			FrameProfiler::NoteReadback(address, waited, ticks);
		}
		void NoteDirect(DirectReadback outcome, uint64_t bytes) {
			using FrameProfiler::Counter;
			switch (outcome) {
				case DirectReadback::Copied: direct = Counter::ReadbackDirect; break;
				case DirectReadback::Unmapped: direct = Counter::ReadbackDirectBlockedUnmapped; break;
				case DirectReadback::Writer: direct = Counter::ReadbackDirectBlockedWriter; break;
				case DirectReadback::Untracked:
					direct = Counter::ReadbackDirectBlockedUntracked;
					break;
				case DirectReadback::Pending: direct = Counter::ReadbackDirectBlockedPending; break;
				case DirectReadback::Size: direct = Counter::ReadbackDirectBlockedSize; break;
			}
			direct_bytes = bytes;
		}
		ProfiledReadMemory(const ProfiledReadMemory&)            = delete;
		ProfiledReadMemory& operator=(const ProfiledReadMemory&) = delete;

		const uint64_t                           t0;
		const uint64_t                           address;
		std::vector<FrameProfiler::AddressRange> waited;
		// The direct readback's outcome (TryReadDirect), when it got that far.
		FrameProfiler::Counter                   direct       = FrameProfiler::Counter::Count;
		uint64_t                                 direct_bytes = 0;
		bool                                     color_metadata = false;
	} profile(vaddr, size);
	// The texture cache watches the DCC/CMask metadata its GPU fast-clear checks recorded
	// (TextureCache::MaterializeColorClear): a readback of it by anyone else is a reader the
	// check left the metadata GPU-dirty for. Metadata never checked on the GPU is not counted.
	profile.color_metadata = profile.t0 != 0 &&
	                         FrameProfiler::t_readback_cause != FrameProfiler::ReadbackCause::Metadata &&
	                         OverlapsWatchedWrites(vaddr, size);
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// A written-back page needs only the wait for its write-back's submit, on this thread. Only
	// for a request inside that page: the resolve hands back the page (and a run after it), not
	// the rest of a longer range, such as an unmap's.
	if (vaddr + size <= Common::AlignDown(vaddr, WritebackPageSize) + WritebackPageSize &&
	    TryResolveWrittenBack(vaddr, is_write)) {
		return;
	}
	// A guest thread waits for its readback itself; the GPU thread's own reads (resource
	// evaluation) wait in place.
	if (!GuestGpu::IsGpuThread()) {
		ReadMemoryFromGuestThread(vaddr, size, is_write);
		return;
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, &profile] {
		ApplyResolvedWritebacks();
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		// Tracker pages are whole 4 KiB pages; GPU writes are byte ranges. A page whose GPU-dirty
		// bytes lie only in its neighbours (a writable binding ending mid-page) holds no GPU data:
		// hand it back without draining the GPU for the neighbours.
		// Only for a request inside the one page: an unmap reads back its whole range.
		const auto fault_page = Common::AlignDown(vaddr, CACHING_PAGESIZE);
		if (vaddr + size <= fault_page + CACHING_PAGESIZE &&
		    !HasGpuDirtyBytes(fault_page, CACHING_PAGESIZE)) {
			// A guest thread's download took the page's GPU-dirty bytes and has yet to publish
			// them.
			if (const auto pending = PendingWriteTick(fault_page, CACHING_PAGESIZE); pending != 0) {
				m_scheduler.Wait(pending, WaitSite::ReadbackPendingPublish);
				m_scheduler.WaitPriorityOperations(pending, WaitSite::ReadbackPendingPublish);
			}
			m_memory_tracker.UnmarkRegionAsGpuModified(fault_page, CACHING_PAGESIZE);
		} else {
			uint64_t direct_bytes  = 0;
			uint64_t blocking_tick = 0;
			auto     direct = TryReadDirect(buffer, vaddr, size, direct_bytes, &blocking_tick);
			if (blocking_tick != 0) {
				// Only the writer (or the untracked stores) blocks the direct read: wait for that
				// tick alone (submitting the open command buffer when the writer is in it), not
				// for a download behind everything recorded since.
				profile.NoteDirect(direct, 0);
				m_scheduler.Wait(blocking_tick, WaitSite::ReadbackGpuThread);
				direct = TryReadDirect(m_slot_buffers[FindBuffer(vaddr, size)], vaddr, size,
				                       direct_bytes);
			}
			profile.NoteDirect(direct, direct_bytes);
			if (direct != DirectReadback::Copied) {
				// Only pages a direct read could not take cost a readback: from now on, writable
				// bindings over them are copied back at the end of their submit.
				m_writeback_hot_pages[vaddr >> WritebackPageBits] = m_scheduler.CurrentTick();
				bool ready = DownloadBufferMemory<false>(buffer, vaddr, size);
				// Pages whose GPU-dirty bytes a guest thread's download took wait for its
				// publish.
				if (const auto pending = ready ? 0 : PendingWriteTick(vaddr, size); pending != 0) {
					m_scheduler.Wait(pending, WaitSite::ReadbackPendingPublish);
					m_scheduler.WaitPriorityOperations(pending, WaitSite::ReadbackPendingPublish);
					ready = true;
				}
				if (ready) {
					m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
				}
			}
		}
		if (is_write) {
			MarkCpuWritten(vaddr, size);
		}
	});
}

BufferCache::DirectReadback BufferCache::TryReadDirect(Buffer& buffer, uint64_t vaddr,
                                                      uint64_t size, uint64_t& copied,
                                                      uint64_t* blocking_tick) {
	// Uncached reads under load cost about 2.5 us per KiB (a GPU copy round trip about 60 us).
	// Larger requests keep the download (an unmap reads back its whole range).
	if (size > TRACKER_PAGE_SIZE) {
		return DirectReadback::Size;
	}
	const auto mapped = buffer.Mapped();
	if (mapped.empty() || reinterpret_cast<uintptr_t>(mapped.data()) % DeviceReadAlignment != 0 ||
	    buffer.Size() % DeviceReadAlignment != 0) {
		return DirectReadback::Unmapped;
	}
	const auto buffer_begin = buffer.CpuAddress();
	const auto begin = std::max(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), buffer_begin);
	const auto end   = std::min(Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE),
	                            buffer_begin + buffer.Size());
	// Stores through GPU-selected V#s and addresses may have written any byte.
	if (!m_scheduler.IsFree(m_untracked_write_tick)) {
		if (blocking_tick != nullptr) {
			*blocking_tick = m_untracked_write_tick;
		}
		return DirectReadback::Untracked;
	}
	// A download or write-back has yet to publish into the pages.
	if (HasPendingGuestWrite(begin, end - begin)) {
		return DirectReadback::Pending;
	}
	// As for a download (DownloadBufferMemory): a fault thread handing written-back pages back
	// either did before the scan, which then skips them, or waits until the pages are back.
	std::unique_lock writeback_lock(m_writeback_mutex, std::defer_lock);
	if (m_writeback_page_count.load(std::memory_order_acquire) != 0) {
		writeback_lock.lock();
	}
	// The GPU-dirty bytes of the GPU-owned pages, by page run.
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	std::vector<std::pair<uint64_t, uint64_t>> dirty;
	uint64_t                                   bytes = 0;
	m_memory_tracker.ForEachDownloadRange<false>(
	    begin, end - begin, [&](uint64_t address, uint64_t length) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, length,
		                                           "direct readback");
		    pages.emplace_back(address, length);
		    m_gpu_modified_ranges.ForEachInRange(address, length, [&](uint64_t first, uint64_t last) {
			    dirty.emplace_back(first, last);
			    bytes += last - first;
		    });
	    });
	// Each byte's last writer has completed: not the open command buffer, not a command still to
	// be recorded for a binding, and its tick (m_write_ticks) done.
	const auto& open   = OpenWrites();
	uint64_t    writer = 0;
	for (const auto [first, last]: dirty) {
		if (std::any_of(m_written_pending.begin(), m_written_pending.end(), [&](const auto& range) {
			    return range.first < last && first < range.first + range.second;
		    })) {
			return DirectReadback::Writer;
		}
		if (open.Intersects(first, last - first)) {
			if (blocking_tick != nullptr) {
				*blocking_tick = m_scheduler.CurrentTick();
			}
			return DirectReadback::Writer;
		}
		writer = std::max(writer, m_write_ticks.Max(first, last));
	}
	if (!m_scheduler.IsFree(writer)) {
		if (blocking_tick != nullptr) {
			*blocking_tick = writer;
		}
		return DirectReadback::Writer;
	}
	// The completed command buffers ended with a host barrier (NoteOpenWrite), so the mapping
	// holds their bytes.
	FenceDeviceReads();
	for (const auto [first, last]: dirty) {
		const auto from = Common::AlignDown(buffer.Offset(first), DeviceReadAlignment);
		const auto to   = Common::AlignUp(buffer.Offset(last), DeviceReadAlignment);
		if (m_direct_staging.size() < to - from + DeviceReadAlignment) {
			m_direct_staging.resize(to - from + DeviceReadAlignment);
		}
		auto* staging = reinterpret_cast<uint8_t*>(Common::AlignUp(
		    reinterpret_cast<uintptr_t>(m_direct_staging.data()), DeviceReadAlignment));
		ReadDeviceMemory(staging, mapped.data() + from, to - from);
		Libs::LibKernel::Memory::WriteBacking(first, staging + (buffer.Offset(first) - from),
		                                      last - first);
	}
	{
		std::unique_lock lock(m_dirty_ranges_mutex);
		for (const auto [address, length]: pages) {
			m_gpu_modified_ranges.Subtract(address, length);
		}
	}
	// The request's pages now hold no GPU-dirty byte. They are handed back lazily: the GPU
	// writes them again soon, and the GPU thread reads them from the backing store.
	const bool lazy = PageManager::SetLazyRelease(true);
	m_memory_tracker.UnmarkRegionAsGpuModified(begin, end - begin);
	(void)PageManager::SetLazyRelease(lazy);
	m_direct_readbacks.fetch_add(1, std::memory_order_relaxed);
	copied = bytes;
	return DirectReadback::Copied;
}

void BufferCache::MergeOpenWrites() {
	for (; m_open_merged < m_open_writes.size(); m_open_merged++) {
		const auto [vaddr, size] = m_open_writes[m_open_merged];
		m_open_written.Add(vaddr, size);
	}
}

void BufferCache::FollowOpenTick() {
	// The log belongs to a submitted command buffer: its ranges take its tick, and the open
	// command buffer starts its own.
	const auto tick = m_scheduler.CurrentTick();
	if (tick == m_open_tick) {
		return;
	}
	MergeOpenWrites();
	m_open_written.ForEach([&](uint64_t begin, uint64_t end) {
		m_write_ticks.Assign(begin, end, m_open_tick);
	});
	m_open_writes.clear();
	m_open_written.Clear();
	m_open_merged = 0;
	m_open_tick   = tick;
	constexpr uint64_t PruneInterval = 64;
	if (tick >= m_write_ticks_pruned + PruneInterval) {
		PruneWriteTicks();
	}
}

void BufferCache::PruneWriteTicks() {
	// Completed ticks read as no writer: their ranges are dropped, so that the map holds about
	// the writes in flight.
	m_write_ticks_pruned = m_scheduler.CurrentTick();
	m_write_ticks.EraseUpTo(m_scheduler.GetMasterSemaphore().KnownGpuTick());
}

void BufferCache::NoteOpenWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	FollowOpenTick();
	m_open_writes.emplace_back(vaddr, size);
	m_scheduler.RequestHostBarrier();
}

const RangeSet& BufferCache::OpenWrites() {
	FollowOpenTick();
	MergeOpenWrites();
	return m_open_written;
}

void BufferCache::NoteUntrackedWrites() {
	if (!m_unchanged.empty()) {
		NoteUnchangedWrite(m_unchanged_begin, m_unchanged_end - m_unchanged_begin, true);
	}
	m_untracked_write_tick = m_scheduler.CurrentTick();
	m_scheduler.RequestHostBarrier();
}

void BufferCache::ReadMemoryFromGuestThread(uint64_t vaddr, uint64_t size, bool is_write) {
	ProfiledDownload profile(vaddr, size);
	auto&            gpu = m_scheduler.Context().GetGpu();
	// The tracker pages the request covers; the download takes their GPU-dirty bytes.
	const auto range_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto range_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	// A non-fault caller (InvalidateMemory) needs the range handed back on return: go again while
	// a page of it waits for a publish recorded after the one waited for.
	for (;;) {
		uint64_t tick         = 0;
		bool     done         = false;
		bool     again        = false;
		gpu.SendCommandSync([&] {
			if (profile.queued == 0) {
				profile.queued = profile.Now();
			}
			ApplyResolvedWritebacks();
			if (is_write && !IsRegionRegistered(vaddr, size)) {
				profile.outcome = FrameProfiler::Counter::DownloadUnregisteredRender;
				done            = true;
				return;
			}
			auto&      buffer     = m_slot_buffers[FindBuffer(vaddr, size)];
			const auto fault_page = Common::AlignDown(vaddr, CACHING_PAGESIZE);
			if (vaddr + size <= fault_page + CACHING_PAGESIZE &&
			    !HasGpuDirtyBytes(fault_page, CACHING_PAGESIZE)) {
				// Another thread's download took the page's GPU-dirty bytes and has yet to
				// publish them: wait for it below, then hand the page back. Its submit is made
				// here, not left to the GPU thread's next one.
				if (const auto pending = PendingWriteTick(fault_page, CACHING_PAGESIZE);
				    pending != 0) {
					if (pending == m_scheduler.CurrentTick()) {
						m_scheduler.Flush();
					}
					profile.outcome = FrameProfiler::Counter::DownloadPendingPublishRender;
					tick            = pending;
					return;
				}
				m_memory_tracker.UnmarkRegionAsGpuModified(fault_page, CACHING_PAGESIZE);
				if (is_write) {
					MarkCpuWritten(vaddr, size);
				}
				profile.outcome = FrameProfiler::Counter::DownloadFastRender;
				done            = true;
				return;
			}
			m_writeback_hot_pages[vaddr >> WritebackPageBits] = m_scheduler.CurrentTick();
			// Pages whose GPU-dirty bytes another thread's download took wait for its publish too.
			if (DownloadBufferMemory<true>(buffer, vaddr, size)) {
				profile.outcome = FrameProfiler::Counter::DownloadCopiedRender;
				tick            = m_scheduler.CurrentTick();
			} else {
				tick = PendingWriteTick(vaddr, size);
			}
			if (tick != 0 && tick == m_scheduler.CurrentTick()) {
				m_scheduler.Flush();
			}
		});
		if (profile.first_done == 0) {
			profile.first_done = profile.Now();
		}
		if (done) {
			return;
		}
		if (tick != 0) {
			const uint64_t wait_t0 = profile.Now();
			m_scheduler.GetMasterSemaphore().Wait(tick, WaitSite::ReadbackGuestThread);
			m_scheduler.WaitPriorityOperations(tick, WaitSite::ReadbackGuestThread);
			profile.wait += wait_t0 != 0 ? FrameProfiler::Ticks() - wait_t0 : 0;
		}
		profile.second_begin = profile.Now();
		gpu.SendCommandSync([&] {
			// Pages the GPU wrote again after the download stay GPU-dirty, and so do pages a
			// download recorded since has yet to publish: the guest faults again.
			const auto ready = [&](uint64_t page) {
				return !HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE) &&
				       !HasPendingGuestWrite(page, TRACKER_PAGE_SIZE);
			};
			for (auto page = range_begin; page < range_end; page += TRACKER_PAGE_SIZE) {
				if (m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) && ready(page)) {
					m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
				}
			}
			if (is_write && !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
				MarkCpuWritten(vaddr, size);
			}
			for (auto page = range_begin; page < range_end && !again; page += TRACKER_PAGE_SIZE) {
				again = m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) &&
				        !HasGpuDirtyBytes(page, TRACKER_PAGE_SIZE);
			}
		});
		profile.second_end = profile.Now();
		if (!again) {
			return;
		}
	}
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	FrameProfiler::Add(FrameProfiler::Counter::BufferJoins);
	FrameProfiler::Add(FrameProfiler::Counter::BufferJoinedBytes, overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);
	if (FrameProfiler::Active()) {
		FrameProfiler::Add(FrameProfiler::Counter::BufferCreations);
		FrameProfiler::Add(FrameProfiler::Counter::BufferCreatedBytes, overlap.end - overlap.begin);
		FrameProfiler::Max(FrameProfiler::Counter::BufferLargestCreated,
		                   overlap.end - overlap.begin);
	}

	constexpr auto flags = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	const auto     id =
	    m_host_readable_buffers
	        ? m_slot_buffers.insert(Buffer::HostReadable {}, m_graphics, m_scheduler,
	                                overlap.begin, flags, overlap.end - overlap.begin)
	        : m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal,
	                                overlap.begin, flags, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	// The joins copied the old buffers' GPU-dirty bytes.
	NoteOpenWrite(overlap.begin, overlap.end - overlap.begin);
	NoteBufferWrite(overlap.begin, overlap.end - overlap.begin);
	// Reads within an epoch only translate (TranslateRead), and pages never protected do not
	// fault to mark themselves pending: the new range takes what the CPU wrote now, in one batch.
	RangeSet created;
	created.Add(overlap.begin, overlap.end - overlap.begin);
	{
		const FrameProfiler::ScopedUploadSource source(FrameProfiler::UploadSource::Create);
		UploadCpuPending(created);
	}
	return id;
}

template <typename Func>
void BufferCache::ForEachMappedPart(uint64_t start, uint64_t end, Func&& func) {
	MappedParts parts;
	if (!m_mapped_parts || m_mapped_parts(start, end - start, parts)) {
		func(start, end);
		return;
	}
	// The runs are listed first: func may take the mapped set's lock.
	for (const auto& [begin, stop]: parts) {
		func(begin, stop);
	}
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	// Only the parts still mapped: a buffer can outlive an unmap of some of its pages, which the
	// unmap left pending with no backing to copy from; they stay pending until mapped again.
	// Each mapped run is one walk, its regions locked until its staging copy is taken, with its
	// own barriers (a staging wrap may submit in between).
	ForEachMappedPart(vaddr, vaddr + size, [&](uint64_t begin, uint64_t end) {
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total_size = 0;
		vk::Buffer                  source;
		m_memory_tracker.ForEachUploadRange(
		    begin, end - begin, is_written,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    copies.emplace_back(total_size, buffer.Offset(address), bytes);
			    total_size += bytes;
		    },
		    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
		if (!source) {
			return;
		}
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		for (const auto& copy: copies) {
			NoteBufferWrite(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	});
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	if (is_written) {
		NoteWrittenRange(vaddr, size);
		if (FrameProfiler::Active()) {
			FrameProfiler::RecordEvent(FrameProfiler::Event::WriteBind, vaddr, size);
			FrameProfiler::RecordEvent(FrameProfiler::Event::WriteShader,
			                           FrameProfiler::CurrentShader(), is_texel_buffer ? 1u : 0u);
		}
	}
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		// The GPU now owns these bytes: prefetched evaluations that read them replay.
		SrtPageWatch::Guest().Unsettle(vaddr, size);
	}
	// Repeat binds of a written range find it in the set already.
	// Only this thread changes the set, so it may read it unlocked.
	if (is_written && !m_gpu_modified_ranges.Contains(vaddr, size)) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	if (is_written) {
		NoteOpenWrite(vaddr, size);
		NoteBufferWrite(vaddr, size, true);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::TranslateRead(uint64_t vaddr, uint64_t size,
                                                        bool is_texel_buffer, BufferId id) {
	if (m_scheduler.Current().IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer read request requires a recording command buffer\n");
	}
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner != nullptr && *owner && m_slot_buffers[*owner].IsInBounds(vaddr, size)) {
			id = *owner;
			ValidateEpochRead(m_slot_buffers[id], vaddr, size);
		} else {
			// A range no buffer holds whole: the buffer created (or joined) for it takes the
			// CPU's bytes as it is created.
			id = FindBuffer(vaddr, size);
		}
	} else {
		ValidateEpochRead(m_slot_buffers[id], vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// Image data over the range is a separate domain, synchronized per binding as before.
	if (is_texel_buffer) {
		(void)SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::ValidateEpochRead(const Buffer& buffer, uint64_t vaddr, uint64_t size) {
	(void)buffer;
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	// A CPU-dirty page under a translated read is a CPU write no acquire covered: a guest race,
	// or a synchronization channel that is not an acquire point.
	if (!m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		return;
	}
	static std::atomic<uint64_t> stale_reads {0};
	const auto                   count = stale_reads.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count <= 64 || (count & (count - 1)) == 0) {
		LOGF("BufferCache: read of CPU-dirty bytes within an epoch (%" PRIu64 " so far), "
		     "addr=0x%016" PRIx64 " size=0x%" PRIx64 " epoch=%" PRIu64 "\n",
		     count, vaddr, size, m_epoch.EpochId());
	}
#else
	(void)vaddr;
	(void)size;
#endif
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		m_scheduler.Context().GpuThreadGuestWrite(vaddr, size, [&](void* to) {
			auto* destination = static_cast<uint32_t*>(to);
			std::fill(destination, destination + size / sizeof(uint32_t), value);
		});
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	CommitWrittenRanges();
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		m_scheduler.Context().GpuThreadGuestWrite(dst_vaddr, size, [&](void* to) {
			std::memcpy(to, reinterpret_cast<const void*>(src_vaddr), size);
		});
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	CommitWrittenRanges();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

uint64_t BufferCache::WriteGeneration(uint64_t vaddr, uint64_t size) {
	std::unique_lock lock(m_write_watch_mutex);
	const auto [watch, inserted] = m_write_watches.try_emplace({vaddr, vaddr + size}, 0);
	if (inserted) {
		watch->second         = ++m_write_generation;
		m_write_watch_largest = std::max(m_write_watch_largest, size);
		m_write_watches_any.store(true, std::memory_order_relaxed);
	}
	return watch->second;
}

void BufferCache::UnwatchWrites(uint64_t vaddr, uint64_t size) {
	std::unique_lock lock(m_write_watch_mutex);
	const auto       first = m_write_watches.lower_bound({vaddr, 0});
	const auto       last  = m_write_watches.lower_bound({vaddr + size, 0});
	if (first != last) {
		m_write_watches.erase(first, last);
		RecomputeLargestWatch();
	}
}

void BufferCache::UnwatchRange(uint64_t vaddr, uint64_t size) {
	std::unique_lock lock(m_write_watch_mutex);
	if (m_write_watches.erase({vaddr, vaddr + size}) != 0) {
		RecomputeLargestWatch();
	}
}

void BufferCache::RecomputeLargestWatch() {
	m_write_watch_largest = 0;
	for (const auto& [range, generation]: m_write_watches) {
		m_write_watch_largest = std::max(m_write_watch_largest, range.second - range.first);
	}
	m_write_watches_any.store(!m_write_watches.empty(), std::memory_order_relaxed);
}

uint64_t BufferCache::WriteNotes() const {
	std::shared_lock lock(m_write_watch_mutex);
	return m_write_notes;
}

bool BufferCache::OverlapsWatchedWrites(uint64_t vaddr, uint64_t size) const {
	if (!m_write_watches_any.load(std::memory_order_relaxed)) {
		return false;
	}
	std::shared_lock lock(m_write_watch_mutex);
	const auto       end = vaddr + size;
	const auto       from =
	    vaddr > m_write_watch_largest ? vaddr - m_write_watch_largest : uint64_t {0};
	for (auto it = m_write_watches.lower_bound({from, 0});
	     it != m_write_watches.end() && it->first.first < end; ++it) {
		if (it->first.second > vaddr) {
			return true;
		}
	}
	return false;
}

void BufferCache::WatchUnchanged(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	m_unchanged.clear();
	m_unchanged_changes.assign(ranges.size(), {});
	m_unchanged_begin   = UINT64_MAX;
	m_unchanged_end     = 0;
	m_unchanged_largest = 0;
	for (size_t i = 0; i < ranges.size(); i++) {
		const auto [vaddr, size] = ranges[i];
		m_unchanged.push_back({vaddr, vaddr + size, i});
		m_unchanged_begin   = std::min(m_unchanged_begin, vaddr);
		m_unchanged_end     = std::max(m_unchanged_end, vaddr + size);
		m_unchanged_largest = std::max(m_unchanged_largest, size);
	}
	std::sort(m_unchanged.begin(), m_unchanged.end(),
	          [](const auto& a, const auto& b) { return a.begin < b.begin; });
}

BufferCache::RangeChanges BufferCache::UnchangedRangeChanges(size_t index) const {
	EXIT_IF(index >= m_unchanged_changes.size());
	return m_unchanged_changes[index];
}

void BufferCache::NoteUnchangedWrite(uint64_t vaddr, uint64_t size, bool gpu) {
	m_change_serial++;
	const auto end = vaddr + size;
	if (m_unchanged.empty() || vaddr >= m_unchanged_end || end <= m_unchanged_begin) {
		return;
	}
	// Ranges are at most m_unchanged_largest long: none beginning before `from` reaches vaddr.
	const auto from  = vaddr > m_unchanged_largest ? vaddr - m_unchanged_largest : uint64_t {0};
	const auto first =
	    std::lower_bound(m_unchanged.begin(), m_unchanged.end(), from,
	                     [](const auto& range, uint64_t at) { return range.begin < at; });
	for (auto it = first; it != m_unchanged.end() && it->begin < end; ++it) {
		if (it->end <= vaddr) {
			continue;
		}
		auto& changes = m_unchanged_changes[it->index];
		if (!gpu) {
			changes.other_write = true;
		} else if (changes.first_gpu_write == 0) {
			changes.first_gpu_write = m_change_serial;
		}
	}
}

void BufferCache::NoteBufferWrite(uint64_t vaddr, uint64_t size, bool gpu) {
	NoteUnchangedWrite(vaddr, size, gpu);
	if (!m_write_watches_any.load(std::memory_order_relaxed)) {
		return;
	}
	std::unique_lock lock(m_write_watch_mutex);
	const auto       end  = vaddr + size;
	const auto       from = vaddr > m_write_watch_largest ? vaddr - m_write_watch_largest : uint64_t {0};
	bool             moved = false;
	for (auto it = m_write_watches.lower_bound({from, 0});
	     it != m_write_watches.end() && it->first.first < end; ++it) {
		if (it->first.second > vaddr) {
			if (!moved) {
				++m_write_generation;
				++m_write_notes;
				moved = true;
			}
			it->second = m_write_generation;
		}
	}
}

bool BufferCache::TryWriteGpuOrdered(uint64_t vaddr, const void* data, uint64_t size) {
	// The guest's copy of a GPU-dirty page is stale and the CPU cannot read it without a
	// readback, which then returns these bytes too. A page that is not GPU-dirty is read
	// directly, so it must get the bytes in guest memory.
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < vaddr + size;
	     page += TRACKER_PAGE_SIZE) {
		if (!m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
			return false;
		}
	}
	const auto id     = FindBuffer(vaddr, size);
	auto&      buffer = m_slot_buffers[id];
	SrtPageWatch::Guest().Unsettle(vaddr, size);
	TouchBuffer(buffer);
	WriteDataBuffer(buffer, vaddr, data, size);
	NoteBufferWrite(vaddr, size);
	// Write-through: with no older download or write-back left to publish into these pages
	// (both register pending guest writes), guest memory takes the bytes too and they stop being
	// GPU-dirty. Reads of them (often the GPU thread's own resource evaluation, right after) are
	// then served from guest memory, even while the page stays protected for the GPU's other
	// bytes.
	if (!HasPendingGuestWrite(vaddr, size) &&
	    Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		{
			std::unique_lock lock(m_dirty_ranges_mutex);
			m_gpu_modified_ranges.Subtract(vaddr, size);
		}
		// A page left with no GPU-written byte is no longer the GPU's.
		for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < vaddr + size;
		     page += TRACKER_PAGE_SIZE) {
			if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
			}
		}
		return true;
	}
	// The bytes stay GPU-dirty: a CPU read gets them through the usual readback.
	NoteWrittenRange(vaddr, size);
	CommitWrittenRanges();
	if (!m_gpu_modified_ranges.Contains(vaddr, size)) {
		std::unique_lock lock(m_dirty_ranges_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
	}
	return true;
}

uint8_t* BufferCache::WriteThroughView(uint64_t vaddr, uint64_t size) {
	// Larger stores (DMA fills and copies) upload their pages in one batch instead of streaming
	// every byte through the staging ring.
	constexpr uint64_t MaxWriteThroughSize = 64 * 1024;
	if (size == 0 || size > MaxWriteThroughSize || !GuestRange {vaddr, size}.Valid()) {
		return nullptr;
	}
	// A page the GPU owns holds stale guest bytes, and a download still on its way (a pending
	// guest write) would publish over the store.
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size) || HasGpuDirtyBytes(vaddr, size) ||
	    HasPendingGuestWrite(vaddr, size)) {
		return nullptr;
	}
	Libs::LibKernel::Memory::GuestBackingView view {};
	if (!Libs::LibKernel::Memory::GetGuestBackingView(vaddr, size, &view)) {
		return nullptr;
	}
	return view.pointer;
}

void BufferCache::WriteThrough(uint64_t vaddr, const uint8_t* bytes, uint64_t size) {
	FrameProfiler::Add(FrameProfiler::Counter::WriteThroughs);
	// A clean page stays clean, as buffer and guest memory both take the bytes. A CPU-dirty one
	// is pending already: the acquire that uploads it reads these bytes from guest memory too.
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			WriteDataBuffer(buffer, start, bytes + (start - vaddr), finish - start);
		}
	}
	NoteBufferWrite(vaddr, size);
}

bool BufferCache::IsCleanForConcurrentRead(uint64_t vaddr, uint64_t size) const {
	// Dirty bytes first: a download notes its pending write before it drops them.
	{
		std::shared_lock lock(m_dirty_ranges_mutex);
		if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
			return false;
		}
	}
	return !HasPendingGuestWrite(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::NotePendingGuestWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	// A download or write-back will publish into these pages.
	SrtPageWatch::Guest().Unsettle(vaddr, size);
	const auto      tick = m_scheduler.CurrentTick();
	std::lock_guard lock(m_pending_writes_mutex);
	for (auto page = Common::AlignDown(vaddr, CACHING_PAGESIZE); page < vaddr + size;
	     page += CACHING_PAGESIZE) {
		auto& pending = m_pending_writes[page];
		pending.count++;
		pending.tick = std::max(pending.tick, tick);
	}
}

void BufferCache::RetickPendingGuestWrites(uint64_t buffer_address,
                                           std::span<const vk::BufferCopy> copies) {
	const auto      tick = m_scheduler.CurrentTick();
	std::lock_guard lock(m_pending_writes_mutex);
	for (const auto& copy: copies) {
		const auto vaddr = buffer_address + copy.srcOffset;
		for (auto page = Common::AlignDown(vaddr, CACHING_PAGESIZE); page < vaddr + copy.size;
		     page += CACHING_PAGESIZE) {
			const auto found = m_pending_writes.find(page);
			EXIT_IF(found == m_pending_writes.end());
			found->second.tick = std::max(found->second.tick, tick);
		}
	}
}

void BufferCache::DonePendingGuestWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	std::lock_guard lock(m_pending_writes_mutex);
	for (auto page = Common::AlignDown(vaddr, CACHING_PAGESIZE); page < vaddr + size;
	     page += CACHING_PAGESIZE) {
		const auto found = m_pending_writes.find(page);
		EXIT_IF(found == m_pending_writes.end());
		if (--found->second.count == 0) {
			m_pending_writes.erase(found);
		}
	}
}

uint64_t BufferCache::FirstPendingGuestWrite(uint64_t vaddr) const {
	std::lock_guard lock(m_pending_writes_mutex);
	const auto      found = m_pending_writes.lower_bound(Common::AlignDown(vaddr, CACHING_PAGESIZE));
	return found != m_pending_writes.end() ? found->first : UINT64_MAX;
}

uint64_t BufferCache::PendingWriteTick(uint64_t vaddr, uint64_t size) const {
	std::lock_guard lock(m_pending_writes_mutex);
	uint64_t        tick = 0;
	for (auto found = m_pending_writes.lower_bound(Common::AlignDown(vaddr, CACHING_PAGESIZE));
	     found != m_pending_writes.end() && found->first < vaddr + size; ++found) {
		tick = std::max(tick, found->second.tick);
	}
	return tick;
}

bool BufferCache::HasPendingGuestWrite(uint64_t vaddr, uint64_t size) const {
	std::lock_guard lock(m_pending_writes_mutex);
	const auto      found = m_pending_writes.lower_bound(Common::AlignDown(vaddr, CACHING_PAGESIZE));
	return found != m_pending_writes.end() && found->first < vaddr + size;
}

bool BufferCache::MayBeInUseByGpu(uint64_t vaddr, uint64_t size) {
	return IsRegionRegistered(vaddr, size) || HasPendingGuestWrite(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick, WaitSite::GarbageCollector);
	m_scheduler.WaitPriorityOperations(completion_tick, WaitSite::GarbageCollector);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	// Take before walking: writes queued during synchronization remain pending.
	const auto pending = m_memory_tracker.TakeBdaSync();
	pending.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			SynchronizeBuffersInRange(start, finish - start);
		});
	});
}

void BufferCache::UploadCpuPending(const RangeSet& pending) {
	struct Upload {
		Buffer* buffer;
		size_t  first;
		size_t  count;
	};
	std::vector<Upload>         uploads;
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	const auto                  collect    = [&](uint64_t vaddr, uint64_t end) {
		auto it = m_buffers.upper_bound(vaddr);
		if (it != m_buffers.begin()) {
			--it;
		}
		for (; it != m_buffers.end() && it->first < end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto start  = std::max(buffer.CpuAddress(), vaddr);
			const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
			if (start >= finish) {
				continue;
			}
			const auto first = copies.size();
			// The pages turn clean (and write-watched) during the walk; a CPU write after that
			// faults and is pending again, whether or not the copy below took it.
			m_memory_tracker.ForEachUploadRange(
			    start, finish - start, false,
			    [&](uint64_t address, uint64_t bytes) noexcept {
				    copies.emplace_back(total_size, buffer.Offset(address), bytes);
				    total_size += bytes;
			    },
			    []() noexcept {});
			if (copies.size() == first) {
				continue;
			}
			// Pending ranges come in address order, so a buffer's copies are consecutive.
			if (uploads.empty() || uploads.back().buffer != &buffer) {
				uploads.push_back({&buffer, first, 0});
			}
			uploads.back().count = copies.size() - uploads.back().first;
		}
	};
	const uint64_t profile_t0     = FrameProfiler::Start();
	const uint64_t protect_calls0 = FrameProfiler::t_protect_calls;
	const uint64_t protect_ticks0 = FrameProfiler::t_protect_ticks;
	pending.ForEach([&](uint64_t start, uint64_t end) { ForEachMappedPart(start, end, collect); });
	if (profile_t0 != 0) {
		using FrameProfiler::Counter;
		using FrameProfiler::UploadCounter;
		FrameProfiler::Add(UploadCounter(Counter::UploadCallsAcquire));
		if (!copies.empty()) {
			FrameProfiler::RecordEvent(FrameProfiler::Event::EpochUpload, total_size, copies.size());
			FrameProfiler::Add(UploadCounter(Counter::UploadNonEmptyAcquire));
			FrameProfiler::Add(UploadCounter(Counter::UploadBytesAcquire), total_size);
			FrameProfiler::Add(UploadCounter(Counter::UploadCopiesAcquire), copies.size());
			FrameProfiler::AddSince(Counter::UploadWalkTime, profile_t0);
			// The walk re-protects the pages it cleans: its share of syscalls.
			FrameProfiler::Add(Counter::UploadWalkProtectCalls,
			                   FrameProfiler::t_protect_calls - protect_calls0);
			FrameProfiler::Add(Counter::UploadWalkProtectTime,
			                   FrameProfiler::t_protect_ticks - protect_ticks0);
		}
	}
	if (copies.empty()) {
		FrameProfiler::AddSince(FrameProfiler::Counter::UploadTime, profile_t0);
		return;
	}

	// The epoch upload does not go through UploadCopies, which counts the other uploads.
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	// One staging reservation: a wrap waiting on this submit then happens before any copy is
	// recorded, never between a copy and the data it reads.
	const uint64_t copy_t0         = profile_t0 != 0 ? FrameProfiler::Ticks() : 0;
	auto [staging, staging_offset] = m_staging_buffer.Map(total_size, 4);
	std::unique_ptr<Buffer> temporary;
	if (staging == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
		                                     vk::BufferUsageFlagBits::eTransferSrc, total_size);
		staging        = temporary->Mapped().data();
		staging_offset = 0;
	}
	for (const auto& upload: uploads) {
		for (size_t i = upload.first; i < upload.first + upload.count; i++) {
			auto& copy = copies[i];
			std::memcpy(staging + copy.srcOffset,
			            reinterpret_cast<const void*>(upload.buffer->CpuAddress() + copy.dstOffset),
			            copy.size);
			copy.srcOffset += staging_offset;
		}
	}
	FrameProfiler::AddSince(FrameProfiler::Counter::UploadCopyTime, copy_t0);
	vk::Buffer source;
	if (temporary) {
		FrameProfiler::Add(FrameProfiler::Counter::UploadTemporaryStaging);
		temporary->Flush(0, total_size);
		source = temporary->Handle();
		m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	} else {
		m_staging_buffer.Commit();
		source = m_staging_buffer.Handle();
	}

	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("BufferCache: CPU-write upload requires a recording command buffer\n");
	}
	command.EndRendering();
	const auto        native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	for (const auto& upload: uploads) {
		native.copyBuffer(source, upload.buffer->Handle(), static_cast<uint32_t>(upload.count),
		                  copies.data() + upload.first);
		for (size_t i = upload.first; i < upload.first + upload.count; i++) {
			NoteBufferWrite(upload.buffer->CpuAddress() + copies[i].dstOffset, copies[i].size);
		}
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                       nullptr);
	FrameProfiler::AddSince(FrameProfiler::Counter::UploadTime, profile_t0);
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
