#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// The commands of one command buffer, recorded on the GPU thread and replayed into a Vulkan
// command buffer by the submit thread (CommandScheduler) while they are recorded: each command is
// a trivially copyable closure, and the arrays and structures its arguments point to are copied
// into the log. The recording thread publishes commands in batches and closes the log at its
// submit; the replaying thread follows what is published.
class CommandLog {
public:
	CommandLog() = default;
	KYTY_CLASS_NO_COPY(CommandLog);

	template <typename F>
	void Push(F&& command) {
		using Command = std::decay_t<F>;
		static_assert(std::is_trivially_copyable_v<Command> &&
		              std::is_trivially_destructible_v<Command>);
		void* storage = Allocate(sizeof(Command), alignof(Command));
		new (storage) Command(std::forward<F>(command));
		const auto chunk = m_count / EntriesPerChunk;
		EXIT_IF(chunk >= MaxEntryChunks);
		if (m_entries[chunk] == nullptr) {
			m_entries[chunk] = std::make_unique<Entry[]>(EntriesPerChunk);
		}
		m_entries[chunk][m_count % EntriesPerChunk] = {
		    [](const void* payload, vk::CommandBuffer buffer) {
			    (*static_cast<const Command*>(payload))(buffer);
		    },
		    storage};
		if (++m_count % PublishBatch == 0) {
			Publish();
		}
	}

	// A copy of count elements that lives until the log is reset; null for none.
	template <typename T>
	[[nodiscard]] T* Copy(const T* data, size_t count) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (data == nullptr || count == 0) {
			return nullptr;
		}
		auto* copy = static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
		std::memcpy(static_cast<void*>(copy), data, sizeof(T) * count);
		return copy;
	}
	[[nodiscard]] const void* CopyBytes(const void* data, size_t size) {
		return Copy(static_cast<const std::byte*>(data), size);
	}

	// Recording thread: the commands so far may be replayed.
	void Publish() noexcept {
		m_state.store(m_count, std::memory_order_release);
		m_state.notify_one();
	}
	// Recording thread, at the submit: no command follows.
	void Close() noexcept {
		m_state.store(m_count | Closed, std::memory_order_release);
		m_state.notify_one();
	}
	// Replaying thread: replays every command into the buffer as it is published, until the log
	// is closed.
	void Replay(vk::CommandBuffer buffer) const {
		size_t done = 0;
		for (;;) {
			const auto state     = m_state.load(std::memory_order_acquire);
			const auto published = static_cast<size_t>(state & ~Closed);
			for (; done < published; done++) {
				const auto& entry = m_entries[done / EntriesPerChunk][done % EntriesPerChunk];
				entry.invoke(entry.payload, buffer);
			}
			if ((state & Closed) != 0) {
				return;
			}
			m_state.wait(state, std::memory_order_acquire);
		}
	}
	// Replaying thread, once replayed: keeps the storage for the next command buffer.
	void Reset() noexcept {
		m_count  = 0;
		m_chunk  = 0;
		m_offset = 0;
		m_state.store(0, std::memory_order_relaxed);
	}

private:
	static constexpr size_t   ChunkBytes      = 256 * 1024;
	static constexpr size_t   EntriesPerChunk = 4096;
	static constexpr size_t   MaxEntryChunks  = 1024;
	static constexpr size_t   PublishBatch    = 128;
	static constexpr uint64_t Closed          = uint64_t {1} << 63u;

	struct Entry {
		void (*invoke)(const void* payload, vk::CommandBuffer buffer);
		const void* payload;
	};
	struct Chunk {
		std::unique_ptr<std::byte[]> bytes;
		size_t                       size = 0;
	};

	void* Allocate(size_t size, size_t alignment) {
		for (;;) {
			if (m_chunk < m_chunks.size()) {
				auto&      chunk = m_chunks[m_chunk];
				const auto base  = reinterpret_cast<uintptr_t>(chunk.bytes.get());
				const auto at    = (base + m_offset + alignment - 1) & ~(alignment - 1);
				if (at + size <= base + chunk.size) {
					m_offset = at + size - base;
					return reinterpret_cast<void*>(at);
				}
				m_chunk++;
				m_offset = 0;
				continue;
			}
			const auto bytes = std::max(ChunkBytes, size + alignment);
			m_chunks.push_back({std::make_unique<std::byte[]>(bytes), bytes});
		}
	}

	// Chunks never move: the replaying thread reads entries while more are added.
	std::array<std::unique_ptr<Entry[]>, MaxEntryChunks> m_entries;
	size_t                                               m_count = 0;
	std::atomic<uint64_t>                                m_state {0};
	std::vector<Chunk>                                   m_chunks;
	size_t                                               m_chunk  = 0;
	size_t                                               m_offset = 0;
};

// What CommandBuffer::Handle returns: the vk::CommandBuffer calls the renderer makes, either
// made at once (a scheduler that records directly) or logged for the submit thread.
class CommandRecorder {
public:
	CommandRecorder() noexcept = default;
	explicit CommandRecorder(vk::CommandBuffer direct) noexcept: m_direct(direct) {}
	explicit CommandRecorder(CommandLog* log) noexcept: m_log(log) {}

	explicit operator bool() const noexcept { return m_log != nullptr || m_direct; }
	[[nodiscard]] bool operator==(const CommandRecorder& other) const noexcept {
		return Identity() == other.Identity();
	}
	// Which command buffer the commands go to, for caches of its state (a log is reused only
	// after a new command buffer has begun).
	[[nodiscard]] const void* Identity() const noexcept {
		return m_log != nullptr ? static_cast<const void*>(m_log)
		                        : static_cast<const void*>(static_cast<VkCommandBuffer>(m_direct));
	}
	// For a recorder outside this interface (third-party): a direct recorder's handle only.
	[[nodiscard]] vk::CommandBuffer Native() const {
		EXIT_IF(m_log != nullptr || !m_direct);
		return m_direct;
	}

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		Record([=](vk::CommandBuffer b) { b.bindPipeline(bind_point, pipeline); });
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		Record([=](vk::CommandBuffer b) { b.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		Record([=](vk::CommandBuffer b) { b.dispatchIndirect(buffer, offset); });
	}
	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex,
	          uint32_t first_instance) const {
		Record([=](vk::CommandBuffer b) {
			b.draw(vertices, instances, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index,
	                 int32_t vertex_offset, uint32_t first_instance) const {
		Record([=](vk::CommandBuffer b) {
			b.drawIndexed(indices, instances, first_index, vertex_offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		Record([=](vk::CommandBuffer b) { b.drawIndirect(buffer, offset, count, stride); });
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		Record([=](vk::CommandBuffer b) {
			b.drawIndexedIndirect(buffer, offset, count, stride);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		Record([=](vk::CommandBuffer b) { b.drawMeshTasksEXT(x, y, z); });
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Record([=](vk::CommandBuffer b) { b.bindIndexBuffer(buffer, offset, type); });
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (Direct()) {
			m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
			return;
		}
		const auto* b0 = m_log->Copy(buffers, count);
		const auto* o0 = m_log->Copy(offsets, count);
		const auto* s0 = m_log->Copy(sizes, count);
		const auto* t0 = m_log->Copy(strides, count);
		Log([=](vk::CommandBuffer b) { b.bindVertexBuffers2(first, count, b0, o0, s0, t0); });
	}
	void bindVertexBuffers(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                       const vk::DeviceSize* offsets) const {
		if (Direct()) {
			m_direct.bindVertexBuffers(first, count, buffers, offsets);
			return;
		}
		const auto* b0 = m_log->Copy(buffers, count);
		const auto* o0 = m_log->Copy(offsets, count);
		Log([=](vk::CommandBuffer b) { b.bindVertexBuffers(first, count, b0, o0); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) const {
		if (Direct()) {
			m_direct.bindDescriptorSets(bind_point, layout, first, count, sets, dynamic_count,
			                            dynamic_offsets);
			return;
		}
		const auto* s0 = m_log->Copy(sets, count);
		const auto* d0 = m_log->Copy(dynamic_offsets, dynamic_count);
		Log([=](vk::CommandBuffer b) {
			b.bindDescriptorSets(bind_point, layout, first, count, s0, dynamic_count, d0);
		});
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		if (Direct()) {
			m_direct.pushConstants(layout, stages, offset, size, values);
			return;
		}
		const auto* v0 = m_log->CopyBytes(values, size);
		Log([=](vk::CommandBuffer b) { b.pushConstants(layout, stages, offset, size, v0); });
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (Direct()) {
			m_direct.pushDescriptorSetKHR(bind_point, layout, set, count, writes);
			return;
		}
		const auto* w0 = CopyWrites(writes, count);
		Log([=](vk::CommandBuffer b) {
			b.pushDescriptorSetKHR(bind_point, layout, set, count, w0);
		});
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set,
	                          vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, writes.size(), writes.data());
	}
	void pushDescriptorSet(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                       uint32_t set, uint32_t count,
	                       const vk::WriteDescriptorSet* writes) const {
		if (Direct()) {
			m_direct.pushDescriptorSet(bind_point, layout, set, count, writes);
			return;
		}
		const auto* w0 = CopyWrites(writes, count);
		Log([=](vk::CommandBuffer b) { b.pushDescriptorSet(bind_point, layout, set, count, w0); });
	}
	void pushDescriptorSet(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                       uint32_t                                     set,
	                       vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
		pushDescriptorSet(bind_point, layout, set, writes.size(), writes.data());
	}

	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		if (Direct()) {
			m_direct.pipelineBarrier(src, dst, flags, memory_count, memory, buffer_count, buffers,
			                         image_count, images);
			return;
		}
		const auto* m0 = CopyChainless(memory, memory_count);
		const auto* b0 = CopyChainless(buffers, buffer_count);
		const auto* i0 = CopyChainless(images, image_count);
		Log([=](vk::CommandBuffer b) {
			b.pipelineBarrier(src, dst, flags, memory_count, m0, buffer_count, b0, image_count, i0);
		});
	}
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const {
		if (Direct()) {
			m_direct.pipelineBarrier2(dependency);
			return;
		}
		auto* d0 = m_log->Copy(&dependency, 1);
		EXIT_IF(d0->pNext != nullptr);
		d0->pMemoryBarriers =
		    CopyChainless(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		d0->pBufferMemoryBarriers =
		    CopyChainless(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
		d0->pImageMemoryBarriers =
		    CopyChainless(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		Log([=](vk::CommandBuffer b) { b.pipelineBarrier2(d0); });
	}
	void pipelineBarrier2(const vk::DependencyInfo* dependency) const {
		pipelineBarrier2(*dependency);
	}

	void beginRendering(const vk::RenderingInfo& info) const {
		if (Direct()) {
			m_direct.beginRendering(info);
			return;
		}
		auto* r0 = m_log->Copy(&info, 1);
		EXIT_IF(r0->pNext != nullptr);
		r0->pColorAttachments  = CopyChainless(info.pColorAttachments, info.colorAttachmentCount);
		r0->pDepthAttachment   = CopyChainless(info.pDepthAttachment, 1);
		r0->pStencilAttachment = CopyChainless(info.pStencilAttachment, 1);
		Log([=](vk::CommandBuffer b) { b.beginRendering(r0); });
	}
	void beginRendering(const vk::RenderingInfo* info) const { beginRendering(*info); }
	void endRendering() const {
		Record([](vk::CommandBuffer b) { b.endRendering(); });
	}
	void beginConditionalRenderingEXT(const vk::ConditionalRenderingBeginInfoEXT* info) const {
		if (Direct()) {
			m_direct.beginConditionalRenderingEXT(info);
			return;
		}
		const auto* c0 = CopyChainless(info, 1);
		Log([=](vk::CommandBuffer b) { b.beginConditionalRenderingEXT(c0); });
	}
	void endConditionalRenderingEXT() const {
		Record([](vk::CommandBuffer b) { b.endConditionalRenderingEXT(); });
	}
	void clearAttachments(uint32_t count, const vk::ClearAttachment* attachments,
	                      uint32_t rect_count, const vk::ClearRect* rects) const {
		if (Direct()) {
			m_direct.clearAttachments(count, attachments, rect_count, rects);
			return;
		}
		const auto* a0 = m_log->Copy(attachments, count);
		const auto* r0 = m_log->Copy(rects, rect_count);
		Log([=](vk::CommandBuffer b) { b.clearAttachments(count, a0, rect_count, r0); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout,
	                     const vk::ClearColorValue* color, uint32_t range_count,
	                     const vk::ImageSubresourceRange* ranges) const {
		if (Direct()) {
			m_direct.clearColorImage(image, layout, color, range_count, ranges);
			return;
		}
		const auto* c0 = m_log->Copy(color, 1);
		const auto* r0 = m_log->Copy(ranges, range_count);
		Log([=](vk::CommandBuffer b) { b.clearColorImage(image, layout, c0, range_count, r0); });
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue& color,
	                     vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
		clearColorImage(image, layout, &color, ranges.size(), ranges.data());
	}
	void waitEvents(uint32_t count, const vk::Event* events, vk::PipelineStageFlags src,
	                vk::PipelineStageFlags dst, uint32_t memory_count,
	                const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                const vk::ImageMemoryBarrier* images) const {
		if (Direct()) {
			m_direct.waitEvents(count, events, src, dst, memory_count, memory, buffer_count,
			                    buffers, image_count, images);
			return;
		}
		const auto* e0 = m_log->Copy(events, count);
		const auto* m0 = CopyChainless(memory, memory_count);
		const auto* b0 = CopyChainless(buffers, buffer_count);
		const auto* i0 = CopyChainless(images, image_count);
		Log([=](vk::CommandBuffer b) {
			b.waitEvents(count, e0, src, dst, memory_count, m0, buffer_count, b0, image_count, i0);
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t range_count,
	                            const vk::ImageSubresourceRange* ranges) const {
		if (Direct()) {
			m_direct.clearDepthStencilImage(image, layout, value, range_count, ranges);
			return;
		}
		const auto* v0 = m_log->Copy(value, 1);
		const auto* r0 = m_log->Copy(ranges, range_count);
		Log([=](vk::CommandBuffer b) {
			b.clearDepthStencilImage(image, layout, v0, range_count, r0);
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue&                     value,
	                            vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
		clearDepthStencilImage(image, layout, &value, ranges.size(), ranges.data());
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t data) const {
		Record([=](vk::CommandBuffer b) { b.fillBuffer(buffer, offset, size, data); });
	}
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const {
		if (Direct()) {
			m_direct.updateBuffer(buffer, offset, size, data);
			return;
		}
		const auto* d0 = m_log->CopyBytes(data, size);
		Log([=](vk::CommandBuffer b) { b.updateBuffer(buffer, offset, size, d0); });
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		if (Direct()) {
			m_direct.copyBuffer(source, destination, count, regions);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) { b.copyBuffer(source, destination, count, r0); });
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const {
		if (Direct()) {
			m_direct.copyImage(source, source_layout, destination, destination_layout, count,
			                   regions);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) {
			b.copyImage(source, source_layout, destination, destination_layout, count, r0);
		});
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (Direct()) {
			m_direct.copyBufferToImage(source, destination, layout, count, regions);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) {
			b.copyBufferToImage(source, destination, layout, count, r0);
		});
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (Direct()) {
			m_direct.copyImageToBuffer(source, layout, destination, count, regions);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) {
			b.copyImageToBuffer(source, layout, destination, count, r0);
		});
	}
	void blitImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageBlit* regions, vk::Filter filter) const {
		if (Direct()) {
			m_direct.blitImage(source, source_layout, destination, destination_layout, count,
			                   regions, filter);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) {
			b.blitImage(source, source_layout, destination, destination_layout, count, r0, filter);
		});
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, uint32_t count,
	                  const vk::ImageResolve* regions) const {
		if (Direct()) {
			m_direct.resolveImage(source, source_layout, destination, destination_layout, count,
			                      regions);
			return;
		}
		const auto* r0 = m_log->Copy(regions, count);
		Log([=](vk::CommandBuffer b) {
			b.resolveImage(source, source_layout, destination, destination_layout, count, r0);
		});
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination,
	                vk::ArrayProxy<const vk::BufferCopy> const& regions) const {
		copyBuffer(source, destination, regions.size(), regions.data());
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout                     destination_layout,
	               vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
		copyImage(source, source_layout, destination, destination_layout, regions.size(),
		          regions.data());
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyBufferToImage(source, destination, layout, regions.size(), regions.data());
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
		copyImageToBuffer(source, layout, destination, regions.size(), regions.data());
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout                        destination_layout,
	                  vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
		resolveImage(source, source_layout, destination, destination_layout, regions.size(),
		             regions.data());
	}
	void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool,
	                    uint32_t query) const {
		Record([=](vk::CommandBuffer b) { b.writeTimestamp(stage, pool, query); });
	}
	void resetQueryPool(vk::QueryPool pool, uint32_t first, uint32_t count) const {
		Record([=](vk::CommandBuffer b) { b.resetQueryPool(pool, first, count); });
	}
	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
		if (Direct()) {
			m_direct.setViewport(first, count, viewports);
			return;
		}
		const auto* v0 = m_log->Copy(viewports, count);
		Log([=](vk::CommandBuffer b) { b.setViewport(first, count, v0); });
	}
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
		if (Direct()) {
			m_direct.setScissor(first, count, scissors);
			return;
		}
		const auto* s0 = m_log->Copy(scissors, count);
		Log([=](vk::CommandBuffer b) { b.setScissor(first, count, s0); });
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (Direct()) {
			m_direct.setViewportWithCount(count, viewports);
			return;
		}
		const auto* v0 = m_log->Copy(viewports, count);
		Log([=](vk::CommandBuffer b) { b.setViewportWithCount(count, v0); });
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (Direct()) {
			m_direct.setScissorWithCount(count, scissors);
			return;
		}
		const auto* s0 = m_log->Copy(scissors, count);
		Log([=](vk::CommandBuffer b) { b.setScissorWithCount(count, s0); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (Direct()) {
			m_direct.setColorWriteEnableEXT(count, enables);
			return;
		}
		const auto* e0 = m_log->Copy(enables, count);
		Log([=](vk::CommandBuffer b) { b.setColorWriteEnableEXT(count, e0); });
	}
	void setBlendConstants(const float constants[4]) const {
		if (Direct()) {
			m_direct.setBlendConstants(constants);
			return;
		}
		const auto* c0 = m_log->Copy(constants, 4);
		Log([=](vk::CommandBuffer b) { b.setBlendConstants(c0); });
	}
	void setLineWidth(float width) const {
		Record([=](vk::CommandBuffer b) { b.setLineWidth(width); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		Record([=](vk::CommandBuffer b) { b.setDepthBias(constant, clamp, slope); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer b) { b.setDepthBiasEnable(enable); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer b) { b.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer b) { b.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		Record([=](vk::CommandBuffer b) { b.setDepthCompareOp(op); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer b) { b.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		Record([=](vk::CommandBuffer b) { b.setDepthBounds(min_bounds, max_bounds); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		Record([=](vk::CommandBuffer b) { b.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		Record([=](vk::CommandBuffer b) {
			b.setStencilOp(faces, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Record([=](vk::CommandBuffer b) { b.setStencilCompareMask(faces, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags faces, uint32_t mask) const {
		Record([=](vk::CommandBuffer b) { b.setStencilWriteMask(faces, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags faces, uint32_t reference) const {
		Record([=](vk::CommandBuffer b) { b.setStencilReference(faces, reference); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		Record([=](vk::CommandBuffer b) { b.setAttachmentFeedbackLoopEnableEXT(aspects); });
	}

private:
	[[nodiscard]] bool Direct() const noexcept { return m_log == nullptr; }

	// A command whose arguments are values: made at once, or logged.
	template <typename F>
	void Record(F&& command) const {
		if (Direct()) {
			EXIT_IF(!m_direct);
			command(m_direct);
			return;
		}
		m_log->Push(std::forward<F>(command));
	}
	template <typename F>
	void Log(F&& command) const {
		m_log->Push(std::forward<F>(command));
	}

	// Structures whose pNext the renderer leaves null.
	template <typename T>
	[[nodiscard]] T* CopyChainless(const T* data, size_t count) const {
		auto* copy = m_log->Copy(data, count);
		for (size_t i = 0; copy != nullptr && i < count; i++) {
			EXIT_IF(copy[i].pNext != nullptr);
		}
		return copy;
	}

	[[nodiscard]] const vk::WriteDescriptorSet* CopyWrites(const vk::WriteDescriptorSet* writes,
	                                                       uint32_t count) const {
		auto* copy = m_log->Copy(writes, count);
		for (uint32_t i = 0; i < count; i++) {
			auto& write = copy[i];
			const auto n = write.descriptorCount;
			write.pBufferInfo      = m_log->Copy(writes[i].pBufferInfo, n);
			write.pImageInfo       = m_log->Copy(writes[i].pImageInfo, n);
			write.pTexelBufferView = m_log->Copy(writes[i].pTexelBufferView, n);
			if (write.pNext == nullptr) {
				continue;
			}
			// Acceleration structures are the one chained write the renderer makes.
			const auto* chained =
			    static_cast<const vk::WriteDescriptorSetAccelerationStructureKHR*>(write.pNext);
			EXIT_IF(chained->sType !=
			            vk::StructureType::eWriteDescriptorSetAccelerationStructureKHR ||
			        chained->pNext != nullptr);
			auto* structures = m_log->Copy(chained, 1);
			structures->pAccelerationStructures = m_log->Copy(
			    chained->pAccelerationStructures, chained->accelerationStructureCount);
			write.pNext = structures;
		}
		return copy;
	}

	vk::CommandBuffer m_direct = nullptr;
	CommandLog*       m_log    = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
