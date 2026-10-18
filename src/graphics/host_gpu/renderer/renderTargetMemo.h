#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETMEMO_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics {

// Draws mostly keep their targets: the last resolve per colour slot and of the depth target, keyed
// by the register bytes it decoded. Its texture-cache answer holds while its resolve stamp does
// (TextureCache::ReuseImage), the same proof the texture resolve cache uses.
struct RenderTargetMemo {
	static_assert(std::is_trivially_copyable_v<HW::RenderTarget>);
	static_assert(std::is_trivially_copyable_v<HW::DepthRenderTarget>);

	// Byte copies: equal bytes mean equal registers; differing padding only costs a full resolve.
	template <typename T>
	struct Key {
		std::array<unsigned char, sizeof(T)> bytes {};

		void Set(const T& regs) { std::memcpy(bytes.data(), &regs, sizeof(T)); }
		[[nodiscard]] bool Matches(const T& regs) const {
			return std::memcmp(bytes.data(), &regs, sizeof(T)) == 0;
		}
	};

	struct Color {
		Key<HW::RenderTarget>      regs;
		uint32_t                   mask         = 0;
		uint32_t                   slice_offset = 0;
		bool                       exact_format = false;
		bool                       valid        = false;
		// The resolved target (its desc as FindImage adjusted it) and the view's base layer
		// before the adjustment, which places fast-clear metadata.
		RenderColorInfo            info;
		uint32_t                   metadata_base_layer = 0;
		TextureCache::ResolveStamp stamp;
	};

	struct Depth {
		Key<HW::DepthRenderTarget> regs;
		bool                       valid = false;
		TextureCache::ImageDesc    desc;
		ImageId                    image_id;
		uint32_t                   metadata_base_layer = 0;
		TextureCache::ResolveStamp stamp;
	};

	std::array<Color, RENDER_COLOR_ATTACHMENTS_MAX> color;
	Depth                                           depth;
	// Lifetime totals (tests): resolves that took the full path.
	uint64_t full_resolves = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERTARGETMEMO_H_
