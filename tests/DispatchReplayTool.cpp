// dispatch_replay_tool <capture-dir> [options]
//
// Replays one captured compute dispatch on the GPU, offline: the guest shader goes through the
// recompiler as built (so emitter changes take effect), its pipeline is created as the renderer
// creates it, the captured buffers, BDA page table, images and samplers are bound as they were,
// and the dispatch runs a number of times between GPU timestamps. The outputs of the first run
// are compared with what the dispatch wrote in the game. Not a test: an offline benchmark and
// oracle for recompiler changes. A capture holds game data; it stays outside the repository.
//
// The capture directory holds meta.txt, pre.bin (contents before the dispatch) and post.bin
// (the bytes the dispatch changed); its parent holds compile-<hash>-<program>.bin (the
// recompiler inputs) and spv-<hash>-<program>.spv (the SPIR-V the game ran).
//
// Options:
//   --repeats N        timed runs (default 20)
//   --game-spirv       run the captured SPIR-V instead of recompiling
//   --spirv FILE       run FILE instead of recompiling
//   --save-spirv FILE  write the SPIR-V that runs
//   --compile-only     stop once the pipeline is built (with RADV_DEBUG=shaders: its ISA)
//   --dump-ir PREFIX   write the decoded guest program and the IR to PREFIX.decoded/.ir
//   --save-output FILE write the written objects' bytes after the first run
//   --reference FILE   compare the first run with a --save-output file instead of the capture
//   --unordered N      compare a buffer that differs as a multiset of N-byte records (a
//                      dispatch that appends with atomics writes them in any order)
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/BindlessImages.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace Libs::Graphics;

[[noreturn]] void Fail(const char* format, ...) {
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fprintf(stderr, "\n");
	std::exit(1);
}

void Check(vk::Result result, const char* what) {
	if (result != vk::Result::eSuccess) Fail("%s failed: %s", what, vk::to_string(result).c_str());
}

struct Mapped {
	const uint8_t* data = nullptr;
	size_t         size = 0;
	explicit Mapped(const std::string& path) {
		const int fd = open(path.c_str(), O_RDONLY);
		if (fd < 0) Fail("cannot open %s", path.c_str());
		struct stat st {};
		fstat(fd, &st);
		size = static_cast<size_t>(st.st_size);
		if (size != 0) {
			data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
			if (data == MAP_FAILED) Fail("cannot map %s", path.c_str());
		}
		close(fd);
	}
};

std::vector<uint8_t> ReadFile(const std::string& path) {
	std::ifstream file(path, std::ios::binary);
	if (!file) Fail("cannot open %s", path.c_str());
	return {std::istreambuf_iterator<char>(file), {}};
}

// ---- capture description ----

struct Obj {
	int      kind = 0; // 0 cache buffer, 1 other buffer range, 2 BDA page table
	uint64_t size = 0, cpu = 0, bda = 0, pre = 0;
	std::vector<std::array<uint64_t, 3>> diffs; // offset, size, post.bin offset
};
struct Img {
	uint32_t flags = 0, type = 0, format = 0, w = 1, h = 1, d = 1, mips = 1, layers = 1, samples = 1,
	         usage = 0;
};
struct View {
	size_t   image = 0;
	int      type = 0, format = 0, r = 0, g = 0, b = 0, a = 0;
	uint32_t aspect = 0, base_mip = 0, mips = 0, base_layer = 0, layers = 0, usage = 0;
	float    min_lod = -1;
};
struct Sub {
	size_t   image = 0;
	uint32_t aspect = 0, mip = 0, layer = 0, w = 0, h = 0, d = 0;
	uint64_t bytes = 0, pre = 0;
	int      layout = 0;
	int64_t  post = -1;
};
struct Elem {
	long     a = -1, b = -1; // buffer: obj, -; image: sampler, view
	uint64_t offset = 0, range = 0;
	int      layout = 0;
};
struct Write {
	uint32_t          set = 0, binding = 0;
	int               type = 0;
	std::vector<Elem> elems;
};

struct Capture {
	uint64_t                                 hash = 0, frame = 0, program = 0;
	uint32_t                                 groups[3] {};
	std::vector<uint32_t>                    push;
	std::vector<Obj>                         objs;
	std::vector<std::pair<uint64_t, std::pair<long, uint64_t>>> ptes; // page -> obj (-1 zero), off
	std::vector<Img>                         images;
	std::vector<View>                        views;
	std::vector<Sub>                         subs;
	std::vector<vk::SamplerCreateInfo>       samplers;
	std::vector<Write>                       writes;
};

Capture Load(const std::string& dir) {
	Capture c;
	std::ifstream file(dir + "/meta.txt");
	if (!file) Fail("cannot open %s/meta.txt", dir.c_str());
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream in(line);
		std::string        tag;
		in >> tag;
		if (tag == "hash") {
			in >> std::hex >> c.hash;
		} else if (tag == "frame") {
			in >> c.frame;
		} else if (tag == "program") {
			in >> c.program;
		} else if (tag == "groups") {
			in >> c.groups[0] >> c.groups[1] >> c.groups[2];
		} else if (tag == "push") {
			uint32_t v = 0;
			while (in >> std::hex >> v) c.push.push_back(v);
		} else if (tag == "obj") {
			size_t i = 0;
			Obj    o;
			in >> i >> o.kind >> o.size >> std::hex >> o.cpu >> o.bda >> std::dec >> o.pre;
			if (i != c.objs.size()) Fail("obj order");
			c.objs.push_back(o);
		} else if (tag == "pd") {
			size_t   i = 0;
			uint64_t off = 0, size = 0, at = 0;
			in >> i >> off >> size >> at;
			c.objs.at(i).diffs.push_back({off, size, at});
		} else if (tag == "pte") {
			uint64_t    page = 0;
			std::string obj;
			in >> page >> obj;
			if (obj == "zero") {
				c.ptes.push_back({page, {-1, 0}});
			} else {
				uint64_t off = 0;
				in >> off;
				c.ptes.push_back({page, {std::stol(obj), off}});
			}
		} else if (tag == "img") {
			size_t i = 0;
			Img    m;
			in >> i >> m.flags >> m.type >> m.format >> m.w >> m.h >> m.d >> m.mips >> m.layers >>
			    m.samples >> m.usage;
			c.images.push_back(m);
		} else if (tag == "view") {
			size_t      i = 0;
			View        v;
			std::string lod;
			in >> i >> v.image >> v.type >> v.format >> v.r >> v.g >> v.b >> v.a >> v.aspect >>
			    v.base_mip >> v.mips >> v.base_layer >> v.layers >> v.usage >> lod;
			v.min_lod = std::strtof(lod.c_str(), nullptr);
			c.views.push_back(v);
		} else if (tag == "sub") {
			size_t i = 0;
			Sub    s;
			in >> i >> s.image >> s.aspect >> s.mip >> s.layer >> s.w >> s.h >> s.d >> s.bytes >>
			    s.layout >> s.pre;
			c.subs.push_back(s);
		} else if (tag == "ps") {
			size_t  i  = 0;
			int64_t at = 0;
			in >> i >> at;
			c.subs.at(i).post = at;
		} else if (tag == "sampler") {
			size_t                i = 0;
			int                   mag, min, mip, u, v, w, compare_op, border;
			uint32_t              aniso, compare, unnorm;
			std::string           bias, max_aniso, min_lod, max_lod;
			in >> i >> mag >> min >> mip >> u >> v >> w >> bias >> aniso >> max_aniso >> compare >>
			    compare_op >> min_lod >> max_lod >> border >> unnorm;
			vk::SamplerCreateInfo s {};
			s.magFilter               = static_cast<vk::Filter>(mag);
			s.minFilter               = static_cast<vk::Filter>(min);
			s.mipmapMode              = static_cast<vk::SamplerMipmapMode>(mip);
			s.addressModeU            = static_cast<vk::SamplerAddressMode>(u);
			s.addressModeV            = static_cast<vk::SamplerAddressMode>(v);
			s.addressModeW            = static_cast<vk::SamplerAddressMode>(w);
			s.mipLodBias              = std::strtof(bias.c_str(), nullptr);
			s.anisotropyEnable        = aniso;
			s.maxAnisotropy           = std::strtof(max_aniso.c_str(), nullptr);
			s.compareEnable           = compare;
			s.compareOp               = static_cast<vk::CompareOp>(compare_op);
			s.minLod                  = std::strtof(min_lod.c_str(), nullptr);
			s.maxLod                  = std::strtof(max_lod.c_str(), nullptr);
			s.borderColor             = static_cast<vk::BorderColor>(border);
			s.unnormalizedCoordinates = unnorm;
			c.samplers.push_back(s);
		} else if (tag == "write") {
			Write  w;
			size_t count = 0;
			in >> w.set >> w.binding >> w.type >> count;
			c.writes.push_back(w);
		} else if (tag == "b") {
			Elem e;
			in >> e.a >> e.offset >> e.range;
			c.writes.back().elems.push_back(e);
		} else if (tag == "i") {
			Elem e;
			in >> e.a >> e.b >> e.layout;
			c.writes.back().elems.push_back(e);
		}
	}
	return c;
}

// ---- compile inputs ----

struct Reader {
	std::vector<uint8_t> bytes;
	size_t               at = 0;
	void Raw(void* out, size_t size) {
		if (at + size > bytes.size()) Fail("compile file truncated");
		std::memcpy(out, bytes.data() + at, size);
		at += size;
	}
	uint32_t U32() {
		uint32_t v;
		Raw(&v, 4);
		return v;
	}
	uint64_t U64() {
		uint64_t v;
		Raw(&v, 8);
		return v;
	}
	template <typename T>
	std::vector<T> Vec() {
		const auto count = U64(), size = U64();
		if (size != sizeof(T)) Fail("compile file element size %" PRIu64 " != %zu", size, sizeof(T));
		std::vector<T> v(count);
		Raw(v.data(), count * sizeof(T));
		return v;
	}
};

struct Compiled {
	std::vector<uint32_t> spirv;
	uint32_t              wave_size = 64;
};

std::string g_dump_ir;

Compiled Recompile(const std::string& path, const std::vector<uint32_t>* override_spirv) {
	using namespace ShaderRecompiler;
	Reader r {ReadFile(path)};
	if (r.U32() != 0x4b434d50u) Fail("%s: not a compile capture", path.c_str());
	const auto hash = r.U64();
	(void)r.U64();
	if (r.U32() != 0) Fail("linked shaders are not supported");
	CompileOptions options;
	options.stage                = static_cast<ShaderType>(r.U32());
	options.wave_size            = r.U32();
	options.user_data_base       = r.U32();
	options.maintenance8_enabled = r.U32() != 0;
	options.loop_watchdog_limit  = r.U32();
	options.untracked_writes     = r.U32() != 0;
	options.bindless_images      = r.U32() != 0;
	options.table_calls          = r.U32() != 0;
	options.emit_debug_info      = r.U32() != 0;
	(void)r.U32();
	(void)r.U32();
	options.dump_ir              = !g_dump_ir.empty();
	options.early_dump           = false;
	options.shader_hash          = hash;
	options.dump_label           = "replay";
	const auto push_cursor       = r.U32();
	const auto user_data         = r.Vec<uint32_t>();
	const auto back_code         = r.Vec<uint32_t>();
	const auto code              = r.Vec<uint32_t>();
	options.bindless_image_pcs   = r.Vec<uint32_t>();
	ShaderComputeInputInfo input {};
	if (r.U64() != sizeof(input)) Fail("ShaderComputeInputInfo layout changed");
	r.Raw(&input, sizeof(input));
	input.stage = {};
	IR::ResourceSpecialization spec;
	spec.buffers  = r.Vec<IR::ResourceSpecialization::Buffer>();
	spec.images   = r.Vec<IR::ResourceSpecialization::Image>();
	spec.samplers = r.Vec<IR::ResourceSpecialization::Sampler>();
	options.user_data           = user_data;
	options.back_code           = back_code;
	options.input_info.compute  = &input;
	Compiled out;
	out.wave_size = options.wave_size;
	if (override_spirv != nullptr) {
		out.spirv = *override_spirv;
		return out;
	}
	auto source     = PrepareShaderSource(code, options);
	auto translated = TranslateProgram(source.decoded, options);
	const auto t0   = std::chrono::steady_clock::now();
	auto result     = CompileProgram(std::move(translated), options, spec, push_cursor);
	std::printf("recompiled in %.1f ms: %zu words\n",
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
	            result.spirv.size());
	if (!g_dump_ir.empty()) {
		std::ofstream(g_dump_ir + ".decoded") << result.decoded_dump;
		std::ofstream(g_dump_ir + ".ir") << result.ir_dump;
	}
	out.spirv     = std::move(result.spirv);
	out.wave_size = result.program.wave_size;
	return out;
}

// ---- Vulkan ----

struct Gpu {
	vk::Instance                       instance;
	vk::PhysicalDevice                 physical;
	vk::Device                         device;
	vk::Queue                          queue;
	uint32_t                           family = 0;
	vk::CommandPool                    pool;
	vk::PhysicalDeviceMemoryProperties memory {};
	vk::PhysicalDeviceProperties       properties {};
	uint32_t                           min_subgroup = 0, max_subgroup = 0, max_push = 0;

	struct Chunk {
		vk::DeviceMemory memory;
		uint64_t         size = 0, used = 0;
		void*            mapped = nullptr;
	};
	std::map<uint32_t, std::vector<Chunk>> arenas;

	void Init() {
		static vk::detail::DynamicLoader loader;
		VULKAN_HPP_DEFAULT_DISPATCHER.init(
		    loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
		vk::ApplicationInfo app {};
		app.pApplicationName = "dispatch_replay_tool";
		app.apiVersion       = VK_API_VERSION_1_3;
		vk::InstanceCreateInfo instance_info {};
		instance_info.pApplicationInfo = &app;
		Check(vk::createInstance(&instance_info, nullptr, &instance), "vkCreateInstance");
		VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
		uint32_t count = 0;
		(void)instance.enumeratePhysicalDevices(&count, nullptr);
		std::vector<vk::PhysicalDevice> devices(count);
		(void)instance.enumeratePhysicalDevices(&count, devices.data());
		for (auto d: devices) {
			vk::PhysicalDeviceProperties p {};
			d.getProperties(&p);
			if (p.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || physical == nullptr) physical = d;
		}
		if (physical == nullptr) Fail("no Vulkan device");
		physical.getProperties(&properties);
		physical.getMemoryProperties(&memory);
		vk::PhysicalDeviceVulkan13Properties p13 {};
		vk::PhysicalDevicePushDescriptorPropertiesKHR push {};
		p13.pNext = &push;
		vk::PhysicalDeviceProperties2 p2 {};
		p2.pNext = &p13;
		physical.getProperties2(&p2);
		min_subgroup = p13.minSubgroupSize;
		max_subgroup = p13.maxSubgroupSize;
		max_push     = push.maxPushDescriptors;
		uint32_t queues = 0;
		physical.getQueueFamilyProperties(&queues, nullptr);
		std::vector<vk::QueueFamilyProperties> families(queues);
		physical.getQueueFamilyProperties(&queues, families.data());
		for (uint32_t i = 0; i < queues; i++) {
			if ((families[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) ==
			    (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) {
				family = i;
				break;
			}
		}
		// Every supported feature, with robustness as the emulator sets it (vulkanWindow.cpp).
		vk::PhysicalDeviceRobustness2FeaturesEXT       robustness2 {};
		vk::PhysicalDeviceImageViewMinLodFeaturesEXT   min_lod {};
		vk::PhysicalDeviceMaintenance8FeaturesKHR      maintenance8 {};
		vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic64 {};
		vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup {};
		vk::PhysicalDeviceVulkan13Features             f13 {};
		vk::PhysicalDeviceVulkan12Features             f12 {};
		vk::PhysicalDeviceVulkan11Features             f11 {};
		vk::PhysicalDeviceFeatures2                    features {};
		features.pNext     = &f11;
		f11.pNext          = &f12;
		f12.pNext          = &f13;
		f13.pNext          = &robustness2;
		robustness2.pNext  = &min_lod;
		min_lod.pNext      = &maintenance8;
		maintenance8.pNext = &image_atomic64;
		image_atomic64.pNext = &workgroup;
		physical.getFeatures2(&features);
		features.features.robustBufferAccess = VK_TRUE;
		robustness2.nullDescriptor           = VK_TRUE;
		robustness2.robustImageAccess2       = robustness2.robustImageAccess2;
		f12.bufferDeviceAddressCaptureReplay = VK_FALSE;
		f12.bufferDeviceAddressMultiDevice   = VK_FALSE;
		std::vector<const char*> extensions {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		                                     VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
		                                     VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME};
		uint32_t ext_count = 0;
		(void)physical.enumerateDeviceExtensionProperties(nullptr, &ext_count, nullptr);
		std::vector<vk::ExtensionProperties> available(ext_count);
		(void)physical.enumerateDeviceExtensionProperties(nullptr, &ext_count, available.data());
		const auto has = [&](const char* name) {
			return std::any_of(available.begin(), available.end(),
			                   [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
		};
		// Optional extensions: their feature structs leave the chain when absent.
		void** tail = &robustness2.pNext;
		*tail       = nullptr;
		for (auto [name, node, next]: std::initializer_list<std::tuple<const char*, void*, void**>> {
		         {VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, &min_lod, &min_lod.pNext},
		         {VK_KHR_MAINTENANCE_8_EXTENSION_NAME, &maintenance8, &maintenance8.pNext},
		         {VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, &image_atomic64, &image_atomic64.pNext},
		         {VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME, &workgroup, &workgroup.pNext}}) {
			if (!has(name)) continue;
			if (std::strcmp(name, VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME) != 0) extensions.push_back(name);
			*tail = node;
			tail  = next;
			*tail = nullptr;
		}
		const float             priority = 1.0f;
		vk::DeviceQueueCreateInfo queue_info {};
		queue_info.queueFamilyIndex = family;
		queue_info.queueCount       = 1;
		queue_info.pQueuePriorities = &priority;
		vk::DeviceCreateInfo create {};
		create.pNext                   = &features;
		create.queueCreateInfoCount    = 1;
		create.pQueueCreateInfos       = &queue_info;
		create.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		create.ppEnabledExtensionNames = extensions.data();
		Check(physical.createDevice(&create, nullptr, &device), "vkCreateDevice");
		VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
		queue = device.getQueue(family, 0);
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = family;
		Check(device.createCommandPool(&pool_info, nullptr, &pool), "vkCreateCommandPool");
		std::printf("device: %s, subgroups %u-%u, push descriptors %u\n", properties.deviceName.data(),
		            min_subgroup, max_subgroup, max_push);
	}

	uint32_t MemoryType(uint32_t bits, vk::MemoryPropertyFlags flags) const {
		for (uint32_t i = 0; i < memory.memoryTypeCount; i++)
			if ((bits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
		Fail("no memory type");
	}

	// Suballocates; chunks of 1 GiB or more.
	std::pair<vk::DeviceMemory, uint64_t> Allocate(const vk::MemoryRequirements& req,
	                                               vk::MemoryPropertyFlags flags, void** mapped = nullptr) {
		const auto type   = MemoryType(req.memoryTypeBits, flags);
		auto&      chunks = arenas[type];
		for (auto& c: chunks) {
			const auto at = (c.used + req.alignment - 1) / req.alignment * req.alignment;
			if (at + req.size <= c.size) {
				c.used = at + req.size;
				if (mapped) *mapped = static_cast<uint8_t*>(c.mapped) + at;
				return {c.memory, at};
			}
		}
		Chunk c;
		c.size = std::max<uint64_t>(req.size, 1ull << 30);
		vk::MemoryAllocateFlagsInfo flags_info {};
		flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
		vk::MemoryAllocateInfo info {};
		info.pNext           = &flags_info;
		info.allocationSize  = c.size;
		info.memoryTypeIndex = type;
		Check(device.allocateMemory(&info, nullptr, &c.memory), "vkAllocateMemory");
		if (flags & vk::MemoryPropertyFlagBits::eHostVisible)
			Check(device.mapMemory(c.memory, 0, VK_WHOLE_SIZE, {}, &c.mapped), "vkMapMemory");
		c.used = req.size;
		if (mapped) *mapped = c.mapped;
		chunks.push_back(c);
		return {c.memory, 0};
	}

	struct Buf {
		vk::Buffer buffer;
		uint64_t   size = 0;
		uint64_t   address = 0;
		uint8_t*   mapped = nullptr;
	};
	Buf MakeBuffer(uint64_t size, bool host = false) {
		Buf b;
		b.size = std::max<uint64_t>(size, 4);
		vk::BufferCreateInfo info {};
		info.size  = b.size;
		info.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
		             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
		             vk::BufferUsageFlagBits::eUniformBuffer;
		Check(device.createBuffer(&info, nullptr, &b.buffer), "vkCreateBuffer");
		vk::MemoryRequirements req {};
		device.getBufferMemoryRequirements(b.buffer, &req);
		void* mapped = nullptr;
		const auto [mem, off] = Allocate(req,
		    host ? vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
		         : vk::MemoryPropertyFlagBits::eDeviceLocal, host ? &mapped : nullptr);
		Check(device.bindBufferMemory(b.buffer, mem, off), "vkBindBufferMemory");
		vk::BufferDeviceAddressInfo address {};
		address.buffer = b.buffer;
		b.address      = device.getBufferAddress(&address);
		b.mapped       = static_cast<uint8_t*>(mapped);
		return b;
	}

	vk::CommandBuffer Begin() {
		vk::CommandBufferAllocateInfo info {};
		info.commandPool        = pool;
		info.level              = vk::CommandBufferLevel::ePrimary;
		info.commandBufferCount = 1;
		vk::CommandBuffer cb;
		Check(device.allocateCommandBuffers(&info, &cb), "vkAllocateCommandBuffers");
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		Check(cb.begin(&begin), "vkBeginCommandBuffer");
		return cb;
	}
	void Submit(vk::CommandBuffer cb) {
		Check(cb.end(), "vkEndCommandBuffer");
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &cb;
		Check(queue.submit(1, &submit, nullptr), "vkQueueSubmit");
		Check(queue.waitIdle(), "vkQueueWaitIdle");
		device.freeCommandBuffers(pool, 1, &cb);
	}
};

void FullBarrier(vk::CommandBuffer cb) {
	vk::MemoryBarrier b {};
	b.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	b.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
	                   {}, 1, &b, 0, nullptr, 0, nullptr);
}

vk::ImageAspectFlags FullAspect(vk::Format format) {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD32Sfloat:
		case vk::Format::eX8D24UnormPack32: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

// Streams host bytes into device buffers through one staging buffer.
struct Uploader {
	Gpu&              gpu;
	Gpu::Buf          staging;
	uint64_t          used = 0;
	vk::CommandBuffer cb;
	explicit Uploader(Gpu& g): gpu(g) {
		staging = gpu.MakeBuffer(256ull << 20, true);
		cb      = gpu.Begin();
	}
	void Flush() {
		gpu.Submit(cb);
		cb   = gpu.Begin();
		used = 0;
	}
	// Copies size bytes from src to dst at offset (or to image subresource when image is set).
	void Buffer(vk::Buffer dst, uint64_t offset, const uint8_t* src, uint64_t size) {
		while (size != 0) {
			if (used == staging.size) Flush();
			const auto n = std::min(size, staging.size - used);
			std::memcpy(staging.mapped + used, src, n);
			vk::BufferCopy region {used, offset, n};
			cb.copyBuffer(staging.buffer, dst, 1, &region);
			used += (n + 255) & ~uint64_t {255};
			used = std::min(used, staging.size);
			src += n;
			offset += n;
			size -= n;
		}
	}
	void Image(vk::Image dst, const vk::BufferImageCopy& where, const uint8_t* src, uint64_t size) {
		if (size > staging.size) Fail("subresource larger than the staging buffer");
		if (used + size > staging.size) Flush();
		std::memcpy(staging.mapped + used, src, size);
		auto region         = where;
		region.bufferOffset = used;
		cb.copyBufferToImage(staging.buffer, dst, vk::ImageLayout::eTransferDstOptimal, 1, &region);
		used = std::min(staging.size, used + ((size + 255) & ~uint64_t {255}));
	}
};

struct Downloader {
	Gpu&     gpu;
	Gpu::Buf staging;
	explicit Downloader(Gpu& g): gpu(g) { staging = gpu.MakeBuffer(256ull << 20, true); }
	void Buffer(vk::Buffer src, uint64_t offset, uint8_t* dst, uint64_t size) {
		while (size != 0) {
			const auto n  = std::min(size, staging.size);
			auto       cb = gpu.Begin();
			vk::BufferCopy region {offset, 0, n};
			cb.copyBuffer(src, staging.buffer, 1, &region);
			FullBarrier(cb);
			gpu.Submit(cb);
			std::memcpy(dst, staging.mapped, n);
			dst += n;
			offset += n;
			size -= n;
		}
	}
};

} // namespace

int main(int argc, char** argv) {
	std::string              dir, spirv_path, save_spirv, save_output, reference;
	uint32_t                 repeats    = 20;
	uint64_t                 unordered  = 0;
	bool                     game_spirv = false;
	bool                     compile_only = false;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--repeats" && i + 1 < argc) repeats = std::strtoul(argv[++i], nullptr, 0);
		else if (a == "--game-spirv") game_spirv = true;
		else if (a == "--compile-only") compile_only = true;
		else if (a == "--spirv" && i + 1 < argc) spirv_path = argv[++i];
		else if (a == "--save-spirv" && i + 1 < argc) save_spirv = argv[++i];
		else if (a == "--dump-ir" && i + 1 < argc) g_dump_ir = argv[++i];
		else if (a == "--save-output" && i + 1 < argc) save_output = argv[++i];
		else if (a == "--reference" && i + 1 < argc) reference = argv[++i];
		else if (a == "--unordered" && i + 1 < argc) unordered = std::strtoull(argv[++i], nullptr, 0);
		else dir = a;
	}
	if (dir.empty()) Fail("usage: %s <capture-dir> [options]", argv[0]);
	while (dir.size() > 1 && dir.back() == '/') dir.pop_back();

	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions config;
	config.printf_direction = Config::LogDirection::Silent;
	Config::Load(config);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();

	auto capture = Load(dir);
	if (compile_only) {
		// Only the layouts, from the writes, are needed.
		capture.objs.clear();
		capture.ptes.clear();
		capture.images.clear();
		capture.views.clear();
		capture.subs.clear();
		capture.samplers.clear();
	}
	const auto parent  = dir.substr(0, dir.find_last_of('/'));
	char       name[64];
	std::snprintf(name, sizeof(name), "%016" PRIx64 "-%" PRIu64, capture.hash, capture.program);
	const auto game_path = parent + "/spv-" + name + ".spv";
	std::vector<uint32_t> game;
	{
		const auto bytes = ReadFile(game_path);
		game.resize(bytes.size() / 4);
		std::memcpy(game.data(), bytes.data(), game.size() * 4);
	}
	std::vector<uint32_t> chosen;
	if (!spirv_path.empty()) {
		const auto bytes = ReadFile(spirv_path);
		chosen.resize(bytes.size() / 4);
		std::memcpy(chosen.data(), bytes.data(), chosen.size() * 4);
	}
	const auto compiled = Recompile(parent + "/compile-" + name + ".bin",
	                                game_spirv ? &game : (spirv_path.empty() ? nullptr : &chosen));
	std::printf("shader %016" PRIx64 " frame %" PRIu64 ": %s SPIR-V, %s the game's\n", capture.hash,
	            capture.frame, game_spirv ? "game" : (spirv_path.empty() ? "recompiled" : "given"),
	            compiled.spirv == game ? "identical to" : "differs from");
	{
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
		std::string          messages;
		tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t&,
		                             const char* message) { messages += message + std::string("\n"); });
		spvtools::ValidatorOptions validator;
		validator.SetScalarBlockLayout(true);
		if (!tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator))
			Fail("invalid SPIR-V:\n%s", messages.c_str());
	}
	if (!save_spirv.empty()) {
		std::ofstream out(save_spirv, std::ios::binary);
		out.write(reinterpret_cast<const char*>(compiled.spirv.data()),
		          static_cast<std::streamsize>(compiled.spirv.size() * 4));
	}

	Gpu gpu;
	gpu.Init();
	auto&        device = gpu.device;
	const Mapped pre(compile_only ? std::string("/dev/null") : dir + "/pre.bin");
	const Mapped post(compile_only ? std::string("/dev/null") : dir + "/post.bin");

	// Buffers.
	std::vector<Gpu::Buf> bufs(capture.objs.size());
	Uploader              up(gpu);
	const auto            t_upload = std::chrono::steady_clock::now();
	for (size_t i = 0; i < capture.objs.size(); i++) {
		const auto& o = capture.objs[i];
		bufs[i]       = gpu.MakeBuffer(o.size);
		if (o.kind == 2) {
			up.cb.fillBuffer(bufs[i].buffer, 0, VK_WHOLE_SIZE, 0);
		} else {
			up.Buffer(bufs[i].buffer, 0, pre.data + o.pre, o.size);
		}
	}
	auto zero_page = gpu.MakeBuffer(64 * 1024);
	up.cb.fillBuffer(zero_page.buffer, 0, VK_WHOLE_SIZE, 0);
	{
		FullBarrier(up.cb);
		std::vector<uint64_t> entries;
		for (const auto& [page, target]: capture.ptes) {
			entries.push_back(target.first < 0 ? zero_page.address
			                                   : bufs[target.first].address + target.second);
		}
		// Contiguous pages go in one copy.
		for (size_t i = 0; i < capture.ptes.size();) {
			size_t j = i + 1;
			while (j < capture.ptes.size() && capture.ptes[j].first == capture.ptes[j - 1].first + 1) j++;
			up.Buffer(bufs[0].buffer, capture.ptes[i].first * 8,
			          reinterpret_cast<const uint8_t*>(entries.data() + i), (j - i) * 8);
			i = j;
		}
	}
	// Images.
	struct ImgObj {
		vk::Image image;
		vk::Format format;
	};
	std::vector<ImgObj> images;
	for (const auto& m: capture.images) {
		vk::ImageCreateInfo info {};
		info.flags         = static_cast<vk::ImageCreateFlags>(m.flags);
		info.imageType     = static_cast<vk::ImageType>(m.type);
		info.format        = static_cast<vk::Format>(m.format);
		info.extent        = vk::Extent3D {m.w, m.h, m.d};
		info.mipLevels     = m.mips;
		info.arrayLayers   = m.layers;
		info.samples       = static_cast<vk::SampleCountFlagBits>(m.samples);
		info.tiling        = vk::ImageTiling::eOptimal;
		const auto known   = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
		                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
		                   vk::ImageUsageFlagBits::eColorAttachment |
		                   vk::ImageUsageFlagBits::eDepthStencilAttachment;
		info.usage = (static_cast<vk::ImageUsageFlags>(m.usage) & known) |
		             vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
		info.initialLayout = vk::ImageLayout::eUndefined;
		vk::Image image;
		Check(device.createImage(&info, nullptr, &image), "vkCreateImage");
		vk::MemoryRequirements req {};
		device.getImageMemoryRequirements(image, &req);
		const auto [mem, off] = gpu.Allocate(req, vk::MemoryPropertyFlagBits::eDeviceLocal);
		Check(device.bindImageMemory(image, mem, off), "vkBindImageMemory");
		vk::ImageMemoryBarrier to {};
		to.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		to.oldLayout           = vk::ImageLayout::eUndefined;
		to.newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to.srcQueueFamilyIndex = to.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to.image               = image;
		to.subresourceRange    = {FullAspect(info.format), 0, VK_REMAINING_MIP_LEVELS, 0,
		                          VK_REMAINING_ARRAY_LAYERS};
		up.cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
		                      {}, 0, nullptr, 0, nullptr, 1, &to);
		images.push_back({image, info.format});
	}
	const auto upload_subs = [&](Uploader& u, bool only_written) {
		for (const auto& s: capture.subs) {
			if (only_written && s.post < 0) continue;
			vk::BufferImageCopy where {};
			where.imageSubresource = {static_cast<vk::ImageAspectFlagBits>(s.aspect), s.mip, s.layer, 1};
			where.imageExtent      = vk::Extent3D {s.w, s.h, s.d};
			u.Image(images[s.image].image, where, pre.data + s.pre, s.bytes);
		}
	};
	const auto to_layouts = [&](vk::CommandBuffer cb, bool from_transfer, bool only_written) {
		for (const auto& s: capture.subs) {
			if (only_written && s.post < 0) continue;
			vk::ImageMemoryBarrier b {};
			b.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
			b.dstAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			b.oldLayout           = from_transfer ? vk::ImageLayout::eTransferDstOptimal
			                                      : static_cast<vk::ImageLayout>(s.layout);
			b.newLayout           = from_transfer ? static_cast<vk::ImageLayout>(s.layout)
			                                      : vk::ImageLayout::eTransferDstOptimal;
			b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			b.image               = images[s.image].image;
			b.subresourceRange    = {FullAspect(images[s.image].format), s.mip, 1, s.layer, 1};
			if (b.oldLayout == b.newLayout) continue;
			cb.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                   vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 0, nullptr, 1, &b);
		}
	};
	upload_subs(up, false);
	to_layouts(up.cb, true, false);
	up.Flush();
	std::printf("uploaded %zu buffers, %zu images in %.1f s\n", bufs.size(), images.size(),
	            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_upload).count());

	// Views and samplers.
	std::vector<vk::ImageView> views;
	for (const auto& v: capture.views) {
		vk::ImageViewUsageCreateInfo usage {};
		usage.usage = static_cast<vk::ImageUsageFlags>(v.usage) &
		              (vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
		               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
		               vk::ImageUsageFlagBits::eColorAttachment |
		               vk::ImageUsageFlagBits::eDepthStencilAttachment);
		vk::ImageViewMinLodCreateInfoEXT min_lod {};
		if (v.min_lod >= 0) {
			min_lod.minLod = v.min_lod;
			usage.pNext    = &min_lod;
		}
		vk::ImageViewCreateInfo info {};
		info.pNext            = &usage;
		info.image            = images.at(v.image).image;
		info.viewType         = static_cast<vk::ImageViewType>(v.type);
		info.format           = static_cast<vk::Format>(v.format);
		info.components       = vk::ComponentMapping {static_cast<vk::ComponentSwizzle>(v.r),
		                                              static_cast<vk::ComponentSwizzle>(v.g),
		                                              static_cast<vk::ComponentSwizzle>(v.b),
		                                              static_cast<vk::ComponentSwizzle>(v.a)};
		info.subresourceRange = {static_cast<vk::ImageAspectFlags>(v.aspect), v.base_mip, v.mips,
		                         v.base_layer, v.layers};
		vk::ImageView view;
		Check(device.createImageView(&info, nullptr, &view), "vkCreateImageView");
		views.push_back(view);
	}
	std::vector<vk::Sampler> samplers;
	for (const auto& s: capture.samplers) {
		vk::Sampler sampler;
		Check(device.createSampler(&s, nullptr, &sampler), "vkCreateSampler");
		samplers.push_back(sampler);
	}

	// Layouts: set 0 from the writes, set 1 as the bindless heap's.
	namespace Bindless = ShaderRecompiler::IR::Bindless;
	std::vector<vk::DescriptorSetLayoutBinding> set0;
	bool                                        uses_set1 = false;
	uint64_t                                    set0_count = 0;
	for (const auto& w: capture.writes) {
		if (w.set != 0) {
			uses_set1 = true;
			continue;
		}
		set0.push_back({w.binding, static_cast<vk::DescriptorType>(w.type),
		                static_cast<uint32_t>(w.elems.size()), vk::ShaderStageFlagBits::eCompute, nullptr});
		set0_count += w.elems.size();
	}
	const bool push_descriptors = set0_count <= gpu.max_push;
	vk::DescriptorSetLayoutCreateInfo layout0_info {};
	layout0_info.flags        = push_descriptors ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                                             : vk::DescriptorSetLayoutCreateFlags {};
	layout0_info.bindingCount = static_cast<uint32_t>(set0.size());
	layout0_info.pBindings    = set0.data();
	std::array<vk::DescriptorSetLayout, 2> layouts {};
	Check(device.createDescriptorSetLayout(&layout0_info, nullptr, &layouts[0]), "set 0 layout");
	{
		std::vector<vk::DescriptorSetLayoutBinding> bindings;
		std::vector<vk::DescriptorBindingFlags>     flags;
		for (const auto binding: {Bindless::TableBinding, Bindless::MissBinding}) {
			bindings.push_back({binding, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll,
			                    nullptr});
			flags.emplace_back();
		}
		for (uint32_t c = 0; c < Bindless::ClassCount; c++) {
			bindings.push_back({Bindless::FirstImageBinding + c, vk::DescriptorType::eSampledImage,
			                    Bindless::HeapSlots, vk::ShaderStageFlagBits::eAll, nullptr});
			flags.emplace_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
		}
		for (const auto binding: {Bindless::SamplerBinding, Bindless::CompareSamplerBinding}) {
			bindings.push_back({binding, vk::DescriptorType::eSampler, Bindless::HeapSlots,
			                    vk::ShaderStageFlagBits::eAll, nullptr});
			flags.emplace_back(vk::DescriptorBindingFlagBits::ePartiallyBound);
		}
		vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
		flags_info.bindingCount  = static_cast<uint32_t>(flags.size());
		flags_info.pBindingFlags = flags.data();
		vk::DescriptorSetLayoutCreateInfo info {};
		info.pNext        = &flags_info;
		info.bindingCount = static_cast<uint32_t>(bindings.size());
		info.pBindings    = bindings.data();
		Check(device.createDescriptorSetLayout(&info, nullptr, &layouts[1]), "set 1 layout");
	}
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        ShaderRecompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = uses_set1 ? 2u : 1u;
	pipeline_layout_info.pSetLayouts            = layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	vk::PipelineLayout pipeline_layout;
	Check(device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout), "pipeline layout");

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = compiled.spirv.size() * 4;
	module_info.pCode    = compiled.spirv.data();
	vk::ShaderModule module;
	Check(device.createShaderModule(&module_info, nullptr, &module), "shader module");
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {};
	vk::ComputePipelineCreateInfo                         pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	if (compiled.wave_size >= gpu.min_subgroup && compiled.wave_size <= gpu.max_subgroup) {
		subgroup.requiredSubgroupSize = compiled.wave_size;
		pipeline_info.stage.pNext     = &subgroup;
	}
	pipeline_info.layout = pipeline_layout;
	vk::Pipeline pipeline;
	const auto   t_pipeline = std::chrono::steady_clock::now();
	Check(device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline), "pipeline");
	std::printf("pipeline built in %.1f ms (wave%u)\n",
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pipeline).count(),
	            compiled.wave_size);
	if (compile_only) {
		std::fflush(stdout);
		std::_Exit(0);
	}

	// Descriptor writes.
	std::vector<std::vector<vk::DescriptorBufferInfo>> buffer_infos;
	std::vector<std::vector<vk::DescriptorImageInfo>>  image_infos;
	std::array<std::vector<vk::WriteDescriptorSet>, 2> writes;
	for (const auto& w: capture.writes) {
		vk::WriteDescriptorSet write {};
		write.dstBinding      = w.binding;
		write.descriptorType  = static_cast<vk::DescriptorType>(w.type);
		write.descriptorCount = static_cast<uint32_t>(w.elems.size());
		const bool buffer     = write.descriptorType == vk::DescriptorType::eStorageBuffer ||
		                    write.descriptorType == vk::DescriptorType::eUniformBuffer;
		if (buffer) {
			auto& infos = buffer_infos.emplace_back();
			for (const auto& e: w.elems) {
				if (e.a < 0) infos.push_back({nullptr, 0, VK_WHOLE_SIZE});
				else infos.push_back({bufs[e.a].buffer, e.offset, e.range});
			}
			write.pBufferInfo = infos.data();
		} else {
			auto& infos = image_infos.emplace_back();
			for (const auto& e: w.elems) {
				infos.push_back({e.a < 0 ? vk::Sampler {} : samplers[e.a],
				                 e.b < 0 ? vk::ImageView {} : views[e.b],
				                 static_cast<vk::ImageLayout>(e.layout)});
			}
			write.pImageInfo = infos.data();
		}
		writes[w.set == 0 ? 0 : 1].push_back(write);
	}
	vk::DescriptorPool pool;
	vk::DescriptorSet  set0_set, set1_set;
	{
		const std::array sizes {
		    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 256},
		    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 64},
		    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, Bindless::ClassCount * Bindless::HeapSlots + 256},
		    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 256},
		    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 2 * Bindless::HeapSlots + 64},
		};
		vk::DescriptorPoolCreateInfo info {};
		info.maxSets       = 2;
		info.poolSizeCount = static_cast<uint32_t>(sizes.size());
		info.pPoolSizes    = sizes.data();
		Check(device.createDescriptorPool(&info, nullptr, &pool), "descriptor pool");
		const std::array<vk::DescriptorSetLayout, 2> alloc {layouts[0], layouts[1]};
		vk::DescriptorSetAllocateInfo allocate {};
		allocate.descriptorPool     = pool;
		allocate.descriptorSetCount = push_descriptors ? 1 : 2;
		allocate.pSetLayouts        = push_descriptors ? &alloc[1] : alloc.data();
		std::array<vk::DescriptorSet, 2> sets {};
		Check(device.allocateDescriptorSets(&allocate, sets.data()), "descriptor sets");
		if (push_descriptors) {
			set1_set = sets[0];
		} else {
			set0_set = sets[0];
			set1_set = sets[1];
			for (auto& w: writes[0]) w.dstSet = set0_set;
			device.updateDescriptorSets(static_cast<uint32_t>(writes[0].size()), writes[0].data(), 0, nullptr);
		}
		for (auto& w: writes[1]) w.dstSet = set1_set;
		if (!writes[1].empty())
			device.updateDescriptorSets(static_cast<uint32_t>(writes[1].size()), writes[1].data(), 0, nullptr);
	}

	// What the dispatch writes: restored before each run from pristine copies.
	std::vector<size_t> written;
	for (size_t i = 0; i < capture.objs.size(); i++)
		if (!capture.objs[i].diffs.empty() || capture.objs[i].kind == 1) written.push_back(i);
	std::vector<Gpu::Buf> pristine;
	{
		auto cb = gpu.Begin();
		for (const auto i: written) {
			pristine.push_back(gpu.MakeBuffer(capture.objs[i].size));
			vk::BufferCopy region {0, 0, capture.objs[i].size};
			cb.copyBuffer(bufs[i].buffer, pristine.back().buffer, 1, &region);
		}
		gpu.Submit(cb);
	}
	vk::QueryPool        queries;
	vk::QueryPoolCreateInfo query_info {};
	query_info.queryType  = vk::QueryType::eTimestamp;
	query_info.queryCount = 2;
	Check(device.createQueryPool(&query_info, nullptr, &queries), "query pool");

	const auto run = [&]() {
		// Restore.
		Uploader u(gpu);
		for (size_t k = 0; k < written.size(); k++) {
			vk::BufferCopy region {0, 0, capture.objs[written[k]].size};
			u.cb.copyBuffer(pristine[k].buffer, bufs[written[k]].buffer, 1, &region);
		}
		to_layouts(u.cb, false, true);
		upload_subs(u, true);
		to_layouts(u.cb, true, true);
		FullBarrier(u.cb);
		u.cb.resetQueryPool(queries, 0, 2);
		u.cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		if (push_descriptors) {
			u.cb.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, pipeline_layout, 0,
			                          static_cast<uint32_t>(writes[0].size()), writes[0].data());
		} else {
			u.cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &set0_set, 0,
			                        nullptr);
		}
		if (uses_set1)
			u.cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, Bindless::DescriptorSet,
			                        1, &set1_set, 0, nullptr);
		std::array<uint32_t, ShaderRecompiler::IR::PushData::DwordCount> push {};
		std::copy_n(capture.push.begin(), std::min(capture.push.size(), push.size()), push.begin());
		u.cb.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
		u.cb.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queries, 0);
		u.cb.dispatch(capture.groups[0], capture.groups[1], capture.groups[2]);
		u.cb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queries, 1);
		FullBarrier(u.cb);
		gpu.Submit(u.cb);
		u.cb = nullptr;
		std::array<uint64_t, 2> stamps {};
		Check(device.getQueryPoolResults(queries, 0, 2, sizeof(stamps), stamps.data(), 8,
		                                 vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
		      "timestamps");
		return static_cast<double>(stamps[1] - stamps[0]) * gpu.properties.limits.timestampPeriod * 1e-6;
	};

	// First run: outputs against the capture.
	const double first = run();
	{
		Downloader down(gpu);
		std::vector<uint8_t> actual, expected, saved;
		uint64_t              mismatched = 0, compared = 0;
		std::vector<uint8_t>  ref;
		if (!reference.empty()) ref = ReadFile(reference);
		uint64_t ref_at = 0;
		for (const auto i: written) {
			const auto& o = capture.objs[i];
			actual.resize(o.size);
			down.Buffer(bufs[i].buffer, 0, actual.data(), o.size);
			if (!reference.empty()) {
				expected.assign(ref.begin() + static_cast<ptrdiff_t>(ref_at),
				                ref.begin() + static_cast<ptrdiff_t>(ref_at + o.size));
				ref_at += o.size;
			} else {
				expected.assign(pre.data + o.pre, pre.data + o.pre + o.size);
				for (const auto& [off, size, at]: o.diffs) std::memcpy(expected.data() + off, post.data + at, size);
			}
			if (!save_output.empty()) saved.insert(saved.end(), actual.begin(), actual.end());
			uint64_t bad = 0, first_bad = UINT64_MAX;
			for (uint64_t k = 0; k < o.size; k++) {
				if (actual[k] != expected[k]) {
					bad++;
					first_bad = std::min(first_bad, k);
				}
			}
			if (bad != 0 && unordered != 0) {
				const auto records = [&](const std::vector<uint8_t>& bytes) {
					std::vector<std::string> list;
					for (uint64_t k = 0; k + unordered <= bytes.size(); k += unordered)
						list.emplace_back(reinterpret_cast<const char*>(bytes.data() + k), unordered);
					std::sort(list.begin(), list.end());
					return list;
				};
				if (records(actual) == records(expected)) {
					std::printf("  object %zu: same %" PRIu64 "-byte records in another order\n", i,
					            unordered);
					bad = 0;
				}
			}
			compared += o.size;
			if (bad != 0) {
				std::printf("  object %zu (kind %d, %" PRIu64 " bytes): %" PRIu64
				            " bytes differ, first at %" PRIu64 "\n",
				            i, o.kind, o.size, bad, first_bad);
			}
			mismatched += bad;
		}
		// Written images: read back through a buffer.
		for (size_t si = 0; si < capture.subs.size(); si++) {
			const auto& s = capture.subs[si];
			const bool storage = std::any_of(capture.writes.begin(), capture.writes.end(), [&](const auto& w) {
				return w.type == static_cast<int>(vk::DescriptorType::eStorageImage) &&
				       std::any_of(w.elems.begin(), w.elems.end(), [&](const Elem& e) {
					       return e.b >= 0 && capture.views[e.b].image == s.image;
				       });
			});
			if (!storage) continue;
			auto tmp = gpu.MakeBuffer(s.bytes);
			auto cb  = gpu.Begin();
			vk::BufferImageCopy where {};
			where.imageSubresource = {static_cast<vk::ImageAspectFlagBits>(s.aspect), s.mip, s.layer, 1};
			where.imageExtent      = vk::Extent3D {s.w, s.h, s.d};
			FullBarrier(cb);
			cb.copyImageToBuffer(images[s.image].image, static_cast<vk::ImageLayout>(s.layout) ==
			                         vk::ImageLayout::eGeneral ? vk::ImageLayout::eGeneral
			                                                   : vk::ImageLayout::eGeneral,
			                     tmp.buffer, 1, &where);
			gpu.Submit(cb);
			actual.resize(s.bytes);
			down.Buffer(tmp.buffer, 0, actual.data(), s.bytes);
			const uint8_t* want = s.post >= 0 ? post.data + s.post : pre.data + s.pre;
			if (!reference.empty()) {
				want = ref.data() + ref_at;
				ref_at += s.bytes;
			}
			if (!save_output.empty()) saved.insert(saved.end(), actual.begin(), actual.end());
			uint64_t bad = 0;
			for (uint64_t k = 0; k < s.bytes; k++) bad += actual[k] != want[k];
			compared += s.bytes;
			if (bad != 0) std::printf("  storage image %zu: %" PRIu64 " bytes differ\n", s.image, bad);
			mismatched += bad;
		}
		std::printf("output: %s (%" PRIu64 " of %" PRIu64 " bytes differ from the %s)\n",
		            mismatched == 0 ? "MATCH" : "MISMATCH", mismatched, compared,
		            reference.empty() ? "capture" : "reference");
		if (!save_output.empty()) {
			std::ofstream out(save_output, std::ios::binary);
			out.write(reinterpret_cast<const char*>(saved.data()), static_cast<std::streamsize>(saved.size()));
		}
	}
	std::vector<double> times {first};
	for (uint32_t i = 1; i < repeats; i++) times.push_back(run());
	std::sort(times.begin(), times.end());
	std::printf("gpu time over %zu runs: min %.3f ms, median %.3f ms, max %.3f ms\n", times.size(),
	            times.front(), times[times.size() / 2], times.back());
	std::fflush(stdout);
	std::_Exit(0);
}
