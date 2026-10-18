#include "common/common.h"
#include "common/frameProfiler.h"
#include "common/virtualMemory.h"

#if defined(__linux__)

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <mutex>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace Common::VirtualMemory {

namespace {

constexpr uint64_t PageSize = 4096;
// SIGBUS instead of a fault message; write protection of shared memory (the memfd views) and of
// pages not yet populated (a store to one must fault too).
constexpr uint64_t RequiredFeatures =
    UFFD_FEATURE_SIGBUS | UFFD_FEATURE_WP_HUGETLBFS_SHMEM | UFFD_FEATURE_WP_UNPOPULATED;

int g_uffd = -1;

bool Arm(int uffd, uint64_t address, uint64_t size, bool protect) {
	uffdio_writeprotect request {};
	request.range.start = address;
	request.range.len   = size;
	request.mode        = protect ? UFFDIO_WRITEPROTECT_MODE_WP : 0;
	int result          = 0;
	do {
		result = ::ioctl(uffd, UFFDIO_WRITEPROTECT, &request);
	} while (result != 0 && (errno == EAGAIN || errno == EINTR));
	return result == 0;
}

bool Register(int uffd, uint64_t address, uint64_t size) {
	uffdio_register request {};
	request.range.start = address;
	request.range.len   = size;
	request.mode        = UFFDIO_REGISTER_MODE_WP;
	return ::ioctl(uffd, UFFDIO_REGISTER, &request) == 0 &&
	       (request.ioctls & (uint64_t {1} << _UFFDIO_WRITEPROTECT)) != 0;
}

// Registers, arms and disarms a page of each kind of guest memory: an anonymous private page and
// a shared memfd page, neither populated.
bool SelfTest(int uffd) {
	bool  ok        = true;
	void* anonymous = ::mmap(nullptr, PageSize, PROT_READ | PROT_WRITE,
	                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (anonymous == MAP_FAILED) {
		return false;
	}
	const auto anonymous_address = reinterpret_cast<uint64_t>(anonymous);
	ok = Register(uffd, anonymous_address, PageSize) && Arm(uffd, anonymous_address, PageSize, true) &&
	     Arm(uffd, anonymous_address, PageSize, false);
	::munmap(anonymous, PageSize);
	const int fd = static_cast<int>(::syscall(SYS_memfd_create, "KytyWriteProtectProbe", 0));
	if (fd < 0) {
		return false;
	}
	void* shared = ok && ::ftruncate(fd, PageSize) == 0
	                   ? ::mmap(nullptr, PageSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
	                   : MAP_FAILED;
	::close(fd);
	if (shared == MAP_FAILED) {
		return false;
	}
	const auto shared_address = reinterpret_cast<uint64_t>(shared);
	ok = Register(uffd, shared_address, PageSize) && Arm(uffd, shared_address, PageSize, true) &&
	     Arm(uffd, shared_address, PageSize, false);
	::munmap(shared, PageSize);
	return ok;
}

int Open() {
	// Unprivileged processes may only handle user-mode faults; a kernel-mode access to an armed
	// page fails with EFAULT, as it does on a read-only page.
	const int uffd = static_cast<int>(
	    ::syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
	if (uffd < 0) {
		return -1;
	}
	uffdio_api api {};
	api.api      = UFFD_API;
	api.features = RequiredFeatures;
	if (::ioctl(uffd, UFFDIO_API, &api) != 0 || (api.features & RequiredFeatures) != RequiredFeatures ||
	    (api.ioctls & (uint64_t {1} << _UFFDIO_REGISTER)) == 0 || !SelfTest(uffd)) {
		::close(uffd);
		return -1;
	}
	return uffd;
}

int Uffd() {
	static std::once_flag once;
	std::call_once(once, [] { g_uffd = Open(); });
	return g_uffd;
}

} // namespace

bool WriteProtectSupported() {
	return Uffd() >= 0;
}

bool RegisterWriteProtect(uint64_t address, uint64_t size) {
	const int uffd = Uffd();
	return uffd >= 0 && size != 0 && Register(uffd, address, size);
}

bool WriteProtect(uint64_t address, uint64_t size, bool protect) {
	const int uffd = Uffd();
	if (uffd < 0 || size == 0) {
		return false;
	}
	const auto begin = address & ~(PageSize - 1);
	const auto end   = (address + size + PageSize - 1) & ~(PageSize - 1);
	return Arm(uffd, begin, end - begin, protect);
}

} // namespace Common::VirtualMemory

#else

namespace Common::VirtualMemory {

bool WriteProtectSupported() {
	return false;
}

bool RegisterWriteProtect(uint64_t /*address*/, uint64_t /*size*/) {
	return false;
}

bool WriteProtect(uint64_t /*address*/, uint64_t /*size*/, bool /*protect*/) {
	return false;
}

} // namespace Common::VirtualMemory

#endif

namespace Common::VirtualMemory {

namespace {

bool ProtectCounted(uint64_t address, uint64_t size, Mode mode) {
	const uint64_t t0 = FrameProfiler::Start();
	const bool     ok = Protect(address, size, mode);
	if (t0 != 0) {
		const auto role = FrameProfiler::CurrentRole();
		FrameProfiler::Add(FrameProfiler::Counter::TrackingMprotectCallsRender, role);
		FrameProfiler::Add(FrameProfiler::Counter::TrackingMprotectTimeRender, role,
		                   FrameProfiler::Ticks() - t0);
	}
	return ok;
}

} // namespace

bool ProtectForTracking(uint64_t address, uint64_t size, Mode mode, bool from_no_access,
                        bool registered) {
	if (!registered || mode == Mode::NoAccess) {
		// A NoAccess page keeps its write protection: the way back sets it.
		return ProtectCounted(address, size, mode);
	}
	const bool     watch = mode == Mode::Read || mode == Mode::ExecuteRead;
	const uint64_t t0    = FrameProfiler::Start();
	const bool     armed = WriteProtect(address, size, watch);
	if (t0 != 0) {
		const auto role = FrameProfiler::CurrentRole();
		FrameProfiler::Add(FrameProfiler::Counter::TrackingWriteProtectCallsRender, role);
		FrameProfiler::Add(FrameProfiler::Counter::TrackingWriteProtectTimeRender, role,
		                   FrameProfiler::Ticks() - t0);
	}
	if (!armed) {
		return ProtectCounted(address, size, mode);
	}
	return !from_no_access ||
	       ProtectCounted(address, size, IsExecute(mode) ? Mode::ExecuteReadWrite : Mode::ReadWrite);
}

} // namespace Common::VirtualMemory
