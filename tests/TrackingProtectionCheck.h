#ifndef KYTY_TESTS_TRACKINGPROTECTIONCHECK_H_
#define KYTY_TESTS_TRACKINGPROTECTIONCHECK_H_

// Shared by the page-manager and memory-tracker tests' ProtectGuestHostMemory stubs.
//
// Checks the page manager's from_no_access claim against the protection it applied before: a
// change that says no page of its range is NoAccess must not find one. And, with write protection
// on (Enable, where the host supports it), applies changes as the guest address space does
// (VirtualMemory::ProtectForTracking over a registered range), so a write watch arms write
// protection and leaves the pages writable.

#include "common/virtualMemory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace TrackingProtectionCheck {

using Mode = Common::VirtualMemory::Mode;

inline std::mutex                   g_mutex;
inline std::map<uint64_t, Mode>     g_applied; // page -> protection last applied
inline std::map<uint64_t, uint64_t> g_registered; // begin -> end
inline bool                         g_write_protect = false;

constexpr uint64_t PageSize = 4096;

// Records a change, after checking from_no_access. Returns whether the range is registered for
// write protection (all of it).
inline bool Note(uint64_t vaddr, uint64_t size, Mode mode, bool from_no_access) {
	std::lock_guard lock(g_mutex);
	for (auto page = vaddr & ~(PageSize - 1); page < vaddr + size; page += PageSize) {
		auto it = g_applied.find(page);
		if (!from_no_access && it != g_applied.end() && it->second == Mode::NoAccess) {
			std::fprintf(stderr,
			             "TrackingProtectionCheck: change claims no NoAccess page, page 0x%llx is\n",
			             static_cast<unsigned long long>(page));
			std::abort();
		}
		g_applied[page] = mode;
	}
	auto it = g_registered.upper_bound(vaddr);
	if (it == g_registered.begin()) {
		return false;
	}
	--it;
	return it->first <= vaddr && vaddr + size <= it->second;
}

// The range's pages are writable and unwatched again (a new allocation).
inline void Forget(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(g_mutex);
	g_applied.erase(g_applied.lower_bound(vaddr), g_applied.lower_bound(vaddr + size));
	for (auto it = g_registered.begin(); it != g_registered.end();) {
		if (it->first >= vaddr && it->second <= vaddr + size) {
			it = g_registered.erase(it);
		} else {
			++it;
		}
	}
}

// With write protection on, registers a new allocation for it (as GuestAddressSpace does).
inline void NoteAllocation(uint64_t vaddr, uint64_t size) {
	Forget(vaddr, size);
	if (g_write_protect && Common::VirtualMemory::RegisterWriteProtect(vaddr, size)) {
		std::lock_guard lock(g_mutex);
		g_registered[vaddr] = vaddr + size;
	}
}

// Turns write protection on for the allocations made from now on. False where the host lacks it.
inline bool Enable(bool enable) {
	g_write_protect = enable && Common::VirtualMemory::WriteProtectSupported();
	return g_write_protect;
}

// The ProtectGuestHostMemory stub's change on a registered range.
inline bool ApplyWriteProtected(uint64_t vaddr, uint64_t size, Mode mode, bool from_no_access) {
	return Common::VirtualMemory::ProtectForTracking(vaddr, size, mode, from_no_access, true);
}

// Whether the page is populated and write-protected (/proc/self/pagemap bit 57, PM_UFFD_WP).
inline bool IsWriteProtected(const void* address) {
#if defined(__linux__)
	const int fd = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
	if (fd < 0) {
		return false;
	}
	uint64_t   entry  = 0;
	const auto offset = static_cast<off_t>(reinterpret_cast<uintptr_t>(address) / PageSize * 8);
	const bool read   = ::pread(fd, &entry, sizeof(entry), offset) == sizeof(entry);
	::close(fd);
	return read && (entry & (uint64_t {1} << 57u)) != 0;
#else
	(void)address;
	return false;
#endif
}

} // namespace TrackingProtectionCheck

#endif // KYTY_TESTS_TRACKINGPROTECTIONCHECK_H_
