#ifndef KYTY_COMMON_VIRTUALMEMORY_H_
#define KYTY_COMMON_VIRTUALMEMORY_H_

#include "common/common.h"

namespace Common {

namespace VirtualMemory {

enum class Mode : uint32_t {
	NoAccess         = 0,
	Read             = 1,
	Write            = 2,
	ReadWrite        = Read | Write,
	Execute          = 4,
	ExecuteRead      = Execute | Read,
	ExecuteWrite     = Execute | Write,
	ExecuteReadWrite = Execute | Read | Write,
};

inline bool IsExecute(Mode mode) {
	return (mode == Mode::Execute || mode == Mode::ExecuteRead || mode == Mode::ExecuteWrite ||
	        mode == Mode::ExecuteReadWrite);
}

void Init();

uint64_t Alloc(uint64_t address, uint64_t size, Mode mode);
uint64_t AllocAligned(uint64_t address, uint64_t size, Mode mode, uint64_t alignment);
bool     AllocFixed(uint64_t address, uint64_t size, Mode mode);
bool     Commit(uint64_t address, uint64_t size, Mode mode);
uint64_t Reserve(uint64_t address, uint64_t size);
uint64_t ReserveAligned(uint64_t address, uint64_t size, uint64_t alignment);
bool     ReserveFixed(uint64_t address, uint64_t size);
bool     Decommit(uint64_t address, uint64_t size);
bool     Free(uint64_t address);
bool     FreeRange(uint64_t address, uint64_t size);
bool     Protect(uint64_t address, uint64_t size, Mode mode);
bool     FlushInstructionCache(uint64_t address, uint64_t size);

// Write protection without a protection change (Linux userfaultfd, synchronous, SIGBUS): a write
// to a write-protected page of a registered range raises SIGBUS on the writing thread, decoded as
// an access-violation write, and is retried once the page is unprotected. Arming and disarming take
// the address space's lock for reading only and never split a mapping, unlike Protect.
// Whether the host supports it; checked once (a capability, not a setting).
bool WriteProtectSupported();
// Registers a mapped range for write protection; its pages start unprotected. A mapping that
// replaces part of the range drops that part's registration.
bool RegisterWriteProtect(uint64_t address, uint64_t size);
// Arms (protect) or disarms the range's pages, which must all be registered: false when some are
// not (pages before the first unregistered one may have changed).
bool WriteProtect(uint64_t address, uint64_t size, bool protect);
// A write-tracking protection change: mode is NoAccess, (Execute)Read for a write watch, or
// (Execute)ReadWrite. On a registered range a write watch is armed write protection, armed before
// the pages turn accessible, and only the NoAccess edges change the protection: from_no_access
// says part of the range may be NoAccess now (over-approximating costs a protection change).
// Elsewhere, or if arming fails (a range remapped since it was registered), the protection is
// set to mode: the same range releases its watch the same way.
bool ProtectForTracking(uint64_t address, uint64_t size, Mode mode, bool from_no_access,
                        bool registered);

} // namespace VirtualMemory

} // namespace Common

#endif /* KYTY_COMMON_VIRTUALMEMORY_H_ */
