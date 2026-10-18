#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <semaphore>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>
#include <csignal>
#include <limits.h>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
#if defined(__APPLE__)
  mach_vm_address_t region_address =
      reinterpret_cast<mach_vm_address_t>(address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  Check(mach_vm_region(mach_task_self(), &region_address, &region_size,
                       VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &info_count,
                       &object_name) == KERN_SUCCESS,
        "mach_vm_region failed");
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  return (info.protection & VM_PROT_WRITE) != 0
             ? PAGE_READWRITE
             : (info.protection & VM_PROT_READ) != 0 ? PAGE_READONLY
                                                     : PAGE_NOACCESS;
#else
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
#endif
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
#if defined(__APPLE__)
  mach_vm_address_t raw_address = reinterpret_cast<mach_vm_address_t>(address);
  const auto flags = address != nullptr ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
  if (mach_vm_allocate(mach_task_self(), &raw_address, size, flags) !=
      KERN_SUCCESS) {
    return nullptr;
  }
  if (mach_vm_protect(mach_task_self(), raw_address, size, false,
                      static_cast<vm_prot_t>(ToHostProt(protection))) !=
      KERN_SUCCESS) {
    mach_vm_deallocate(mach_task_self(), raw_address, size);
    return nullptr;
  }
  void *raw = reinterpret_cast<void *>(raw_address);
  AllocationSizes()[raw] = size;
  return raw;
#else
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
#endif
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
#if defined(__APPLE__)
  const int ok = mach_vm_deallocate(mach_task_self(),
                                    reinterpret_cast<mach_vm_address_t>(address),
                                    it->second) == KERN_SUCCESS
                     ? 1
                     : 0;
#else
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
#endif
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
#if defined(__APPLE__)
  return mach_vm_protect(mach_task_self(),
                         reinterpret_cast<mach_vm_address_t>(address), size,
                         false, static_cast<vm_prot_t>(ToHostProt(protection))) ==
                 KERN_SUCCESS
             ? 1
             : 0;
#else
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
#endif
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;
std::mutex g_protection_log_mutex;

struct ProtectionGate {
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
};
ProtectionGate *g_protection_gate = nullptr;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  if (g_protection_gate != nullptr && mode == Common::VirtualMemory::Mode::ReadWrite) {
    g_protection_gate->entered.release();
    g_protection_gate->resume.acquire();
  }
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  {
    std::lock_guard lock(g_protection_log_mutex);
    g_protection_calls++;
    g_protection_log.push_back({vaddr, size, mode});
  }
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *AllocateFixedGuestRange(uint64_t size, uintptr_t offset) {
  // Keep the mapping inside the tracker's guest address space and preserve the
  // requested region alignment. A fixed address can be occupied by the host
  // process (notably by the macOS runner's ASLR layout).
  constexpr uintptr_t first_base = 0x0000000200000000ull;
  constexpr uintptr_t stride = 0x0000000100000000ull;
  for (uintptr_t attempt = 0; attempt < 256; attempt++) {
    auto *wanted = reinterpret_cast<void *>(first_base + offset + attempt * stride);
    auto *memory = static_cast<uint8_t *>(
        VirtualAlloc(wanted, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (memory == wanted) {
      return memory;
    }
    if (memory != nullptr) {
      Check(VirtualFree(memory, 0, MEM_RELEASE) != 0,
            "release unexpected fixed allocation failed");
    }
  }
  Check(false, "no free fixed guest address found");
  return nullptr;
}

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  const auto size = manager.GetPageSize() * pages;
  return AllocateFixedGuestRange(size, 0x10000);
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr uint64_t extended_end = Libs::LibKernel::Memory::kExtendedMemoryBase +
                                    Libs::LibKernel::Memory::kExtendedMemorySize;
  constexpr GuestRange last_byte{extended_end - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == extended_end);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase, 1}.Valid() &&
            !GuestRange{Libs::Graphics::LOWER_ADDRESS_SIZE, 1}.Valid() &&
            !GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase - 1, 2}.Valid() &&
            !GuestRange{extended_end - 1, 2}.Valid(),
        "extended range and gap boundaries are enforced");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestBdaSyncDirtiness() {
  constexpr auto page = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.TakeBdaSync().Empty(), "new tracker queued unowned memory");
  tracker.QueueBdaSync(address, 3 * page);
  Check(tracker.IsRegionCpuModified(address, page), "new region was clean");
  auto pending = tracker.TakeBdaSync();
  Check(pending.Contains(address, 3 * page) && !pending.Intersects(address + 3 * page, page),
        "mapped or registered range did not bound initial BDA synchronization");
  tracker.ForEachUploadRange(address, 3 * page, false,
                             [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(tracker.TakeBdaSync().Empty(), "clean upload queued redundant BDA work");

  tracker.MarkRegionAsCpuModified(address + 17, 1);
  pending = tracker.TakeBdaSync();
  Check(pending.Contains(address, page) && !pending.Intersects(address + page, page) &&
            tracker.TakeBdaSync().Empty(),
        "byte dirtiness did not drain exactly its whole tracker page");
  // A write after Take belongs to the next drain, even when its page was just taken.
  tracker.MarkRegionAsCpuModified(address + 31, 1);
  Check(tracker.TakeBdaSync().Contains(address, page), "re-dirty event was lost");
  tracker.ForEachUploadRange(address, page, true,
                             [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(tracker.TakeBdaSync().Empty(), "GPU ownership queued an unnecessary upload");
  tracker.InvalidateRegion(address + 7, 1, [&] {
    tracker.UnmarkRegionAsGpuModified(address, page);
    tracker.MarkRegionAsCpuModified(address, page);
  });
  Check(tracker.TakeBdaSync().Contains(address, page),
        "GPU-to-CPU invalidation did not queue BDA synchronization");
  tracker.InvalidateRegion(address + page + 7, 1, [] {});
  Check(tracker.TakeBdaSync().Contains(address + page, page),
        "CPU invalidation did not queue BDA synchronization");
  tracker.UntrackMemory(address, 3 * page);
  Check(tracker.TakeBdaSync().Contains(address, 3 * page),
        "untracked memory did not queue its newly dirty pages");
  Release(memory);
}

// A take (TakeBdaSync) returns the set a hold queued into only once the hold is dropped; a hold
// queued after the take belongs to the next set and does not hold that take.
void TestBdaSyncTakeWaitsForHolds() {
  constexpr auto page = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  (void)tracker.TakeBdaSync();
  Check(!tracker.HasBdaSync(), "an empty queue reported pending BDA work");

  std::atomic<bool> taken{false};
  Libs::Graphics::RangeSet result;
  {
    auto hold = tracker.HoldBdaSync(address + 17, 1);
    Check(tracker.HasBdaSync(), "a hold did not queue its range");
    std::jthread taker([&] {
      result = tracker.TakeBdaSync();
      taken.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Check(!taken.load(std::memory_order_acquire), "a take returned before the hold it took");
    // Queued into the next set: the waiting take must not wait for it as well.
    auto later = tracker.HoldBdaSync(address + page, 1);
    { auto drop = std::move(hold); }
    taker.join();
    Check(taken.load(std::memory_order_acquire) && result.Contains(address, page) &&
              !result.Intersects(address + page, page),
          "a take did not return its held range alone");
  }
  const auto next = tracker.TakeBdaSync();
  Check(next.Contains(address + page, page) && !tracker.HasBdaSync(),
        "a hold queued after a take was lost");
  tracker.UntrackMemory(address, 2 * page);
  (void)tracker.TakeBdaSync();
  Release(memory);
}

void TestBdaSyncConcurrentDrain() {
  constexpr auto page = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, page, false,
                             [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  (void)tracker.TakeBdaSync();
  ProtectionGate gate;
  g_protection_gate = &gate;
  std::jthread writer([&] { tracker.MarkRegionAsCpuModified(address + 17, 1); });
  gate.entered.acquire();
  // Ownership is still changing. Publishing its event before this point would let
  // a drain consume it before the writer has finished releasing page protection.
  Check(tracker.TakeBdaSync().Empty(), "BDA event preceded its state transition");
  gate.resume.release();
  writer.join();
  g_protection_gate = nullptr;
  Check(tracker.TakeBdaSync().Contains(address, page) &&
            tracker.IsRegionCpuModified(address, page),
        "concurrent drain lost a completed CPU dirty transition");
  tracker.UntrackMemory(address, page);
  Release(memory);
}

// A lazy release (BufferCache's direct readback hand-back) leaves a GPU-written page unreadable
// while its watchers would let it be read: the GPU writing it again needs no protection change,
// and a fault applies what the watchers ask.
void TestLazyRelease() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});
  tracker.MarkRegionAsGpuModified(address, page_size);
  Check(Protection(memory) == PAGE_NOACCESS, "a GPU-written page stayed readable");

  const auto lazy_release = [&] {
    Check(!PageManager::SetLazyRelease(true), "lazy release was already set");
    tracker.UnmarkRegionAsGpuModified(address, page_size);
    Check(PageManager::SetLazyRelease(false), "lazy release changed");
  };
  lazy_release();
  Check(!tracker.IsRegionGpuModified(address, page_size) && Protection(memory) == PAGE_NOACCESS,
        "a lazy release changed the page's protection");
  tracker.MarkRegionAsGpuModified(address, page_size);
  Check(Protection(memory) == PAGE_NOACCESS && !page_manager.ApplyDeferred(address),
        "a GPU write after a lazy release left the page as left");

  lazy_release();
  Check(page_manager.ApplyDeferred(address) && Protection(memory) == PAGE_READONLY &&
            !page_manager.ApplyDeferred(address),
        "a fault after a lazy release did not apply the watchers' protection once");
  Check(Protection(memory + page_size) == PAGE_READONLY, "the neighbour page changed");

  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

// An upload walk with deferred protection (BufferCache's upload helper) leaves the pages it
// cleans writable until ApplyProtection, which protects them as their watchers now ask: a page
// another change made unreadable meanwhile stays so, and a page another watcher took was
// protected at once.
void TestDeferredUploadProtection() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto walk = [&](std::vector<Libs::Graphics::GuestRange> &deferred) {
    tracker.MarkRegionAsCpuModified(address, page_size * 3);
    Check(PageManager::SetDeferredWriteWatches(&deferred) == nullptr, "a sink was already set");
    uint64_t uploaded = 0;
    tracker.ForEachUploadRange(
        address, page_size * 3, false,
        [&](uint64_t, uint64_t size) noexcept { uploaded += size; }, []() noexcept {});
    Check(PageManager::SetDeferredWriteWatches(nullptr) == &deferred, "the sink changed");
    Check(uploaded == page_size * 3 && !tracker.IsRegionCpuModified(address, page_size * 3),
          "the deferred walk did not clean its pages");
  };
  const auto apply = [&](const std::vector<Libs::Graphics::GuestRange> &deferred) {
    for (const auto &range : deferred) {
      page_manager.ApplyProtection(range.address, range.size);
    }
  };

  std::vector<Libs::Graphics::GuestRange> deferred;
  walk(deferred);
  Check(!deferred.empty() && IsWritable(memory) && IsWritable(memory + page_size * 2),
        "a deferred walk protected its pages itself");
  apply(deferred);
  Check(Protection(memory) == PAGE_READONLY && Protection(memory + page_size * 2) == PAGE_READONLY,
        "ApplyProtection did not watch the cleaned pages");

  // A GPU write over page 1 before the helper: it is unreadable at once and stays so.
  deferred.clear();
  walk(deferred);
  tracker.MarkRegionAsGpuModified(address + page_size, page_size);
  Check(Protection(memory + page_size) == PAGE_NOACCESS && IsWritable(memory),
        "a GPU write over a deferred page did not protect it at once");
  apply(deferred);
  Check(Protection(memory) == PAGE_READONLY && Protection(memory + page_size) == PAGE_NOACCESS &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "ApplyProtection undid a later protection");
  tracker.UnmarkRegionAsGpuModified(address + page_size, page_size);

  // Another write watcher on page 2 before the helper: protected at once.
  deferred.clear();
  walk(deferred);
  page_manager.UpdatePageWatchers<true>(address + page_size * 2, page_size);
  Check(Protection(memory + page_size * 2) == PAGE_READONLY && IsWritable(memory),
        "a new write watcher on a deferred page did not protect it at once");
  apply(deferred);
  page_manager.UpdatePageWatchers<false>(address + page_size * 2, page_size);
  Check(Protection(memory + page_size * 2) == PAGE_READONLY,
        "the walk's own watcher was lost");

  tracker.UntrackMemory(address, page_size * 3);
  Release(memory);
}

void TestCleanUploadPreservesOwnership() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, region_size * 2, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});

  for (const auto start : {address + 63 * page_size,
                           address + region_size - page_size}) {
    const auto before = start - page_size;
    const auto after = start + 2 * page_size;
    tracker.MarkRegionAsCpuModified(before, page_size);
    tracker.MarkRegionAsCpuModified(after, page_size);
    uint32_t ranges = 0;
    uint32_t completions = 0;
    ResetProtectionLog();
    tracker.ForEachUploadRange(
        start, 2 * page_size, false,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 1 && g_protection_calls == 0 &&
              !tracker.IsRegionCpuModified(start, 2 * page_size),
          "clean read-only upload changed dirty state or protection");

    tracker.ForEachUploadRange(
        start, 2 * page_size, true,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 2 &&
              tracker.IsRegionGpuModified(start, page_size) &&
              tracker.IsRegionGpuModified(start + page_size, page_size) &&
              Protection(reinterpret_cast<void *>(start)) == PAGE_NOACCESS &&
              Protection(reinterpret_cast<void *>(start + page_size)) ==
                  PAGE_NOACCESS &&
              tracker.IsRegionCpuModified(before, page_size) &&
              tracker.IsRegionCpuModified(after, page_size) &&
              !tracker.IsRegionGpuModified(before, page_size) &&
              !tracker.IsRegionGpuModified(after, page_size) &&
              IsWritable(reinterpret_cast<void *>(before)) &&
              IsWritable(reinterpret_cast<void *>(after)),
          "clean written upload lost GPU ownership or changed dirty neighbors");
    tracker.UnmarkRegionAsGpuModified(start, 2 * page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void BenchmarkCleanUploads() {
  constexpr uint64_t size = 256ull * 1024 * 1024;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(size, 0);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, size, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});
  std::puts("chunk_bytes,sweeps,calls,elapsed_ns,ns_per_sweep,ns_per_call");
  for (const uint64_t chunk : {size, Libs::Graphics::TRACKER_REGION_SIZE,
                               uint64_t{64 * 1024}}) {
    uint64_t ranges = 0;
    uint64_t completions = 0;
    uint64_t sweeps = 0;
    ResetProtectionLog();
    const auto start = std::chrono::steady_clock::now();
    std::chrono::nanoseconds elapsed{};
    do {
      for (uint64_t offset = 0; offset < size; offset += chunk) {
        tracker.ForEachUploadRange(
            address + offset, chunk, false,
            [&](uint64_t, uint64_t) noexcept { ranges++; },
            [&]() noexcept { completions++; });
      }
      sweeps++;
      elapsed = std::chrono::steady_clock::now() - start;
    } while (elapsed < std::chrono::milliseconds(250));
    Check(ranges == 0 && completions == sweeps * (size / chunk) &&
              g_protection_calls == 0,
          "clean upload benchmark changed ranges, completion, or protection");
    std::printf("%llu,%llu,%llu,%lld,%.2f,%.2f\n",
                static_cast<unsigned long long>(chunk),
                static_cast<unsigned long long>(sweeps),
                static_cast<unsigned long long>(completions),
                static_cast<long long>(elapsed.count()),
                static_cast<double>(elapsed.count()) / sweeps,
                static_cast<double>(elapsed.count()) / completions);
  }
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestRangeInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = AllocateFixedGuestRange(size, 0x1000000);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

// InvalidateRegion's on_cpu_dirty runs before a region's pages turn CPU-dirty and writable,
// and what it returns lives until they have. BufferCache notes the pages pending under a
// lock there, so an acquire either runs before a CPU write can reach them or finds them
// both pending and dirty.
void TestInvalidationHoldsAcrossCpuDirty() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(
      address, page_size * 2, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(!tracker.IsRegionCpuModified(address, page_size * 2) && !IsWritable(memory),
        "invalidation hook setup did not leave the pages clean and protected");

  struct Hold {
    MemoryTracker *tracker;
    uint8_t *memory;
    uint64_t size;
    int *released;
    ~Hold() {
      Check(tracker->IsRegionCpuModified(reinterpret_cast<uint64_t>(memory), size) &&
                IsWritable(memory),
            "invalidation released its hook before the pages turned CPU-dirty");
      (*released)++;
    }
  };
  int calls = 0;
  int released = 0;
  tracker.InvalidateRegion(
      address + 16, 32, [] { Check(false, "clean invalidation requested a GPU flush"); },
      [&](uint64_t vaddr, uint64_t size) {
        calls++;
        Check(vaddr <= address + 16 && vaddr + size >= address + 48,
              "invalidation hook did not cover the range");
        Check(!tracker.IsRegionCpuModified(address, page_size) && !IsWritable(memory),
              "invalidation hook ran after the pages turned writable");
        return Hold{&tracker, memory, page_size, &released};
      });
  Check(calls == 1 && released == 1, "invalidation hook did not run once per region part");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  for (const bool is_written : {false, true}) {
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    uint32_t ranges = 0;
    tracker.ForEachUploadRange(
        boundary - page_size, page_size * 2, is_written,
        [&](uint64_t start, uint64_t size) noexcept {
          Check(start == boundary - page_size + ranges * page_size &&
                    size == page_size,
                "cold upload did not visit both region boundaries exactly");
          ranges++;
        },
        [&]() noexcept {
          Check(ranges == 2 &&
                    Protection(reinterpret_cast<void *>(boundary - page_size)) ==
                        PAGE_READONLY &&
                    Protection(reinterpret_cast<void *>(boundary)) == PAGE_READONLY,
                "cold upload completed before clearing and protecting both regions");
        });
    for (const auto page : {boundary - page_size, boundary}) {
      Check(!tracker.IsRegionCpuModified(page, page_size) &&
                tracker.IsRegionGpuModified(page, page_size) == is_written &&
                Protection(reinterpret_cast<void *>(page)) ==
                    (is_written ? PAGE_NOACCESS : PAGE_READONLY),
            "cross-region upload did not preserve final ownership and protection");
    }
    Check(IsWritable(reinterpret_cast<void *>(boundary - 2 * page_size)) &&
              IsWritable(reinterpret_cast<void *>(boundary + page_size)),
          "cross-region upload changed neighboring pages");
    tracker.UnmarkRegionAsGpuModified(boundary - page_size, 2 * page_size);
    tracker.UntrackMemory(address, region_size * 2);
  }
  Release(memory);
}

void TestConcurrentColdUploads() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::counting_semaphore<2> start{0};
  std::counting_semaphore<2> upload_entered{0};
  std::counting_semaphore<2> finish_upload{0};
  std::vector<std::jthread> workers;
  for (const auto page : {address, address + region_size}) {
    workers.emplace_back([&, page] {
      start.acquire();
      uint32_t ranges = 0;
      tracker.ForEachUploadRange(
          page, page_size, true,
          [&](uint64_t upload_address, uint64_t upload_size) noexcept {
            Check(upload_address == page && upload_size == page_size,
                  "concurrent cold upload lost its dirty page");
            ranges++;
          },
          [&]() noexcept {
            Check(ranges == 1, "concurrent cold upload skipped its dirty page");
            upload_entered.release();
            finish_upload.acquire();
          });
    });
  }
  start.release(2);
  const bool first_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  const bool second_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release(2);
  for (auto &worker : workers) {
    worker.join();
  }
  Check(first_entered && second_entered,
        "cold writable uploads serialized disjoint regions");
  for (const auto page : {address, address + region_size}) {
    Check(!tracker.IsRegionCpuModified(page, page_size) &&
              tracker.IsRegionGpuModified(page, page_size) &&
              Protection(reinterpret_cast<void *>(page)) == PAGE_NOACCESS,
          "concurrent cold upload did not retain GPU ownership");
    tracker.UnmarkRegionAsGpuModified(page, page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

struct UploadCounter {
  uint32_t uploads = 0;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
};

void Sync(MemoryTracker &tracker, uint64_t vaddr, uint64_t size,
          bool is_written, UploadCounter &counter) {
  tracker.ForEachUploadRange(
      vaddr, size, is_written,
      [&](uint64_t address, uint64_t bytes) noexcept {
        counter.ranges.push_back({address, bytes});
      },
      [&]() noexcept { counter.uploads++; });
}

uint8_t *AllocateRegions(uintptr_t base, uint64_t regions) {
  const auto size = Libs::Graphics::TRACKER_REGION_SIZE * regions;
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), size,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base),
        "region-aligned allocation failed");
  return memory;
}

void TestRepeatedSyncSkipsUnchangedRange() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadCounter written;
  Sync(tracker, address + 16, page_size * 4, true, written);
  Check(written.uploads == 1 && written.ranges.size() == 1 &&
            written.ranges[0].first == address &&
            written.ranges[0].second == page_size * 5,
        "first written sync did not upload its pages");
  written = {};
  ResetProtectionLog();
  Sync(tracker, address + 16, page_size * 4, true, written);
  Check(written.uploads == 0 && written.ranges.empty() &&
            g_protection_calls == 0,
        "repeated written sync of a GPU-owned range was not skipped");

  // A read of a range still GPU-owned throughout is just as redundant.
  Sync(tracker, address + 16, page_size * 4, false, written);
  Check(written.uploads == 0, "read sync of a GPU-owned range was not skipped");

  UploadCounter read;
  Sync(tracker, address + page_size * 8, page_size * 4, false, read);
  Check(read.uploads == 1 && read.ranges.size() == 1,
        "first read sync did not upload its pages");
  read = {};
  Sync(tracker, address + page_size * 8, page_size * 4, false, read);
  Check(read.uploads == 0, "repeated read sync of a clean range was not skipped");

  // The shortcut is keyed on the exact range; a subrange walks and finds nothing.
  Sync(tracker, address + page_size * 9, page_size, false, read);
  Check(read.uploads == 1 && read.ranges.empty(),
        "clean subrange sync uploaded pages");

  // A written sync does not ride on a read-only record.
  read = {};
  Sync(tracker, address + page_size * 8, page_size * 4, true, read);
  Check(read.uploads == 1 && read.ranges.empty() &&
            tracker.IsRegionGpuModified(address + page_size * 8, page_size * 4),
        "written sync after a read sync skipped its GPU marking");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 16);
  tracker.MarkRegionAsCpuModified(address, page_size * 16);
  tracker.UntrackMemory(address, page_size * 16);
  Release(memory);
}

void TestSyncRecordInvalidation() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto written_range = address;
  const auto read_range = address + page_size * 8;
  const auto other_page = address + page_size * 15;

  UploadCounter counter;
  Sync(tracker, written_range, page_size * 4, true, counter);
  Sync(tracker, read_range, page_size * 4, false, counter);
  Sync(tracker, other_page, page_size, false, counter);

  // A GPU page handed back to the CPU and written: the next written sync must
  // upload it and take it back.
  tracker.UnmarkRegionAsGpuModified(written_range + page_size * 2, page_size);
  tracker.MarkRegionAsCpuModified(written_range + page_size * 2, 8);
  counter = {};
  Sync(tracker, written_range, page_size * 4, true, counter);
  Check(counter.uploads == 1 && counter.ranges.size() == 1 &&
            counter.ranges[0].first == written_range + page_size * 2 &&
            counter.ranges[0].second == page_size &&
            tracker.IsRegionGpuModified(written_range + page_size * 2, page_size) &&
            !tracker.IsRegionCpuModified(written_range, page_size * 4),
        "written sync missed a page handed back to the CPU");

  // A handed-back page nobody wrote still needs GPU ownership again.
  tracker.UnmarkRegionAsGpuModified(written_range + page_size, page_size);
  counter = {};
  Sync(tracker, written_range, page_size * 4, true, counter);
  Check(counter.uploads == 1 && counter.ranges.empty() &&
            tracker.IsRegionGpuModified(written_range + page_size, page_size),
        "written sync did not reacquire a handed-back page");

  // A CPU write through a fault: the next read sync uploads exactly that page.
  tracker.InvalidateRegion(read_range + page_size * 3 + 100, 8, [] {
    Check(false, "clean page invalidation requested a GPU flush");
  });
  counter = {};
  Sync(tracker, read_range, page_size * 4, false, counter);
  Check(counter.uploads == 1 && counter.ranges.size() == 1 &&
            counter.ranges[0].first == read_range + page_size * 3 &&
            counter.ranges[0].second == page_size,
        "read sync missed a CPU write");

  // A CPU write elsewhere costs a walk but uploads nothing here.
  tracker.MarkRegionAsCpuModified(other_page, 8);
  counter = {};
  Sync(tracker, read_range, page_size * 4, false, counter);
  Check(counter.uploads == 1 && counter.ranges.empty(),
        "unrelated CPU write changed a clean range's upload");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 16);
  tracker.MarkRegionAsCpuModified(address, page_size * 16);
  tracker.UntrackMemory(address, page_size * 16);
  Release(memory);
}

void TestWrittenSyncSkipsGpuOwnedRegions() {
  constexpr uintptr_t base = 0x0000000204000000ull;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateRegions(base, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  // Partial first and last regions around a whole one.
  const auto begin = address + page_size * 7;
  const auto size = region_size * 3 - page_size * 12;

  UploadCounter counter;
  Sync(tracker, begin, size, true, counter);
  Check(tracker.IsRegionGpuModified(begin, size) &&
            !tracker.IsRegionCpuModified(begin, size),
        "multi-region written sync did not take ownership");

  // Hand back one page of the whole middle region and one of the last.
  const auto middle_page = address + region_size + page_size * 100;
  const auto last_page = address + region_size * 2 + page_size * 5;
  for (const auto page : {middle_page, last_page}) {
    tracker.UnmarkRegionAsGpuModified(page, page_size);
    tracker.MarkRegionAsCpuModified(page, page_size);
  }
  counter = {};
  ResetProtectionLog();
  Sync(tracker, begin, size, true, counter);
  bool touched_first_region = false;
  for (const auto &call : g_protection_log) {
    touched_first_region |= call.address < address + region_size;
  }
  Check(counter.uploads == 1 && counter.ranges.size() == 2 &&
            counter.ranges[0].first == middle_page &&
            counter.ranges[1].first == last_page && !touched_first_region,
        "multi-region written sync did not upload only the handed-back pages");
  for (auto page = begin; page < begin + size; page += page_size) {
    Check(tracker.IsRegionGpuModified(page, page_size),
          "multi-region written sync left a page CPU-owned");
  }

  tracker.UnmarkRegionAsGpuModified(address, region_size * 3);
  tracker.MarkRegionAsCpuModified(address, region_size * 3);
  tracker.UntrackMemory(address, region_size * 3);
  Release(memory);
}

void TestUnlockedModificationQueries() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadCounter counter;
  Sync(tracker, address, page_size * 4, false, counter);
  Check(!tracker.IsRegionCpuModified(address, page_size * 4) &&
            !tracker.IsRegionGpuModified(address, page_size * 4),
        "clean region answered modified");
  Sync(tracker, address + page_size, page_size, true, counter);
  Check(tracker.IsRegionGpuModified(address, page_size * 4) &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size * 4),
        "GPU-dirty page was not reported exactly");
  tracker.MarkRegionAsCpuModified(address + page_size * 3, 8);
  Check(tracker.IsRegionCpuModified(address, page_size * 4) &&
            !tracker.IsRegionCpuModified(address, page_size * 3),
        "CPU-dirty page was not reported exactly");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 4);
  tracker.MarkRegionAsCpuModified(address, page_size * 4);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// A fault thread hands GPU pages back to the CPU (BufferCache::TryResolveWrittenBack)
// while the GPU thread keeps re-binding writable ranges. As in the buffer cache, a
// page is handed back only while no bind of it is in flight. Once the fault thread
// stops, one more bind of each range must own every page again.
void TestConcurrentHandBacksAreNeverMissed() {
  constexpr uintptr_t base = 0x0000000208000000ull;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateRegions(base, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  constexpr std::pair<uint64_t, uint64_t> ranges[] = {
      {page_size * 3, region_size * 3 - page_size * 8},
      {region_size, region_size},
      {page_size * 10, page_size * 20},
      {region_size * 2 + page_size * 50, page_size}};
  constexpr uint64_t first_page = 3;
  constexpr uint64_t last_page = (region_size * 3 - page_size * 5) / page_size;

  std::mutex writeback_mutex;
  std::set<uint64_t> writeback_pages;
  const auto bind = [&](uint64_t offset, uint64_t size) {
    {
      std::lock_guard lock(writeback_mutex);
      writeback_pages.erase(writeback_pages.lower_bound(offset / page_size),
                            writeback_pages.upper_bound((offset + size - 1) /
                                                        page_size));
    }
    tracker.ForEachUploadRange(
        address + offset, size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
  };
  for (const auto &[offset, size] : ranges) {
    bind(offset, size);
  }

  std::atomic_bool done{false};
  std::atomic_uint64_t handed_back{0};
  std::jthread faults([&] {
    uint64_t random = 0x9e3779b97f4a7c15ull;
    while (!done.load(std::memory_order_acquire)) {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      const auto page = first_page + random % (last_page - first_page);
      std::lock_guard lock(writeback_mutex);
      if (writeback_pages.erase(page) == 0) {
        continue;
      }
      tracker.UnmarkRegionAsGpuModified(address + page * page_size, page_size);
      if ((random & 1) != 0) {
        tracker.MarkRegionAsCpuModified(address + page * page_size, 8);
      }
      handed_back.fetch_add(1, std::memory_order_relaxed);
    }
  });
  uint64_t random = 0xda942042e4dd58b5ull;
  for (uint32_t pass = 0; pass < 20000; pass++) {
    const auto &[offset, size] = ranges[pass % std::size(ranges)];
    bind(offset, size);
    // RecordWritebacks: some GPU-owned pages become eligible for hand-back.
    std::lock_guard lock(writeback_mutex);
    for (uint32_t i = 0; i < 4; i++) {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      writeback_pages.insert(first_page + random % (last_page - first_page));
    }
  }
  done.store(true, std::memory_order_release);
  faults.join();
  Check(handed_back.load(std::memory_order_relaxed) != 0,
        "hand-back thread never ran");

  for (const auto &[offset, size] : ranges) {
    bind(offset, size);
  }
  for (auto page = first_page; page < last_page; page++) {
    Check(tracker.IsRegionGpuModified(address + page * page_size, page_size) &&
              !tracker.IsRegionCpuModified(address + page * page_size, page_size),
          "a page handed back during repeated binds was never reacquired");
  }

  tracker.UnmarkRegionAsGpuModified(address, region_size * 3);
  tracker.MarkRegionAsCpuModified(address, region_size * 3);
  tracker.UntrackMemory(address, region_size * 3);
  Release(memory);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
MemoryTracker *g_fault_tracker = nullptr;
std::atomic_uint64_t g_fault_begin{0};
std::atomic_uint64_t g_fault_end{0};

// What BufferCache::InvalidateMemory does for a write to a clean protected page.
bool HandleTrackedWrite(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  const auto address = info.access_violation_vaddr;
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      address < g_fault_begin.load() || address >= g_fault_end.load()) {
    return false;
  }
  g_fault_tracker->InvalidateRegion(address, 1, [] { std::_Exit(3); });
  return true;
}

// CPU writes fault and invalidate pages on one thread while another keeps
// re-synchronizing the same ranges (taking the repeat shortcut whenever it can)
// into a copy. Once the writer stops, one more sync must bring every write over.
void TestConcurrentCpuWritesAreNeverMissed() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr uint64_t pages = 64;
  constexpr uint64_t size = page_size * pages;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = Allocate(harness.page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::vector<uint8_t> copy(size);

  const auto sync = [&](uint64_t first_page, uint64_t count) {
    std::vector<std::pair<uint64_t, uint64_t>> uploads;
    tracker.ForEachUploadRange(
        address + first_page * page_size, count * page_size, false,
        [&](uint64_t range, uint64_t bytes) noexcept {
          uploads.push_back({range, bytes});
        },
        [&]() noexcept {
          for (const auto &[range, bytes] : uploads) {
            std::memcpy(copy.data() + (range - address),
                        reinterpret_cast<const void *>(range), bytes);
          }
        });
  };
  // A few ranges bound over and over, overlapping each other.
  constexpr std::pair<uint64_t, uint64_t> ranges[] = {
      {0, pages}, {0, 16}, {8, 8}, {20, 30}, {63, 1}, {32, 32}};
  for (const auto &[first, count] : ranges) {
    sync(first, count);
  }

  g_fault_tracker = &tracker;
  g_fault_begin.store(address);
  g_fault_end.store(address + size);
  Check(Common::HostException::InstallHandler(HandleTrackedWrite),
        "install tracked-write handler failed");

  std::atomic_bool done{false};
  std::jthread writer([&] {
    uint64_t random = 0x2545f4914f6cdd1dull;
    for (uint64_t value = 1; value <= 4000; value++) {
      random ^= random << 13;
      random ^= random >> 7;
      random ^= random << 17;
      const auto slot = random % (size / sizeof(uint64_t));
      reinterpret_cast<volatile uint64_t *>(memory)[slot] = value;
    }
    done.store(true, std::memory_order_release);
  });
  uint64_t passes = 0;
  while (!done.load(std::memory_order_acquire)) {
    const auto &[first, count] = ranges[passes++ % std::size(ranges)];
    sync(first, count);
  }
  writer.join();
  for (const auto &[first, count] : ranges) {
    sync(first, count);
  }
  Check(std::memcmp(copy.data(), memory, size) == 0,
        "a CPU write racing repeated syncs never reached the copy");

  g_fault_end.store(g_fault_begin.load());
  tracker.MarkRegionAsCpuModified(address, size);
  tracker.UntrackMemory(address, size);
  Release(memory);
}
#endif

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  } else if (std::strcmp(name, "recursive-tracking-lock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    lock.lock();
  } else if (std::strcmp(name, "non-owner-tracking-unlock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    std::thread worker([&] { lock.unlock(); });
    worker.join();
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
#if defined(__APPLE__)
  std::vector<char> path(PATH_MAX);
  uint32_t path_size = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    path.resize(path_size);
    Check(_NSGetExecutablePath(path.data(), &path_size) == 0,
          "_NSGetExecutablePath failed");
  }
#endif
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
#if defined(__APPLE__)
    ::execl(path.data(), "MemoryTrackerTests", "--death", name, nullptr);
#else
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
#endif
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload",
                           "recursive-tracking-lock", "non-owner-tracking-unlock"}) {
    CheckDeathCase(name);
  }
}

// Run-ahead (MemoryTracker::InvalidateFault / InvalidateAhead). Pages are numbered from the
// allocation; history epochs are passed in directly.
struct AheadHarness {
  explicit AheadHarness(uint64_t pages = 64)
      : page_size(harness.page_manager.GetPageSize()),
        memory(Allocate(harness.page_manager, pages)),
        address(reinterpret_cast<uint64_t>(memory)), size(page_size * pages) {
    Clean(0, pages);
  }
  ~AheadHarness() {
    tracker().MarkRegionAsCpuModified(address, size);
    tracker().UntrackMemory(address, size);
    Release(memory);
  }
  MemoryTracker &tracker() { return harness.tracker; }
  uint64_t Page(uint64_t page) const { return address + page * page_size; }
  // A write fault on the page; returns both run-ahead candidates in pages.
  std::pair<uint64_t, uint64_t> FaultBoth(uint64_t page, uint64_t epoch = 0) {
    const auto ahead = tracker().InvalidateFault(
        Page(page) + 8, epoch, [] { Check(false, "run-ahead fault requested a GPU flush"); },
        [](uint64_t, uint64_t) noexcept { return 0; });
    Check(ahead.up_bytes % page_size == 0 && ahead.down_bytes % page_size == 0,
          "run-ahead candidate was not whole pages");
    return {ahead.up_bytes / page_size, ahead.down_bytes / page_size};
  }
  // The candidate above the page.
  uint64_t Fault(uint64_t page, uint64_t epoch = 0) { return FaultBoth(page, epoch).first; }
  // The candidate below the page.
  uint64_t FaultDown(uint64_t page, uint64_t epoch = 0) { return FaultBoth(page, epoch).second; }
  // Faults each page of [first, last), building write history.
  void FaultEach(uint64_t first, uint64_t last, uint64_t epoch = 0) {
    for (auto page = first; page < last; page++) {
      (void)Fault(page, epoch);
    }
  }
  // As FaultEach, from the top page down.
  void FaultEachDown(uint64_t first, uint64_t last, uint64_t epoch = 0) {
    for (auto page = last; page > first; page--) {
      (void)Fault(page - 1, epoch);
    }
  }
  uint64_t Ahead(uint64_t page, uint64_t pages, uint64_t epoch = 0) {
    return tracker().InvalidateAhead(Page(page), pages * page_size,
                                     Libs::Graphics::AheadDirection::Up, epoch,
                                     [](uint64_t, uint64_t) noexcept { return 0; }) /
           page_size;
  }
  // The downward window of pages [end - pages, end), walked from end - 1.
  uint64_t Below(uint64_t end, uint64_t pages, uint64_t epoch = 0) {
    return tracker().InvalidateAhead(Page(end - pages), pages * page_size,
                                     Libs::Graphics::AheadDirection::Down, epoch,
                                     [](uint64_t, uint64_t) noexcept { return 0; }) /
           page_size;
  }
  void Clean(uint64_t page, uint64_t pages) {
    tracker().ForEachUploadRange(
        Page(page), pages * page_size, false, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
  }
  bool CpuDirty(uint64_t page) {
    return tracker().IsRegionCpuModified(Page(page), page_size);
  }

  TrackerHarness harness;
  uint64_t page_size;
  uint8_t *memory;
  uint64_t address;
  uint64_t size;
};

void TestAheadFollowsRun() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEach(10, 14);
  const auto ahead = h.Fault(14);
  Check(ahead == Libs::Graphics::AHEAD_MIN_PAGES,
        "a fault continuing a 4-page run did not offer a 4-page window");
  ResetProtectionLog();
  Check(h.Ahead(15, ahead) == ahead, "run-ahead did not mark its whole window");
  Check(g_protection_log.size() == 1 && g_protection_log[0].address == h.Page(15) &&
            g_protection_log[0].size == ahead * h.page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite,
        "run-ahead did not unprotect its window in one call");
  Check(h.CpuDirty(15) && h.CpuDirty(18) && IsWritable(h.memory + 18 * h.page_size) &&
            !h.CpuDirty(19) && !IsWritable(h.memory + 19 * h.page_size),
        "run-ahead marked the wrong pages");
  // The run continues at the window's end: the run below is now 9 pages.
  h.Clean(0, 64);
  h.FaultEach(10, 19);
  Check(h.Fault(19) == 4, "run-ahead window was not cut to the write history");
}

void TestAheadNeedsRun() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  ResetProtectionLog();
  Check(h.Fault(14) == 0, "a fault without a run below offered run-ahead");
  Check(g_protection_log.size() == 1 && g_protection_log[0].address == h.Page(14) &&
            g_protection_log[0].size == h.page_size,
        "a fault without a run unprotected more than its page");
  Check(!h.CpuDirty(15), "a fault without a run marked the next page");
}

void TestAheadStopsAtGpuDirty() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.tracker().ForEachUploadRange(
      h.Page(17), h.page_size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  h.FaultEach(10, 14);
  Check(h.Fault(14) == 2, "run-ahead window passed a GPU-dirty page");
  Check(h.Ahead(15, 4) == 2 && h.CpuDirty(16) &&
            h.tracker().IsRegionGpuModified(h.Page(17), h.page_size) && !h.CpuDirty(17),
        "run-ahead marked past a GPU-dirty page");
  h.tracker().UnmarkRegionAsGpuModified(h.Page(17), h.page_size);
}

void TestAheadStopsAtHistoryGap() {
  AheadHarness h;
  h.FaultEach(10, 15);
  h.FaultEach(16, 17);
  h.FaultEach(18, 24);
  h.Clean(0, 64);
  // Page 15 has no history but is CPU-dirty already: passed over. Page 17 ends the prefix.
  h.tracker().MarkRegionAsCpuModified(h.Page(15), h.page_size);
  h.FaultEach(10, 14);
  Check(h.Fault(14) == 2, "run-ahead did not pass a CPU-dirty page and stop at a history gap");
  Check(h.Ahead(15, 3) == 2 && h.CpuDirty(16) && !h.CpuDirty(17),
        "run-ahead marked a page outside the write history");
  // A prefix of CPU-dirty pages only is empty.
  h.Clean(0, 64);
  h.tracker().MarkRegionAsCpuModified(h.Page(16), h.page_size);
  Check(h.Ahead(16, 4) == 0 && !h.CpuDirty(17), "run-ahead marked a page that never faulted");
}

void TestAheadHistoryAges() {
  AheadHarness h;
  h.FaultEach(10, 24, 0);
  h.Clean(0, 64);
  h.FaultEach(10, 14, 1);
  Check(h.Fault(14, 1) == 4, "write history did not last one epoch");
  h.Clean(0, 64);
  h.FaultEach(10, 14, 2);
  Check(h.Fault(14, 2) == 0 && h.Ahead(15, 4, 2) == 0,
        "write history outlived two epochs");
  // A gap of more than one epoch clears both halves.
  h.Clean(0, 64);
  h.FaultEach(10, 24, 2);
  h.Clean(0, 64);
  h.FaultEach(10, 14, 4);
  Check(h.Fault(14, 4) == 0, "write history survived an epoch gap");
}

// A fault that read an older epoch before waiting for the region lock keeps the history.
void TestAheadHistoryStaleEpoch() {
  AheadHarness h;
  h.FaultEach(10, 24, 1);
  h.Clean(0, 64);
  h.FaultEach(10, 14, 1);
  Check(h.Fault(14, 0) == 4, "a stale history epoch cleared the write history");
  h.Clean(0, 64);
  h.FaultEach(10, 14, 1);
  Check(h.Fault(14, 1) == 4, "a stale history epoch moved the history back");
}

// The window grows with the run below the fault and stops at AHEAD_MAX_PAGES.
void TestAheadWindowFollowsRunLength() {
  AheadHarness h(128);
  h.FaultEach(10, 128);
  h.Clean(0, 128);
  h.FaultEach(10, 19);
  Check(h.Fault(19) == 9, "a 9-page run did not offer a 9-page window");
  h.Clean(0, 128);
  h.FaultEach(10, 42);
  Check(h.Fault(42) == Libs::Graphics::AHEAD_MAX_PAGES,
        "a 32-page run did not offer the largest window");
  h.Clean(0, 128);
  h.FaultEach(10, 50);
  Check(h.Fault(50) == Libs::Graphics::AHEAD_MAX_PAGES,
        "a 40-page run offered more than the largest window");
}

void TestAheadStaysInRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  const auto fault = [&](uint64_t vaddr) {
    return tracker.InvalidateFault(
        vaddr, 0, [] { Check(false, "region-end fault requested a GPU flush"); },
        [](uint64_t, uint64_t) noexcept { return 0; });
  };
  const auto clean = [&] {
    tracker.ForEachUploadRange(
        boundary - 16 * page_size, 32 * page_size, false, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
  };
  clean();
  for (auto page = boundary - 16 * page_size; page < boundary + 16 * page_size;
       page += page_size) {
    (void)fault(page);
  }
  clean();
  for (auto page = boundary - 6 * page_size; page < boundary - 2 * page_size;
       page += page_size) {
    (void)fault(page);
  }
  Check(fault(boundary - 2 * page_size).up_bytes == page_size,
        "run-ahead window crossed the region end");
  for (auto page = boundary - 2 * page_size; page < boundary - page_size; page += page_size) {
    (void)fault(page);
  }
  Check(fault(boundary - page_size).up_bytes == 0, "run-ahead offered pages past the region end");
  Check(!tracker.IsRegionCpuModified(boundary, page_size) &&
            !IsWritable(memory + (boundary - address)),
        "run-ahead touched the next region");
  tracker.MarkRegionAsCpuModified(address, region_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

// As TestInvalidationHoldsAcrossCpuDirty: an upload that takes the pending set waits for the
// hold, so it finds the run-ahead pages CPU-dirty.
void TestAheadHeldUntilWritable() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEach(10, 14);
  const auto ahead = h.Fault(14);
  struct Hold {
    AheadHarness *h;
    uint64_t pages;
    int *released;
    ~Hold() {
      Check(h->tracker().IsRegionCpuModified(h->Page(15), pages * h->page_size) &&
                IsWritable(h->memory + 15 * h->page_size) &&
                IsWritable(h->memory + (14 + pages) * h->page_size),
            "run-ahead released its hold before the pages turned CPU-dirty");
      (*released)++;
    }
  };
  int calls = 0;
  int released = 0;
  const auto marked = h.tracker().InvalidateAhead(
      h.Page(15), ahead * h.page_size, Libs::Graphics::AheadDirection::Up, 0,
      [&](uint64_t vaddr, uint64_t size) {
        calls++;
        Check(vaddr == h.Page(15) && size == ahead * h.page_size,
              "run-ahead hold did not cover the window");
        Check(!h.CpuDirty(15) && !IsWritable(h.memory + 15 * h.page_size),
              "run-ahead hold ran after the pages turned writable");
        return Hold{&h, ahead, &released};
      });
  Check(marked == ahead * h.page_size && calls == 1 && released == 1,
        "run-ahead hold did not run once");
}

void TestAheadRecheckUnderLock() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEach(10, 14);
  const auto ahead = h.Fault(14);
  Check(ahead == 4, "recheck setup did not offer a window");
  // The GPU writes page 16 between the candidate and the mark.
  h.tracker().ForEachUploadRange(
      h.Page(16), h.page_size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(h.Ahead(15, ahead) == 1 && h.CpuDirty(15) && !h.CpuDirty(16) &&
            h.tracker().IsRegionGpuModified(h.Page(16), h.page_size),
        "run-ahead marked a page that turned GPU-dirty after its candidate");
  h.tracker().UnmarkRegionAsGpuModified(h.Page(16), h.page_size);
}

// The texture cache's write watch on a page outlives run-ahead's buffer-side unwatch.
void TestAheadKeepsImageWatch() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.harness.page_manager.UpdatePageWatchers<true>(h.Page(16), h.page_size);
  h.FaultEach(10, 14);
  const auto ahead = h.Fault(14);
  ResetProtectionLog();
  Check(h.Ahead(15, ahead) == ahead, "image watch setup did not mark the window");
  bool touched = false;
  for (const auto &call : g_protection_log) {
    touched |= call.address < h.Page(17) && h.Page(16) < call.address + call.size;
  }
  Check(!touched && Protection(h.memory + 16 * h.page_size) == PAGE_READONLY &&
            IsWritable(h.memory + 15 * h.page_size) && IsWritable(h.memory + 17 * h.page_size),
        "run-ahead unprotected a page an image still watches");
  h.harness.page_manager.UpdatePageWatchers<false>(h.Page(16), h.page_size);
}

// Downward run-ahead: a run written from the top down opens the window below the fault.
void TestAheadDownFollowsRun() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEachDown(20, 24);
  const auto [up, down] = h.FaultBoth(19);
  Check(up == 0 && down == Libs::Graphics::AHEAD_MIN_PAGES,
        "a fault continuing a 4-page run downwards did not offer a 4-page window below");
  ResetProtectionLog();
  Check(h.Below(19, down) == down, "downward run-ahead did not mark its whole window");
  Check(g_protection_log.size() == 1 && g_protection_log[0].address == h.Page(15) &&
            g_protection_log[0].size == down * h.page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite,
        "downward run-ahead did not unprotect its window in one call");
  Check(h.CpuDirty(15) && h.CpuDirty(18) && IsWritable(h.memory + 15 * h.page_size) &&
            !h.CpuDirty(14) && !IsWritable(h.memory + 14 * h.page_size),
        "downward run-ahead marked the wrong pages");
  // The run continues below the window: the run above is now 9 pages.
  h.Clean(0, 64);
  h.FaultEachDown(15, 24);
  Check(h.FaultDown(14) == 4, "downward run-ahead window was not cut to the write history");
}

void TestAheadDownNeedsRun() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  ResetProtectionLog();
  Check(h.FaultBoth(19) == std::pair<uint64_t, uint64_t>{0, 0},
        "a fault without a run above offered run-ahead");
  Check(g_protection_log.size() == 1 && g_protection_log[0].address == h.Page(19) &&
            g_protection_log[0].size == h.page_size,
        "a fault without a run unprotected more than its page");
  Check(!h.CpuDirty(18), "a fault without a run marked the page below");
}

void TestAheadDownStopsAtGpuDirty() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.tracker().ForEachUploadRange(
      h.Page(16), h.page_size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  h.FaultEachDown(20, 24);
  Check(h.FaultDown(19) == 2, "downward run-ahead window passed a GPU-dirty page");
  Check(h.Below(19, 4) == 2 && h.CpuDirty(17) &&
            h.tracker().IsRegionGpuModified(h.Page(16), h.page_size) && !h.CpuDirty(16),
        "downward run-ahead marked past a GPU-dirty page");
  h.tracker().UnmarkRegionAsGpuModified(h.Page(16), h.page_size);
}

void TestAheadDownStopsAtHistoryGap() {
  AheadHarness h;
  h.FaultEach(10, 16);
  h.FaultEach(17, 18);
  h.FaultEach(19, 24);
  h.Clean(0, 64);
  // Page 18 has no history but is CPU-dirty already: passed over. Page 16 ends the prefix.
  h.tracker().MarkRegionAsCpuModified(h.Page(18), h.page_size);
  h.FaultEachDown(20, 24);
  Check(h.FaultDown(19) == 2,
        "downward run-ahead did not pass a CPU-dirty page and stop at a history gap");
  Check(h.Below(19, 3) == 2 && h.CpuDirty(17) && !h.CpuDirty(16),
        "downward run-ahead marked a page outside the write history");
  // A prefix of CPU-dirty pages only is empty.
  h.Clean(0, 64);
  h.tracker().MarkRegionAsCpuModified(h.Page(17), h.page_size);
  Check(h.Below(18, 4) == 0 && !h.CpuDirty(16),
        "downward run-ahead marked a page that never faulted");
}

void TestAheadDownHistoryAges() {
  AheadHarness h;
  h.FaultEach(10, 24, 0);
  h.Clean(0, 64);
  h.FaultEachDown(20, 24, 1);
  Check(h.FaultDown(19, 1) == 4, "write history did not last one epoch downwards");
  h.Clean(0, 64);
  h.FaultEachDown(20, 24, 2);
  Check(h.FaultDown(19, 2) == 0 && h.Below(19, 4, 2) == 0,
        "write history outlived two epochs downwards");
  h.Clean(0, 64);
  h.FaultEach(10, 24, 2);
  h.Clean(0, 64);
  h.FaultEachDown(20, 24, 4);
  Check(h.FaultDown(19, 4) == 0, "write history survived an epoch gap downwards");
}

// A fault between two CPU-dirty pages offers a window on each side.
void TestAheadBothDirections() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  (void)h.Fault(14);
  (void)h.Fault(16);
  Check(h.FaultBoth(15) == std::pair<uint64_t, uint64_t>{4, 4},
        "a fault between two runs did not offer both windows");
  ResetProtectionLog();
  Check(h.Ahead(16, 4) == 4 && h.Below(15, 4) == 4, "a fault between two runs did not mark both");
  Check(g_protection_log.size() == 2, "the two run-ahead windows did not take one call each");
  Check(h.CpuDirty(11) && h.CpuDirty(19) && !h.CpuDirty(10) && !h.CpuDirty(20),
        "the two run-ahead windows marked the wrong pages");
}

// The downward re-check under the lock walks down from the window's top page: a page turning
// GPU-dirty inside the window ends the mark there, not a prefix from its bottom.
void TestAheadDownRecheckUnderLock() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEachDown(20, 24);
  const auto down = h.FaultDown(19);
  Check(down == 4, "downward recheck setup did not offer a window");
  // The GPU writes page 17 between the candidate and the mark.
  h.tracker().ForEachUploadRange(
      h.Page(17), h.page_size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(h.Below(19, down) == 1 && h.CpuDirty(18) && !h.CpuDirty(17) && !h.CpuDirty(16) &&
            !h.CpuDirty(15) && h.tracker().IsRegionGpuModified(h.Page(17), h.page_size),
        "downward run-ahead re-checked its window in the wrong direction");
  h.tracker().UnmarkRegionAsGpuModified(h.Page(17), h.page_size);
}

// The downward window stops at the region's first page.
void TestAheadDownStaysInRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  const auto fault = [&](uint64_t vaddr) {
    return tracker.InvalidateFault(
        vaddr, 0, [] { Check(false, "region-start fault requested a GPU flush"); },
        [](uint64_t, uint64_t) noexcept { return 0; });
  };
  const auto clean = [&] {
    tracker.ForEachUploadRange(
        boundary - 16 * page_size, 32 * page_size, false, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
  };
  clean();
  for (auto page = boundary - 16 * page_size; page < boundary + 16 * page_size;
       page += page_size) {
    (void)fault(page);
  }
  clean();
  for (auto page = boundary + 6 * page_size; page > boundary + 2 * page_size;
       page -= page_size) {
    (void)fault(page);
  }
  Check(fault(boundary + 2 * page_size).down_bytes == 2 * page_size,
        "downward run-ahead window crossed the region start");
  (void)fault(boundary + page_size);
  Check(fault(boundary).down_bytes == 0, "downward run-ahead offered pages before the region");
  Check(!tracker.IsRegionCpuModified(boundary - page_size, page_size) &&
            !IsWritable(memory + (boundary - page_size - address)),
        "downward run-ahead touched the previous region");
  tracker.MarkRegionAsCpuModified(address, region_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

// As TestAheadHeldUntilWritable, downwards: the hold covers the whole window even when the
// re-check marks only its top part, and is released once that part is CPU-dirty.
void TestAheadDownHeldUntilWritable() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.FaultEachDown(20, 24);
  const auto down = h.FaultDown(19);
  Check(down == 4, "downward hold setup did not offer a window");
  // The GPU writes page 16 between the candidate and the mark: only [17, 19) is marked.
  h.tracker().ForEachUploadRange(
      h.Page(16), h.page_size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  struct Hold {
    AheadHarness *h;
    int *released;
    ~Hold() {
      Check(h->tracker().IsRegionCpuModified(h->Page(17), 2 * h->page_size) &&
                IsWritable(h->memory + 17 * h->page_size) &&
                IsWritable(h->memory + 18 * h->page_size),
            "downward run-ahead released its hold before the pages turned CPU-dirty");
      (*released)++;
    }
  };
  int calls = 0;
  int released = 0;
  const auto marked = h.tracker().InvalidateAhead(
      h.Page(15), down * h.page_size, Libs::Graphics::AheadDirection::Down, 0,
      [&](uint64_t vaddr, uint64_t size) {
        calls++;
        Check(vaddr == h.Page(15) && size == down * h.page_size,
              "downward run-ahead hold did not cover the window");
        Check(!h.CpuDirty(18) && !IsWritable(h.memory + 18 * h.page_size),
              "downward run-ahead hold ran after the pages turned writable");
        return Hold{&h, &released};
      });
  Check(marked == 2 * h.page_size && calls == 1 && released == 1,
        "downward run-ahead hold did not run once");
  Check(!h.CpuDirty(15) && !h.CpuDirty(16), "downward run-ahead marked below a GPU-dirty page");
  h.tracker().UnmarkRegionAsGpuModified(h.Page(16), h.page_size);
}

// As TestAheadKeepsImageWatch, downwards.
void TestAheadDownKeepsImageWatch() {
  AheadHarness h;
  h.FaultEach(10, 24);
  h.Clean(0, 64);
  h.harness.page_manager.UpdatePageWatchers<true>(h.Page(17), h.page_size);
  h.FaultEachDown(20, 24);
  const auto down = h.FaultDown(19);
  ResetProtectionLog();
  Check(h.Below(19, down) == down, "downward image watch setup did not mark the window");
  bool touched = false;
  for (const auto &call : g_protection_log) {
    touched |= call.address < h.Page(18) && h.Page(17) < call.address + call.size;
  }
  Check(!touched && Protection(h.memory + 17 * h.page_size) == PAGE_READONLY &&
            IsWritable(h.memory + 16 * h.page_size) && IsWritable(h.memory + 18 * h.page_size),
        "downward run-ahead unprotected a page an image still watches");
  h.harness.page_manager.UpdatePageWatchers<false>(h.Page(17), h.page_size);
}

void TestNonFaultInvalidationNoHistory() {
  AheadHarness h;
  for (uint64_t page = 10; page < 24; page++) {
    h.tracker().InvalidateRegion(h.Page(page), h.page_size,
                                 [] { Check(false, "clean invalidation requested a flush"); });
  }
  h.Clean(0, 64);
  h.FaultEach(10, 14);
  Check(h.Fault(14) == 0 && h.Ahead(15, 4) == 0,
        "a non-fault invalidation made pages eligible for run-ahead");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-clean-upload") == 0) {
    BenchmarkCleanUploads();
    return 0;
  }
  TestGuestRange();
  TestRangeSet();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestBdaSyncDirtiness();
  TestBdaSyncConcurrentDrain();
  TestBdaSyncTakeWaitsForHolds();
  TestLazyRelease();
  TestDeferredUploadProtection();
  TestCleanUploadPreservesOwnership();
  TestRangeInvalidation();
  TestInvalidationHoldsAcrossCpuDirty();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestConcurrentColdUploads();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestRepeatedSyncSkipsUnchangedRange();
  TestSyncRecordInvalidation();
  TestWrittenSyncSkipsGpuOwnedRegions();
  TestUnlockedModificationQueries();
  TestConcurrentHandBacksAreNeverMissed();
  TestFatalPaths();
  TestAheadFollowsRun();
  TestAheadNeedsRun();
  TestAheadStopsAtGpuDirty();
  TestAheadStopsAtHistoryGap();
  TestAheadHistoryAges();
  TestAheadHistoryStaleEpoch();
  TestAheadWindowFollowsRunLength();
  TestAheadStaysInRegion();
  TestAheadHeldUntilWritable();
  TestAheadRecheckUnderLock();
  TestNonFaultInvalidationNoHistory();
  TestAheadKeepsImageWatch();
  TestAheadDownFollowsRun();
  TestAheadDownNeedsRun();
  TestAheadDownStopsAtGpuDirty();
  TestAheadDownStopsAtHistoryGap();
  TestAheadDownHistoryAges();
  TestAheadBothDirections();
  TestAheadDownRecheckUnderLock();
  TestAheadDownStaysInRegion();
  TestAheadDownHeldUntilWritable();
  TestAheadDownKeepsImageWatch();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
  // Last: it leaves its write-fault handler installed.
  TestConcurrentCpuWritesAreNeverMissed();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
