#ifndef GRAPHICS_GUEST_GPU_QUEUE_PARK_H
#define GRAPHICS_GUEST_GPU_QUEUE_PARK_H

#include <cstdint>

// Parking a queue on a host-GPU tick (GPU thread). A dispatch packet's resource materialization
// that would wait for the host GPU to read GPU-written memory back can, while another queue is
// runnable, give up instead: the readback sets the pending tick and returns without the bytes,
// every later read of the materialization fails at once, and the dispatch returns before it
// records anything. The command processor then suspends the packet, which runs again from its
// start once the tick has completed (GuestGpu::ParkInsteadOfWait).
//
// Three scopes must all be open for a readback to park:
//   DispatchScope     the command processor executes a dispatch packet in ProcessPm4;
//   RestartableScope  the pipeline cache evaluates a cached plan, whose outputs the next
//                     evaluation rewrites;
//   ReadScope         the read is the materialization's own read of guest memory.
namespace Libs::Graphics::QueuePark {

struct State {
	bool     dispatch     = false;
	uint32_t restartable  = 0;
	uint32_t reads        = 0;
	uint64_t pending_tick = 0;
};

inline thread_local State t_state;

// A readback parked the dispatch: the reads after it fail, and the dispatch returns.
[[nodiscard]] inline bool Pending() noexcept {
	return t_state.pending_tick != 0;
}

[[nodiscard]] inline bool CanPark() noexcept {
	return t_state.dispatch && t_state.restartable != 0 && t_state.reads != 0 && !Pending();
}

// The command processor: takes the tick a parked dispatch waits for (0: none).
[[nodiscard]] inline uint64_t TakePending() noexcept {
	const auto tick      = t_state.pending_tick;
	t_state.pending_tick = 0;
	return tick;
}

class DispatchScope {
public:
	explicit DispatchScope(bool enabled) noexcept: m_previous(t_state.dispatch) {
		t_state.dispatch = enabled;
	}
	~DispatchScope() noexcept { t_state.dispatch = m_previous; }
	DispatchScope(const DispatchScope&)            = delete;
	DispatchScope& operator=(const DispatchScope&) = delete;

private:
	bool m_previous;
};

class RestartableScope {
public:
	RestartableScope() noexcept { t_state.restartable++; }
	~RestartableScope() noexcept { t_state.restartable--; }
	RestartableScope(const RestartableScope&)            = delete;
	RestartableScope& operator=(const RestartableScope&) = delete;
};

class ReadScope {
public:
	ReadScope() noexcept { t_state.reads++; }
	~ReadScope() noexcept { t_state.reads--; }
	ReadScope(const ReadScope&)            = delete;
	ReadScope& operator=(const ReadScope&) = delete;
};

// A page fault handled on the GPU thread inside such a read waits as before: its access is
// retried, so it cannot give up.
class NoParkScope {
public:
	NoParkScope() noexcept: m_previous(t_state.reads) { t_state.reads = 0; }
	~NoParkScope() noexcept { t_state.reads = m_previous; }
	NoParkScope(const NoParkScope&)            = delete;
	NoParkScope& operator=(const NoParkScope&) = delete;

private:
	uint32_t m_previous;
};

} // namespace Libs::Graphics::QueuePark

#endif /* GRAPHICS_GUEST_GPU_QUEUE_PARK_H */
