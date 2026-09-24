#include <cassert>
#include <chrono>

#include "../../patches/sonic/pose_watchdog.hpp"

int main() {
  using namespace std::chrono;
  using Clock = PoseWatchdog::Clock;
  const auto t0 = Clock::time_point{};
  PoseWatchdog guard{milliseconds{300}};

  assert(!guard.Expired(t0 + seconds{5}));  // planner mode
  guard.EnterStream(t0);
  assert(!guard.Expired(t0 + milliseconds{300}));
  assert(guard.Expired(t0 + milliseconds{301}));  // no first token

  guard.EnterStream(t0);
  guard.ValidToken(t0 - milliseconds{1});  // stale token from previous session
  assert(guard.Expired(t0 + milliseconds{301}));

  guard.EnterStream(t0);
  guard.ValidToken(t0 + milliseconds{100});
  assert(!guard.Expired(t0 + milliseconds{399}));
  assert(guard.Expired(t0 + milliseconds{401}));  // inference stopped

  guard.EnterStream(t0 + seconds{2});
  guard.ValidToken(t0 + seconds{2} + milliseconds{100});
  guard.ValidToken(t0 + seconds{2} + milliseconds{50});  // out of order
  assert(guard.Expired(t0 + seconds{2} + milliseconds{401}));

  guard.LeaveStream();
  assert(!guard.Expired(t0 + seconds{10}));
  guard.EnterStream(t0 + seconds{10});
  assert(guard.Expired(t0 + seconds{10} + milliseconds{301}));
}
