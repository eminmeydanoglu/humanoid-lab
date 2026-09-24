#pragma once

#include <chrono>
#include <optional>

// Input-thread-only state. Reception alone does not refresh this watchdog:
// callers must pass the time of a successfully decoded Protocol v4 token.
class PoseWatchdog {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit PoseWatchdog(std::chrono::milliseconds timeout) : timeout_(timeout) {}

  void EnterStream(TimePoint now) {
    entered_at_ = now;
    last_valid_at_.reset();
  }

  void ValidToken(TimePoint when) {
    if (entered_at_ && when >= *entered_at_ &&
        (!last_valid_at_ || when > *last_valid_at_)) {
      last_valid_at_ = when;
    }
  }

  bool Expired(TimePoint now) const {
    if (!entered_at_) return false;
    const TimePoint anchor = last_valid_at_.value_or(*entered_at_);
    return now - anchor > timeout_;
  }

  void LeaveStream() {
    entered_at_.reset();
    last_valid_at_.reset();
  }

 private:
  std::chrono::milliseconds timeout_;
  std::optional<TimePoint> entered_at_;
  std::optional<TimePoint> last_valid_at_;
};
