#pragma once

#include <cstdint>

namespace mrn_sync
{
struct TimeGateConfig
{
  int64_t max_message_age_ns{300000000};
  int64_t max_clock_offset_ns{20000000};
  int64_t max_offset_uncertainty_ns{10000000};
  int64_t max_future_skew_ns{50000000};
  bool reject_if_unknown_offset{true};
};
}  // namespace mrn_sync
