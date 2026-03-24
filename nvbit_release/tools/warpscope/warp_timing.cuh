#ifndef WARP_TIMING_CUH
#define WARP_TIMING_CUH

#include <stdint.h>

#define WS_MAX_WARPS 4096

namespace warpscope {

struct warp_timing_record {
    uint64_t start_clock;
    uint64_t end_clock;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t warp_id;
    uint32_t sm_id;
    uint32_t active_threads_start;
    uint32_t active_threads_end;
    uint32_t valid;  // 0=unused, 1=has start, 3=complete
};

struct warp_timing_buffer {
    warp_timing_record records[WS_MAX_WARPS];
    uint32_t count;  // atomically incremented to assign slots
};

} // namespace warpscope

#endif // WARP_TIMING_CUH
