// inject_funcs.cu — Device-side instrumentation functions for warpscope.
// Injected at IPOINT_BEFORE on first instruction and every EXIT/RET/BRK.

#include <cstdint>
#include "utils/utils.h"
#include "warp_timing.cuh"

using namespace warpscope;

// Injected at IPOINT_BEFORE on the first instruction of each kernel.
extern "C" __device__ __noinline__ void
warpscope_timer_start(int pred, uint64_t pbuffer) {
    int active_mask = __ballot_sync(__activemask(), 1); //gets a bitmask of which threads(lanes) in the warp are active
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1; //first active lane id 

    if (first_laneid == laneid) {
        warp_timing_buffer *buf = (warp_timing_buffer *)pbuffer;
        uint32_t slot = atomicAdd(&buf->count, 1); //atomic adds buf->count  to claim a slot in the shared buffer
        if (slot < WS_MAX_WARPS) {
            int4 cta = get_ctaid();
            buf->records[slot].start_clock = clock64();
            buf->records[slot].block_x = cta.x;
            buf->records[slot].block_y = cta.y;
            buf->records[slot].block_z = cta.z;
            buf->records[slot].warp_id = get_warpid();
            buf->records[slot].sm_id = get_smid();
            buf->records[slot].active_threads_start = __popc(active_mask);
            buf->records[slot].valid = 1;
        }
    }
}

// Injected at IPOINT_BEFORE on every EXIT/RET/BRK instruction.
extern "C" __device__ __noinline__ void
warpscope_timer_end(int pred, uint64_t pbuffer) {
    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (first_laneid == laneid) {
        warp_timing_buffer *buf = (warp_timing_buffer *)pbuffer;
        int4 cta = get_ctaid();
        uint32_t my_warp = get_warpid();
        uint32_t active_threads = __popc(active_mask);
        uint64_t clk = clock64();

        uint32_t n = buf->count < WS_MAX_WARPS ? buf->count : WS_MAX_WARPS;
        for (uint32_t i = 0; i < n; i++) {
            if (buf->records[i].valid == 1 &&
                buf->records[i].block_x == (uint32_t)cta.x &&
                buf->records[i].block_y == (uint32_t)cta.y &&
                buf->records[i].block_z == (uint32_t)cta.z &&
                buf->records[i].warp_id == my_warp) {
                buf->records[i].end_clock = clk;
                buf->records[i].active_threads_end = active_threads;
                buf->records[i].valid = 3;
                break;
            }
        }
    }
}
