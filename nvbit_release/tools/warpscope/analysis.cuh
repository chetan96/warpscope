#ifndef ANALYSIS_CUH
#define ANALYSIS_CUH

#include <string>
#include <vector>
#include <cstdint>
#include "warp_timing.cuh"

typedef struct CUctx_st *CUcontext;

namespace warpscope {
namespace analysis {

// Pointer to managed memory buffer (set by warpscope.cu)
extern warp_timing_buffer *timing_buffer;

// Parsed per-warp result from the last kernel analysis
struct warp_result {
    uint32_t bx, by, bz, warp_id, sm_id;
    uint64_t duration;
    uint32_t active_start;
    uint32_t active_end;
};

// Overhead timing record (populated by warpscope.cu when WARPSCOPE_BENCHMARK=1)
struct overhead_record {
    double sync_ms;      // cudaDeviceSynchronize() time
    double analysis_ms;  // analysis::on_kernel_exit() time
    double spec_ms;      // specialization::on_kernel_exit() time
};

void init(float threshold);
void tool_init(CUcontext ctx);
void on_kernel_launch(const std::string &kernel_name);
void on_kernel_exit();
void record_overhead(const overhead_record &rec);
void term(CUcontext ctx);

// Get the parsed results and max duration from the most recent kernel analysis.
const std::vector<warp_result>& get_last_results();
uint64_t get_last_max_duration();
const std::string& get_current_kernel_name();

} // namespace analysis
} // namespace warpscope

#endif // ANALYSIS_CUH
