// analysis.cu — Host-side idle warp analysis and cross-run tracking.

#include "analysis.cuh"
#include "wsout.hh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace warpscope {
namespace analysis {

// ---- Configuration ---------------------------------------------------------
static float idle_threshold = 0.5f;

// ---- Managed memory buffer -------------------------------------------------
warp_timing_buffer *timing_buffer = nullptr;

// ---- Last kernel results (exposed to specialization engine) ----------------
static std::vector<warp_result> last_results;
static uint64_t last_max_duration = 0;

// ---- Cross-run tracking ----------------------------------------------------
struct warp_key {
    uint32_t bx, by, bz, warp_id, sm_id;
    bool operator<(const warp_key &o) const {
        if (bx != o.bx) return bx < o.bx;
        if (by != o.by) return by < o.by;
        if (bz != o.bz) return bz < o.bz;
        return warp_id < o.warp_id;
    }
};

struct kernel_stats {
    uint32_t num_runs = 0;
    std::vector<std::tuple<size_t, size_t, float>> run_history; // (total, idle, fraction)
    std::map<warp_key, uint32_t> idle_counts;
    std::map<warp_key, uint32_t> appearance_counts;
};
static std::map<std::string, kernel_stats> all_kernel_stats;

// ---- State -----------------------------------------------------------------
static std::string current_kernel_name_str;

// ---- Per-kernel analysis ---------------------------------------------------
static void analyse_kernel() {
    if (!timing_buffer) return;

    uint32_t n = timing_buffer->count;
    if (n > WS_MAX_WARPS) n = WS_MAX_WARPS;
    if (n == 0) return;

    last_results.clear();
    last_max_duration = 0;

    for (uint32_t i = 0; i < n; i++) {
        auto &r = timing_buffer->records[i];
        if (r.valid == 3 && r.end_clock > r.start_clock) {
            uint64_t dur = r.end_clock - r.start_clock;
            last_results.push_back({r.block_x, r.block_y, r.block_z,
                                    r.warp_id, r.sm_id, dur,
                                    r.active_threads_start, r.active_threads_end});
            if (dur > last_max_duration) last_max_duration = dur;
        }
    }

    if (last_results.empty() || last_max_duration == 0) return;

    std::sort(last_results.begin(), last_results.end(),
              [](const warp_result &a, const warp_result &b) {
                  return a.duration < b.duration;
              });

    size_t num_idle = 0;
    size_t num_partial = 0;
    for (auto &r : last_results) {
        float utilization = (float)r.duration / (float)last_max_duration;
        if (utilization < idle_threshold) num_idle++;
        if (r.active_start < 32 || r.active_end < 32) num_partial++;
    }

    // Print per-kernel report
    auto &s = wsout_stream();
    auto old_flags = s.flags();
    s << std::dec;

    wsout() << "\n--- Warp Utilization Report: kernel [" << current_kernel_name_str << "] ---\n";
    wsout() << "Total warps: " << last_results.size()
            << " | Idle (<" << (int)(idle_threshold * 100) << "% utilization): " << num_idle
            << " | Partial occupancy: " << num_partial << "\n";
    wsout() << "Max warp duration: " << last_max_duration << " cycles\n";

    if (num_idle > 0) {
        wsout() << "\nIdle warps (candidates for warp specialization):\n";
        wsout() << std::setw(12) << "Block"
                << std::setw(8) << "Warp"
                << std::setw(6) << "SM"
                << std::setw(14) << "Duration"
                << std::setw(10) << "Util%"
                << std::setw(16) << "ActiveThreads"
                << "\n";
        for (auto &r : last_results) {
            float util = 100.0f * (float)r.duration / (float)last_max_duration;
            if (util >= idle_threshold * 100.0f) break;
            char block_str[32];
            snprintf(block_str, sizeof(block_str), "(%d,%d,%d)", r.bx, r.by, r.bz);
            wsout() << std::setw(12) << block_str
                    << std::setw(8) << r.warp_id
                    << std::setw(6) << r.sm_id
                    << std::setw(14) << r.duration
                    << std::setw(9) << std::fixed << std::setprecision(1) << util << "%"
                    << std::setw(8) << r.active_start << "/" << r.active_end
                    << "\n";
        }
    }

    wsout() << "\nBusiest warps:\n";
    wsout() << std::setw(12) << "Block"
            << std::setw(8) << "Warp"
            << std::setw(6) << "SM"
            << std::setw(14) << "Duration"
            << std::setw(10) << "Util%"
            << std::setw(16) << "ActiveThreads"
            << "\n";
    size_t start_idx = last_results.size() > 5 ? last_results.size() - 5 : 0;
    for (size_t i = start_idx; i < last_results.size(); i++) {
        auto &r = last_results[i];
        float util = 100.0f * (float)r.duration / (float)last_max_duration;
        char block_str[32];
        snprintf(block_str, sizeof(block_str), "(%d,%d,%d)", r.bx, r.by, r.bz);
        wsout() << std::setw(12) << block_str
                << std::setw(8) << r.warp_id
                << std::setw(6) << r.sm_id
                << std::setw(14) << r.duration
                << std::setw(9) << std::fixed << std::setprecision(1) << util << "%"
                << std::setw(8) << r.active_start << "/" << r.active_end
                << "\n";
    }
    s.flags(old_flags);

    // Update cross-run stats
    auto &ks = all_kernel_stats[current_kernel_name_str];
    ks.num_runs++;
    float idle_frac = (float)num_idle / (float)last_results.size();
    ks.run_history.push_back({last_results.size(), num_idle, idle_frac});
    for (auto &r : last_results) {
        warp_key wk{r.bx, r.by, r.bz, r.warp_id, r.sm_id};
        ks.appearance_counts[wk]++;
        float util = (float)r.duration / (float)last_max_duration;
        if (util < idle_threshold) {
            ks.idle_counts[wk]++;
        }
    }

    // Reset buffer for next kernel
    timing_buffer->count = 0;
    memset(timing_buffer->records, 0, sizeof(timing_buffer->records));
}

// ---- Public API ------------------------------------------------------------

void init(float threshold) {
    idle_threshold = threshold;
}

void tool_init(CUcontext ctx) {
    (void)ctx;
}

void on_kernel_launch(const std::string &kernel_name) {
    current_kernel_name_str = kernel_name;
    if (timing_buffer) {
        timing_buffer->count = 0;
    }
}

void on_kernel_exit() {
    analyse_kernel();
}

const std::vector<warp_result>& get_last_results() {
    return last_results;
}

uint64_t get_last_max_duration() {
    return last_max_duration;
}

const std::string& get_current_kernel_name() {
    return current_kernel_name_str;
}

void term(CUcontext ctx) {
    (void)ctx;
    if (all_kernel_stats.empty()) return;

    wsout() << "\n========== Warpscope Cross-Run Summary ==========\n";
    for (auto &[kname, ks] : all_kernel_stats) {
        if (ks.num_runs < 1) continue;

        auto old_flags = wsout_stream().flags();
        wsout_stream() << std::dec;
        wsout() << "\nKernel [" << kname << "] - " << ks.num_runs << " run(s)\n";

        bool has_idle_pattern = false;
        for (size_t i = 0; i < ks.run_history.size(); i++) {
            auto [total, idle, frac] = ks.run_history[i];
            wsout() << "  Run " << (i + 1) << ": " << idle << "/" << total
                    << " warps idle (" << std::fixed << std::setprecision(1)
                    << (frac * 100) << "%)\n";
            if (idle > 0) has_idle_pattern = true;
        }

        if (has_idle_pattern) {
            size_t min_idle = SIZE_MAX, max_idle = 0;
            for (auto &[total, idle, frac] : ks.run_history) {
                min_idle = std::min(min_idle, idle);
                max_idle = std::max(max_idle, idle);
            }
            if (min_idle == max_idle && min_idle > 0) {
                wsout() << "  >> CONSISTENT: " << min_idle
                        << " warps are idle in every run.\n";
            } else if (min_idle > 0) {
                wsout() << "  >> PATTERN: " << min_idle << "-" << max_idle
                        << " warps idle across runs.\n";
            }
        }

        // Show consistently idle warps
        std::vector<std::pair<warp_key, float>> consistently_idle;
        for (auto &[wk, idle_count] : ks.idle_counts) {
            uint32_t appearances = ks.appearance_counts[wk];
            float idle_rate = (float)idle_count / (float)appearances;
            if (idle_rate > 0.5f) {
                consistently_idle.push_back({wk, idle_rate});
            }
        }

        if (consistently_idle.empty() && !has_idle_pattern) {
            wsout() << "  No idle warps detected.\n";
        } else if (!consistently_idle.empty()) {
            std::sort(consistently_idle.begin(), consistently_idle.end(),
                      [](const auto &a, const auto &b) {
                          return a.second > b.second;
                      });

            wsout() << "  " << consistently_idle.size()
                    << " consistently idle warp(s):\n";
            wsout() << std::setw(12) << "Block"
                    << std::setw(8) << "Warp"
                    << std::setw(6) << "SM"
                    << std::setw(14) << "Idle Rate"
                    << "\n";
            for (auto &[wk, rate] : consistently_idle) {
                char block_str[32];
                snprintf(block_str, sizeof(block_str), "(%d,%d,%d)",
                         wk.bx, wk.by, wk.bz);
                wsout() << std::setw(12) << block_str
                        << std::setw(8) << wk.warp_id
                        << std::setw(6) << wk.sm_id
                        << std::setw(12) << std::fixed << std::setprecision(0)
                        << (rate * 100) << "%"
                        << "\n";
            }
        }
        wsout_stream().flags(old_flags);
    }
    wsout() << "\n=================================================\n";
}

} // namespace analysis
} // namespace warpscope
