// specialization.cu — Warp specialization recommendation engine.
// Consumes per-kernel warp timing data from analysis::get_last_results()
// and produces pattern classification + producer/consumer split recommendations.

#include "specialization.cuh"
#include "analysis.cuh"
#include "wsout.hh"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

namespace warpscope {
namespace specialization {

// ---- Configuration ---------------------------------------------------------
static std::string json_output_path;
static bool json_enabled = false;

// ---- Per-block recommendation ----------------------------------------------
struct block_rec {
    uint32_t bx, by, bz;
    uint32_t total_warps;
    uint32_t idle_warps;
    uint32_t compute_warps;
    float avg_idle_util;
    float avg_compute_util;
    uint32_t suggested_producers;
    uint32_t suggested_consumers;
};

// ---- Per-kernel recommendation ---------------------------------------------
struct kernel_rec {
    std::string name;
    uint32_t num_runs;
    float consistency_score;
    bool recommend_specialization;
    std::string pattern;
    float avg_idle_fraction;
    std::vector<block_rec> blocks;
};

static std::map<std::string, std::vector<float>> idle_fractions_by_kernel;
static std::map<std::string, std::vector<std::vector<block_rec>>> block_recs_by_kernel;

// ---- Pattern classification ------------------------------------------------
static std::string classify_pattern(const std::vector<block_rec> &blocks) {
    if (blocks.empty()) return "balanced";

    bool has_idle = false;
    for (auto &b : blocks) {
        if (b.idle_warps > 0) has_idle = true;
    }
    if (!has_idle) return "balanced";

    // Check if idle blocks are at the front (causal triangle)
    bool front_idle = true;
    bool seen_busy = false;
    for (auto &b : blocks) {
        float block_idle_frac = (float)b.idle_warps / (float)b.total_warps;
        if (block_idle_frac > 0.5f && seen_busy) { front_idle = false; break; }
        if (block_idle_frac <= 0.5f) seen_busy = true;
    }

    // Check if entire blocks are idle vs partial
    size_t fully_idle_blocks = 0;
    size_t partially_idle_blocks = 0;
    for (auto &b : blocks) {
        if (b.idle_warps == b.total_warps) fully_idle_blocks++;
        else if (b.idle_warps > 0) partially_idle_blocks++;
    }

    // Check for monotonically increasing utilization (causal triangle)
    bool monotonic = true;
    for (size_t i = 1; i < blocks.size(); i++) {
        if (blocks[i].avg_compute_util < blocks[i-1].avg_compute_util - 0.05f) {
            monotonic = false;
            break;
        }
    }

    if (front_idle && monotonic && blocks.size() > 2) return "causal_triangle";
    if (fully_idle_blocks > 0 && partially_idle_blocks == 0) return "uniform_idle";

    // Check for alternating pattern within blocks
    // (odd/even warps idle — need to look at within-block structure)
    // For now, classify as divergent if idle warps are scattered
    if (partially_idle_blocks > fully_idle_blocks) return "divergent";

    return "tail_idle";
}

// ---- Compute per-block recommendations from warp results -------------------
static std::vector<block_rec> compute_block_recs(
    const std::vector<analysis::warp_result> &results,
    uint64_t max_duration, float threshold)
{
    // Group warps by block
    struct block_key {
        uint32_t bx, by, bz;
        bool operator<(const block_key &o) const {
            if (bx != o.bx) return bx < o.bx;
            if (by != o.by) return by < o.by;
            return bz < o.bz;
        }
    };

    std::map<block_key, std::vector<const analysis::warp_result*>> by_block;
    for (auto &r : results) {
        by_block[{r.bx, r.by, r.bz}].push_back(&r);
    }

    std::vector<block_rec> recs;
    for (auto &[bk, warps] : by_block) {
        block_rec br{};
        br.bx = bk.bx; br.by = bk.by; br.bz = bk.bz;
        br.total_warps = warps.size();

        float idle_util_sum = 0, compute_util_sum = 0;
        for (auto *w : warps) {
            float util = (float)w->duration / (float)max_duration;
            if (util < threshold) {
                br.idle_warps++;
                idle_util_sum += util;
            } else {
                br.compute_warps++;
                compute_util_sum += util;
            }
        }

        br.avg_idle_util = br.idle_warps > 0 ? idle_util_sum / br.idle_warps : 0;
        br.avg_compute_util = br.compute_warps > 0 ? compute_util_sum / br.compute_warps : 0;

        // Recommendation: repurpose idle warps as producers
        br.suggested_producers = br.idle_warps;
        br.suggested_consumers = br.compute_warps;
        // Clamp: at least 1 consumer if any warps are active
        if (br.suggested_consumers == 0 && br.total_warps > 0) {
            br.suggested_consumers = 1;
            if (br.suggested_producers > 0) br.suggested_producers--;
        }

        recs.push_back(br);
    }

    // Sort by block coordinates
    std::sort(recs.begin(), recs.end(), [](const block_rec &a, const block_rec &b) {
        if (a.bx != b.bx) return a.bx < b.bx;
        if (a.by != b.by) return a.by < b.by;
        return a.bz < b.bz;
    });

    return recs;
}

// ---- JSON output -----------------------------------------------------------
static void write_json(const std::vector<kernel_rec> &kernels) {
    std::ofstream out(json_output_path);
    if (!out.is_open()) {
        wsout() << "Failed to open JSON output file: " << json_output_path << "\n";
        return;
    }

    out << "{\n";
    out << "  \"tool\": \"warpscope\",\n";
    out << "  \"kernels\": [\n";
    for (size_t k = 0; k < kernels.size(); k++) {
        auto &kr = kernels[k];
        out << "    {\n";
        out << "      \"name\": \"" << kr.name << "\",\n";
        out << "      \"num_runs\": " << kr.num_runs << ",\n";
        out << "      \"consistency_score\": " << std::fixed << std::setprecision(2) << kr.consistency_score << ",\n";
        out << "      \"recommend_specialization\": " << (kr.recommend_specialization ? "true" : "false") << ",\n";
        out << "      \"pattern\": \"" << kr.pattern << "\",\n";
        out << "      \"avg_idle_fraction\": " << std::setprecision(3) << kr.avg_idle_fraction << ",\n";
        out << "      \"blocks\": [\n";
        for (size_t b = 0; b < kr.blocks.size(); b++) {
            auto &br = kr.blocks[b];
            out << "        {\n";
            out << "          \"block\": [" << br.bx << ", " << br.by << ", " << br.bz << "],\n";
            out << "          \"total_warps\": " << br.total_warps << ",\n";
            out << "          \"idle_warps\": " << br.idle_warps << ",\n";
            out << "          \"compute_warps\": " << br.compute_warps << ",\n";
            out << "          \"avg_idle_util\": " << std::setprecision(3) << br.avg_idle_util << ",\n";
            out << "          \"avg_compute_util\": " << std::setprecision(3) << br.avg_compute_util << ",\n";
            out << "          \"suggested_producers\": " << br.suggested_producers << ",\n";
            out << "          \"suggested_consumers\": " << br.suggested_consumers << "\n";
            out << "        }" << (b + 1 < kr.blocks.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (k + 1 < kernels.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();
    wsout() << "JSON report written to: " << json_output_path << "\n";
}

// ---- Public API ------------------------------------------------------------

void init(const std::string &json_path) {
    json_output_path = json_path;
    json_enabled = !json_path.empty();
}

void on_kernel_exit() {
    auto &results = analysis::get_last_results();
    uint64_t max_dur = analysis::get_last_max_duration();
    const std::string &kname = analysis::get_current_kernel_name();

    if (results.empty() || max_dur == 0) return;

    // Compute block-level recommendations
    float threshold = 0.5f; // Use same default; could be passed from config
    auto blocks = compute_block_recs(results, max_dur, threshold);

    // Track idle fraction
    size_t idle_count = 0;
    for (auto &b : blocks) idle_count += b.idle_warps;
    float idle_frac = (float)idle_count / (float)results.size();
    idle_fractions_by_kernel[kname].push_back(idle_frac);
    block_recs_by_kernel[kname].push_back(blocks);
}

void term(CUcontext ctx) {
    (void)ctx;

    if (idle_fractions_by_kernel.empty()) return;

    std::vector<kernel_rec> kernel_recs;

    wsout() << "\n========== Warp Specialization Recommendations ==========\n";

    for (auto &[kname, fracs] : idle_fractions_by_kernel) {
        auto &all_blocks = block_recs_by_kernel[kname];
        uint32_t num_runs = fracs.size();

        // Compute consistency score
        float mean_frac = 0;
        for (float f : fracs) mean_frac += f;
        mean_frac /= num_runs;

        float variance = 0;
        for (float f : fracs) variance += (f - mean_frac) * (f - mean_frac);
        variance /= num_runs;
        // Consistency = 1 - normalized std dev (higher = more consistent)
        float consistency = mean_frac > 0 ? 1.0f - sqrtf(variance) / mean_frac : 1.0f;
        if (consistency < 0) consistency = 0;
        if (consistency > 1) consistency = 1;

        // Use last run's block recs as representative
        auto &blocks = all_blocks.back();
        std::string pattern = classify_pattern(blocks);
        bool recommend = (mean_frac > 0.1f && consistency > 0.5f && pattern != "balanced");

        kernel_rec kr;
        kr.name = kname;
        kr.num_runs = num_runs;
        kr.consistency_score = consistency;
        kr.recommend_specialization = recommend;
        kr.pattern = pattern;
        kr.avg_idle_fraction = mean_frac;
        kr.blocks = blocks;
        kernel_recs.push_back(kr);

        // Print human-readable recommendation
        auto old_flags = wsout_stream().flags();
        wsout_stream() << std::dec;

        wsout() << "\nKernel [" << kname << "] (" << num_runs << " runs)\n";
        wsout() << "  Pattern: " << pattern << "\n";
        wsout() << "  Avg idle fraction: " << std::fixed << std::setprecision(1)
                << (mean_frac * 100) << "%\n";
        wsout() << "  Consistency score: " << std::setprecision(2) << consistency << "\n";

        if (recommend) {
            uint32_t total_producers = 0, total_consumers = 0;
            for (auto &b : blocks) {
                total_producers += b.suggested_producers;
                total_consumers += b.suggested_consumers;
            }

            wsout() << "  >> RECOMMENDATION: Enable warp specialization\n";
            wsout() << "  >> Suggested split: " << total_producers << " producer warps, "
                    << total_consumers << " consumer warps\n";

            if (pattern == "causal_triangle") {
                wsout() << "  >> Strategy: Use position-dependent producer/consumer ratio.\n";
                wsout() << "     Early blocks (low sequence positions) → more producers (prefetch K/V tiles).\n";
                wsout() << "     Late blocks (high sequence positions) → more consumers (full compute).\n";
            } else if (pattern == "uniform_idle") {
                wsout() << "  >> Strategy: Entire idle blocks can become data prefetchers.\n";
                wsout() << "     Route them to load next-layer weights or KV cache tiles.\n";
            } else if (pattern == "divergent") {
                wsout() << "  >> Strategy: Within each block, assign idle warps to shared memory loads.\n";
                wsout() << "     Use warp_id-based branching for producer/consumer roles.\n";
            }

            wsout() << "  Per-block breakdown:\n";
            wsout() << std::setw(12) << "Block"
                    << std::setw(10) << "Warps"
                    << std::setw(8) << "Idle"
                    << std::setw(12) << "Producers"
                    << std::setw(12) << "Consumers"
                    << std::setw(12) << "IdleUtil%"
                    << "\n";
            for (auto &b : blocks) {
                if (b.idle_warps == 0 && blocks.size() > 8) continue; // skip all-busy blocks in large grids
                char block_str[32];
                snprintf(block_str, sizeof(block_str), "(%d,%d,%d)", b.bx, b.by, b.bz);
                wsout() << std::setw(12) << block_str
                        << std::setw(10) << b.total_warps
                        << std::setw(8) << b.idle_warps
                        << std::setw(12) << b.suggested_producers
                        << std::setw(12) << b.suggested_consumers
                        << std::setw(11) << std::fixed << std::setprecision(1)
                        << (b.avg_idle_util * 100) << "%"
                        << "\n";
            }
        } else {
            wsout() << "  >> No specialization recommended";
            if (pattern == "balanced") {
                wsout() << " (all warps are well-utilized).\n";
            } else {
                wsout() << " (pattern too inconsistent or idle fraction too low).\n";
            }
        }
        wsout_stream().flags(old_flags);
    }

    wsout() << "\n=========================================================\n";

    // Write JSON if enabled
    if (json_enabled) {
        write_json(kernel_recs);
    }
}

} // namespace specialization
} // namespace warpscope
