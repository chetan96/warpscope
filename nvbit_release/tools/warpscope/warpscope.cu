// warpscope.cu — Standalone NVBit tool for idle warp detection and
// warp specialization recommendations.
//
// Usage: LD_PRELOAD=warpscope.so ./your_cuda_program
//
// Environment variables:
//   WARPSCOPE_THRESHOLD  — idle utilization threshold (default 0.5)
//   WARPSCOPE_VERBOSE    — verbosity (0=normal, 1=detailed)
//   WARPSCOPE_LOGFILE    — redirect output to file (default stderr)
//   WARPSCOPE_JSON       — path for JSON recommendation report
//   WARPSCOPE_KERNEL     — comma-separated kernel whitelist

#include "nvbit_tool.h"
#include "nvbit.h"
#include "utils/channel.hpp"

#include <cstring>
#include <string>
#include <unordered_set>

#include "warp_timing.cuh"
#include "analysis.cuh"
#include "specialization.cuh"
#include "wsout.hh"

// ---- Managed memory buffer (device + host accessible) ----------------------
static __managed__ warpscope::warp_timing_buffer ws_timing_buffer;

// ---- State -----------------------------------------------------------------
static int verbose = 0;
static std::unordered_set<CUfunction> instrumented_functions;
static std::unordered_set<std::string> kernel_whitelist;

// ---- Utility ---------------------------------------------------------------
static inline std::string cut_kernel_name(const std::string &name) {
    size_t pos = name.find_first_of("<(");
    return (pos != std::string::npos) ? name.substr(0, pos) : name;
}

template<typename T>
static inline uint64_t tobits64(T value) {
    static_assert(sizeof(T) == sizeof(uint64_t), "T must be 64 bits");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

static void parse_kernel_whitelist(const char *str) {
    if (!str || str[0] == '\0') return;
    std::string s(str);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        std::string name = s.substr(start, end - start);
        if (!name.empty()) kernel_whitelist.insert(name);
        start = end + 1;
    }
}

// ---- NVBit Callbacks -------------------------------------------------------

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    // Read configuration
    float threshold = 0.5f;
    const char *env;

    env = getenv("WARPSCOPE_THRESHOLD");
    if (env) {
        float t = atof(env);
        if (t > 0.0f && t < 1.0f) threshold = t;
    }

    env = getenv("WARPSCOPE_VERBOSE");
    if (env) verbose = atoi(env);

    env = getenv("WARPSCOPE_LOGFILE");
    if (env && env[0] != '\0') {
        std::string logfile(env);
        set_out_file(logfile);
    }

    std::string json_path;
    env = getenv("WARPSCOPE_JSON");
    if (env && env[0] != '\0') json_path = env;

    env = getenv("WARPSCOPE_KERNEL");
    parse_kernel_whitelist(env);

    // Initialize modules
    warpscope::analysis::init(threshold);
    warpscope::specialization::init(json_path);

    // Print banner
    std::string pad(70, '-');
    wsout() << pad << "\n";
    wsout() << "Warpscope — Idle Warp Detector & Specialization Advisor\n";
    wsout() << "Threshold: " << (int)(threshold * 100) << "% | Verbose: " << verbose;
    if (!json_path.empty()) wsout_stream() << " | JSON: " << json_path;
    if (!kernel_whitelist.empty()) {
        wsout_stream() << " | Kernels: ";
        for (auto &k : kernel_whitelist) wsout_stream() << k << " ";
    }
    wsout_stream() << "\n";
    wsout() << pad << "\n";
}

static void instrument_function(CUcontext ctx, CUfunction func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);
    related_functions.push_back(func);

    for (auto f : related_functions) {
        if (!instrumented_functions.insert(f).second) continue;

        auto instrs = nvbit_get_instrs(ctx, f);
        if (instrs.empty()) continue;

        if (verbose) {
            wsout() << "Instrumenting " << nvbit_get_func_name(ctx, f)
                    << " (" << instrs.size() << " instructions)\n";
        }

        // IPOINT_BEFORE on first instruction → warp start time
        Instr *first_instr = instrs[0];
        nvbit_insert_call(first_instr, "warpscope_timer_start", IPOINT_BEFORE);
        nvbit_add_call_arg_guard_pred_val(first_instr);
        nvbit_add_call_arg_const_val64(first_instr, tobits64(&ws_timing_buffer), false);

        // IPOINT_BEFORE on every EXIT/RET/BRK → warp end time
        for (auto instr : instrs) {
            std::string opcode(instr->getOpcode());
            if (opcode == "EXIT" || opcode == "RET" || opcode == "BRK") {
                nvbit_insert_call(instr, "warpscope_timer_end", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, tobits64(&ws_timing_buffer), false);
            }
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel ||
        cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
        cbid == API_CUDA_cuLaunchCooperativeKernelMultiDevice) {

        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            std::string kernel_name = nvbit_get_func_name(ctx, p->f);
            std::string short_name = cut_kernel_name(kernel_name);

            // Check whitelist
            bool enable = true;
            if (!kernel_whitelist.empty()) {
                enable = kernel_whitelist.count(short_name) > 0;
            }

            if (enable) {
                instrument_function(ctx, p->f);
                warpscope::analysis::on_kernel_launch(short_name);
            }
            nvbit_enable_instrumented(ctx, p->f, enable);

        } else {
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                wsout() << "Kernel error: " << cudaGetErrorString(err) << "\n";
            }
            warpscope::analysis::on_kernel_exit();
            warpscope::specialization::on_kernel_exit();
        }
    }
}

void nvbit_tool_init(CUcontext ctx) {
    wsout() << "Initializing warpscope context...\n";
    memset(&ws_timing_buffer, 0, sizeof(ws_timing_buffer));
    warpscope::analysis::timing_buffer = &ws_timing_buffer;
    warpscope::analysis::tool_init(ctx);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    warpscope::analysis::term(ctx);
    warpscope::specialization::term(ctx);
}
