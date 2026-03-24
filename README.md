# Warpscope

A standalone [NVBit](https://github.com/NVlabs/NVBit)-based tool for detecting idle GPU warps and recommending warp specialization strategies for CUDA kernels.

Forked from [Nixnan](https://github.com/LLNL/nixnan) (floating-point exception detector). Warpscope adds a new NVBit tool that instruments CUDA kernels at the binary level to measure per-warp execution time, identify underutilized warps, and suggest producer/consumer warp splits for optimization.

## What It Does

1. **Idle Warp Detection** -- Injects `clock64()` timing probes at the first instruction (`IPOINT_BEFORE`) and every `EXIT`/`RET`/`BRK` instruction of each kernel. Measures per-warp duration and flags warps below a utilization threshold as "idle".

2. **Cross-Run Consistency Tracking** -- Runs the same kernel multiple times and identifies warps that are *consistently* idle, confirming the pattern is deterministic and exploitable.

3. **Warp Specialization Recommendations** -- Classifies the workload pattern (`causal_triangle`, `uniform_idle`, `divergent`, `balanced`) and suggests how to split warps into producers (data loaders) and consumers (compute).

4. **JSON Reports** -- Optionally outputs a machine-readable JSON report with per-block producer/consumer recommendations.

No source code modification is needed -- warpscope instruments any CUDA binary at runtime via `LD_PRELOAD`.

## Requirements

- Linux x86_64
- CUDA 13+ (CUDA 12 may work but NVBit 1.7.7.3 recommends 13+)
- GPU with compute capability >= 8.6 (Ampere, Ada Lovelace, Hopper)

## Setup

```bash
# Build everything (downloads NVBit automatically)
make

# Or build just warpscope
cd nvbit_release/tools/warpscope
make ARCH=sm_89   # Set to your GPU's SM version
```

This produces `warpscope.so` in the project root.

## Usage

```bash
# Basic usage -- instrument any CUDA program
LD_PRELOAD=warpscope.so ./your_cuda_program

# Custom idle threshold (default 0.5 = 50%)
WARPSCOPE_THRESHOLD=0.3 LD_PRELOAD=warpscope.so ./your_program

# Output JSON recommendations to a file
WARPSCOPE_JSON=report.json LD_PRELOAD=warpscope.so ./your_program

# Redirect human-readable output to a log file
WARPSCOPE_LOGFILE=warpscope.log LD_PRELOAD=warpscope.so ./your_program

# Only instrument specific kernels
WARPSCOPE_KERNEL=flash_attention_causal_fwd LD_PRELOAD=warpscope.so ./your_program

# Verbose mode (show per-instruction details)
WARPSCOPE_VERBOSE=1 LD_PRELOAD=warpscope.so ./your_program
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WARPSCOPE_THRESHOLD` | `0.5` | Utilization fraction below which a warp is "idle" (0.0-1.0) |
| `WARPSCOPE_VERBOSE` | `0` | Verbosity level (0=normal, 1=detailed) |
| `WARPSCOPE_LOGFILE` | stderr | Path to redirect human-readable output |
| `WARPSCOPE_JSON` | (none) | Path for JSON recommendation report |
| `WARPSCOPE_KERNEL` | (all) | Comma-separated kernel name whitelist |

## Example Output

### Causal (Decoder) Flash Attention

```bash
$ LD_PRELOAD=warpscope.so ./flash_attention_causal

#warpscope: --- Warp Utilization Report: kernel [flash_attention_causal_fwd] ---
#warpscope: Total warps: 16 | Idle (<50% utilization): 8 | Partial occupancy: 0
#warpscope: Max warp duration: 3250346629 cycles

#warpscope: Idle warps (candidates for warp specialization):
#warpscope:        Block    Warp    SM      Duration     Util%   ActiveThreads
#warpscope:      (0,0,0)       1     0     670603387     20.6%      32/32
#warpscope:      (0,0,0)       0     0     670603394     20.6%      32/32
#warpscope:      (1,0,0)       3     2    1506517739     46.3%      32/32
#warpscope:      (1,0,0)       0     2    1506517757     46.3%      32/32

#warpscope: Busiest warps:
#warpscope:      (3,0,0)       1     6    3250346629    100.0%      32/32
```

### Specialization Recommendation

```
#warpscope: ========== Warp Specialization Recommendations ==========

#warpscope: Kernel [flash_attention_causal_fwd] (3 runs)
#warpscope:   Pattern: causal_triangle
#warpscope:   Avg idle fraction: 50.0%
#warpscope:   Consistency score: 1.00
#warpscope:   >> RECOMMENDATION: Enable warp specialization
#warpscope:   >> Suggested split: 8 producer warps, 8 consumer warps
#warpscope:   >> Strategy: Use position-dependent producer/consumer ratio.
#warpscope:      Early blocks (low sequence positions) -> more producers (prefetch K/V tiles).
#warpscope:      Late blocks (high sequence positions) -> more consumers (full compute).
```

### JSON Output

```json
{
  "tool": "warpscope",
  "kernels": [{
    "name": "flash_attention_causal_fwd",
    "num_runs": 3,
    "consistency_score": 1.00,
    "recommend_specialization": true,
    "pattern": "causal_triangle",
    "avg_idle_fraction": 0.500,
    "blocks": [
      {"block": [0,0,0], "total_warps": 4, "idle_warps": 4, "suggested_producers": 3, "suggested_consumers": 1},
      {"block": [1,0,0], "total_warps": 4, "idle_warps": 4, "suggested_producers": 3, "suggested_consumers": 1},
      {"block": [2,0,0], "total_warps": 4, "idle_warps": 0, "suggested_producers": 0, "suggested_consumers": 4},
      {"block": [3,0,0], "total_warps": 4, "idle_warps": 0, "suggested_producers": 0, "suggested_consumers": 4}
    ]
  }]
}
```

## Example Programs

The `examples/` directory contains test cases:

| File | Description |
|---|---|
| `flash_attention_causal.cu` | Causal (decoder) Flash Attention 2 with shared memory -- shows ~50% idle warps from the triangular causal mask |
| `idle_warp_test.cu` | Synthetic tests: imbalanced, divergent, and tail-effect kernels |
| `idle_warp_ml.cu` | Real ML patterns: MoE routing, padded batch attention, speculative decoding, sparse attention, token pruning |
| `basic.cu` | Tensor core FP16 matrix multiply (from nixnan) |

Compile and run any example:

```bash
cd examples
nvcc -arch=compute_89 -lineinfo flash_attention_causal.cu -o flash_attention_causal
LD_PRELOAD=../warpscope.so ./flash_attention_causal
```

## How It Works

Warpscope uses NVBit to instrument CUDA kernels at the GPU binary (SASS) level:

1. **At kernel launch**, warpscope iterates over all SASS instructions and injects:
   - `warpscope_timer_start()` at `IPOINT_BEFORE` on the first instruction -- records `clock64()`, `get_warpid()`, `get_smid()`, active thread count via `__popc(__activemask())`
   - `warpscope_timer_end()` at `IPOINT_BEFORE` on every `EXIT`/`RET`/`BRK` instruction -- captures end timestamp

2. **After kernel completion**, the host reads the managed memory buffer containing per-warp timing records, computes utilization as `duration / max_duration`, and flags warps below the threshold.

3. **Across multiple runs**, the tool tracks which warps are consistently idle and computes a consistency score.

4. **At program exit**, the specialization engine classifies the workload pattern and outputs recommendations.

## Workload Patterns Detected

| Pattern | Description | Example |
|---|---|---|
| `causal_triangle` | Early blocks idle, late blocks busy (monotonically increasing utilization) | Decoder-style causal attention |
| `uniform_idle` | Entire blocks are idle while others are fully busy | MoE with imbalanced routing, token pruning |
| `divergent` | Alternating idle/busy warps within blocks | Kernels with branch divergence |
| `tail_idle` | Last warps in blocks are idle | Tail effects from non-aligned problem sizes |
| `balanced` | All warps well-utilized | Encoder-style full attention |

## Project Structure

```
nvbit_release/tools/warpscope/
    warpscope.cu         # Main NVBit tool (callbacks, instrumentation)
    inject_funcs.cu      # Device-side timer functions
    warp_timing.cuh      # Shared data structures (managed memory)
    analysis.cu/cuh      # Per-kernel analysis, cross-run tracking
    specialization.cu/cuh # Pattern classification, recommendations
    wsout.cc/hh          # Logging helpers
    Makefile             # Build
```

## Nixnan (Original Tool)

The original Nixnan FP exception detector is still available at `nvbit_release/tools/nixnan/`. See the nixnan-specific documentation in `Tutorial.md`.

## License

See [LICENSE](LICENSE).
