# Warpscope & Nixnan Tutorial
### Comprehensive Guide to GPU Warp Analysis and Floating-Point Exception Detection

## Table of Contents

1. [Introduction](#introduction)
2. [Part I: Warpscope — Idle Warp Detection & Specialization](#warpscope)
   - [Background: Why Warp Utilization Matters](#warp-background)
   - [Installation](#warpscope-installation)
   - [Basic Usage](#warpscope-usage)
   - [Environment Variables](#warpscope-env-vars)
   - [Understanding the Output](#warpscope-output)
   - [Workload Pattern Classification](#warpscope-patterns)
   - [Case Studies](#warpscope-case-studies)
   - [JSON Report Format](#warpscope-json)
3. [Part II: Nixnan — Floating-Point Exception Detection](#nixnan)
   - [Background: Why FP Exception Detection Matters](#fp-background)
   - [Basic Usage](#nixnan-usage)
   - [Environment Variables](#nixnan-env-vars)
   - [Advanced Features](#nixnan-advanced)
   - [Case Studies](#nixnan-case-studies)
4. [How NVBit Instrumentation Works](#nvbit-internals)
5. [Performance Considerations](#performance)
6. [Troubleshooting](#troubleshooting)
7. [References](#references)

---

## Introduction <a name="introduction"></a>

This repository contains two standalone NVBit tools for GPU program analysis:

| Tool | Purpose | Location |
|------|---------|----------|
| **Warpscope** | Idle warp detection and warp specialization recommendations | `nvbit_release/tools/warpscope/` |
| **Nixnan** | Floating-point exception detection (NaN, Inf, subnormal, div-by-zero) | `nvbit_release/tools/nixnan/` |

Both tools use [NVBit](https://github.com/NVlabs/NVBit) to instrument CUDA programs at the GPU binary (SASS) level via `LD_PRELOAD` — no source code modification or recompilation needed.

### System Requirements

- **OS**: Linux x86_64
- **CUDA**: 13+ recommended (12 may work but NVBit 1.7.7.3 recommends 13+)
- **GPU**: Compute capability >= 8.6 (Ampere, Ada Lovelace, Hopper)
- **Build**: GCC, Make, nvcc

### Quick Start

```bash
# Build both tools
make

# Run any CUDA program with warpscope
LD_PRELOAD=warpscope.so ./your_program

# Run any CUDA program with nixnan
LD_PRELOAD=nixnan.so ./your_program
```

---

## Part I: Warpscope — Idle Warp Detection & Specialization <a name="warpscope"></a>

### Background: Why Warp Utilization Matters <a name="warp-background"></a>

A GPU executes threads in groups of 32 called **warps**. When a kernel launches, the GPU scheduler assigns warps to Streaming Multiprocessors (SMs). In an ideal kernel, all warps do equal work. In practice, several common patterns cause warp imbalance:

| Pattern | Cause | Example |
|---------|-------|---------|
| **Causal mask** | Decoder attention: query i attends to keys 0..i only | GPT, LLaMA, Mistral |
| **Token pruning** | Skipping "easy" tokens at intermediate layers | DynamicViT, early-exit transformers |
| **MoE routing** | Uneven token distribution across experts | Mixtral, Switch Transformer |
| **Padded batching** | Variable-length sequences padded to max length | Production LLM serving |
| **Speculative decoding** | Rejected candidate tokens skip verification | Medusa, EAGLE |

Idle warps waste GPU cycles. **Warp specialization** repurposes these idle warps as data prefetchers — they load the next tile of data into shared memory while the busy warps compute on the current tile. This is the core technique behind Flash Attention 3.

Warpscope automatically detects which warps are idle, whether the pattern is consistent, and recommends producer/consumer splits.

### Installation <a name="warpscope-installation"></a>

```bash
git clone <your-fork-url>
cd warpscope

# Build (downloads NVBit automatically on first run)
make

# Or build only warpscope with a specific GPU architecture
cd nvbit_release/tools/warpscope
make ARCH=sm_89    # RTX 4090
# make ARCH=sm_86  # RTX 3090 / A100
# make ARCH=sm_90  # H100
```

This produces `warpscope.so` in the project root.

### Basic Usage <a name="warpscope-usage"></a>

```bash
# Instrument any CUDA program
LD_PRELOAD=warpscope.so ./your_cuda_program

# With a PyTorch script
LD_PRELOAD=warpscope.so python train.py

# Generate JSON report
WARPSCOPE_JSON=report.json LD_PRELOAD=warpscope.so ./your_program
```

### Environment Variables <a name="warpscope-env-vars"></a>

| Variable | Default | Description |
|---|---|---|
| `WARPSCOPE_THRESHOLD` | `0.5` | Utilization fraction below which a warp is "idle" (0.0–1.0) |
| `WARPSCOPE_VERBOSE` | `0` | Verbosity (0=normal, 1=per-function instrumentation details) |
| `WARPSCOPE_LOGFILE` | stderr | Redirect output to a file |
| `WARPSCOPE_JSON` | (none) | Path for JSON recommendation report |
| `WARPSCOPE_KERNEL` | (all) | Comma-separated kernel name whitelist |

### Understanding the Output <a name="warpscope-output"></a>

Warpscope produces three sections of output:

#### 1. Per-Kernel Warp Utilization Report

Printed after each kernel completes:

```
#warpscope: --- Warp Utilization Report: kernel [flash_attention_causal_fwd] ---
#warpscope: Total warps: 32 | Idle (<50% utilization): 16 | Partial occupancy: 0
#warpscope: Max warp duration: 13108278004 cycles

#warpscope: Idle warps (candidates for warp specialization):
#warpscope:        Block    Warp    SM      Duration     Util%   ActiveThreads
#warpscope:      (0,0,0)       3     0    1253064546      9.6%      32/32
#warpscope:      (1,0,0)       1     2    2850740116     21.7%      32/32
#warpscope:      (2,0,0)       1     4    4520595942     34.5%      32/32
#warpscope:      (3,0,0)       1     6    6190783424     47.2%      32/32

#warpscope: Busiest warps:
#warpscope:      (7,0,0)       2    14   13108278004    100.0%      32/32
```

**How to read this:**
- **Block (0,0,0)**: The CUDA block coordinates (blockIdx.x, y, z)
- **Warp 3**: Hardware warp ID within the SM
- **SM 0**: Which Streaming Multiprocessor this warp ran on
- **Duration**: Wall-clock cycles from first instruction to EXIT
- **Util%**: `duration / max_duration_across_all_warps * 100`
- **ActiveThreads**: How many of the 32 lanes were active at start/end

#### 2. Cross-Run Summary

Printed at program exit, aggregating all runs of each kernel:

```
#warpscope: ========== Warpscope Cross-Run Summary ==========
#warpscope: Kernel [flash_attention_causal_fwd] - 3 run(s)
#warpscope:   Run 1: 16/32 warps idle (50.0%)
#warpscope:   Run 2: 16/32 warps idle (50.0%)
#warpscope:   Run 3: 16/32 warps idle (50.0%)
#warpscope:   >> CONSISTENT: 16 warps are idle in every run.
```

This confirms whether the pattern is deterministic (exploitable) or varies with scheduling (harder to optimize).

#### 3. Specialization Recommendations

```
#warpscope: ========== Warp Specialization Recommendations ==========
#warpscope: Kernel [flash_attention_causal_fwd] (3 runs)
#warpscope:   Pattern: causal_triangle
#warpscope:   Avg idle fraction: 50.0%
#warpscope:   Consistency score: 1.00
#warpscope:   >> RECOMMENDATION: Enable warp specialization
#warpscope:   >> Suggested split: 16 producer warps, 16 consumer warps
#warpscope:   >> Strategy: Use position-dependent producer/consumer ratio.
```

### Workload Pattern Classification <a name="warpscope-patterns"></a>

Warpscope classifies idle warp patterns into five categories:

| Pattern | What it looks like | Typical cause | Specialization strategy |
|---|---|---|---|
| `causal_triangle` | Blocks 0..N/2 are idle, N/2..N are busy. Utilization increases monotonically. | Causal (decoder) attention mask | Position-dependent split: early blocks = more producers, late blocks = more consumers |
| `uniform_idle` | Entire blocks are idle while others are fully busy | MoE cold experts, token pruning | Idle blocks become full-time data prefetchers |
| `divergent` | Within each block, some warps idle, others busy | Branch divergence, odd/even workload | Warp-ID-based producer/consumer branching within blocks |
| `tail_idle` | Last few warps in blocks are idle | Non-aligned problem sizes (seq_len % block_m != 0) | Tail warps prefetch next tile |
| `balanced` | All warps ~100% utilized | Well-optimized kernel | No specialization needed |

### Case Studies <a name="warpscope-case-studies"></a>

#### Case Study 1: Causal Flash Attention (Decoder Transformers)

In decoder-style transformers (GPT, LLaMA), the attention mask is lower-triangular: query at position `i` only attends to keys `0..i`. This means:

- Block 0 (rows 0–127): each thread attends to at most 128 keys (~13% of sequence)
- Block 7 (rows 896–1023): each thread attends to up to 1024 keys (100%)

```bash
cd examples
nvcc -arch=compute_89 -lineinfo flash_attention_causal.cu -o flash_attention_causal
LD_PRELOAD=../warpscope.so ./flash_attention_causal
```

**Result**: 50% idle warps, consistency score 1.00, pattern `causal_triangle`. Blocks handling early sequence positions finish 5–10x faster than blocks handling late positions, but are held at `__syncthreads()` barriers.

**Why this matters**: This is exactly the insight that motivated Flash Attention 3's warp specialization — early-position warps can prefetch K/V tiles via TMA while late-position warps compute.

#### Case Study 2: Divergent Workload

```bash
nvcc -arch=compute_89 -lineinfo idle_warp_test.cu -o idle_warp_test
LD_PRELOAD=../warpscope.so ./idle_warp_test
```

The divergent kernel assigns heavy work to even warps (2000 iterations) and light work to odd warps (10 iterations). Warpscope detects 16/32 warps at ~28% utilization with consistency score 1.00.

#### Case Study 3: Token Pruning in FFN Layers

```bash
nvcc -arch=compute_89 -lineinfo idle_warp_ml.cu -o idle_warp_ml
LD_PRELOAD=../warpscope.so ./idle_warp_ml
```

When 50% of tokens are pruned at an FFN layer, blocks for pruned tokens skip the matrix multiply but are held at `__syncthreads()`. Warpscope identifies 34 consistently idle warps across runs.

### JSON Report Format <a name="warpscope-json"></a>

```bash
WARPSCOPE_JSON=report.json LD_PRELOAD=warpscope.so ./your_program
```

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
      {
        "block": [0, 0, 0],
        "total_warps": 4,
        "idle_warps": 4,
        "compute_warps": 0,
        "avg_idle_util": 0.096,
        "avg_compute_util": 0.000,
        "suggested_producers": 3,
        "suggested_consumers": 1
      }
    ]
  }]
}
```

---

## Part II: Nixnan — Floating-Point Exception Detection <a name="nixnan"></a>

### Background <a name="fp-background"></a>

NVIDIA GPUs lack hardware-level FP exception traps. Exceptional values (NaN, Inf) propagate silently through computations. Nixnan detects these at runtime by instrumenting every floating-point SASS instruction.

### Basic Usage <a name="nixnan-usage"></a>

```bash
LD_PRELOAD=nixnan.so ./your_cuda_program
```

### Environment Variables <a name="nixnan-env-vars"></a>

| Variable | Type | Default | Description |
|---|---|---|---|
| `INSTR_BEGIN` | Integer | 0 | Beginning of instruction interval to instrument |
| `INSTR_END` | Integer | UINT32_MAX | End of instruction interval |
| `TOOL_VERBOSE` | Integer | 0 | Enable verbose output |
| `ENABLE_FUN_DETAIL` | Integer | 0 | Show detailed function info |
| `PRINT_ILL_INSTR` | Integer | 0 | Print the offending instruction |
| `SAMPLING` | Integer | 0 | Instrument every Nth kernel invocation |
| `INSTR_MEM` | Integer | 0 | Instrument memory load/store for NaN detection |
| `HISTOGRAM` | Integer | 0 | Enable FP exponent range tracking |
| `BIN_SPEC_FILE` | String | (none) | JSON spec for targeted range monitoring |
| `LOGFILE` | String | stderr | Redirect output to file |
| `LINE_INFO` | Integer | 1 | Enable source line info (may cause crashes, set to 0 if needed) |

### Advanced Features <a name="nixnan-advanced"></a>

#### Tensor Core Monitoring

Nixnan instruments HMMA (half-precision MMA) instructions used by Tensor Cores:

```
#nixnan: error [infinity] detected in operand 0 of instruction
  HMMA.16816.F16 R20, R4.reuse, R16, RZ ; in function WMMAF16TensorCore of type f16
```

#### Exponent Histogram Tracking

```bash
HISTOGRAM=1 LD_PRELOAD=nixnan.so ./your_program
```

Outputs the exponent range per format (e.g., `Exponent range for f16: [-5, 3]`).

#### Targeted Range Monitoring

Create a JSON spec file:

```json
{
    "count": 128,
    "bf16": [],
    "f16": [[0,5],[-4,-1]],
    "f32": [],
    "f64": []
}
```

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=nixnan.so ./your_program
```

### Case Studies <a name="nixnan-case-studies"></a>

#### Flash Attention with Extreme Inputs

The `examples/basic.cu` test case injects extreme FP16 values into a tensor-core matrix multiply:

```bash
cd examples
nvcc -arch=compute_89 -lineinfo basic.cu -o basic
LD_PRELOAD=../nixnan.so ./basic
```

Nixnan detects:
- **Infinity** from `max_normal * max_normal` overflow
- **NaN** from `inf + (-inf)`
- **Subnormals** from `min_normal * 0.5` underflow

---

## How NVBit Instrumentation Works <a name="nvbit-internals"></a>

Both tools use NVBit's binary instrumentation framework:

1. **`LD_PRELOAD`** loads the tool as a shared library before the CUDA program starts
2. **`nvbit_at_cuda_event()`** is called on every CUDA driver API call (kernel launches, memory ops, etc.)
3. Before a kernel launches, the tool calls **`nvbit_get_instrs()`** to get all SASS instructions
4. **`nvbit_insert_call()`** injects a device function call before or after specific instructions:
   - `IPOINT_BEFORE`: call runs before the instruction
   - `IPOINT_AFTER`: call runs after the instruction
5. **`nvbit_add_call_arg_*()`** passes arguments (register values, constants, pointers) to the injected function
6. The injected device function runs on the GPU alongside the original kernel code

**Warpscope** injects `clock64()` timing probes at the first instruction and every EXIT/RET/BRK instruction.

**Nixnan** injects FP classification checks (NaN, Inf, subnormal, div-by-zero) around every floating-point instruction.

---

## Performance Considerations <a name="performance"></a>

| Tool | Mode | Typical Slowdown |
|------|------|-----------------|
| Warpscope | Default | 2–10x (only instruments first/last instructions) |
| Nixnan | Basic detection | 10–50x (instruments every FP instruction) |
| Nixnan | With line info | 20–100x |
| Nixnan | With sampling=64 | 2–10x |
| Nixnan | Memory instrumentation | 50–200x |

Warpscope has lower overhead than Nixnan because it only instruments 2 points per kernel (start + exits), while Nixnan instruments every floating-point instruction.

---

## Troubleshooting <a name="troubleshooting"></a>

### "instrumentation function not found in binary"

The tool's SM architecture doesn't match your GPU. Rebuild with the correct ARCH:

```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Rebuild (e.g., for compute capability 8.9)
cd nvbit_release/tools/warpscope && make clean && make ARCH=sm_89
```

### All warps show 100% utilization

This means the kernel is well-balanced — no idle warps to exploit. This is expected for:
- Non-causal (encoder) attention
- Element-wise operations
- Well-optimized matrix multiplies

### Crashes with LINE_INFO=1 (nixnan)

Disable line info: `LINE_INFO=0 LD_PRELOAD=nixnan.so ./your_program`

### Output mixed with program output

Redirect to a file:
```bash
WARPSCOPE_LOGFILE=warpscope.log LD_PRELOAD=warpscope.so ./your_program
LOGFILE=nixnan.log LD_PRELOAD=nixnan.so ./your_program
```

### NVBit: only one tool per process

You cannot `LD_PRELOAD` both `warpscope.so` and `nixnan.so` simultaneously. Run them in separate executions.

---

## References <a name="references"></a>

### Papers

1. **GPU-FPX**: Li, X., et al. (2023). "Design and Evaluation of GPU-FPX: A Low-Overhead Tool for Floating-Point Exception Detection in NVIDIA GPUs." *HPDC '23*. https://doi.org/10.1145/3588195.3592991

2. **Flash Attention 2**: Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."

3. **Flash Attention 3**: Shah, J., Dao, T., et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision."

4. **NVBit**: Villa, O., et al. (2019). "NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs." *MICRO '19*.

### Tools and Resources

- [NVBit](https://github.com/NVlabs/NVBit) — NVIDIA Binary Instrumentation Tool
- [GPU-FPX](https://github.com/LLNL/GPU-FPX) — Original FP exception detector
- [SC'25 Tutorial on FP Analysis Tools](https://fpanalysistools.org)
- [NVIDIA CUDA FP Documentation](https://docs.nvidia.com/cuda/floating-point/)

---

*This tutorial covers both the Warpscope idle warp detector and the Nixnan FP exception detector. For tool-specific details, see `README.md` and `CLAUDE.md`.*
