# CLAUDE.md — Project Guide for Warpscope

## Project Overview

Warpscope is a standalone NVBit tool for idle warp detection and warp specialization recommendations. It instruments CUDA kernels at the binary (SASS) level via `LD_PRELOAD` — no source modification needed.

Forked from nixnan (FP exception detector). Warpscope is an independent tool at `nvbit_release/tools/warpscope/`.

## Build Commands

```bash
# Build everything (downloads NVBit, builds nixnan + warpscope)
make

# Build only warpscope
cd nvbit_release/tools/warpscope && make ARCH=sm_89

# Clean
cd nvbit_release/tools/warpscope && make clean

# Compile an example
cd examples && nvcc -arch=compute_89 -lineinfo flash_attention_causal.cu -o flash_attention_causal

# Run with warpscope
LD_PRELOAD=../warpscope.so ./flash_attention_causal

# Run with JSON output
WARPSCOPE_JSON=report.json LD_PRELOAD=../warpscope.so ./flash_attention_causal
```

## Architecture

The tool has 3 layers:

1. **Device instrumentation** (`inject_funcs.cu`) — device functions injected into kernel SASS via NVBit. Compiled with `-Xptxas -astoolspatch --keep-device-functions`.

2. **NVBit callbacks** (`warpscope.cu`) — hooks into kernel launches, calls instrumentation, manages the managed memory buffer. This is the only file that includes `nvbit.h` / `nvbit_tool.h`.

3. **Host analysis** (`analysis.cu`, `specialization.cu`) — reads timing data from managed memory after `cudaDeviceSynchronize()`, computes metrics, generates reports. Compiled without `-dc` to avoid nvlink symbol conflicts with `warpscope.cu`.

## Key Design Decisions

- **Managed memory** (not NVBit channel): The timing buffer uses `__managed__` memory. This is simpler than the channel-based approach used by nixnan — no background receiver thread needed. We just `cudaDeviceSynchronize()` after kernel exit and read the buffer directly.

- **EXIT instruction instrumentation**: We instrument ALL `EXIT`/`RET`/`BRK` instructions (not just the last instruction in the instruction list) because warps may exit via different code paths. This was a bug we hit during development.

- **Separate compilation for analysis.cu**: Must be compiled without `-dc` (device compilation) because it doesn't contain device code but includes CUDA runtime headers. If compiled with `-dc`, nvlink produces "multiple definition" errors against `warpscope.cu`.

- **No IDLE_WARP env var gating**: Unlike the nixnan integration (where `IDLE_WARP=1` was needed to enable the feature), warpscope is always active — it's the tool's sole purpose.

## File Roles

| File | Compiled With | Role |
|---|---|---|
| `warpscope.cu` | `-dc` (device code) | NVBit callbacks, instrument_function, managed buffer decl |
| `inject_funcs.cu` | `-astoolspatch` | Device timer start/end functions |
| `analysis.cu` | no `-dc` | Host-side per-kernel analysis, cross-run tracking |
| `specialization.cu` | no `-dc` | Pattern classification, recommendation engine |
| `wsout.cc` | g++ (pure C++) | Logging with `#warpscope:` prefix |
| `warp_timing.cuh` | header only | Shared data structures |

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `WARPSCOPE_THRESHOLD` | `0.5` | Idle utilization threshold |
| `WARPSCOPE_VERBOSE` | `0` | Verbosity |
| `WARPSCOPE_LOGFILE` | stderr | Log file path |
| `WARPSCOPE_JSON` | (none) | JSON report output path |
| `WARPSCOPE_KERNEL` | (all) | Kernel name whitelist (comma-separated) |

## GPU Compatibility

- Default build targets `sm_89` (RTX 4090 / Ada Lovelace)
- Change with `make ARCH=sm_86` for Ampere, `make ARCH=sm_90` for Hopper
- NVBit 1.7.7.3 requires CUDA 13+ for reliable instrumentation

## Testing

Best test cases for verifying idle warp detection:

- `examples/flash_attention_causal.cu` — 50% idle warps consistently (causal mask creates triangular workload)
- `examples/idle_warp_test.cu` — divergent kernel shows 16/32 warps at ~28% utilization
- `examples/idle_warp_ml.cu` — token pruning shows consistently idle warps in FFN layers

## Common Issues

- **"instrumentation function warpscope_timer_start not found"**: ARCH mismatch. Rebuild with the correct SM version for your GPU (check `nvidia-smi --query-gpu=compute_cap --format=csv`).
- **All warps show 100% utilization**: The kernel may have warps that exit too fast for the timer to capture meaningful differences. This is expected for well-balanced kernels like non-causal Flash Attention.
- **WS_MAX_WARPS overflow**: If a kernel launches >4096 warps, some records are silently dropped. Increase `WS_MAX_WARPS` in `warp_timing.cuh` if needed.

## Relationship to Nixnan

Nixnan (FP exception detector) is untouched at `nvbit_release/tools/nixnan/`. The two tools are independent — you can use either via `LD_PRELOAD`. They cannot be used simultaneously on the same program (NVBit limitation: one tool per process).
