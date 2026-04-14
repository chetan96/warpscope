// flash_attention_specialized.cu — Optimized Flash Attention 2 variants.
// Demonstrates improvements informed by warpscope idle warp analysis:
//   Variant A: Double-buffered KV pipeline (1 sync vs 2 per KV tile)
//   Variant B: Dynamic block scheduling (atomic work-stealing)
//   Variant C: Early-exit warp repurposing (idle warps prefetch next KV tile)
//
// Compile: nvcc -arch=sm_89 -lineinfo flash_attention_specialized.cu -o flash_attention_specialized
// Run:     ./flash_attention_specialized [filter]
// Warpscope: WARPSCOPE_JSON=report.json LD_PRELOAD=../warpscope.so ./flash_attention_specialized

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── Baseline causal FA2 (single-buffered, 2 syncs per KV tile) ────────────

__global__ void fa2_causal_baseline(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len, int head_dim, int block_m, float scale)
{
    const int row_start = blockIdx.x * block_m;
    const int tid = threadIdx.x;
    const int qi = row_start + tid;
    const bool valid_query = (tid < block_m && qi < seq_len);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    const int BLOCK_N = 32;
    extern __shared__ half smem[];
    half* K_tile = smem;
    half* V_tile = smem + BLOCK_N * head_dim;

    int max_kv = min(seq_len, row_start + block_m);

    for (int kv_start = 0; kv_start < max_kv; kv_start += BLOCK_N) {
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;

        int load_iters = (BLOCK_N * head_dim + block_m - 1) / block_m;
        for (int li = 0; li < load_iters; li++) {
            int load_idx = tid + li * block_m;
            if (load_idx < BLOCK_N * head_dim) {
                int kv_row = load_idx / head_dim;
                int kv_col = load_idx % head_dim;
                int global_kv = kv_start + kv_row;
                if (global_kv < seq_len) {
                    K_tile[load_idx] = K[global_kv * head_dim + kv_col];
                    V_tile[load_idx] = V[global_kv * head_dim + kv_col];
                } else {
                    K_tile[load_idx] = __float2half(0.0f);
                    V_tile[load_idx] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        if (valid_query) {
            for (int j = 0; j < tile_size; j++) {
                int kj = kv_start + j;
                if (kj > qi) break;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(Q[qi * head_dim + d]) *
                             __half2float(K_tile[j * head_dim + d]);
                }
                score *= scale;

                float m_old = m_i;
                m_i = fmaxf(m_i, score);
                float correction = expf(m_old - m_i);
                float p_ij = expf(score - m_i);
                l_i = l_i * correction + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * correction +
                             p_ij * __half2float(V_tile[j * head_dim + d]);
                }
            }
        }
        __syncthreads();
    }

    if (valid_query) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = __float2half(acc[d] / l_i);
        }
    }
}

// ─── Variant A: Double-Buffered KV Pipeline ────────────────────────────────
// Two sets of K_tile/V_tile in shared memory. Loads for tile t+1 are issued
// BEFORE computing tile t, allowing the GPU scheduler to overlap memory
// requests with attention compute. Reduces syncs from 2 to 1 per iteration.

__global__ void fa2_causal_doublebuf(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len, int head_dim, int block_m, float scale)
{
    const int row_start = blockIdx.x * block_m;
    const int tid = threadIdx.x;
    const int qi = row_start + tid;
    const bool valid_query = (tid < block_m && qi < seq_len);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    const int BLOCK_N = 32;
    // Double buffer: 2 × (K_tile + V_tile)
    extern __shared__ half smem[];
    int tile_elems = BLOCK_N * head_dim;
    half* K_buf[2] = { smem, smem + 2 * tile_elems };
    half* V_buf[2] = { smem + tile_elems, smem + 3 * tile_elems };

    int max_kv = min(seq_len, row_start + block_m);
    int num_kv_tiles = (max_kv + BLOCK_N - 1) / BLOCK_N;

    int load_iters = (BLOCK_N * head_dim + block_m - 1) / block_m;

    // Prologue: load first tile into buffer 0
    if (num_kv_tiles > 0) {
        for (int li = 0; li < load_iters; li++) {
            int load_idx = tid + li * block_m;
            if (load_idx < tile_elems) {
                int kv_row = load_idx / head_dim;
                int kv_col = load_idx % head_dim;
                int global_kv = kv_row;
                if (global_kv < seq_len) {
                    K_buf[0][load_idx] = K[global_kv * head_dim + kv_col];
                    V_buf[0][load_idx] = V[global_kv * head_dim + kv_col];
                } else {
                    K_buf[0][load_idx] = __float2half(0.0f);
                    V_buf[0][load_idx] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();
    }

    for (int t = 0; t < num_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kv_start = t * BLOCK_N;
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;
        int next_kv_start = (t + 1) * BLOCK_N;

        // Issue loads for NEXT tile first (overlaps with compute below)
        if (t + 1 < num_kv_tiles) {
            for (int li = 0; li < load_iters; li++) {
                int load_idx = tid + li * block_m;
                if (load_idx < tile_elems) {
                    int kv_row = load_idx / head_dim;
                    int kv_col = load_idx % head_dim;
                    int global_kv = next_kv_start + kv_row;
                    if (global_kv < seq_len) {
                        K_buf[nxt][load_idx] = K[global_kv * head_dim + kv_col];
                        V_buf[nxt][load_idx] = V[global_kv * head_dim + kv_col];
                    } else {
                        K_buf[nxt][load_idx] = __float2half(0.0f);
                        V_buf[nxt][load_idx] = __float2half(0.0f);
                    }
                }
            }
        }

        // Compute attention for current tile
        if (valid_query) {
            for (int j = 0; j < tile_size; j++) {
                int kj = kv_start + j;
                if (kj > qi) break;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(Q[qi * head_dim + d]) *
                             __half2float(K_buf[cur][j * head_dim + d]);
                }
                score *= scale;

                float m_old = m_i;
                m_i = fmaxf(m_i, score);
                float correction = expf(m_old - m_i);
                float p_ij = expf(score - m_i);
                l_i = l_i * correction + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * correction +
                             p_ij * __half2float(V_buf[cur][j * head_dim + d]);
                }
            }
        }

        __syncthreads();  // single sync: ensures both load and compute are done
    }

    if (valid_query) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = __float2half(acc[d] / l_i);
        }
    }
}

// ─── Variant B: Dynamic Block Scheduling ───────────────────────────────────
// Instead of static blockIdx.x → query tile mapping, blocks atomically grab
// the next unprocessed query tile. This interleaves heavy (late) and light
// (early) causal blocks across SMs, improving utilization balance.

__device__ int g_next_block_B;

__global__ void fa2_causal_dynamic(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len, int head_dim, int block_m, float scale,
    int num_blocks_total)
{
    const int tid = threadIdx.x;

    // Dynamic work-stealing: grab next query tile
    __shared__ int my_block_id;
    if (tid == 0) {
        my_block_id = atomicAdd(&g_next_block_B, 1);
    }
    __syncthreads();

    if (my_block_id >= num_blocks_total) return;  // no more work

    const int row_start = my_block_id * block_m;
    const int qi = row_start + tid;
    const bool valid_query = (tid < block_m && qi < seq_len);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    const int BLOCK_N = 32;
    extern __shared__ half smem[];
    half* K_tile = smem;
    half* V_tile = smem + BLOCK_N * head_dim;

    int max_kv = min(seq_len, row_start + block_m);

    for (int kv_start = 0; kv_start < max_kv; kv_start += BLOCK_N) {
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;

        int load_iters = (BLOCK_N * head_dim + block_m - 1) / block_m;
        for (int li = 0; li < load_iters; li++) {
            int load_idx = tid + li * block_m;
            if (load_idx < BLOCK_N * head_dim) {
                int kv_row = load_idx / head_dim;
                int kv_col = load_idx % head_dim;
                int global_kv = kv_start + kv_row;
                if (global_kv < seq_len) {
                    K_tile[load_idx] = K[global_kv * head_dim + kv_col];
                    V_tile[load_idx] = V[global_kv * head_dim + kv_col];
                } else {
                    K_tile[load_idx] = __float2half(0.0f);
                    V_tile[load_idx] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        if (valid_query) {
            for (int j = 0; j < tile_size; j++) {
                int kj = kv_start + j;
                if (kj > qi) break;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(Q[qi * head_dim + d]) *
                             __half2float(K_tile[j * head_dim + d]);
                }
                score *= scale;

                float m_old = m_i;
                m_i = fmaxf(m_i, score);
                float correction = expf(m_old - m_i);
                float p_ij = expf(score - m_i);
                l_i = l_i * correction + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * correction +
                             p_ij * __half2float(V_tile[j * head_dim + d]);
                }
            }
        }
        __syncthreads();
    }

    if (valid_query) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = __float2half(acc[d] / l_i);
        }
    }
}

// ─── Variant C: Early-Exit Warp Repurposing ────────────────────────────────
// Combines double-buffering with productive use of idle barrier time.
// In causal attention, warps handling early query rows finish their inner loop
// quickly. Instead of idling at __syncthreads(), these warps detect they have
// no more causal work for the current tile and switch to prefetching the next
// KV tile into the alternate buffer.
//
// This is the key experiment: can we convert the ~50% causal idle time
// into useful prefetch work?

__global__ void fa2_causal_repurpose(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len, int head_dim, int block_m, float scale)
{
    const int row_start = blockIdx.x * block_m;
    const int tid = threadIdx.x;
    const int qi = row_start + tid;
    const bool valid_query = (tid < block_m && qi < seq_len);
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    const int BLOCK_N = 32;
    extern __shared__ half smem[];
    int tile_elems = BLOCK_N * head_dim;
    half* K_buf[2] = { smem, smem + 2 * tile_elems };
    half* V_buf[2] = { smem + tile_elems, smem + 3 * tile_elems };

    // Shared coordination: how many warps are done with compute for this tile
    __shared__ int idle_warp_count;

    int max_kv = min(seq_len, row_start + block_m);
    int num_kv_tiles = (max_kv + BLOCK_N - 1) / BLOCK_N;
    int load_iters = (BLOCK_N * head_dim + block_m - 1) / block_m;

    // Prologue: load first tile into buffer 0
    if (num_kv_tiles > 0) {
        for (int li = 0; li < load_iters; li++) {
            int load_idx = tid + li * block_m;
            if (load_idx < tile_elems) {
                int kv_row = load_idx / head_dim;
                int kv_col = load_idx % head_dim;
                if (kv_row < seq_len) {
                    K_buf[0][load_idx] = K[kv_row * head_dim + kv_col];
                    V_buf[0][load_idx] = V[kv_row * head_dim + kv_col];
                } else {
                    K_buf[0][load_idx] = __float2half(0.0f);
                    V_buf[0][load_idx] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();
    }

    for (int t = 0; t < num_kv_tiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kv_start = t * BLOCK_N;
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;
        int next_kv_start = (t + 1) * BLOCK_N;

        // Reset idle warp tracking
        if (tid == 0) idle_warp_count = 0;
        // (idle_warp_count tracks how many warps have no causal work)
        __syncthreads();

        // --- Phase 1: Compute attention for current tile ---
        // Each warp determines if it has causal work in this tile.
        // A warp is "idle" for this tile if all its query rows have qi < kv_start
        // (i.e., the warp's queries only attend to keys before this tile).
        int warp_first_qi = row_start + warp_id * 32;
        int warp_last_qi = min(warp_first_qi + 31, seq_len - 1);
        bool warp_has_work = valid_query && (warp_last_qi >= kv_start);

        if (warp_has_work && valid_query) {
            for (int j = 0; j < tile_size; j++) {
                int kj = kv_start + j;
                if (kj > qi) break;

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __half2float(Q[qi * head_dim + d]) *
                             __half2float(K_buf[cur][j * head_dim + d]);
                }
                score *= scale;

                float m_old = m_i;
                m_i = fmaxf(m_i, score);
                float correction = expf(m_old - m_i);
                float p_ij = expf(score - m_i);
                l_i = l_i * correction + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * correction +
                             p_ij * __half2float(V_buf[cur][j * head_dim + d]);
                }
            }
        }

        // --- Phase 2: Idle warps prefetch next tile ---
        // Warps that had no causal work (or are beyond seq_len) help prefetch.
        if (!warp_has_work && t + 1 < num_kv_tiles) {
            // Signal that this warp is idle
            if (lane_id == 0) {
                atomicAdd(&idle_warp_count, 1);
            }
            __syncwarp();

            // Each idle warp contributes to prefetching
            // We use (warp_id, lane_id) within the idle warp pool
            // Simple approach: each idle warp loads a strided portion of the tile
            int idle_tid = lane_id;  // within this warp
            int idle_stride = 32;    // one warp at a time, simple striding

            for (int i = idle_tid; i < tile_elems; i += idle_stride) {
                int kv_row = i / head_dim;
                int kv_col = i % head_dim;
                int global_kv = next_kv_start + kv_row;
                if (global_kv < seq_len) {
                    K_buf[nxt][i] = K[global_kv * head_dim + kv_col];
                    V_buf[nxt][i] = V[global_kv * head_dim + kv_col];
                } else {
                    K_buf[nxt][i] = __float2half(0.0f);
                    V_buf[nxt][i] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // --- Phase 3: If no idle warps did prefetch, all warps do it together ---
        // This handles the case where all warps had compute work (late blocks)
        if (t + 1 < num_kv_tiles && idle_warp_count == 0) {
            for (int li = 0; li < load_iters; li++) {
                int load_idx = tid + li * block_m;
                if (load_idx < tile_elems) {
                    int kv_row = load_idx / head_dim;
                    int kv_col = load_idx % head_dim;
                    int global_kv = next_kv_start + kv_row;
                    if (global_kv < seq_len) {
                        K_buf[nxt][load_idx] = K[global_kv * head_dim + kv_col];
                        V_buf[nxt][load_idx] = V[global_kv * head_dim + kv_col];
                    } else {
                        K_buf[nxt][load_idx] = __float2half(0.0f);
                        V_buf[nxt][load_idx] = __float2half(0.0f);
                    }
                }
            }
            __syncthreads();
        }
    }

    if (valid_query) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = __float2half(acc[d] / l_i);
        }
    }
}

// ─── Benchmark Harness ─────────────────────────────────────────────────────

struct FAConfig {
    const char* name;
    int seq_len;
    int head_dim;
    int block_m;
    bool causal;
};

static FAConfig configs[] = {
    // Category A — Causal seq_len sweep
    {"causal_seq128",    128,  64, 128, true},
    {"causal_seq256",    256,  64, 128, true},
    {"causal_seq512",    512,  64, 128, true},
    {"causal_seq1024",  1024,  64, 128, true},
    {"causal_seq2048",  2048,  64, 128, true},
    {"causal_seq4096",  4096,  64, 128, true},

    // Category B — Non-causal comparison
    {"full_seq128",      128,  64, 128, false},
    {"full_seq512",      512,  64, 128, false},
    {"full_seq2048",    2048,  64, 128, false},

    // Category C — Varying block_m
    {"causal_bm32",     1024,  64,  32, true},
    {"causal_bm64",     1024,  64,  64, true},
    {"causal_bm128",    1024,  64, 128, true},
    {"causal_bm256",    1024,  64, 256, true},
    {"causal_bm512",    1024,  64, 512, true},

    // Category D — Real LLM configs
    {"llama2_7b",       1024, 128, 128, true},
    {"llama2_7b_long",  2048, 128, 128, true},
    {"gpt2_small",      1024,  64, 128, true},
    {"gpt2_medium",     1024, 128, 128, true},

    // Category E — Non-aligned
    {"causal_noalign_500", 500, 64, 128, true},
    {"causal_noalign_300", 300, 64, 128, true},
};
static const int num_configs = sizeof(configs) / sizeof(configs[0]);

// FLOP computation
double compute_flops(int seq_len, int head_dim, bool causal) {
    if (causal) {
        return 2.0 * (double)seq_len * ((double)seq_len + 1.0) * (double)head_dim;
    } else {
        return 4.0 * (double)seq_len * (double)seq_len * (double)head_dim;
    }
}

void fill_random(half* mat, size_t n, float stddev) {
    for (size_t i = 0; i < n; i++) {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        mat[i] = __float2half(stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2));
    }
}

// CPU reference for correctness
void cpu_attention_ref(const half* Q, const half* K, const half* V, float* ref_O,
                       int seq_len, int head_dim, bool causal) {
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int i = 0; i < seq_len; i++) {
        float max_score = -INFINITY;
        float* scores = (float*)malloc(seq_len * sizeof(float));
        for (int j = 0; j < seq_len; j++) {
            if (causal && j > i) { scores[j] = -INFINITY; continue; }
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += __half2float(Q[i * head_dim + d]) * __half2float(K[j * head_dim + d]);
            scores[j] = dot * scale;
            max_score = fmaxf(max_score, scores[j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++)
                val += (scores[j] / sum_exp) * __half2float(V[j * head_dim + d]);
            ref_O[i * head_dim + d] = val;
        }
        free(scores);
    }
}

bool verify_attention(const half* gpu_O, const float* ref_O, int seq_len, int head_dim) {
    int errors = 0;
    for (int i = 0; i < seq_len * head_dim && errors < 10; i++) {
        float gpu_val = __half2float(gpu_O[i]);
        float ref_val = ref_O[i];
        float diff = fabsf(gpu_val - ref_val);
        float rel = diff / (fabsf(ref_val) + 1e-6f);
        if (rel > 1e-2f && diff > 1e-3f) {
            if (errors == 0)
                printf("  MISMATCH at [%d,%d]: gpu=%.6f ref=%.6f rel=%.4f\n",
                       i / head_dim, i % head_dim, gpu_val, ref_val, rel);
            errors++;
        }
    }
    return errors == 0;
}

// Timing helper
float time_fa_kernel(void (*launch_fn)(const half*, const half*, const half*,
                                        half*, int, int, int, float,
                                        int, int, int),
                     const half* d_Q, const half* d_K, const half* d_V,
                     half* d_O, int seq_len, int head_dim, int block_m,
                     float scale, int num_blocks, int smem_bytes,
                     int num_runs)
{
    // Warmup
    for (int i = 0; i < 2; i++) {
        launch_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim, block_m, scale,
                  num_blocks, smem_bytes, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float times[32];
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int run = 0; run < num_runs; run++) {
        CHECK_CUDA(cudaMemset(d_O, 0, (size_t)seq_len * head_dim * sizeof(half)));
        CHECK_CUDA(cudaEventRecord(start));
        launch_fn(d_Q, d_K, d_V, d_O, seq_len, head_dim, block_m, scale,
                  num_blocks, smem_bytes, 0);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[run], start, stop));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::sort(times, times + num_runs);
    return times[num_runs / 2];
}

// Launch wrappers (uniform signature for time_fa_kernel)
void launch_baseline(const half* Q, const half* K, const half* V, half* O,
                     int seq_len, int head_dim, int block_m, float scale,
                     int num_blocks, int smem_bytes, int /*unused*/) {
    fa2_causal_baseline<<<num_blocks, block_m, smem_bytes>>>(
        Q, K, V, O, seq_len, head_dim, block_m, scale);
}

void launch_doublebuf(const half* Q, const half* K, const half* V, half* O,
                      int seq_len, int head_dim, int block_m, float scale,
                      int num_blocks, int smem_bytes, int /*unused*/) {
    int dbuf_smem = smem_bytes * 2;  // double buffer
    fa2_causal_doublebuf<<<num_blocks, block_m, dbuf_smem>>>(
        Q, K, V, O, seq_len, head_dim, block_m, scale);
}

void launch_dynamic(const half* Q, const half* K, const half* V, half* O,
                    int seq_len, int head_dim, int block_m, float scale,
                    int num_blocks, int smem_bytes, int /*unused*/) {
    // Reset global counter
    int zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(g_next_block_B, &zero, sizeof(int)));
    fa2_causal_dynamic<<<num_blocks, block_m, smem_bytes>>>(
        Q, K, V, O, seq_len, head_dim, block_m, scale, num_blocks);
}

void launch_repurpose(const half* Q, const half* K, const half* V, half* O,
                      int seq_len, int head_dim, int block_m, float scale,
                      int num_blocks, int smem_bytes, int /*unused*/) {
    int dbuf_smem = smem_bytes * 2;  // double buffer
    fa2_causal_repurpose<<<num_blocks, block_m, dbuf_smem>>>(
        Q, K, V, O, seq_len, head_dim, block_m, scale);
}

// Non-causal baseline for Category B
void launch_full(const half* Q, const half* K, const half* V, half* O,
                 int seq_len, int head_dim, int block_m, float scale,
                 int num_blocks, int smem_bytes, int /*unused*/) {
    // Reuse the baseline kernel structure but non-causal
    // For simplicity, Category B configs just use baseline (no optimization needed
    // since they have no idle warps)
    fa2_causal_baseline<<<num_blocks, block_m, smem_bytes>>>(
        Q, K, V, O, seq_len, head_dim, block_m, scale);
}

void run_comparison(const FAConfig& cfg, bool do_verify) {
    int seq_len = cfg.seq_len;
    int head_dim = cfg.head_dim;
    int block_m = cfg.block_m;
    int num_blocks = (seq_len + block_m - 1) / block_m;
    int warps_per_block = (block_m + 31) / 32;
    int total_warps = num_blocks * warps_per_block;

    printf("\n─── %s: seq=%d d=%d bm=%d causal=%s ───\n",
           cfg.name, seq_len, head_dim, block_m, cfg.causal ? "YES" : "no");
    printf("  Grid: %d blocks  Warps/block: %d  Total warps: %d\n",
           num_blocks, warps_per_block, total_warps);

    size_t mat_elems = (size_t)seq_len * head_dim;
    size_t mat_bytes = mat_elems * sizeof(half);

    half* h_Q = (half*)malloc(mat_bytes);
    half* h_K = (half*)malloc(mat_bytes);
    half* h_V = (half*)malloc(mat_bytes);
    srand(42);
    fill_random(h_Q, mat_elems, 0.1f);
    fill_random(h_K, mat_elems, 0.1f);
    fill_random(h_V, mat_elems, 0.5f);

    half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, mat_bytes));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, mat_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, mat_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, mat_bytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)head_dim);
    int smem_bytes = 2 * 32 * head_dim * (int)sizeof(half);

    const int NUM_RUNS = 5;
    double flops = compute_flops(seq_len, head_dim, cfg.causal);

    if (!cfg.causal) {
        // Non-causal: just run baseline (no optimization variants needed)
        float baseline_ms = time_fa_kernel(launch_baseline, d_Q, d_K, d_V, d_O,
                                           seq_len, head_dim, block_m, scale,
                                           num_blocks, smem_bytes, NUM_RUNS);
        double baseline_tflops = flops / (baseline_ms * 1e9);

        printf("  Baseline: %.3f ms  %.3f TFLOPS\n", baseline_ms, baseline_tflops);
        printf("COMPARE\t%s\t%.4f\t%.3f\t-\t-\t-\t-\t-\t-\t-\t-\t-\n",
               cfg.name, baseline_ms, baseline_tflops);

        goto cleanup;
    }

    {
        // Causal: run all 4 variants
        float baseline_ms = time_fa_kernel(launch_baseline, d_Q, d_K, d_V, d_O,
                                           seq_len, head_dim, block_m, scale,
                                           num_blocks, smem_bytes, NUM_RUNS);
        double baseline_tflops = flops / (baseline_ms * 1e9);

        float dbuf_ms = time_fa_kernel(launch_doublebuf, d_Q, d_K, d_V, d_O,
                                       seq_len, head_dim, block_m, scale,
                                       num_blocks, smem_bytes, NUM_RUNS);
        double dbuf_tflops = flops / (dbuf_ms * 1e9);

        float dynamic_ms = time_fa_kernel(launch_dynamic, d_Q, d_K, d_V, d_O,
                                          seq_len, head_dim, block_m, scale,
                                          num_blocks, smem_bytes, NUM_RUNS);
        double dynamic_tflops = flops / (dynamic_ms * 1e9);

        float repurpose_ms = time_fa_kernel(launch_repurpose, d_Q, d_K, d_V, d_O,
                                            seq_len, head_dim, block_m, scale,
                                            num_blocks, smem_bytes, NUM_RUNS);
        double repurpose_tflops = flops / (repurpose_ms * 1e9);

        // Correctness verification on small shapes
        if (do_verify) {
            float* ref_O = (float*)malloc(seq_len * head_dim * sizeof(float));
            half* h_O = (half*)malloc(mat_bytes);
            cpu_attention_ref(h_Q, h_K, h_V, ref_O, seq_len, head_dim, cfg.causal);

            // Check each variant
            const char* variant_names[] = {"baseline", "doublebuf", "dynamic", "repurpose"};
            void (*launchers[])(const half*, const half*, const half*, half*,
                                int, int, int, float, int, int, int) = {
                launch_baseline, launch_doublebuf, launch_dynamic, launch_repurpose
            };

            for (int v = 0; v < 4; v++) {
                CHECK_CUDA(cudaMemset(d_O, 0, mat_bytes));
                launchers[v](d_Q, d_K, d_V, d_O, seq_len, head_dim, block_m,
                            scale, num_blocks, smem_bytes, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaMemcpy(h_O, d_O, mat_bytes, cudaMemcpyDeviceToHost));
                bool ok = verify_attention(h_O, ref_O, seq_len, head_dim);
                printf("  Correctness [%s]: %s\n", variant_names[v], ok ? "PASS" : "FAIL");
            }

            free(ref_O);
            free(h_O);
        }

        printf("  Baseline:      %8.3f ms  %8.3f TFLOPS\n", baseline_ms, baseline_tflops);
        printf("  Double-buf:    %8.3f ms  %8.3f TFLOPS  (%.2fx)\n",
               dbuf_ms, dbuf_tflops, baseline_ms / dbuf_ms);
        printf("  Dynamic:       %8.3f ms  %8.3f TFLOPS  (%.2fx)\n",
               dynamic_ms, dynamic_tflops, baseline_ms / dynamic_ms);
        printf("  Repurpose:     %8.3f ms  %8.3f TFLOPS  (%.2fx)\n",
               repurpose_ms, repurpose_tflops, baseline_ms / repurpose_ms);

        // Machine-parseable
        printf("COMPARE\t%s\t%.4f\t%.3f\t%.4f\t%.3f\t%.2f\t%.4f\t%.3f\t%.2f\t%.4f\t%.3f\t%.2f\n",
               cfg.name,
               baseline_ms, baseline_tflops,
               dbuf_ms, dbuf_tflops, baseline_ms / dbuf_ms,
               dynamic_ms, dynamic_tflops, baseline_ms / dynamic_ms,
               repurpose_ms, repurpose_tflops, baseline_ms / repurpose_ms);
    }

cleanup:
    free(h_Q); free(h_K); free(h_V);
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
}

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("BLOCK_N (KV tile width): 32\n\n");

    printf("COMPARE\tname\tbaseline_ms\tbaseline_tflops\tdbuf_ms\tdbuf_tflops\tdbuf_speedup\tdynamic_ms\tdynamic_tflops\tdynamic_speedup\trepurpose_ms\trepurpose_tflops\trepurpose_speedup\n");

    for (int i = 0; i < num_configs; i++) {
        if (filter && !strstr(configs[i].name, filter)) continue;

        bool do_verify = (configs[i].seq_len <= 256);
        run_comparison(configs[i], do_verify);
    }

    printf("\nDone.\n");
    return 0;
}
