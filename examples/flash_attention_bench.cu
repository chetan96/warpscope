// flash_attention_bench.cu — Flash Attention 2 benchmark harness.
// Tests causal and non-causal FA2 across 20 configurations to measure
// latency and TFLOPS, and to create idle warp patterns for warpscope analysis.
//
// Compile: nvcc -arch=sm_89 -lineinfo flash_attention_bench.cu -o flash_attention_bench
// Run:     ./flash_attention_bench [filter]
// Warpscope: WARPSCOPE_JSON=report.json LD_PRELOAD=../warpscope.so ./flash_attention_bench

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

// ─── Causal Flash Attention 2 kernel ────────────────────────────────────────
// Each block handles block_m query rows. The causal mask means query at
// global row qi only attends to keys at positions 0..qi.
// Early blocks finish much faster → idle warp opportunity.

__global__ void flash_attention_causal_fwd(
    const half* __restrict__ Q,   // [seq_len, head_dim]
    const half* __restrict__ K,   // [seq_len, head_dim]
    const half* __restrict__ V,   // [seq_len, head_dim]
    half*       __restrict__ O,   // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    int block_m,
    float scale)
{
    const int row_start = blockIdx.x * block_m;
    const int tid = threadIdx.x;
    const int qi = row_start + tid;

    const bool valid_query = (tid < block_m && qi < seq_len);

    // Per-row accumulators (in FP32)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    // Shared memory for K and V tiles
    const int BLOCK_N = 32;
    extern __shared__ half smem[];
    half* K_tile = smem;
    half* V_tile = smem + BLOCK_N * head_dim;

    // Causal: no need to go beyond last query's position
    int max_kv = min(seq_len, row_start + block_m);

    for (int kv_start = 0; kv_start < max_kv; kv_start += BLOCK_N) {
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;

        // Cooperative load of K and V tiles into shared memory
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
                if (kj > qi) break;  // CAUSAL MASK

                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = __half2float(Q[qi * head_dim + d]);
                    float k_val = __half2float(K_tile[j * head_dim + d]);
                    score += q_val * k_val;
                }
                score *= scale;

                float m_old = m_i;
                m_i = fmaxf(m_i, score);
                float correction = expf(m_old - m_i);
                float p_ij = expf(score - m_i);

                l_i = l_i * correction + p_ij;
                for (int d = 0; d < head_dim; d++) {
                    float v_val = __half2float(V_tile[j * head_dim + d]);
                    acc[d] = acc[d] * correction + p_ij * v_val;
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

// ─── Non-causal Flash Attention 2 kernel ────────────────────────────────────
__global__ void flash_attention_full_fwd(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len,
    int head_dim,
    int block_m,
    float scale)
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

    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {
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

// ─── CPU reference (brute-force attention, FP32 accumulation) ───────────────
void cpu_attention_ref(const half* Q, const half* K, const half* V, float* ref_O,
                       int seq_len, int head_dim, bool causal) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int i = 0; i < seq_len; i++) {
        // Compute scores
        float max_score = -INFINITY;
        float* scores = (float*)malloc(seq_len * sizeof(float));
        for (int j = 0; j < seq_len; j++) {
            if (causal && j > i) {
                scores[j] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(Q[i * head_dim + d]) *
                       __half2float(K[j * head_dim + d]);
            }
            scores[j] = dot * scale;
            max_score = fmaxf(max_score, scores[j]);
        }
        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j < seq_len; j++) {
            scores[j] /= sum_exp;
        }
        // Weighted sum of V
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                val += scores[j] * __half2float(V[j * head_dim + d]);
            }
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
                printf("  MISMATCH at [%d,%d]: gpu=%.6f ref=%.6f diff=%.6f rel=%.4f\n",
                       i / head_dim, i % head_dim, gpu_val, ref_val, diff, rel);
            errors++;
        }
    }
    return errors == 0;
}

// ─── Random initialization ─────────────────────────────────────────────────
void fill_random(half* mat, size_t n, float stddev) {
    for (size_t i = 0; i < n; i++) {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        mat[i] = __float2half(stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2));
    }
}

// ─── FLOP computation ──────────────────────────────────────────────────────
// QK matmul: each query row i computes dot products with keys.
// PV matmul: attention weights × V.
// Non-causal: 2 * S * S * d (QK) + 2 * S * S * d (PV) = 4 * S^2 * d
// Causal: sum_{i=0}^{S-1} 2*(i+1)*d (QK) + same (PV) = 2*S*(S+1)*d
double compute_flops(int seq_len, int head_dim, bool causal) {
    if (causal) {
        return 2.0 * (double)seq_len * ((double)seq_len + 1.0) * (double)head_dim;
    } else {
        return 4.0 * (double)seq_len * (double)seq_len * (double)head_dim;
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
    // Category A — Causal seq_len sweep (d=64, block_m=128)
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

    // Category C — Varying block_m (seq=1024, d=64)
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

    // Category E — Non-aligned seq_len (compound idle: causal + tail)
    {"causal_noalign_500", 500, 64, 128, true},
    {"causal_noalign_300", 300, 64, 128, true},
};
static const int num_configs = sizeof(configs) / sizeof(configs[0]);

void run_benchmark(const FAConfig& cfg, bool do_verify) {
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

    if (cfg.causal) {
        printf("  Causal workload per block:\n");
        for (int b = 0; b < num_blocks && b < 8; b++) {
            int row_start = b * block_m;
            int first_qi = row_start;
            int last_qi = min(row_start + block_m - 1, seq_len - 1);
            int max_kv = min(seq_len, row_start + block_m);
            int kv_tiles = (max_kv + 31) / 32;
            int last_block_kv_tiles = (seq_len + 31) / 32;
            printf("    Block %d: rows [%d..%d], %d/%d KV tiles (%.0f%% of last block)\n",
                   b, first_qi, last_qi, kv_tiles, last_block_kv_tiles,
                   100.0f * kv_tiles / last_block_kv_tiles);
        }
        if (num_blocks > 8) printf("    ... (%d more blocks)\n", num_blocks - 8);
    }

    // Check for non-aligned tail
    int tail_threads_oob = 0;
    if (seq_len % block_m != 0) {
        int last_block_start = (num_blocks - 1) * block_m;
        tail_threads_oob = block_m - (seq_len - last_block_start);
        printf("  Non-aligned: last block has %d/%d threads OOB\n",
               tail_threads_oob, block_m);
    }

    size_t mat_elems = (size_t)seq_len * head_dim;
    size_t mat_bytes = mat_elems * sizeof(half);

    // Host allocation
    half* h_Q = (half*)malloc(mat_bytes);
    half* h_K = (half*)malloc(mat_bytes);
    half* h_V = (half*)malloc(mat_bytes);
    srand(42);
    fill_random(h_Q, mat_elems, 0.1f);
    fill_random(h_K, mat_elems, 0.1f);
    fill_random(h_V, mat_elems, 0.5f);

    // Device allocation
    half *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, mat_bytes));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, mat_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, mat_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, mat_bytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)head_dim);
    int smem_bytes = 2 * 32 * head_dim * (int)sizeof(half);  // K_tile + V_tile

    // Select kernel
    auto launch_kernel = [&]() {
        if (cfg.causal) {
            flash_attention_causal_fwd<<<num_blocks, block_m, smem_bytes>>>(
                d_Q, d_K, d_V, d_O, seq_len, head_dim, block_m, scale);
        } else {
            flash_attention_full_fwd<<<num_blocks, block_m, smem_bytes>>>(
                d_Q, d_K, d_V, d_O, seq_len, head_dim, block_m, scale);
        }
    };

    // Warmup
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMemset(d_O, 0, mat_bytes));
        launch_kernel();
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n", cudaGetErrorString(err));
            goto cleanup;
        }
    }

    {
        // Timed runs
        const int NUM_RUNS = 5;
        float times[NUM_RUNS];

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        for (int run = 0; run < NUM_RUNS; run++) {
            CHECK_CUDA(cudaMemset(d_O, 0, mat_bytes));
            CHECK_CUDA(cudaEventRecord(start));
            launch_kernel();
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            CHECK_CUDA(cudaEventElapsedTime(&times[run], start, stop));
        }

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        // Stats
        std::sort(times, times + NUM_RUNS);
        float median_ms = times[NUM_RUNS / 2];
        float min_ms = times[0];
        float max_ms = times[NUM_RUNS - 1];
        double flops = compute_flops(seq_len, head_dim, cfg.causal);
        double tflops = flops / (median_ms * 1e9);

        printf("  Latency: min=%.3f ms  median=%.3f ms  max=%.3f ms\n",
               min_ms, median_ms, max_ms);
        printf("  TFLOPS:  %.3f\n", tflops);

        // Machine-parseable line
        printf("RESULT\t%s\t%d\t%d\t%d\t%s\t%d\t%d\t%d\t%.4f\t%.3f\n",
               cfg.name, seq_len, head_dim, block_m,
               cfg.causal ? "causal" : "full",
               num_blocks, warps_per_block, total_warps,
               median_ms, tflops);

        // Correctness verification
        if (do_verify) {
            half* h_O = (half*)malloc(mat_bytes);
            CHECK_CUDA(cudaMemcpy(h_O, d_O, mat_bytes, cudaMemcpyDeviceToHost));
            float* ref_O = (float*)malloc(seq_len * head_dim * sizeof(float));
            cpu_attention_ref(h_Q, h_K, h_V, ref_O, seq_len, head_dim, cfg.causal);
            bool ok = verify_attention(h_O, ref_O, seq_len, head_dim);
            printf("  Correctness: %s\n", ok ? "PASS" : "FAIL");
            free(h_O);
            free(ref_O);
        }
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

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("BLOCK_N (KV tile width): 32\n\n");

    printf("RESULT\tname\tseq_len\thead_dim\tblock_m\tcausal\tnum_blocks\twarps_per_block\ttotal_warps\tmedian_ms\ttflops\n");

    for (int i = 0; i < num_configs; i++) {
        if (filter && !strstr(configs[i].name, filter)) continue;

        // Verify correctness only on small shapes (seq_len <= 256)
        bool do_verify = (configs[i].seq_len <= 256);

        run_benchmark(configs[i], do_verify);
    }

    printf("\nDone.\n");
    return 0;
}
