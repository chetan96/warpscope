// flash_attention_causal.cu — Causal (decoder) Flash Attention 2 with shared memory
//
// In decoder-style transformers (GPT, LLaMA, Mistral), the attention mask
// is causal: position i can only attend to positions 0..i. The upper triangle
// of the attention matrix is masked to -infinity.
//
// This creates a triangular workload:
//   - Query row 0:   attends to 1 key   (almost no work)
//   - Query row N/2: attends to N/2 keys (half work)
//   - Query row N-1: attends to N keys   (full work)
//
// Warps handling early rows finish much faster → idle warp opportunity.
//
// This kernel uses shared memory + __syncthreads() like production FA2,
// which forces idle warps to stay alive at barriers.
//
// Compile: nvcc -arch=compute_89 -lineinfo flash_attention_causal.cu -o flash_attention_causal
// Run:     IDLE_WARP=1 LD_PRELOAD=../nixnan.so ./flash_attention_causal

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --------------------------------------------------------------------------
// Causal Flash Attention 2 kernel with shared memory tiling
//
// Each block handles BLOCK_M query rows. Threads cooperatively load
// K/V tiles into shared memory, then each thread computes attention
// for its assigned query row — but only over causally valid keys.
//
// The causal mask means: query at global row qi can only attend to
// keys at positions 0..qi. If the entire KV tile is above qi for all
// threads in the block, the tile is skipped entirely.
// --------------------------------------------------------------------------
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

    // Even threads beyond seq_len participate in shared memory loads
    // but skip the compute. This is how real FA2 works.
    const bool valid_query = (tid < block_m && qi < seq_len);

    // Per-row accumulators (in FP32)
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;
    }

    // Shared memory for K and V tiles
    // Layout: K_tile[BLOCK_N][head_dim], V_tile[BLOCK_N][head_dim]
    const int BLOCK_N = 32;
    extern __shared__ half smem[];
    half* K_tile = smem;                              // [BLOCK_N * head_dim]
    half* V_tile = smem + BLOCK_N * head_dim;         // [BLOCK_N * head_dim]

    // Causal mask: the latest query in this block can attend up to row_start + block_m - 1.
    // So we only need to iterate over KV tiles up to that position.
    int max_kv = min(seq_len, row_start + block_m);  // causal: no need to go beyond last query's position

    // Iterate over KV tiles
    for (int kv_start = 0; kv_start < max_kv; kv_start += BLOCK_N) {
        int kv_end = min(kv_start + BLOCK_N, seq_len);
        int tile_size = kv_end - kv_start;

        // --- Cooperative load of K and V tiles into shared memory ---
        // All threads participate, even idle ones. This is critical:
        // __syncthreads() requires ALL threads in the block to participate.
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
        __syncthreads();  // <-- ALL threads must hit this, even idle ones

        // --- Compute attention for this tile (only valid queries) ---
        if (valid_query) {
            for (int j = 0; j < tile_size; j++) {
                int kj = kv_start + j;

                // CAUSAL MASK: skip keys beyond this query's position
                if (kj > qi) break;

                // Dot product: Q[qi] . K[kj]
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = __half2float(Q[qi * head_dim + d]);
                    float k_val = __half2float(K_tile[j * head_dim + d]);
                    score += q_val * k_val;
                }
                score *= scale;

                // Online softmax update
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

        __syncthreads();  // <-- wait before loading next tile
    }

    // Write output
    if (valid_query) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] = __float2half(acc[d] / l_i);
        }
    }
}

// --------------------------------------------------------------------------
// Non-causal version for comparison (same kernel structure, no mask)
// --------------------------------------------------------------------------
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
                int kj = kv_start + j;
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

// --------------------------------------------------------------------------
// Host
// --------------------------------------------------------------------------
void fill_random(half* mat, size_t n, float stddev) {
    for (size_t i = 0; i < n; i++) {
        float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        float u2 = (float)(rand() + 1) / ((float)RAND_MAX + 1.0f);
        mat[i] = __float2half(stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2));
    }
}

struct Config {
    const char* name;
    int seq_len;
    int head_dim;
    int block_m;
    bool causal;
};

void run_config(const Config& cfg) {
    printf("\n================================================================\n");
    printf("  %s\n", cfg.name);
    printf("  seq_len=%d, head_dim=%d, block_m=%d, causal=%s\n",
           cfg.seq_len, cfg.head_dim, cfg.block_m, cfg.causal ? "YES" : "no");

    int num_blocks = (cfg.seq_len + cfg.block_m - 1) / cfg.block_m;
    int warps_per_block = (cfg.block_m + 31) / 32;
    printf("  Grid: %d blocks × %d warps = %d total warps\n",
           num_blocks, warps_per_block, num_blocks * warps_per_block);

    if (cfg.causal) {
        // Show the workload triangle
        printf("  Causal workload per block:\n");
        for (int b = 0; b < num_blocks; b++) {
            int row_start = b * cfg.block_m;
            int first_qi = row_start;
            int last_qi = min(row_start + cfg.block_m - 1, cfg.seq_len - 1);
            // First thread attends to first_qi+1 keys, last to last_qi+1
            printf("    Block %d: rows [%d..%d], attend to [%d..%d] keys (%.0f%% of full)\n",
                   b, first_qi, last_qi,
                   first_qi + 1, last_qi + 1,
                   100.0f * (first_qi + last_qi + 2) / (2.0f * cfg.seq_len));
        }
    }
    printf("================================================================\n");

    size_t mat_elems = cfg.seq_len * cfg.head_dim;
    size_t mat_bytes = mat_elems * sizeof(half);

    half *h_Q = (half*)malloc(mat_bytes);
    half *h_K = (half*)malloc(mat_bytes);
    half *h_V = (half*)malloc(mat_bytes);
    fill_random(h_Q, mat_elems, 0.1f);
    fill_random(h_K, mat_elems, 0.1f);
    fill_random(h_V, mat_elems, 0.5f);

    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, mat_bytes);
    cudaMalloc(&d_K, mat_bytes);
    cudaMalloc(&d_V, mat_bytes);
    cudaMalloc(&d_O, mat_bytes);
    cudaMemcpy(d_Q, h_Q, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, mat_bytes, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)cfg.head_dim);
    int smem_bytes = 2 * 32 * cfg.head_dim * sizeof(half);  // K_tile + V_tile

    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        cudaMemset(d_O, 0, mat_bytes);
        if (cfg.causal) {
            flash_attention_causal_fwd<<<num_blocks, cfg.block_m, smem_bytes>>>(
                d_Q, d_K, d_V, d_O, cfg.seq_len, cfg.head_dim, cfg.block_m, scale);
        } else {
            flash_attention_full_fwd<<<num_blocks, cfg.block_m, smem_bytes>>>(
                d_Q, d_K, d_V, d_O, cfg.seq_len, cfg.head_dim, cfg.block_m, scale);
        }
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n", cudaGetErrorString(err));
            break;
        }
    }

    half h_out[8];
    cudaMemcpy(h_out, d_O, min((size_t)8, mat_elems) * sizeof(half), cudaMemcpyDeviceToHost);
    printf("  Output[0:8] = ");
    for (int i = 0; i < 8 && i < (int)mat_elems; i++)
        printf("%.4f ", __half2float(h_out[i]));
    printf("\n");

    free(h_Q); free(h_K); free(h_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

int main() {
    srand(42);

    Config configs[] = {
        // Non-causal baseline for comparison
        {"NON-CAUSAL: seq=512, d=64, block=128 (encoder-style, full attention)",
         512, 64, 128, false},

        // Causal — this is where idle warps should appear
        {"CAUSAL: seq=512, d=64, block=128 (GPT-style decoder)",
         512, 64, 128, true},

        // LLaMA-2 decoder prefill
        {"CAUSAL: seq=1024, d=128, block=128 (LLaMA-2 prefill)",
         1024, 128, 128, true},

        // Short prompt — extreme triangle effect
        {"CAUSAL: seq=256, d=64, block=128 (short prompt, 2 blocks)",
         256, 64, 128, true},

        // Larger tile — more warps per block, more idle potential
        {"CAUSAL: seq=512, d=64, block=256 (8 warps/block, steep triangle)",
         512, 64, 256, true},
    };

    for (auto& cfg : configs) {
        run_config(cfg);
    }

    return 0;
}
