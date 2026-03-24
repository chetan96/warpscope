// idle_warp_ml.cu — Real-world ML kernels where idle warps appear.
// These are simplified but faithful representations of production patterns
// in transformer-based systems.
//
// Compile: nvcc -arch=compute_89 -lineinfo idle_warp_ml.cu -o idle_warp_ml
// Run:     IDLE_WARP=1 LD_PRELOAD=../nixnan.so ./idle_warp_ml

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// =========================================================================
// Example 1: Mixture of Experts (MoE) — Mixtral, Switch Transformer, DBRX
//
// In MoE, each token is routed to a subset of experts (typically top-2 out
// of 8). The router assigns tokens to experts, but the distribution is
// almost never uniform. Some experts get many tokens ("hot experts") and
// some get very few ("cold experts"). All experts are launched in the same
// kernel with the same grid, so warps assigned to cold experts finish
// early and sit idle while hot expert warps are still computing.
//
// This is the #1 source of warp underutilization in production MoE models.
// =========================================================================
__global__ void moe_expert_kernel(
    const float* __restrict__ input,    // [total_tokens, hidden_dim]
    float*       __restrict__ output,   // [total_tokens, hidden_dim]
    const int*   __restrict__ expert_offsets,  // [num_experts + 1] — start index per expert
    const int*   __restrict__ expert_counts,   // [num_experts] — tokens per expert
    const float* __restrict__ expert_weights,  // [num_experts, hidden_dim, hidden_dim]
    int hidden_dim,
    int num_experts)
{
    int expert_id = blockIdx.x;  // one block per expert
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (expert_id >= num_experts) return;

    int token_count = expert_counts[expert_id];
    int offset = expert_offsets[expert_id];

    // Each warp handles one token's MLP forward pass
    // If this expert got fewer tokens than warps, some warps are idle
    if (warp_id >= token_count) return;  // <-- IDLE WARP

    int token_idx = offset + warp_id;
    int lane = tid % 32;

    // Simplified MLP: output = ReLU(input * W)
    // Each lane handles hidden_dim/32 elements
    int elems_per_lane = (hidden_dim + 31) / 32;
    for (int i = 0; i < elems_per_lane; i++) {
        int col = lane * elems_per_lane + i;
        if (col >= hidden_dim) break;

        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            sum += input[token_idx * hidden_dim + k] *
                   expert_weights[expert_id * hidden_dim * hidden_dim + k * hidden_dim + col];
        }
        output[token_idx * hidden_dim + col] = fmaxf(sum, 0.0f);  // ReLU
    }
}

// =========================================================================
// Example 2: Padded Batch Attention — Variable-length serving
//
// In production LLM serving (vLLM, TensorRT-LLM, TGI), a batch contains
// requests of different lengths. The naive approach pads all sequences to
// the max length. Warps processing padding tokens do useless work.
//
// Real systems use PagedAttention to mitigate this, but many custom
// kernels and older systems still pad.
// =========================================================================
__global__ void padded_batch_attention(
    const half* __restrict__ Q,       // [batch, max_seq_len, head_dim]
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    const int*  __restrict__ seq_lens, // actual length per batch item
    int max_seq_len,
    int head_dim,
    int block_m)
{
    int batch_idx = blockIdx.y;
    int row_start = blockIdx.x * block_m;
    int tid = threadIdx.x;
    int qi = row_start + tid;

    int actual_len = seq_lens[batch_idx];

    // Threads beyond the actual sequence length are wasted
    if (qi >= actual_len) return;  // <-- IDLE WARP for padding

    int head_offset = batch_idx * max_seq_len * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    // Only attend to actual tokens, not padding
    for (int kj = 0; kj < actual_len; kj++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(Q[head_offset + qi * head_dim + d]) *
                     __half2float(K[head_offset + kj * head_dim + d]);
        }
        score *= scale;

        float m_old = m_i;
        m_i = fmaxf(m_i, score);
        float correction = expf(m_old - m_i);
        float p_ij = expf(score - m_i);
        l_i = l_i * correction + p_ij;

        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction +
                     p_ij * __half2float(V[head_offset + kj * head_dim + d]);
        }
    }

    for (int d = 0; d < head_dim; d++) {
        O[head_offset + qi * head_dim + d] = __float2half(acc[d] / l_i);
    }
}

// =========================================================================
// Example 3: Speculative Decoding verification
//
// In speculative decoding (used by Medusa, EAGLE, Lookahead), a small
// draft model proposes K candidate tokens, then the large model verifies
// them in parallel. After verification, only some candidates are accepted.
// The next iteration only needs to process accepted tokens, but the
// kernel is launched for all K candidates. Rejected candidates = idle warps.
//
// Typical: K=5 candidates proposed, 1-3 accepted → 40-80% idle.
// =========================================================================
__global__ void speculative_verify_kernel(
    const float* __restrict__ hidden_states,  // [num_candidates, hidden_dim]
    float*       __restrict__ logits,          // [num_candidates, vocab_size]
    const int*   __restrict__ accepted_mask,   // [num_candidates] — 1=accepted, 0=rejected
    const float* __restrict__ lm_head_weight,  // [vocab_size, hidden_dim]
    int num_candidates,
    int hidden_dim,
    int vocab_size)
{
    int candidate_id = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (candidate_id >= num_candidates) return;

    // Rejected candidates: skip computation entirely
    if (!accepted_mask[candidate_id]) return;  // <-- IDLE WARP

    // Compute logits = hidden_states @ lm_head_weight^T
    // Each warp handles a chunk of the vocabulary
    int vocab_per_warp = (vocab_size + (blockDim.x / 32) - 1) / (blockDim.x / 32);
    int vocab_start = warp_id * vocab_per_warp;
    int vocab_end = min(vocab_start + vocab_per_warp, vocab_size);
    int lane = tid % 32;

    for (int v = vocab_start + lane; v < vocab_end; v += 32) {
        float sum = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            sum += hidden_states[candidate_id * hidden_dim + d] *
                   lm_head_weight[v * hidden_dim + d];
        }
        logits[candidate_id * vocab_size + v] = sum;
    }
}

// =========================================================================
// Example 4: Sparse Attention (BigBird / Longformer pattern)
//
// In sparse attention, each query only attends to a subset of keys
// (local window + global tokens + random tokens). Queries near the
// edges of the sequence attend to fewer keys than queries in the middle.
// This creates load imbalance: middle warps do full work, edge warps
// do less.
// =========================================================================
__global__ void sparse_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int seq_len,
    int head_dim,
    int window_size)   // each query attends to a window of this size
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= seq_len) return;

    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute attention window for this query
    // Edge queries have smaller windows → less work → shorter runtime
    int kv_start = max(0, qi - window_size / 2);
    int kv_end = min(seq_len, qi + window_size / 2);
    int actual_window = kv_end - kv_start;  // varies per query!

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[128];
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    for (int kj = kv_start; kj < kv_end; kj++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(Q[qi * head_dim + d]) *
                     __half2float(K[kj * head_dim + d]);
        }
        score *= scale;

        float m_old = m_i;
        m_i = fmaxf(m_i, score);
        float correction = expf(m_old - m_i);
        float p_ij = expf(score - m_i);
        l_i = l_i * correction + p_ij;

        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction +
                     p_ij * __half2float(V[kj * head_dim + d]);
        }
    }

    for (int d = 0; d < head_dim; d++) {
        O[qi * head_dim + d] = __float2half(acc[d] / l_i);
    }
}

// =========================================================================
// Example 5: Dynamic Token Pruning (used in efficient inference)
//
// Models like DynamicViT, AdaViT, and token-pruned LLMs skip "easy"
// tokens at intermediate layers. A pruning mask marks which tokens
// need full computation. The kernel launches for all tokens, but
// pruned tokens return early.
// =========================================================================
__global__ void token_pruned_ffn(
    const float* __restrict__ input,      // [seq_len, hidden_dim]
    float*       __restrict__ output,     // [seq_len, hidden_dim]
    const int*   __restrict__ prune_mask, // [seq_len] — 1=active, 0=pruned
    const float* __restrict__ W1,         // [hidden_dim, ffn_dim]
    const float* __restrict__ W2,         // [ffn_dim, hidden_dim]
    int seq_len,
    int hidden_dim,
    int ffn_dim)
{
    int token_id = blockIdx.x;
    int tid = threadIdx.x;

    if (token_id >= seq_len) return;
    if (!prune_mask[token_id]) return;  // <-- IDLE WARP for pruned tokens

    int lane = tid % 32;
    int warp_id = tid / 32;

    // FFN: output = W2 * ReLU(W1 * input)
    // Each warp handles a portion of the FFN intermediate dimension
    int ffn_per_warp = (ffn_dim + (blockDim.x / 32) - 1) / (blockDim.x / 32);
    int ffn_start = warp_id * ffn_per_warp;
    int ffn_end = min(ffn_start + ffn_per_warp, ffn_dim);

    // Step 1: intermediate = ReLU(W1 * input)
    extern __shared__ float intermediate[];

    for (int f = ffn_start + lane; f < ffn_end; f += 32) {
        float sum = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            sum += input[token_id * hidden_dim + d] * W1[d * ffn_dim + f];
        }
        intermediate[f] = fmaxf(sum, 0.0f);
    }
    __syncthreads();

    // Step 2: output = W2 * intermediate
    int out_per_warp = (hidden_dim + (blockDim.x / 32) - 1) / (blockDim.x / 32);
    int out_start = warp_id * out_per_warp;
    int out_end = min(out_start + out_per_warp, hidden_dim);

    for (int d = out_start + lane; d < out_end; d += 32) {
        float sum = 0.0f;
        for (int f = 0; f < ffn_dim; f++) {
            sum += intermediate[f] * W2[f * hidden_dim + d];
        }
        output[token_id * hidden_dim + d] = sum;
    }
}


// =========================================================================
// Host-side test harness
// =========================================================================

void test_moe() {
    printf("\n================================================================\n");
    printf("  Example 1: Mixture of Experts (MoE)\n");
    printf("  8 experts, 128 tokens total, imbalanced routing\n");
    printf("  Expert token counts: [32, 28, 24, 16, 12, 8, 4, 4]\n");
    printf("  Launched with 8 warps/block → experts with <8 tokens have idle warps\n");
    printf("================================================================\n");

    int num_experts = 8;
    int hidden_dim = 256;
    int total_tokens = 128;

    // Imbalanced routing — typical in production MoE
    int h_counts[] = {32, 28, 24, 16, 12, 8, 4, 4};
    int h_offsets[9];
    h_offsets[0] = 0;
    for (int i = 0; i < num_experts; i++) h_offsets[i+1] = h_offsets[i] + h_counts[i];

    float *d_input, *d_output, *d_weights;
    int *d_offsets, *d_counts;
    cudaMalloc(&d_input, total_tokens * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, total_tokens * hidden_dim * sizeof(float));
    cudaMalloc(&d_weights, num_experts * hidden_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_offsets, (num_experts + 1) * sizeof(int));
    cudaMalloc(&d_counts, num_experts * sizeof(int));

    // Initialize with random data
    float *h_buf = (float*)malloc(num_experts * hidden_dim * hidden_dim * sizeof(float));
    for (int i = 0; i < num_experts * hidden_dim * hidden_dim; i++)
        h_buf[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_weights, h_buf, num_experts * hidden_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_buf);

    h_buf = (float*)malloc(total_tokens * hidden_dim * sizeof(float));
    for (int i = 0; i < total_tokens * hidden_dim; i++)
        h_buf[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_input, h_buf, total_tokens * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_buf);

    cudaMemcpy(d_offsets, h_offsets, (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, num_experts * sizeof(int), cudaMemcpyHostToDevice);

    // 8 blocks (one per expert), 256 threads (8 warps) per block
    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        moe_expert_kernel<<<num_experts, 256>>>(
            d_input, d_output, d_offsets, d_counts, d_weights, hidden_dim, num_experts);
        cudaDeviceSynchronize();
    }

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_weights);
    cudaFree(d_offsets); cudaFree(d_counts);
}

void test_padded_batch() {
    printf("\n================================================================\n");
    printf("  Example 2: Padded Batch Attention (variable-length serving)\n");
    printf("  Batch of 4 requests, max_len=512, actual: [512, 256, 128, 64]\n");
    printf("  Shorter sequences → entire warps processing padding are idle\n");
    printf("================================================================\n");

    int batch = 4;
    int max_seq = 512;
    int head_dim = 64;
    int block_m = 128;
    int h_lens[] = {512, 256, 128, 64};

    size_t mat_bytes = batch * max_seq * head_dim * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;
    int *d_lens;
    cudaMalloc(&d_Q, mat_bytes);
    cudaMalloc(&d_K, mat_bytes);
    cudaMalloc(&d_V, mat_bytes);
    cudaMalloc(&d_O, mat_bytes);
    cudaMalloc(&d_lens, batch * sizeof(int));

    // Fill with random data
    half *h_buf = (half*)malloc(mat_bytes);
    for (size_t i = 0; i < batch * max_seq * head_dim; i++)
        h_buf[i] = __float2half(0.1f * ((float)rand() / RAND_MAX - 0.5f));
    cudaMemcpy(d_Q, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    free(h_buf);
    cudaMemcpy(d_lens, h_lens, batch * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks_seq = (max_seq + block_m - 1) / block_m;  // 4
    dim3 grid(num_blocks_seq, batch);  // (4, 4) = 16 blocks

    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        padded_batch_attention<<<grid, block_m>>>(
            d_Q, d_K, d_V, d_O, d_lens, max_seq, head_dim, block_m);
        cudaDeviceSynchronize();
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaFree(d_lens);
}

void test_speculative() {
    printf("\n================================================================\n");
    printf("  Example 3: Speculative Decoding Verification\n");
    printf("  8 candidate tokens proposed, only 3 accepted\n");
    printf("  5/8 blocks do no work → 62.5%% idle warps\n");
    printf("================================================================\n");

    int num_candidates = 8;
    int hidden_dim = 256;
    int vocab_size = 1024;  // small for testing
    int h_accepted[] = {1, 0, 1, 0, 0, 1, 0, 0};  // 3/8 accepted

    float *d_hidden, *d_logits, *d_lm_head;
    int *d_accepted;
    cudaMalloc(&d_hidden, num_candidates * hidden_dim * sizeof(float));
    cudaMalloc(&d_logits, num_candidates * vocab_size * sizeof(float));
    cudaMalloc(&d_lm_head, vocab_size * hidden_dim * sizeof(float));
    cudaMalloc(&d_accepted, num_candidates * sizeof(int));

    float *h_buf = (float*)malloc(vocab_size * hidden_dim * sizeof(float));
    for (int i = 0; i < vocab_size * hidden_dim; i++)
        h_buf[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_lm_head, h_buf, vocab_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < num_candidates * hidden_dim; i++)
        h_buf[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_hidden, h_buf, num_candidates * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_buf);
    cudaMemcpy(d_accepted, h_accepted, num_candidates * sizeof(int), cudaMemcpyHostToDevice);

    // One block per candidate, 128 threads (4 warps)
    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        speculative_verify_kernel<<<num_candidates, 128>>>(
            d_hidden, d_logits, d_accepted, d_lm_head,
            num_candidates, hidden_dim, vocab_size);
        cudaDeviceSynchronize();
    }

    cudaFree(d_hidden); cudaFree(d_logits); cudaFree(d_lm_head);
    cudaFree(d_accepted);
}

void test_sparse_attention() {
    printf("\n================================================================\n");
    printf("  Example 4: Sparse Attention (Longformer/BigBird pattern)\n");
    printf("  seq_len=512, window=128. Edge queries attend to fewer keys.\n");
    printf("  Queries at position 0 attend to 64 keys, middle to 128 keys.\n");
    printf("  → Edge warps finish ~2x faster than middle warps.\n");
    printf("================================================================\n");

    int seq_len = 512;
    int head_dim = 64;
    int window_size = 128;

    size_t mat_bytes = seq_len * head_dim * sizeof(half);
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, mat_bytes);
    cudaMalloc(&d_K, mat_bytes);
    cudaMalloc(&d_V, mat_bytes);
    cudaMalloc(&d_O, mat_bytes);

    half *h_buf = (half*)malloc(mat_bytes);
    for (int i = 0; i < seq_len * head_dim; i++)
        h_buf[i] = __float2half(0.1f * ((float)rand() / RAND_MAX - 0.5f));
    cudaMemcpy(d_Q, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_buf, mat_bytes, cudaMemcpyHostToDevice);
    free(h_buf);

    int threads = 128;
    int blocks = (seq_len + threads - 1) / threads;

    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        sparse_attention_kernel<<<blocks, threads>>>(
            d_Q, d_K, d_V, d_O, seq_len, head_dim, window_size);
        cudaDeviceSynchronize();
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
}

void test_token_pruning() {
    printf("\n================================================================\n");
    printf("  Example 5: Dynamic Token Pruning (FFN layer)\n");
    printf("  256 tokens, 50%% pruned at an intermediate layer\n");
    printf("  Pruned tokens skip the FFN entirely → 50%% idle blocks\n");
    printf("================================================================\n");

    int seq_len = 256;
    int hidden_dim = 256;
    int ffn_dim = 512;

    // Prune every other token
    int *h_mask = (int*)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) h_mask[i] = (i % 2 == 0) ? 1 : 0;

    float *d_input, *d_output, *d_W1, *d_W2;
    int *d_mask;
    cudaMalloc(&d_input, seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_output, seq_len * hidden_dim * sizeof(float));
    cudaMalloc(&d_W1, hidden_dim * ffn_dim * sizeof(float));
    cudaMalloc(&d_W2, ffn_dim * hidden_dim * sizeof(float));
    cudaMalloc(&d_mask, seq_len * sizeof(int));

    float *h_buf = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    for (int i = 0; i < hidden_dim * ffn_dim; i++)
        h_buf[i] = 0.01f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_W1, h_buf, hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_buf, ffn_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < seq_len * hidden_dim; i++)
        h_buf[i] = 0.1f * ((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(d_input, h_buf, seq_len * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    free(h_buf);
    cudaMemcpy(d_mask, h_mask, seq_len * sizeof(int), cudaMemcpyHostToDevice);
    free(h_mask);

    // One block per token, 128 threads, shared memory for FFN intermediate
    for (int run = 0; run < 3; run++) {
        printf("  Run %d...\n", run + 1);
        token_pruned_ffn<<<seq_len, 128, ffn_dim * sizeof(float)>>>(
            d_input, d_output, d_mask, d_W1, d_W2,
            seq_len, hidden_dim, ffn_dim);
        cudaDeviceSynchronize();
    }

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_mask);
}

int main() {
    srand(42);
    test_moe();
    test_padded_batch();
    test_speculative();
    test_sparse_attention();
    test_token_pruning();
    return 0;
}
