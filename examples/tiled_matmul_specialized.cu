// tiled_matmul_specialized.cu — Optimized tiled matrix multiplication variants.
// Demonstrates improvements informed by warpscope idle warp analysis:
//   Variant A: Double-buffered pipeline (1 sync vs 2 per iteration)
//   Variant B: Tail-block warp specialization (idle warps prefetch instead of exiting)
//
// Compile: nvcc -arch=sm_89 -lineinfo tiled_matmul_specialized.cu -o tiled_matmul_specialized
// Run:     ./tiled_matmul_specialized [filter]
// Warpscope: WARPSCOPE_JSON=report.json LD_PRELOAD=../warpscope.so ./tiled_matmul_specialized

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── Baseline kernel (single-buffered, 2 syncs per iteration) ───────────────

template <int TILE_M, int TILE_N, int TILE_K, int ROWS_PER_THREAD>
__global__ void tiled_matmul_baseline(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ C,
                                      int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) acc[r] = 0.0f;

    int tid = ty * TILE_N + tx;
    int block_threads = (TILE_M / ROWS_PER_THREAD) * TILE_N;

    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load tile
        for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
            int r = i / TILE_K, c = i % TILE_K;
            int gr = block_row + r, gc = k_tile + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
            int r = i / TILE_N, c = i % TILE_N;
            int gr = k_tile + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();  // sync 1: ensure tile is loaded

        // Compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = Bs[kk][tx];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++)
                acc[r] += As[ty * ROWS_PER_THREAD + r][kk] * b_val;
        }
        __syncthreads();  // sync 2: prevent next load from overwriting current data
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int gr = block_row + ty * ROWS_PER_THREAD + r;
        int gc = block_col + tx;
        if (gr < M && gc < N) C[gr * N + gc] = acc[r];
    }
}

// ─── Variant A: Double-Buffered Pipeline ────────────────────────────────────
// Uses two shared memory buffers. Loads for the next tile are issued BEFORE
// computing the current tile, allowing the GPU scheduler to overlap memory
// requests with ALU work. Only 1 __syncthreads() per iteration (vs 2 above).

template <int TILE_M, int TILE_N, int TILE_K, int ROWS_PER_THREAD>
__global__ void tiled_matmul_doublebuf(const float *__restrict__ A,
                                       const float *__restrict__ B,
                                       float *__restrict__ C,
                                       int M, int N, int K) {
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * TILE_N + tx;
    int block_threads = (TILE_M / ROWS_PER_THREAD) * TILE_N;

    float acc[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) acc[r] = 0.0f;

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prologue: load first tile into buffer 0
    for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
        int r = i / TILE_K, c = i % TILE_K;
        int gr = block_row + r, gc = c;
        As[0][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
        int r = i / TILE_N, c = i % TILE_N;
        int gr = r, gc = block_col + c;
        Bs[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int curr = kt & 1;
        int next = 1 - curr;
        int next_k = (kt + 1) * TILE_K;

        // Issue loads for NEXT tile first (async, will overlap with compute)
        if (kt + 1 < num_k_tiles) {
            for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
                int r = i / TILE_K, c = i % TILE_K;
                int gr = block_row + r, gc = next_k + c;
                As[next][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }
            for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
                int r = i / TILE_N, c = i % TILE_N;
                int gr = next_k + r, gc = block_col + c;
                Bs[next][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        // Compute from current buffer (overlaps with loads above)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = Bs[curr][kk][tx];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++)
                acc[r] += As[curr][ty * ROWS_PER_THREAD + r][kk] * b_val;
        }

        __syncthreads();  // single sync: ensures both load and compute are done
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int gr = block_row + ty * ROWS_PER_THREAD + r;
        int gc = block_col + tx;
        if (gr < M && gc < N) C[gr * N + gc] = acc[r];
    }
}

// ─── Variant B: Tail-Block Warp Specialization ─────────────────────────────
// Informed by warpscope analysis: non-aligned shapes have "tail blocks" where
// warps beyond the valid data range exit early and sit idle. This kernel keeps
// those warps alive and assigns them the ENTIRE prefetch load for the block,
// freeing compute warps to focus on ALU work without interleaving loads.
//
// Non-tail blocks use the standard double-buffer approach (all warps load+compute).
// Tail blocks: compute warps do only compute, producer warps do only prefetch.

template <int TILE_M, int TILE_N, int TILE_K, int ROWS_PER_THREAD>
__global__ void tiled_matmul_tail_specialized(const float *__restrict__ A,
                                              const float *__restrict__ B,
                                              float *__restrict__ C,
                                              int M, int N, int K) {
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * TILE_N + tx;
    int block_threads = (TILE_M / ROWS_PER_THREAD) * TILE_N;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warps_per_block = block_threads / 32;

    // Detect tail blocks
    int valid_rows = min(TILE_M, M - block_row);
    bool is_tail = (valid_rows < TILE_M) || (min(TILE_N, N - block_col) < TILE_N);

    // In tail blocks: warps whose output rows are entirely OOB become producers
    // With TILE_N=32, each warp covers exactly 1 ty value = ROWS_PER_THREAD rows
    int rows_per_warp = ROWS_PER_THREAD;  // 32/TILE_N * ROWS_PER_THREAD = 1*4 = 4
    int compute_warps = is_tail ? min(warps_per_block,
                                      (valid_rows + rows_per_warp - 1) / rows_per_warp)
                                : warps_per_block;
    int producer_warps = warps_per_block - compute_warps;
    bool is_producer = is_tail && (warp_id >= compute_warps);

    float acc[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) acc[r] = 0.0f;

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prologue: all warps load first tile
    for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
        int r = i / TILE_K, c = i % TILE_K;
        int gr = block_row + r, gc = c;
        As[0][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
        int r = i / TILE_N, c = i % TILE_N;
        int gr = r, gc = block_col + c;
        Bs[0][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int curr = kt & 1;
        int next = 1 - curr;
        int next_k = (kt + 1) * TILE_K;

        if (is_tail && producer_warps > 0) {
            // Tail block: split roles
            if (is_producer) {
                // Producers: exclusively prefetch next tile
                if (kt + 1 < num_k_tiles) {
                    int ptid = (warp_id - compute_warps) * 32 + lane_id;
                    int pthreads = producer_warps * 32;
                    for (int i = ptid; i < TILE_M * TILE_K; i += pthreads) {
                        int r = i / TILE_K, c = i % TILE_K;
                        int gr = block_row + r, gc = next_k + c;
                        As[next][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                    }
                    for (int i = ptid; i < TILE_K * TILE_N; i += pthreads) {
                        int r = i / TILE_N, c = i % TILE_N;
                        int gr = next_k + r, gc = block_col + c;
                        Bs[next][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                    }
                }
            } else {
                // Compute warps: only compute (no loads — producers handle it)
                #pragma unroll
                for (int kk = 0; kk < TILE_K; kk++) {
                    float b_val = Bs[curr][kk][tx];
                    #pragma unroll
                    for (int r = 0; r < ROWS_PER_THREAD; r++)
                        acc[r] += As[curr][ty * ROWS_PER_THREAD + r][kk] * b_val;
                }
            }
        } else {
            // Non-tail block: standard double-buffer (all warps load + compute)
            if (kt + 1 < num_k_tiles) {
                for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
                    int r = i / TILE_K, c = i % TILE_K;
                    int gr = block_row + r, gc = next_k + c;
                    As[next][r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
                }
                for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
                    int r = i / TILE_N, c = i % TILE_N;
                    int gr = next_k + r, gc = block_col + c;
                    Bs[next][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
                }
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                float b_val = Bs[curr][kk][tx];
                #pragma unroll
                for (int r = 0; r < ROWS_PER_THREAD; r++)
                    acc[r] += As[curr][ty * ROWS_PER_THREAD + r][kk] * b_val;
            }
        }

        __syncthreads();
    }

    // Store — producers skip (their output rows are OOB anyway)
    if (!is_producer) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            int gr = block_row + ty * ROWS_PER_THREAD + r;
            int gc = block_col + tx;
            if (gr < M && gc < N) C[gr * N + gc] = acc[r];
        }
    }
}

// ─── Benchmark Harness ──────────────────────────────────────────────────────

struct MatmulConfig {
    const char *name;
    int M, N, K;
};

static MatmulConfig configs[] = {
    // Square
    {"square_1024",   1024, 1024, 1024},
    {"square_2048",   2048, 2048, 2048},
    {"square_4096",   4096, 4096, 4096},
    // Tall-skinny
    {"tall_4096x128", 4096,  128, 1024},
    {"tall_4096x256", 4096,  256, 1024},
    {"tall_8192x64",  8192,   64,  512},
    // Short-wide
    {"wide_128x4096",  128, 4096, 1024},
    {"wide_256x4096",  256, 4096, 1024},
    {"wide_64x8192",    64, 8192,  512},
    // Non-aligned
    {"noalign_1000",      1000, 1000, 1000},
    {"noalign_2000x3000", 2000, 3000, 2000},
    {"noalign_4096x1000", 4096, 1000, 1024},
};
static const int num_configs = sizeof(configs) / sizeof(configs[0]);

static const int TILE = 32;
static const int TILE_K_VAL = 32;
static const int ROWS_PER_THREAD = 4;

typedef void (*kernel_fn)(const float *, const float *, float *, int, int, int);

float time_kernel(kernel_fn kernel, dim3 grid, dim3 block,
                  const float *d_A, const float *d_B, float *d_C,
                  int M, int N, int K, int num_runs) {
    // Warmup
    for (int i = 0; i < 2; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float times[32];
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int run = 0; run < num_runs; run++) {
        CHECK_CUDA(cudaEventRecord(start));
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[run], start, stop));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    std::sort(times, times + num_runs);
    return times[num_runs / 2];
}

void cpu_matmul_ref(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void run_comparison(const MatmulConfig &cfg) {
    int M = cfg.M, N = cfg.N, K = cfg.K;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE / ROWS_PER_THREAD);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    int total_blocks = grid.x * grid.y;
    int warps_per_block = (block.x * block.y) / 32;
    bool has_tail = (N % TILE != 0) || (M % TILE != 0);

    printf("\n─── %s: M=%d N=%d K=%d ───\n", cfg.name, M, N, K);
    printf("  Grid: (%d, %d)  Warps/block: %d  Total blocks: %d  Tail: %s\n",
           grid.x, grid.y, warps_per_block, total_blocks, has_tail ? "yes" : "no");

    const int NUM_RUNS = 5;
    double flops = 2.0 * M * N * K;

    // 1) Baseline (single buffer, 2 syncs/iter)
    float baseline_ms = time_kernel(
        tiled_matmul_baseline<32, 32, 32, 4>,
        grid, block, d_A, d_B, d_C, M, N, K, NUM_RUNS);
    double baseline_gflops = flops / (baseline_ms * 1e6);

    // 2) Double-buffered (1 sync/iter, load-compute overlap)
    float dbuf_ms = time_kernel(
        tiled_matmul_doublebuf<32, 32, 32, 4>,
        grid, block, d_A, d_B, d_C, M, N, K, NUM_RUNS);
    double dbuf_gflops = flops / (dbuf_ms * 1e6);

    // 3) Tail-specialized (double-buf + idle warps become producers in tail blocks)
    float tail_ms = time_kernel(
        tiled_matmul_tail_specialized<32, 32, 32, 4>,
        grid, block, d_A, d_B, d_C, M, N, K, NUM_RUNS);
    double tail_gflops = flops / (tail_ms * 1e6);

    // Verify correctness of both variants on this shape
    bool verify = (M <= 1024 && N <= 1024 && K <= 1024);
    if (verify) {
        float *h_C = (float *)malloc(size_C);
        float *ref_C = (float *)malloc(size_C);
        cpu_matmul_ref(h_A, h_B, ref_C, M, N, K);

        // Check double-buffered
        tiled_matmul_doublebuf<32, 32, 32, 4><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        int errs_dbuf = 0;
        for (int i = 0; i < M * N; i++) {
            float rel = fabsf(h_C[i] - ref_C[i]) / (fabsf(ref_C[i]) + 1e-6f);
            if (rel > 1e-3f) errs_dbuf++;
        }

        // Check tail-specialized
        tiled_matmul_tail_specialized<32, 32, 32, 4><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        int errs_tail = 0;
        for (int i = 0; i < M * N; i++) {
            float rel = fabsf(h_C[i] - ref_C[i]) / (fabsf(ref_C[i]) + 1e-6f);
            if (rel > 1e-3f) errs_tail++;
        }

        printf("  Correctness: doublebuf=%s  tail=%s\n",
               errs_dbuf == 0 ? "PASS" : "FAIL", errs_tail == 0 ? "PASS" : "FAIL");
        free(h_C);
        free(ref_C);
    }

    printf("  Baseline:          %8.3f ms  %8.1f GFLOPS\n", baseline_ms, baseline_gflops);
    printf("  Double-buffered:   %8.3f ms  %8.1f GFLOPS  (%.2fx)\n",
           dbuf_ms, dbuf_gflops, baseline_ms / dbuf_ms);
    printf("  Tail-specialized:  %8.3f ms  %8.1f GFLOPS  (%.2fx)\n",
           tail_ms, tail_gflops, baseline_ms / tail_ms);

    // Machine-parseable
    printf("COMPARE\t%s\t%.4f\t%.1f\t%.4f\t%.1f\t%.2f\t%.4f\t%.1f\t%.2f\n",
           cfg.name,
           baseline_ms, baseline_gflops,
           dbuf_ms, dbuf_gflops, baseline_ms / dbuf_ms,
           tail_ms, tail_gflops, baseline_ms / tail_ms);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
}

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : nullptr;

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Tile: %dx%dx%d  ROWS_PER_THREAD=%d\n\n", TILE, TILE, TILE_K_VAL, ROWS_PER_THREAD);

    printf("COMPARE\tname\tbaseline_ms\tbaseline_gflops\tdbuf_ms\tdbuf_gflops\tdbuf_speedup\ttail_ms\ttail_gflops\ttail_speedup\n");

    for (int i = 0; i < num_configs; i++) {
        if (filter && !strstr(configs[i].name, filter)) continue;
        run_comparison(configs[i]);
    }

    printf("\nDone.\n");
    return 0;
}
