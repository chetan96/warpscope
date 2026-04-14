// tiled_matmul.cu — Baseline tiled matrix multiplication with benchmark harness.
// Tests various square and rectangular shapes to measure latency and GFLOPS,
// and to create idle warp patterns for warpscope analysis.
//
// Compile: nvcc -arch=sm_89 -lineinfo tiled_matmul.cu -o tiled_matmul
// Run:     ./tiled_matmul [filter]
// Warpscope: WARPSCOPE_JSON=report.json LD_PRELOAD=../warpscope.so ./tiled_matmul

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

// ─── Tiled MatMul Kernel ────────────────────────────────────────────────────
// C[M×N] = A[M×K] × B[K×N]
// Each block computes a TILE_M × TILE_N tile of C.
// Block dims: (TILE_N, TILE_M / ROWS_PER_THREAD)
// Each thread computes ROWS_PER_THREAD rows of C within its tile column.

template <int TILE_M, int TILE_N, int TILE_K, int ROWS_PER_THREAD>
__global__ void tiled_matmul_kernel(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int M, int N, int K) {
    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    // Block's top-left corner in C
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    // Thread position within the block
    int tx = threadIdx.x;  // column within tile [0, TILE_N)
    int ty = threadIdx.y;  // row-group within tile [0, TILE_M/ROWS_PER_THREAD)

    // Accumulator for this thread's ROWS_PER_THREAD output values
    float acc[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) acc[r] = 0.0f;

    // Linear thread ID for cooperative loading
    int tid = ty * TILE_N + tx;
    int block_threads = (TILE_M / ROWS_PER_THREAD) * TILE_N;

    // Tile across K dimension
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load of A tile [TILE_M × TILE_K]
        for (int i = tid; i < TILE_M * TILE_K; i += block_threads) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = block_row + r;
            int global_c = k_tile + c;
            As[r][c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // Cooperative load of B tile [TILE_K × TILE_N]
        for (int i = tid; i < TILE_K * TILE_N; i += block_threads) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int global_r = k_tile + r;
            int global_c = block_col + c;
            Bs[r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = Bs[kk][tx];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                acc[r] += As[ty * ROWS_PER_THREAD + r][kk] * b_val;
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int global_r = block_row + ty * ROWS_PER_THREAD + r;
        int global_c = block_col + tx;
        if (global_r < M && global_c < N) {
            C[global_r * N + global_c] = acc[r];
        }
    }
}

// ─── Benchmark Harness ──────────────────────────────────────────────────────

struct MatmulConfig {
    const char *name;
    int M, N, K;
};

static MatmulConfig configs[] = {
    // Square (aligned — expect balanced warps)
    {"square_1024",   1024, 1024, 1024},
    {"square_2048",   2048, 2048, 2048},
    {"square_4096",   4096, 4096, 4096},

    // Tall-skinny (asymmetric grid)
    {"tall_4096x128", 4096,  128, 1024},
    {"tall_4096x256", 4096,  256, 1024},
    {"tall_8192x64",  8192,   64,  512},

    // Short-wide (mirror)
    {"wide_128x4096",  128, 4096, 1024},
    {"wide_256x4096",  256, 4096, 1024},
    {"wide_64x8192",    64, 8192,  512},

    // Non-aligned (tail blocks will have idle warps)
    {"noalign_1000",      1000, 1000, 1000},
    {"noalign_2000x3000", 2000, 3000, 2000},
    {"noalign_4096x1000", 4096, 1000, 1024},
};
static const int num_configs = sizeof(configs) / sizeof(configs[0]);

// Tile parameters
static const int TILE = 32;
static const int TILE_K_VAL = 32;
static const int ROWS_PER_THREAD = 4;

void cpu_matmul_ref(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

bool verify_result(const float *gpu_C, const float *ref_C, int M, int N) {
    int errors = 0;
    for (int i = 0; i < M * N && errors < 10; i++) {
        float diff = fabsf(gpu_C[i] - ref_C[i]);
        float rel = diff / (fabsf(ref_C[i]) + 1e-6f);
        if (rel > 1e-3f) {
            if (errors == 0)
                printf("  MISMATCH at [%d,%d]: gpu=%.6f ref=%.6f diff=%.6f\n",
                       i / N, i % N, gpu_C[i], ref_C[i], diff);
            errors++;
        }
    }
    return errors == 0;
}

void run_benchmark(const MatmulConfig &cfg, bool do_verify) {
    int M = cfg.M, N = cfg.N, K = cfg.K;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    // Allocate and init host data
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Grid and block
    dim3 block(TILE, TILE / ROWS_PER_THREAD);  // 32×8 = 256 threads (8 warps)
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    int total_blocks = grid.x * grid.y;
    int warps_per_block = (block.x * block.y) / 32;
    int total_warps = total_blocks * warps_per_block;

    // Compute expected tail idle info
    int tail_x = (N % TILE != 0) ? 1 : 0;
    int tail_y = (M % TILE != 0) ? 1 : 0;
    int tail_blocks = tail_x * grid.y + tail_y * grid.x - tail_x * tail_y;

    printf("\n─── %s: M=%d N=%d K=%d ───\n", cfg.name, M, N, K);
    printf("  Grid: (%d, %d)  Block: (%d, %d)  Warps/block: %d  Total warps: %d\n",
           grid.x, grid.y, block.x, block.y, warps_per_block, total_warps);
    if (tail_blocks > 0)
        printf("  Tail blocks: %d/%d (%.1f%%) — expect idle warps\n",
               tail_blocks, total_blocks, 100.0f * tail_blocks / total_blocks);
    else
        printf("  Fully aligned — no tail blocks\n");

    // Warmup
    for (int i = 0; i < 2; i++) {
        tiled_matmul_kernel<32, 32, 32, 4><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Timed runs
    const int NUM_RUNS = 5;
    float times[NUM_RUNS];

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int run = 0; run < NUM_RUNS; run++) {
        CHECK_CUDA(cudaEventRecord(start));
        tiled_matmul_kernel<32, 32, 32, 4><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&times[run], start, stop));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Compute stats
    std::sort(times, times + NUM_RUNS);
    float median_ms = times[NUM_RUNS / 2];
    float min_ms = times[0];
    float max_ms = times[NUM_RUNS - 1];
    double flops = 2.0 * M * N * K;
    double gflops = flops / (median_ms * 1e6);

    printf("  Latency: min=%.3f ms  median=%.3f ms  max=%.3f ms\n", min_ms, median_ms, max_ms);
    printf("  GFLOPS:  %.1f\n", gflops);

    // Machine-parseable line
    printf("RESULT\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.1f\t%d\t%d\t%d\n",
           cfg.name, M, N, K, TILE, TILE, TILE_K_VAL,
           median_ms, gflops, grid.x, grid.y, total_warps);

    // Correctness verification
    if (do_verify) {
        CHECK_CUDA(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
        float *ref_C = (float *)malloc(size_C);
        cpu_matmul_ref(h_A, h_B, ref_C, M, N, K);
        bool ok = verify_result(h_C, ref_C, M, N);
        printf("  Correctness: %s\n", ok ? "PASS" : "FAIL");
        free(ref_C);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : nullptr;

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Tile config: TILE_M=%d TILE_N=%d TILE_K=%d ROWS_PER_THREAD=%d\n\n",
           TILE, TILE, TILE_K_VAL, ROWS_PER_THREAD);

    printf("RESULT\tname\tM\tN\tK\ttile_m\ttile_n\ttile_k\tmedian_ms\tgflops\tgrid_x\tgrid_y\ttotal_warps\n");

    for (int i = 0; i < num_configs; i++) {
        if (filter && !strstr(configs[i].name, filter)) continue;

        // Verify correctness only on small shapes (M,N,K all <= 256)
        bool do_verify = (configs[i].M <= 256 && configs[i].N <= 256 && configs[i].K <= 256);

        // For the first non-aligned shape, also verify
        if (strcmp(configs[i].name, "noalign_1000") == 0) do_verify = true;

        run_benchmark(configs[i], do_verify);
    }

    printf("\nDone.\n");
    return 0;
}
