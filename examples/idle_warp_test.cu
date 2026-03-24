// idle_warp_test.cu — Test case for the idle warp detector.
// Launches kernels with intentionally unbalanced workloads to create idle warps.
//
// Compile: nvcc -arch=compute_89 -lineinfo idle_warp_test.cu -o idle_warp_test
// Run:     IDLE_WARP=1 LD_PRELOAD=../nixnan.so ./idle_warp_test

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define N 1024

// Test 1: Imbalanced kernel — first few warps do heavy work, rest exit early.
// This simulates a common pattern in reduction/attention kernels where
// trailing warps have no useful work.
__global__ void imbalanced_kernel(float *data, int work_limit) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

    // Only warps below work_limit do computation
    if (warp_id >= work_limit) return;  // <-- these warps are "idle"

    // Simulate heavy FP work
    float val = data[tid];
    for (int i = 0; i < 1000; i++) {
        val = val * 1.001f + 0.001f;
        val = sqrtf(val * val + 1.0f);
    }
    data[tid] = val;
}

// Test 2: Divergent kernel — even warps do heavy work, odd warps do light work.
// Shows how warp specialization could reassign the light warps.
__global__ void divergent_kernel(float *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;

    float val = data[tid];
    if (warp_id % 2 == 0) {
        // Heavy path: 2000 iterations of FP work
        for (int i = 0; i < 2000; i++) {
            val = val * 1.0001f + 0.0001f;
            val = sqrtf(val * val + 1.0f);
        }
    } else {
        // Light path: just 10 iterations
        for (int i = 0; i < 10; i++) {
            val = val + 1.0f;
        }
    }
    data[tid] = val;
}

// Test 3: Tail-effect kernel — grid is larger than needed, some blocks are empty.
// Common when problem size doesn't divide evenly into tiles.
__global__ void tail_effect_kernel(float *data, int actual_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= actual_size) return;  // Tail warps exit immediately

    float val = data[tid];
    for (int i = 0; i < 500; i++) {
        val = val * 1.001f + 0.001f;
    }
    data[tid] = val;
}

void run_test(const char *name, void (*launcher)(float *, int), float *d_data, int param) {
    printf("\n===== %s =====\n", name);
    launcher(d_data, param);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  Error: %s\n", cudaGetErrorString(err));
    }
}

// Wrappers for the test launches
void launch_imbalanced(float *d_data, int work_limit) {
    // 4 blocks × 256 threads = 32 warps, but only work_limit warps compute
    imbalanced_kernel<<<4, 256>>>(d_data, work_limit);
}

void launch_divergent(float *d_data, int /*unused*/) {
    // 4 blocks × 256 threads = 32 warps; even warps heavy, odd warps light
    divergent_kernel<<<4, 256>>>(d_data);
}

void launch_tail(float *d_data, int actual_size) {
    // Launch enough blocks for 1024 threads, but actual work is smaller
    int threads = 256;
    int blocks = (N + threads - 1) / threads;  // 4 blocks
    tail_effect_kernel<<<blocks, threads>>>(d_data, actual_size);
}

int main() {
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data
    float h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Test 1: Only 4 out of 32 warps do work (87.5% idle)
    printf("Test 1: Imbalanced — 4/32 warps active, 28 idle\n");
    run_test("Imbalanced (4 active warps)", launch_imbalanced, d_data, 4);
    // Run again to test cross-run consistency
    run_test("Imbalanced (4 active warps) — run 2", launch_imbalanced, d_data, 4);
    run_test("Imbalanced (4 active warps) — run 3", launch_imbalanced, d_data, 4);

    // Test 2: Even warps heavy, odd warps light
    printf("\nTest 2: Divergent — even warps heavy, odd warps light\n");
    run_test("Divergent workload", launch_divergent, d_data, 0);
    run_test("Divergent workload — run 2", launch_divergent, d_data, 0);

    // Test 3: Tail effect — grid covers 1024 threads but only 300 do work
    printf("\nTest 3: Tail effect — only 300/1024 threads have work\n");
    run_test("Tail effect (300 active)", launch_tail, d_data, 300);
    run_test("Tail effect (300 active) — run 2", launch_tail, d_data, 300);

    cudaFree(d_data);
    return 0;
}
