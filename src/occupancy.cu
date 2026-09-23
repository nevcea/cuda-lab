// https://raw.githubusercontent.com/geohot/gpunoob/refs/heads/master/src/main.rs
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n";                \
            return 1;                                                                                        \
        }                                                                                                    \
    } while (0)

// each thread does iters dependent FMAs so the compiler can't fold the loop away,
// then writes a value we can check for correctness.
__global__ void busy(float* out, int iters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = threadIdx.x;
    for (int i = 0; i < iters; i++)
        a += 1.f;
    out[gid] = a;
}

int main() {
    const int iters = 100000;

    // correctness check on one fixed launch config
    float* d_out;
    CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));
    busy<<<1, 256>>>(d_out, iters);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    float h_out0;
    CHECK(cudaMemcpy(&h_out0, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = h_out0 == (float)iters; // threadIdx.x == 0 for out[0]
    std::cout << (ok ? "OK" : "FAIL") << " (out[0]=" << h_out0 << ", expected " << iters << ")\n";
    cudaFree(d_out);
    if (!ok) return 1;

    // sweep: for each block size (threads per block), grow total threads and time the kernel.
    // mirrors geohot's OpenCL local/global work-size sweep but in CUDA terms:
    // block size = local_work_size, total threads = global_work_size.
    std::cout << "\nblockDim,totalThreads,ms\n";
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int block : {32, 64, 128, 256, 512, 1024}) {
        for (int total = block; total <= (1 << 18); total *= 2) {
            int grid = (total + block - 1) / block;
            CHECK(cudaMalloc(&d_out, (size_t)grid * block * sizeof(float)));

            CHECK(cudaEventRecord(start));
            busy<<<grid, block>>>(d_out, iters);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaGetLastError());

            float ms = 0;
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            std::cout << block << "," << total << "," << std::fixed << std::setprecision(3) << ms << "\n";
            cudaFree(d_out);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
