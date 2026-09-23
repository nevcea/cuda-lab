#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n";                \
            return 1;                                                                                        \
        }                                                                                                    \
    } while (0)

#define BLOCK 256

// unroll last warp (Harris #4): once stride <= 32 only warp 0 works, so swap
// __syncthreads() for __syncwarp(). needs blockDim.x >= 64.
__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float v = i < n ? in[i] : 0.f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    // Harris's volatile-only version races on Volta+ (independent thread
    // scheduling): read, sync, then write each step.
    if (tid < 32) {
        v = s[tid];
        for (int stride = 32; stride > 0; stride >>= 1) {
            v += s[tid + stride];
            __syncwarp();
            s[tid] = v;
            __syncwarp();
        }
    }

    if (tid == 0) atomicAdd(out, s[0]);
}

int main() {
    const int n = 1 << 20;
    float *h_in, *h_out;
    CHECK(cudaMallocManaged(&h_in, n * sizeof(float)));
    CHECK(cudaMallocManaged(&h_out, sizeof(float)));
    for (int i = 0; i < n; i++)
        h_in[i] = 1.f;
    *h_out = 0.f;

    reduce<<<(n + 2 * BLOCK - 1) / (2 * BLOCK), BLOCK>>>(h_in, h_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = *h_out == (float)n;
    std::cout << "sum = " << *h_out << " expected " << n << "\n" << (ok ? "OK" : "FAIL") << "\n";

    cudaFree(h_in);
    cudaFree(h_out);
    return !ok;
}
