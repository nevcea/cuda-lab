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

// sequential addressing (Harris #3): active threads are tid < stride, so
// whole warps go idle together -- no divergence, unlike reduction_la.cu.
// A warp reads 32 consecutive floats, one per bank, so none of
// reduction_si.cu's bank conflicts either.
__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = i < n ? in[i] : 0.f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
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

    reduce<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(h_in, h_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = *h_out == (float)n;
    std::cout << "sum = " << *h_out << " expected " << n << "\n" << (ok ? "OK" : "FAIL") << "\n";

    cudaFree(h_in);
    cudaFree(h_out);
    return !ok;
}
