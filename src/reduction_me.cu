#include <algorithm>
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

// multiple elements per thread (Harris #7): launch only as many blocks as fit
// on the GPU at once and have each thread grid-stride over the input, summing
// in a register. The shared-memory tree and atomicAdd then run once per
// resident block instead of once per 512 inputs.
template <unsigned BS> __global__ void reduce(const float* in, float* out, int n) {
    static_assert(BS >= 64 && (BS & (BS - 1)) == 0, "BS must be a power of 2 >= 64");
    __shared__ float s[BS];
    int tid = threadIdx.x;
    float v = 0.f;
    for (int i = blockIdx.x * BS * 2 + tid; i < n; i += BS * 2 * gridDim.x) {
        v += in[i];
        if (i + BS < n) v += in[i + BS];
    }
    s[tid] = v;
    __syncthreads();

#pragma unroll
    for (unsigned stride = BS / 2; stride > 32; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid < 32) {
        v = s[tid];
#pragma unroll
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

    // one full wave: SMs * resident blocks per SM, capped by what n needs
    int sms, per_sm;
    CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, reduce<BLOCK>, BLOCK, 0));
    int blocks = std::min(sms * per_sm, (n + 2 * BLOCK - 1) / (2 * BLOCK));
    std::cout << sms << " SMs x " << per_sm << " blocks/SM -> " << blocks << " blocks, "
              << (n + blocks * BLOCK - 1) / (blocks * BLOCK) << " elements/thread\n";

    reduce<BLOCK><<<blocks, BLOCK>>>(h_in, h_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = *h_out == (float)n;
    std::cout << "sum = " << *h_out << " expected " << n << "\n" << (ok ? "OK" : "FAIL") << "\n";

    cudaFree(h_in);
    cudaFree(h_out);
    return !ok;
}
