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

// strided index (Harris #2): same pairs as reduction_la.cu, but thread tid
// handles slot 2*stride*tid, so active threads are packed at the low tids and
// warps no longer diverge. The new cost is shared-memory bank conflicts: the
// active lanes of a warp read s[2*stride*tid], and with 32 banks
// (bank = index % 32) that lands 2 lanes per bank at stride 1, 4 at stride 2,
// 8 at stride 4 and up (e.g. stride 16: tids 0..7 -> s[0], s[32], ... all
// bank 0). An n-way conflict splits one access into n serialized ones.
__global__ void reduce(const float* in, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = i < n ? in[i] : 0.f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = 2 * stride * tid;
        if (idx < blockDim.x) s[idx] += s[idx + stride];
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
