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

// complete unrolling (Harris #6): block size is a template parameter, so every
// loop bound is a compile-time constant and nvcc unrolls both loops fully --
// no stride compare/shift/branch left, only the adds and barriers. Harris
// spells the steps out as `if (BS >= 512) ...` chains; a constant-bound loop
// with #pragma unroll compiles to the same thing.
//
// sass/reduction_fu.sass vs sass/reduction_uw.sass:
//   uw 0x0190-0x0230  loop body: SHF (stride>>=1), ISETP, @P1 BRA 0x190 back
//   fu 0x0150-0x01e0  straight line: LDS [tid+0x200] / [tid+0x100] with
//                     immediate offsets, one BAR.SYNC each, no back-branch
//   the warp tail (stride 32..1) was already unrolled in uw: its bounds were
//   constant even there, only blockDim.x / 2 was not.
template <unsigned BS> __global__ void reduce(const float* in, float* out, int n) {
    static_assert(BS >= 64 && (BS & (BS - 1)) == 0, "BS must be a power of 2 >= 64");
    __shared__ float s[BS];
    int tid = threadIdx.x;
    int i = blockIdx.x * BS * 2 + tid;
    float v = i < n ? in[i] : 0.f;
    if (i + BS < n) v += in[i + BS];
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

    // BLOCK is fixed here; Harris dispatches a runtime block size with a
    // switch over reduce<512>, reduce<256>, ... on the host.
    reduce<BLOCK><<<(n + 2 * BLOCK - 1) / (2 * BLOCK), BLOCK>>>(h_in, h_out, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    bool ok = *h_out == (float)n;
    std::cout << "sum = " << *h_out << " expected " << n << "\n" << (ok ? "OK" : "FAIL") << "\n";

    cudaFree(h_in);
    cudaFree(h_out);
    return !ok;
}
