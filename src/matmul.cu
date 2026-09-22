#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n";                \
            std::exit(1);                                                                                    \
        }                                                                                                    \
    } while (0)

__global__ void matmul(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            s += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = s;
    }
}

template <typename T> inline void freeall(T*& ptr, bool is_gpu = false) {
    if (ptr != nullptr) {
        if (is_gpu)
            cudaFree(ptr);
        else
            delete[] ptr;
        ptr = nullptr;
    }
}

int main() {
    const int m = 512;
    const int n = 512;
    const int k = 512;
    const size_t s_a = size_t(m) * k * sizeof(float);
    const size_t s_b = size_t(k) * n * sizeof(float);
    const size_t s_c = size_t(m) * n * sizeof(float);

    float* h_a = new float[size_t(m) * k];
    float* h_b = new float[size_t(k) * n];
    float* h_c = new float[size_t(m) * n];

    for (int i = 0; i < m * k; ++i)
        h_a[i] = 1.0f;
    for (int i = 0; i < k * n; ++i)
        h_b[i] = 2.0f;

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, s_a));
    CHECK(cudaMalloc(&d_b, s_b));
    CHECK(cudaMalloc(&d_c, s_c));
    CHECK(cudaMemcpy(d_a, h_a, s_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, s_b, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16); // 16 * 16 = 256 threads
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(m, n, k, d_a, d_b, d_c);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_c, d_c, s_c, cudaMemcpyDeviceToHost));
    std::cout << "top-left ele c[0] : " << h_c[0] << " expected: " << k * 2.0f << "\n";

    freeall(d_a, true);
    freeall(d_b, true);
    freeall(d_c, true);
    freeall(h_a);
    freeall(h_b);
    freeall(h_c);

    return 0;
}
