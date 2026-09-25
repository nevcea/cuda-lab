#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#pragma comment(lib, "cublas")

#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n";                \
            return 1;                                                                                        \
        }                                                                                                    \
    } while (0)

// Step-by-step matmul optimization (Simon Boehm, "How to Optimize a CUDA
// Matmul Kernel"). Every kernel is checked against cuBLAS on random inputs and
// timed at 4096^3; constant inputs like matmul.cu's would hide indexing bugs.

#define TILE 16

// step 0 baseline: same kernel as matmul_shared.cu
__global__ void matmul_shared(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int r = blockIdx.y * TILE + threadIdx.y;
    int c = blockIdx.x * TILE + threadIdx.x;

    float s = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[threadIdx.y][threadIdx.x] = (r < M && k0 + threadIdx.x < K) ? A[r * K + k0 + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && c < N) ? B[(k0 + threadIdx.y) * N + c] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (r < M && c < N) C[r * N + c] = s;
}

// average ms per call over `reps` runs, after one warm-up call
template <class F> float time_ms(F launch, int reps = 10) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    launch();
    cudaEventRecord(t0);
    for (int i = 0; i < reps; ++i)
        launch();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / reps;
}

int main() {
    const int m = 4096, n = 4096, k = 4096;
    const double flops = 2.0 * m * n * k;

    std::vector<float> h_a(size_t(m) * k), h_b(size_t(k) * n), h_c(size_t(m) * n), h_ref(size_t(m) * n);
    srand(0);
    for (float& x : h_a)
        x = rand() / (float)RAND_MAX * 2.0f - 1.0f;
    for (float& x : h_b)
        x = rand() / (float)RAND_MAX * 2.0f - 1.0f;

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_c, h_c.size() * sizeof(float)));
    CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS is column-major: row-major C = A*B is column-major C^T = B^T * A^T,
    // so pass B first and swap m/n. No transpose flags needed.
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
    const float alpha = 1.0f, beta = 0.0f;
    auto run_cublas = [&] {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b, n, d_a, k, &beta, d_c, n);
    };
    float ms_ref = time_ms(run_cublas);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_ref.data(), d_c, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "cublas         " << ms_ref << " ms  " << flops / ms_ref / 1e6 << " GFLOPS\n";

    bool all_ok = true;
    auto report = [&](const char* name, float ms) {
        float max_err = 0.0f;
        for (size_t i = 0; i < h_c.size(); ++i)
            max_err = std::fmax(max_err, std::fabs(h_c[i] - h_ref[i]));
        bool ok = max_err < 1e-2f;
        all_ok &= ok;
        std::cout << name << ms << " ms  " << flops / ms / 1e6 << " GFLOPS  " << 100.0f * ms_ref / ms
                  << "% of cublas  max_err " << max_err << (ok ? "  OK" : "  FAIL") << "\n";
    };

    CHECK(cudaMemset(d_c, 0, h_c.size() * sizeof(float)));
    float ms = time_ms([&] {
        matmul_shared<<<dim3((n + TILE - 1) / TILE, (m + TILE - 1) / TILE), dim3(TILE, TILE)>>>(m, n, k, d_a,
                                                                                                d_b, d_c);
    });
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    report("shared         ", ms);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return !all_ok;
}
