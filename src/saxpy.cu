#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#define CHECK(x)                                                                                             \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " " << cudaGetErrorString(e) << "\n";                \
            std::exit(1);                                                                                    \
        }                                                                                                    \
    } while (0)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

using clk = std::chrono::steady_clock;

static clk::time_point lap(const char* label, clk::time_point t) {
    auto now = clk::now();
    double ms = std::chrono::duration<double, std::milli>(now - t).count();
    std::cout << std::left << std::setw(8) << label << std::right << std::setw(10) << std::fixed
              << std::setprecision(2) << ms << " ms\n";
    return now;
}

int main() {
    const int n = 1 << 20;
    float *x, *y;
    auto t = clk::now();

    CHECK(cudaMallocManaged(&x, n * sizeof(float)));
    CHECK(cudaMallocManaged(&y, n * sizeof(float)));
    t = lap("alloc", t);

    for (int i = 0; i < n; i++) {
        x[i] = 1.f;
        y[i] = 2.f;
    }
    t = lap("init", t);

    saxpy<<<(n + 255) / 256, 256>>>(n, 3.f, x, y);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    t = lap("kernel", t);

    int bad = 0;
    for (int i = 0; i < n; i++)
        if (y[i] != 5.f) bad++;
    t = lap("verify", t);

    std::cout << "n = " << n << "\n" << (bad ? "FAIL" : "OK") << " (" << bad << " mismatches)\n";
    cudaFree(x);
    cudaFree(y);
    return bad != 0;
}
