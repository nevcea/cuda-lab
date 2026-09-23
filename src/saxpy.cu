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

// MOV R1, c[0x0][0x28]
__global__ void saxpy(int n, float a, const float* x, float* y) {
    /*
    int i = blockIdx.x * blockDim.x + threadIdx.x:
    S2R R4, SR_CTAID.X                     = blockIdx.x
    S2R R3, SR_TID.X                       = threadIdx.x
    IMAD R4, R4, c[0x0][0x0], R3           = R4 * blockDim.x(c[0x0][0x0]) + R3
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
    ISETP.GE.AND P0, PT, R4, c[0x0][0x160], PT ; if (i >= n)

    y[i] = a * x[i] + y[i]:
    MOV R5, 0x4                            = sizeof(float)
    ULDC.64 UR4, c[0x0][0x118]             = memory access descriptor setup
    IMAD.WIDE R2, R4, R5, c[0x0][0x168]    = &x[i]
    IMAD.WIDE R4, R4, R5, c[0x0][0x170]    = &y[i]
    LDG.E R2, [R2.64]                      = x[i]
    LDG.E R7, [R4.64]                      = y[i]
    FFMA R7, R2, c[0x0][0x164], R7         = a*x[i]+y[i]
    STG.E [R4.64], R7                      = y[i] = result
    */
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
