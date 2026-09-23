// https://developer.nvidia.com/blog/using-shared-memory-cuda-cc
#include <cstdio>

// sass/shared_mem.sass: staticReverse and dynamicReverse compile to byte-identical
// SASS (13 instructions each). Shared memory size is launch-time metadata, not
// something the kernel body needs to know -- indexing is always offset-from-0.
__global__ void staticReverse(int* d, int n) {
    __shared__ int s[64];
    int t = threadIdx.x; // S2R R7, SR_TID.X
    int tr = n - t - 1;  // LOP3.LUT R0, RZ, R7, RZ, 0x33 (R0=~t)  then  IADD3 R0, R0, n (R0=~t+n == n-t-1)
    s[t] = d[t];         // IMAD.WIDE &d[t] -> LDG.E R4=d[t] -> STS [R7.X4], R4
    __syncthreads();     // BAR.SYNC.DEFER_BLOCKING
    d[t] = s[tr];        // LDS R5=s[tr] (R0 holds tr) -> STG.E [R2.64], R5 (R2 still &d[t] from above)
}

__global__ void dynamicReverse(int* d, int n) {
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

int main(void) {
    const int n = 64;
    int a[n], r[n], d[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
        r[i] = n - i - 1;
        d[i] = 0;
    }

    int* d_d;
    cudaMalloc(&d_d, n * sizeof(int));

    // run static shrd mem
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    staticReverse<<<1, n>>>(d_d, n);
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

    // run dyn shrd mem
    cudaMemcpy(d_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, n, n * sizeof(int)>>>(d_d, n);
    cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}