# TODO

1. Warp shuffle reduction: `__shfl_down_sync`로 shared memory 없이 warp 내부 reduction을 구현하고 `reduction_me.cu`와 비교합니다. cooperative groups와 CUB `DeviceReduce`를 기준선으로 함께 측정합니다.
2. Matmul 최적화 심화: `matmul_shared.cu`에서 출발하여 register tiling(1D, 2D), `float4` vectorized load, bank conflict 제거 순서로 진행하고 cuBLAS 대비 성능을 측정합니다. 참고 자료는 Simon Boehm의 "How to Optimize a CUDA Matmul Kernel"입니다.
3. Scan (prefix sum): Hillis-Steele 방식과 Blelloch 방식을 구현하고, 여러 block에 걸친 scan으로 확장합니다.
4. Streams와 비동기 전송: pinned memory, `cudaMemcpyAsync`, 여러 stream으로 복사와 kernel 실행을 겹치고 `prof.py`로 timeline을 확인합니다.
5. Tensor Core (WMMA): `wmma` API로 FP16 matmul을 구현합니다. 2번을 마친 뒤에 진행합니다.
