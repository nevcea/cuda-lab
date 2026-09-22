# cuda-study

CUDA 공부용 저장소입니다. `src/`의 각 `.cu` 파일은 독립 실행되는 단일 파일 프로그램이고, 스스로 결과를 검증합니다.

## 환경

- Windows, NVIDIA GPU (`sm_86`, RTX 30 시리즈)
- CUDA Toolkit 13.4, Visual Studio 2022 Build Tools
- [uv](https://docs.astral.sh/uv/)

GPU 아키텍처나 설치 경로가 다르면 `build.py` 상단의 경로와 `-arch` 값을 수정하세요.

## 실행

```bash
uv run build.py matmul
```

`src/matmul.cu`를 `build/matmul.exe`로 빌드한 뒤 바로 실행합니다. VS Code에서는 `Ctrl+Shift+B`로 현재 파일을 빌드하고 실행합니다.
