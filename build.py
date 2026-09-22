# usage: uv run build.py matmul   (src/matmul.cu -> build/matmul.exe, then runs it)
import subprocess
import sys
from pathlib import Path

VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
NVCC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe"

name = sys.argv[1].removesuffix(".cu")
root = Path(__file__).parent
out = root / "build"
out.mkdir(exist_ok=True)
exe = out / f"{name}.exe"

# nvcc needs the MSVC env, so run it in the same cmd session as vcvars64
build = f'call "{VCVARS}" >nul 2>&1 && "{NVCC}" -arch=sm_86 -O2 -Xcompiler /utf-8 -lcublas src/{name}.cu -o "{exe}"'
if subprocess.run(build, shell=True, cwd=root, env={"VSLANG": "1033", **__import__("os").environ}).returncode:
    sys.exit(1)
sys.exit(subprocess.run([exe]).returncode)
