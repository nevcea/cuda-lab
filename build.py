# usage: uv run build.py matmul         (src/matmul.cu -> build/matmul.exe, then runs it)
#        uv run build.py matmul --sass  (also dumps SASS disassembly to sass/matmul.sass)
import subprocess
import sys
from pathlib import Path

VCVARS = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin"
NVCC = rf"{CUDA_BIN}\nvcc.exe"
CUOBJDUMP = rf"{CUDA_BIN}\cuobjdump.exe"

name = sys.argv[1].removesuffix(".cu")
dump_sass = "--sass" in sys.argv[2:]
root = Path(__file__).parent
out = root / "build"
out.mkdir(exist_ok=True)
exe = out / f"{name}.exe"

# nvcc needs the MSVC env, so run it in the same cmd session as vcvars64
build = f'call "{VCVARS}" >nul 2>&1 && "{NVCC}" -arch=sm_86 -O2 -Xcompiler /utf-8 -lcublas src/{name}.cu -o "{exe}"'
if subprocess.run(build, shell=True, cwd=root, env={"VSLANG": "1033", **__import__("os").environ}).returncode:
    sys.exit(1)

if dump_sass:
    sass_dir = root / "sass"
    sass_dir.mkdir(exist_ok=True)
    sass = sass_dir / f"{name}.sass"
    result = subprocess.run([CUOBJDUMP, "--dump-sass", str(exe)], capture_output=True, text=True)
    sass.write_text(result.stdout)
    print(f"SASS written to {sass}")

sys.exit(subprocess.run([exe]).returncode)
