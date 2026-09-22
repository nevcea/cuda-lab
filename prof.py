# usage: uv run prof.py matmul   (profiles build/matmul.exe with Nsight Systems, prints memcpy/kernel times)
import subprocess
import sys
from pathlib import Path

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"

name = sys.argv[1].removesuffix(".cu")
out = Path(__file__).parent / "build"
exe, rep = out / f"{name}.exe", out / f"{name}_prof.nsys-rep"

if not exe.exists():
    sys.exit(f"{exe} not found; run: uv run build.py {name}")
subprocess.run([NSYS, "profile", "--trace=cuda", "--force-overwrite=true", "-o", rep.with_suffix(""), exe], check=True)
subprocess.run([NSYS, "stats", "--force-export=true", "--report", "cuda_gpu_mem_time_sum,cuda_gpu_kern_sum,cuda_api_sum", rep], check=True)
