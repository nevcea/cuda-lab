# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Personal CUDA study repo (Windows, RTX 30-series `sm_86`, CUDA 13.4 via scoop, Visual Studio Build Tools 18). Each `.cu` file in `src/` is a standalone, single-file program with its own `main()`; there is no shared library, test suite, or CMake.

## Commands

- Build + run one file: `uv run build.py <name>` (e.g. `uv run build.py matmul` compiles `src/matmul.cu` to `build/matmul.exe` and runs it). VS Code's default build task (Ctrl+Shift+B) does the same for the open file.
- `build.py` runs `nvcc` in the same `cmd` session as MSVC's `vcvars64.bat` (nvcc needs the MSVC env), with `-arch=sm_86 -O2`. Toolchain paths are hardcoded at the top of the script.
- Profile: `uv run prof.py <name>` runs `build/<name>.exe` under Nsight Systems and prints GPU memcpy, kernel, and CUDA API time tables (build it first with `build.py`). `cudaMallocManaged` migrations don't show up as memcpy on Windows; use `cudaMemcpy` programs to see transfers.
- Format/lint: `pre-commit run --all-files` (clang-format for `.cu`, markdownlint-cli2 for `.md`; config in `.pre-commit-config.yaml`, `.clang-format`, `.markdownlint-cli2.yaml` with MD013 line length disabled). The hooks run on commit and auto-fix files, so a first commit may fail; re-`git add` and commit again. On a fresh clone: `uv tool install pre-commit && pre-commit install`.

## Conventions

- Programs self-verify: each computes a known result on the GPU and prints/returns pass/fail (`matmul` prints `c[0]` vs expected, `saxpy` prints `OK`/`FAIL` and returns nonzero on mismatch).
- CUDA calls are wrapped in a local `CHECK(...)` macro that prints file:line and `return 1` (so it only works inside `main`). Kernel launches are followed by `CHECK(cudaGetLastError())` and `CHECK(cudaDeviceSynchronize())`.
- Host/device pointers are prefixed `h_` / `d_`.
- `build/` is git-ignored; keep build artifacts out of the repo root.

## Working rules

- Commit and push only when the user explicitly asks. Do not offer or suggest committing.
- The user decides what to study or try next. Don't propose next steps or add to-do / "next" content to docs unprompted — record only what was actually done or learned. Only write next-step content, including to `TODO.md`, when the user explicitly asks for it, and only what they decided, not Claude's own suggestions.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, ...), imperative mood, no trailing period.
- `TODO.md` reflects current state: when an item is done, remove it instead of leaving it stale. Don't ask/remind — check before adding new items.
