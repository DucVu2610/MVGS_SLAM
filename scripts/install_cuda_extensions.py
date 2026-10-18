#!/usr/bin/env python3
"""Validate the PyTorch/nvcc pairing and build pinned CUDA extensions."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def cuda_versions():
    torch_cuda = torch.version.cuda
    if not torch_cuda:
        raise RuntimeError(
            "This PyTorch build has no CUDA runtime. Install a CUDA-enabled "
            "torch==2.5.1 build before compiling the 3DGS extensions.")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            f"PyTorch was built for CUDA {torch_cuda}, but nvcc is not on PATH. "
            "Install/enable a matching CUDA toolkit (Kaggle often exposes it "
            "under /usr/local/cuda/bin).")
    output = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        raise RuntimeError(f"Could not parse CUDA version from {nvcc} --version")
    nvcc_cuda = f"{match.group(1)}.{match.group(2)}"
    torch_major, torch_minor = map(int, torch_cuda.split(".")[:2])
    nvcc_major, nvcc_minor = map(int, nvcc_cuda.split(".")[:2])
    if torch_major != nvcc_major:
        raise RuntimeError(
            f"CUDA major-version mismatch: torch={torch_cuda}, nvcc={nvcc_cuda}. "
            "Use a toolkit with the same major version as torch.version.cuda.")
    if torch_minor != nvcc_minor:
        print(
            f"warning: CUDA minor versions differ (torch={torch_cuda}, "
            f"nvcc={nvcc_cuda}); PyTorch permits this but extension compilation "
            "may require the exact toolkit used by the wheel.",
            file=sys.stderr,
        )
    return torch_cuda, nvcc_cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate CUDA versions without installing anything")
    args = parser.parse_args()
    torch_cuda, nvcc_cuda = cuda_versions()
    print(f"CUDA preflight OK: torch={torch_cuda}, nvcc={nvcc_cuda}")
    if args.check_only:
        return
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
        "-r", str(REPOSITORY_ROOT / "requirements-cuda.txt"),
    ], cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
