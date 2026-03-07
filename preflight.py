"""Preflight checks for RunwayShield PoC.

Run:
  python preflight.py

This script verifies:
- Python syntax for all project modules
- Required imports are available

It is intentionally lightweight and cross-platform.
"""

from __future__ import annotations

import importlib
import pathlib
import py_compile
import sys


ROOT = pathlib.Path(__file__).resolve().parent


def compile_all() -> None:
    py_files = [p for p in ROOT.glob("*.py") if p.name != "__init__.py"]
    for p in py_files:
        py_compile.compile(str(p), doraise=True)


def check_imports() -> None:
    required = [
        "streamlit",
        "cv2",
        "numpy",
        "pandas",
        "PIL",
        "torch",
        "open_clip",
        "yaml",
        "ultralytics",
    ]
    for m in required:
        importlib.import_module(m)


def check_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            vram = round(dev.total_memory / 1024 ** 2)
            print(f"[preflight] CUDA available: {dev.name} ({vram} MB VRAM)")
        else:
            print("[preflight] CUDA not available — will run on CPU")
    except Exception as e:
        print(f"[preflight] WARNING: could not check CUDA: {e}")


def main() -> None:
    print("[preflight] compiling python files...")
    compile_all()
    print("[preflight] imports...")
    check_imports()
    check_cuda()
    print("[preflight] OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[preflight] FAILED:", repr(e))
        sys.exit(1)
