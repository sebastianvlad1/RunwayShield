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
        "transformers",
    ]
    for m in required:
        importlib.import_module(m)

    # Transformers 4.51+ injects GPU-CPU sync points (torch_compilable_check) in
    # GroundingDINO's deformable attention, halving inference throughput.
    # Warn if a bad version is installed.
    try:
        import transformers
        from packaging.version import Version
        tv = Version(transformers.__version__)
        if tv >= Version("4.51"):
            print(
                f"[preflight] WARNING: transformers {transformers.__version__} detected. "
                "Versions >= 4.51 degrade GroundingDINO inference speed (~2x slower). "
                "Pin to transformers==4.50.3: pip install transformers==4.50.3"
            )
        else:
            print(f"[preflight] transformers {transformers.__version__} OK")
    except Exception:
        pass


def check_cuda() -> None:
    """Report CUDA availability and GPU info."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            print(f"[preflight] CUDA available: {name} ({vram_mb:.0f} MB VRAM)")
        else:
            print("[preflight] CUDA not available — models will run on CPU")
    except Exception as e:
        print(f"[preflight] CUDA check failed: {e}")


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
