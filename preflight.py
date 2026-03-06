"""Preflight checks for RunwayShield PoC.

Run:
  python preflight.py

This script verifies:
- Python syntax for all project modules
- Required imports are available
- open_clip model can be created (registry + transforms)

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
    ]
    for m in required:
        importlib.import_module(m)


def check_open_clip_create() -> None:
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    assert model is not None


def main() -> None:
    print("[preflight] compiling python files...")
    compile_all()
    print("[preflight] imports...")
    check_imports()
    print("[preflight] open_clip registry...")
    check_open_clip_create()
    print("[preflight] OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[preflight] FAILED:", repr(e))
        sys.exit(1)
