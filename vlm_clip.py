"""Fast VLM classification using OpenCLIP.

This module provides a stable, cross-platform way to classify an incident ROI into:
- person | vehicle | bird | animal | debris | shadow | unknown

It is designed for:
- CPU-first reliability (works on Mac + Windows out of the box)
- deterministic output (no JSON parsing / no generative model fragility)
- minimal latency (single forward pass)

OpenCLIP is open-source and widely used for zero-shot classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger("runway_shield")


@dataclass
class ClipConfig:
    enabled: bool = True
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"  # auto|cpu|cuda|mps

    # If the best class confidence is below this, output becomes unknown.
    unknown_threshold: float = 0.28

    # If shadow wins with >= this, mark real=False.
    shadow_artifact_threshold: float = 0.55

    # Debris detail classification only if debris confidence >= this.
    debris_detail_threshold: float = 0.45


@dataclass
class ClipResult:
    ok: bool
    category: str
    confidence: float
    real: bool
    detail: Optional[str] = None
    error: Optional[str] = None


CATEGORIES = ["person", "vehicle", "bird", "animal", "debris", "shadow", "unknown"]


_CATEGORY_PROMPTS: Dict[str, List[str]] = {
    "person": [
        "a person on a runway",
        "a human on a runway",
        "a pedestrian",
    ],
    "vehicle": [
        "a vehicle on a runway",
        "a car",
        "a truck",
        "a maintenance vehicle",
        "a service vehicle",
    ],
    "bird": [
        "a bird on a runway",
        "birds on a runway",
        "a flock of birds",
    ],
    "animal": [
        "an animal on a runway",
        "a dog",
        "a deer",
        "a fox",
        "wildlife",
    ],
    "debris": [
        "debris on a runway",
        "foreign object debris",
        "a tire on a runway",
        "a traffic cone",
        "a suitcase",
        "trash on asphalt",
    ],
    "shadow": [
        "a shadow on asphalt",
        "a shadow",
        "a reflection on asphalt",
        "sun glare",
        "heat haze",
    ],
    # unknown is handled by thresholding; keep prompts for stability.
    "unknown": [
        "an object",
        "something",
    ],
}

_DEBRIS_DETAIL_LABELS: List[Tuple[str, List[str]]] = [
    ("tire", ["a tire", "a rubber tire"]),
    ("cone", ["a traffic cone", "an orange cone"]),
    ("bag", ["a bag", "a duffel bag", "a backpack"]),
    ("suitcase", ["a suitcase", "a luggage suitcase"]),
    ("tool", ["a tool", "a metal tool"]),
    ("trash", ["trash", "a piece of trash", "garbage"]),
    ("metal", ["a metal object", "a piece of metal"]),
]


def _choose_device(pref: str) -> str:
    pref = (pref or "auto").lower().strip()
    try:
        import torch
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CLIP: CUDA requested but not available, falling back to CPU")
            return "cpu"
        if pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("CLIP: MPS requested but not available, falling back to CPU")
            return "cpu"
        # auto
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception as exc:
        logger.warning("CLIP: device detection failed (%s), falling back to CPU", exc)
        return "cpu"


def _bgr_to_pil(frame_bgr: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class ClipClassifier:
    """OpenCLIP-based ROI classifier."""

    def __init__(self, cfg: ClipConfig):
        self.cfg = cfg
        self._device: str = _choose_device(cfg.device)

        self._model = None
        self._preprocess = None

        self._cat_text_emb = None  # torch.Tensor [C, D]
        self._detail_text_emb = None  # torch.Tensor [K, D]
        self._detail_names: List[str] = []

    @property
    def device(self) -> str:
        return self._device

    def _ensure_loaded(self) -> None:
        if not self.cfg.enabled:
            return
        if self._model is not None:
            return

        import torch
        import open_clip

        device = self._device

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            device=device,
        )
        model.eval()

        # Use half-precision on CUDA for faster inference
        if device == "cuda":
            model = model.half()

        logger.info("CLIP device: %s (half=%s, model=%s)", device, device == "cuda", self.cfg.model_name)

        # Build category text embeddings
        cat_prompts: List[str] = []
        cat_offsets: List[Tuple[int, int]] = []
        for c in CATEGORIES:
            prompts = _CATEGORY_PROMPTS[c]
            start = len(cat_prompts)
            cat_prompts.extend(prompts)
            end = len(cat_prompts)
            cat_offsets.append((start, end))

        tokenizer = open_clip.get_tokenizer(self.cfg.model_name)
        text = tokenizer(cat_prompts)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda")
            if device == "cuda"
            else torch.inference_mode()
        )
        with torch.inference_mode(), autocast_ctx:
            text = text.to(device)
            text_emb = model.encode_text(text)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Aggregate per category by mean (over prompt variants)
        cat_embs = []
        for (s, e) in cat_offsets:
            emb = text_emb[s:e].mean(dim=0, keepdim=True)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            cat_embs.append(emb)
        cat_emb = torch.cat(cat_embs, dim=0)  # [C, D]

        # Debris detail embeddings
        detail_prompts: List[str] = []
        detail_offsets: List[Tuple[int, int]] = []
        detail_names: List[str] = []
        for name, prompts in _DEBRIS_DETAIL_LABELS:
            detail_names.append(name)
            s = len(detail_prompts)
            detail_prompts.extend(prompts)
            e = len(detail_prompts)
            detail_offsets.append((s, e))

        text2 = tokenizer(detail_prompts) if detail_prompts else None
        if text2 is not None:
            with torch.inference_mode(), autocast_ctx:
                text2 = text2.to(device)
                detail_emb = model.encode_text(text2)
                detail_emb = detail_emb / detail_emb.norm(dim=-1, keepdim=True)
            detail_embs = []
            for (s, e) in detail_offsets:
                emb = detail_emb[s:e].mean(dim=0, keepdim=True)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                detail_embs.append(emb)
            detail_cat_emb = torch.cat(detail_embs, dim=0)
        else:
            detail_cat_emb = None

        self._model = model
        self._preprocess = preprocess
        self._cat_text_emb = cat_emb
        self._detail_text_emb = detail_cat_emb
        self._detail_names = detail_names

    def classify_rois(self, roi_bgr_frames: List[np.ndarray]) -> ClipResult:
        """Classify a list of ROI crops (BGR). Uses temporal averaging."""
        if not self.cfg.enabled:
            return ClipResult(ok=False, category="unknown", confidence=0.0, real=True, error="disabled")

        try:
            self._ensure_loaded()

            import torch

            assert self._model is not None
            assert self._preprocess is not None
            assert self._cat_text_emb is not None

            pil_imgs = [_bgr_to_pil(f) for f in roi_bgr_frames if f is not None and f.size > 0]
            if not pil_imgs:
                return ClipResult(ok=False, category="unknown", confidence=0.0, real=True, error="empty_roi")

            imgs = torch.stack([self._preprocess(im) for im in pil_imgs], dim=0).to(self._device)

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda")
                if self._device == "cuda"
                else torch.inference_mode()
            )
            with torch.inference_mode(), autocast_ctx:
                img_emb = self._model.encode_image(imgs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                # temporal average
                img_emb = img_emb.mean(dim=0, keepdim=True)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                logits = (img_emb @ self._cat_text_emb.T) * 100.0
                probs = logits.softmax(dim=-1).squeeze(0)

            probs_np = probs.detach().float().cpu().numpy()
            best_idx = int(np.argmax(probs_np))
            best_cat = CATEGORIES[best_idx]
            best_conf = float(probs_np[best_idx])

            if best_conf < float(self.cfg.unknown_threshold):
                best_cat = "unknown"

            shadow_idx = CATEGORIES.index("shadow")
            shadow_conf = float(probs_np[shadow_idx])
            real = not (best_cat == "shadow" and shadow_conf >= float(self.cfg.shadow_artifact_threshold))

            detail: Optional[str] = None
            if best_cat == "debris" and best_conf >= float(self.cfg.debris_detail_threshold) and self._detail_text_emb is not None:
                with torch.inference_mode(), autocast_ctx:
                    logits2 = (img_emb @ self._detail_text_emb.T) * 100.0
                    probs2 = logits2.softmax(dim=-1).squeeze(0)
                probs2_np = probs2.detach().float().cpu().numpy()
                j = int(np.argmax(probs2_np))
                if 0 <= j < len(self._detail_names):
                    detail = self._detail_names[j]

            return ClipResult(ok=True, category=best_cat, confidence=best_conf, real=real, detail=detail)

        except Exception as e:
            return ClipResult(ok=False, category="unknown", confidence=0.0, real=True, error=f"exception:{type(e).__name__}:{e}")
