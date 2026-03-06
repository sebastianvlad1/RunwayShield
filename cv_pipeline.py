"""Computer vision pipeline for runway hazard candidate detection.

Core pipeline:
- User defines runway polygon.
- We create a runway mask.
- We run change detection (MOG2 + optional median background diff) only inside the mask.
- We filter blobs by area and shape and guard against camera shake.
- We track blobs with an IoU tracker to get stable IDs.

This module has no ML dependencies and is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import defaultdict, deque

import cv2
import numpy as np


# ----------------------------
# Geometry helpers
# ----------------------------


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def sort_polygon_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort points around centroid to reduce self-intersections.

    This does not guarantee a perfect polygon for any arbitrary input, but it is
    more robust than using the raw click order (which can easily self-intersect).
    """
    if len(points) < 3:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    pts = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return pts


def polygon_from_points(points: List[Tuple[float, float]]) -> np.ndarray:
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    return pts


def make_runway_mask(frame_shape: Tuple[int, int], poly: np.ndarray) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def point_in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def clamp_bbox(b: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def crop_roi(frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int], pad: float = 0.20) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox_xyxy, w, h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(round(bw * pad))
    dy = int(round(bh * pad))
    x1p = max(0, x1 - dx)
    y1p = max(0, y1 - dy)
    x2p = min(w, x2 + dx)
    y2p = min(h, y2 + dy)
    return frame_bgr[y1p:y2p, x1p:x2p].copy()


# ----------------------------
# OpenCV blob detector
# ----------------------------


@dataclass
class BlobDetectorConfig:
    min_area: int = 250
    max_area: int = 45000

    history: int = 400
    var_threshold: int = 16
    detect_shadows: bool = True
    learning_rate: float = 0.005

    warmup_max_frames: int = 40
    warmup_median_min_frames: int = 15

    diff_threshold: int = 22

    morph_kernel: int = 3
    morph_iters: int = 2

    max_fg_ratio: float = 0.25  # camera shake / lighting change guard
    min_extent: float = 0.15
    min_solidity: float = 0.30


class BlobDetector:
    def __init__(self, cfg: BlobDetectorConfig):
        self.cfg = cfg
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(cfg.history),
            varThreshold=float(cfg.var_threshold),
            detectShadows=bool(cfg.detect_shadows),
        )
        self._warmup_grays: List[np.ndarray] = []
        self._median_bg: Optional[np.ndarray] = None

    def warmup(self, frame_bgr: np.ndarray, runway_mask: np.ndarray) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _ = self.bg.apply(gray, learningRate=0.5)

        m = cv2.bitwise_and(gray, gray, mask=runway_mask)
        if len(self._warmup_grays) < int(self.cfg.warmup_max_frames):
            self._warmup_grays.append(m)

        if self._median_bg is None and len(self._warmup_grays) >= int(self.cfg.warmup_median_min_frames):
            stack = np.stack(self._warmup_grays, axis=0)
            self._median_bg = np.median(stack, axis=0).astype(np.uint8)

    def detect(self, frame_bgr: np.ndarray, runway_mask: np.ndarray) -> Tuple[List[Tuple[int, int, int, int, float]], np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self.bg.apply(gray, learningRate=float(self.cfg.learning_rate))
        fg = cv2.bitwise_and(fg, runway_mask)

        # shadow suppression: with MOG2, 127 == shadow
        _, fg_bin = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # static change vs median background (keeps static new objects)
        if self._median_bg is not None:
            diff = cv2.absdiff(gray, self._median_bg)
            diff = cv2.bitwise_and(diff, diff, mask=runway_mask)
            _, diff_bin = cv2.threshold(diff, int(self.cfg.diff_threshold), 255, cv2.THRESH_BINARY)
            fg_bin = cv2.bitwise_or(fg_bin, diff_bin)

        fg_ratio = float(np.mean(fg_bin > 0))
        if fg_ratio > float(self.cfg.max_fg_ratio):
            return ([], fg_bin)

        k = np.ones((int(self.cfg.morph_kernel), int(self.cfg.morph_kernel)), np.uint8)
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, k, iterations=int(self.cfg.morph_iters))
        fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, k, iterations=max(1, int(self.cfg.morph_iters) - 1))
        fg_bin = cv2.dilate(fg_bin, k, iterations=1)

        contours, _ = cv2.findContours(fg_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[Tuple[int, int, int, int, float]] = []
        h, w = gray.shape[:2]

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < float(self.cfg.min_area) or area > float(self.cfg.max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 2 or bh <= 2:
                continue

            x1, y1, x2, y2 = clamp_bbox((x, y, x + bw, y + bh), w, h)
            bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))
            extent = float(area / bbox_area)

            hull = cv2.convexHull(c)
            hull_area = float(max(1.0, cv2.contourArea(hull)))
            solidity = float(area / hull_area)

            if extent < float(self.cfg.min_extent):
                continue
            if solidity < float(self.cfg.min_solidity):
                continue

            # heuristic confidence
            conf = 0.2 + 0.5 * min(1.0, extent / 0.7) + 0.3 * min(1.0, solidity / 0.9)
            conf = float(max(0.0, min(1.0, conf)))

            boxes.append((x1, y1, x2, y2, conf))

        return boxes, fg_bin


# ----------------------------
# Simple IoU blob tracker
# ----------------------------


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.25
    max_missed: int = 10


class BlobTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.next_id = 1
        self.tracks: Dict[int, Tuple[int, int, int, int]] = {}
        self.missed: Dict[int, int] = defaultdict(int)

    def update(self, boxes_xyxy: List[Tuple[int, int, int, int]]) -> List[Tuple[Tuple[int, int, int, int], int]]:
        assigned: Dict[int, Tuple[int, int, int, int]] = {}
        used_tracks: set[int] = set()

        for box in boxes_xyxy:
            best_iou = 0.0
            best_tid: Optional[int] = None
            for tid, tbox in self.tracks.items():
                if tid in used_tracks:
                    continue
                i = iou_xyxy(box, tbox)
                if i > best_iou:
                    best_iou = i
                    best_tid = tid
            if best_tid is not None and best_iou >= float(self.cfg.iou_threshold):
                assigned[best_tid] = box
                used_tracks.add(best_tid)
            else:
                tid = self.next_id
                self.next_id += 1
                assigned[tid] = box
                used_tracks.add(tid)

        new_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        new_missed: Dict[int, int] = defaultdict(int)

        for tid, box in assigned.items():
            new_tracks[tid] = box
            new_missed[tid] = 0

        for tid, tbox in self.tracks.items():
            if tid in assigned:
                continue
            m = self.missed.get(tid, 0) + 1
            if m <= int(self.cfg.max_missed):
                new_tracks[tid] = tbox
                new_missed[tid] = m

        self.tracks = new_tracks
        self.missed = new_missed

        return [(box, tid) for tid, box in assigned.items()]
