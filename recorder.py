"""Evidence recorder: snapshots + short clips with pre/post buffers.

Cross-platform notes:
- OpenCV video writing depends on available codecs. We attempt multiple container/codec pairs.
- If no writer can be opened, we keep the pipeline running and simply skip clip output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple
from collections import deque

import cv2
import numpy as np


def ensure_artifacts_dir() -> Path:
    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _try_video_writer(path: Path, fps: int, frame_size: Tuple[int, int], fourcc: str) -> Optional[cv2.VideoWriter]:
    w, h = frame_size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(w), int(h)))
    if writer is None:
        return None
    if not writer.isOpened():
        try:
            writer.release()
        except Exception:
            pass
        return None
    return writer


def create_best_effort_writer(base_path_no_ext: Path, fps: int, frame_size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], Optional[Path]]:
    """Try multiple codec/container pairs.

    Returns (writer, path_used). If writer is None, writing is unavailable.
    """
    # (extension, fourcc)
    candidates = [
        (".mp4", "mp4v"),
        (".avi", "MJPG"),
        (".avi", "XVID"),
    ]

    for ext, fourcc in candidates:
        path = base_path_no_ext.with_suffix(ext)
        w = _try_video_writer(path, fps=fps, frame_size=frame_size, fourcc=fourcc)
        if w is not None:
            return w, path

    return None, None


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


@dataclass
class RecorderConfig:
    fps: int
    frame_size: Tuple[int, int]  # (w, h)
    prebuffer_frames: int
    postbuffer_frames: int


class RecorderManager:
    def __init__(self, cfg: RecorderConfig):
        self.cfg = cfg
        self.prebuffer: Deque[np.ndarray] = deque(maxlen=int(cfg.prebuffer_frames))
        self.active: Dict[str, dict] = {}
        self.out_dir = ensure_artifacts_dir()

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        self.prebuffer.append(frame_bgr.copy())

        done = []
        for inc_id, rec in self.active.items():
            writer: Optional[cv2.VideoWriter] = rec.get("writer")
            if writer is not None:
                writer.write(frame_bgr)
            rec["remaining"] -= 1
            if rec["remaining"] <= 0:
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
                done.append(inc_id)

        for inc_id in done:
            del self.active[inc_id]

    def save_snapshot(self, incident_id: str, frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> str:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(bbox_xyxy, w, h)
        crop = frame_bgr[y1:y2, x1:x2]

        # If bbox is degenerate, save full frame with bbox overlay.
        if crop.size == 0 or (x2 - x1) < 8 or (y2 - y1) < 8:
            out = frame_bgr.copy()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            path = self.out_dir / f"{incident_id}_snapshot_full.jpg"
            cv2.imwrite(str(path), out)
            return str(path)

        path = self.out_dir / f"{incident_id}_snapshot.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    def start_recording(self, incident_id: str) -> Optional[str]:
        base = self.out_dir / f"{incident_id}_evidence"
        writer, path = create_best_effort_writer(base, fps=int(self.cfg.fps), frame_size=self.cfg.frame_size)

        # Even if writing is unavailable, we should not crash.
        if writer is None or path is None:
            self.active[incident_id] = {"writer": None, "remaining": int(self.cfg.postbuffer_frames)}
            return None

        for f in list(self.prebuffer):
            writer.write(f)

        self.active[incident_id] = {"writer": writer, "remaining": int(self.cfg.postbuffer_frames)}
        return str(path)
