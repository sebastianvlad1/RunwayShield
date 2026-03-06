"""Incident model + gating logic.

An incident is opened only when a blob persists on the runway for N of the last M frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Incident:
    incident_id: str
    status: str  # OPEN | DISMISSED | CLOSED
    severity: str  # LOW | MED | HIGH

    first_frame: int
    last_frame: int
    track_id: int

    bbox_xyxy: Tuple[int, int, int, int]
    candidate_conf: float
    detector_label: Optional[str] = None
    predicted_intrusion: bool = False
    trajectory_horizon: int = 0
    predicted_points: Optional[List[Tuple[float, float]]] = None

    evidence_image_path: Optional[str] = None
    evidence_video_path: Optional[str] = None

    # VLM (CLIP) classification
    vlm_category: Optional[str] = None
    vlm_detail: Optional[str] = None
    vlm_real: Optional[bool] = None
    vlm_confidence: Optional[float] = None
    vlm_error: Optional[str] = None
    vlm_processed_ts: Optional[float] = None


def severity_for(category: str, in_centerline: bool) -> str:
    k = (category or "unknown").lower().strip()
    if k in ("person", "vehicle", "animal"):
        return "HIGH"
    if k == "bird":
        return "HIGH" if in_centerline else "MED"
    if k == "debris":
        return "HIGH" if in_centerline else "MED"
    if k == "shadow":
        return "LOW"
    if k == "unknown":
        return "MED"
    return "MED"


def _estimate_centerline(poly_points: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.array(poly_points, dtype=np.float32)
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = (X.T @ X) / max(1, len(pts) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal = eigvecs[:, order[0]]
    perp = np.array([-principal[1], principal[0]], dtype=np.float32)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    return mean, perp


def normalized_across_runway_distance(point_xy: Tuple[float, float], poly_points: List[Tuple[float, float]]) -> float:
    """0=centerline, 1=edge (approx)"""
    mean, perp = _estimate_centerline(poly_points)
    pts = np.array(poly_points, dtype=np.float32)
    proj = (pts - mean) @ perp
    half_width = max(1e-6, float(np.max(np.abs(proj))))

    p = np.array(point_xy, dtype=np.float32)
    d = float((p - mean) @ perp)
    return min(1.0, abs(d) / half_width)


class IncidentEngine:
    def __init__(self, confirm_n: int, window_m: int):
        self.confirm_n = int(confirm_n)
        self.window_m = int(window_m)

        self.track_hist: Dict[int, Deque[bool]] = defaultdict(lambda: deque(maxlen=self.window_m))
        self.track_last_bbox: Dict[int, Tuple[int, int, int, int]] = {}
        self.track_last_conf: Dict[int, float] = {}
        self.track_last_label: Dict[int, str] = {}
        self.track_pred_intrusion: Dict[int, bool] = defaultdict(bool)
        self.track_pred_points: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.track_traj_horizon: Dict[int, int] = defaultdict(int)

        self.incidents: Dict[str, Incident] = {}
        self.incident_by_track: Dict[int, str] = {}

    def _new_incident_id(self) -> str:
        return f"INC-{int(time.time() * 1000)}"

    def update_tracks(self, seen_track_ids: List[int]) -> None:
        """Append False for tracks not seen in this frame."""
        seen = set(seen_track_ids)
        for tid, hist in list(self.track_hist.items()):
            if tid not in seen:
                hist.append(False)

    def on_detection(
        self,
        track_id: int,
        in_runway: bool,
        bbox_xyxy: Tuple[int, int, int, int],
        conf: float,
        detector_label: Optional[str] = None,
        predicted_intrusion: bool = False,
        predicted_points: Optional[List[Tuple[float, float]]] = None,
        trajectory_horizon: int = 0,
    ) -> None:
        self.track_hist[track_id].append(bool(in_runway))
        self.track_last_bbox[track_id] = bbox_xyxy
        self.track_last_conf[track_id] = float(conf)
        if detector_label:
            self.track_last_label[track_id] = str(detector_label)
        self.track_pred_intrusion[track_id] = bool(predicted_intrusion)
        self.track_pred_points[track_id] = list(predicted_points or [])
        self.track_traj_horizon[track_id] = int(trajectory_horizon)

    def maybe_open_incidents(
        self,
        frame_idx: int,
        poly_points: List[Tuple[float, float]],
        bbox_center_fn,
        recorder,
        frame_bgr,
    ) -> List[Incident]:
        new_incidents: List[Incident] = []
        for tid, hist in self.track_hist.items():
            if tid in self.incident_by_track:
                # update existing incident
                inc_id = self.incident_by_track[tid]
                inc = self.incidents[inc_id]
                inc.last_frame = frame_idx
                inc.bbox_xyxy = self.track_last_bbox.get(tid, inc.bbox_xyxy)
                inc.candidate_conf = self.track_last_conf.get(tid, inc.candidate_conf)
                inc.detector_label = self.track_last_label.get(tid, inc.detector_label)
                inc.predicted_intrusion = bool(self.track_pred_intrusion.get(tid, inc.predicted_intrusion))
                inc.predicted_points = self.track_pred_points.get(tid, inc.predicted_points)
                inc.trajectory_horizon = int(self.track_traj_horizon.get(tid, inc.trajectory_horizon))
                continue

            if sum(hist) >= self.confirm_n:
                bbox = self.track_last_bbox.get(tid)
                if bbox is None:
                    continue

                cx, cy = bbox_center_fn(bbox)
                dist = normalized_across_runway_distance((cx, cy), poly_points)
                in_centerline = dist < 0.25

                inc_id = self._new_incident_id()

                # snapshot + clip
                img_path = recorder.save_snapshot(inc_id, frame_bgr, bbox)
                vid_path = recorder.start_recording(inc_id)

                # initial severity: candidate until VLM refines
                severity = severity_for("unknown", in_centerline=in_centerline)

                inc = Incident(
                    incident_id=inc_id,
                    status="OPEN",
                    severity=severity,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    track_id=tid,
                    bbox_xyxy=bbox,
                    candidate_conf=float(self.track_last_conf.get(tid, 0.6)),
                    detector_label=self.track_last_label.get(tid),
                    predicted_intrusion=bool(self.track_pred_intrusion.get(tid, False)),
                    trajectory_horizon=int(self.track_traj_horizon.get(tid, 0)),
                    predicted_points=self.track_pred_points.get(tid),
                    evidence_image_path=img_path,
                    evidence_video_path=vid_path,
                )
                self.incidents[inc_id] = inc
                self.incident_by_track[tid] = inc_id
                new_incidents.append(inc)

        return new_incidents

    def to_table(self) -> pd.DataFrame:
        rows = [asdict(v) for v in self.incidents.values()]
        if not rows:
            return pd.DataFrame(columns=list(Incident.__dataclass_fields__.keys()))
        df = pd.DataFrame(rows)

        status_order = {"OPEN": 0, "DISMISSED": 1, "CLOSED": 2}
        sev_order = {"HIGH": 0, "MED": 1, "LOW": 2}

        df["_status_order"] = df["status"].map(status_order).fillna(99)
        df["_sev_order"] = df["severity"].map(sev_order).fillna(99)

        df = df.sort_values(by=["_status_order", "_sev_order", "first_frame"], ascending=[True, True, False])
        df = df.drop(columns=["_status_order", "_sev_order"])

        preferred = [
            "incident_id",
            "status",
            "severity",
            "vlm_category",
            "vlm_detail",
            "vlm_real",
            "vlm_confidence",
            "first_frame",
            "last_frame",
            "track_id",
            "detector_label",
            "predicted_intrusion",
            "trajectory_horizon",
            "candidate_conf",
            "evidence_video_path",
            "evidence_image_path",
            "vlm_error",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        return df[cols]
