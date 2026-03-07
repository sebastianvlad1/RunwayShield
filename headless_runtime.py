"""RunwayShield headless runtime.

Provides `RunwayShieldRuntime`: the core detection/tracking/incident pipeline
decoupled from any UI. Can be used:
  - From headless_runner.py (CLI, no Streamlit)
  - From app.py (Streamlit) as a shared backend

No Streamlit imports anywhere in this file.

Source modes
------------
  file        Normal file, process as fast as possible, stop at EOF.
  file_live   Process file at real-time pace (sleep between frames), loop
              infinitely at EOF to simulate a live stream.
  webcam      Index-based device (0, 1, …).
  rtsp        Any URL accepted by cv2.VideoCapture (rtsp://, http://, etc.).

For webcam/rtsp sources the loop is infinite; call `stop()` or send
KeyboardInterrupt for a clean shutdown.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from cv_pipeline import (
    GroundingDINOConfig,
    GroundingDINOPipeline,
    TrajectoryConfig,
    TrajectoryPredictor,
    TrackedObject,
    bbox_center,
    clamp_bbox,
    crop_roi,
    make_runway_mask,
    point_in_polygon,
    polygon_from_points,
    resize_keep_aspect,
    sort_polygon_points,
)
from incident import Incident, IncidentEngine, normalized_across_runway_distance, severity_for
from recorder import RecorderConfig, RecorderManager
from vlm_clip import ClipClassifier, ClipConfig

logger = logging.getLogger("runway_shield")


# ---------------------------------------------------------------------------
# Public config dataclass
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """All parameters needed to run the detection pipeline.

    Every field has a sensible default so callers only need to override
    what differs from the baseline.
    """

    # --- Video source ---
    source: str = "0"
    """Path to video file, 'webcam:<index>' (e.g. 'webcam:0'), or an RTSP URL."""

    source_mode: str = "file"
    """One of: file | file_live | webcam | rtsp"""

    loop: bool = True
    """Only used with source_mode='file_live'. Loop file infinitely."""

    # --- Frame sizing/rate ---
    proc_width: int = 960
    proc_fps: int = 15

    # --- Warmup ---
    warmup_secs: float = 3.0

    # --- Incident gating ---
    confirm_n: int = 6
    window_m: int = 10

    # --- Detection (GroundingDINO) ---
    yolo_model: str = "IDEA-Research/grounding-dino-tiny"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.50
    yolo_device: str = "auto"
    yolo_tracker_yaml: str = "botsort.yaml"  # kept for config compat; unused by GroundingDINO
    yolo_classes_en: List[str] = field(default_factory=list)

    # --- Trajectory ---
    horizon_frames: int = 10

    # --- Evidence ---
    prebuffer_secs: float = 5.0
    postbuffer_secs: float = 5.0

    # --- VLM ---
    enable_vlm: bool = False
    vlm_model: str = "ViT-B-32"
    vlm_pretrained: str = "laion2b_s34b_b79k"
    vlm_device: str = "auto"
    vlm_frames: int = 3
    vlm_unknown_threshold: float = 0.28
    vlm_shadow_threshold: float = 0.55
    vlm_debris_detail_threshold: float = 0.45

    # --- Output ---
    output_dir: str = "artifacts"
    log_name: str = "incidents.jsonl"

    # --- Reconnect (live sources) ---
    reconnect_max_attempts: int = 10
    reconnect_base_delay: float = 1.0  # seconds, doubles each attempt


# ---------------------------------------------------------------------------
# FrameResult returned per processed frame
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    new_incidents: List[Incident] = field(default_factory=list)
    """Incidents that just opened this frame."""

    tracked_count: int = 0
    """Number of tracked objects in-runway this frame."""

    effective_fps: float = 0.0
    is_warmup: bool = False


# ---------------------------------------------------------------------------
# Video source abstraction
# ---------------------------------------------------------------------------

class _VideoSource:
    """Wraps cv2.VideoCapture with pacing + reconnect + loop support."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._pace_period: float = 0.0
        self._last_pace_t: float = 0.0
        self._stopped = False

        self._open()

        # Determine frame skip
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 30.0
        if not src_fps or math.isnan(src_fps) or src_fps <= 0:
            src_fps = 30.0
        self._src_fps: float = src_fps
        self._frame_skip: int = max(1, int(round(src_fps / float(cfg.proc_fps))))

        # Pacing for file_live mode
        if cfg.source_mode == "file_live":
            self._pace_period = 1.0 / float(cfg.proc_fps)

    # -- public interface --

    @property
    def src_fps(self) -> float:
        return self._src_fps

    @property
    def frame_skip(self) -> int:
        return self._frame_skip

    @property
    def frame_count(self) -> int:
        """Total frames if known (file); 0 for live sources."""
        if self._cap is None:
            return 0
        fc = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return max(0, fc)

    def read_next(self) -> Optional[np.ndarray]:
        """Return next frame that should be processed (respects frame skip).
        Returns None when source is exhausted (file mode, no loop) or stopped.
        Raises no exception; reconnect is handled internally for live sources.
        """
        while not self._stopped:
            ok, frame = self._read_raw()
            if not ok:
                if self._cfg.source_mode == "file_live" and self._cfg.loop:
                    logger.info("[source] EOF reached — looping from start")
                    self._rewind()
                    continue
                elif self._cfg.source_mode in ("webcam", "rtsp"):
                    if self._reconnect():
                        continue
                    return None
                else:
                    return None  # file mode, normal EOF

            self._frame_idx += 1
            if (self._frame_idx - 1) % self._frame_skip != 0:
                continue

            # Pacing for file_live
            if self._cfg.source_mode == "file_live" and self._pace_period > 0:
                now = time.monotonic()
                if self._last_pace_t > 0:
                    wait = self._pace_period - (now - self._last_pace_t)
                    if wait > 0:
                        time.sleep(wait)
                self._last_pace_t = time.monotonic()

            return frame

        return None

    def stop(self) -> None:
        self._stopped = True
        self._release()

    # -- internals --

    def _cv2_source(self) -> str:
        mode = self._cfg.source_mode
        src = self._cfg.source
        if mode == "webcam":
            # support both int index and "webcam:N"
            idx = src.replace("webcam:", "").strip()
            return str(int(idx)) if idx.isdigit() else src
        return src  # file, file_live, rtsp

    def _open(self) -> bool:
        src = self._cv2_source()
        logger.info(f"[source] opening: {src}")
        try:
            cap_src: cv2.VideoCapture
            if src.isdigit():
                cap_src = cv2.VideoCapture(int(src))
            else:
                cap_src = cv2.VideoCapture(src)
            if cap_src.isOpened():
                self._cap = cap_src
                self._frame_idx = 0
                return True
            cap_src.release()
        except Exception as e:
            logger.warning(f"[source] open error: {e}")
        return False

    def _read_raw(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        try:
            ok, frame = self._cap.read()
            return ok, frame
        except Exception as e:
            logger.warning(f"[source] read error: {e}")
            return False, None

    def _rewind(self) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_idx = 0

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _reconnect(self) -> bool:
        max_attempts = self._cfg.reconnect_max_attempts
        base_delay = self._cfg.reconnect_base_delay
        self._release()
        for attempt in range(1, max_attempts + 1):
            delay = min(base_delay * (2 ** (attempt - 1)), 30.0)
            logger.warning(f"[source] reconnect attempt {attempt}/{max_attempts} in {delay:.1f}s …")
            time.sleep(delay)
            if self._open():
                logger.info("[source] reconnected successfully")
                return True
        logger.error("[source] exhausted reconnect attempts, giving up")
        return False


# ---------------------------------------------------------------------------
# Main runtime class
# ---------------------------------------------------------------------------

class RunwayShieldRuntime:
    """Core pipeline orchestrator — no UI dependencies.

    Typical headless usage::

        cfg = RuntimeConfig(source="runway.mp4", source_mode="file_live", ...)
        runtime = RunwayShieldRuntime(cfg, poly_points=[(x,y), ...])
        for result in runtime.run():
            for inc in result.new_incidents:
                print(inc.incident_id, inc.severity)
        runtime.shutdown()

    Callback-based usage (from Streamlit or another UI)::

        def on_frame(frame_bgr, result):
            ...  # update your UI

        runtime = RunwayShieldRuntime(cfg, poly_points=[...], frame_callback=on_frame)
        runtime.run_blocking()
    """

    def __init__(
        self,
        cfg: RuntimeConfig,
        poly_points: List[Tuple[float, float]],
        frame_callback: Optional[Callable[[np.ndarray, FrameResult], None]] = None,
        incident_callback: Optional[Callable[[Incident], None]] = None,
    ) -> None:
        if len(poly_points) < 3:
            raise ValueError("poly_points must contain at least 3 points")

        self._cfg = cfg
        self._callbacks_frame = frame_callback
        self._callbacks_incident = incident_callback
        self._stopped = False
        self._processed = 0
        self._t0: float = 0.0
        # Last frame state — exposed for Streamlit overlay
        self._last_frame: Optional[np.ndarray] = None
        self._last_tracked: List[TrackedObject] = []
        self._last_traj_map: dict = {}

        # Normalise polygon
        self._poly_points = sort_polygon_points(poly_points)
        self._poly = polygon_from_points(self._poly_points)

        # Resolve output paths
        self._art_dir = Path(cfg.output_dir)
        self._art_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._art_dir / cfg.log_name

        # Build video source (needed to get first frame for mask sizing)
        self._video = _VideoSource(cfg)

        # Grab first frame to establish frame size
        first = self._video.read_next()
        if first is None:
            raise RuntimeError(f"Cannot read first frame from source: {cfg.source}")
        first = resize_keep_aspect(first, target_width=cfg.proc_width)
        self._frame_h, self._frame_w = first.shape[:2]
        self._first_frame = first

        # Build runway mask
        self._runway_mask = make_runway_mask(first.shape, self._poly)

        effective_fps = cfg.proc_fps

        # Detection pipeline (GroundingDINO + IoU tracker)
        self._yolo = GroundingDINOPipeline(
            GroundingDINOConfig(
                model_name=cfg.yolo_model,
                conf_threshold=cfg.yolo_conf,
                iou_threshold=cfg.yolo_iou,
                device=cfg.yolo_device,
                classes_en=cfg.yolo_classes_en,
            )
        )

        # Trajectory predictor
        self._traj = TrajectoryPredictor(TrajectoryConfig(horizon_frames=cfg.horizon_frames))

        # Incident engine
        self._engine = IncidentEngine(confirm_n=cfg.confirm_n, window_m=cfg.window_m)

        # Recorder
        prebuf = max(1, int(cfg.prebuffer_secs * effective_fps))
        postbuf = max(1, int(cfg.postbuffer_secs * effective_fps))
        self._rec = RecorderManager(
            RecorderConfig(
                fps=effective_fps,
                frame_size=(self._frame_w, self._frame_h),
                prebuffer_frames=prebuf,
                postbuffer_frames=postbuf,
            )
        )

        # VLM classifier
        self._clip: Optional[ClipClassifier] = None
        if cfg.enable_vlm:
            self._clip = ClipClassifier(
                ClipConfig(
                    enabled=True,
                    model_name=cfg.vlm_model,
                    pretrained=cfg.vlm_pretrained,
                    device=cfg.vlm_device,
                    unknown_threshold=cfg.vlm_unknown_threshold,
                    shadow_artifact_threshold=cfg.vlm_shadow_threshold,
                    debris_detail_threshold=cfg.vlm_debris_detail_threshold,
                )
            )

        self._warmup_frames = max(0, int(cfg.warmup_secs * effective_fps))

        # Push first frame into prebuffer
        self._rec.push_frame(self._first_frame)

    # -- public properties --

    @property
    def first_frame(self) -> np.ndarray:
        """BGR first frame (useful for polygon preview in Streamlit)."""
        return self._first_frame

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._frame_w, self._frame_h)

    @property
    def processed_count(self) -> int:
        return self._processed

    @property
    def source_fps(self) -> float:
        return self._video.src_fps

    @property
    def frame_skip(self) -> int:
        return self._video.frame_skip

    @property
    def frame_count(self) -> int:
        return self._video.frame_count

    # -- stop signal --

    def stop(self) -> None:
        """Signal the run loop to exit cleanly."""
        self._stopped = True

    # -- main API --

    def run(self) -> Iterator[FrameResult]:
        """Generator: yields a FrameResult per processed frame.

        Stops when:
        - source is exhausted (file mode without loop)
        - stop() has been called
        - KeyboardInterrupt (propagated to caller)
        """
        self._t0 = time.time()
        self._save_jsonl({"event": "RUN_START", "source": self._cfg.source,
                          "mode": self._cfg.source_mode, "ts": time.time()})

        logger.info(f"[runtime] start — source={self._cfg.source} mode={self._cfg.source_mode} "
                    f"warmup={self._warmup_frames}f proc_fps={self._cfg.proc_fps}")

        try:
            while not self._stopped:
                frame = self._video.read_next()
                if frame is None:
                    break

                result = self._process_frame(frame)
                yield result

                if self._callbacks_frame is not None:
                    self._callbacks_frame(frame, result)

        finally:
            self._finalize()

    def run_blocking(self) -> None:
        """Consume run() fully — convenience for CLI usage."""
        for _ in self.run():
            pass

    def shutdown(self) -> None:
        """Release all resources. Safe to call multiple times."""
        self.stop()
        self._video.stop()

    # -- internal frame processor --

    def _process_frame(self, raw_frame: np.ndarray) -> FrameResult:
        frame = resize_keep_aspect(raw_frame, target_width=self._cfg.proc_width)
        if frame.shape[0] != self._frame_h or frame.shape[1] != self._frame_w:
            frame = cv2.resize(frame, (self._frame_w, self._frame_h), interpolation=cv2.INTER_AREA)

        self._rec.push_frame(frame)
        is_warmup = self._processed < self._warmup_frames

        if is_warmup:
            tracked_objects: List[TrackedObject] = []
        else:
            tracked_objects = self._yolo.infer_and_track(frame)

        trajectory_map = self._traj.update_and_predict(tracked_objects)

        seen_tids: List[int] = []
        for obj in tracked_objects:
            x1, y1, x2, y2 = clamp_bbox(obj.bbox_xyxy, self._frame_w, self._frame_h)
            cx, cy = bbox_center((x1, y1, x2, y2))
            in_runway_now = point_in_polygon(self._poly, cx, cy)

            pred = trajectory_map.get(int(obj.track_id))
            future_points = pred.future_points if pred is not None else []
            will_enter_runway = any(point_in_polygon(self._poly, px, py) for px, py in future_points)

            if not in_runway_now and not will_enter_runway:
                continue

            seen_tids.append(int(obj.track_id))
            self._engine.on_detection(
                track_id=int(obj.track_id),
                in_runway=bool(in_runway_now or will_enter_runway),
                bbox_xyxy=(x1, y1, x2, y2),
                conf=float(obj.confidence),
                detector_label=str(obj.class_name),
                predicted_intrusion=bool(will_enter_runway and not in_runway_now),
                predicted_points=future_points,
                trajectory_horizon=int(self._cfg.horizon_frames),
            )

        self._engine.update_tracks(seen_tids)

        new_incidents: List[Incident] = []
        if not is_warmup:
            new_incidents = self._engine.maybe_open_incidents(
                frame_idx=self._processed,
                poly_points=self._poly_points,
                bbox_center_fn=bbox_center,
                recorder=self._rec,
                frame_bgr=frame,
            )

        # Classify new incidents with VLM
        for inc in new_incidents:
            self._save_jsonl({"event": "INCIDENT_OPEN", "ts": time.time(),
                              "incident_id": inc.incident_id,
                              "track_id": inc.track_id,
                              "severity": inc.severity,
                              "bbox": list(inc.bbox_xyxy),
                              "detector_label": inc.detector_label})

            if self._clip is not None:
                rois: List[np.ndarray] = []
                prebuf = list(self._rec.prebuffer)
                k = int(self._cfg.vlm_frames)
                if k > 1 and len(prebuf) > 1:
                    idxs = np.linspace(0, len(prebuf) - 1, num=k - 1).round().astype(int).tolist()
                    for idx in idxs:
                        rois.append(crop_roi(prebuf[idx], inc.bbox_xyxy, pad=0.20))
                rois.append(crop_roi(frame, inc.bbox_xyxy, pad=0.20))

                res = self._clip.classify_rois(rois)
                inc.vlm_processed_ts = time.time()
                inc.vlm_category = res.category
                inc.vlm_detail = res.detail
                inc.vlm_real = bool(res.real)
                inc.vlm_confidence = float(res.confidence)
                inc.vlm_error = res.error

                cx, cy = bbox_center(inc.bbox_xyxy)
                dist = normalized_across_runway_distance((cx, cy), self._poly_points)
                inc.severity = severity_for(res.category, in_centerline=(dist < 0.25))

                if res.ok and (res.real is False):
                    inc.status = "DISMISSED"
                    self._save_jsonl({"event": "INCIDENT_DISMISSED", "ts": time.time(),
                                      "incident_id": inc.incident_id,
                                      "category": res.category, "conf": res.confidence})
                else:
                    self._save_jsonl({"event": "INCIDENT_CLASSIFIED", "ts": time.time(),
                                      "incident_id": inc.incident_id,
                                      "category": res.category, "detail": res.detail,
                                      "conf": res.confidence, "real": res.real})

            if self._callbacks_incident is not None:
                self._callbacks_incident(inc)

        elapsed = time.time() - self._t0
        fps_eff = self._processed / max(1e-6, elapsed)

        result = FrameResult(
            frame_index=self._processed,
            timestamp=time.time(),
            new_incidents=new_incidents,
            tracked_count=len(seen_tids),
            effective_fps=fps_eff,
            is_warmup=is_warmup,
        )

        # Store for Streamlit overlay access
        self._last_frame = frame
        self._last_tracked = tracked_objects
        self._last_traj_map = trajectory_map

        self._processed += 1
        return result

    def _finalize(self) -> None:
        self._save_jsonl({"event": "RUN_END", "ts": time.time(),
                          "processed_frames": self._processed,
                          "incidents": len(self._engine.incidents)})
        self._video.stop()
        logger.info(f"[runtime] finished — {self._processed} frames processed, "
                    f"{len(self._engine.incidents)} incidents total")

    def _save_jsonl(self, record: dict) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[runtime] jsonl write error: {e}")

    # -- helpers exposed for Streamlit overlay drawing --

    def get_last_overlay(
        self,
        frame: Optional[np.ndarray] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        trajectory_map: Optional[dict] = None,
        en_to_ro_label: Optional[Dict[str, str]] = None,
    ) -> Optional[np.ndarray]:
        """Draw detections + polygon on frame; returns BGR overlay copy.

        When called with no arguments, uses the last processed frame and
        detection results stored internally (for Streamlit use).
        Returns None if no frame has been processed yet.
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return None
        if tracked_objects is None:
            tracked_objects = self._last_tracked
        if trajectory_map is None:
            trajectory_map = self._last_traj_map
        overlay = frame.copy()
        cv2.polylines(overlay, [self._poly], isClosed=True, color=(0, 255, 0), thickness=2)

        en_to_ro = en_to_ro_label or {}
        for obj in tracked_objects:
            x1, y1, x2, y2 = clamp_bbox(obj.bbox_xyxy, self._frame_w, self._frame_h)
            cx, cy = bbox_center((x1, y1, x2, y2))
            pred = trajectory_map.get(int(obj.track_id))
            future_pts = pred.future_points if pred is not None else []
            in_runway = point_in_polygon(self._poly, cx, cy)
            will_enter = any(point_in_polygon(self._poly, px, py) for px, py in future_pts)
            if not in_runway and not will_enter:
                continue
            color = (0, 0, 255) if in_runway else (0, 255, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            lbl = en_to_ro.get(obj.class_name, obj.class_name)
            cv2.putText(overlay, f"id#{obj.track_id} {lbl} {obj.confidence:.2f}",
                        (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, color, 2, cv2.LINE_AA)
            if future_pts:
                for px, py in future_pts:
                    cv2.circle(overlay, (int(px), int(py)), 2, (255, 200, 0), -1)
                pts_arr = np.array([[int(px), int(py)] for px, py in future_pts],
                                   dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts_arr], isClosed=False, color=(255, 200, 0), thickness=2)
        return overlay

    def incidents_table(self):
        """Return pd.DataFrame of all incidents (delegates to engine)."""
        return self._engine.to_table()
