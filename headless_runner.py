"""RunwayShield headless CLI runner.

Runs the full detection/tracking/incident pipeline without any UI.
Designed for integration into external applications or automated pipelines.

Usage examples
--------------

  # Process a video file (finish at EOF):
  python headless_runner.py --source path/to/video.mp4 --polygon polygon.json

  # Simulate a live stream from a file (real-time pacing, loop infinitely):
  python headless_runner.py --source path/to/video.mp4 --mode file_live --polygon polygon.json

  # Webcam index 0:
  python headless_runner.py --source 0 --mode webcam --polygon polygon.json

  # RTSP stream:
  python headless_runner.py --source rtsp://user:pass@192.168.1.10/stream --mode rtsp --polygon polygon.json

Polygon file format (JSON)
--------------------------
  {"points": [[x1,y1], [x2,y2], [x3,y3], ...]}

  Or YAML:
  points:
    - [x1, y1]
    - [x2, y2]

Minimum 3 points required.

Output
------
  - artifacts/incidents.jsonl  (event log, JSONL format)
  - artifacts/*.jpg            (incident snapshots)
  - artifacts/*.mp4            (evidence clips with pre/post buffer)
  - Stdout: JSON lines per new incident (for piping to external consumers)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from headless_runtime import FrameResult, RunwayShieldRuntime, RuntimeConfig
from incident import Incident


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Polygon loading
# ---------------------------------------------------------------------------

def load_polygon(path_str: str) -> List[Tuple[float, float]]:
    """Load polygon points from a JSON or YAML file.

    Expected JSON: {"points": [[x,y], ...]}
    Expected YAML: points:\n  - [x, y]\\n  ...
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Polygon file not found: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    raw = data.get("points", [])
    if not raw or len(raw) < 3:
        raise ValueError(
            f"Polygon file must contain at least 3 points under 'points' key. "
            f"Got: {raw}"
        )

    points: List[Tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
        elif isinstance(item, dict):
            points.append((float(item["x"]), float(item["y"])))
        else:
            raise ValueError(f"Cannot parse polygon point: {item!r}")

    return points


# ---------------------------------------------------------------------------
# Config loading from YAML (optional, merges with CLI args)
# ---------------------------------------------------------------------------

def _load_yolo_cfg_from_yaml(yaml_path: Path) -> dict:
    """Load detection config (YOLO classes etc.) from project config.yaml."""
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return data.get("detection", {})


def _build_runtime_config(args: argparse.Namespace, detection_cfg: dict) -> RuntimeConfig:
    classes_raw = detection_cfg.get("classes", [])
    classes_en = []
    for c in classes_raw:
        if isinstance(c, dict):
            en = str(c.get("en", "")).strip()
            ro = str(c.get("ro", "")).strip()
            if en:
                classes_en.append(en)
            elif ro:
                classes_en.append(ro)

    return RuntimeConfig(
        source=str(args.source),
        source_mode=str(args.mode),
        loop=bool(args.loop),
        proc_width=int(args.proc_width),
        proc_fps=int(args.proc_fps),
        warmup_secs=float(args.warmup_secs),
        confirm_n=int(args.confirm_n),
        window_m=int(args.window_m),
        yolo_model=str(detection_cfg.get("model", "yolov8s-world.pt")),
        yolo_conf=float(args.yolo_conf or detection_cfg.get("conf_threshold", 0.25)),
        yolo_iou=float(args.yolo_iou or detection_cfg.get("iou_threshold", 0.50)),
        yolo_tracker_yaml=str(detection_cfg.get("tracker_yaml", "botsort.yaml")),
        yolo_classes_en=classes_en,
        horizon_frames=int(args.horizon_frames),
        prebuffer_secs=float(args.prebuffer_secs),
        postbuffer_secs=float(args.postbuffer_secs),
        enable_vlm=bool(args.enable_vlm),
        vlm_model=str(args.vlm_model),
        vlm_pretrained=str(args.vlm_pretrained),
        vlm_device=str(args.vlm_device),
        vlm_frames=int(args.vlm_frames),
        vlm_unknown_threshold=float(args.vlm_unknown_threshold),
        vlm_shadow_threshold=float(args.vlm_shadow_threshold),
        vlm_debris_detail_threshold=float(args.vlm_debris_detail_threshold),
        output_dir=str(args.output_dir),
        log_name="incidents.jsonl",
        reconnect_max_attempts=int(args.reconnect_attempts),
        reconnect_base_delay=float(args.reconnect_delay),
    )


# ---------------------------------------------------------------------------
# Incident output
# ---------------------------------------------------------------------------

def _print_incident(inc: Incident) -> None:
    """Emit one JSON line to stdout per new incident (for piping)."""
    record = {
        "incident_id": inc.incident_id,
        "status": inc.status,
        "severity": inc.severity,
        "track_id": inc.track_id,
        "detector_label": inc.detector_label,
        "bbox": list(inc.bbox_xyxy),
        "vlm_category": inc.vlm_category,
        "vlm_detail": inc.vlm_detail,
        "vlm_confidence": inc.vlm_confidence,
        "vlm_real": inc.vlm_real,
        "evidence_image": inc.evidence_image_path,
        "evidence_video": inc.evidence_video_path,
    }
    print(json.dumps(record, ensure_ascii=False, default=str), flush=True)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="headless_runner",
        description="RunwayShield headless pipeline — no UI required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", required=True,
                   help="Video path, webcam index (0), or RTSP URL.")
    p.add_argument("--mode", default="file",
                   choices=["file", "file_live", "webcam", "rtsp"],
                   help="Source mode. Use 'file_live' to simulate live from a file.")
    p.add_argument("--polygon", required=True,
                   help="Path to polygon JSON/YAML file (min 3 points).")
    p.add_argument("--config", default="config.yaml",
                   help="Project config.yaml with YOLO class definitions.")
    p.add_argument("--loop", action="store_true", default=False,
                   help="In file_live mode, loop file infinitely at EOF.")
    p.add_argument("--output-dir", default="artifacts",
                   help="Directory for evidence + JSONL log.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging.")

    perf = p.add_argument_group("Performance")
    perf.add_argument("--proc-width", type=int, default=960,
                      help="Processing frame width in pixels.")
    perf.add_argument("--proc-fps", type=int, default=15,
                      help="Target processing FPS (file_live pacing).")

    gating = p.add_argument_group("Incident gating")
    gating.add_argument("--confirm-n", type=int, default=6,
                        help="Frames track must be in-runway to open incident.")
    gating.add_argument("--window-m", type=int, default=10,
                        help="Sliding window size for N-of-M gating.")
    gating.add_argument("--warmup-secs", type=float, default=3.0,
                        help="Background warmup seconds before detection starts.")

    yolo = p.add_argument_group("YOLO detection")
    yolo.add_argument("--yolo-conf", type=float, default=None,
                      help="YOLO confidence threshold (overrides config.yaml).")
    yolo.add_argument("--yolo-iou", type=float, default=None,
                      help="YOLO IoU threshold (overrides config.yaml).")
    yolo.add_argument("--horizon-frames", type=int, default=10,
                      help="Kalman trajectory prediction horizon (frames).")

    ev = p.add_argument_group("Evidence buffers")
    ev.add_argument("--prebuffer-secs", type=float, default=5.0,
                    help="Seconds of pre-incident video to include in clip.")
    ev.add_argument("--postbuffer-secs", type=float, default=5.0,
                    help="Seconds of post-incident video to include in clip.")

    vlm = p.add_argument_group("VLM classification (optional)")
    vlm.add_argument("--enable-vlm", action="store_true",
                     help="Enable OpenCLIP classification of incident ROI.")
    vlm.add_argument("--vlm-model", default="ViT-B-32",
                     choices=["ViT-B-32", "ViT-L-14"])
    vlm.add_argument("--vlm-pretrained", default="laion2b_s34b_b79k")
    vlm.add_argument("--vlm-device", default="auto",
                     choices=["auto", "cpu", "cuda", "mps"])
    vlm.add_argument("--vlm-frames", type=int, default=3,
                     help="Number of ROI frames for temporal averaging.")
    vlm.add_argument("--vlm-unknown-threshold", type=float, default=0.28)
    vlm.add_argument("--vlm-shadow-threshold", type=float, default=0.55)
    vlm.add_argument("--vlm-debris-detail-threshold", type=float, default=0.45)

    live = p.add_argument_group("Live source / reconnect")
    live.add_argument("--reconnect-attempts", type=int, default=10,
                      help="Max reconnect attempts for webcam/RTSP on failure.")
    live.add_argument("--reconnect-delay", type=float, default=1.0,
                      help="Base delay in seconds between reconnect attempts (doubles each time).")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("runner")

    # Validate confirm_n <= window_m
    if args.confirm_n > args.window_m:
        parser.error(f"--confirm-n ({args.confirm_n}) must be <= --window-m ({args.window_m})")

    # Load polygon
    try:
        poly_points = load_polygon(args.polygon)
    except (FileNotFoundError, ValueError, KeyError) as e:
        log.error(f"Polygon error: {e}")
        return 1

    # Load detection config
    detection_cfg = _load_yolo_cfg_from_yaml(Path(args.config))

    # Build runtime config
    cfg = _build_runtime_config(args, detection_cfg)

    log.info(f"Source: {cfg.source} | Mode: {cfg.source_mode} | "
             f"Loop: {cfg.loop} | Polygon: {len(poly_points)} points")

    # Build runtime
    try:
        runtime = RunwayShieldRuntime(cfg, poly_points=poly_points)
    except (RuntimeError, ValueError) as e:
        log.error(f"Runtime init failed: {e}")
        return 1

    log.info(f"Frame size: {runtime.frame_size} | Source FPS: {runtime.source_fps:.1f} | "
             f"Frame skip: {runtime.frame_skip} | Warmup frames: {int(cfg.warmup_secs * cfg.proc_fps)}")

    # Graceful shutdown on SIGINT/SIGTERM
    def _handle_signal(sig, _frame):
        log.info(f"Signal {sig} received — stopping …")
        runtime.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    # Run
    total_incidents = 0
    try:
        for result in runtime.run():
            for inc in result.new_incidents:
                total_incidents += 1
                _print_incident(inc)
                log.info(
                    f"[INCIDENT] {inc.incident_id} | severity={inc.severity} | "
                    f"label={inc.detector_label} | vlm={inc.vlm_category} | status={inc.status}"
                )

            if result.frame_index % 50 == 0 and not result.is_warmup:
                log.info(
                    f"frame={result.frame_index} | fps={result.effective_fps:.1f} | "
                    f"tracked={result.tracked_count} | incidents_total={total_incidents}"
                )
    except Exception as e:
        log.error(f"Unexpected error in run loop: {e}", exc_info=True)
        runtime.shutdown()
        return 1

    runtime.shutdown()
    log.info(f"Done. Processed {runtime.processed_count} frames, "
             f"{total_incidents} incidents. Artifacts in '{cfg.output_dir}/'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
