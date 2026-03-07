from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from cv_pipeline import (
    resize_keep_aspect,
    sort_polygon_points,
)
from headless_runtime import RunwayShieldRuntime, RuntimeConfig
from vlm_clip import CATEGORIES


# ----------------------------
# Utils
# ----------------------------


def _default_detection_cfg() -> dict:
    return {
        "model": "yolov8s-world.pt",
        "conf_threshold": 0.25,
        "iou_threshold": 0.5,
        "tracker_yaml": "botsort.yaml",
        "classes": [
            {"id": 0, "ro": "persoana cu rucsac", "en": "person with backpack"},
            {"id": 1, "ro": "vehicul de pista", "en": "runway vehicle"},
            {"id": 2, "ro": "pasare", "en": "bird"},
            {"id": 3, "ro": "resturi pe pista", "en": "runway debris"},
        ],
    }


def load_yolo_world_config(path: Path) -> dict:
    if not path.exists():
        cfg = {"detection": _default_detection_cfg()}
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return cfg

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    detection = data.get("detection") or {}
    merged = _default_detection_cfg()
    merged.update({k: v for k, v in detection.items() if k in merged and v is not None})
    classes = detection.get("classes") if isinstance(detection, dict) else None
    if isinstance(classes, list) and classes:
        merged["classes"] = classes
    return {"detection": merged}


def yolo_class_prompts_en(cfg: dict) -> List[str]:
    classes = cfg.get("detection", {}).get("classes", [])
    prompts: List[str] = []
    for c in classes:
        if not isinstance(c, dict):
            continue
        en = str(c.get("en", "")).strip()
        ro = str(c.get("ro", "")).strip()
        if en:
            prompts.append(en)
        elif ro:
            prompts.append(ro)
    return prompts


def yolo_label_maps(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    en_to_ro: Dict[str, str] = {}
    en_to_en: Dict[str, str] = {}
    classes = cfg.get("detection", {}).get("classes", [])
    for c in classes:
        if not isinstance(c, dict):
            continue
        en = str(c.get("en", "")).strip()
        ro = str(c.get("ro", "")).strip()
        if not en:
            continue
        en_to_ro[en] = ro or en
        en_to_en[en] = en
    return en_to_ro, en_to_en


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="RunwayShield PoC", layout="wide")

st.title("RunwayShield — Automatic Runway Hazard Detection")
st.caption("YOLO-World + BoT-SORT + Kalman trajectory prediction → N-of-M incident gating → evidence")

config_path = Path("config.yaml")
yolo_cfg_data = load_yolo_world_config(config_path)
prompt_classes_en = yolo_class_prompts_en(yolo_cfg_data)
en_to_ro_label, _ = yolo_label_maps(yolo_cfg_data)

loop_sim: bool = True  # default; overridden in sidebar when simulate-live is selected

with st.sidebar:
    st.header("Video Source")
    source_mode = st.radio(
        "Source type",
        options=["Upload file", "Live — webcam", "Live — RTSP/URL", "Simulate live (file)"],
        index=0,
        help=(
            "Upload file: process offline video.\n"
            "Live — webcam: use a connected camera.\n"
            "Live — RTSP/URL: use a network stream.\n"
            "Simulate live (file): replay file at real-time pace, looping infinitely."
        ),
    )

    uploaded = None
    live_webcam_idx = 0
    live_rtsp_url = ""
    live_file_path = ""

    if source_mode == "Upload file":
        uploaded = st.file_uploader("Upload runway video", type=["mp4", "mov", "mkv", "avi"])
    elif source_mode == "Live — webcam":
        live_webcam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    elif source_mode == "Live — RTSP/URL":
        live_rtsp_url = st.text_input("Stream URL", placeholder="rtsp://user:pass@192.168.1.10/stream")
    else:  # Simulate live (file)
        sim_uploaded = st.file_uploader("Upload video to simulate live", type=["mp4", "mov", "mkv", "avi"],
                                        key="sim_upload")
        loop_sim = st.checkbox("Loop infinitely", value=True)
        if sim_uploaded is not None:
            _sim_id = f"{sim_uploaded.name}_{sim_uploaded.size}"
            if (st.session_state.get("_sim_upload_id") != _sim_id
                    or not Path(st.session_state.get("_sim_upload_path", "")).exists()):
                art_tmp = Path("artifacts/tmp")
                art_tmp.mkdir(parents=True, exist_ok=True)
                suffix = Path(sim_uploaded.name).suffix.lower()
                if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
                    suffix = ".mp4"
                _sim_path = str(art_tmp / f"sim_{int(time.time()*1000)}{suffix}")
                with open(_sim_path, "wb") as _f:
                    _f.write(sim_uploaded.getbuffer())
                st.session_state["_sim_upload_id"] = _sim_id
                st.session_state["_sim_upload_path"] = _sim_path
            live_file_path = st.session_state.get("_sim_upload_path", "")

    st.header("Performance")
    proc_width = st.select_slider("Processing width (px)", options=[640, 800, 960, 1120, 1280], value=960)
    proc_fps = st.select_slider("Process FPS (approx)", options=[5, 10, 15, 20, 30], value=15)

    st.header("Incident gating")
    window_m = st.number_input("Window M (frames)", min_value=5, max_value=120, value=10, step=1)
    confirm_n = st.number_input("Confirm N (in-runway frames)", min_value=2, max_value=60, value=6, step=1)

    st.header("YOLO-World detection")
    yolo_conf = st.slider("YOLO confidence", 0.05, 0.95, float(yolo_cfg_data["detection"]["conf_threshold"]), 0.01)
    yolo_iou = st.slider("YOLO IoU", 0.05, 0.95, float(yolo_cfg_data["detection"]["iou_threshold"]), 0.01)
    warmup_secs = st.number_input("Warm-up seconds", min_value=0, max_value=30, value=3, step=1)

    st.header("Trajectory prediction")
    horizon_frames = st.slider("Prediction horizon N (frames)", 3, 60, 10, 1)

    st.header("Evidence")
    prebuffer_secs = st.number_input("Pre-buffer seconds", min_value=1, max_value=30, value=5, step=1)
    postbuffer_secs = st.number_input("Post-buffer seconds", min_value=1, max_value=30, value=5, step=1)

    st.header("VLM classification (fast + stable)")
    enable_vlm = st.checkbox("Enable VLM classify", value=False)

    vlm_model = "ViT-B-32"
    vlm_pretrained = "laion2b_s34b_b79k"
    vlm_device = "auto"
    unknown_thr = 0.28
    shadow_thr = 0.55
    debris_detail_thr = 0.45
    vlm_frames = 3

    if enable_vlm:
        st.caption("Uses OpenCLIP (zero-shot). First run downloads ~400MB weights once.")
        vlm_model = st.selectbox("CLIP model", ["ViT-B-32", "ViT-L-14"], index=0)
        vlm_pretrained = st.selectbox(
            "Pretrained weights",
            ["laion2b_s34b_b79k", "laion2b_s32b_b82k"],
            index=0,
        )
        vlm_device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        vlm_frames = st.slider("ROI frames (temporal)", 1, 5, 3, 1)
        unknown_thr = st.slider("Unknown threshold", 0.05, 0.60, 0.28, 0.01)
        shadow_thr = st.slider("Shadow artefact threshold", 0.20, 0.95, 0.55, 0.01)
        debris_detail_thr = st.slider("Debris detail threshold", 0.10, 0.95, 0.45, 0.01)

    st.divider()
    st.caption(f"YOLO classes loaded from `{config_path}`")
    for cls in yolo_cfg_data["detection"]["classes"]:
        st.write(f"- {cls.get('ro', '-')}: `{cls.get('en', '-')}`")

    run_btn = st.button("▶ Start", type="primary")


# ----------------------------
# Resolve video source + first-frame probe (cached per source in session_state)
# Caching avoids re-opening webcam/RTSP on every sidebar widget interaction
# and prevents duplicate temp files for uploaded videos.
# ----------------------------

_source_str: str = ""
_source_mode_key: str = "file"

if source_mode == "Upload file":
    if uploaded is None:
        st.info("Upload a runway video to begin.")
        st.stop()
    _upload_id = f"{uploaded.name}_{uploaded.size}"
    if (st.session_state.get("_upload_id") != _upload_id
            or not Path(st.session_state.get("_upload_path", "")).exists()):
        art_tmp = Path("artifacts/tmp")
        art_tmp.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
            suffix = ".mp4"
        _path = str(art_tmp / f"upload_{int(time.time()*1000)}{suffix}")
        with open(_path, "wb") as _f:
            _f.write(uploaded.getbuffer())
        _cap = cv2.VideoCapture(_path)
        _ok, _raw = _cap.read()
        _cap.release()
        if not _ok or _raw is None:
            st.error("Could not read first frame from the uploaded video.")
            st.stop()
        st.session_state["_upload_id"] = _upload_id
        st.session_state["_upload_path"] = _path
        st.session_state["_upload_frame"] = _raw
    _source_str = st.session_state["_upload_path"]
    _first_frame_raw = st.session_state["_upload_frame"]
    _source_mode_key = "file"

elif source_mode == "Live — webcam":
    _source_str = str(int(live_webcam_idx))
    _source_mode_key = "webcam"
    _probe_key = f"_probe_webcam_{_source_str}"
    if _probe_key not in st.session_state:
        _cap = cv2.VideoCapture(int(_source_str))
        _ok, _raw = _cap.read()
        _cap.release()
        if not _ok or _raw is None:
            st.error(f"Could not read from camera index {_source_str}. Check the camera.")
            st.stop()
        st.session_state[_probe_key] = _raw
    _first_frame_raw = st.session_state[_probe_key]

elif source_mode == "Live — RTSP/URL":
    if not live_rtsp_url.strip():
        st.info("Enter a stream URL to begin.")
        st.stop()
    _source_str = live_rtsp_url.strip()
    _source_mode_key = "rtsp"
    _probe_key = f"_probe_rtsp_{_source_str}"
    if _probe_key not in st.session_state:
        _cap = cv2.VideoCapture(_source_str)
        _ok, _raw = _cap.read()
        _cap.release()
        if not _ok or _raw is None:
            st.error("Could not read first frame from the stream. Check the URL.")
            st.stop()
        st.session_state[_probe_key] = _raw
    _first_frame_raw = st.session_state[_probe_key]

else:  # Simulate live
    if not live_file_path:
        st.info("Upload a video to simulate live stream.")
        st.stop()
    _source_str = live_file_path
    _source_mode_key = "file_live"
    _probe_key = f"_probe_sim_{_source_str}"
    if _probe_key not in st.session_state:
        _cap = cv2.VideoCapture(_source_str)
        _ok, _raw = _cap.read()
        _cap.release()
        if not _ok or _raw is None:
            st.error("Could not read first frame from the simulated video.")
            st.stop()
        st.session_state[_probe_key] = _raw
    _first_frame_raw = st.session_state[_probe_key]

_first_frame = resize_keep_aspect(_first_frame_raw, target_width=int(proc_width))
frame_h, frame_w = _first_frame.shape[:2]


# ----------------------------
# Polygon definition UI
# ----------------------------

st.subheader("1) Define runway zone")
st.write("Define the runway polygon (minimum 3 points). Then press Start.")

colA, colB = st.columns([1.2, 0.8], gap="large")

with colA:
    st.image(
        Image.fromarray(cv2.cvtColor(_first_frame, cv2.COLOR_BGR2RGB)),
        caption="First frame — enter runway points on the right",
        use_container_width=True,
    )

poly_points: List[Tuple[float, float]] = []

with colB:
    st.markdown("### Polygon points")
    st.caption(
        "Enter points manually or use the rectangle helper below."
    )

    use_rect = st.checkbox("Quick rectangle helper", value=False)
    if use_rect:
        x1 = st.slider("Rect x_min", 0, frame_w - 1, int(frame_w * 0.10))
        x2 = st.slider("Rect x_max", 1, frame_w, int(frame_w * 0.90))
        y1 = st.slider("Rect y_min", 0, frame_h - 1, int(frame_h * 0.35))
        y2 = st.slider("Rect y_max", 1, frame_h, int(frame_h * 0.90))
        if x2 <= x1:
            st.error("Rectangle invalid: x_max must be > x_min")
        elif y2 <= y1:
            st.error("Rectangle invalid: y_max must be > y_min")
        else:
            poly_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    manual = st.text_area(
        "Manual points (one per line: x,y)",
        value="",
        placeholder="Example:\n120.5, 340.2\n980.0, 360.0\n900.0, 620.0\n150.0, 600.0",
        help="Paste coordinate pairs, one per line.",
    )

    if manual.strip():
        parsed: List[Tuple[float, float]] = []
        for line in manual.strip().splitlines():
            line = line.strip().replace(";", ",")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 2:
                continue
            try:
                parsed.append((float(parts[0]), float(parts[1])))
            except Exception:
                continue
        if len(parsed) >= 3:
            poly_points = parsed

    if len(poly_points) >= 3:
        preview = _first_frame.copy()
        pts_arr = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)
        for (px, py) in poly_points:
            cv2.circle(preview, (int(px), int(py)), 4, (0, 255, 0), -1)
        st.image(
            Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)),
            caption="Runway polygon preview",
            use_container_width=True,
        )

    if len(poly_points) < 3:
        st.warning("Need at least 3 points.")
    else:
        st.success(f"{len(poly_points)} points captured.")

    st.dataframe(pd.DataFrame(poly_points, columns=["x", "y"]).round(1),
                 use_container_width=True, height=220)

    st.markdown("### YOLO classes (RO)")
    st.write(", ".join([en_to_ro_label.get(c, c) for c in prompt_classes_en]))
    st.markdown("### VLM categories")
    st.write(", ".join(CATEGORIES))


if not run_btn:
    st.stop()

if len(poly_points) < 3:
    st.error("Need at least 3 points to create a runway polygon.")
    st.stop()

if int(confirm_n) > int(window_m):
    st.error("Confirm N must be <= Window M (otherwise no incident can ever open).")
    st.stop()


# ----------------------------
# Build runtime config
# ----------------------------

_loop_flag = True
if _source_mode_key == "file_live":
    # Only file_live has configureable loop; live sources always loop by nature
    _loop_flag = loop_sim

runtime_cfg = RuntimeConfig(
    source=_source_str,
    source_mode=_source_mode_key,
    loop=_loop_flag,
    proc_width=int(proc_width),
    proc_fps=int(proc_fps),
    warmup_secs=float(warmup_secs),
    confirm_n=int(confirm_n),
    window_m=int(window_m),
    yolo_model=str(yolo_cfg_data["detection"]["model"]),
    yolo_conf=float(yolo_conf),
    yolo_iou=float(yolo_iou),
    yolo_tracker_yaml=str(yolo_cfg_data["detection"].get("tracker_yaml", "botsort.yaml")),
    yolo_classes_en=prompt_classes_en,
    horizon_frames=int(horizon_frames),
    prebuffer_secs=float(prebuffer_secs),
    postbuffer_secs=float(postbuffer_secs),
    enable_vlm=bool(enable_vlm),
    vlm_model=str(vlm_model),
    vlm_pretrained=str(vlm_pretrained),
    vlm_device=str(vlm_device),
    vlm_frames=int(vlm_frames),
    vlm_unknown_threshold=float(unknown_thr),
    vlm_shadow_threshold=float(shadow_thr),
    vlm_debris_detail_threshold=float(debris_detail_thr),
    output_dir="artifacts",
)


# ----------------------------
# Build RunwayShieldRuntime
# ----------------------------

try:
    runtime = RunwayShieldRuntime(runtime_cfg, poly_points=poly_points)
except (RuntimeError, ValueError) as _e:
    st.error(f"Failed to initialise detection runtime: {_e}")
    st.stop()

frame_count = runtime.frame_count  # 0 for live sources

# ----------------------------
# Processing UI placeholders
# ----------------------------

st.subheader("2) Processing")

_is_live = _source_mode_key in ("webcam", "rtsp", "file_live")

if _is_live:
    st.info(
        f"Live mode: **{source_mode}**. "
        "Processing runs until you stop the app or close the browser tab."
    )

video_ph = st.empty()
stats_ph = st.empty()
inc_ph = st.empty()

if not _is_live:
    progress = st.progress(0.0)
else:
    progress = None

alerts_count = 0
t0 = time.time()
_frame_idx = 0  # raw frame counter for progress (file mode only)

# ----------------------------
# Main processing loop
# ----------------------------

for result in runtime.run():
    # Track raw frame index for progress bar (file mode)
    _frame_idx += runtime.frame_skip

    # Draw overlay using stored last-frame state in runtime
    overlay = runtime.get_last_overlay(en_to_ro_label=en_to_ro_label)
    if overlay is not None:
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        video_ph.image(rgb, channels="RGB",
                       caption=f"Frame {result.frame_index} | FPS {result.effective_fps:.1f}")

    # Toast on new incidents
    for inc in result.new_incidents:
        alerts_count += 1
        st.toast(f"🚨 Incident: {inc.incident_id} [{inc.severity}]", icon="🚨")
        if inc.vlm_category and inc.vlm_category != "unknown":
            status_icon = "✅" if inc.status == "DISMISSED" else "🧠"
            st.toast(
                f"{status_icon} {inc.vlm_category} "
                f"({inc.vlm_confidence:.2f}) — {inc.status}",
                icon=status_icon,
            )

    inc_ph.dataframe(runtime.incidents_table(), use_container_width=True, height=300)

    elapsed = time.time() - t0
    fps_eff = result.effective_fps
    source_fps_label = f"{runtime.source_fps:.1f}" if not _is_live else "live"

    stats_ph.markdown(
        f"""
        **Stats**
        - Source: `{source_mode}` | Source FPS: `{source_fps_label}` | Effective FPS: `{fps_eff:.1f}`
        - Processed frames: `{result.frame_index}` | In-runway tracks: `{result.tracked_count}`
        - Incidents opened: `{alerts_count}` | Elapsed: `{elapsed:.0f}s`
        - Log: `artifacts/incidents.jsonl`
        """
    )

    if progress is not None and frame_count > 0:
        progress.progress(min(1.0, _frame_idx / frame_count))

runtime.shutdown()

if not _is_live:
    st.success(
        "Done. Check `artifacts/` for evidence clips + snapshots, "
        "and `artifacts/incidents.jsonl` for the audit log."
    )

