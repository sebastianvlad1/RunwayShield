from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from cv_pipeline import (
    BlobDetector,
    BlobDetectorConfig,
    BlobTracker,
    TrackerConfig,
    bbox_center,
    clamp_bbox,
    crop_roi,
    make_runway_mask,
    point_in_polygon,
    polygon_from_points,
    resize_keep_aspect,
    sort_polygon_points,
)
from incident import IncidentEngine, normalized_across_runway_distance, severity_for
from recorder import RecorderConfig, RecorderManager, ensure_artifacts_dir
from vlm_clip import ClipClassifier, ClipConfig, CATEGORIES


# ----------------------------
# Utils
# ----------------------------


def save_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_points(canvas_json: dict) -> List[Tuple[float, float]]:
    if not canvas_json:
        return []
    objs = canvas_json.get("objects", [])
    pts: List[Tuple[float, float]] = []
    for o in objs:
        left = float(o.get("left", 0.0))
        top = float(o.get("top", 0.0))
        r = float(o.get("radius", 4.0))
        pts.append((left + r, top + r))
    return pts


@st.cache_resource(show_spinner=False)
def get_clip_classifier(cfg: ClipConfig) -> ClipClassifier:
    return ClipClassifier(cfg)


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="RunwayShield PoC", layout="wide")

st.title("RunwayShield — Automatic Runway Hazard Detection")
st.caption("OpenCV change detection → N-of-M incident gating → evidence → fast VLM classification (OpenCLIP)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload runway video", type=["mp4", "mov", "mkv", "avi"])

    st.header("Performance")
    proc_width = st.select_slider("Processing width (px)", options=[640, 800, 960, 1120, 1280], value=960)
    proc_fps = st.select_slider("Process FPS (approx)", options=[5, 10, 15, 20, 30], value=15)

    st.header("Incident gating")
    window_m = st.number_input("Window M (frames)", min_value=5, max_value=120, value=10, step=1)
    confirm_n = st.number_input("Confirm N (in-runway frames)", min_value=2, max_value=60, value=6, step=1)

    st.header("OpenCV detection")
    min_area = st.number_input("Min blob area (px^2)", min_value=20, max_value=20000, value=250, step=10)
    max_area = st.number_input("Max blob area (px^2)", min_value=100, max_value=400000, value=45000, step=500)
    warmup_secs = st.number_input("Warm-up seconds", min_value=0, max_value=30, value=3, step=1)

    with st.expander("Advanced detection", expanded=False):
        bg_history = st.slider("MOG2 history", min_value=100, max_value=1500, value=400, step=50)
        var_threshold = st.slider("MOG2 varThreshold", min_value=4, max_value=64, value=16, step=2)
        detect_shadows = st.checkbox("MOG2 detectShadows", value=True)
        learning_rate = st.slider("Learning rate", 0.0, 0.05, 0.005, 0.001)
        diff_threshold = st.slider("Median absdiff threshold", min_value=5, max_value=60, value=22, step=1)
        morph_kernel = st.select_slider("Morph kernel", options=[3, 5, 7], value=3)
        morph_iters = st.slider("Morph iterations", min_value=1, max_value=5, value=2, step=1)
        max_fg_ratio = st.slider("Max foreground ratio (shake guard)", 0.05, 0.60, 0.25, 0.05)
        min_extent = st.slider("Min extent", 0.05, 0.50, 0.15, 0.01)
        min_solidity = st.slider("Min solidity", 0.05, 0.80, 0.30, 0.01)

    st.header("Tracking")
    tracker_iou = st.slider("Tracker IoU", 0.05, 0.70, 0.25, 0.05)
    tracker_max_missed = st.slider("Tracker max missed", 1, 60, 10, 1)

    st.header("Evidence")
    prebuffer_secs = st.number_input("Pre-buffer seconds", min_value=1, max_value=30, value=5, step=1)
    postbuffer_secs = st.number_input("Post-buffer seconds", min_value=1, max_value=30, value=5, step=1)

    st.header("VLM classification (fast + stable)")
    enable_vlm = st.checkbox("Enable VLM classify", value=True)

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
        # pretrained tags depend on model; keep one stable default per choice.
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
    run_btn = st.button("▶ Start", type="primary")


if uploaded is None:
    st.info("Upload a runway video to begin.")
    st.stop()

# Save upload to artifacts/tmp preserving extension
art_tmp = Path("artifacts/tmp")
art_tmp.mkdir(parents=True, exist_ok=True)

suffix = Path(uploaded.name).suffix.lower()
if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
    suffix = ".mp4"

video_path = art_tmp / f"upload_{int(time.time()*1000)}{suffix}"
with open(video_path, "wb") as f:
    f.write(uploaded.getbuffer())

cap = cv2.VideoCapture(str(video_path))
ok, first = cap.read()
if not ok:
    st.error("Could not read the video file.")
    st.stop()

first = resize_keep_aspect(first, target_width=int(proc_width))
frame_h, frame_w = first.shape[:2]

st.subheader("1) Define runway zone")
st.write("Define the runway polygon (minimum 3 points). Then press Start.")

colA, colB = st.columns([1.2, 0.8], gap="large")

with colA:
    st.image(
        Image.fromarray(cv2.cvtColor(first, cv2.COLOR_BGR2RGB)),
        caption="First frame (enter runway points on the right)",
        use_container_width=True,
    )

poly_points: List[Tuple[float, float]] = []

with colB:
    st.markdown("### Polygon points")
    st.caption(
        "This build avoids the canvas dependency (which often breaks across Streamlit versions). "
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
        help="If canvas doesn't work, paste points here.",
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
        preview = first.copy()
        pts = np.array(poly_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for (x, y) in poly_points:
            cv2.circle(preview, (int(x), int(y)), 4, (0, 255, 0), -1)
        st.image(
            Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)),
            caption="Runway polygon preview",
            use_container_width=True,
        )

    if len(poly_points) < 3:
        st.warning("Need at least 3 points.")
    else:
        st.success(f"{len(poly_points)} points captured.")

    st.dataframe(pd.DataFrame(poly_points, columns=["x", "y"]).round(1), use_container_width=True, height=220)

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

# Stabilize polygon ordering
poly_points = sort_polygon_points(poly_points)
poly = polygon_from_points(poly_points)
runway_mask = make_runway_mask(first.shape, poly)

# Reset capture
cap.release()
cap = cv2.VideoCapture(str(video_path))

src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or math.isnan(src_fps) or src_fps <= 0:
    src_fps = 30.0

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
frame_skip = max(1, int(round(src_fps / float(proc_fps))))

effective_fps = int(proc_fps)

# Components
engine = IncidentEngine(confirm_n=int(confirm_n), window_m=int(window_m))

det_cfg = BlobDetectorConfig(
    min_area=int(min_area),
    max_area=int(max_area),
    history=int(bg_history),
    var_threshold=int(var_threshold),
    detect_shadows=bool(detect_shadows),
    learning_rate=float(learning_rate),
    diff_threshold=int(diff_threshold),
    morph_kernel=int(morph_kernel),
    morph_iters=int(morph_iters),
    max_fg_ratio=float(max_fg_ratio),
    min_extent=float(min_extent),
    min_solidity=float(min_solidity),
)

detector = BlobDetector(det_cfg)
tracker = BlobTracker(TrackerConfig(iou_threshold=float(tracker_iou), max_missed=int(tracker_max_missed)))

prebuffer_frames = max(1, int(float(prebuffer_secs) * effective_fps))
postbuffer_frames = max(1, int(float(postbuffer_secs) * effective_fps))

rec = RecorderManager(
    RecorderConfig(
        fps=effective_fps,
        frame_size=(frame_w, frame_h),
        prebuffer_frames=prebuffer_frames,
        postbuffer_frames=postbuffer_frames,
    )
)

warmup_frames = max(0, int(float(warmup_secs) * effective_fps))

clip_cfg = ClipConfig(
    enabled=bool(enable_vlm),
    model_name=str(vlm_model),
    pretrained=str(vlm_pretrained),
    device=str(vlm_device),
    unknown_threshold=float(unknown_thr),
    shadow_artifact_threshold=float(shadow_thr),
    debris_detail_threshold=float(debris_detail_thr),
)

clip_classifier = get_clip_classifier(clip_cfg) if enable_vlm else None

# Logging
art_dir = ensure_artifacts_dir()
log_path = art_dir / "incidents.jsonl"
save_jsonl(log_path, {"event": "RUN_START", "video": str(video_path), "ts": time.time()})

# UI placeholders
st.subheader("2) Processing")
video_ph = st.empty()
stats_ph = st.empty()
inc_ph = st.empty()
progress = st.progress(0.0)

processed = 0
frame_idx = 0
alerts_count = 0

# warmup uses processed frame counter

t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    frame = resize_keep_aspect(frame, target_width=int(proc_width))
    if frame.shape[0] != frame_h or frame.shape[1] != frame_w:
        frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

    rec.push_frame(frame)

    # Warmup
    if processed < warmup_frames:
        detector.warmup(frame, runway_mask)
        boxes_with_conf, fgmask = ([], np.zeros((frame_h, frame_w), dtype=np.uint8))
    else:
        boxes_with_conf, fgmask = detector.detect(frame, runway_mask)

    boxes_xyxy = [(b[0], b[1], b[2], b[3]) for b in boxes_with_conf]
    conf_map = {(b[0], b[1], b[2], b[3]): float(b[4]) for b in boxes_with_conf}

    tracked = tracker.update(boxes_xyxy)

    seen_tids: List[int] = []
    for box, tid in tracked:
        x1, y1, x2, y2 = clamp_bbox(box, frame_w, frame_h)
        cx, cy = bbox_center((x1, y1, x2, y2))
        if not point_in_polygon(poly, cx, cy):
            continue
        seen_tids.append(int(tid))
        conf = conf_map.get((x1, y1, x2, y2), 0.6)
        engine.on_detection(int(tid), True, (x1, y1, x2, y2), float(conf))

    engine.update_tracks(seen_tids)

    new_incidents = []
    if processed >= warmup_frames:
        new_incidents = engine.maybe_open_incidents(
            frame_idx=processed,
            poly_points=poly_points,
            bbox_center_fn=bbox_center,
            recorder=rec,
            frame_bgr=frame,
        )

    # VLM classify immediately (fast) on incident open
    for inc in new_incidents:
        alerts_count += 1
        st.toast(f"🚨 Incident opened: {inc.incident_id}", icon="🚨")
        save_jsonl(log_path, {"event": "INCIDENT_OPEN", "ts": time.time(), "incident_id": inc.incident_id})

        if clip_classifier is not None:
            # sample 1..vlm_frames from prebuffer + current
            rois: List[np.ndarray] = []
            prebuf = list(rec.prebuffer)
            k = int(vlm_frames)
            if k > 1 and len(prebuf) > 1:
                idxs = np.linspace(0, len(prebuf) - 1, num=k - 1).round().astype(int).tolist()
                for idx in idxs:
                    rois.append(crop_roi(prebuf[idx], inc.bbox_xyxy, pad=0.20))
            rois.append(crop_roi(frame, inc.bbox_xyxy, pad=0.20))

            res = clip_classifier.classify_rois(rois)
            inc.vlm_processed_ts = time.time()
            inc.vlm_category = res.category
            inc.vlm_detail = res.detail
            inc.vlm_real = bool(res.real)
            inc.vlm_confidence = float(res.confidence)
            inc.vlm_error = res.error

            # refine severity
            cx, cy = bbox_center(inc.bbox_xyxy)
            dist = normalized_across_runway_distance((cx, cy), poly_points)
            inc.severity = severity_for(res.category, in_centerline=(dist < 0.25))

            if res.ok and (res.real is False):
                inc.status = "DISMISSED"
                st.toast(f"✅ Dismissed as artefact: {res.category} ({res.confidence:.2f})", icon="✅")
                save_jsonl(log_path, {"event": "INCIDENT_DISMISSED", "ts": time.time(), "incident_id": inc.incident_id, "category": res.category, "conf": res.confidence})
            else:
                label = res.detail or res.category
                st.toast(f"🧠 Classified: {label} ({res.confidence:.2f}) on {clip_classifier.device}", icon="🧠")
                save_jsonl(log_path, {"event": "INCIDENT_CLASSIFIED", "ts": time.time(), "incident_id": inc.incident_id, "category": res.category, "detail": res.detail, "conf": res.confidence, "real": res.real})

    # Overlay
    overlay = frame.copy()
    cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    for box, tid in tracked:
        x1, y1, x2, y2 = clamp_bbox(box, frame_w, frame_h)
        cx, cy = bbox_center((x1, y1, x2, y2))
        if not point_in_polygon(poly, cx, cy):
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"blob#{tid}",
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    video_ph.image(rgb, channels="RGB", caption=f"Processed frame {processed}")

    inc_ph.dataframe(engine.to_table(), use_container_width=True, height=300)

    processed += 1
    elapsed = time.time() - t0
    fps_eff = processed / max(1e-6, elapsed)

    stats_ph.markdown(
        f"""
        **Stats**  
        - Source FPS: `{src_fps:.1f}` | Target FPS: `{proc_fps}` | Effective FPS: `{fps_eff:.1f}`  
        - Frame skip: `{frame_skip}` | Processed frames: `{processed}`  
        - Incidents opened: `{alerts_count}`  
        - Log: `{log_path}`
        """
    )

    if frame_count > 0:
        progress.progress(min(1.0, frame_idx / frame_count))

    frame_idx += 1

cap.release()
save_jsonl(log_path, {"event": "RUN_END", "ts": time.time(), "processed_frames": processed, "incidents": alerts_count})

st.success("Done. Check `artifacts/` for evidence clips + snapshots, and `artifacts/incidents.jsonl` for the audit log.")
