# RunwayShield PoC — Technical Briefing pentru LLM

## 🎯 Ce Este Acest Proiect

**RunwayShield** este un sistem de detecție automată a hazardurilor pe pistele de aeroporturi. Detectează obiecte noi/mobile într-o zonă de runway definită de utilizator și clasifică incidentele folosind computer vision + vision-language models.

**Scop**: PoC (proof-of-concept) pentru procesare video offline cu focus pe stabilitate cross-platform (Windows/Mac/Linux) și funcționare pe CPU.

---

## 📦 Stack Tehnic

```python
# Core dependencies
opencv-python>=4.8      # Computer vision pipeline
torch>=2.1              # CLIP model backend
open_clip_torch>=2.24   # Zero-shot classification
streamlit==1.55.0       # UI framework
numpy>=1.24             # Numerical operations
pandas>=2.0             # Data display
pillow>=10.0            # Image handling
```

**Entry point**: `python -m streamlit run app.py`

---

## 🏗️ Arhitectură (5 Module Principale)

### 1. **app.py** — Orchestrator & UI
- Streamlit application (single-page)
- User uploads video → defines runway polygon → runs detection
- Integrează toate componentele în main processing loop
- Frame skip logic pentru target FPS
- Event logging în `artifacts/incidents.jsonl`

### 2. **cv_pipeline.py** — Computer Vision Core
**Exports**:
- `BlobDetector(config)` — Detectează obiecte noi folosind:
  - MOG2 background subtractor
  - Median background diff (pentru obiecte statice noi)
  - Morphological operations (noise removal)
  - Shape filtering (extent, solidity)
  - Camera shake guard (max foreground ratio)
  
- `BlobTracker(config)` — IoU tracker simplu pentru ID-uri stabile
- Helper functions: `make_runway_mask()`, `point_in_polygon()`, `bbox_center()`, `crop_roi()`

**Workflow**:
```python
detector.warmup(frame, runway_mask)  # 3-5 secunde stabilizare
boxes, fgmask = detector.detect(frame, runway_mask)
tracked = tracker.update(boxes)  # [(bbox, track_id), ...]
```

### 3. **incident.py** — Business Logic & Gating
**Core class**: `IncidentEngine(confirm_n, window_m)`

**N-of-M Gating Logic**:
- Un blob devine incident DOAR dacă apare în runway în **N din ultimele M frames**
- Previne false positives de la noise temporar
- State tracking: `track_hist[track_id]` = deque of bool (in_runway)

**Severity calculation**:
- `severity_for(category, in_centerline)` → HIGH/MED/LOW
- HIGH: person, vehicle, animal (oricând); bird/debris în centerline
- MED: bird/debris pe margine; unknown
- LOW: shadow

**Dataclass**: `Incident` cu fields pentru bbox, frames, VLM results, evidence paths

### 4. **recorder.py** — Evidence Recording
**Components**:
- `RecorderManager(config)` cu prebuffer/postbuffer circular buffers
- `save_snapshot(incident_id, frame, bbox)` → crops & saves JPG
- `start_recording(incident_id)` → writes MP4 cu prebuffer + live + postbuffer

**Codec fallback strategy**:
```python
# Încearcă în ordine: mp4v, MJPG, XVID
# Dacă toate eșuează → None (no crash, doar lipsește clip)
```

### 5. **vlm_clip.py** — Fast Classification
**Class**: `ClipClassifier(config)`

**Metodă core**:
```python
result = classifier.classify_rois(roi_frames: List[np.ndarray])
# → ClipResult(category, confidence, real, detail)
```

**Categories**: `person | vehicle | bird | animal | debris | shadow | unknown`

**Tehnică**:
- OpenCLIP ViT-B-32 (400MB, download o dată)
- Text prompts pentru fiecare categorie (multiple variants)
- Image encoding → cosine similarity cu text embeddings
- Temporal averaging peste 1-3 ROI frames pentru stabilitate
- Shadow thresholding pentru artefact detection (`real=False`)
- Debris detail classification (tire/cone/bag/suitcase/...)

---

## 🔄 Fluxul de Execuție (Main Loop)

```python
# 1. Setup (app.py)
cap = cv2.VideoCapture(video_path)
runway_mask = make_runway_mask(frame_shape, polygon)
detector = BlobDetector(config)
tracker = BlobTracker(config)
engine = IncidentEngine(confirm_n=6, window_m=10)
recorder = RecorderManager(config)
clip_classifier = ClipClassifier(config)

# 2. Processing loop
while True:
    ok, frame = cap.read()
    frame = resize_keep_aspect(frame, target_width)
    
    recorder.push_frame(frame)  # Circular buffer
    
    # Warmup first N seconds
    if processed < warmup_frames:
        detector.warmup(frame, runway_mask)
        continue
    
    # Detection
    boxes, fgmask = detector.detect(frame, runway_mask)
    tracked = tracker.update(boxes)
    
    # Update incident engine
    for bbox, track_id in tracked:
        cx, cy = bbox_center(bbox)
        in_runway = point_in_polygon(polygon, cx, cy)
        engine.on_detection(track_id, in_runway, bbox, conf)
    
    engine.update_tracks(seen_track_ids)
    
    # Check for new incidents (N-of-M gate)
    new_incidents = engine.maybe_open_incidents(
        frame_idx, polygon, recorder, frame
    )
    
    # Classify immediately (fast, <100ms on CPU)
    for inc in new_incidents:
        rois = [crop_roi(f, inc.bbox) for f in prebuffer_sample]
        result = clip_classifier.classify_rois(rois)
        
        inc.vlm_category = result.category
        inc.vlm_confidence = result.confidence
        inc.vlm_real = result.real
        
        if not result.real:
            inc.status = "DISMISSED"  # Shadow/artefact
        
        # Save evidence
        recorder.save_snapshot(inc.incident_id, frame, bbox)
        recorder.start_recording(inc.incident_id)
        
        # Log to JSONL
        save_jsonl(log_path, {"event": "INCIDENT_OPEN", ...})
```

---

## 📋 Convenții de Cod

### Type Hints Stricte
```python
from __future__ import annotations
def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
```

### Dataclasses pentru Config
```python
@dataclass
class BlobDetectorConfig:
    min_area: int = 250
    max_area: int = 45000
    # ...
```

### Defensive Programming
- Toate operațiile fragile au fallbacks
- `clamp_bbox()` pentru orice bbox înainte de crop
- Video writer failures → continue cu warning
- CLIP failures → category="unknown", continue

### No External State
- No database (JSONL append-only pentru audit)
- No config files (toate în Streamlit sidebar)
- In-memory state în dictionaries/deques

---

## 🗂️ Structura de Fișiere

```
runwayshield_poc/
├── app.py              # Entry point + UI + main loop
├── cv_pipeline.py      # BlobDetector + BlobTracker + geometry utils
├── incident.py         # IncidentEngine + N-of-M gating + severity
├── vlm_clip.py         # ClipClassifier + prompts + device selection
├── recorder.py         # RecorderManager + evidence saving
├── preflight.py        # Smoke tests (imports + CLIP registry)
├── requirements.txt    # Dependencies
├── README.md           # User documentation
└── artifacts/          # Output directory
    ├── incidents.jsonl     # Event log
    ├── INC-*_snapshot.jpg  # Incident crops
    └── INC-*_evidence.mp4  # Pre+post buffer clips
```

---

## 🔍 Cum Să Navighezi Codul

### Pentru debugging detection issues:
1. Check `cv_pipeline.py` → `BlobDetector.detect()` line 172
2. Parametri critici: `min_area`, `max_fg_ratio`, `min_extent`, `min_solidity`
3. Visualizations: `fgmask` returnat de `detect()`

### Pentru incident gating logic:
1. Check `incident.py` → `IncidentEngine.maybe_open_incidents()` line 113
2. Core: `if sum(hist) >= self.confirm_n`
3. Hist update: `on_detection()` + `update_tracks()`

### Pentru VLM classification:
1. Check `vlm_clip.py` → `ClipClassifier.classify_rois()` line 171
2. Prompts: `_CATEGORY_PROMPTS` dictionary line 45
3. Temporal averaging: line 196 (`img_emb.mean(dim=0)`)

### Pentru evidence paths:
1. Check `recorder.py` → `RecorderManager.start_recording()` line 106
2. Codec fallback: `create_best_effort_writer()` line 35

---

## ⚙️ Key Parameters (Tuning)

| Parameter | Location | Default | Impact |
|-----------|----------|---------|--------|
| `min_area` | BlobDetectorConfig | 250 | Min blob size (px²) to detect |
| `max_fg_ratio` | BlobDetectorConfig | 0.25 | Camera shake guard threshold |
| `confirm_n` | IncidentEngine | 6 | Persistence frames required |
| `window_m` | IncidentEngine | 10 | Sliding window size |
| `unknown_threshold` | ClipConfig | 0.28 | Min confidence pentru category |
| `shadow_threshold` | ClipConfig | 0.55 | Threshold pentru artefact dismissal |

---

## 🚨 Gotchas & Limitări

1. **Polygon definition**: Manual entry sau rect helper (no visual canvas)
2. **Video codecs**: Poate eșua pe Windows exotic → evidence clips missing
3. **Warmup requirement**: Primele `warmup_secs` NU detectează (background learning)
4. **CPU-only safe**: CUDA/MPS funcționează dar nu e garantat pe toate sistemele
5. **No streaming**: Only offline video files
6. **No persistence**: Restart = all state lost (doar JSONL log rămâne)

---

## 💡 Când Să Modifici Ce

**Vrei mai multe/mai puține detecții?**
→ Adjustează `min_area`, `max_area`, `min_extent`, `min_solidity` în `app.py` sidebar

**Prea multe false positives?**
→ Crește `confirm_n` (mai multe frames de persistență) în `app.py` line 86

**Classification slabă?**
→ Modifică `_CATEGORY_PROMPTS` în `vlm_clip.py` lines 45-80

**More evidence context?**
→ Crește `prebuffer_secs` / `postbuffer_secs` în `app.py` lines 100-101

---

## 📊 Diagrame de Arhitectură

### Pipeline Flow
```
Video Upload
    ↓
Frame Extraction (with skip for FPS target)
    ↓
Runway Mask Application
    ↓
Background Subtraction (MOG2 + Median Diff)
    ↓
Blob Detection (Contours + Filtering)
    ↓
IoU Tracking (Stable IDs)
    ↓
N-of-M Gating (Persistence Check)
    ↓
Incident Creation
    ↓
CLIP Classification (Zero-shot)
    ↓
Evidence Recording (Snapshot + Clip)
    ↓
JSONL Logging
```

### Component Dependencies
```
app.py
  ├── cv_pipeline.py
  │     ├── BlobDetector (MOG2)
  │     └── BlobTracker (IoU)
  ├── incident.py
  │     └── IncidentEngine (N-of-M)
  ├── recorder.py
  │     └── RecorderManager (Buffers)
  └── vlm_clip.py
        └── ClipClassifier (OpenCLIP)
```

---

## 🧪 Testing & Validation

### Preflight Checks
```bash
python preflight.py
```
Verifică:
- Python syntax pentru toate modulele
- Imports disponibile
- OpenCLIP model registry functional

### Manual Testing Flow
1. Run: `python -m streamlit run app.py`
2. Upload test video cu runway vizibil
3. Define polygon (minim 3 puncte)
4. Set parameters în sidebar
5. Press "Start" → verifică:
   - Detecții vizibile în overlay
   - Incidents table se populează
   - Evidence files în `artifacts/`
   - JSONL log entries

---

## 🔧 Troubleshooting Common Issues

### Nu detectează nimic
- Check warmup period terminat (primele N secunde sunt skip)
- Verifică polygon coverage (runway mask)
- Lower `min_area` sau relax `min_extent`/`min_solidity`
- Check `max_fg_ratio` (camera shake poate bloca detecția)

### Prea multe false positives
- Increase `confirm_n` (mai strictă persistență)
- Increase `min_area` (ignore noise mic)
- Adjust `min_extent` / `min_solidity` (filtrare shape)
- Enable VLM classification pentru auto-dismiss shadows

### CLIP classification lentă
- Reduce `vlm_frames` (1 în loc de 3)
- Device stuck pe CPU → check torch.cuda.is_available()
- Model download încă în progres → wait for completion

### Video writer eșuează
- Check codec availability: `cv2.VideoWriter_fourcc(*'mp4v')`
- Fallback la MJPG/XVID dacă mp4v unavailable
- Evidence clips lipsesc DAR snapshots funcționează

---

## 📚 Resurse & Referințe

### OpenCV Background Subtraction
- MOG2: Gaussian Mixture-based Background/Foreground Segmentation
- History buffer: adaptare graduală la schimbări de lighting
- Shadow detection: pixel value 127 în mask

### OpenCLIP
- Zero-shot image classification via text prompts
- CLIP: Contrastive Language-Image Pre-training
- ViT-B-32: Vision Transformer Base, patch size 32
- LAION-2B: training dataset (2 billion image-text pairs)

### Streamlit Caching
- `@st.cache_resource`: pentru modele ML (persist across reruns)
- Session state minimal (doar UI widgets)

---

**Acest codebase este simplu, bine modularizat și transparent. Toate dependencies sunt explicit typed. Începe de la `app.py` main loop și urmărește flow-ul descris mai sus.**
