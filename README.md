# RunwayShield PoC (CV trigger + VLM classify)

This PoC detects **new / moving objects** inside a **runway polygon** using OpenCV change detection, opens incidents via an **N-of-M persistence gate**, saves **evidence (snapshot + clip)**, and (optionally) classifies the incident ROI using a **fast vision-language model (CLIP via open_clip)**.

Why CLIP (open_clip) instead of a large generative VLM (e.g. Qwen2.5-VL)?
- It is **much faster and more stable** on CPU/Mac/Windows.
- It is deterministic, small, and ideal for **category classification** (person / vehicle / bird / animal / debris / shadow / unknown).

## Quickstart

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python preflight.py
python -m streamlit run app.py
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python preflight.py
python -m streamlit run app.py
```

## How it works (high level)
1. Upload a runway video.
2. Define runway area by clicking points (polygon).
3. For each processed frame:
   - OpenCV change detection (MOG2 + median background diff) finds candidate blobs **inside the runway mask**.
   - Filters remove noise (area/shape, camera-shake guard).
   - IoU tracking gives stable `track_id`.
   - N-of-M gating opens an incident only if the blob persists.
4. Optional: VLM classification runs on 1-3 ROI crops around the blob and outputs:
   - `category` in {person, vehicle, bird, animal, debris, shadow, unknown}
   - `real` (true/false)
   - `confidence` (0..1)
   - (optional) debris detail (tire / cone / bag / ...)

Artifacts are written to `artifacts/`.
