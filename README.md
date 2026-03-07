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

## Source modes

RunwayShield supports four video source modes, available in both the Streamlit UI and the headless CLI:

| Mode | Description |
|---|---|
| `file` | Process an offline video file (default) |
| `file_live` | Replay a file at real-time pace, optionally looping infinitely |
| `webcam` | Read from a locally connected camera by device index |
| `rtsp` | Read from an RTSP/HTTP network stream |

## Headless / CLI usage

Use `headless_runner.py` to integrate RunwayShield into another application without any Streamlit dependency.
The runner emits one **JSON line per incident** to stdout, which is easy to pipe to downstream consumers.

### Polygon format

Create a `polygon.json` file (or `polygon.yaml`):
```json
{
  "points": [
    [120, 340],
    [980, 360],
    [900, 620],
    [150, 600]
  ]
}
```

### Examples

**Offline video:**
```bash
python headless_runner.py --source video.mp4 --polygon polygon.json
```

**Simulate live (loop file at real-time pace):**
```bash
python headless_runner.py --source video.mp4 --mode file_live --polygon polygon.json --loop
```

**Webcam:**
```bash
python headless_runner.py --source 0 --mode webcam --polygon polygon.json
```

**RTSP stream with auto-reconnect:**
```bash
python headless_runner.py --source rtsp://user:pass@192.168.1.10/stream --mode rtsp \
    --polygon polygon.json --reconnect-attempts 10 --reconnect-delay 2.0
```

**With VLM classification:**
```bash
python headless_runner.py --source video.mp4 --polygon polygon.json \
    --enable-vlm --vlm-model ViT-B-32 --vlm-device cpu
```

### Output format (stdout)

Each incident line is a JSON object, for example:
```json
{"event": "INCIDENT_OPEN", "ts": 1710000000.0, "incident_id": "INC-0001", "track_id": 3, "severity": "HIGH", "bbox": [120, 340, 200, 420], "detector_label": "person walking"}
```

All incidents are also written to `artifacts/incidents.jsonl`.

### Full CLI options

```
python headless_runner.py --help
```

Key flags:
- `--source` — path / URL / camera index
- `--mode` — `file` | `file_live` | `webcam` | `rtsp`
- `--polygon` — path to JSON or YAML polygon file
- `--proc-width` — processing width in pixels (default 960)
- `--proc-fps` — target processing FPS (default 15)
- `--confirm-n` / `--window-m` — N-of-M gating (default 6/10)
- `--loop` — loop file infinitely in `file_live` mode
- `--enable-vlm` — enable CLIP classification
- `--reconnect-attempts` / `--reconnect-delay` — live stream reconnect settings

