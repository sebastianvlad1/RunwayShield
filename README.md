# RunwayShield PoC (GroundingDINO + VLM classify)

This PoC detects hazards inside a **runway polygon** using **GroundingDINO + IoU tracking**, opens incidents via an **N-of-M persistence gate**, stores **evidence (snapshot + clip)**, and (optionally) classifies the incident ROI with a **fast vision-language model (CLIP via open_clip)**.

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

## GPU/CUDA setup (important)

By default, `pip install torch` may install a CPU build depending on your environment.
If you want NVIDIA GPU acceleration, install the CUDA build of PyTorch explicitly.

### Windows + NVIDIA (recommended)
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# Install CUDA-enabled PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install project deps
python -m pip install -r requirements.txt

# Verify
python preflight.py
```

Expected preflight output on a GPU machine:
```text
[preflight] CUDA available: <GPU name> (<VRAM> MB VRAM)
```

### Quick runtime check
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### If Task Manager shows 0% GPU

This app uses CUDA/compute engines, not 3D rendering.
In Windows Task Manager, set the GPU graph to `Cuda` or `Compute_0`/`Compute_1`
instead of `3D`, or check usage with `nvidia-smi`.

### Expected processing speed

GroundingDINO (the zero-shot detection backbone) is computationally heavy.
Typical throughput on consumer GPUs:

| GPU | FPS (GroundingDINO-tiny) |
|-----|--------------------------|
| GTX 1650 Ti (4 GB) | ~0.4 FPS |
| RTX 3060 / 3070 | ~1–2 FPS |
| RTX 4080 / A100 | ~4–8 FPS |

This is expected. The app uses frame-skip gating so not every frame is analysed —
configure `frame_skip` in `config.yaml` to trade coverage for UI responsiveness.

> **Note:** `torch.compile()` acceleration is not available on Windows (requires Triton,
> which is Linux-only). On Linux the speed roughly doubles.

### Optional: HF token to avoid download/rate-limit warnings

Set `HF_TOKEN` to reduce Hugging Face Hub rate-limit warnings and improve model download reliability.

PowerShell example:
```powershell
$env:HF_TOKEN = "<your_token_here>"
```

## How it works (high level)
1. Upload a runway video.
2. Define runway area by clicking points (polygon).
3. For each processed frame:
  - GroundingDINO detects objects from configured text classes.
  - IoU tracker provides stable `track_id` values.
  - Kalman trajectory prediction (`horizon_frames`) marks tracks that are already in runway or predicted to enter runway.
  - N-of-M gating opens an incident only if the track persists.
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
{"incident_id": "INC-1710000000000", "status": "OPEN", "severity": "MED", "track_id": 3, "detector_label": "person walking", "bbox": [120, 340, 200, 420], "vlm_category": "person", "vlm_detail": null, "vlm_confidence": 0.91, "vlm_real": true, "evidence_image": "artifacts/INC-1710000000000_snapshot.jpg", "evidence_video": "artifacts/INC-1710000000000_evidence.mp4"}
```

All incidents are also written to `artifacts/incidents.jsonl`.

### Full CLI options

```
python headless_runner.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--source` | **required** | Video path, webcam index (`0`), or RTSP URL |
| `--polygon` | **required** | Path to polygon JSON or YAML file (min 3 points) |
| `--config` | `config.yaml` | Path to project YAML detection config (YOLO model/classes/threshold defaults) |
| `--mode` | `file` | `file` \| `file_live` \| `webcam` \| `rtsp` |
| `--loop` | `False` | Loop file infinitely at EOF (only for `file_live`) |
| `--output-dir` | `artifacts` | Directory for evidence clips, snapshots, and JSONL log |
| `--verbose` | `False` | Enable DEBUG logging |
| **Performance** | | |
| `--proc-width` | `640` | Processing frame width in pixels |
| `--proc-fps` | `30` | Target processing FPS (also controls `file_live` pacing) |
| `--infer-width` | `640` | Resolution sent to GroundingDINO (separate from display) |
| `--dino-every` | `1` | Run GroundingDINO once every N frames; Kalman-predict in between |
| `--crop-pad` | `0.20` | Padding factor around runway bounding-rect for inference crop |
| **Incident gating** | | |
| `--confirm-n` | `3` | DINO-confirmed frames a track must be in-runway to open an incident |
| `--window-m` | `5` | Sliding window size for N-of-M gating (DINO frames only) |
| `--warmup-secs` | `5.0` | Seconds to skip detection at startup (background stabilisation) |
| **YOLO detection** | | |
| `--yolo-conf` | `0.35` | YOLO confidence threshold (overrides `config.yaml`) |
| `--yolo-iou` | `0.60` | YOLO IoU threshold (overrides `config.yaml`) |
| `--horizon-frames` | `6` | Kalman trajectory prediction horizon (frames ahead) |
| **Evidence buffers** | | |
| `--prebuffer-secs` | `3.0` | Seconds of pre-incident video included in evidence clip |
| `--postbuffer-secs` | `5.0` | Seconds of post-incident video included in evidence clip |
| **VLM classification** | | |
| `--enable-vlm` | `False` | Enable OpenCLIP classification of incident ROI |
| `--vlm-model` | `ViT-B-32` | CLIP model (`ViT-B-32` or `ViT-L-14`) |
| `--vlm-pretrained` | `laion2b_s34b_b79k` | Pretrained weights tag |
| `--vlm-device` | `auto` | `auto` \| `cpu` \| `cuda` \| `mps` |
| `--vlm-frames` | `3` | Number of ROI frames for temporal averaging |
| `--vlm-unknown-threshold` | `0.28` | Below this confidence → category becomes `unknown` |
| `--vlm-shadow-threshold` | `0.55` | Shadow confidence above this → `real=false`, incident dismissed |
| `--vlm-debris-detail-threshold` | `0.45` | Min debris confidence to run sub-classification |
| **Live source / reconnect** | | |
| `--reconnect-attempts` | `10` | Max reconnect attempts for webcam/RTSP on failure |
| `--reconnect-delay` | `1.0` | Base delay in seconds between reconnect attempts (doubles each time) |

