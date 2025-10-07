## Touchless Lock System (TLS)

Computer‑vision palm verification pipeline for a touchless door lock. Streams video, detects palms with MediaPipe, extracts robust LBP+geometry features, and verifies against registered users. Now includes a modern FastAPI web interface for palm snapshot collection.

### Highlights
- **FastAPI Web Server**: Modern web interface for palm detection and collection
- Real‑time palm detection with MediaPipe Hands
- Tiled‑LBP features plus optional palm‑geometry descriptors
- Registration and verification flows with similarity matching
- Snapshot collection via HTTP endpoints
- Modular, testable design

---

## Quick Start

### Prerequisites
- Linux recommended
- Python 3.11 or 3.12 strongly recommended (best wheels support)
- Webcam supported by OpenCV

If you must use Python 3.13, you may need system build headers and newer wheels.

### Setup
```bash
cd TLS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Run

**Web Server Mode (Recommended):**

*Local Mode (localhost only):*
```bash
python launcher.py web
# or
python web_server.py
```
Then open your browser to: http://localhost:8000

*Network Mode (accessible from other devices):*
```bash
python launcher.py web --network
# Auto-detects your local IP and makes server accessible from other devices
# Access from other devices: http://YOUR_AUTO_DETECTED_IP:8000
```

```

*Windows Easy Start:*
```bash
start_network_server.bat
```

**Desktop Mode (Legacy):**
```bash
python launcher.py desktop
# or
python main.py
```
Keys:
- q: Quit
- r: Start registration flow
- v: Start verification flow

---

## Network Configuration

### Quick Network Setup

**Option 1: Automatic Setup (Windows)**
```bash
start_network_server.bat
```

**Option 2: Manual Setup**
```bash
# Check your network configuration
python network_config.py

# Start with auto-detected IP (recommended)
python launcher.py web --network

# Or specify IP manually
python launcher.py web --network --host YOUR_LOCAL_IP
```

### Access URLs

- **Local:** http://localhost:8000
- **Network (share this):** http://YOUR_LOCAL_IP:8000

### Firewall Configuration

**Windows:**
```bash
# Allow port 8000 through Windows Firewall
netsh advfirewall firewall add rule name="TLS Web Server" dir=in action=allow protocol=TCP localport=8000
```

**Linux:**
```bash
# Allow port 8000 through UFW
sudo ufw allow 8000
```

## ESP32 Integration (Endpoints)

The ESP32 acts as a client and communicates with the Python FastAPI server.

### Flow Overview
- Default mode is verification.
- Registration mode can be switched via an endpoint (from the Web UI or the ESP32).
- Snapshots are collected until 5 are received, then a decision is returned in the response to the 5th POST.
- The Web UI shows live preview and thumbnails of recent annotated frames.

### Endpoints

1) POST `/esp32/hello`
- Headers: `Content-Type: application/json`
- Body:
```
{"deviceId":"cam1"}
```
- Response:
```
{"ok":true,"mode":"verification","targetSnapshots":5}
```

2) POST `/esp32/snapshot`
- Headers: `Content-Type: application/json`
- Body (send one image per request):
```
{"deviceId":"cam1","image":"data:image/jpeg;base64,BASE64_BYTES"}
```
- Responses:
  - While collecting (1–4):
```
{"ok":true,"completed":false,"received":n,"target":5}
```
  - On the 5th (decision returned here):
    - Verification:
```
{"ok":true,"completed":true,"action":"verify","result":"granted","userId":123,"name":"Alice"}
```
```
{"ok":true,"completed":true,"action":"verify","result":"denied"}
```
    - Registration:
```
{"ok":true,"completed":true,"action":"register","result":"success","userId":123,"message":"Registration successful for Alice"}
```
```
{"ok":false,"completed":true,"action":"register","result":"failed","message":"Registration failed"}
```

3) POST `/mode/registration`
- Switch to registration mode for the device; clears counters.
- Body:
```
{"deviceId":"cam1","userName":"Alice"}
```
- Response:
```
{"ok":true,"mode":"registration","targetSnapshots":5}
```

4) POST `/mode/verification`
- Switch back to verification mode for the device; clears counters.
- Body:
```
{"deviceId":"cam1"}
```
- Response:
```
{"ok":true,"mode":"verification","targetSnapshots":5}
```

5) GET `/status`
- Returns current UI-driving info:
```
{
  "uiEnabled": true,
  "activeDeviceId": "cam1",
  "mode": "verification",
  "snapshotsCollected": 3,
  "targetSnapshots": 5,
  "lastResult": null,
  "lastAnnotatedImage": "data:image/jpeg;base64,...",
  "recentAnnotatedImages": ["data:image/jpeg;base64,...", ...]
}
```

### ESP32 Client Pseudocode
```
POST /esp32/hello { deviceId }
for i in 1..5:
  POST /esp32/snapshot { deviceId, image: dataUrl }
  if response.completed == true:
    if response.action == 'verify' and response.result == 'granted': unlock
    else if response.action == 'register' and response.result == 'success': show success
    else: deny/fail
    break
```

---

## CLI Options
```bash
python main.py \
  --source 0 \
  --width 640 --height 480 \
  --max-fps 30 \
  --detection-confidence 0.7 \
  --tracking-confidence 0.6 \
  --use-geometry \
  --similarity-threshold 0.92 \
  --camera-buffer-size 3 \
  --cv2-threads 2 \
  --display \
  --esp32-enabled --esp32-port /dev/ttyUSB0 \
  --save-snaps --snaps-dir snapshots
```
Notes:
- `--use-geometry` toggles geometry features on top of LBP.
- `--save-snaps` stores palm ROI images and feature files to `--snaps-dir`.

---

## Project Structure
```
TLS/
  launcher.py             # Main launcher script (web/desktop modes)
  web_server.py           # FastAPI web server for palm snapshot collection
  main.py                 # Desktop app entry; orchestrates camera, detector, flows, UI
  camera.py               # VideoCapture wrapper integrating detection per frame
  detector.py             # MediaPipe-based palm detector + ROI builder
  preprocessing.py        # Optional ROI preprocessing helper (if present)
  feature_extraction.py   # LBP + optional geometry feature extraction
  registration.py         # Registration flow; averages features; stores in DB
  verification.py         # Verification flow; compares features to templates
  db.py                   # SQLite helpers for users and palm templates
  schema.sql              # Database schema
  utils/
    config.py             # Centralized runtime configuration constants
    palm.py               # Palm-facing heuristic/utilities
  requirements.txt
  README.md
```

---

## Architecture & Data Flow
1. Camera capture (`camera.VideoCapture`)
   - Reads frames from source
   - Calls `PalmDetector.detect` to annotate frame and return palm detections
2. Palm detection (`detector.PalmDetector`)
   - Runs MediaPipe Hands (max hands from `utils/config.py`)
   - Validates palm orientation via `utils.palm.is_palm_facing_camera`
   - Builds palm ROI and preprocesses (96×96)
   - Returns `PalmDetection`: `(bbox, palm_roi, landmarks, handedness, confidence)`
3. Registration (`registration.PalmRegistrar`)
   - Validates orientation, handedness consistency, and LBP variance
   - Extracts features once per detection via `PalmFeatureExtractor`
   - Averages valid feature vectors to a template and stores in DB

4. Verification (`verification.PalmVerifier` / `verify_palm_with_features`)
   - Validates and extracts features for current detections
   - Loads all user templates and computes cosine similarity
   - Grants access if best score exceeds threshold

---

## Modules

### `main.py`
- Parses CLI, sets logging, configures OpenCV threading
- Instantiates `PalmDetector`, `PalmRegistrar`, `VideoCapture`
- Event loop manages registration and verification flows
- Optional ESP32 signaling hooks

### `camera.py`
- Wrapper around OpenCV capture
- Holds a `PalmDetector` instance
- `get_palm_frame()` → `(success, annotated_frame, palm_detections)`
- Handles resize/buffer; saves ROIs if `--save-snaps`

### `detector.py`
- `DetectorConfig`: ROI size, confidences, padding, drawing, etc.
- `PalmDetector.detect(frame)`:
  - MediaPipe Hands inference
  - Orientation check using `utils.palm`
  - ROI extraction + preprocessing to 96×96
  - Optional debug drawing

### `feature_extraction.py`
- `LBPExtractor`: tiled-LBP histograms; returns `(vector, variance, tile_variances)`
- `GeometryFeatureExtractor`: landmark statistics (mean/std/max-distance)
- `PalmFeatureExtractor`: concatenates LBP and optional geometry into single vector
- `cosine_similarity` helper for verification

### `registration.py`
- `RegistrarConfig`: thresholds, `use_geometry`, snapshot options
- `PalmRegistrar.register_user_with_features(...)`:
  - Validates each detection once
  - Averages features; writes to DB via `db.save_palm_template`

### `verification.py`
- `PalmVerifier`: validates/extracts features and compares against templates
- `verify_palm_with_features(...)`: scans all users and returns first/best match

### `db.py` / `schema.sql`
- SQLite-based storage for users and templates
- `create_user`, `save_palm_template`, `load_user_templates`

### `utils/config.py`
- Central constants: detector thresholds, ROI size, similarity threshold, UI/window

### `utils/palm.py`
- Shared palm orientation heuristic

---

## Snapshots & Saved Artifacts
When `--save-snaps` is enabled:
- ROI images: saved by `camera.VideoCapture` as `roi_*.png` and optionally preprocessed `roi96_*.png`

---

## Tuning
- `DETECTION_CONFIDENCE`, `TRACKING_CONFIDENCE`: MediaPipe sensitivity
- `VARIANCE_THRESHOLD`: gate low-contrast LBP ROIs
- `similarity_threshold`: matching strictness (default ~0.92)
- Try `--use-geometry` or lower `similarity_threshold` if false negatives occur

---

## Troubleshooting
- Prefer Python 3.11/3.12 for prebuilt wheels (numpy/mediapipe/opencv)
- On Python 3.13 you may need `python3.13-devel`, BLAS/LAPACK, etc.
- If no frames: check `--source`, reduce resolution, verify camera permissions
- Few detections: improve lighting; ensure palm faces camera
- Registration issues: maintain consistent handedness and stable pose

---

## Development
- Clear naming, early returns, minimal inline comments
- Put new tunables in `utils/config.py`
- Keep hardware-specific code isolated (e.g., ESP32 signaling)

---
