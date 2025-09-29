## Touchless Lock System (TLS)

Computer‑vision palm verification pipeline for a touchless door lock. Streams video, detects palms with MediaPipe, extracts robust LBP+geometry features, and verifies against registered users. Optional ESP32 signaling can unlock a door on successful verification.

### Highlights
- Real‑time palm detection with MediaPipe Hands
- Tiled‑LBP features plus optional palm‑geometry descriptors
- Registration and verification flows with similarity matching
- Snapshot saving for ROIs and extracted features when enabled
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
```bash
python main.py
```
Keys:
- q: Quit
- r: Start registration flow

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
  main.py                 # App entry; orchestrates camera, detector, flows, UI
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
