## Palm Detection (Pluggable Backends)

Modular Python project for real-time palm/hand detection. Default backend uses MediaPipe, with a swappable architecture ready for a future Edge Impulse (.eim) model.

### Features
- **Modular architecture**: `camera.py`, `detector.py`, `main.py`.
- **MediaPipe backend**: Landmarks and bounding boxes drawn in real time.
- **Edge Impulse stub**: Interface in place; raises `NotImplementedError` until implemented.
- **Config flags**: `immediate_forwarding`, `buffered_mode` placeholders for ESP32 workflows.
- **Logging**: Clear messages for detection status.

### Requirements
- Python 3.12 (recommended for MediaPipe compatibility)
- Install dependencies:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py --backend mediapipe --source 0
```

- Press `q` to quit.
- Use `--log-level DEBUG` for more verbose logs.
- `--source` accepts a webcam index (e.g., `0`) or a stream URL.

### Backends
- **mediapipe**: Fully functional. Requires the `mediapipe` package.
- **edgeimpulse**: Not implemented yet. Intended to load a `.eim` model via `edge-impulse-linux` in the future.

### Design Notes
- `PalmDetector` abstracts detection. Swap backends with `--backend` or via code.
- `VideoCapture` handles input from webcam or URLs; ESP32 support can plug in via URL stream.
- Code is documented with docstrings and raises clear errors for invalid configuration.

### Future Extensions
- Recognition/classification on top of detection
- SD card logging and retrieval
- Servo motor control based on gesture events
