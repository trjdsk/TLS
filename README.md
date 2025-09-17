# TLS Palm Recognition System

A robust, real-time palm recognition system built with Python, OpenCV, MediaPipe Hands, and SQLite. This system provides secure palm-based authentication with advanced features including handedness detection, palm orientation normalization, and strict verification protocols.

## 🚀 Features

### Core Capabilities
- **Real-time Palm Detection**: Uses MediaPipe Hands for accurate hand tracking and landmark detection
- **Handedness-Aware Recognition**: Separate registration and verification for left and right hands
- **Palm Orientation Normalization**: Consistent palm orientation for improved recognition accuracy
- **Strict Verification Protocols**: High-precision matching with variance checking
- **Robust Registration Process**: 15-snapshot registration with hand side selection and name collection
- **SQLite Database**: Persistent storage with automatic schema migration

### Advanced Features
- **Modular Design**: Easy integration with Edge Impulse models (planned)
- **Configurable Thresholds**: Adjustable similarity and variance parameters
- **Handedness Enforcement**: Toggleable left/right hand matching
- **Palm-Only Mode**: Filters to open palm detections only (default)
- **Palm Region Focus**: Extracts palm-only regions excluding fingers (default)

## 📋 Requirements

- Python 3.8+
- OpenCV 4.7+
- MediaPipe 0.10.14
- NumPy 1.26+
- scikit-image 0.19.0+

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TLS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**:
   ```bash
   python main.py
   ```

## 🎮 Usage

### Basic Operation

**Start the system**:
```bash
python main.py
```

**Controls**:
- `r` - Start palm registration
- `c` - Cancel current registration
- `q` - Quit the application

### Registration Process

1. Press `r` to start registration
2. **Select hand side**: Choose between Left or Right hand
3. The system will capture 15 snapshots with movement guidance
4. Follow the on-screen instructions for varied hand positions
5. **Enter your name**: After successful capture, enter your name in the terminal
6. Registration completes automatically when 15 valid snapshots are captured

### Verification Process

- **Automatic**: The system continuously verifies palms in real-time
- **5 Snapshot Requirement**: Captures 5 snapshots for verification
- **Strict Matching**: All snapshots must meet similarity thresholds
- **1-Second Cooldown**: Completely stops detection for 1 second after each verification result

## ⚙️ Configuration

### Command Line Options

#### Detection Settings
```bash
--backend {mediapipe,edgeimpulse}  # Detection backend (default: mediapipe)
--source SOURCE                    # Camera index or stream URL (default: 0)
--width WIDTH                      # Desired frame width
--height HEIGHT                    # Desired frame height
```

#### Palm Detection Modes
```bash
--no-palm-only                    # Disable palm-only filtering (allow all hand poses)
--no-palm-region-only             # Show full hand landmarks instead of palm region only
```

#### Performance Tuning
```bash
--max-fps FPS                     # Limit processing FPS (default: 30.0)
--process-every N                 # Process every Nth frame when not registering (default: 1)
--cv2-threads N                   # Set OpenCV thread count
```

#### MediaPipe Settings
```bash
--mp-max-hands N                  # Maximum number of hands to detect (default: 2)
--mp-det-conf CONF                # Minimum detection confidence (default: 0.5)
--mp-track-conf CONF              # Minimum tracking confidence (default: 0.5)
```

#### Verification Settings
```bash
--enforce-handedness              # Enforce handedness matching during verification (default: enabled)
--no-enforce-handedness           # Disable handedness matching during verification
```

#### System Settings
```bash
--log-level LEVEL                 # Logging level (default: INFO)
--display                         # Show window and accept hotkeys (default: enabled)
--no-display                      # Run headless mode
```

### Example Commands

**Basic usage with default settings**:
```bash
python main.py
```

**High-performance mode**:
```bash
python main.py --max-fps 60 --cv2-threads 4 --process-every 2
```

**Debug mode with verbose logging**:
```bash
python main.py --log-level DEBUG --no-display
```

**Allow all hand poses**:
```bash
python main.py --no-palm-only --no-palm-region-only
```

## 🏗️ Architecture

### System Components

```
main.py              # Main application entry point
├── camera.py        # Video capture abstraction
├── detector.py      # Palm detection backends (MediaPipe/Edge Impulse)
├── registration.py  # Palm registration and embedding extraction
├── verification.py  # Palm verification and matching
└── schema.sql       # Database schema definition
```

### Database Schema

```sql
CREATE TABLE registered_palms (
    user_id TEXT,
    handedness TEXT,
    name TEXT,
    embeddings BLOB NOT NULL,
    PRIMARY KEY (user_id, handedness)
);
```

### Key Features Explained

#### Palm-Only Mode (Default)
- **`--palm-only`**: Filters detections to open palms only using finger extension heuristics
- **`--palm-region-only`**: Extracts palm-only bounding boxes excluding fingers
- **Impact**: Improves recognition accuracy by focusing on palm features rather than finger positions

#### Handedness Detection
- **Automatic Detection**: Uses MediaPipe's handedness classification
- **Separate Storage**: Left and right hands stored independently
- **Verification Filtering**: Only compares against same-handedness registrations

#### Palm Normalization
- **Orientation Correction**: Uses wrist-to-middle-finger-MCP vector for consistent orientation
- **Rotation**: Automatically rotates palm to upward orientation
- **Consistency**: Reduces left/right hand confusion

#### Verification Thresholds
- **Similarity Threshold**: 0.92 (individual snapshot matching)
- **Average Similarity**: 0.90 (overall match quality)
- **Peak Similarity**: 0.95 (best single match)
- **Variance Limit**: 0.05 (consistency requirement)

## 🔧 Development

### Adding New Detection Backends

The system supports pluggable detection backends. To add a new backend:

1. **Implement the backend class** in `detector.py`
2. **Add backend selection** in `PalmDetector.__init__()`
3. **Update command line options** in `main.py`

### Customizing Embedding Extraction

The `extract_embedding()` function in `registration.py` is designed to be easily replaceable:

```python
def extract_embedding(roi_bgr: np.ndarray) -> np.ndarray:
    # Current implementation uses multi-feature approach
    # Replace with Edge Impulse model or other embedding extractor
    pass
```

### Database Migration

The system automatically migrates existing databases to support handedness:

- **Existing Users**: Automatically marked as "Right" handedness
- **Schema Updates**: Handled transparently on first run
- **Backward Compatibility**: Preserves all existing data

## 📊 Performance

### Recommended Settings

**High Accuracy**:
```bash
python main.py --mp-det-conf 0.7 --mp-track-conf 0.7
```

**High Performance**:
```bash
python main.py --max-fps 60 --process-every 2 --cv2-threads 4
```

**Balanced**:
```bash
python main.py --max-fps 30 --mp-det-conf 0.5 --mp-track-conf 0.5
```

### System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 4GB+ recommended
- **Camera**: USB webcam or built-in camera
- **OS**: Linux, Windows, macOS

## 🐛 Troubleshooting

### Common Issues

**Camera not detected**:
```bash
python main.py --source 1  # Try different camera index
```

**Low detection accuracy**:
```bash
python main.py --mp-det-conf 0.7 --mp-track-conf 0.7
```

**Performance issues**:
```bash
python main.py --max-fps 15 --process-every 3 --cv2-threads 1
```

**Database errors**:
- Delete `palms.db` to reset database
- Check file permissions in project directory

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python main.py --log-level DEBUG
```

## 📝 License

[Add your license information here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the debug logs with `--log-level DEBUG`

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional security measures and performance optimizations.
