# Touchless Lock System - Refactored

A Python-based palmprint biometric verification system using MediaPipe for hand detection and Local Binary Pattern (LBP) features for biometric matching. This system is designed to run on ESP32-CAM and provides real-time palm detection and verification.

## 🚀 Key Features

- **MediaPipe-Only Detection**: Uses MediaPipe Hands for robust real-time palm detection
- **LBP Feature Extraction**: Implements Local Binary Pattern features for palmprint recognition
- **Hand Geometry Features**: Optional hand geometry features using MediaPipe landmarks
- **SQLite Database**: Stores feature vectors for user registration and verification
- **ESP32-Friendly**: Optimized for low-memory environments
- **Real-time Processing**: Designed for 10-15 FPS performance
- **Modular Architecture**: Clean, maintainable code structure

## 📋 Requirements

- Python 3.8+
- OpenCV 4.7+
- MediaPipe 0.10.14
- NumPy 1.26+
- scikit-image 0.19.0+

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python test_system.py
   ```

## 🏗️ Architecture

### Core Modules

- **`detector.py`**: MediaPipe-based palm detection and cropping
- **`feature_extraction.py`**: LBP and hand geometry feature extraction
- **`db.py`**: SQLite database management for user templates
- **`registration.py`**: User registration with feature extraction
- **`verification.py`**: Palm verification using feature similarity
- **`camera.py`**: Camera interface with palm detection integration
- **`main.py`**: Main application orchestrating the workflow

### Feature Extraction Pipeline

1. **Palm Detection**: MediaPipe Hands detects hand landmarks
2. **Palm Cropping**: Extract palm region using landmarks 0-17
3. **Preprocessing**: Convert to grayscale, resize to 96x96, histogram equalization
4. **LBP Features**: Compute Local Binary Pattern histogram (26 features)
5. **Geometry Features**: Optional hand geometry features (14 features)
6. **Feature Vector**: Combined LBP + geometry features (40 total)

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

-- Palm templates table
CREATE TABLE palm_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    handedness TEXT NOT NULL,
    feature_vector BLOB NOT NULL,
    feature_type TEXT NOT NULL DEFAULT 'LBP',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
```

## 🚀 Usage

### Basic Usage

```bash
# Run the main application
python main.py

# With custom settings
python main.py --detection-confidence 0.8 --similarity-threshold 0.9 --use-geometry
```

### Command Line Options

- `--source`: Camera index or stream URL (default: 0)
- `--detection-confidence`: MediaPipe detection confidence (default: 0.7)
- `--tracking-confidence`: MediaPipe tracking confidence (default: 0.6)
- `--use-geometry`: Include hand geometry features (default: True)
- `--similarity-threshold`: Feature similarity threshold (default: 0.85)
- `--max-fps`: Limit processing FPS (default: 30)
- `--save-snaps`: Save palm snapshots to disk
- `--esp32-enabled`: Enable ESP32 communication

### Interactive Controls

- **'r'**: Start user registration
- **'c'**: Cancel registration
- **'q'**: Quit application

## 🔧 Configuration

### Feature Extraction Parameters

```python
# LBP parameters
LBP_RADIUS = 3          # LBP radius
LBP_N_POINTS = 24       # Number of LBP points
LBP_METHOD = 'uniform'  # LBP method

# Geometry features
USE_GEOMETRY = True     # Include hand geometry
SIMILARITY_METHOD = 'cosine'  # Similarity calculation method
```

### Verification Thresholds

```python
# MediaPipe confidence
MIN_DETECTION_CONFIDENCE = 0.7

# Feature similarity
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold
```

## 📊 Performance

### Memory Usage
- **LBP Features**: 26 × 4 bytes = 104 bytes per template
- **Geometry Features**: 14 × 4 bytes = 56 bytes per template
- **Total per User**: ~160 bytes (very ESP32-friendly)

### Processing Speed
- **MediaPipe Detection**: ~10-15 FPS on ESP32-CAM
- **Feature Extraction**: ~5-10ms per palm
- **Verification**: ~1-2ms per comparison

## 🧪 Testing

Run the test suite to verify system functionality:

```bash
python test_system.py
```

The test suite covers:
- Feature extraction functionality
- Database operations
- MediaPipe detector initialization
- Verification system integration

## 🔄 Migration from Edge Impulse

This refactored system replaces the previous Edge Impulse + TFLite pipeline:

### What Changed
- ❌ **Removed**: Edge Impulse TFLite model and C++ wrapper
- ❌ **Removed**: TensorFlow dependencies
- ❌ **Removed**: ORB/SIFT/SURF feature matching
- ✅ **Added**: MediaPipe-only detection
- ✅ **Added**: LBP feature extraction
- ✅ **Added**: Hand geometry features
- ✅ **Added**: Feature vector similarity matching

### Database Migration
The database schema has been updated to store feature vectors instead of descriptors. Existing databases will need to be migrated or recreated.

## 🐛 Troubleshooting

### Common Issues

1. **MediaPipe not detecting hands**
   - Check camera permissions
   - Ensure good lighting
   - Adjust detection confidence threshold

2. **Low verification accuracy**
   - Increase similarity threshold
   - Enable geometry features
   - Ensure consistent palm positioning

3. **Performance issues**
   - Reduce max FPS
   - Disable geometry features
   - Lower camera resolution

### Debug Mode

Enable debug logging for detailed information:

```bash
python main.py --log-level DEBUG
```

## 📈 Future Enhancements

- **Multi-modal Features**: Combine with fingerprint or face recognition
- **Adaptive Thresholds**: Dynamic threshold adjustment based on environment
- **Template Updates**: Incremental learning for improved accuracy
- **Cloud Integration**: Remote template storage and synchronization
- **Mobile App**: Companion app for user management

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test suite for examples

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional security measures and thorough testing.
