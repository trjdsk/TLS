# TFLite Palm Detection Integration

This document describes the integration of TensorFlow Lite models with computer vision techniques for palm detection and validation.

## Overview

The integration provides a computer vision-based approach for palm detection:

1. **Computer Vision Detector** (`detector.py`): A complete pipeline using OpenCV thresholding and contour detection for ROI detection and TFLite for palm classification.

## Pipeline

```
Frame → Computer Vision Detection → Crop & Preprocess → TFLite Inference → Result
```

### Steps:
1. **Computer Vision** detects hand regions using thresholding and contour detection
2. **Crop** hand regions from the original frame
3. **Preprocess**: Convert to grayscale → Resize to 96×96 → Normalize [0,1] → Reshape (1,96,96,1)
4. **TFLite Inference** for palm classification
5. **Result**: Palm/Not_Palm prediction with confidence

## Installation

### 1. Update Requirements

The `requirements.txt` has been updated with protobuf compatibility:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `protobuf==4.25.3` (compatible with TensorFlow Lite)
- `tensorflow==2.17.1` (includes TensorFlow Lite interpreter)
- `opencv-python>=4.7` (for computer vision operations)

### 2. Model Files

The TFLite model is located at:
```
/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite
```

## Usage

### Standalone TFLite Detector

```python
from tflite_palm_detector import TFLitePalmDetector

# Initialize detector
detector = TFLitePalmDetector(
    model_path="/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite",
    palm_threshold=0.5
)

# Process frame
annotated, detections = detector.detect(frame)

# Check results
for detection in detections:
    print(f"Palm: {detection.is_palm}, Confidence: {detection.confidence:.3f}")
```

### Command Line Usage

```bash
# Webcam
python tflite_palm_detector.py --model "/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite"

# ESP32-CAM stream
python tflite_palm_detector.py --model "/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite" --source http://192.168.1.100:81/stream

# Custom threshold
python tflite_palm_detector.py --model "/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite" --threshold 0.7
```

### Integrated Detector

```python
from detector import PalmDetector

# Initialize with TFLite model
detector = PalmDetector()
detector.load_tflite_model("/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite")

# Use as before
annotated, detections = detector.detect(frame)
```

### Example Script

```bash
# Use the integrated detector with TFLite
python example_tflite_detection.py --tflite-model "/home/semei/VS Code/TLS/model/tflite-model/tflite_learn_781277_3.tflite"
```

## API Reference

### TFLitePalmDetector

#### Constructor
```python
TFLitePalmDetector(
    model_path: str,
    max_num_hands: int = 1,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    palm_threshold: float = 0.5
)
```

#### Methods

- `detect(frame)`: Main detection method
- `detect_hands(frame)`: Detect hand bounding boxes
- `preprocess_crop(crop)`: Preprocess crop for inference
- `infer_palm(tensor)`: Run TFLite inference
- `close()`: Clean up resources

### PalmDetector (Enhanced)

#### New Methods

- `load_tflite_model(model_path)`: Load TFLite model
- `_validate_palm_tflite(roi)`: Validate using TFLite

## Data Structures

### TFLiteDetectionResult
```python
@dataclass
class TFLiteDetectionResult:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    is_palm: bool                    # True if classified as palm
    confidence: float                # Confidence score
    processed_crop: Optional[np.ndarray]  # Preprocessed (1,96,96,1) tensor
```

## Testing

Run the test script to verify the integration:

```bash
python test_tflite_model.py
```

This will test:
- Computer vision detection
- TFLite model loading
- Preprocessing pipeline
- Inference functionality

## Video Sources

The system supports multiple video sources:

### Webcam
```python
cap = cv2.VideoCapture(0)  # Default webcam
```

### ESP32-CAM HTTP Stream
```python
cap = cv2.VideoCapture("http://192.168.1.100:81/stream")
```

### File
```python
cap = cv2.VideoCapture("video.mp4")
```

## Performance Considerations

### TensorFlow Lite Integration
- **tensorflow.lite**: Integrated with full TensorFlow framework
- **Interpreter**: Optimized for inference with full TensorFlow ecosystem

### Model Optimization
- The TFLite model is optimized for mobile/edge deployment
- Input: 96×96 grayscale images
- Output: Binary classification (palm/not_palm)

### Memory Usage
- TFLite interpreter is lightweight
- Computer vision operations have minimal memory footprint
- Suitable for real-time processing

## Troubleshooting

### Common Issues

1. **Protobuf Version Conflicts**
   ```
   Solution: Use protobuf==4.25.3
   ```

2. **TFLite Import Errors**
   ```bash
   pip install tensorflow==2.17.1
   ```

3. **Model Loading Failures**
   - Check model file path
   - Verify model format (.tflite)
   - Check file permissions

4. **OpenCV Issues**
   ```bash
   pip install opencv-python>=4.7
   ```

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use command line:
```bash
python tflite_palm_detector.py --model model.tflite --verbose
```

## Integration Benefits

1. **Flexibility**: Support both Edge Impulse and TFLite models
2. **Performance**: Optimized TFLite runtime for edge deployment
3. **Compatibility**: Protobuf version compatibility
4. **Real-time**: Suitable for live video processing
5. **Modular**: Clean separation of detection and classification

## Future Enhancements

- Support for multiple TFLite models
- Batch processing capabilities
- Model quantization options
- Custom preprocessing pipelines
- Performance benchmarking tools
