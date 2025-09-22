"""Palm detection module using MediaPipe + TFLite Edge Impulse model.

Provides PalmDetector class that:
1. Uses MediaPipe Hands to find hand bounding boxes
2. Crops ROI and resizes to 96x96x1 grayscale
3. Normalizes and quantizes for INT8 TFLite model
4. Runs Edge Impulse inference for palm classification
5. Applies debouncing (must detect "palm" ≥3/5 frames)
6. Returns annotated frame and palm crops
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PalmDetection:
    """Represents a detected palm with bounding box and metadata."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    palm_roi: np.ndarray  # 96x96 grayscale palm crop
    is_valid_palm: bool  # True if TFLite confirms palm
    tflite_score: float  # TFLite confidence score
    handedness: Optional[str] = None  # "Left" or "Right"


class PalmDetector:
    """Palm detector using MediaPipe + TFLite Edge Impulse model."""
    
    def __init__(self, model_path: str, max_num_hands: int = 2, 
                 detection_confidence: float = 0.7, tracking_confidence: float = 0.6,
                 palm_threshold: float = 0.85, smoothing_window: int = 5):
        """
        Initialize palm detector.
        
        Args:
            model_path: Path to TFLite model file
            max_num_hands: Maximum number of hands to detect
            detection_confidence: MediaPipe detection confidence threshold
            tracking_confidence: MediaPipe tracking confidence threshold
            palm_threshold: TFLite palm classification threshold
            smoothing_window: Number of frames for confidence smoothing
        """
        self.model_path = model_path
        self.palm_threshold = palm_threshold
        self.smoothing_window = smoothing_window
        
        # Load TFLite model
        self._load_tflite_model()
        
        # Initialize MediaPipe
        self._init_mediapipe(max_num_hands, detection_confidence, tracking_confidence)
        
        # Confidence history for debouncing
        self.confidence_history = deque(maxlen=smoothing_window)
        
        logger.info("PalmDetector initialized with model: %s", model_path)
    
    def _load_tflite_model(self):
        """Load and initialize TFLite model."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.labels = ["not_palm", "palm"]
            self.input_index = self.input_details[0]['index']
            self.output_index = self.output_details[0]['index']
            
            # Expected input size: 96x96x1
            self.INPUT_H, self.INPUT_W, self.INPUT_C = 96, 96, 1
            
            logger.info("TFLite model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load TFLite model: %s", e)
            raise
    
    def _init_mediapipe(self, max_num_hands: int, detection_confidence: float, tracking_confidence: float):
        """Initialize MediaPipe hands detection."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        logger.info("MediaPipe hands initialized")
    
    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for TFLite inference.
        
        Args:
            roi: BGR image crop
            
        Returns:
            Preprocessed input tensor [1, 96, 96, 1]
        """
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Lighting normalization (CLAHE for uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_eq = clahe.apply(roi_gray)
        
        # Resize to 96x96
        roi_resized = cv2.resize(roi_eq, (self.INPUT_W, self.INPUT_H))
        
        # Normalize to [-128, 127] range for int8 quantization
        roi_norm = roi_resized.astype(np.float32) / 255.0  # [0,1]
        roi_int8 = ((roi_norm - 0.5) * 255).astype(np.int8)  # center around 0
        
        # Shape: [1,96,96,1]
        input_data = np.expand_dims(roi_int8, axis=(0, -1))
        return input_data
    
    def _run_tflite_inference(self, roi: np.ndarray) -> Tuple[bool, float]:
        """
        Run TFLite inference on palm ROI.
        
        Args:
            roi: BGR image crop
            
        Returns:
            Tuple of (is_palm, confidence_score)
        """
        try:
            # Preprocess ROI
            input_data = self._preprocess_roi(roi)
            
            # Run inference
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_index)[0]
            
            # Get prediction
            pred_index = int(np.argmax(output_data))
            confidence = output_data[pred_index] / 128.0  # adjust scaling if needed
            label = self.labels[pred_index]
            
            is_palm = (label == "palm" and confidence > self.palm_threshold)
            return is_palm, confidence
            
        except Exception as e:
            logger.error("TFLite inference failed: %s", e)
            return False, 0.0
    
    def _get_handedness(self, hand_landmarks) -> str:
        """
        Determine handedness from MediaPipe landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            "Left" or "Right"
        """
        # Use wrist (landmark 0) and middle finger MCP (landmark 9) to determine handedness
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        
        # If middle finger MCP is to the right of wrist, it's a right hand
        if middle_mcp.x > wrist.x:
            return "Right"
        else:
            return "Left"
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PalmDetection]]:
        """
        Detect palms in frame using MediaPipe + TFLite.
        
        Args:
            frame: BGR input frame
            
        Returns:
            Tuple of (annotated_frame, list_of_palm_detections)
        """
        annotated_frame = frame.copy()
        detections = []
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box from landmarks
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Expand box slightly
                pad = 20
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                
                # Crop ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                
                # Run TFLite inference
                is_palm, confidence = self._run_tflite_inference(roi)
                
                # Add confidence to history for debouncing
                self.confidence_history.append(confidence)
                
                # Smooth confidence over last N frames
                smoothed_conf = np.mean(self.confidence_history)
                
                # Apply debouncing: must detect "palm" ≥3/5 frames
                palm_count = sum(1 for c in self.confidence_history if c > self.palm_threshold)
                is_valid_palm = (palm_count >= 3) and is_palm
                
                # Get handedness
                handedness = self._get_handedness(hand_landmarks)
                
                # Create palm crop (96x96 grayscale)
                palm_roi = self._preprocess_roi(roi).squeeze()  # Remove batch and channel dims
                
                # Create detection object
                detection = PalmDetection(
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    palm_roi=palm_roi,
                    is_valid_palm=is_valid_palm,
                    tflite_score=smoothed_conf,
                    handedness=handedness
                )
                detections.append(detection)
                
                # Draw annotations
                if is_valid_palm:
                    color = (0, 255, 0)  # Green for valid palm
                    label_text = f"Palm: {smoothed_conf:.2f}"
                else:
                    color = (0, 0, 255)  # Red for invalid/no palm
                    label_text = f"Not Palm: {smoothed_conf:.2f}"
                
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(annotated_frame, label_text, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(annotated_frame, f"Hand: {handedness}", (x_min, y_max + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, 
                                             self.mp_hands.HAND_CONNECTIONS)
        
        return annotated_frame, detections
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            logger.info("PalmDetector closed")
        except Exception as e:
            logger.error("Error closing PalmDetector: %s", e)


def detect_palms(frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        frame: BGR input frame
        
    Returns:
        Tuple of (annotated_frame, list_of_palm_crops)
    """
    # Create detector instance (this should be done once and reused)
    detector = PalmDetector("model/tflite-model/tflite_learn_781277_3.tflite")
    
    try:
        annotated_frame, detections = detector.detect(frame)
        palm_crops = [det.palm_roi for det in detections if det.is_valid_palm]
        return annotated_frame, palm_crops
    finally:
        detector.close()
