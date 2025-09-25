"""Palm detection module using MediaPipe Hands.

Provides PalmDetector class that:
1. Uses MediaPipe Hands to find palm bounding boxes (excludes fingers)
2. Crops ROI and converts to 96x96 grayscale (no TFLite here)
3. Returns annotated frame and palm crops for downstream processing
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from preprocessing import preprocess_roi_96


@dataclass
class PalmDetection:
    """Represents a detected palm with bounding box and metadata."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    palm_roi: np.ndarray  # 96x96 grayscale palm crop
    is_valid_palm: bool  # True if TFLite confirms palm
    tflite_score: float  # TFLite confidence score
    handedness: Optional[str] = None  # "Left" or "Right"


class PalmDetector:
    """Palm detector using MediaPipe; produces palm-only crops."""
    
    def __init__(self, model_path: str | None = None, max_num_hands: int = 2, 
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
        self.model_path = model_path  # kept for compatibility (unused here)
        self.palm_threshold = palm_threshold
        self.smoothing_window = smoothing_window
        
        # Initialize MediaPipe
        self._init_mediapipe(max_num_hands, detection_confidence, tracking_confidence)
        
        logger.info("PalmDetector initialized (MediaPipe only; downstream model handled elsewhere)")

    
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
    
    def _compute_palm_bbox(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Compute a palm-only bounding box that includes full palm area while excluding fingers.

        Strategy:
        1. Use wrist (0) and palm base landmarks (1, 2, 5, 9, 13, 17) for palm outline
        2. Add extra padding to ensure full palm coverage
        3. Use palm center to determine if we need asymmetric padding
        """
        h, w, _ = frame_shape
        
        # Palm landmarks: wrist + palm base + MCP joints (excluding finger tips)
        palm_landmarks = [0, 1, 2, 5, 9, 13, 17]  # wrist, palm base, MCPs
        
        # Get coordinates of palm landmarks
        xs = [hand_landmarks.landmark[i].x * w for i in palm_landmarks]
        ys = [hand_landmarks.landmark[i].y * h for i in palm_landmarks]
        
        # Calculate palm center for asymmetric padding
        palm_center_x = np.mean(xs)
        palm_center_y = np.mean(ys)
        
        # Initial bounding box from palm landmarks
        x_min, x_max = int(max(0, min(xs))), int(min(w, max(xs)))
        y_min, y_max = int(max(0, min(ys))), int(min(h, max(ys)))
        
        # Calculate palm dimensions for proportional padding
        palm_width = x_max - x_min
        palm_height = y_max - y_min
        
        # Add generous padding to ensure full palm coverage
        # Use 20-25% of palm size as padding, with minimum of 15px
        pad_x = max(15, int(palm_width * 0.25))
        pad_y = max(15, int(palm_height * 0.25))
        
        # Apply padding
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        # Ensure minimum size for palm detection
        min_size = 40
        if (x_max - x_min) < min_size:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - min_size // 2)
            x_max = min(w, center_x + min_size // 2)
        
        if (y_max - y_min) < min_size:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - min_size // 2)
            y_max = min(h, center_y + min_size // 2)

        return x_min, y_min, x_max - x_min, y_max - y_min
    
    # TFLite inference removed from detector; handled in verification module
    
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
                # Compute palm-only bounding box (exclude fingers)
                x_min, y_min, w_box, h_box = self._compute_palm_bbox(hand_landmarks, frame.shape)
                x_max, y_max = x_min + w_box, y_min + h_box

                # Crop ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                
                # Preprocess to 96x96 grayscale (no model gating here)
                roi_96 = preprocess_roi_96(roi)

                # Get handedness
                handedness = self._get_handedness(hand_landmarks)
                
                palm_roi = roi_96
                
                # Create detection object
                detection = PalmDetection(
                    bbox=(x_min, y_min, w_box, h_box),
                    palm_roi=palm_roi,
                    is_valid_palm=True,  # gating deferred; mark as candidate
                    tflite_score=0.0,
                    handedness=handedness
                )
                detections.append(detection)
                
                # Draw annotations
                color = (0, 255, 0)
                label_text = "Palm candidate"
                
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
