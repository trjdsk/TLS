"""Palm detection module using MediaPipe Hands.

Provides PalmDetector class that:
1. Uses MediaPipe Hands to find palm bounding boxes (excludes fingers)
2. Crops ROI and converts to 96x96 grayscale with histogram equalization
3. Returns annotated frame and palm crops with landmarks for feature extraction
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

from preprocessing import preprocess_roi_96


@dataclass
class PalmDetection:
    """Represents a detected palm with bounding box and metadata."""
    bbox: Tuple[int, int, int, int]
    palm_roi: np.ndarray
    landmarks: Any
    handedness: Optional[str] = None
    confidence: float = 1.0


class PalmDetector:
    """Palm detector using MediaPipe; produces palm-only crops."""
    
    def __init__(self, max_num_hands: int = 2, detection_confidence: float = 0.7, tracking_confidence: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        logger.info("PalmDetector initialized")

    def _compute_palm_bbox(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[int,int,int,int]:
        h, w, _ = frame_shape
        palm_landmarks = [0, 1, 2, 5, 9, 13, 17]
        xs = [hand_landmarks.landmark[i].x * w for i in palm_landmarks]
        ys = [hand_landmarks.landmark[i].y * h for i in palm_landmarks]
        x_min, x_max = int(max(0,min(xs))), int(min(w,max(xs)))
        y_min, y_max = int(max(0,min(ys))), int(min(h,max(ys)))
        palm_width, palm_height = x_max-x_min, y_max-y_min
        pad_x, pad_y = max(15,int(palm_width*0.25)), max(15,int(palm_height*0.25))
        x_min, y_min = max(0,x_min-pad_x), max(0,y_min-pad_y)
        x_max, y_max = min(w,x_max+pad_x), min(h,y_max+pad_y)
        min_size = 40
        if (x_max-x_min)<min_size: x_min,x_max = max(0,(x_min+x_max)//2 - min_size//2), min(w,(x_min+x_max)//2 + min_size//2)
        if (y_max-y_min)<min_size: y_min,y_max = max(0,(y_min+y_max)//2 - min_size//2), min(h,(y_min+y_max)//2 + min_size//2)
        return x_min, y_min, x_max-x_min, y_max-y_min

    def _resize_with_padding(self, img, size=(96,96)):
        h, w = img.shape[:2]
        scale = min(size[0]/h, size[1]/w)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w,new_h))
        padded = np.zeros(size, dtype=resized.dtype)
        padded[:new_h,:new_w] = resized
        return padded

    def _preprocess_palm_roi(self, roi: np.ndarray, landmarks=None) -> np.ndarray:
        if roi is None or roi.size==0:
            raise ValueError("Empty ROI")
        if len(roi.shape)==3: gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else: gray = roi.copy()
        resized = self._resize_with_padding(gray)
        equalized = cv2.equalizeHist(resized)
        return equalized.astype(np.uint8)

    def _is_palm_facing_camera(self, landmarks) -> bool:
        fingertip_indices = [4,8,12,16,20]
        visible = sum(1 for idx in fingertip_indices if landmarks.landmark[idx].z < landmarks.landmark[idx-1].z)
        return visible>=4

    def _get_handedness(self, hand_landmarks) -> str:
        return "Right" if hand_landmarks.landmark[9].x > hand_landmarks.landmark[0].x else "Left"

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PalmDetection]]:
        annotated = frame.copy()
        detections = []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if not self._is_palm_facing_camera(hand_landmarks):
                    logger.debug("Palm not facing camera")
                    continue
                x_min,y_min,w_box,h_box = self._compute_palm_bbox(hand_landmarks, frame.shape)
                roi = frame[y_min:y_min+h_box, x_min:x_min+w_box]
                if roi.size==0: continue
                palm_roi = self._preprocess_palm_roi(roi, hand_landmarks)
                handedness = self._get_handedness(hand_landmarks)
                confidence = 1.0
                if results.multi_handedness:
                    confidence = results.multi_handedness[idx].classification[0].score
                det = PalmDetection(bbox=(x_min,y_min,w_box,h_box), palm_roi=palm_roi,
                                    landmarks=hand_landmarks, handedness=handedness, confidence=confidence)
                detections.append(det)
                color=(0,255,0)
                cv2.rectangle(annotated,(x_min,y_min),(x_min+w_box,y_min+h_box),color,2)
                cv2.putText(annotated,f"Palm({handedness})",(x_min,y_min-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                cv2.putText(annotated,f"Conf:{confidence:.2f}",(x_min,y_min+h_box+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                self.mp_drawing.draw_landmarks(annotated, hand_landmarks,self.mp_hands.HAND_CONNECTIONS)
        return annotated, detections

    def close(self):
        if hasattr(self,'hands'): self.hands.close()
        logger.info("PalmDetector closed")
