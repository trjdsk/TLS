"""Palm detection pipeline: MediaPipe Hands + Edge Impulse validation.

Pipeline:
1. MediaPipe Hands detects hand landmarks
2. Extract palm ROI (wrist to MCP joints, excluding fingers)
3. Preprocess ROI into 96x96 grayscale uint8
4. Validate ROI with Edge Impulse model
5. Return validated palm ROIs ready for verification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import cv2

try:
    import mediapipe as mp  # type: ignore
except Exception as exc:
    mp = None
    _logging = logging.getLogger(__name__)
    _logging.debug("mediapipe import failed: %s", exc)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Represents a validated palm detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Optional[np.ndarray]  # shape: (21, 2) in pixel coords
    score: Optional[float]
    handedness: Optional[str]        # "Left" / "Right"
    is_valid_palm: bool              # True if EI model confirms palm
    palm_roi: Optional[np.ndarray]   # Preprocessed 96x96 grayscale ROI


class PalmDetector:
    """Palm detection + Edge Impulse validation."""

    def __init__(self,
                 max_num_hands: int = 1,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 palm_threshold: float = 0.5) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for palm detection")

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._palm_threshold = palm_threshold

        # Load Edge Impulse wrapper
        try:
            from model_wrapper import EdgeImpulseModel
            self._ei = EdgeImpulseModel()
            if self._ei.is_initialized:
                logger.info("Edge Impulse model initialized")
            else:
                logger.warning("Failed to initialize Edge Impulse model")
                self._ei = None
        except Exception as exc:
            logger.warning("Edge Impulse model not available: %s", exc)
            self._ei = None

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect palms and validate with Edge Impulse."""
        detections: List[DetectionResult] = []
        annotated = frame_bgr.copy()

        # Step 1: run MediaPipe Hands
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            h, w = annotated.shape[:2]
            handedness_list = getattr(results, "multi_handedness", None)

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coords = np.array(
                    [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark],
                    dtype=np.int32,
                )

                # Handedness & confidence
                score, handedness = None, None
                if handedness_list and idx < len(handedness_list):
                    try:
                        handedness_info = handedness_list[idx].classification[0]
                        score = float(handedness_info.score)
                        handedness = handedness_info.label
                    except Exception:
                        pass

                # Step 2: extract palm bbox
                palm_bbox = self._extract_palm_bbox(coords)
                if palm_bbox is None:
                    continue
                x_min, y_min, w_box, h_box = palm_bbox
                x_max, y_max = x_min + w_box, y_min + h_box

                # Pad a little
                pad = 10
                x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
                x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

                palm_roi = frame_bgr[y_min:y_max, x_min:x_max]

                # Step 3: validate with EI
                is_valid, processed_roi, ei_score = self._validate_palm(palm_roi)

                # Log detection scores
                logger.info("Detection - MediaPipe confidence: %.3f, Edge Impulse score: %.3f, handedness: %s, valid_palm: %s",
                           score or 0.0, ei_score or 0.0, handedness or "unknown", is_valid)

                detections.append(
                    DetectionResult(
                        bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                        landmarks=coords,
                        score=score,
                        handedness=handedness,
                        is_valid_palm=is_valid,
                        palm_roi=processed_roi,
                    )
                )

                # Draw bbox
                color = (0, 255, 0) if is_valid else (0, 0, 255)
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

        return annotated, detections

    def _extract_palm_bbox(self, landmarks: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Palm = wrist + MCP joints."""
        if landmarks.shape != (21, 2):
            return None
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        pts = landmarks[palm_indices]
        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
        if x_max - x_min < 20 or y_max - y_min < 20:
            return None
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def _preprocess_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Convert to 96x96 grayscale uint8."""
        if roi_bgr.size == 0:
            return np.zeros((96, 96), dtype=np.uint8)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
        resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _validate_palm(self, roi_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Run EI model and check if ROI is palm."""
        if self._ei is None or not self._ei.is_initialized:
            return False, None, None

        processed = self._preprocess_roi(roi_bgr)
        
        try:
            # Use the Edge Impulse model's is_palm method
            is_palm = self._ei.is_palm(processed, self._palm_threshold)
            
            # Get detailed scores for logging
            palm_score = None
            try:
                scores, predicted_class = self._ei.predict(processed)
                palm_score = scores[1] if len(scores) > 1 else scores[0]
                logger.debug("Edge Impulse prediction: scores=%s, predicted=%s, palm_score=%.3f, threshold=%.3f", 
                           scores, predicted_class, palm_score, self._palm_threshold)
            except Exception as exc:
                logger.warning("Could not get detailed Edge Impulse scores: %s", exc)
            
            logger.debug("Palm validation result: %s (threshold=%.3f)", is_palm, self._palm_threshold)
            return is_palm, processed, palm_score
            
        except Exception as exc:
            logger.error("Edge Impulse validation failed: %s", exc)
            return False, processed, None

    def close(self) -> None:
        if hasattr(self, "_hands"):
            self._hands.close()


__all__ = ["PalmDetector", "DetectionResult"]
