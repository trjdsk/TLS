"""
Refactored Palm detection module using MediaPipe Hands.

Key improvements:
- Centralized config constants
- Uses shared palm-facing check from utils.palm
- Defensive input checks and ROI bounds checks
- Optional use of external `preprocess_roi_96` (imported safely)
- Cleaner logging and resource cleanup
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils import config
from utils.palm import is_palm_facing_camera

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Optional preprocessing helper
try:
    from preprocessing import preprocess_roi_96  # optional helper
    _HAS_PREPROCESS_HELPER = True
except Exception:
    preprocess_roi_96 = None
    _HAS_PREPROCESS_HELPER = False
    logger.debug("preprocess_roi_96 not available; using built-in preprocessing.")


@dataclass
class DetectorConfig:
    size: Tuple[int, int] = config.ROI_SIZE
    min_detection_confidence: float = config.DETECTION_CONFIDENCE
    min_tracking_confidence: float = config.TRACKING_CONFIDENCE
    palm_padding_ratio: float = 0.25
    palm_min_size: int = 40
    palm_normal_angle_thresh_deg: float = 60.0
    fingertip_visibility_req: int = 4
    handedness_confidence_threshold: float = 0.0
    debug_draw: bool = True


@dataclass
class PalmDetection:
    bbox: Tuple[int, int, int, int]
    palm_roi: np.ndarray
    landmarks: Any
    handedness: Optional[str] = None
    confidence: float = 0.0


class PalmDetector:
    def __init__(self, config: DetectorConfig = DetectorConfig(), max_num_hands: int = 2):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        logger.info("PalmDetector initialized with config: %s", self.config)

    # ---------- Utilities ----------
    def _validate_frame(self, frame: np.ndarray):
        if frame is None:
            raise ValueError("Frame is None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be HxWx3 BGR image")

    def _compute_palm_bbox(self, hand_landmarks, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        h, w, _ = frame_shape
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        xs, ys = [], []
        for i in palm_indices:
            lm = hand_landmarks.landmark[i]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        x_min, x_max = int(max(0, min(xs))), int(min(w - 1, max(xs)))
        y_min, y_max = int(max(0, min(ys))), int(min(h - 1, max(ys)))
        palm_w, palm_h = x_max - x_min, y_max - y_min
        pad_x = max(15, int(palm_w * self.config.palm_padding_ratio))
        pad_y = max(15, int(palm_h * self.config.palm_padding_ratio))
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        # enforce minimal size
        if (x_max - x_min) < self.config.palm_min_size:
            cx = (x_min + x_max) // 2
            half = self.config.palm_min_size // 2
            x_min, x_max = max(0, cx - half), min(w, cx + half)
        if (y_max - y_min) < self.config.palm_min_size:
            cy = (y_min + y_max) // 2
            half = self.config.palm_min_size // 2
            y_min, y_max = max(0, cy - half), min(h, cy + half)
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def _resize_with_padding(self, img: np.ndarray, size: Tuple[int, int]):
        h, w = img.shape[:2]
        scale = min(size[0] / h, size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Invalid resize computed")
        resized = cv2.resize(img, (new_w, new_h))
        padded = np.zeros(size, dtype=resized.dtype)
        padded[0:new_h, 0:new_w] = resized
        return padded

    def _builtin_preprocess(self, roi: np.ndarray) -> np.ndarray:
        if roi is None or roi.size == 0:
            raise ValueError("Empty ROI")
        gray = roi.copy() if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = self._resize_with_padding(gray, self.config.size)
        equalized = cv2.equalizeHist(resized)
        return equalized.astype(np.uint8)

    def _preprocess_palm_roi(self, roi: np.ndarray) -> np.ndarray:
        if _HAS_PREPROCESS_HELPER and callable(preprocess_roi_96):
            try:
                return preprocess_roi_96(roi)
            except Exception as e:
                logger.debug("preprocess_roi_96 failed: %s; falling back to builtin", e)
        return self._builtin_preprocess(roi)

    def _get_handedness(self, hand_landmarks) -> str:
        try:
            return "Right" if hand_landmarks.landmark[9].x > hand_landmarks.landmark[0].x else "Left"
        except Exception:
            return "Unknown"

    # ---------- Public detect API ----------
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PalmDetection]]:
        self._validate_frame(frame)
        annotated = frame.copy()
        detections: List[PalmDetection] = []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.hands.process(rgb)
        except Exception as e:
            logger.exception("MediaPipe Hands processing exception: %s", e)
            return annotated, detections

        if not results or not getattr(results, "multi_hand_landmarks", None):
            return annotated, detections

        multi_hands = results.multi_hand_landmarks
        multi_handedness = getattr(results, "multi_handedness", None)

        for idx, hand_landmarks in enumerate(multi_hands):
            try:
                # Unified palm-facing check from utils.palm
                if not is_palm_facing_camera(
                    hand_landmarks,
                    angle_thresh_deg=self.config.palm_normal_angle_thresh_deg,
                    fingertip_req=self.config.fingertip_visibility_req,
                ):
                    logger.debug("Palm not facing camera (skipping)")
                    continue

                x_min, y_min, w_box, h_box = self._compute_palm_bbox(hand_landmarks, frame.shape)
                x0, x1 = max(0, x_min), min(frame.shape[1], x_min + w_box)
                y0, y1 = max(0, y_min), min(frame.shape[0], y_min + h_box)
                if x1 <= x0 or y1 <= y0:
                    logger.debug("Computed empty ROI bounds")
                    continue
                roi = frame[y0:y1, x0:x1]
                if roi is None or roi.size == 0:
                    logger.debug("Empty ROI after slicing; skipping")
                    continue

                palm_roi = self._preprocess_palm_roi(roi)
                handedness = self._get_handedness(hand_landmarks)

                confidence = 0.0
                if multi_handedness and idx < len(multi_handedness):
                    try:
                        score = float(multi_handedness[idx].classification[0].score)
                        if score >= self.config.handedness_confidence_threshold:
                            confidence = score
                    except Exception:
                        logger.debug("Failed to read handedness score; confidence=0.0")

                detections.append(PalmDetection(
                    bbox=(x0, y0, x1 - x0, y1 - y0),
                    palm_roi=palm_roi,
                    landmarks=hand_landmarks,
                    handedness=handedness,
                    confidence=confidence
                ))

                if self.config.debug_draw:
                    color = (0, 255, 0)
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(annotated, f"Palm({handedness})", (x0, max(0, y0 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(annotated, f"Conf:{confidence:.2f}", (x0, min(annotated.shape[0] - 5, y1 + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    try:
                        self.mp_drawing.draw_landmarks(annotated, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    except Exception:
                        logger.debug("Drawing landmarks failed; continuing.")

            except Exception as e:
                logger.exception("Exception handling a detected hand: %s", e)
                continue

        return annotated, detections

    def close(self):
        """Safely close the MediaPipe Hands instance."""
        try:
            if hasattr(self, "hands") and self.hands:
                # Only close if the internal graph is still active
                if getattr(self.hands, "_graph", None) is not None:
                    self.hands.close()
                    logger.info("PalmDetector hands closed")
                else:
                    logger.debug("MediaPipe hands _graph already None; skipping close")
        except Exception:
            logger.exception("Error while closing MediaPipe hands")

def __del__(self):
    try:
        self.close()
    except Exception:
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
