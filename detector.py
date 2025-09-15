"""Palm/hand detection backends.

Defines a unified PalmDetector interface with pluggable backends.
Current backends:
- mediapipe: real-time hand detection with landmarks + bounding boxes
- edgeimpulse: stubbed for future Edge Impulse .eim model integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception as exc:  # pragma: no cover - optional import for backend
    mp = None  # type: ignore
    _logging = logging.getLogger(__name__)
    _logging.debug("mediapipe import failed: %s", exc)


logger = logging.getLogger(__name__)


BackendName = Literal["mediapipe", "edgeimpulse"]


@dataclass
class DetectionResult:
    """Represents a single hand detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: Optional[np.ndarray]  # shape: (21, 2) in pixel coords, or None
    score: Optional[float]


class PalmDetector:
    """Unified detector facade for palm/hand detection.

    Instantiate with a backend name and optional configuration.
    """

    def __init__(self, backend: BackendName = "mediapipe", **kwargs: Any) -> None:
        self.backend: BackendName = backend
        self._impl: Any

        if backend == "mediapipe":
            self._impl = _MediaPipeHandDetector(**kwargs)
        elif backend == "edgeimpulse":
            self._impl = _EdgeImpulsePalmDetectorStub(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Run detection on a BGR frame.

        Returns annotated frame and list of detections.
        """
        return self._impl.detect(frame_bgr)

    def close(self) -> None:
        """Release backend resources if any."""
        close_method = getattr(self._impl, "close", None)
        if callable(close_method):
            close_method()


class _MediaPipeHandDetector:
    """MediaPipe Hands backend implementation."""

    def __init__(self, max_num_hands: int = 2, detection_confidence: float = 0.5, tracking_confidence: float = 0.5, palm_only: bool = False, palm_region_only: bool = False) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for the mediapipe backend")
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._palm_only = palm_only
        self._palm_region_only = palm_region_only

    @staticmethod
    def _is_open_palm(landmarks_xy: np.ndarray) -> bool:
        """Heuristic: consider palm open if 4+ fingers (excluding thumb) are extended.

        Uses relative distance from wrist (0) to finger TIP vs PIP joints.
        This is orientation-agnostic by comparing distances, not raw axes.
        """
        if landmarks_xy.shape != (21, 2):
            return False

        wrist = landmarks_xy[0]
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]

        def dist(a: np.ndarray, b: np.ndarray) -> float:
            v = a.astype(np.float32) - b.astype(np.float32)
            return float(np.linalg.norm(v))

        extended_count = 0
        for tip_idx, pip_idx in finger_pairs:
            tip_farther = dist(landmarks_xy[tip_idx], wrist) > dist(landmarks_xy[pip_idx], wrist) * 1.1
            if tip_farther:
                extended_count += 1

        thumb_extended = dist(landmarks_xy[4], wrist) > dist(landmarks_xy[2], wrist) * 1.05
        if thumb_extended:
            extended_count += 1

        return extended_count >= 4

    @staticmethod
    def _palm_polygon(landmarks_xy: np.ndarray) -> np.ndarray:
        """Construct a polygon approximating the palm area (excluding fingers).

        Uses a subset of palm-base landmarks and returns a convex hull.
        """
        # Landmarks that approximate palm base: wrist + MCP/CMC joints
        indices = [0, 1, 2, 5, 9, 13, 17]
        pts = landmarks_xy[indices]
        import cv2
        hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
        return hull.astype(int)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self._hands.process(frame_rgb)

        detections: List[DetectionResult] = []
        annotated = frame_bgr.copy()

        if results.multi_hand_landmarks:
            image_h, image_w = annotated.shape[:2]
            handedness_list = getattr(results, "multi_handedness", None)

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Collect pixel coordinates
                coords = []
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * image_w)
                    y_px = int(lm.y * image_h)
                    coords.append((x_px, y_px))
                coords_np = np.array(coords, dtype=np.int32)

                # Palm-only detection filter (hand pose)
                if self._palm_only and not self._is_open_palm(coords_np):
                    continue

                # Confidence from handedness if available
                score: Optional[float] = None
                if handedness_list and idx < len(handedness_list):
                    try:
                        score = float(handedness_list[idx].classification[0].score)
                    except Exception:
                        score = None

                import cv2
                if self._palm_region_only:
                    # Compute palm polygon and bbox
                    palm_poly = self._palm_polygon(coords_np)
                    x_min = int(np.min(palm_poly[:, 0]))
                    y_min = int(np.min(palm_poly[:, 1]))
                    x_max = int(np.max(palm_poly[:, 0]))
                    y_max = int(np.max(palm_poly[:, 1]))
                    w = max(0, x_max - x_min)
                    h = max(0, y_max - y_min)

                    # Draw only palm polygon and bbox, no finger landmarks
                    cv2.polylines(annotated, [palm_poly.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)
                    cv2.rectangle(annotated, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 255), 2)
                    if score is not None:
                        cv2.putText(
                            annotated,
                            f"conf: {score:.2f}",
                            (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    detections.append(
                        DetectionResult(
                            bbox=(x_min, y_min, w, h),
                            landmarks=coords_np,
                            score=score,
                        )
                    )
                else:
                    # Full hand landmarks and bbox (includes fingers)
                    self._mp_drawing.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                    )
                    x_min = int(np.min(coords_np[:, 0]))
                    y_min = int(np.min(coords_np[:, 1]))
                    x_max = int(np.max(coords_np[:, 0]))
                    y_max = int(np.max(coords_np[:, 1]))
                    w = max(0, x_max - x_min)
                    h = max(0, y_max - y_min)
                    cv2.rectangle(annotated, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
                    if score is not None:
                        cv2.putText(
                            annotated,
                            f"conf: {score:.2f}",
                            (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    detections.append(
                        DetectionResult(
                            bbox=(x_min, y_min, w, h),
                            landmarks=coords_np,
                            score=score,
                        )
                    )
        else:
            logger.debug("No hands detected in current frame")

        return annotated, detections

    def close(self) -> None:
        self._hands.close()


class _EdgeImpulsePalmDetectorStub:
    """Placeholder for Edge Impulse .eim backend.

    Raises NotImplementedError for detection, but keeps initialization shape.
    """

    def __init__(self, model_path: Optional[str] = None, **_: Any) -> None:
        self.model_path = model_path
        logger.info("Edge Impulse backend selected, but not implemented yet. model_path=%s", model_path)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult]]:
        raise NotImplementedError("Edge Impulse backend is not implemented yet. Provide a .eim model and implement inference.")

    def close(self) -> None:
        return None


__all__ = ["PalmDetector", "DetectionResult"]
