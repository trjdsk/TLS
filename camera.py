"""Camera input module with palm detection integration.

Provides camera interface that integrates with palm detection system.
Returns annotated frames with palm bounding boxes and palm crops for processing.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union, List
import os
import time

import cv2
import numpy as np

from detector import PalmDetector, PalmDetection

logger = logging.getLogger(__name__)


class VideoCapture:
    """Abstraction for video input with palm detection.

    Currently supports laptop webcam via integer index and URL streams.
    Integrates with palm detection to provide annotated frames and palm crops.

    Attributes:
        source: Camera index (int) or stream URL (str).
        width: Optional desired frame width.
        height: Optional desired frame height.
        detector: PalmDetector instance for palm detection.
    """

    def __init__(self, source: Union[int, str] = 0, width: Optional[int] = None, 
                 height: Optional[int] = None, buffer_size: Optional[int] = None,
                 detector: Optional[PalmDetector] = None,
                 save_snaps: bool = False, snaps_dir: Optional[str] = None) -> None:
        self.source: Union[int, str] = source
        self.width: Optional[int] = width
        self.height: Optional[int] = height
        self.buffer_size: Optional[int] = buffer_size
        self.detector: Optional[PalmDetector] = detector
        self._cap: Optional[cv2.VideoCapture] = None
        self.save_snaps: bool = save_snaps
        self.snaps_dir: str = snaps_dir or "snapshots"
        # Snapshot delay mechanism
        self._hand_detected_time: Optional[float] = None
        self._snapshot_delay: float = 0.5  # 0.5 seconds delay
        self._snapshot_taken: bool = False

    def open(self) -> None:
        """Open the video source and apply optional resolution settings."""
        logger.debug("Opening video source: %s", self.source)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Apply optional resolution
        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        # Apply optional buffer size (best-effort; backend-dependent)
        if self.buffer_size is not None:
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.buffer_size))
            except Exception:
                logger.debug("CAP_PROP_BUFFERSIZE not supported by backend")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source.

        Returns:
            A tuple of (ret, frame). ret is True if a frame was read successfully.
        """
        if self._cap is None:
            raise RuntimeError("VideoCapture is not opened. Call open() first.")
        ret, frame = self._cap.read()
        if not ret:
            logger.debug("Failed to read frame from source: %s", self.source)
        return ret, frame

    def get_palm_frame(self) -> Tuple[bool, Optional[np.ndarray], List[PalmDetection]]:
        """
        Get frame with palm detection and return annotated frame + palm detections.
        
        Returns:
            Tuple of (success, annotated_frame, palm_detections)
            - success: True if frame was read successfully
            - annotated_frame: Frame with palm bounding boxes and annotations
            - palm_detections: List of PalmDetection objects with crops and landmarks
        """
        ret, frame = self.read()
        if not ret or frame is None:
            return False, None, []
        
        if self.detector is None:
            # No detector available, return original frame
            return True, frame, []
        
        try:
            # Run palm detection
            annotated_frame, detections = self.detector.detect(frame)

            # Optionally save snapshots of each detection (original ROI and 96x96)
            # with 0.5s delay after hand detection
            if self.save_snaps and detections:
                current_time = time.time()
                
                # If this is the first detection or enough time has passed since last snapshot
                if self._hand_detected_time is None:
                    self._hand_detected_time = current_time
                    self._snapshot_taken = False
                    logger.debug("Hand detected, starting 0.5s delay timer")
                elif not self._snapshot_taken and (current_time - self._hand_detected_time) >= self._snapshot_delay:
                    # Delay has passed, take snapshot
                    try:
                        os.makedirs(self.snaps_dir, exist_ok=True)
                        ts_ms = int(current_time * 1000)
                        for idx, det in enumerate(detections):
                            x, y, w, h = det.bbox
                            x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
                            orig_roi = frame[y:y2, x:x2]
                            hand = det.handedness or "Unknown"
                            base = f"{ts_ms}_{idx}_{hand}"
                            # Save original ROI (BGR)
                            try:
                                cv2.imwrite(os.path.join(self.snaps_dir, f"roi_{base}.png"), orig_roi)
                            except Exception:
                                logger.debug("Failed to save original ROI snapshot")
                            # Save 96x96 grayscale
                            try:
                                cv2.imwrite(os.path.join(self.snaps_dir, f"roi96_{base}.png"), det.palm_roi)
                            except Exception:
                                logger.debug("Failed to save 96x96 ROI snapshot")
                        self._snapshot_taken = True
                        logger.debug("Snapshot taken after 0.5s delay")
                    except Exception:
                        logger.debug("Snapshot saving failed")
            elif not detections and self._hand_detected_time is not None:
                # No hand detected, reset the timer
                self._hand_detected_time = None
                self._snapshot_taken = False
            
            return True, annotated_frame, detections
            
        except Exception as e:
            logger.error("Palm detection failed: %s", e)
            return True, frame, []

    def release(self) -> None:
        """Release the video resource."""
        if self._cap is not None:
            logger.debug("Releasing video source: %s", self.source)
            self._cap.release()
            self._cap = None


def get_palm_frame(cap: VideoCapture) -> Tuple[Optional[np.ndarray], List[PalmDetection]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        cap: VideoCapture instance with palm detector
        
    Returns:
        Tuple of (annotated_frame, palm_detections)
    """
    success, annotated_frame, palm_detections = cap.get_palm_frame()
    if success:
        return annotated_frame, palm_detections
    else:
        return None, []


__all__ = ["VideoCapture", "get_palm_frame"]
