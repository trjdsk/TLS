"""Camera input module.

Provides a simple wrapper over OpenCV's VideoCapture to support
webcam indices and network stream URLs (e.g., ESP32 streams in future).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class VideoCapture:
    """Abstraction for video input.

    Currently supports laptop webcam via integer index and URL streams.

    Attributes:
        source: Camera index (int) or stream URL (str).
        width: Optional desired frame width.
        height: Optional desired frame height.
    """

    def __init__(self, source: Union[int, str] = 0, width: Optional[int] = None, height: Optional[int] = None, buffer_size: Optional[int] = None) -> None:
        self.source: Union[int, str] = source
        self.width: Optional[int] = width
        self.height: Optional[int] = height
        self.buffer_size: Optional[int] = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None

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

    def release(self) -> None:
        """Release the video resource."""
        if self._cap is not None:
            logger.debug("Releasing video source: %s", self.source)
            self._cap.release()
            self._cap = None


__all__ = ["VideoCapture"]
