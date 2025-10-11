"""Shared preprocessing utilities for palm detection and biometrics.

Provides basic image preprocessing functions for palm ROI normalization.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_roi_96(roi_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR ROI to grayscale and resize to 96x96 (uint8).
    Optimized for JPEG inputs from ESP32-CAM.

    Args:
        roi_bgr: BGR input image crop.

    Returns:
        Grayscale 96x96 image as uint8.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("Empty ROI provided to preprocess_roi_96")

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
        
        # Optimize for JPEG inputs: apply denoising to reduce compression artifacts
        if roi_bgr.ndim == 3:  # Only for color images (likely from JPEG)
            gray = cv2.medianBlur(gray, 3)  # Remove JPEG compression noise
        
        resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)
    except Exception as exc:
        logger.error("Failed preprocessing ROI to 96x96: %s", exc)
        raise


def preprocess_palm(img: np.ndarray) -> np.ndarray:
    """Basic palm preprocessing with histogram equalization.

    Args:
        img: Input palm image (96x96 grayscale)
        
    Returns:
        Preprocessed palm image with histogram equalization
    """
    if img is None or img.size == 0:
        raise ValueError("Empty image provided to preprocess_palm")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization for consistent lighting
    equalized = cv2.equalizeHist(img)
    
    return equalized


