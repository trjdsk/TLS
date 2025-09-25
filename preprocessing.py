"""Shared preprocessing utilities for palm detection and biometrics."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects

logger = logging.getLogger(__name__)


def preprocess_roi_96(roi_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR ROI to grayscale and resize to 96x96 (uint8).

    Args:
        roi_bgr: BGR input image crop.

    Returns:
        Grayscale 96x96 image as uint8.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        raise ValueError("Empty ROI provided to preprocess_roi_96")

    try:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr
        resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)
    except Exception as exc:
        logger.error("Failed preprocessing ROI to 96x96: %s", exc)
        raise


def preprocess_palm(img: np.ndarray) -> np.ndarray:
    """Enhance only deep palm creases as binary (0/255) uint8 mask.

    Steps: CLAHE -> Canny -> Morph Open/Close -> Skeletonize -> Remove small objects.
    Falls back to Canny edges if result is empty.
    """
    if img is None or img.size == 0:
        raise ValueError("Empty image provided to preprocess_palm")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast normalization
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_eq = clahe.apply(img)

    # Edge detection
    edges = cv2.Canny(img_eq, 50, 120)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Skeletonize and remove small objects
    skel = skeletonize(closed > 0)
    cleaned = remove_small_objects(skel, min_size=20, connectivity=2)

    if np.sum(cleaned) == 0:
        cleaned = edges > 0

    return (cleaned.astype(np.uint8) * 255)


