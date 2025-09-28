"""Centralized palm-facing heuristics used by detector, verifier, registrar."""

from __future__ import annotations
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


def is_palm_facing_camera(
    landmarks: Any,
    handedness: str = "Right",
    angle_thresh_deg: float = 60.0,
    fingertip_req: int = 4
) -> bool:
    """
    Estimate if the palm is facing the camera using combined heuristics.

    Heuristics:
      1. Palm-normal angle test:
         Uses cross product of (5-0) and (17-0) vectors to approximate palm orientation.
      2. Fingertip visibility test:
         Checks if fingertip z < preceding joint z for majority of fingers.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList or equivalent object with .landmark
        handedness: "Left" or "Right"
        angle_thresh_deg: Maximum deviation from camera axis for palm-normal
        fingertip_req: Minimum number of fingertips that must pass z-test

    Returns:
        True if palm is likely facing the camera, False otherwise
    """
    if landmarks is None:
        logger.debug("Palm facing check failed: landmarks is None")
        return False

    try:
        # Basic landmark access
        def get_coord(idx: int) -> np.ndarray:
            lm = landmarks.landmark[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=float)

        wrist = get_coord(0)
        index_mcp = get_coord(5)
        pinky_mcp = get_coord(17)
        middle_mcp = get_coord(9)
    except Exception as e:
        logger.debug("Palm facing check failed: invalid landmark access (%s)", e)
        return False

    # --- Palm-normal angle heuristic ---
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    normal_ok = False
    if norm_len > 1e-6:
        normal /= norm_len
        cam_vec = np.array([0.0, 0.0, -1.0])
        cosang = float(np.dot(normal, cam_vec))
        angle_deg = float(np.degrees(np.arccos(np.clip(abs(cosang), -1.0, 1.0))))
        # Invert for left hand
        if handedness.lower().startswith("left"):
            angle_deg = 180.0 - angle_deg
        normal_ok = angle_deg <= angle_thresh_deg

    # --- Fingertip visibility heuristic ---
    fingertip_indices = [4, 8, 12, 16, 20]
    visible = 0
    for idx in fingertip_indices:
        try:
            if landmarks.landmark[idx].z < landmarks.landmark[idx - 1].z:
                visible += 1
        except Exception:
            continue
    fingertip_ok = visible >= fingertip_req

    logger.debug(
        "Palm facing check: handedness=%s, normal_ok=%s, fingertip_ok=%s (visible=%d)",
        handedness, normal_ok, fingertip_ok, visible
    )

    return normal_ok or fingertip_ok
