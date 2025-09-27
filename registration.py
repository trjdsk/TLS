"""Palm registration module using LBP and hand geometry features.

Extracts Local Binary Pattern (LBP) features and optional hand geometry features
from palm crops and stores them in database for biometric verification.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from db import create_user, save_palm_template
from feature_extraction import PalmFeatureExtractor
from detector import PalmDetection

logger = logging.getLogger(__name__)


class PalmRegistrar:
    """Handles palm registration with feature extraction and validation."""

    def __init__(self, use_geometry: bool = True, min_confidence: float = 0.7,
                 min_lbp_variance: float = 0.005):
        """
        Initialize registrar.

        Args:
            use_geometry: Whether to include hand geometry features
            min_confidence: Minimum MediaPipe confidence to accept a palm
            min_lbp_variance: Minimum variance of LBP histogram to accept a palm
        """
        self.use_geometry = use_geometry
        self.min_confidence = min_confidence
        self.min_lbp_variance = min_lbp_variance
        self.extractor = PalmFeatureExtractor(use_geometry=use_geometry)

    def _is_palm_facing_camera(self, landmarks) -> bool:
        """
        Estimate if the palm is facing the camera using landmarks.
        """
        import numpy as np
        p0 = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        p5 = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
        p17 = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])

        v1 = p5 - p0
        v2 = p17 - p0
        normal = np.cross(v1, v2)

        # Negative z toward camera
        return normal[2] < 0

    def _validate_detection(self, detection: PalmDetection) -> bool:
        """Validate a palm detection for registration."""
        if detection.confidence < self.min_confidence:
            logger.debug("Palm detection confidence too low: %.2f < %.2f",
                         detection.confidence, self.min_confidence)
            return False

        if detection.palm_roi is None or detection.palm_roi.size == 0:
            logger.debug("Empty palm ROI")
            return False

        if detection.palm_roi.shape != (96, 96):
            logger.debug("Invalid palm ROI shape: %s", detection.palm_roi.shape)
            return False

        if detection.landmarks is None:
            logger.debug("No landmarks available for geometry features")
            return False

        if not self._is_palm_facing_camera(detection.landmarks):
            logger.debug("Palm not facing camera, rejecting detection")
            return False

        # Check LBP variance
        lbp_features = self.extractor.lbp_extractor.extract_lbp_features(detection.palm_roi)
        if np.var(lbp_features) < self.min_lbp_variance:
            logger.debug("LBP variance too low: %.5f < %.5f, rejecting detection",
                         np.var(lbp_features), self.min_lbp_variance)
            return False

        return True

    def register_user_with_features(self, palm_detections: Sequence[PalmDetection],
                                    name: str = "Unknown", handedness: str = "Right") -> Tuple[bool, Optional[int]]:
        """
        Register a new user with validated palm feature vectors.

        Args:
            palm_detections: List of PalmDetection objects
            name: User name
            handedness: "Left" or "Right"

        Returns:
            Tuple of (success, user_id)
        """
        if not palm_detections:
            logger.error("No palm detections provided for registration")
            return False, None

        # Validate detections
        valid_detections = [d for d in palm_detections if self._validate_detection(d)]
        if not valid_detections:
            logger.error("No valid palm detections found for registration")
            return False, None

        # Extract features from valid detections
        all_features = []
        for detection in valid_detections:
            features = self.extractor.extract_features(detection.palm_roi, detection.landmarks)
            if features is not None and len(features) > 0:
                all_features.append(features)

        if not all_features:
            logger.error("No features extracted from valid palm detections")
            return False, None

        # Average features for robust template
        combined_features = np.mean(all_features, axis=0)

        # Create user in database
        user_id = create_user(name)
        feature_type = "LBP+Geometry" if self.use_geometry else "LBP"

        # Save palm template
        save_palm_template(user_id, handedness, combined_features, feature_type)

        logger.info("Successfully registered user %s (ID: %d) with %d features from %d detections",
                    name, user_id, len(combined_features), len(valid_detections))

        return True, user_id
