from __future__ import annotations
import logging
import numpy as np
from typing import List, Optional, Sequence, Tuple

from db import get_all_templates, get_user_name
from feature_extraction import PalmFeatureExtractor, calculate_feature_similarity
from detector import PalmDetection

logger = logging.getLogger(__name__)

class PalmVerifier:
    """Verifies palm identity using LBP and geometry features with robust checks."""

    def __init__(self, use_geometry: bool = True, similarity_method: str = "cosine",
                 similarity_threshold: float = 0.92, min_confidence: float = 0.7,
                 min_lbp_variance: float = 0.005):
        """
        Initialize palm verifier.

        Args:
            use_geometry: Whether to include hand geometry features
            similarity_method: Similarity method ('cosine', 'euclidean', 'correlation')
            similarity_threshold: Minimum similarity score for verification
            min_confidence: Minimum MediaPipe confidence for valid detection
            min_lbp_variance: Minimum variance of LBP histogram to accept a palm
        """
        self.use_geometry = use_geometry
        self.similarity_method = similarity_method
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.min_lbp_variance = min_lbp_variance

        # Initialize feature extractor
        self.extractor = PalmFeatureExtractor(use_geometry=use_geometry)

        logger.info(
            "PalmVerifier initialized: geometry=%s, method=%s, threshold=%.2f, min_conf=%.2f, min_var=%.5f",
            use_geometry, similarity_method, similarity_threshold, min_confidence, min_lbp_variance
        )

    def _is_palm_facing_camera(self, landmarks) -> bool:
        p0 = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        p5 = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
        p17 = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])

        normal = np.cross(p5 - p0, p17 - p0)
        return normal[2] < 0

    def _validate_detection(self, detection: PalmDetection) -> bool:
        """Validate that a palm detection is suitable for verification."""
        # Check MediaPipe confidence
        if detection.confidence < self.min_confidence:
            logger.debug("Palm detection confidence too low: %.2f < %.2f",
                         detection.confidence, self.min_confidence)
            return False

        # Check ROI size and quality
        if detection.palm_roi is None or detection.palm_roi.size == 0:
            logger.debug("Empty palm ROI")
            return False

        if detection.palm_roi.shape != (96, 96):
            logger.debug("Invalid palm ROI shape: %s", detection.palm_roi.shape)
            return False

        # Check landmarks availability
        if self.use_geometry and detection.landmarks is None:
            logger.debug("No landmarks available for geometry features")
            return False

        # Check if palm is facing camera
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

    def verify_palm(
        self,
        palm_detections: Sequence[PalmDetection],
        handedness: Optional[str] = None
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        if not palm_detections:
            return False, None, None

        live_features = []
        for det in palm_detections:
            if self._validate_detection(det):
                feats = self.extractor.extract_features(det.palm_roi, det.landmarks)
                if feats is not None and len(feats) > 0:
                    live_features.append(feats)

        if not live_features:
            return False, None, None

        stored_templates = get_all_templates(handedness=handedness)
        if not stored_templates:
            return False, None, None

        best_score = 0.0
        best_user = None
        best_name = None

        for template_id, user_id, template_handedness, stored_feats, _ in stored_templates:
            for live_vec in live_features:
                if len(live_vec) != len(stored_feats):
                    continue
                sim = calculate_feature_similarity(live_vec, stored_feats, self.similarity_method)
                if sim > best_score:
                    best_score = sim
                    best_user = user_id
                    best_name = get_user_name(user_id)

        # Only accept matches above similarity threshold
        if best_score >= self.similarity_threshold:
            return True, best_user, best_name
        else:
            return False, None, None


def verify_palm_with_features(
    palm_detections: Sequence[PalmDetection],
    handedness: Optional[str] = None,
    use_geometry: bool = True,
    similarity_threshold: float = 0.92
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Convenience function to verify palm using feature matching.

    Args:
        palm_detections: List of PalmDetection objects with ROI and landmarks
        handedness: Filter by handedness ("Left" or "Right"), None for any
        use_geometry: Whether to include hand geometry features
        similarity_threshold: Minimum similarity score for verification

    Returns:
        Tuple of (is_match, matched_user_id, matched_name)
    """
    verifier = PalmVerifier(use_geometry=use_geometry, similarity_threshold=similarity_threshold)
    return verifier.verify_palm(palm_detections, handedness)
