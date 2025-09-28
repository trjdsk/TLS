"""
Palm registration module using tiled LBP and optional geometry features.

Key changes:
- Enforce handedness consistency across registration samples
- Single LBP extraction per detection (variance used for gating)
- Save averaged feature vector as template via save_palm_template()
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Sequence, Optional, Any, Tuple, List

import numpy as np
from utils import config

# Safe imports from db
try:
    from db import create_user, save_palm_template
except Exception:
    create_user = None  # type: ignore
    save_palm_template = None  # type: ignore

from feature_extraction import PalmFeatureExtractor
from detector import PalmDetection

# Shared palm utility
try:
    from utils.palm import is_palm_facing_camera
    _HAS_SHARED_PALM_UTIL = True
except Exception:
    _HAS_SHARED_PALM_UTIL = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class RegistrarConfig:
    expected_roi_shape: Tuple[int, int] = config.ROI_SIZE
    min_detection_confidence: float = config.DETECTION_CONFIDENCE
    min_lbp_variance: float = config.VARIANCE_THRESHOLD
    use_geometry: bool = config.DEFAULT_USE_GEOMETRY
    handedness_default: str = "Right"
    feature_type_label: str = "LBP+Geometry"


class PalmRegistrar:
    def __init__(self, config: Optional[RegistrarConfig] = None):
        self.config = config or RegistrarConfig()
        self.use_geometry = bool(self.config.use_geometry)
        self.extractor = PalmFeatureExtractor(use_geometry=self.use_geometry)
        logger.info("PalmRegistrar initialized: %s", self.config)

    def _is_palm_facing(self, landmarks: Any) -> bool:
        if landmarks is None:
            return False
        if _HAS_SHARED_PALM_UTIL:
            try:
                return bool(is_palm_facing_camera(landmarks))
            except Exception:
                pass
        # fallback simple check
        try:
            p0 = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
            p5 = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
            p17 = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])
            v1, v2 = p5 - p0, p17 - p0
            normal = np.cross(v1, v2)
            return bool(normal[2] < 0)
        except Exception:
            return False

    def _validate_detection(self, detection: PalmDetection) -> Tuple[bool, Optional[np.ndarray], Optional[float], Optional[List[float]]]:
        """
        Validate detection and return extracted LBP once.
        Returns (valid, feature_vector, global_variance, tile_variances)
        """
        if detection is None:
            return False, None, None, None

        if detection.palm_roi is None or detection.palm_roi.size == 0:
            return False, None, None, None

        try:
            conf = float(detection.confidence) if detection.confidence is not None else 0.0
        except Exception:
            conf = 0.0
        if conf < self.config.min_detection_confidence:
            return False, None, None, None

        if not self._is_palm_facing(detection.landmarks):
            return False, None, None, None

        try:
            feat_vec, global_var, tile_vars = self.extractor.extract(detection.palm_roi, detection.landmarks)
        except Exception as e:
            logger.debug("LBP extraction failed during validation: %s", e)
            return False, None, None, None

        if feat_vec is None or feat_vec.size == 0:
            return False, None, None, None

        if global_var < self.config.min_lbp_variance:
            logger.debug("Global LBP variance too low: %s < %s", global_var, self.config.min_lbp_variance)
            return False, None, None, None

        return True, feat_vec, global_var, tile_vars

    def register_user_with_features(
        self,
        palm_detections: Sequence[PalmDetection],
        name: str = "Unknown",
        handedness: Optional[str] = None,
    ) -> Tuple[bool, Optional[int]]:
        if not palm_detections:
            logger.error("No palm_detections provided to register")
            return False, None

        if create_user is None or save_palm_template is None:
            logger.error("Database functions not available")
            return False, None

        extracted: List[np.ndarray] = []
        detected_handedness: Optional[str] = None
        for d in palm_detections:
            valid, feat_vec, gvar, tile_vars = self._validate_detection(d)
            if not valid:
                continue
            det_hand = d.handedness or handedness or self.config.handedness_default
            if detected_handedness is None:
                detected_handedness = det_hand
            elif det_hand != detected_handedness:
                logger.warning("Conflicting handedness in registration samples (%s vs %s); skipping sample", det_hand, detected_handedness)
                continue
            extracted.append(feat_vec)

        if not extracted:
            logger.warning("No valid extracted features for registration")
            return False, None

        try:
            stacked = np.stack(extracted, axis=0)
            combined = np.mean(stacked, axis=0)
        except Exception as e:
            logger.exception("Failed to combine features: %s", e)
            return False, None

        try:
            uid = create_user(name)
            if uid is None:
                logger.error("create_user returned None")
                return False, None
            feature_label = self.config.feature_type_label
            tid = save_palm_template(int(uid), detected_handedness or (handedness or self.config.handedness_default), combined, feature_label)
            if tid is None:
                logger.error("save_palm_template failed")
                return False, None
            logger.info("Registered user %s (id=%s) template_id=%s features=%d", name, uid, tid, combined.size)
            return True, int(uid)
        except Exception as e:
            logger.exception("Database error while saving template: %s", e)
            return False, None
