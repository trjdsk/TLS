"""
Palm verification module.

Key changes:
- Use tiled-LBP extractor
- Enforce handedness match between detection and stored templates
- Use LBP variance gating and cosine similarity for final decision
- Provide helper verify_palm_with_features() scanning all users
"""

from __future__ import annotations
import logging
from typing import Sequence, Optional, Tuple, List, Any

import numpy as np
from utils import config
from feature_extraction import PalmFeatureExtractor, cosine_similarity
from detector import PalmDetection

# DB interface
try:
    import db
    from db import load_user_templates
except Exception:
    db = None  # type: ignore
    load_user_templates = None  # type: ignore

# Shared palm util
try:
    from utils.palm import is_palm_facing_camera
    _HAS_SHARED_PALM_UTIL = True
except Exception:
    _HAS_SHARED_PALM_UTIL = False

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PalmVerifier:
    def __init__(self, use_geometry: bool = config.DEFAULT_USE_GEOMETRY,
                 min_lbp_variance: float = config.VARIANCE_THRESHOLD,
                 similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD):
        self.use_geometry = bool(use_geometry)
        self.min_lbp_variance = float(min_lbp_variance)
        self.similarity_threshold = float(similarity_threshold)
        self.extractor = PalmFeatureExtractor(use_geometry=self.use_geometry)
        logger.info("PalmVerifier initialized: geometry=%s min_var=%s sim_thresh=%s",
                    self.use_geometry, self.min_lbp_variance, self.similarity_threshold)

    def _is_palm_facing(self, landmarks: Any) -> bool:
        if landmarks is None:
            return False
        if _HAS_SHARED_PALM_UTIL:
            try:
                return bool(is_palm_facing_camera(landmarks))
            except Exception:
                pass
        try:
            p0 = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
            p5 = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
            p17 = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])
            v1, v2 = p5 - p0, p17 - p0
            normal = np.cross(v1, v2)
            return bool(normal[2] < 0)
        except Exception:
            return False

    def _validate_and_extract(self, detection: PalmDetection) -> Optional[Tuple[np.ndarray, float, List[float]]]:
        """Return (feat_vec, global_var, tile_vars) or None if invalid."""
        if detection is None or detection.palm_roi is None or detection.palm_roi.size == 0:
            return None
        if not self._is_palm_facing(detection.landmarks):
            return None
        try:
            feats, gvar, tile_vars = self.extractor.extract(detection.palm_roi, detection.landmarks)
        except Exception as e:
            logger.debug("Extraction failed: %s", e)
            return None
        if feats is None or feats.size == 0 or gvar < self.min_lbp_variance:
            logger.debug("Rejecting due to low variance: %s < %s", gvar, self.min_lbp_variance)
            return None
        return feats, gvar, tile_vars

    def verify_user(self, palm_detections: Sequence[PalmDetection], user_id: int) -> Tuple[bool, float]:
        """
        Verify palm detections against templates of a specific user.
        Returns (success, best_score).
        """
        if not palm_detections or load_user_templates is None:
            logger.error("Invalid input or DB loader unavailable")
            return False, 0.0
        try:
            templates = load_user_templates(int(user_id))
        except Exception as e:
            logger.exception("Failed to load templates for user %s: %s", user_id, e)
            return False, 0.0
        if not templates:
            logger.debug("No templates for user %s", user_id)
            return False, 0.0

        valid_extracted = []
        for d in palm_detections:
            res = self._validate_and_extract(d)
            if res is not None:
                feats, gvar, tile_vars = res
                valid_extracted.append((d, feats, gvar, tile_vars))

        if not valid_extracted:
            logger.debug("No valid detections for verification")
            return False, 0.0

        best_score = -1.0
        for tpl_id, tpl_handedness, tpl_features, tpl_ftype in templates:
            tpl_vec = np.asarray(tpl_features).ravel()
            if tpl_vec.size == 0:
                continue
            for d, feats, gvar, tile_vars in valid_extracted:
                det_hand = (d.handedness or "").strip()
                if tpl_handedness and det_hand and tpl_handedness != det_hand:
                    logger.debug("Handedness mismatch: det=%s tpl=%s -> skip", det_hand, tpl_handedness)
                    continue
                sim = cosine_similarity(feats, tpl_vec)
                if sim > best_score:
                    best_score = sim

        if best_score < 0:
            return False, 0.0
        success = best_score >= self.similarity_threshold
        logger.info("User %s verification result: success=%s score=%.3f", user_id, success, best_score)
        return success, float(best_score)


def verify_palm_with_features(
    palm_detections: Sequence[PalmDetection],
    handedness: Optional[str] = None,
    use_geometry: bool = config.DEFAULT_USE_GEOMETRY,
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Iterate through all users and return first/best match.
    Returns: (is_match, matched_user_id, matched_user_name)
    """
    verifier = PalmVerifier(use_geometry=use_geometry, similarity_threshold=similarity_threshold)
    if db is None:
        logger.error("Database module not available")
        return False, None, None
    try:
        cur = db.db.cursor()
        cur.execute("SELECT id, name FROM users")
        users = cur.fetchall()
    except Exception as e:
        logger.exception("Failed to enumerate users: %s", e)
        return False, None, None

    best_score = -1.0
    best_user_id: Optional[int] = None
    best_user_name: Optional[str] = None

    for row in users:
        try:
            uid = int(row["id"])
            uname = row["name"]
        except Exception:
            continue
        ok, score = verifier.verify_user(palm_detections, uid)
        if ok and score > best_score:
            best_score = score
            best_user_id = uid
            best_user_name = uname
        elif not ok and score > best_score:
            best_score = score

    if best_user_id is not None and best_score >= similarity_threshold:
        return True, best_user_id, best_user_name
    return False, None, None
