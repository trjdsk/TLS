"""Palm verification module: TFLite palm gating + biometric matching.

Flow:
1) Edge Impulse TFLite int8 classification (palm vs not palm)
2) If palm -> extract deep crease skeleton via shared preprocessing
3) Match features/structures vs registered templates
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from db import get_all_templates, get_user_name
from registration import PalmFeatureExtractor
from preprocessing import preprocess_palm
import tensorflow as tf


class TFLitePalmClassifier:
    """Edge Impulse TFLite int8 classifier wrapper."""

    def __init__(self, model_path: str = "model/tflite-model/tflite_learn_781277_3.tflite") -> None:
        self.model_path = model_path
        self._load()

    def _load(self) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

    def _to_int8_input(self, gray_96: np.ndarray) -> np.ndarray:
        # Expect uint8 grayscale 96x96 → int8 centered
        roi_norm = gray_96.astype(np.float32) / 255.0
        roi_int8 = ((roi_norm - 0.5) * 255).astype(np.int8)
        return np.expand_dims(roi_int8, axis=(0, -1))

    def infer(self, gray_96: np.ndarray) -> Tuple[bool, float]:
        input_tensor = self._to_int8_input(gray_96)
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_index)[0]
        pred_index = int(np.argmax(output_data))
        confidence = float(output_data[pred_index]) / 128.0
        is_palm = (pred_index == 1)
        return is_palm, confidence

logger = logging.getLogger(__name__)


class PalmVerifier:
    """Verifies palm identity using feature matching."""
    
    def __init__(self, feature_type: str = "ORB", ratio_threshold: float = 0.75, 
                 ransac_threshold: float = 5.0, min_matches: int = 10):
        """
        Initialize palm verifier.
        
        Args:
            feature_type: "ORB", "SIFT", or "SURF"
            ratio_threshold: Lowe's ratio test threshold (0.75 recommended)
            ransac_threshold: RANSAC threshold for homography estimation
            min_matches: Minimum number of good matches required
        """
        self.feature_type = feature_type.upper()
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        
        # Initialize feature extractor
        self.extractor = PalmFeatureExtractor(feature_type=feature_type)
        # Initialize classifier for gating
        self.classifier = TFLitePalmClassifier()
        
        # Initialize matcher based on feature type
        self._init_matcher()
        
        logger.info("PalmVerifier initialized with %s (ratio=%.2f, ransac=%.1f, min_matches=%d)", 
                   feature_type, ratio_threshold, ransac_threshold, min_matches)
    
    def _init_matcher(self):
        """Initialize feature matcher."""
        if self.feature_type == "ORB":
            # Use Hamming distance for ORB (binary features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use L2 distance for SIFT/SURF (float features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def _apply_ratio_test(self, matches: List) -> List:
        """
        Apply Lowe's ratio test to filter good matches.
        
        Args:
            matches: List of matches from knnMatch
            
        Returns:
            List of good matches
        """
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def _apply_ransac_filter(self, keypoints1: List, keypoints2: List, 
                           matches: List) -> Tuple[List, Optional[np.ndarray]]:
        """
        Apply RANSAC to filter matches using homography estimation.
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Tuple of (filtered_matches, homography_matrix)
        """
        if len(matches) < 4:
            return matches, None
        
        # Extract matched points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            # Find homography using RANSAC
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, self.ransac_threshold)
            
            if mask is not None:
                # Filter matches using RANSAC mask
                filtered_matches = [matches[i] for i in range(len(matches)) if mask[i]]
                return filtered_matches, homography
            else:
                return matches, None
                
        except Exception as e:
            logger.warning("RANSAC homography estimation failed: %s", e)
            return matches, None
    
    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                      keypoints1: List, keypoints2: List) -> Tuple[int, float]:
        """
        Match features between two sets of descriptors.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            
        Returns:
            Tuple of (num_good_matches, match_ratio)
        """
        if descriptors1 is None or descriptors2 is None:
            return 0, 0.0
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return 0, 0.0
        
        try:
            # Find matches using knnMatch
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            # Apply ratio test
            good_matches = self._apply_ratio_test(matches)
            
            if len(good_matches) < self.min_matches:
                return len(good_matches), 0.0
            
            # Apply RANSAC filtering
            filtered_matches, homography = self._apply_ransac_filter(
                keypoints1, keypoints2, good_matches)
            
            # Calculate match ratio
            match_ratio = len(filtered_matches) / max(len(descriptors1), len(descriptors2))
            
            logger.debug("Feature matching: %d/%d good matches, ratio=%.3f", 
                        len(filtered_matches), len(good_matches), match_ratio)
            
            return len(filtered_matches), match_ratio
            
        except Exception as e:
            logger.error("Feature matching failed: %s", e)
            return 0, 0.0
    
    def verify_palm(self, palm_crops: Sequence[np.ndarray], handedness: Optional[str] = None,
                   match_threshold: float = 0.15, palm_threshold: float = 0.8) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Verify palm against stored templates.
        
        Args:
            palm_crops: List of 96x96 grayscale palm crops
            handedness: Filter by handedness ("Left" or "Right"), None for any
            match_threshold: Minimum match ratio for verification
            
        Returns:
            Tuple of (is_match, matched_user_id, matched_name)
        """
        if not palm_crops:
            return False, None, None
        
        try:
            # Gate by TFLite palm classifier and extract features from live palm crops
            live_features = []
            for crop in palm_crops:
                # crop is 96x96 grayscale. Run EI TFLite gate first
                is_palm, conf = self.classifier.infer(crop)
                logger.info("TFLite palm gate: is_palm=%s conf=%.2f", str(is_palm), conf)
                if not is_palm or conf < palm_threshold:
                    continue

                # Apply biometric crease preprocessing for reproducibility
                crease = preprocess_palm(crop)
                keypoints, descriptors = self.extractor.extract_features(crease)
                if descriptors is not None and len(descriptors) > 0:
                    live_features.append((keypoints, descriptors))
            
            if not live_features:
                logger.warning("No features extracted from live palm crops")
                return False, None, None
            
            # Get all stored templates
            stored_templates = get_all_templates(handedness=handedness)
            
            if not stored_templates:
                logger.warning("No stored templates found for verification")
                return False, None, None
            
            best_match_score = 0.0
            best_match_user = None
            best_match_name = None
            
            # Compare with each stored template
            for template_id, user_id, template_handedness, stored_descriptors in stored_templates:
                # Calculate average match score across all live crops
                total_matches = 0
                total_ratio = 0.0
                valid_comparisons = 0
                
                for live_keypoints, live_descriptors in live_features:
                    num_matches, match_ratio = self.match_features(
                        live_descriptors, stored_descriptors, 
                        live_keypoints, []  # No keypoints for stored descriptors
                    )
                    
                    if num_matches > 0:
                        total_matches += num_matches
                        total_ratio += match_ratio
                        valid_comparisons += 1
                
                if valid_comparisons > 0:
                    avg_ratio = total_ratio / valid_comparisons
                    avg_matches = total_matches / valid_comparisons
                    
                    # Combined score: ratio + normalized match count
                    combined_score = avg_ratio + (avg_matches / 100.0)
                    
                    logger.info("User %d: avg_ratio=%.3f, avg_matches=%.1f, combined=%.3f", 
                               user_id, avg_ratio, avg_matches, combined_score)
                    
                    if combined_score > best_match_score and avg_ratio >= match_threshold:
                        best_match_score = combined_score
                        best_match_user = user_id
                        best_match_name = get_user_name(user_id)
            
            # Determine if verification is successful
            is_match = (best_match_user is not None and best_match_score >= match_threshold)
            
            if is_match:
                logger.info("Verification successful: User %d (%s), score=%.3f", 
                           best_match_user, best_match_name, best_match_score)
            else:
                logger.info("Verification failed: best_score=%.3f, threshold=%.3f", 
                           best_match_score, match_threshold)
            
            return is_match, best_match_user, best_match_name
            
        except Exception as e:
            logger.error("Palm verification failed: %s", e)
            return False, None, None


def verify_palm(embeddings_list: Sequence[np.ndarray], db_path: str = "palms.db", 
                similarity_threshold: float = 0.92, min_avg_similarity: float = 0.90,
                min_peak_similarity: float = 0.95, handedness: Optional[str] = None,
                max_variance: float = 0.05) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Legacy function for backward compatibility.
    Converts embeddings to features and verifies palm.
    """
    logger.warning("Using legacy verify_palm function - consider using PalmVerifier")
    
    # This is a temporary solution - should be replaced with proper feature matching
    # For now, return a dummy result
    return False, None, None


def verify_palm_with_features(palm_crops: Sequence[np.ndarray], handedness: Optional[str] = None,
                             feature_type: str = "ORB", match_threshold: float = 0.15) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Convenience function to verify palm using feature matching.
    
    Args:
        palm_crops: List of 96x96 grayscale palm crops
        handedness: Filter by handedness ("Left" or "Right"), None for any
        feature_type: "ORB", "SIFT", or "SURF"
        match_threshold: Minimum match ratio for verification
        
    Returns:
        Tuple of (is_match, matched_user_id, matched_name)
    """
    verifier = PalmVerifier(feature_type=feature_type)
    return verifier.verify_palm(palm_crops, handedness, match_threshold)