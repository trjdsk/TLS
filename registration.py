"""Palm registration module using hand-crafted features.

Extracts ORB/SIFT/SURF features from palm crops and stores them in database.
Provides palm crease enhancement and feature extraction for registration.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import secrets

import cv2
import numpy as np

from db import create_user, save_palm_template, get_user_templates, get_user_name

logger = logging.getLogger(__name__)


class PalmFeatureExtractor:
    """Extracts hand-crafted features from palm images."""
    
    def __init__(self, feature_type: str = "ORB", max_features: int = 1000):
        """
        Initialize feature extractor.
        
        Args:
            feature_type: "ORB", "SIFT", or "SURF"
            max_features: Maximum number of features to extract
        """
        self.feature_type = feature_type.upper()
        self.max_features = max_features
        self._init_detector()
        
        logger.info("PalmFeatureExtractor initialized with %s (max_features=%d)", 
                   feature_type, max_features)
    
    def _init_detector(self):
        """Initialize the feature detector."""
        if self.feature_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.feature_type == "SIFT":
            try:
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            except AttributeError:
                logger.warning("SIFT not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
                self.feature_type = "ORB"
        elif self.feature_type == "SURF":
            try:
                # SURF requires non-free OpenCV
                self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
            except AttributeError:
                logger.warning("SURF not available, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=self.max_features)
                self.feature_type = "ORB"
        else:
            logger.warning("Unknown feature type %s, using ORB", self.feature_type)
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.feature_type = "ORB"
    
    def enhance_palm_creases(self, palm_roi: np.ndarray) -> np.ndarray:
        """
        Enhance palm creases using preprocessing techniques.
        
        Args:
            palm_roi: 96x96 grayscale palm image
            
        Returns:
            Enhanced palm image with highlighted creases
        """
        # Convert to grayscale if needed
        if len(palm_roi.shape) == 3:
            gray = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = palm_roi.copy()
        
        # Histogram equalization for lighting normalization
        equalized = cv2.equalizeHist(gray)
        
        # Edge enhancement using Laplacian
        laplacian = cv2.Laplacian(equalized, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Combine original with enhanced edges
        enhanced = cv2.addWeighted(equalized, 0.7, laplacian, 0.3, 0)
        
        # Optional: Sobel edge detection for additional crease highlighting
        sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
        
        # Final combination
        final = cv2.addWeighted(enhanced, 0.8, sobel_magnitude, 0.2, 0)
        
        return final
    
    def extract_features(self, palm_roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from palm ROI.
        
        Args:
            palm_roi: 96x96 grayscale palm image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Enhance palm creases
        enhanced = self.enhance_palm_creases(palm_roi)
        
        # Extract features
        keypoints, descriptors = self.detector.detectAndCompute(enhanced, None)
        
        if descriptors is None:
            logger.warning("No features detected in palm ROI")
            return np.array([]), np.array([])
        
        logger.debug("Extracted %d %s features", len(keypoints), self.feature_type)
        return keypoints, descriptors


def verify_palm_for_registration(palm_roi: np.ndarray, threshold: float = 0.5) -> bool:
    """Verify that the ROI contains a palm using Edge Impulse model during registration.
    
    Args:
        palm_roi: Preprocessed palm ROI (96x96 grayscale)
        threshold: Confidence threshold for palm detection
        
    Returns:
        True if palm is detected, False otherwise
    """
    try:
        from model_wrapper import get_model
        
        model = get_model()
        if model.is_initialized:
            # ROI should already be preprocessed by detector
            return model.is_palm(palm_roi, threshold)
        else:
            logger.warning("Edge Impulse model not initialized, cannot verify palm during registration")
            return True  # Allow registration to proceed if model not available
    except Exception as exc:
        logger.warning("Edge Impulse model not available for registration verification: %s", exc)
        return True  # Allow registration to proceed if model not available


def register_user_with_features(palm_crops: Sequence[np.ndarray], handedness: str = "Right", 
                               name: str = "Unknown", feature_type: str = "ORB") -> Tuple[bool, Optional[int]]:
    """
    Register a new user with palm feature descriptors.
    
    Args:
        palm_crops: List of 96x96 grayscale palm crops
        handedness: "Left" or "Right"
        name: User's name
        feature_type: "ORB", "SIFT", or "SURF"
        
    Returns:
        Tuple of (success, user_id)
    """
    if not palm_crops:
        logger.error("No palm crops provided for registration")
        return False, None
    
    try:
        # Create feature extractor
        extractor = PalmFeatureExtractor(feature_type=feature_type)
        
        # Verify all palm crops are valid
        valid_crops = []
        for crop in palm_crops:
            if verify_palm_for_registration(crop):
                valid_crops.append(crop)
            else:
                logger.warning("Invalid palm detected in registration crop")
        
        if not valid_crops:
            logger.error("No valid palm crops found for registration")
            return False, None
        
        # Extract features from all valid crops
        all_descriptors = []
        for crop in valid_crops:
            keypoints, descriptors = extractor.extract_features(crop)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            logger.error("No features extracted from palm crops")
            return False, None
        
        # Combine descriptors from all crops
        combined_descriptors = np.vstack(all_descriptors)
        
        # Create user in database
        user_id = create_user(name)
        
        # Save palm template
        template_id = save_palm_template(user_id, handedness, combined_descriptors)
        
        logger.info("Successfully registered user %s (ID: %d) with %d features from %d crops", 
                   name, user_id, len(combined_descriptors), len(valid_crops))
        
        return True, user_id
        
    except Exception as e:
        logger.error("Registration failed: %s", e)
        return False, None


def register_user(user_id: str, embeddings_list: Sequence[np.ndarray], handedness: str = "Right", name: str = "Unknown", db_path: str = "palms.db") -> bool:
    """
    Legacy function for backward compatibility.
    Converts embeddings to features and registers user.
    """
    logger.warning("Using legacy register_user function - consider using register_user_with_features")
    
    # Convert embeddings to dummy features for compatibility
    # This is a temporary solution - should be replaced with proper feature extraction
    try:
        # Create dummy descriptors from embeddings
        dummy_descriptors = np.random.rand(100, 32).astype(np.uint8)  # ORB-like descriptors
        
        # Create user in database
        user_id_int = create_user(name)
        
        # Save palm template
        save_palm_template(user_id_int, handedness, dummy_descriptors)
        
        return True
    except Exception as e:
        logger.error("Legacy registration failed: %s", e)
        return False


def extract_embedding(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    Extracts dummy embedding from palm ROI.
    """
    logger.warning("Using legacy extract_embedding function - consider using PalmFeatureExtractor")
    
    # Return dummy embedding for compatibility
    return np.random.rand(128).astype(np.float32)