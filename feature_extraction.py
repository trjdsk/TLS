"""Feature extraction module for palmprint biometrics.

Implements Local Binary Pattern (LBP) features and optional hand geometry features
for robust palmprint recognition using MediaPipe landmarks.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import exposure

logger = logging.getLogger(__name__)


class LBPFeatureExtractor:
    """Local Binary Pattern feature extractor for palmprint recognition."""
    
    def __init__(self, radius: int = 3, n_points: int = 24, method: str = 'uniform'):
        """
        Initialize LBP feature extractor.
        
        Args:
            radius: Radius of the circular pattern
            n_points: Number of points in the circular pattern
            method: LBP method ('uniform', 'nri_uniform', 'var')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        
        logger.info("LBPFeatureExtractor initialized: radius=%d, n_points=%d, method=%s", 
                   radius, n_points, method)
    
    def extract_lbp_features(self, palm_image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale LBP features from palm image for enhanced discrimination.
        
        Args:
            palm_image: 96x96 grayscale palm image
            
        Returns:
            Combined LBP feature vector (histogram)
        """
        if palm_image is None or palm_image.size == 0:
            raise ValueError("Empty palm image provided")
        
        # Ensure grayscale
        if len(palm_image.shape) == 3:
            gray = cv2.cvtColor(palm_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = palm_image.copy()
        
        # Apply histogram equalization for lighting normalization
        equalized = exposure.equalize_adapthist(gray, clip_limit=0.03)
        equalized = (equalized * 255).astype(np.uint8)
        
        # Extract multi-scale LBP features for better discrimination
        all_features = []
        
        # Multiple scales: fine and coarse texture patterns
        scales = [(1, 8), (2, 16), (3, 24)]  # (radius, n_points)
        
        for radius, n_points in scales:
            # Compute LBP
            lbp = local_binary_pattern(equalized, n_points, radius, method=self.method)
            
            # Compute histogram
            if self.method == 'uniform':
                n_bins = n_points + 2
            else:
                n_bins = int(lbp.max()) + 1
            
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-7)
            all_features.extend(hist)
        
        # Add texture statistics for additional discrimination
        texture_features = self._extract_texture_statistics(equalized)
        all_features.extend(texture_features)
        
        combined_features = np.array(all_features, dtype=np.float32)
        
        logger.debug("Extracted multi-scale LBP features: %d total features", len(combined_features))
        return combined_features
    
    def _extract_texture_statistics(self, image: np.ndarray) -> np.ndarray:
        """Extract additional texture statistics for better discrimination."""
        # Local standard deviation using 3x3 kernel
        kernel = np.ones((3, 3), np.float32) / 9
        mean_img = cv2.filter2D(image.astype(np.float32), -1, kernel)
        std_img = np.sqrt(cv2.filter2D((image.astype(np.float32) - mean_img) ** 2, -1, kernel))
        
        # Texture statistics
        features = [
            np.mean(std_img),           # Average local standard deviation
            np.std(std_img),            # Standard deviation of local std
            np.percentile(std_img, 25), # 25th percentile
            np.percentile(std_img, 75), # 75th percentile
        ]
        
        return np.array(features, dtype=np.float32)
    


class HandGeometryExtractor:
    """Hand geometry feature extractor using MediaPipe landmarks."""
    
    def __init__(self):
        """Initialize hand geometry extractor."""
        logger.info("HandGeometryExtractor initialized")
    
    def extract_geometry_features(self, landmarks) -> np.ndarray:
        """
        Extract hand geometry features from MediaPipe landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Geometry feature vector
        """
        if landmarks is None:
            return np.array([])
        
        features = []
        
        # Palm width (distance between landmarks 5 and 17 - MCP joints)
        palm_width = self._calculate_distance(landmarks.landmark[5], landmarks.landmark[17])
        features.append(palm_width)
        
        # Palm length (distance from wrist to middle of palm)
        wrist = landmarks.landmark[0]
        palm_center = self._get_palm_center(landmarks)
        palm_length = self._calculate_distance(wrist, palm_center)
        features.append(palm_length)
        
        # Finger lengths (from MCP to tip)
        finger_lengths = []
        finger_pairs = [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]  # thumb, index, middle, ring, pinky
        
        for tip_idx, mcp_idx in finger_pairs:
            length = self._calculate_distance(landmarks.landmark[tip_idx], landmarks.landmark[mcp_idx])
            finger_lengths.append(length)
        
        features.extend(finger_lengths)
        
        # Finger widths (MCP to PIP distance)
        finger_widths = []
        width_pairs = [(3, 2), (6, 5), (10, 9), (14, 13), (18, 17)]  # thumb, index, middle, ring, pinky
        
        for pip_idx, mcp_idx in width_pairs:
            width = self._calculate_distance(landmarks.landmark[pip_idx], landmarks.landmark[mcp_idx])
            finger_widths.append(width)
        
        features.extend(finger_widths)
        
        # Palm area (approximate using key palm landmarks)
        palm_landmarks = [0, 1, 2, 5, 9, 13, 17]  # wrist, palm base, MCPs
        palm_points = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                               for i in palm_landmarks])
        palm_area = self._calculate_polygon_area(palm_points)
        features.append(palm_area)
        
        # Aspect ratios
        features.append(palm_length / (palm_width + 1e-7))  # palm aspect ratio
        
        # Normalize features
        features = np.array(features, dtype=np.float32)
        
        logger.debug("Extracted geometry features: %d features", len(features))
        return features
    
    def _calculate_distance(self, point1, point2) -> float:
        """Calculate Euclidean distance between two 3D points."""
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _get_palm_center(self, landmarks) -> Any:
        """Get approximate palm center from landmarks."""
        # Use average of key palm landmarks
        palm_landmarks = [1, 2, 5, 9, 13, 17]  # palm base and MCPs
        
        x = np.mean([landmarks.landmark[i].x for i in palm_landmarks])
        y = np.mean([landmarks.landmark[i].y for i in palm_landmarks])
        z = np.mean([landmarks.landmark[i].z for i in palm_landmarks])
        
        # Create a simple point-like object
        class Point:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        return Point(x, y, z)
    
    def _calculate_polygon_area(self, points: np.ndarray) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(points) < 3:
            return 0.0
        
        x = points[:, 0]
        y = points[:, 1]
        
        # Shoelace formula
        area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] 
                            for i in range(-1, len(x) - 1)))
        return area


class PalmFeatureExtractor:
    """Combined palm feature extractor using LBP and optional geometry features."""
    
    def __init__(self, use_geometry: bool = True, lbp_radius: int = 3, 
                 lbp_n_points: int = 24, lbp_method: str = 'uniform'):
        """
        Initialize combined feature extractor.
        
        Args:
            use_geometry: Whether to include hand geometry features
            lbp_radius: LBP radius
            lbp_n_points: LBP number of points
            lbp_method: LBP method
        """
        self.use_geometry = use_geometry
        self.lbp_extractor = LBPFeatureExtractor(lbp_radius, lbp_n_points, lbp_method)
        self.geometry_extractor = HandGeometryExtractor() if use_geometry else None
        
        logger.info("PalmFeatureExtractor initialized: geometry=%s, LBP=(%d,%d,%s)", 
                   use_geometry, lbp_radius, lbp_n_points, lbp_method)
    
    def extract_features(self, palm_image: np.ndarray, landmarks: Optional[Any] = None) -> np.ndarray:
        """
        Extract combined features from palm image and landmarks.
        
        Args:
            palm_image: 96x96 grayscale palm image
            landmarks: Optional MediaPipe landmarks for geometry features
            
        Returns:
            Combined feature vector
        """
        # Extract LBP features
        lbp_features = self.lbp_extractor.extract_lbp_features(palm_image)
        
        if self.use_geometry and landmarks is not None:
            # Extract geometry features
            geometry_features = self.geometry_extractor.extract_geometry_features(landmarks)
            
            # Combine features
            combined_features = np.concatenate([lbp_features, geometry_features])
        else:
            combined_features = lbp_features
        
        logger.debug("Extracted combined features: %d total (%d LBP + %d geometry)", 
                    len(combined_features), len(lbp_features), 
                    len(combined_features) - len(lbp_features) if self.use_geometry else 0)
        
        return combined_features.astype(np.float32)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get feature dimensions for this extractor.
        
        Returns:
            Dictionary with feature dimensions
        """
        # Multi-scale LBP dimensions: (8+2) + (16+2) + (24+2) + 4 texture stats = 58
        lbp_dim = 58
        
        geometry_dim = 0
        if self.use_geometry:
            # 2 (palm width, length) + 5 (finger lengths) + 5 (finger widths) + 1 (area) + 1 (aspect ratio)
            geometry_dim = 14
        
        return {
            'lbp': lbp_dim,
            'geometry': geometry_dim,
            'total': lbp_dim + geometry_dim
        }


def calculate_feature_similarity(features1: np.ndarray, features2: np.ndarray, 
                               method: str = 'cosine') -> float:
    """
    Calculate similarity between two feature vectors.
    
    Args:
        features1: First feature vector
        features2: Second feature vector
        method: Similarity method ('cosine', 'euclidean', 'correlation')
        
    Returns:
        Similarity score (higher is more similar)
    """
    if len(features1) != len(features2):
        raise ValueError("Feature vectors must have the same length")
    
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    elif method == 'euclidean':
        # Euclidean distance (convert to similarity)
        distance = np.linalg.norm(features1 - features2)
        # Convert distance to similarity (0-1 range)
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)
    
    elif method == 'correlation':
        # Pearson correlation coefficient
        correlation = np.corrcoef(features1, features2)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


__all__ = [
    'LBPFeatureExtractor',
    'HandGeometryExtractor', 
    'PalmFeatureExtractor',
    'calculate_feature_similarity'
]
