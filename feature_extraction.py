"""
Palm feature extraction module with tiled LBP.

Provides:
- LBPExtractor: tiled LBP (default grid 3x3) -> (feature_vector, variance, tile_variances)
- PalmFeatureExtractor: wrapper that optionally appends geometry features
- cosine_similarity helper
- Variance gating for rejecting low-quality ROIs
"""

from __future__ import annotations
import logging
from typing import Tuple, List, Optional, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class LBPExtractor:
    def __init__(self, grid: Tuple[int, int] = (3, 3), hist_bins: int = 256):
        self.grid = grid
        self.hist_bins = int(hist_bins)

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Empty image")
        if img.ndim == 3 and img.shape[2] == 3:
            # ESP32-CAM optimized grayscale conversion
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing for ESP32-CAM JPEG quality
            # Apply bilateral filter to reduce noise while preserving edges
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply unsharp masking to enhance texture details
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            
            return gray
        return img.copy()

    def _lbp_img(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        lbp = np.zeros((h, w), dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0),
                   (1, -1), (0, -1)]
        for dy, dx in offsets:
            shifted = np.zeros_like(gray)
            src_y1 = max(0, -dy)
            src_y2 = h - max(0, dy)
            dst_y1 = max(0, dy)
            dst_y2 = h - max(0, -dy)
            src_x1 = max(0, -dx)
            src_x2 = w - max(0, dx)
            dst_x1 = max(0, dx)
            dst_x2 = w - max(0, -dx)
            shifted[dst_y1:dst_y2, dst_x1:dst_x2] = gray[src_y1:src_y2, src_x1:src_x2]
            lbp = (lbp << 1) | (shifted >= gray).astype(np.uint8)
        return lbp

    def _hist(self, patch: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([patch], [0], None, [self.hist_bins], [0, 256]).flatten().astype(np.float32)
        s = hist.sum()
        if s <= 0:
            return hist
        return hist / s

    def extract(self, roi_bgr: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        gray = self._to_gray(roi_bgr)
        h, w = gray.shape[:2]
        if h < 32 or w < 32:
            gray = cv2.resize(gray, (max(32, w), max(32, h)), interpolation=cv2.INTER_LINEAR)
            h, w = gray.shape[:2]

        # ESP32-CAM specific: Apply additional preprocessing for better LBP
        # Normalize intensity to improve feature consistency
        gray = cv2.equalizeHist(gray)
        
        lbp = self._lbp_img(gray)
        rows, cols = self.grid
        tile_h = lbp.shape[0] // rows
        tile_w = lbp.shape[1] // cols

        features = []
        tile_variances: List[float] = []
        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * tile_h, c * tile_w
                y1 = (r + 1) * tile_h if r < rows - 1 else lbp.shape[0]
                x1 = (c + 1) * tile_w if c < cols - 1 else lbp.shape[1]
                patch = lbp[y0:y1, x0:x1]
                tile_variances.append(float(np.var(patch)))
                hist = self._hist(patch)
                features.append(hist)

        if not features:
            return np.array([], dtype=np.float32), 0.0, []

        feature_vector = np.concatenate(features).astype(np.float32)
        global_var = float(np.var(gray))
        return feature_vector, global_var, tile_variances

    @staticmethod
    def chi2_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return float("inf")
        num = (a - b) ** 2
        denom = a + b + eps
        return float(0.5 * np.sum(num / denom))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return -1.0
        denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
        return float(np.dot(a, b) / denom)


class GeometryFeatureExtractor:
    def extract(self, landmarks: Any) -> np.ndarray:
        if landmarks is None or not hasattr(landmarks, "landmark"):
            return np.array([], dtype=np.float32)
        try:
            coords = np.array([[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks.landmark], dtype=np.float32)
            mean = np.mean(coords, axis=0)
            std = np.std(coords, axis=0)
            max_dist = float(np.max(np.linalg.norm(coords - mean, axis=1)))
            return np.concatenate([mean, std, [max_dist]]).astype(np.float32)
        except Exception as e:
            logger.debug("Geometry extraction failed: %s", e)
            return np.array([], dtype=np.float32)


class PalmFeatureExtractor:
    def __init__(self, use_geometry: bool = True, lbp_grid: Tuple[int, int] = (3, 3), hist_bins: int = 256):
        self.use_geometry = bool(use_geometry)
        self.lbp = LBPExtractor(grid=lbp_grid, hist_bins=hist_bins)
        self.geometry = GeometryFeatureExtractor() if self.use_geometry else None

    def extract(self, palm_roi: np.ndarray, landmarks: Optional[Any] = None) -> Tuple[np.ndarray, float, List[float]]:
        feats, global_var, tile_vars = self.lbp.extract(palm_roi)
        if feats.size == 0:
            return feats, global_var, tile_vars
        if self.use_geometry and self.geometry is not None and landmarks is not None:
            geom = self.geometry.extract(landmarks)
            if geom.size > 0:
                feats = np.concatenate([feats, geom.astype(np.float32)])
        return feats, global_var, tile_vars

    def extract_lbp_only(self, palm_roi: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        return self.lbp.extract(palm_roi)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return LBPExtractor.cosine_similarity(a, b)
