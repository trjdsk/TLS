"""Palm registration utilities: embeddings, DB, snapshot validation, and IO.

This module provides a minimal, robust registration flow backend:
- Stub embedding extractor (to be replaced by Edge Impulse later)
- SQLite persistence for registered palms
- Helpers for palm ROI extraction, IoU, and debug saving
"""
from __future__ import annotations

import io
import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


DB_PATH = "palms.db"


@dataclass
class Snapshot:
    """Represents a validated palm snapshot and its derived data."""
    roi_bgr: np.ndarray
    bbox_xywh: Tuple[int, int, int, int]
    landmarks_xy: Optional[np.ndarray]
    embedding: Optional[np.ndarray] = None


 


def ensure_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        # Check if old schema exists and migrate if needed
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='registered_palms'")
        if cur.fetchone() is not None:
            # Check if old schema (without handedness column)
            cur = conn.execute("PRAGMA table_info(registered_palms)")
            columns = [row[1] for row in cur.fetchall()]
            if 'handedness' not in columns:
                logger.info("Migrating database schema to support handedness...")
                # Create new table with handedness support
                conn.execute("""
                    CREATE TABLE registered_palms_new (
                        user_id TEXT,
                        handedness TEXT,
                        name TEXT,
                        embeddings BLOB NOT NULL,
                        PRIMARY KEY (user_id, handedness)
                    )
                """)
                # Migrate existing data (assume all existing palms are right-handed)
                conn.execute("""
                    INSERT INTO registered_palms_new (user_id, handedness, name, embeddings)
                    SELECT user_id, 'Right', 'Unknown', embeddings FROM registered_palms
                """)
                # Drop old table and rename new one
                conn.execute("DROP TABLE registered_palms")
                conn.execute("ALTER TABLE registered_palms_new RENAME TO registered_palms")
                logger.info("Database migration completed. Existing palms marked as 'Right' handedness with 'Unknown' name.")
            elif 'name' not in columns:
                logger.info("Migrating database schema to support names...")
                # Add name column to existing table
                conn.execute("ALTER TABLE registered_palms ADD COLUMN name TEXT DEFAULT 'Unknown'")
                logger.info("Database migration completed. Added name column with 'Unknown' default.")
        
        # Create table if it doesn't exist (new installations)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_palms (
                user_id TEXT,
                handedness TEXT,
                name TEXT,
                embeddings BLOB NOT NULL,
                PRIMARY KEY (user_id, handedness)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def serialize_embeddings(embeddings_list: Sequence[np.ndarray]) -> bytes:
    # Store as a single stacked float32 array for portability
    stacked = np.stack([e.astype(np.float32, copy=False) for e in embeddings_list], axis=0)
    buf = io.BytesIO()
    np.save(buf, stacked)
    return buf.getvalue()


def verify_palm_for_registration(roi_bgr: np.ndarray, threshold: float = 0.5) -> bool:
    """Verify that the ROI contains a palm using Edge Impulse model during registration.
    
    Args:
        roi_bgr: Preprocessed palm ROI (96x96 grayscale)
        threshold: Confidence threshold for palm detection
        
    Returns:
        True if palm is detected, False otherwise
    """
    try:
        from model_wrapper import get_model
        
        model = get_model()
        if model.is_initialized:
            # ROI should already be preprocessed by detector
            return model.is_palm(roi_bgr, threshold)
        else:
            logger.warning("Edge Impulse model not initialized, cannot verify palm during registration")
            return True  # Allow registration to proceed if model not available
    except Exception as exc:
        logger.warning("Edge Impulse model not available for registration verification: %s", exc)
        return True  # Allow registration to proceed if model not available


def register_user(user_id: str, embeddings_list: Sequence[np.ndarray], handedness: str = "Right", name: str = "Unknown", db_path: str = DB_PATH) -> bool:
    ensure_db(db_path)
    try:
        payload = serialize_embeddings(embeddings_list)
    except Exception as exc:
        logger.error("Failed to serialize embeddings: %s", exc)
        return False

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            # Insert or gracefully reject duplicates (now based on user_id + handedness)
            cur = conn.execute("SELECT 1 FROM registered_palms WHERE user_id = ? AND handedness = ?", (user_id, handedness))
            if cur.fetchone() is not None:
                logger.warning("Duplicate user_id '%s' with handedness '%s' rejected.", user_id, handedness)
                return False
            conn.execute(
                "INSERT INTO registered_palms (user_id, handedness, name, embeddings) VALUES (?, ?, ?, ?)",
                (user_id, handedness, name, payload),
            )
        return True
    except sqlite3.DatabaseError as exc:
        logger.error("DB error during registration: %s", exc)
        return False
    finally:
        conn.close()


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def palm_bbox_from_landmarks(landmarks_xy: np.ndarray) -> Tuple[int, int, int, int]:
    """Approximate palm-only bbox using convex hull of wrist + MCP/CMC points."""
    indices = [0, 1, 2, 5, 9, 13, 17]
    pts = landmarks_xy[indices].astype(np.int32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    x_min = int(np.min(hull[:, 0]))
    y_min = int(np.min(hull[:, 1]))
    x_max = int(np.max(hull[:, 0]))
    y_max = int(np.max(hull[:, 1]))
    return (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))


def crop_roi(frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int], pad: int = 8) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox_xywh
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    return frame_bgr[y0:y1, x0:x1].copy()


def normalize_palm_orientation(roi_bgr: np.ndarray, landmarks_xy: Optional[np.ndarray]) -> np.ndarray:
    """Normalize palm orientation using wrist to middle finger MCP vector.
    
    Args:
        roi_bgr: Palm ROI image
        landmarks_xy: Hand landmarks in pixel coordinates (21, 2) or None
        
    Returns:
        Normalized ROI with consistent orientation
    """
    if landmarks_xy is None or landmarks_xy.shape != (21, 2):
        # If no landmarks available, return original ROI
        return roi_bgr
    
    # Get wrist (0) and middle finger MCP (9) landmarks
    wrist = landmarks_xy[0].astype(np.float32)
    middle_mcp = landmarks_xy[9].astype(np.float32)
    
    # Compute vector from wrist to middle finger MCP
    palm_vector = middle_mcp - wrist
    
    # Calculate angle relative to vertical (pointing upward)
    # atan2 gives angle from positive x-axis, we want from positive y-axis (upward)
    angle_rad = np.arctan2(palm_vector[0], -palm_vector[1])  # Negative y because image y increases downward
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    # Only rotate if angle is significant (> 5 degrees)
    if abs(angle_deg) > 5.0:
        # Get image center
        h, w = roi_bgr.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
        
        # Rotate the image
        normalized_roi = cv2.warpAffine(roi_bgr, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return normalized_roi
    else:
        # Angle is small, no rotation needed
        return roi_bgr


def extract_embedding(roi_bgr: np.ndarray) -> np.ndarray:
    """Extract palm embedding using Edge Impulse model.

    Uses the Edge Impulse trained model to extract discriminative features
    for palm recognition. The ROI should already be preprocessed (96x96 grayscale).
    """
    try:
        # Try to use Edge Impulse model first
        from model_wrapper import get_model
        
        model = get_model()
        if model.is_initialized:
            # ROI should already be preprocessed by detector
            embedding = model.get_embedding(roi_bgr, embedding_size=128)
            logger.debug("Extracted embedding using Edge Impulse model, shape: %s", embedding.shape)
            return embedding
        else:
            logger.warning("Edge Impulse model not initialized, falling back to custom features")
    except Exception as exc:
        logger.warning("Edge Impulse model not available, falling back to custom features: %s", exc)
    
    # Fallback to custom feature extraction
    return _extract_custom_embedding(roi_bgr)


def _extract_custom_embedding(roi_bgr: np.ndarray) -> np.ndarray:
    """Fallback custom palm embedding extractor using multiple feature types (4096 dimensions).

    Uses a combination of approaches to create highly discriminative embeddings:
    1. Raw pixel features with quantization
    2. Histogram features
    3. Texture features
    4. Geometric features
    """
    # Resize to 64x64 for consistent processing
    roi_resized = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. Quantized pixel features (2048 features)
    # Quantize to reduce noise and make more discriminative
    quantized = (gray // 4) * 4  # Quantize to 64 levels (256/4)
    pixel_features = quantized.astype(np.float32).flatten()
    # Normalize to 0-1 range
    pixel_features = pixel_features / 255.0
    features.extend(pixel_features)
    
    # 2. Histogram features (1024 features)
    # Create histograms at different scales
    for scale in [1, 2, 4]:
        if scale > 1:
            scaled = cv2.resize(gray, (64//scale, 64//scale), interpolation=cv2.INTER_AREA)
        else:
            scaled = gray
        
        # Intensity histogram
        hist = cv2.calcHist([scaled], [0], None, [64], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        features.extend(hist)
        
        # Gradient histogram
        grad_x = cv2.Sobel(scaled, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(scaled, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [64], [0, 256])
        grad_hist = grad_hist.flatten() / (grad_hist.sum() + 1e-8)
        features.extend(grad_hist)
    
    # 3. Texture features (512 features)
    # Local Binary Pattern-like features
    lbp_features = []
    for i in range(1, 63):  # Skip border pixels
        for j in range(1, 63):
            center = gray[i, j]
            # Compare with 8 neighbors
            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j-1], gray[i, j+1],
                gray[i+1, j-1], gray[i+1, j], gray[i+1, j+1]
            ]
            # Create binary pattern
            pattern = sum(1 << k for k, neighbor in enumerate(neighbors) if neighbor > center)
            lbp_features.append(pattern / 255.0)  # Normalize to 0-1
    
    # Take every 4th feature to reduce dimensionality
    features.extend(lbp_features[::4][:512])
    
    # 4. Geometric features (512 features)
    # Analyze the shape and structure
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Regional analysis
    regions = [
        edges[:32, :32],      # top-left
        edges[:32, 32:],      # top-right  
        edges[32:, :32],      # bottom-left
        edges[32:, 32:]       # bottom-right
    ]
    
    for region in regions:
        # Edge density
        edge_density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
        features.append(edge_density)
        
        # Edge distribution
        if np.sum(region > 0) > 0:
            y_coords, x_coords = np.where(region > 0)
            if len(y_coords) > 0:
                # Spatial distribution
                features.extend([
                    np.mean(y_coords) / region.shape[0],
                    np.mean(x_coords) / region.shape[1],
                    np.std(y_coords) / region.shape[0],
                    np.std(x_coords) / region.shape[1]
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Convert to numpy array and ensure exactly 4096 dimensions
    feature_vector = np.array(features, dtype=np.float32)
    
    # Ensure exactly 4096 dimensions
    if len(feature_vector) != 4096:
        if len(feature_vector) < 4096:
            feature_vector = np.pad(feature_vector, (0, 4096 - len(feature_vector)), 'constant')
        else:
            feature_vector = feature_vector[:4096]
    
    # L2 normalization
    norm = np.linalg.norm(feature_vector) + 1e-8
    feature_vector /= norm
    
    return feature_vector


 


def validate_consistency(bboxes: Sequence[Tuple[int, int, int, int]], min_iou: float = 0.15) -> bool:
    """Check that consecutive bboxes belong to the same palm with modest movement."""
    if len(bboxes) <= 1:
        return False
    for prev, curr in zip(bboxes[:-1], bboxes[1:]):
        if compute_iou(prev, curr) < min_iou:
            return False
    return True




