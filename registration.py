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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_palms (
                user_id TEXT PRIMARY KEY,
                embeddings BLOB NOT NULL
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


def register_user(user_id: str, embeddings_list: Sequence[np.ndarray], db_path: str = DB_PATH) -> bool:
    ensure_db(db_path)
    try:
        payload = serialize_embeddings(embeddings_list)
    except Exception as exc:
        logger.error("Failed to serialize embeddings: %s", exc)
        return False

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            # Insert or gracefully reject duplicates
            cur = conn.execute("SELECT 1 FROM registered_palms WHERE user_id = ?", (user_id,))
            if cur.fetchone() is not None:
                logger.warning("Duplicate user_id '%s' rejected.", user_id)
                return False
            conn.execute(
                "INSERT INTO registered_palms (user_id, embeddings) VALUES (?, ?)",
                (user_id, payload),
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


def extract_embedding(roi_bgr: np.ndarray) -> np.ndarray:
    """Stub embedding extractor.

    Strategy: resize to 64x64 grayscale, L2-normalize flattened vector.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    vec = resized.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(vec) + 1e-8
    vec /= norm
    return vec


 


def validate_consistency(bboxes: Sequence[Tuple[int, int, int, int]], min_iou: float = 0.15) -> bool:
    """Check that consecutive bboxes belong to the same palm with modest movement."""
    if len(bboxes) <= 1:
        return False
    for prev, curr in zip(bboxes[:-1], bboxes[1:]):
        if compute_iou(prev, curr) < min_iou:
            return False
    return True


