"""Palm verification module.

Handles palm verification against registered users in the database.
"""
from __future__ import annotations

import io
import sqlite3
import logging
from typing import Optional, Sequence, Tuple

import numpy as np

from registration import DB_PATH, ensure_db, extract_embedding

logger = logging.getLogger(__name__)


def deserialize_embeddings(blob: bytes) -> np.ndarray:
    """Load embeddings from database blob."""
    buf = io.BytesIO(blob)
    return np.load(buf)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def verify_palm(embeddings_list: Sequence[np.ndarray], db_path: str = DB_PATH, 
                similarity_threshold: float = 0.92, min_avg_similarity: float = 0.90,
                min_peak_similarity: float = 0.95, handedness: Optional[str] = None,
                max_variance: float = 0.05) -> Tuple[bool, Optional[str], Optional[str]]:
    """Verify palm against registered users in database.
    
    Args:
        embeddings_list: List of embedding vectors from verification snapshots
        db_path: Path to SQLite database
        similarity_threshold: Minimum similarity for individual snapshot match
        min_avg_similarity: Minimum average similarity across all snapshots
        min_peak_similarity: Minimum peak similarity (best single match)
        handedness: Filter by handedness ("Left" or "Right"), None for any
        max_variance: Maximum allowed standard deviation of similarity scores
    
    Returns:
        Tuple of (is_match, matched_user_id or None, matched_name or None)
    """
    if not embeddings_list:
        return False, None, None
    
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # Filter by handedness if specified
        if handedness is not None:
            cur = conn.execute("SELECT user_id, name, embeddings FROM registered_palms WHERE handedness = ?", (handedness,))
        else:
            cur = conn.execute("SELECT user_id, name, embeddings FROM registered_palms")
        rows = cur.fetchall()
        
        if not rows:
            logger.warning("No registered palms found in database")
            return False, None, None
        
        logger.info("Found %d registered users in database", len(rows))
        
        # Log embedding dimensions for debugging
        if embeddings_list:
            logger.info("Verification embeddings shape: %s", [e.shape for e in embeddings_list])
        
        best_match_score = 0.0
        best_match_user = None
        best_match_name = None
        
        for user_id, name, embeddings_blob in rows:
            try:
                stored_embeddings = deserialize_embeddings(embeddings_blob)
                # stored_embeddings shape: (N, embedding_dim)
                
                # Compare each verification embedding against all stored embeddings for this user
                match_count = 0
                similarities = []
                for verify_emb in embeddings_list:
                    max_similarity = 0.0
                    for stored_emb in stored_embeddings:
                        sim = cosine_similarity(verify_emb, stored_emb)
                        max_similarity = max(max_similarity, sim)
                    
                    similarities.append(max_similarity)
                    if max_similarity >= similarity_threshold:
                        match_count += 1
                
                # Log similarity scores for debugging
                avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
                max_sim = max(similarities) if similarities else 0.0
                
                # Calculate variance (standard deviation) of similarity scores
                sim_variance = 0.0
                if len(similarities) > 1:
                    sim_variance = np.std(similarities)
                
                logger.info("User %s: similarities=%s, avg=%.3f, max=%.3f, std=%.3f, matches=%d/%d", 
                           user_id, [f"{s:.3f}" for s in similarities], avg_sim, max_sim, sim_variance, match_count, len(embeddings_list))
                
                # Also print to console for immediate feedback
                print(f"User {user_id}: similarities={[f'{s:.3f}' for s in similarities]}, avg={avg_sim:.3f}, max={max_sim:.3f}, std={sim_variance:.3f}, matches={match_count}/{len(embeddings_list)}")
                
                # Require ALL snapshots to match (5/5) AND very high average similarity
                # AND at least one snapshot must have very high similarity
                # AND the similarity must be significantly above random chance (0.1)
                # AND variance must be below threshold (consistent similarity scores)
                if (match_count >= len(embeddings_list) and avg_sim > min_avg_similarity and max_sim > min_peak_similarity 
                    and avg_sim > 0.1 and max_sim > 0.15 and sim_variance <= max_variance):
                    avg_score = match_count / len(embeddings_list)
                    if avg_score > best_match_score:
                        best_match_score = avg_score
                        best_match_user = user_id
                        best_match_name = name
                        
            except Exception as exc:
                logger.error("Error processing embeddings for user %s: %s", user_id, exc)
                continue
        
        result = best_match_user is not None, best_match_user, best_match_name
        logger.info("Verification result: match=%s, user=%s, name=%s", result[0], result[1], result[2])
        return result
        
    except sqlite3.DatabaseError as exc:
        logger.error("DB error during verification: %s", exc)
        return False, None, None
    finally:
        conn.close()


def test_embedding_compatibility(roi_bgr: np.ndarray) -> np.ndarray:
    """Test function to check embedding extractor output."""
    embedding = extract_embedding(roi_bgr)
    logger.info("Generated embedding shape: %s, norm: %.3f", embedding.shape, np.linalg.norm(embedding))
    return embedding
