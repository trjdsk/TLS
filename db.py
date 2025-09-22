"""Database management module for palm verification system.

Manages SQLite database with users and palm feature descriptors.
Stores hand-crafted features (ORB/SIFT/SURF) instead of embeddings.
"""
from __future__ import annotations

import io
import sqlite3
import logging
from typing import List, Optional, Tuple, Any
import pickle
import numpy as np

logger = logging.getLogger(__name__)

DB_PATH = "palms.db"


def ensure_db(db_path: str = DB_PATH) -> None:
    """Ensure database exists with proper schema."""
    conn = sqlite3.connect(db_path)
    try:
        # Create users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL
            )
        """)
        
        # Create palm_templates table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS palm_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                handedness TEXT NOT NULL,
                descriptors BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        
        # Create index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_palm_templates_user_handedness 
            ON palm_templates(user_id, handedness)
        """)
        
        conn.commit()
        logger.info("Database schema ensured at: %s", db_path)
        
    except sqlite3.DatabaseError as e:
        logger.error("Database error: %s", e)
        raise
    finally:
        conn.close()


def serialize_descriptors(descriptors: np.ndarray) -> bytes:
    """Serialize feature descriptors to bytes for database storage."""
    try:
        # Use pickle for serialization (handles numpy arrays well)
        return pickle.dumps(descriptors)
    except Exception as e:
        logger.error("Failed to serialize descriptors: %s", e)
        raise


def deserialize_descriptors(blob: bytes) -> np.ndarray:
    """Deserialize feature descriptors from database blob."""
    try:
        return pickle.loads(blob)
    except Exception as e:
        logger.error("Failed to deserialize descriptors: %s", e)
        raise


def create_user(name: str, db_path: str = DB_PATH) -> int:
    """
    Create a new user in the database.
    
    Args:
        name: User's name
        db_path: Path to database file
        
    Returns:
        User ID of created user
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            cursor = conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
            user_id = cursor.lastrowid
            logger.info("Created user: %s (ID: %d)", name, user_id)
            return user_id
    except sqlite3.DatabaseError as e:
        logger.error("Failed to create user: %s", e)
        raise
    finally:
        conn.close()


def save_palm_template(user_id: int, handedness: str, descriptors: np.ndarray, 
                      db_path: str = DB_PATH) -> int:
    """
    Save palm feature descriptors for a user.
    
    Args:
        user_id: User ID
        handedness: "Left" or "Right"
        descriptors: Feature descriptors (ORB/SIFT/SURF)
        db_path: Path to database file
        
    Returns:
        Template ID of saved template
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            # Serialize descriptors
            descriptors_blob = serialize_descriptors(descriptors)
            
            # Insert template
            cursor = conn.execute("""
                INSERT INTO palm_templates (user_id, handedness, descriptors) 
                VALUES (?, ?, ?)
            """, (user_id, handedness, descriptors_blob))
            
            template_id = cursor.lastrowid
            logger.info("Saved palm template for user %d (%s hand), template ID: %d", 
                       user_id, handedness, template_id)
            return template_id
            
    except sqlite3.DatabaseError as e:
        logger.error("Failed to save palm template: %s", e)
        raise
    finally:
        conn.close()


def get_user_templates(user_id: int, handedness: Optional[str] = None, 
                      db_path: str = DB_PATH) -> List[Tuple[int, np.ndarray]]:
    """
    Get palm templates for a user.
    
    Args:
        user_id: User ID
        handedness: Filter by handedness ("Left" or "Right"), None for all
        db_path: Path to database file
        
    Returns:
        List of (template_id, descriptors) tuples
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        if handedness is not None:
            cursor = conn.execute("""
                SELECT id, descriptors FROM palm_templates 
                WHERE user_id = ? AND handedness = ?
            """, (user_id, handedness))
        else:
            cursor = conn.execute("""
                SELECT id, descriptors FROM palm_templates 
                WHERE user_id = ?
            """, (user_id,))
        
        templates = []
        for template_id, descriptors_blob in cursor.fetchall():
            descriptors = deserialize_descriptors(descriptors_blob)
            templates.append((template_id, descriptors))
        
        logger.info("Retrieved %d templates for user %d", len(templates), user_id)
        return templates
        
    except sqlite3.DatabaseError as e:
        logger.error("Failed to get user templates: %s", e)
        raise
    finally:
        conn.close()


def get_all_templates(handedness: Optional[str] = None, 
                     db_path: str = DB_PATH) -> List[Tuple[int, int, str, np.ndarray]]:
    """
    Get all palm templates from database.
    
    Args:
        handedness: Filter by handedness ("Left" or "Right"), None for all
        db_path: Path to database file
        
    Returns:
        List of (template_id, user_id, handedness, descriptors) tuples
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        if handedness is not None:
            cursor = conn.execute("""
                SELECT pt.id, pt.user_id, pt.handedness, pt.descriptors 
                FROM palm_templates pt
                WHERE pt.handedness = ?
            """, (handedness,))
        else:
            cursor = conn.execute("""
                SELECT pt.id, pt.user_id, pt.handedness, pt.descriptors 
                FROM palm_templates pt
            """)
        
        templates = []
        for template_id, user_id, hand, descriptors_blob in cursor.fetchall():
            descriptors = deserialize_descriptors(descriptors_blob)
            templates.append((template_id, user_id, hand, descriptors))
        
        logger.info("Retrieved %d templates from database", len(templates))
        return templates
        
    except sqlite3.DatabaseError as e:
        logger.error("Failed to get all templates: %s", e)
        raise
    finally:
        conn.close()


def get_user_name(user_id: int, db_path: str = DB_PATH) -> Optional[str]:
    """
    Get user name by user ID.
    
    Args:
        user_id: User ID
        db_path: Path to database file
        
    Returns:
        User name or None if not found
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.DatabaseError as e:
        logger.error("Failed to get user name: %s", e)
        return None
    finally:
        conn.close()


def delete_user(user_id: int, db_path: str = DB_PATH) -> bool:
    """
    Delete user and all their palm templates.
    
    Args:
        user_id: User ID to delete
        db_path: Path to database file
        
    Returns:
        True if user was deleted, False if not found
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            # Delete palm templates first (foreign key constraint)
            cursor = conn.execute("DELETE FROM palm_templates WHERE user_id = ?", (user_id,))
            templates_deleted = cursor.rowcount
            
            # Delete user
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            user_deleted = cursor.rowcount > 0
            
            if user_deleted:
                logger.info("Deleted user %d and %d templates", user_id, templates_deleted)
            else:
                logger.warning("User %d not found for deletion", user_id)
            
            return user_deleted
            
    except sqlite3.DatabaseError as e:
        logger.error("Failed to delete user: %s", e)
        return False
    finally:
        conn.close()


def get_database_stats(db_path: str = DB_PATH) -> dict:
    """
    Get database statistics.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Dictionary with database statistics
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # Count users
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        # Count templates
        cursor = conn.execute("SELECT COUNT(*) FROM palm_templates")
        template_count = cursor.fetchone()[0]
        
        # Count by handedness
        cursor = conn.execute("""
            SELECT handedness, COUNT(*) FROM palm_templates 
            GROUP BY handedness
        """)
        handedness_counts = dict(cursor.fetchall())
        
        return {
            'users': user_count,
            'templates': template_count,
            'handedness_counts': handedness_counts
        }
        
    except sqlite3.DatabaseError as e:
        logger.error("Failed to get database stats: %s", e)
        return {}
    finally:
        conn.close()


def list_users(db_path: str = DB_PATH) -> List[Tuple[int, str, int]]:
    """
    List all users with their template counts.
    
    Args:
        db_path: Path to database file
        
    Returns:
        List of (user_id, name, template_count) tuples
    """
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("""
            SELECT u.id, u.name, COUNT(pt.id) as template_count
            FROM users u
            LEFT JOIN palm_templates pt ON u.id = pt.user_id
            GROUP BY u.id, u.name
            ORDER BY u.id
        """)
        return cursor.fetchall()
    except sqlite3.DatabaseError as e:
        logger.error("Failed to list users: %s", e)
        return []
    finally:
        conn.close()
