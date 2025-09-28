"""Database utilities for palm registration and verification.

Provides a simple SQLite wrapper with safe transaction handling and
utility functions for user and palm template management.
"""

from __future__ import annotations
import sqlite3
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    """SQLite wrapper with safe connection and transaction handling."""

    def __init__(self, db_path: str | Path = "palm_auth.db") -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish connection if not already connected."""
        if self._conn is None:
            logger.debug("Connecting to database at %s", self.db_path)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create required tables if they do not exist."""
        logger.debug("Ensuring database schema exists")
        with self.transaction() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS palm_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    handedness TEXT CHECK(handedness IN ('Left','Right')) NOT NULL,
                    features BLOB NOT NULL,
                    feature_type TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

    def close(self) -> None:
        """Close the database connection safely."""
        if self._conn is not None:
            logger.debug("Closing database connection")
            self._conn.close()
            self._conn = None

    def cursor(self) -> sqlite3.Cursor:
        """Get a cursor, ensuring connection exists."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn.cursor()

    def commit(self) -> None:
        if self._conn:
            self._conn.commit()

    def rollback(self) -> None:
        if self._conn:
            self._conn.rollback()

    @contextmanager
    def transaction(self) -> sqlite3.Cursor:
        """
        Context manager for safe transactions:
            with db.transaction() as cur:
                cur.execute(...)
        """
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception as e:
            logger.error("Transaction failed, rolling back: %s", e)
            self._conn.rollback()
            raise
        finally:
            cur.close()


# --- High-level functions using Database ---
db = Database()


def create_user(name: str) -> int:
    """Insert a new user and return its ID."""
    with db.transaction() as cur:
        cur.execute("INSERT INTO users (name) VALUES (?)", (name,))
        return cur.lastrowid


def save_palm_template(user_id: int,
                       handedness: str,
                       features: Any,
                       feature_type: str) -> int:
    """Save palm template for a given user."""
    import pickle
    features_blob = pickle.dumps(features)

    with db.transaction() as cur:
        cur.execute("""
            INSERT INTO palm_templates (user_id, handedness, features, feature_type)
            VALUES (?, ?, ?, ?)
        """, (user_id, handedness, features_blob, feature_type))
        return cur.lastrowid


def load_user_templates(user_id: int) -> List[Tuple[int, str, Any, str]]:
    """Load all templates for a given user ID."""
    import pickle
    with db.transaction() as cur:
        cur.execute("SELECT * FROM palm_templates WHERE user_id = ?", (user_id,))
        rows = cur.fetchall()
    return [
        (row["id"], row["handedness"], pickle.loads(row["features"]), row["feature_type"])
        for row in rows
    ]


def find_user_by_name(name: str) -> Optional[int]:
    """Find a user ID by name."""
    with db.transaction() as cur:
        cur.execute("SELECT id FROM users WHERE name = ?", (name,))
        row = cur.fetchone()
        return row["id"] if row else None
