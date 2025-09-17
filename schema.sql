-- Database schema for TLS palm registration
-- Recreate the SQLite database structure with:
--   sqlite3 palms.db < schema.sql

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS registered_palms (
    user_id TEXT,
    handedness TEXT,
    name TEXT,
    embeddings BLOB NOT NULL,
    PRIMARY KEY (user_id, handedness)
);


