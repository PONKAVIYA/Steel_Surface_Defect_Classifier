"""SQLite connection and schema for inspection traceability.

Images are NOT stored in the database: the original file bytes are written to
<inspections_dir>/images/YYYY/MM/<inspection_id>.<ext> and the record keeps the relative path.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1
VALID_SOURCES = ("upload", "hardware", "simulated_hardware", "demo_sample")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id        TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,              -- ISO-8601, UTC
    source               TEXT NOT NULL CHECK (source IN ({", ".join(repr(s) for s in VALID_SOURCES)})),
    device_id            TEXT,
    original_filename    TEXT,
    image_path           TEXT,                       -- relative to the inspections directory
    image_sha256         TEXT,
    image_width          INTEGER,
    image_height         INTEGER,
    predicted_class      TEXT NOT NULL,
    predicted_index      INTEGER NOT NULL,
    confidence           REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probabilities_json   TEXT NOT NULL,
    review_needed        INTEGER NOT NULL,
    review_threshold     REAL,
    quality_suitable     INTEGER,
    quality_json         TEXT,
    model_version        TEXT NOT NULL,
    preprocessing_json   TEXT NOT NULL,
    inference_ms         REAL,
    operator_note        TEXT,
    verified_label       TEXT,                        -- optional label confirmed by a person
    verified_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_class ON inspections(predicted_class);
"""


class DatabaseError(RuntimeError):
    """Raised for any database failure, with a message suitable for the UI."""


def init_db(db_path: str | Path) -> Path:
    db_path = Path(db_path)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseError(f"Cannot create the inspection folder {db_path.parent}: {exc}") from exc
    try:
        with sqlite3.connect(db_path, timeout=10) as conn:
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] > SCHEMA_VERSION:
                raise DatabaseError(f"Database schema v{row[0]} is newer than this software (v{SCHEMA_VERSION}).")
    except sqlite3.DatabaseError as exc:
        raise DatabaseError(f"Cannot open inspection database {db_path}: {exc}") from exc
    return db_path


@contextmanager
def connect(db_path: str | Path):
    try:
        conn = sqlite3.connect(db_path, timeout=10)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Cannot connect to inspection database {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        conn.close()
