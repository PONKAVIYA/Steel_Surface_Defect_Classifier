"""Inspection records: create, read, filter, summarise, export."""
from __future__ import annotations

import hashlib
import io
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.database.db import VALID_SOURCES, DatabaseError, connect, init_db

SOURCE_LABELS = {
    "upload": "Uploaded image",
    "hardware": "Hardware capture",
    "simulated_hardware": "Simulated hardware (not a real device)",
    "demo_sample": "Demo sample",
}


class RecordValidationError(ValueError):
    pass


def format_local(dt: datetime) -> str:
    """Local time with a numeric UTC offset (time-zone abbreviations are unreliable across systems)."""
    off = dt.strftime("%z")
    return dt.strftime("%Y-%m-%d %H:%M:%S") + (f" UTC{off[:3]}:{off[3:]}" if off else "")


def new_inspection_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"INS-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3).upper()}"


@dataclass
class InspectionRecord:
    inspection_id: str
    created_at: str
    source: str
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    review_needed: bool
    model_version: str
    preprocessing: dict
    review_threshold: float | None = None
    quality: dict = field(default_factory=dict)
    quality_suitable: bool | None = None
    inference_ms: float | None = None
    operator_note: str = ""
    device_id: str | None = None
    original_filename: str | None = None
    image_path: str | None = None
    image_sha256: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    verified_label: str | None = None
    verified_at: str | None = None
    problems: list[str] = field(default_factory=list)  # filled when a stored row is damaged

    def validate(self) -> None:
        if self.source not in VALID_SOURCES:
            raise RecordValidationError(f"Invalid source '{self.source}'. Valid: {VALID_SOURCES}")
        if self.predicted_class not in CLASS_NAMES:
            raise RecordValidationError(f"Unknown predicted class '{self.predicted_class}'")
        if CLASS_NAMES[self.predicted_index] != self.predicted_class:
            raise RecordValidationError("predicted_index does not match predicted_class")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise RecordValidationError(f"confidence must be in [0, 1], got {self.confidence}")
        if set(self.probabilities) != set(CLASS_NAMES):
            raise RecordValidationError("probabilities must contain exactly the six classes")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-3:
            raise RecordValidationError("probabilities must sum to 1")
        if self.verified_label is not None and self.verified_label not in CLASS_NAMES:
            raise RecordValidationError(f"Unknown verified label '{self.verified_label}'")
        if len(self.operator_note or "") > 2000:
            raise RecordValidationError("operator note is limited to 2000 characters")

    @property
    def display_class(self) -> str:
        return DISPLAY_NAMES.get(self.predicted_class, self.predicted_class)

    @property
    def created_local(self) -> datetime:
        return datetime.fromisoformat(self.created_at).astimezone()


def _row_to_record(row) -> InspectionRecord:
    problems = []

    def _json(col, default):
        try:
            return json.loads(row[col]) if row[col] else default
        except (TypeError, json.JSONDecodeError):
            problems.append(f"{col} is not valid JSON")
            return default

    rec = InspectionRecord(
        inspection_id=row["inspection_id"], created_at=row["created_at"], source=row["source"],
        predicted_class=row["predicted_class"], predicted_index=row["predicted_index"],
        confidence=row["confidence"], probabilities=_json("probabilities_json", {}),
        review_needed=bool(row["review_needed"]), model_version=row["model_version"],
        preprocessing=_json("preprocessing_json", {}), review_threshold=row["review_threshold"],
        quality=_json("quality_json", {}),
        quality_suitable=None if row["quality_suitable"] is None else bool(row["quality_suitable"]),
        inference_ms=row["inference_ms"], operator_note=row["operator_note"] or "", device_id=row["device_id"],
        original_filename=row["original_filename"], image_path=row["image_path"],
        image_sha256=row["image_sha256"], image_width=row["image_width"], image_height=row["image_height"],
        verified_label=row["verified_label"], verified_at=row["verified_at"],
    )
    if rec.predicted_class not in CLASS_NAMES:
        problems.append(f"unknown class '{rec.predicted_class}'")
    try:
        datetime.fromisoformat(rec.created_at)
    except (TypeError, ValueError):
        problems.append("invalid timestamp")
        rec.created_at = "1970-01-01T00:00:00+00:00"
    rec.problems = problems
    return rec


class InspectionRepository:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "inspections.db"
        self.images_dir = self.base_dir / "images"
        init_db(self.db_path)

    # ------------------------------------------------------------------ write
    def save(self, record: InspectionRecord, image_bytes: bytes | None = None, image_ext: str = ".png") -> InspectionRecord:
        record.validate()
        written: Path | None = None
        if image_bytes:
            ts = datetime.fromisoformat(record.created_at)
            rel = Path("images") / f"{ts:%Y}" / f"{ts:%m}" / f"{record.inspection_id}{image_ext.lower()}"
            dest = self.base_dir / rel
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(image_bytes)
            except OSError as exc:
                raise DatabaseError(f"Could not store inspection image: {exc}") from exc
            written = dest
            record.image_path = rel.as_posix()
            record.image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO inspections (inspection_id, created_at, source, device_id, original_filename,
                       image_path, image_sha256, image_width, image_height, predicted_class, predicted_index,
                       confidence, probabilities_json, review_needed, review_threshold, quality_suitable,
                       quality_json, model_version, preprocessing_json, inference_ms, operator_note,
                       verified_label, verified_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (record.inspection_id, record.created_at, record.source, record.device_id,
                     record.original_filename, record.image_path, record.image_sha256, record.image_width,
                     record.image_height, record.predicted_class, record.predicted_index, float(record.confidence),
                     json.dumps(record.probabilities), int(record.review_needed), record.review_threshold,
                     None if record.quality_suitable is None else int(record.quality_suitable),
                     json.dumps(record.quality), record.model_version, json.dumps(record.preprocessing),
                     record.inference_ms, record.operator_note or "", record.verified_label, record.verified_at))
        except DatabaseError:
            if written is not None:  # do not leave orphaned images behind
                written.unlink(missing_ok=True)
            raise
        return record

    def set_verified_label(self, inspection_id: str, label: str | None) -> None:
        if label is not None and label not in CLASS_NAMES:
            raise RecordValidationError(f"Unknown label '{label}'")
        stamp = datetime.now(timezone.utc).isoformat() if label else None
        with connect(self.db_path) as conn:
            cur = conn.execute("UPDATE inspections SET verified_label=?, verified_at=? WHERE inspection_id=?",
                               (label, stamp, inspection_id))
            if cur.rowcount == 0:
                raise DatabaseError(f"Inspection {inspection_id} not found")

    def update_note(self, inspection_id: str, note: str) -> None:
        if len(note) > 2000:
            raise RecordValidationError("operator note is limited to 2000 characters")
        with connect(self.db_path) as conn:
            cur = conn.execute("UPDATE inspections SET operator_note=? WHERE inspection_id=?", (note, inspection_id))
            if cur.rowcount == 0:
                raise DatabaseError(f"Inspection {inspection_id} not found")

    # ------------------------------------------------------------------ read
    def get(self, inspection_id: str) -> InspectionRecord | None:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM inspections WHERE inspection_id=?", (inspection_id,)).fetchone()
        return _row_to_record(row) if row else None

    def query(self, id_contains: str | None = None, classes: list[str] | None = None,
              date_from: date | None = None, date_to: date | None = None, review_only: bool = False,
              sources: list[str] | None = None, limit: int | None = None) -> list[InspectionRecord]:
        """Filter records. Dates are calendar days in the local time zone of this computer (inclusive)."""
        sql, args = ["SELECT * FROM inspections WHERE 1=1"], []
        if id_contains:
            sql.append("AND inspection_id LIKE ? ESCAPE '\\'")
            esc = id_contains.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            args.append(f"%{esc}%")
        if classes:
            sql.append(f"AND predicted_class IN ({','.join('?' * len(classes))})")
            args.extend(classes)
        if sources:
            sql.append(f"AND source IN ({','.join('?' * len(sources))})")
            args.extend(sources)
        local_tz = datetime.now().astimezone().tzinfo
        if date_from:
            start = datetime.combine(date_from, time.min, tzinfo=local_tz).astimezone(timezone.utc)
            sql.append("AND created_at >= ?")
            args.append(start.isoformat())
        if date_to:
            end = datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=local_tz).astimezone(timezone.utc)
            sql.append("AND created_at < ?")
            args.append(end.isoformat())
        if review_only:
            sql.append("AND review_needed = 1")
        sql.append("ORDER BY created_at DESC")
        if limit:
            sql.append("LIMIT ?")
            args.append(int(limit))
        with connect(self.db_path) as conn:
            rows = conn.execute(" ".join(sql), args).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self) -> int:
        with connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0])

    def image_file(self, record: InspectionRecord) -> Path | None:
        """Absolute path of the stored image, or None if there is none or it is missing on disk."""
        if not record.image_path:
            return None
        p = (self.base_dir / record.image_path).resolve()
        try:
            p.relative_to(self.base_dir.resolve())  # refuse paths escaping the inspections folder
        except ValueError:
            return None
        return p if p.is_file() else None

    # ------------------------------------------------------------------ summaries / export
    @staticmethod
    def to_dataframe(records: list[InspectionRecord]) -> pd.DataFrame:
        cols = ["inspection_id", "created_at_local", "source", "device_id", "predicted_class", "display_class",
                "confidence", "review_needed", "review_threshold", "quality_suitable", "n_quality_warnings",
                "verified_label", "model_version", "inference_ms", "operator_note", "original_filename",
                "image_path"] + [f"p_{c}" for c in CLASS_NAMES]
        rows = []
        for r in records:
            d = asdict(r)
            d["created_at_local"] = format_local(r.created_local)
            d["display_class"] = r.display_class
            d["n_quality_warnings"] = sum(1 for i in (r.quality.get("issues") or []) if i.get("severity") == "warning")
            for c in CLASS_NAMES:
                d[f"p_{c}"] = r.probabilities.get(c)
            rows.append({k: d.get(k) for k in cols})
        return pd.DataFrame(rows, columns=cols)

    def export_csv(self, records: list[InspectionRecord]) -> bytes:
        """CSV export. `label_status` makes explicit that predicted_class is a model prediction unless a
        person recorded a verified_label."""
        df = self.to_dataframe(records)
        verified = df["verified_label"].notna() & (df["verified_label"].astype(str) != "")
        df.insert(5, "label_status", np.where(verified, "verified by operator", "model prediction (unverified)"))
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    @staticmethod
    def class_counts(records: list[InspectionRecord]) -> dict[str, int]:
        counts = {c: 0 for c in CLASS_NAMES}
        for r in records:
            if r.predicted_class in counts:
                counts[r.predicted_class] += 1
        return counts
