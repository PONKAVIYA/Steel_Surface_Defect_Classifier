"""Inspection traceability store: create, read, filter, export, error handling."""
from __future__ import annotations

import csv
import io
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.config import CLASS_NAMES
from src.database.db import DatabaseError, init_db
from src.database.repository import (InspectionRecord, InspectionRepository, RecordValidationError,
                                     new_inspection_id)


def make_record(cls="scratches", conf=0.9, source="upload", review=False, when=None, **kw) -> InspectionRecord:
    probs = {c: (1 - conf) / 5 for c in CLASS_NAMES}
    probs[cls] = conf
    when = when or datetime.now(timezone.utc)
    return InspectionRecord(
        inspection_id=kw.pop("inspection_id", new_inspection_id(when)), created_at=when.isoformat(), source=source,
        predicted_class=cls, predicted_index=CLASS_NAMES.index(cls), confidence=conf, probabilities=probs,
        review_needed=review, model_version="test-model", preprocessing={"input_size": 64}, **kw)


@pytest.fixture
def repo(tmp_path):
    return InspectionRepository(tmp_path / "insp")


def test_create_and_retrieve_with_image(repo):
    rec = repo.save(make_record(operator_note="coil 17"), image_bytes=b"fake-image-bytes", image_ext=".png")
    got = repo.get(rec.inspection_id)
    assert got.predicted_class == "scratches" and got.operator_note == "coil 17"
    assert got.probabilities["scratches"] == pytest.approx(0.9)
    path = repo.image_file(got)
    assert path is not None and path.read_bytes() == b"fake-image-bytes"
    assert got.image_sha256 and got.image_path.startswith("images/")
    assert repo.get("INS-DOES-NOT-EXIST") is None


def test_ids_are_unique():
    assert len({new_inspection_id() for _ in range(500)}) == 500


def test_filters(repo):
    now = datetime.now(timezone.utc)
    repo.save(make_record("scratches", inspection_id="INS-A-1", when=now))
    repo.save(make_record("crazing", inspection_id="INS-B-2", review=True, conf=0.4, when=now))
    repo.save(make_record("crazing", inspection_id="INS-C-3", source="demo_sample", when=now - timedelta(days=10)))
    assert {r.inspection_id for r in repo.query(classes=["crazing"])} == {"INS-B-2", "INS-C-3"}
    assert [r.inspection_id for r in repo.query(id_contains="b-2")] == ["INS-B-2"]
    assert [r.inspection_id for r in repo.query(review_only=True)] == ["INS-B-2"]
    assert [r.inspection_id for r in repo.query(sources=["demo_sample"])] == ["INS-C-3"]
    today = datetime.now().astimezone().date()
    assert {r.inspection_id for r in repo.query(date_from=today, date_to=today)} == {"INS-A-1", "INS-B-2"}
    old = (now - timedelta(days=10)).astimezone().date()
    assert [r.inspection_id for r in repo.query(date_from=old, date_to=old)] == ["INS-C-3"]
    assert repo.query(id_contains="%") == []  # LIKE wildcards are escaped
    assert repo.count() == 3
    assert repo.class_counts(repo.query())["crazing"] == 2


def test_csv_export(repo):
    repo.save(make_record("patches", inspection_id="INS-X-1"))
    rec = repo.save(make_record("inclusion", inspection_id="INS-X-2"))
    repo.set_verified_label(rec.inspection_id, "pitted_surface")
    data = repo.export_csv(repo.query())
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    assert len(rows) == 2
    by_id = {r["inspection_id"]: r for r in rows}
    assert by_id["INS-X-1"]["label_status"] == "model prediction (unverified)"
    assert by_id["INS-X-2"]["label_status"] == "verified by operator"
    assert by_id["INS-X-2"]["verified_label"] == "pitted_surface"
    assert {f"p_{c}" for c in CLASS_NAMES} <= set(rows[0])


def test_invalid_records_rejected(repo):
    with pytest.raises(RecordValidationError):
        repo.save(make_record(source="satellite"))
    bad = make_record()
    bad.predicted_class = "rust"
    with pytest.raises(RecordValidationError):
        repo.save(bad)
    bad = make_record()
    bad.probabilities["crazing"] += 0.5
    with pytest.raises(RecordValidationError, match="sum"):
        repo.save(bad)
    bad = make_record()
    bad.predicted_index = 0
    with pytest.raises(RecordValidationError):
        repo.save(bad)
    with pytest.raises(RecordValidationError):
        repo.set_verified_label("whatever", "rust")
    assert repo.count() == 0


def test_missing_image_and_damaged_row(repo):
    rec = repo.save(make_record(), image_bytes=b"img", image_ext=".jpg")
    repo.image_file(rec).unlink()
    assert repo.image_file(repo.get(rec.inspection_id)) is None  # missing file handled, record intact
    with sqlite3.connect(repo.db_path) as conn:
        conn.execute("UPDATE inspections SET probabilities_json='{not json', image_path='../../etc/passwd'")
    got = repo.get(rec.inspection_id)
    assert got.problems and "probabilities_json" in got.problems[0]
    assert repo.image_file(got) is None  # path traversal refused
    assert len(repo.to_dataframe([got])) == 1


def test_duplicate_id_fails_without_orphan_image(repo):
    rec = make_record(inspection_id="INS-DUP")
    repo.save(rec, image_bytes=b"first", image_ext=".png")
    with pytest.raises(DatabaseError):
        repo.save(make_record(inspection_id="INS-DUP", when=datetime(2020, 1, 1, tzinfo=timezone.utc)),
                  image_bytes=b"second", image_ext=".png")
    images = list((repo.base_dir / "images").rglob("*.png"))
    assert len(images) == 1 and images[0].read_bytes() == b"first"


def test_database_errors_are_reported(tmp_path):
    bad = tmp_path / "file_not_dir"
    bad.write_text("x")
    with pytest.raises(DatabaseError):
        InspectionRepository(bad)
    corrupt = tmp_path / "corrupt" / "inspections.db"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"this is not a sqlite database" * 100)
    with pytest.raises(DatabaseError):
        init_db(corrupt)


def test_update_note_and_verification(repo):
    rec = repo.save(make_record())
    repo.update_note(rec.inspection_id, "checked by QA")
    repo.set_verified_label(rec.inspection_id, "scratches")
    got = repo.get(rec.inspection_id)
    assert got.operator_note == "checked by QA" and got.verified_label == "scratches" and got.verified_at
    repo.set_verified_label(rec.inspection_id, None)
    assert repo.get(rec.inspection_id).verified_label is None
    with pytest.raises(DatabaseError):
        repo.update_note("INS-NOPE", "x")

