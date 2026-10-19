"""HTTP API, simulated camera, drop-folder adapter and the hardware loop script."""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.config import CLASS_NAMES, load_config
from src.data.synthetic import synthetic_texture
from src.database.repository import InspectionRepository
from src.hardware.interface import (FolderDropAdapter, HardwareNotConnected, HardwareTimeout,
                                    build_result_message)
from src.hardware.simulator import SimulatedCameraAdapter
from tests.conftest import ROOT, write_config

os.environ["STEEL_API_NO_AUTOLOAD"] = "1"
from src.api.service import create_app  # noqa: E402


def img_bytes(fmt="PNG", cls="pitted_surface", seed=3) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(synthetic_texture(cls, np.random.default_rng(seed))).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client(temp_cfg):
    app = create_app(temp_cfg)
    return TestClient(app), app


# ----------------------------------------------------------------------------- API
def test_health_and_model(client):
    c, _ = client
    assert c.get("/v1/health").json() == {"status": "ok", "model_loaded": True, "model_error": None}
    info = c.get("/v1/model").json()
    assert info["class_names"] == list(CLASS_NAMES) and info["input_size"] == 64


def test_inspect_saves_record(client, temp_cfg):
    c, app = client
    r = c.post("/v1/inspect", files={"file": ("frame.png", img_bytes(), "image/png")},
               data={"device_id": "CAM-7", "source": "hardware", "operator_note": "line 2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_model_prediction"] is True and body["predicted_class"] in CLASS_NAMES
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-4
    assert body["saved"] and body["device_id"] == "CAM-7" and body["simulated"] is False
    rec = app.state.service.repository.get(body["inspection_id"])
    assert rec.source == "hardware" and rec.operator_note == "line 2"
    assert app.state.service.repository.image_file(rec) is not None


def test_inspect_without_saving(client):
    c, app = client
    r = c.post("/v1/inspect", files={"file": ("f.bmp", img_bytes("BMP"), "image/bmp")},
               data={"save": "false", "source": "simulated_hardware"})
    assert r.status_code == 200 and r.json()["saved"] is False and r.json()["simulated"] is True
    assert app.state.service.repository.count() == 0


@pytest.mark.parametrize("payload,status,code", [
    (b"\x89PNG\r\n\x1a\ngarbage", 400, "IMAGE_DECODE_ERROR"),
    (b"", 400, "EMPTY_FILE"),
])
def test_inspect_bad_images(client, payload, status, code):
    c, _ = client
    r = c.post("/v1/inspect", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_inspect_unsupported_format_and_source(client):
    c, _ = client
    buf = io.BytesIO()
    Image.new("L", (200, 200), 100).save(buf, format="TIFF")
    r = c.post("/v1/inspect", files={"file": ("x.tif", buf.getvalue(), "image/tiff")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    r = c.post("/v1/inspect", files={"file": ("x.png", img_bytes(), "image/png")}, data={"source": "satellite"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_SOURCE"


def test_inspect_too_large(temp_cfg):
    cfg = dict(temp_cfg)
    cfg["quality"] = dict(cfg["quality"], max_upload_mb=0.001)
    c = TestClient(create_app(cfg))
    r = c.post("/v1/inspect", files={"file": ("x.png", img_bytes(), "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_api_without_model_never_predicts(temp_cfg, tmp_path):
    cfg = dict(temp_cfg)
    cfg["inference"] = dict(cfg["inference"], model_path=str(tmp_path / "missing.pt"))
    c = TestClient(create_app(cfg))
    h = c.get("/v1/health").json()
    assert h["model_loaded"] is False and "not found" in h["model_error"]
    r = c.post("/v1/inspect", files={"file": ("x.png", img_bytes(), "image/png")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "predicted_class" not in r.json()
    assert c.get("/v1/model").status_code == 503


def test_openapi_schema_documents_endpoints(client):
    c, _ = client
    paths = c.get("/openapi.json").json()["paths"]
    assert {"/v1/health", "/v1/model", "/v1/inspect"} <= set(paths)


# ----------------------------------------------------------------------------- simulator
def test_simulator_marks_frames_as_simulated(tmp_path):
    sim = SimulatedCameraAdapter(ROOT / "samples")
    with pytest.raises(HardwareNotConnected):
        sim.acquire()
    sim.connect()
    a, b = sim.acquire(), sim.acquire()
    assert a.source == b.source == "simulated_hardware" and a.metadata["simulated"]
    assert a.filename != b.filename and a.image_bytes
    sim.inject_fault("timeout")
    with pytest.raises(HardwareTimeout):
        sim.acquire()
    sim.inject_fault("corrupt_frame")
    assert sim.acquire().metadata["fault"] == "corrupt_frame"
    sim.inject_fault("disconnect")
    with pytest.raises(HardwareNotConnected):
        sim.acquire()
    assert not sim.connected
    with pytest.raises(Exception):
        SimulatedCameraAdapter(tmp_path / "empty").connect()


def test_result_message_formats():
    summary = {"inspection_id": "INS-1", "predicted_class": "scratches", "predicted_display_name": "Scratches",
               "confidence": 0.97, "review_needed": False, "model_version": "v", "source": "hardware",
               "quality": {"suitable": False, "issues": [{"code": "too_dark", "severity": "warning"},
                                                         {"code": "unexpected_size", "severity": "info"}]}}
    msg = build_result_message(summary)
    assert msg["status"] == "ok" and msg["quality_warnings"] == ["too_dark"] and msg["is_model_prediction"]
    err = build_result_message({}, None, {"code": "IMAGE_DECODE_ERROR", "message": "bad"})
    assert err["status"] == "error" and "predicted_class" not in err
    json.dumps(msg)


# ----------------------------------------------------------------------------- drop folder
def test_folder_drop_adapter(tmp_path):
    drop = tmp_path / "inbox"
    ad = FolderDropAdapter(drop, "CAM-1", timeout_s=0.3, poll_s=0.05)
    with pytest.raises(HardwareNotConnected):
        ad.acquire()
    ad.connect()
    with pytest.raises(HardwareTimeout):
        ad.acquire()
    tmp = drop / "frame_001.tmp"
    tmp.write_bytes(img_bytes())
    (drop / "frame_001.json").write_text(json.dumps({"device_id": "CAM-9", "captured_at": "2026-01-01T00:00:00Z"}))
    tmp.rename(drop / "frame_001.png")  # atomic hand-over, as documented
    cap = ad.acquire()
    assert cap.source == "hardware" and cap.device_id == "CAM-9" and cap.captured_at.startswith("2026")
    assert (drop / "processed" / "frame_001.png").is_file()
    ad.send_result(cap, {"status": "ok"})
    assert json.loads((drop / "results" / "frame_001.result.json").read_text()) == {"status": "ok"}
    assert ad.status()["pending_files"] == 0


# ----------------------------------------------------------------------------- hardware loop script
def test_hardware_loop_with_simulator(temp_cfg, tmp_path):
    from scripts import hardware_loop
    cfg_path = temp_cfg["_test_config_path"]
    assert hardware_loop.main(["--config", cfg_path, "--adapter", "simulated", "--count", "3"]) == 0
    repo = InspectionRepository(load_config(cfg_path)["paths"]["inspections_dir"])
    recs = repo.query()
    assert len(recs) == 3 and {r.source for r in recs} == {"simulated_hardware"}


def test_hardware_loop_reports_missing_model(tmp_path):
    from scripts import hardware_loop
    cfg_path = write_config(tmp_path, {"inference": {"model_path": str(tmp_path / "none.pt")},
                                       "paths": {"inspections_dir": str(tmp_path / "i")}})
    assert hardware_loop.main(["--config", str(cfg_path), "--adapter", "simulated", "--count", "1"]) == 2
