"""Streamlit app smoke tests using streamlit.testing (headless, no browser)."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config
from src.database.repository import InspectionRepository
from tests.conftest import ROOT, write_config

APP = str(ROOT / "app.py")
PAGES = ["Overview", "Inspect surface", "Inspection history", "Analytics", "Model evaluation"]


def _app(monkeypatch, cfg_path) -> AppTest:
    monkeypatch.setenv("STEEL_CONFIG", str(cfg_path))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _goto(at: AppTest, page: str) -> AppTest:
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, at.exception
    return at


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [t.value for t in at.title]
    parts += [e.value for e in at.error] + [i.value for i in at.info] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [h.value for h in at.header] + [h.value for h in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_all_pages_render(monkeypatch, temp_cfg):
    at = _app(monkeypatch, temp_cfg["_test_config_path"])
    for page in PAGES:
        _goto(at, page)
    assert "No evaluation results are available yet" in _text(at)  # temp config has no reports


def test_inspect_classify_and_save_workflow(monkeypatch, temp_cfg):
    at = _app(monkeypatch, temp_cfg["_test_config_path"])
    _goto(at, "Inspect surface")
    at.radio(key="inspect_source").set_value("Use a demo sample").run()
    assert not at.button(key="classify_btn").disabled
    at.button(key="classify_btn").click().run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Model prediction with a softmax score" in text
    assert "not a confirmed engineering diagnosis" in text
    assert "Approximate model explanation" in text
    at.text_area(key=[k.key for k in at.text_area][0]).input("demo note").run()
    at.button(key="save_btn").click().run()
    assert not at.exception, at.exception
    assert "Saved as inspection INS-" in _text(at)
    repo = InspectionRepository(load_config(temp_cfg["_test_config_path"])["paths"]["inspections_dir"])
    recs = repo.query()
    assert len(recs) == 1 and recs[0].source == "demo_sample" and recs[0].operator_note == "demo note"

    # Second image: the previous result must not leak into the new one.
    at.selectbox(key="demo_sample").set_value(at.selectbox(key="demo_sample").options[-1]).run()
    assert "Model prediction with a softmax score" not in _text(at)

    _goto(at, "Inspection history")
    assert "1 of 1 inspections match" in _text(at)
    _goto(at, "Analytics")
    assert not at.exception


def test_simulated_camera_capture(monkeypatch, temp_cfg):
    at = _app(monkeypatch, temp_cfg["_test_config_path"])
    _goto(at, "Inspect surface")
    at.radio(key="inspect_source").set_value("Capture from camera").run()
    at.button(key="capture_btn").click().run()
    at.button(key="classify_btn").click().run()
    assert not at.exception, at.exception
    text = _text(at)
    assert "Simulated camera" in text and "Model prediction with a softmax score" in text


def test_app_without_model_disables_classification(monkeypatch, tmp_path):
    cfg_path = write_config(tmp_path, {"inference": {"model_path": str(tmp_path / "no_model.pt")},
                                       "paths": {"inspections_dir": str(tmp_path / "insp"),
                                                 "reports_dir": str(tmp_path / "reports")}})
    at = _app(monkeypatch, cfg_path)
    assert "No trained model is available" in _text(at)
    _goto(at, "Inspect surface")
    at.radio(key="inspect_source").set_value("Use a demo sample").run()
    assert at.button(key="classify_btn").disabled
    assert "real classification is unavailable" in _text(at)
    assert "Model prediction with a softmax score" not in _text(at)


@pytest.mark.real_data
def test_app_with_deployed_model_and_reports(monkeypatch, tmp_path):
    from tests.conftest import DEPLOYED_MODEL, FINAL_REPORT
    if not (DEPLOYED_MODEL.is_file() and FINAL_REPORT.is_file()):
        pytest.skip("BLOCKED: trained model / evaluation report not present")
    cfg_path = write_config(tmp_path, {"paths": {"inspections_dir": str(tmp_path / "insp")}})
    at = _app(monkeypatch, cfg_path)
    assert "Measured on the held-out test split" in _text(at)
    _goto(at, "Model evaluation")
    text = _text(at)
    assert "held-out test split" in text and "Per-class results" in text
    _goto(at, "Inspect surface")
    at.radio(key="inspect_source").set_value("Use a demo sample").run()
    at.button(key="classify_btn").click().run()
    assert not at.exception, at.exception
    assert "Model prediction with a softmax score" in _text(at)
