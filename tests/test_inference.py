"""Predictor, Grad-CAM, image-quality warnings, review threshold and INT8 export."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from src.config import CLASS_NAMES, load_config
from src.data.preprocessing import ImageLoadError, load_image
from src.data.synthetic import synthetic_texture
from src.inference.explainability import grad_cam, overlay_heatmap
from src.inference.image_quality import assess_image_quality
from src.inference.predictor import ModelUnavailableError, Predictor
from src.models.checkpoints import load_checkpoint
from src.models.cnn import SteelCNN
from src.models.quantization import load_int8, quantize_static, save_int8
from src.training.calibration import calibrate_review_threshold
from tests.conftest import DEPLOYED_MODEL, FINAL_REPORT, REAL_MANIFEST, ROOT, requires_real_model


def _img(cls="scratches", seed=1, size=200) -> Image.Image:
    return Image.fromarray(synthetic_texture(cls, np.random.default_rng(seed), size), mode="L")


def _bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def predictor(tiny_checkpoint):
    return Predictor.from_checkpoint(tiny_checkpoint)


# ----------------------------------------------------------------------------- predictor
def test_prediction_is_valid(predictor):
    r = predictor.predict(_bytes(_img()))
    assert r.class_name in CLASS_NAMES and CLASS_NAMES[r.class_index] == r.class_name
    probs = np.array(list(r.probabilities.values()))
    assert len(probs) == 6 and np.all(np.isfinite(probs)) and abs(probs.sum() - 1) < 1e-5
    assert r.confidence == pytest.approx(probs.max())
    assert r.display_name[0].isupper() and "_" not in r.display_name
    assert r.inference_ms > 0 and r.model_version == predictor.model_version
    assert r.input_array.shape == (64, 64)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP"])
def test_supported_formats(predictor, fmt):
    ref = predictor.predict(_bytes(_img(), "PNG"))
    r = predictor.predict(_bytes(_img(), fmt))
    assert r.class_name == ref.class_name


def test_color_and_odd_sized_inputs(predictor):
    color = _img().convert("RGB")
    assert predictor.predict(_bytes(color)).class_name in CLASS_NAMES
    assert predictor.predict(_bytes(_img(size=200).resize((320, 180)))).class_name in CLASS_NAMES
    assert predictor.predict(_bytes(_img().resize((40, 40)))).class_name in CLASS_NAMES


def test_corrupted_input_raises(predictor):
    with pytest.raises(ImageLoadError):
        predictor.predict(b"\x89PNG\r\n\x1a\nbroken")


def test_no_state_contamination_between_images(predictor):
    a, b = _bytes(_img("scratches", 1)), _bytes(_img("patches", 2))
    r1 = predictor.predict(a)
    predictor.predict(b)
    predictor.explain(b)
    r3 = predictor.predict(a)
    assert r1.probabilities == r3.probabilities


def test_missing_or_invalid_model_never_predicts(tmp_path):
    with pytest.raises(ModelUnavailableError, match="not found"):
        Predictor.from_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "bad.pt").write_bytes(b"not a model")
    with pytest.raises(ModelUnavailableError):
        Predictor.from_checkpoint(tmp_path / "bad.pt")


def test_review_threshold_logic(tiny_checkpoint):
    always = Predictor.from_checkpoint(tiny_checkpoint, review_threshold=1.01)
    never = Predictor.from_checkpoint(tiny_checkpoint, review_threshold=0.0)
    img = _bytes(_img())
    assert always.predict(img).review_needed and "below the review threshold" in always.predict(img).review_reason
    assert not never.predict(img).review_needed


def test_threshold_calibration():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 200)
    probs = np.full((200, 6), 0.02)
    probs[np.arange(200), y] = 0.9  # confident and correct
    wrong = np.arange(10)
    probs[wrong] = 0.02
    probs[wrong, (y[wrong] + 1) % 6] = 0.6  # less confident and wrong
    probs /= probs.sum(1, keepdims=True)
    c = calibrate_review_threshold(probs, y, target_accuracy=0.99, min_threshold=0.5)
    assert c["target_reachable"] and c["val_errors_flagged"] == 10
    assert c["val_accepted_accuracy"] == 1.0
    c2 = calibrate_review_threshold(probs, y, target_accuracy=0.5, min_threshold=0.5)
    assert c2["threshold"] == 0.5  # already met at the floor


# ----------------------------------------------------------------------------- Grad-CAM
@pytest.mark.parametrize("layer,native", [("block4", 8), ("block3", 16)])
def test_gradcam_shape_and_range(predictor, layer, native):
    e = predictor.explain(_bytes(_img()), layer=layer)
    if e.ok:
        assert e.heatmap.shape == (64, 64) and np.all(np.isfinite(e.heatmap))
        assert e.heatmap.min() >= 0 and e.heatmap.max() == pytest.approx(1.0)
        ov = overlay_heatmap(predictor.predict(_bytes(_img())).input_array, e.heatmap, 256)
        assert ov.shape == (256, 256, 3) and ov.dtype == np.uint8
    else:
        assert "No heatmap" in e.message
    assert e.native_resolution == (native, native)


def test_gradcam_every_class_and_bad_args(predictor):
    for i in range(6):
        e = predictor.explain(_bytes(_img()), target_index=i)
        assert e.target_name == CLASS_NAMES[i]
    with pytest.raises(ValueError):
        predictor.explain(_bytes(_img()), layer="block9")
    with pytest.raises(ValueError):
        predictor.explain(_bytes(_img()), target_index=6)


def test_gradcam_refuses_to_fabricate_empty_map():
    m = SteelCNN().eval()
    with torch.no_grad():
        m.classifier.weight.zero_()  # class score independent of the features -> zero gradients
    e = grad_cam(m, torch.rand(1, 1, 64, 64))
    assert not e.ok and e.heatmap is None


# ----------------------------------------------------------------------------- image quality
@pytest.fixture(scope="module")
def cfg():
    return load_config()


THRESH = {"min_mean_brightness": 30, "max_mean_brightness": 230, "min_contrast_std": 7,
          "min_laplacian_var": 10, "max_clipped_fraction": 0.3}


def _codes(report):
    return {i.code for i in report.warnings}


def load_image_png(img):
    return load_image(_bytes(img))


def test_quality_normal_image_has_no_warnings(cfg):
    assert _codes(assess_image_quality(load_image_png(_img()), cfg, 64, THRESH)) == set()


@pytest.mark.parametrize("value,code", [(5, "too_dark"), (250, "too_bright")])
def test_quality_exposure(cfg, value, code):
    arr = np.clip(np.random.default_rng(0).normal(value, 4, (200, 200)), 0, 255).astype(np.uint8)
    assert code in _codes(assess_image_quality(load_image_png(Image.fromarray(arr)), cfg, 64, THRESH))


def test_quality_flat_blurry_colour_shape_size(cfg):
    flat = Image.new("L", (200, 200), 120)
    assert {"low_contrast", "blurry"} <= _codes(assess_image_quality(load_image_png(flat), cfg, 64, THRESH))
    blurred = _img().filter(ImageFilter.GaussianBlur(6))
    assert "blurry" in _codes(assess_image_quality(load_image_png(blurred), cfg, 64, THRESH))
    colour = Image.merge("RGB", [_img(), Image.new("L", (200, 200), 30), Image.new("L", (200, 200), 200)])
    assert "color_image" in _codes(assess_image_quality(load_image_png(colour), cfg, 64, THRESH))
    wide = _img().resize((400, 200))
    assert "non_square" in _codes(assess_image_quality(load_image_png(wide), cfg, 64, THRESH))
    tiny = _img().resize((32, 32))
    assert "too_small" in _codes(assess_image_quality(load_image_png(tiny), cfg, 64, THRESH))
    gray_rgb = _img().convert("RGB")
    rep = assess_image_quality(load_image_png(gray_rgb), cfg, 64, THRESH)
    assert "color_image" not in _codes(rep) and any(i.code == "rgb_grayscale_content" for i in rep.issues)


def test_quality_messages_never_claim_a_defect(cfg):
    flat = Image.new("L", (300, 120), 2)
    rep = assess_image_quality(load_image_png(flat), cfg, 64, THRESH)
    assert rep.warnings
    for issue in rep.issues:
        assert "defect" not in issue.message.lower()


# ----------------------------------------------------------------------------- INT8
def test_int8_quantization_roundtrip(tiny_checkpoint, synthetic_manifest, synthetic_root, tmp_path):
    from src.data.dataset import SteelDefectDataset
    fp32, prep, _ = load_checkpoint(tiny_checkpoint)
    train = SteelDefectDataset(synthetic_manifest, synthetic_root, "train", prep, "none")
    test = SteelDefectDataset(synthetic_manifest, synthetic_root, "test", prep, "none")
    q, engine = quantize_static(fp32, train.x)
    path = save_int8(tmp_path / "q.pt", q, fp32.config, engine, {})
    q2, meta = load_int8(path)
    with torch.no_grad():
        out = q2(test.x)
        ref = fp32(test.x)
    assert out.shape == ref.shape and torch.isfinite(out).all()
    assert (out.argmax(1) == ref.argmax(1)).float().mean() >= 0.8
    assert meta["engine"] == engine


# ----------------------------------------------------------------------------- real model
@pytest.mark.real_data
@requires_real_model
def test_deployed_model_on_demo_samples():
    p = Predictor.from_checkpoint(DEPLOYED_MODEL)
    assert p.threshold_source.startswith("validation")
    ckpt_thr = p.meta["review_threshold"]["threshold"]
    assert p.review_threshold == pytest.approx(ckpt_thr)
    samples = sorted((ROOT / "samples").glob("*.jpg"))
    assert len(samples) == 12
    for s in samples:
        r = p.predict(s)
        assert r.class_name in CLASS_NAMES
        assert r.review_needed == (r.confidence < ckpt_thr)
        e = p.explain(s)
        assert (e.ok and e.heatmap.shape == (p.input_size, p.input_size)) or not e.ok


@pytest.mark.real_data
@requires_real_model
def test_int8_report_uses_same_test_split():
    qpath = ROOT / "artifacts" / "reports" / "quantization" / "quantization.json"
    if not qpath.is_file():
        pytest.skip("BLOCKED: INT8 export has not been run")
    q = json.loads(qpath.read_text())
    assert q["status"] == "ok"
    import pandas as pd
    n_test = int((pd.read_csv(REAL_MANIFEST)["split"] == "test").sum())
    assert q["n_test"] == n_test
    final = json.loads(FINAL_REPORT.read_text())
    assert q["source_model_version"] == final["model_version"]
    assert q["fp32"]["accuracy"] == pytest.approx(final["metrics"]["accuracy"])
    q8, _ = load_int8(ROOT / "artifacts" / "models" / "steel_cnn_int8.pt")
    size = final["preprocessing"]["input_size"]
    assert q8(torch.rand(2, 1, size, size)).shape == (2, 6)
