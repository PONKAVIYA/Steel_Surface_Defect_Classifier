"""Shared preprocessing pipeline and training-only augmentation."""
from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import SteelDefectDataset, augment_tensor
from src.data.preprocessing import (ImageLoadError, PreprocessingConfig, load_image, preprocess_pil,
                                    preprocess_to_tensor, to_grayscale)


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def gradient():
    arr = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (200, 1))
    return Image.fromarray(arr, mode="L")


@pytest.mark.parametrize("size", [64, 100, 128])
def test_resize_and_normalize(gradient, size):
    arr = preprocess_pil(gradient, PreprocessingConfig(input_size=size))
    assert arr.shape == (size, size) and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0
    assert arr.max() > 0.95 and arr.min() < 0.05


def test_tensor_shape(gradient):
    t = preprocess_to_tensor(_encode(gradient, "PNG"), PreprocessingConfig())
    assert tuple(t.shape) == (1, 1, 64, 64) and t.dtype == torch.float32


def test_grayscale_conversion_modes():
    rgb = Image.new("RGB", (10, 10), (100, 100, 100))
    assert to_grayscale(rgb).mode == "L" and np.asarray(to_grayscale(rgb)).max() == 100
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))  # fully transparent -> white background
    assert np.asarray(to_grayscale(rgba)).min() == 255
    i16 = Image.fromarray(np.full((10, 10), 65535, dtype=np.uint16))
    assert np.asarray(to_grayscale(i16)).max() == 255
    pal = rgb.convert("P")
    assert to_grayscale(pal).mode == "L"


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP"])
def test_supported_formats_load(gradient, fmt):
    img = load_image(_encode(gradient, fmt))
    assert img.format == fmt


@pytest.mark.parametrize("fmt", ["TIFF", "GIF"])
def test_unsupported_formats_rejected(gradient, fmt):
    with pytest.raises(ImageLoadError, match="Unsupported image format"):
        load_image(_encode(gradient, fmt))


def test_corrupted_empty_and_missing_inputs(tmp_path, gradient):
    good = _encode(gradient, "PNG")
    with pytest.raises(ImageLoadError, match="Could not decode"):
        load_image(good[: len(good) // 3])
    with pytest.raises(ImageLoadError, match="empty"):
        load_image(b"")
    with pytest.raises(ImageLoadError, match="not found"):
        load_image(tmp_path / "missing.png")
    with pytest.raises(ImageLoadError, match="Could not decode"):
        load_image(b"hello world, not an image")


def test_png_jpeg_bmp_give_near_identical_inputs(gradient):
    cfg = PreprocessingConfig()
    a = preprocess_pil(load_image(_encode(gradient, "PNG")), cfg)
    b = preprocess_pil(load_image(_encode(gradient, "BMP")), cfg)
    c = preprocess_pil(load_image(_encode(gradient, "JPEG")), cfg)
    assert np.array_equal(a, b)
    assert np.abs(a - c).max() < 0.05  # JPEG is lossy


def test_color_image_preprocessed_to_single_channel():
    img = Image.new("RGB", (200, 200), (200, 30, 30))
    arr = preprocess_pil(img, PreprocessingConfig())
    assert arr.shape == (64, 64)


def test_config_validation():
    with pytest.raises(ValueError):
        PreprocessingConfig(input_size=8)
    with pytest.raises(ValueError):
        PreprocessingConfig(resample="magic")
    with pytest.raises(ValueError):
        PreprocessingConfig(grayscale=False)


def test_training_dataset_matches_inference_preprocessing(synthetic_manifest, synthetic_root):
    """The tensors used for training are exactly what the inference path produces for the same file."""
    prep = PreprocessingConfig(input_size=64)
    ds = SteelDefectDataset(synthetic_manifest, synthetic_root, "val", prep, "none")
    for i in range(3):
        path = synthetic_root / ds.rows.iloc[i]["rel_path"]
        expected = preprocess_to_tensor(path, prep)[0]
        assert torch.equal(ds[i][0], expected)


def test_augmentation_only_on_training_split(synthetic_manifest, synthetic_root):
    prep = PreprocessingConfig()
    with pytest.raises(ValueError, match="only allowed on the training split"):
        SteelDefectDataset(synthetic_manifest, synthetic_root, "val", prep, "flip")
    with pytest.raises(ValueError, match="only allowed on the training split"):
        SteelDefectDataset(synthetic_manifest, synthetic_root, "test", prep, "flip_rot90")
    val = SteelDefectDataset(synthetic_manifest, synthetic_root, "val", prep, "none")
    with pytest.raises(ValueError):
        val.with_augmentation("flip", 0)
    assert torch.equal(val[0][0], val[0][0])  # deterministic, unaugmented


def test_augmentations_preserve_content():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    assert torch.equal(augment_tensor(x, "none"), x)
    g = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(64):
        y = augment_tensor(x, "flip_rot90", g)
        assert y.shape == x.shape
        assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)
        seen.add(tuple(y.flatten().tolist()))
    assert len(seen) == 8  # all dihedral orientations reachable
    with pytest.raises(ValueError):
        augment_tensor(x, "elastic")
