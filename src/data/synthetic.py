"""SYNTHETIC texture images for software pipeline tests ONLY.

These procedurally generated patterns loosely mimic the six defect families so that the pipeline
(discovery -> split -> training -> evaluation -> inference -> storage) can be exercised without the
real dataset. Results on synthetic data say nothing about NEU classification accuracy.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import CLASS_NAMES

SHORT = {"crazing": "Cr", "inclusion": "In", "patches": "Pa", "pitted_surface": "PS",
         "rolled-in_scale": "RS", "scratches": "Sc"}


def synthetic_texture(cls: str, rng: np.random.Generator, size: int = 200) -> np.ndarray:
    base = rng.uniform(90, 160)
    img = np.full((size, size), base, np.float32) + rng.normal(0, 6, (size, size)).astype(np.float32)
    if cls == "crazing":
        for _ in range(int(rng.integers(25, 40))):
            p = rng.integers(0, size, 2)
            q = p + rng.integers(-30, 30, 2)
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), base - 25, 1)
    elif cls == "inclusion":
        for _ in range(int(rng.integers(3, 7))):
            c = tuple(int(v) for v in rng.integers(20, size - 20, 2))
            cv2.ellipse(img, c, (int(rng.integers(3, 7)), int(rng.integers(15, 40))), 0, 0, 360, base - 60, -1)
    elif cls == "patches":
        for _ in range(int(rng.integers(2, 5))):
            c = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.circle(img, c, int(rng.integers(25, 60)), base - 45, -1)
        img = cv2.GaussianBlur(img, (0, 0), 6)
    elif cls == "pitted_surface":
        for _ in range(int(rng.integers(60, 120))):
            c = tuple(int(v) for v in rng.integers(0, size, 2))
            cv2.circle(img, c, int(rng.integers(1, 3)), base - 50, -1)
    elif cls == "rolled-in_scale":
        noise = rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32)
        img += cv2.resize(noise, (size, size), interpolation=cv2.INTER_NEAREST) * 25
    elif cls == "scratches":
        vertical = rng.random() < 0.5
        for _ in range(int(rng.integers(1, 4))):
            x = int(rng.integers(10, size - 10))
            w = int(rng.integers(2, 6))
            if vertical:
                img[:, x:x + w] = base + 70
            else:
                img[x:x + w, :] = base + 70
    else:
        raise ValueError(cls)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(root: str | Path, per_class: int = 12, seed: int = 0, layout: str = "neu-det",
                               size: int = 200, classes=CLASS_NAMES) -> Path:
    """layout: 'neu-det' (train_images/<class>_<n>.jpg), 'class-folders' (<class>/<n>.png), 'neu-cls' (Cr_1.bmp)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    for cls in classes:
        for i in range(1, per_class + 1):
            arr = synthetic_texture(cls, rng, size)
            img = Image.fromarray(arr, mode="L")
            if layout == "neu-det":
                path = root / "train_images" / f"{cls}_{i}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.convert("RGB").save(path, quality=95)
            elif layout == "class-folders":
                path = root / cls / f"img_{i:03d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
            elif layout == "neu-cls":
                path = root / f"{SHORT[cls]}_{i}.bmp"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
            else:
                raise ValueError(f"unknown layout {layout}")
    (root / "SYNTHETIC_DATA_README.txt").write_text(
        "SYNTHETIC images for software tests only. Not NEU data. Do not report accuracy on these.\n",
        encoding="utf-8")
    return root
