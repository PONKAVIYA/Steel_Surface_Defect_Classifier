"""Per-image statistics shared by dataset validation and inference-time quality checks."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image

from src.data.preprocessing import to_grayscale

REFERENCE_SIZE = (200, 200)  # NEU native resolution; blur/brightness stats are computed at this scale


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def highpass_signature(img: Image.Image, size: int = 64, sigma: float = 3.0) -> np.ndarray:
    """Illumination-independent fingerprint for near-duplicate detection.

    The image is reduced to size x size, its low-frequency illumination (Gaussian blur, sigma px)
    is subtracted, and the result is zero-meaned and L2-normalised. Two signatures' dot product is
    their normalised cross-correlation. Plain perceptual hashes / thumbnails are dominated by the
    lighting gradient most NEU images share and flag thousands of false duplicates.
    """
    gray = to_grayscale(img).resize((size, size), Image.Resampling.BOX)
    arr = np.asarray(gray, dtype=np.float32)
    hp = arr - cv2.GaussianBlur(arr, (0, 0), sigma)
    hp -= hp.mean()
    norm = float(np.linalg.norm(hp))
    return (hp / norm if norm > 1e-6 else hp).astype(np.float32)


def channel_difference(img: Image.Image) -> float:
    """Mean absolute difference between colour channels (0 for true grayscale content)."""
    if img.mode not in ("RGB", "RGBA"):
        return 0.0
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float((np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3.0)


def gray_stats(img: Image.Image) -> dict[str, float]:
    """Brightness, contrast, sharpness (variance of Laplacian) and clipping, at the reference scale."""
    gray = to_grayscale(img)
    if gray.size != REFERENCE_SIZE:
        gray = gray.resize(REFERENCE_SIZE, Image.Resampling.BOX if gray.width > REFERENCE_SIZE[0]
                           else Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.uint8)
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "lap_var": float(lap.var()),
        "clipped_frac": float(((arr <= 2) | (arr >= 253)).mean()),
    }
