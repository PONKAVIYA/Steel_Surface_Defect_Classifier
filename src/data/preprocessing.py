"""The single preprocessing pipeline shared by training, evaluation, inference and the API.

Pipeline (identical everywhere):
  1. decode image with Pillow (BMP / JPEG / PNG)
  2. convert to 8-bit grayscale ("L"; 16-bit inputs are rescaled to 8-bit first)
  3. resize to input_size x input_size with the configured filter (bilinear, antialiased by Pillow)
  4. convert to float32 and divide by 255 -> values in [0, 1]
  5. add channel (and batch) dimension -> tensor of shape (1, S, S) / (1, 1, S, S)
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"BMP", "JPEG", "PNG"}
SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}

ImageInput = Union[str, Path, bytes, BinaryIO, Image.Image]


class ImageLoadError(ValueError):
    """Raised when an image cannot be decoded or has an unsupported format."""


@dataclass(frozen=True)
class PreprocessingConfig:
    input_size: int = 64
    resample: str = "bilinear"
    grayscale: bool = True
    normalize: str = "unit"

    def __post_init__(self):
        if self.input_size < 16:
            raise ValueError(f"input_size must be >= 16 (got {self.input_size})")
        if self.input_size % 16 != 0 and self.input_size < 64:
            raise ValueError("input_size below 64 must be divisible by 16 for the 4-block CNN")
        if self.resample not in _RESAMPLE:
            raise ValueError(f"Unknown resample '{self.resample}'. Options: {sorted(_RESAMPLE)}")
        if not self.grayscale:
            raise ValueError("The model is trained on grayscale images; grayscale must be true.")
        if self.normalize != "unit":
            raise ValueError("Only 'unit' normalisation (pixel/255) is implemented.")

    @classmethod
    def from_config(cls, cfg: dict, input_size: int | None = None) -> "PreprocessingConfig":
        p = cfg.get("preprocessing", {})
        return cls(
            input_size=int(input_size or p.get("input_size", 64)),
            resample=p.get("resample", "bilinear"),
            grayscale=bool(p.get("grayscale", True)),
            normalize=p.get("normalize", "unit"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(source: ImageInput, check_format: bool = True) -> Image.Image:
    """Decode an image from a path, raw bytes, a file object or an existing PIL image.

    Raises ImageLoadError with a human-readable message for unreadable or unsupported files.
    """
    if isinstance(source, Image.Image):
        return source
    name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "<uploaded image>")
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ImageLoadError(f"Image file not found: {path}")
            with open(path, "rb") as fh:
                data = fh.read()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.read()
        if not data:
            raise ImageLoadError(f"Image '{name}' is empty (0 bytes).")
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated / corrupted files
        img = Image.open(io.BytesIO(data))
        img.load()
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageLoadError(f"Could not decode image '{name}': {exc}") from exc
    if check_format and img.format not in SUPPORTED_FORMATS:
        raise ImageLoadError(
            f"Unsupported image format '{img.format}' for '{name}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    return img


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert any Pillow image mode to 8-bit grayscale ('L')."""
    if img.mode == "L":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        maxval = 65535.0 if arr.max(initial=0) > 255 else 255.0
        arr = np.clip(arr / maxval * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        # Flatten transparency onto white before conversion so transparent pixels are not black.
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    return img.convert("L")


def preprocess_pil(img: Image.Image, cfg: PreprocessingConfig) -> np.ndarray:
    """Grayscale -> resize -> [0,1] float32 array of shape (S, S)."""
    gray = to_grayscale(img)
    size = (cfg.input_size, cfg.input_size)
    if gray.size != size:
        gray = gray.resize(size, resample=_RESAMPLE[cfg.resample])
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr


def preprocess_to_tensor(source: ImageInput, cfg: PreprocessingConfig) -> torch.Tensor:
    """Full pipeline -> tensor of shape (1, 1, S, S) ready for the model."""
    img = load_image(source)
    arr = preprocess_pil(img, cfg)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
