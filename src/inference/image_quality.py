"""Inspection image quality checks.

Thresholds come from the TRAINING split statistics (artifacts/splits/quality_reference.json):
an image outside the range the model was trained on gets a warning that the prediction may be
unreliable. A quality warning never means the steel is defective, and a clean report does not mean
the prediction is correct.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from PIL import Image

from src.config import resolve_path
from src.data.image_stats import channel_difference, gray_stats
from src.data.preprocessing import SUPPORTED_FORMATS

QUALITY_DISCLAIMER = ("Quality checks describe the photo, not the steel: a warning means the prediction may be "
                      "unreliable, not that the surface is defective.")


@dataclass
class QualityIssue:
    code: str
    severity: str  # "warning" (may affect reliability) | "info"
    message: str


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    thresholds_source: str = ""

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def suitable(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {"suitable": self.suitable, "issues": [asdict(i) for i in self.issues], "stats": self.stats,
                "thresholds_source": self.thresholds_source}


def load_quality_thresholds(cfg: dict) -> tuple[dict, str]:
    ref_path = resolve_path(cfg["paths"]["splits_dir"]) / "quality_reference.json"
    if ref_path.is_file():
        try:
            with open(ref_path, encoding="utf-8") as fh:
                ref = json.load(fh)
            return dict(ref["thresholds"]), f"training-split statistics ({ref.get('n_images', '?')} images)"
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return dict(cfg["quality"]["fallback"]), "configured fallback values"


def assess_image_quality(img: Image.Image, cfg: dict, input_size: int, thresholds: dict | None = None,
                         source_name: str = "") -> QualityReport:
    qcfg = cfg["quality"]
    if thresholds is None:
        thresholds, src = load_quality_thresholds(cfg)
    else:
        src = "provided thresholds"
    issues: list[QualityIssue] = []
    w, h = img.size
    fmt = img.format or "unknown"

    if img.format and img.format not in SUPPORTED_FORMATS:
        issues.append(QualityIssue("unsupported_format", "warning",
                                   f"Format {fmt} is not one of the supported formats {sorted(SUPPORTED_FORMATS)}."))
    exp_w, exp_h = qcfg["expected_size"]
    if (w, h) != (exp_w, exp_h):
        issues.append(QualityIssue("unexpected_size", "info",
                                   f"Image is {w}x{h}; training images were {exp_w}x{exp_h}. "
                                   f"It is resized to {input_size}x{input_size} for the model."))
    aspect_dev = abs(w / h - 1.0) if h else 1.0
    if aspect_dev > float(qcfg["max_aspect_deviation"]):
        issues.append(QualityIssue("non_square", "warning",
                                   f"Aspect ratio {w}:{h} is not square; resizing to a square stretches surface "
                                   "features. Crop a square region of interest for a more reliable result."))
    if min(w, h) < input_size:
        issues.append(QualityIssue("too_small", "warning",
                                   f"Image ({w}x{h}) is smaller than the model input ({input_size}x{input_size}); "
                                   "upsampling cannot recover fine detail."))
    if max(w, h) > 4 * exp_w:
        issues.append(QualityIssue("large_field_of_view", "warning",
                                   f"Image is much larger ({w}x{h}) than the training patches ({exp_w}x{exp_h}). "
                                   "Small defects may vanish when downscaled; inspect a cropped patch instead."))
    if "A" in img.getbands():
        issues.append(QualityIssue("alpha_channel", "info", "Transparency is flattened onto white before conversion."))
    chdiff = channel_difference(img)
    if chdiff > float(qcfg["max_color_channel_diff"]):
        issues.append(QualityIssue("color_image", "warning",
                                   f"Colour image (mean channel difference {chdiff:.1f}). The model was trained on "
                                   "grayscale images; it is converted using luminance, but colour cues are lost "
                                   "and colour lighting may shift results."))
    elif img.mode in ("RGB", "RGBA"):
        issues.append(QualityIssue("rgb_grayscale_content", "info",
                                   "Stored as RGB but the content is grayscale; converted to a single channel."))

    stats = gray_stats(img)
    stats["channel_diff"] = chdiff
    stats["width"], stats["height"], stats["format"] = w, h, fmt
    t = thresholds
    if stats["mean"] < float(t["min_mean_brightness"]):
        issues.append(QualityIssue("too_dark", "warning",
                                   f"Very dark image (mean brightness {stats['mean']:.0f}/255; training images "
                                   f">= {t['min_mean_brightness']:.0f}). Check lighting or exposure."))
    if stats["mean"] > float(t["max_mean_brightness"]):
        issues.append(QualityIssue("too_bright", "warning",
                                   f"Very bright image (mean brightness {stats['mean']:.0f}/255; training images "
                                   f"<= {t['max_mean_brightness']:.0f}). Check lighting or exposure."))
    if stats["std"] < float(t["min_contrast_std"]):
        issues.append(QualityIssue("low_contrast", "warning",
                                   f"Very low contrast (std {stats['std']:.1f}; training images >= "
                                   f"{t['min_contrast_std']:.1f}). The surface texture may not be visible."))
    if stats["lap_var"] < float(t["min_laplacian_var"]):
        issues.append(QualityIssue("blurry", "warning",
                                   f"Little fine detail (variance of Laplacian {stats['lap_var']:.1f}; training "
                                   f"images >= {t['min_laplacian_var']:.1f}). The image may be out of focus or "
                                   "motion-blurred."))
    if stats["clipped_frac"] > float(t["max_clipped_fraction"]):
        issues.append(QualityIssue("clipped", "warning",
                                   f"{stats['clipped_frac'] * 100:.0f}% of pixels are saturated black/white "
                                   "(over- or under-exposed)."))
    return QualityReport(issues, stats, src)
