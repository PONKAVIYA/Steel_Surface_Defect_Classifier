"""Grad-CAM for the custom CNN (Selvaraju et al., 2017).

For a target class c, Grad-CAM weights each channel k of a convolutional activation A^k by the
spatially averaged gradient of the class logit y^c, then sums and rectifies:
    L^c = ReLU( sum_k  mean_ij(dy^c / dA^k_ij) * A^k )
The map is computed on the post-ReLU activation of one conv block (before pooling), so its native
resolution is coarse: 8x8 for block4 at 64 px input (16x16 for block3). It is upsampled to the input
size for display. This is an APPROXIMATE indication of which regions raised the class score - it is
not a segmentation of the physical defect and does not show that the prediction is correct.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.models.cnn import BLOCK_NAMES, SteelCNN
from src.utils.plotting import HEATMAP_CMAP

EXPLANATION_DISCLAIMER = (
    "Approximate model explanation (Grad-CAM): brighter regions increased the score of the explained class. "
    "It is not a measured defect boundary and does not prove the prediction is correct."
)


@dataclass
class ExplanationResult:
    ok: bool
    method: str
    layer: str
    target_index: int
    target_name: str
    heatmap: np.ndarray | None  # (S, S) float32 in [0, 1], aligned with the model input
    native_resolution: tuple[int, int]
    message: str

    def to_dict(self) -> dict:
        return {"ok": self.ok, "method": self.method, "layer": self.layer, "target_index": self.target_index,
                "target_name": self.target_name, "native_resolution": list(self.native_resolution),
                "message": self.message}


def grad_cam(model: SteelCNN, x: torch.Tensor, target_index: int | None = None,
             layer: str = "block4") -> ExplanationResult:
    if layer not in BLOCK_NAMES:
        raise ValueError(f"layer must be one of {BLOCK_NAMES}")
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("grad_cam expects a single image tensor of shape (1, 1, S, S)")
    model.eval()
    with torch.enable_grad():
        logits, act = model.forward_with_activation(x, layer)
        if target_index is None:
            target_index = int(logits.argmax(1).item())
        if not 0 <= target_index < logits.shape[1]:
            raise ValueError(f"target_index {target_index} out of range")
        grads = torch.autograd.grad(logits[0, target_index], act)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True)).detach()
    native = (int(cam.shape[-2]), int(cam.shape[-1]))
    name = CLASS_NAMES[target_index]
    if not torch.isfinite(cam).all():
        return ExplanationResult(False, "Grad-CAM", layer, target_index, name, None, native,
                                 "Grad-CAM produced non-finite values; no heatmap is shown.")
    peak = float(cam.max())
    if peak <= 1e-8:
        return ExplanationResult(
            False, "Grad-CAM", layer, target_index, name, None, native,
            f"Grad-CAM found no positive evidence for '{DISPLAY_NAMES[name]}' at {layer} (all-zero map). "
            "No heatmap is shown rather than displaying an uninformative one.")
    up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].clamp(min=0)
    heat = (up / up.max().clamp(min=1e-12)).cpu().numpy().astype(np.float32)  # normalise after upsampling
    return ExplanationResult(True, "Grad-CAM", layer, target_index, name, heat, native,
                             f"Grad-CAM at {layer} (native {native[0]}x{native[1]}, upsampled to input size).")


def colorize_heatmap(heatmap: np.ndarray, size: int) -> np.ndarray:
    """Heatmap -> RGB uint8 image (size x size) using a perceptually uniform colormap."""
    h = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    rgba = colormaps[HEATMAP_CMAP](np.asarray(h, dtype=np.float32) / 255.0)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(gray_input: np.ndarray, heatmap: np.ndarray, size: int = 256, alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap over the model-input image (both upscaled to `size` for display)."""
    base = Image.fromarray((np.clip(gray_input, 0, 1) * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.BILINEAR)
    base_rgb = np.stack([np.asarray(base)] * 3, axis=-1).astype(np.float32)
    heat_rgb = colorize_heatmap(heatmap, size).astype(np.float32)
    return np.clip((1 - alpha) * base_rgb + alpha * heat_rgb, 0, 255).astype(np.uint8)
