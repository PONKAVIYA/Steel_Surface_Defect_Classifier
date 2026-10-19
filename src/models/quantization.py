"""Optional INT8 post-training static quantization (PyTorch eager mode, torch.ao.quantization).

Workflow: copy the FP32 model -> wrap with Quant/DeQuant stubs -> fuse conv+ReLU per block ->
attach the backend qconfig -> calibrate observers on TRAINING images -> convert to INT8.
The FP32 checkpoint is never modified. Grad-CAM needs gradients and therefore uses the FP32 model.
"""
from __future__ import annotations

import copy
import io
import warnings
from pathlib import Path

import torch
from torch import nn

from src.config import CLASS_NAMES
from src.models.cnn import BLOCK_NAMES, ModelConfig, SteelCNN

INT8_FORMAT = "steel-defect-cnn-int8/v1"


class QuantizationUnavailable(RuntimeError):
    pass


class QuantizableSteelCNN(nn.Module):
    def __init__(self, fp32: SteelCNN):
        super().__init__()
        from torch.ao.quantization import DeQuantStub, QuantStub
        self.quant = QuantStub()
        self.model = copy.deepcopy(fp32).cpu().eval()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.model(self.quant(x)))


def pick_engine() -> str:
    engines = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):
        if e in engines:
            return e
    raise QuantizationUnavailable(f"No supported quantized engine in this PyTorch build ({engines}).")


def _prepare(fp32: SteelCNN, engine: str) -> QuantizableSteelCNN:
    from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare
    if fp32.config.use_batchnorm:
        raise QuantizationUnavailable("INT8 export is implemented for the baseline (no BatchNorm) model only.")
    torch.backends.quantized.engine = engine
    q = QuantizableSteelCNN(fp32).eval()
    fuse_modules(q.model, [[f"{b}.conv", f"{b}.relu"] for b in BLOCK_NAMES], inplace=True)
    q.qconfig = get_default_qconfig(engine)
    prepare(q, inplace=True)
    return q


def quantize_static(fp32: SteelCNN, calibration: torch.Tensor, engine: str | None = None,
                    batch_size: int = 64) -> tuple[nn.Module, str]:
    """Return (INT8 model, engine). `calibration` must be preprocessed TRAINING images (N, 1, S, S)."""
    from torch.ao.quantization import convert
    engine = engine or pick_engine()
    q = _prepare(fp32, engine)
    with torch.no_grad():
        for i in range(0, len(calibration), batch_size):
            q(calibration[i:i + batch_size])
    convert(q, inplace=True)
    return q, engine


def save_int8(path: str | Path, qmodel: nn.Module, model_cfg: ModelConfig, engine: str, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save({"format": INT8_FORMAT, "state_dict": qmodel.state_dict(), "model_config": model_cfg.to_dict(),
                "engine": engine, "class_names": list(CLASS_NAMES), **meta}, buf)
    path.write_bytes(buf.getvalue())
    return path


def load_int8(path: str | Path) -> tuple[nn.Module, dict]:
    """Rebuild the INT8 graph (prepare + convert on an empty FP32 model) and load the quantized weights."""
    from torch.ao.quantization import convert
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"INT8 model not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != INT8_FORMAT or list(payload.get("class_names", [])) != list(CLASS_NAMES):
        raise ValueError(f"{path} is not a compatible INT8 model file")
    cfg = ModelConfig.from_dict(payload["model_config"])
    q = _prepare(SteelCNN(cfg), payload["engine"])
    with warnings.catch_warnings():
        # Converting the empty (uncalibrated) graph warns about default qparams; the real scales and
        # zero points are restored by load_state_dict on the next line.
        warnings.filterwarnings("ignore", message=".*must run observer.*")
        convert(q, inplace=True)
    q.load_state_dict(payload["state_dict"])
    q.eval()
    out = q(torch.zeros(1, 1, cfg.input_size, cfg.input_size))
    if out.shape != (1, len(CLASS_NAMES)) or not torch.isfinite(out).all():
        raise ValueError("INT8 model produced invalid output")
    return q, {k: v for k, v in payload.items() if k != "state_dict"}
