"""Saving and safely loading model checkpoints (weights + class mapping + configuration together)."""
from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.data.preprocessing import PreprocessingConfig
from src.models.cnn import ModelConfig, SteelCNN

CHECKPOINT_FORMAT = "steel-defect-cnn/v1"
REQUIRED_KEYS = {"format", "model_state", "model_config", "class_names", "preprocessing", "model_version"}


class CheckpointError(RuntimeError):
    """Missing, corrupted or incompatible checkpoint."""


def weights_digest(state_dict: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:10]


def make_model_version(model_cfg: ModelConfig, augment: str, seed: int, state_dict: dict) -> str:
    bn = "-bn" if model_cfg.use_batchnorm else ""
    return f"steelcnn{bn}-{model_cfg.input_size}px-{augment}-s{seed}-{weights_digest(state_dict)}"


def save_checkpoint(path: str | Path, model: SteelCNN, preprocessing: PreprocessingConfig,
                    extra: dict[str, Any] | None = None, model_version: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": str(torch.__version__),  # TorchVersion objects are rejected by weights_only loading
        "model_state": state,
        "model_config": model.config.to_dict(),
        "class_names": list(CLASS_NAMES),
        "display_names": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
        "preprocessing": preprocessing.to_dict(),
        "model_version": model_version or f"steelcnn-{model.config.input_size}px-{weights_digest(state)}",
    }
    if extra:
        payload.update(extra)
    # Write atomically so a crash never leaves a half-written checkpoint behind.
    buf = io.BytesIO()
    torch.save(payload, buf)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"
                    ) -> tuple[SteelCNN, PreprocessingConfig, dict]:
    """Load and validate a checkpoint. Returns (model in eval mode, preprocessing config, metadata).

    Validation: file exists and decodes with weights_only=True (no arbitrary code execution),
    format tag, class list identical to the fixed mapping, weights match the architecture
    (strict), and a dummy forward pass yields finite logits of shape (1, 6).
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Model checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # corrupted / truncated / not a torch file
        raise CheckpointError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint {path} has an unexpected structure.")
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise CheckpointError(f"Checkpoint {path} is missing fields: {sorted(missing)}")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError(f"Unsupported checkpoint format '{payload['format']}' (expected {CHECKPOINT_FORMAT}).")
    if list(payload["class_names"]) != list(CLASS_NAMES):
        raise CheckpointError(
            f"Checkpoint class order {payload['class_names']} does not match the fixed mapping {list(CLASS_NAMES)}."
        )
    try:
        model_cfg = ModelConfig.from_dict(payload["model_config"])
        prep = PreprocessingConfig(**payload["preprocessing"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"Invalid configuration stored in {path}: {exc}") from exc
    if prep.input_size != model_cfg.input_size:
        raise CheckpointError("Preprocessing input_size does not match the model input_size.")
    model = SteelCNN(model_cfg)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"Weights in {path} do not match the architecture: {exc}") from exc
    model.to(map_location).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, model_cfg.input_size, model_cfg.input_size, device=map_location))
    if out.shape != (1, len(CLASS_NAMES)) or not torch.isfinite(out).all():
        raise CheckpointError(f"Checkpoint {path} produced invalid output {tuple(out.shape)}.")
    meta = {k: v for k, v in payload.items() if k != "model_state"}
    meta["path"] = str(path)
    meta["file_size_bytes"] = path.stat().st_size
    return model, prep, meta
