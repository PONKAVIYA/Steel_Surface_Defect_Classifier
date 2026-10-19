"""Inference: load the trained checkpoint once, then classify images with the training preprocessing.

If the checkpoint is missing or invalid, `Predictor.from_checkpoint` raises ModelUnavailableError.
There is no fallback: the application never shows random or hard-coded predictions.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.data.preprocessing import ImageInput, PreprocessingConfig, load_image, preprocess_pil
from src.inference.explainability import ExplanationResult, grad_cam
from src.models.checkpoints import CheckpointError, load_checkpoint
from src.models.cnn import SteelCNN, count_parameters

PREDICTION_DISCLAIMER = ("Model prediction, not a confirmed engineering diagnosis. The confidence is the network's "
                         "softmax score: it is not a calibrated probability that the prediction is correct.")


class ModelUnavailableError(RuntimeError):
    """No usable trained model - real classification is not possible."""


@dataclass
class PredictionResult:
    class_index: int
    class_name: str
    display_name: str
    confidence: float
    probabilities: dict[str, float]
    review_needed: bool
    review_threshold: float | None
    review_reason: str
    inference_ms: float
    model_version: str
    preprocessing: dict
    input_array: np.ndarray | None = field(default=None, repr=False)  # preprocessed (S, S) image

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("input_array", None)
        return d

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])


class Predictor:
    def __init__(self, model: SteelCNN, preprocessing: PreprocessingConfig, meta: dict,
                 review_threshold: float | None = None):
        self.model = model.eval()
        self.preprocessing = preprocessing
        self.meta = meta
        stored = (meta.get("review_threshold") or {}).get("threshold")
        self.review_threshold = float(review_threshold) if review_threshold is not None else (
            float(stored) if stored is not None else None)
        self.threshold_source = ("config override" if review_threshold is not None else
                                 "validation calibration stored in checkpoint" if stored is not None else "none")
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path: str | Path, review_threshold: float | None = None) -> "Predictor":
        try:
            model, prep, meta = load_checkpoint(path)
        except CheckpointError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        return cls(model, prep, meta, review_threshold)

    @property
    def model_version(self) -> str:
        return str(self.meta.get("model_version", "unknown"))

    @property
    def input_size(self) -> int:
        return self.preprocessing.input_size

    def info(self) -> dict:
        t = self.meta.get("training", {})
        return {
            "model_version": self.model_version,
            "checkpoint": self.meta.get("path"),
            "file_size_bytes": self.meta.get("file_size_bytes"),
            "created_at": self.meta.get("created_at"),
            "input_size": self.input_size,
            "model_config": self.meta.get("model_config"),
            "preprocessing": self.preprocessing.to_dict(),
            "n_params": count_parameters(self.model),
            "class_names": list(CLASS_NAMES),
            "review_threshold": self.review_threshold,
            "review_threshold_source": self.threshold_source,
            "best_val_acc": t.get("best_val_acc"),
            "augment": (t.get("settings") or {}).get("augment"),
            "seed": (t.get("settings") or {}).get("seed"),
            "selection": self.meta.get("selection"),
        }

    def preprocess(self, source: ImageInput) -> tuple[np.ndarray, torch.Tensor]:
        img = load_image(source)
        arr = preprocess_pil(img, self.preprocessing)
        return arr, torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    def predict(self, source: ImageInput) -> PredictionResult:
        t0 = time.perf_counter()
        arr, x = self.preprocess(source)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy().astype(np.float64)
        elapsed = (time.perf_counter() - t0) * 1000
        if probs.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(probs)):
            raise RuntimeError(f"Model returned invalid scores {probs}")
        idx = int(probs.argmax())
        conf = float(probs[idx])
        if self.review_threshold is None:
            review, reason = False, "No review threshold configured."
        elif conf < self.review_threshold:
            review, reason = True, (f"Top score {conf:.3f} is below the review threshold {self.review_threshold:.3f} "
                                    "selected on validation data. A person should check this image.")
        else:
            review, reason = False, f"Top score {conf:.3f} >= review threshold {self.review_threshold:.3f}."
        name = CLASS_NAMES[idx]
        return PredictionResult(
            class_index=idx, class_name=name, display_name=DISPLAY_NAMES[name], confidence=conf,
            probabilities={c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            review_needed=review, review_threshold=self.review_threshold, review_reason=reason,
            inference_ms=elapsed, model_version=self.model_version, preprocessing=self.preprocessing.to_dict(),
            input_array=arr,
        )

    def explain(self, source: ImageInput, target_index: int | None = None, layer: str = "block4") -> ExplanationResult:
        _, x = self.preprocess(source)
        with self._lock:  # serialise gradient computation when shared across app sessions
            return grad_cam(self.model, x, target_index, layer)
