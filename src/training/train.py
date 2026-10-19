"""Training loop with validation tracking, best-checkpoint saving, early stopping and overfitting detection."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import AUGMENTATION_POLICIES, SteelDefectDataset
from src.data.preprocessing import PreprocessingConfig
from src.models.checkpoints import make_model_version, save_checkpoint
from src.models.cnn import ModelConfig, SteelCNN, count_parameters
from src.utils.reproducibility import select_device, set_seed


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainSettings:
    input_size: int = 64
    augment: str = "flip"
    seed: int = 42
    use_batchnorm: bool = False
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    monitor: str = "val_loss"
    device: str = "auto"
    overfit_window: int = 8

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "TrainSettings":
        t = cfg["training"]
        s = cls(
            input_size=int(cfg["preprocessing"]["input_size"]),
            augment=cfg["augmentation"]["policy"],
            seed=int(cfg["seed"]),
            use_batchnorm=bool(cfg["model"].get("use_batchnorm", False)),
            batch_size=int(t["batch_size"]),
            learning_rate=float(t["learning_rate"]),
            max_epochs=int(t["max_epochs"]),
            patience=int(t["early_stopping_patience"]),
            monitor=t.get("monitor", "val_loss"),
            device=t.get("device", "auto"),
            overfit_window=int(t.get("overfit_window", 8)),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        if s.augment not in AUGMENTATION_POLICIES:
            raise TrainingError(f"Unknown augmentation policy {s.augment}")
        if s.monitor not in ("val_loss", "val_acc"):
            raise TrainingError("monitor must be val_loss or val_acc")
        return s

    def run_name(self) -> str:
        bn = "_bn" if self.use_batchnorm else ""
        return f"cnn{bn}_{self.input_size}px_{self.augment}_s{self.seed}"


@dataclass
class TrainResult:
    run_dir: Path
    checkpoint_path: Path
    history: list[dict]
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    stopped_early: bool
    overfitting_warning: str | None
    n_params: int
    train_seconds: float
    model_version: str
    settings: dict = field(default_factory=dict)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)  # raw logits: CrossEntropyLoss applies log-softmax internally
            if not torch.isfinite(loss):
                raise TrainingError(f"Non-finite loss encountered ({loss.item()}). Check data and learning rate.")
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def detect_overfitting(history: list[dict], window: int) -> str | None:
    """Flag when validation loss has risen for `window` epochs since its minimum while training accuracy kept improving."""
    if len(history) <= window:
        return None
    best_idx = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    since = len(history) - 1 - best_idx
    if since < window:
        return None
    vl_best, vl_last = history[best_idx]["val_loss"], history[-1]["val_loss"]
    ta_best, ta_last = history[best_idx]["train_acc"], history[-1]["train_acc"]
    if vl_last > vl_best * 1.10 and ta_last >= ta_best:
        return (f"Possible overfitting: validation loss rose from {vl_best:.4f} (epoch {best_idx + 1}) to "
                f"{vl_last:.4f} while training accuracy went {ta_best:.3f} -> {ta_last:.3f}. "
                "The best-validation checkpoint is kept.")
    return None


def train_model(manifest: pd.DataFrame, dataset_root: str | Path, settings: TrainSettings,
                runs_dir: str | Path, preprocessing_resample: str = "bilinear",
                log: Callable[[str], None] = print, run_name: str | None = None,
                cached_datasets: dict | None = None) -> TrainResult:
    set_seed(settings.seed)
    device = select_device(settings.device)
    prep = PreprocessingConfig(input_size=settings.input_size, resample=preprocessing_resample)
    run_dir = Path(runs_dir) / (run_name or settings.run_name())
    run_dir.mkdir(parents=True, exist_ok=True)

    if cached_datasets is not None and settings.input_size in cached_datasets:
        base_train, val_ds = cached_datasets[settings.input_size]
        train_ds = base_train.with_augmentation(settings.augment, settings.seed)
    else:
        train_ds = SteelDefectDataset(manifest, dataset_root, "train", prep, settings.augment, settings.seed)
        val_ds = SteelDefectDataset(manifest, dataset_root, "val", prep, "none")
        if cached_datasets is not None:
            cached_datasets[settings.input_size] = (train_ds, val_ds)

    g = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model_cfg = ModelConfig(input_size=settings.input_size, use_batchnorm=settings.use_batchnorm)
    model = SteelCNN(model_cfg).to(device)
    n_params = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss()
    log(f"[{run_dir.name}] device={device} params={n_params:,} train={len(train_ds)} val={len(val_ds)} "
        f"augment={settings.augment}")

    history: list[dict] = []
    best_score, best_epoch, best_state, bad_epochs = math.inf, 0, None, 0
    stopped_early = False
    t0 = time.perf_counter()
    for epoch in range(1, settings.max_epochs + 1):
        tl, ta = _run_epoch(model, train_loader, criterion, device, optimizer)
        vl, va = _run_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        score = vl if settings.monitor == "val_loss" else -va
        improved = score < best_score - 1e-6
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            log(f"[{run_dir.name}] epoch {epoch:3d} train_loss={tl:.4f} train_acc={ta:.3f} "
                f"val_loss={vl:.4f} val_acc={va:.3f}{'  *best' if improved else ''}")
        if bad_epochs >= settings.patience:
            stopped_early = True
            log(f"[{run_dir.name}] early stopping at epoch {epoch} (no {settings.monitor} improvement "
                f"for {settings.patience} epochs); best epoch {best_epoch}")
            break
    train_seconds = time.perf_counter() - t0
    if best_state is None:
        raise TrainingError("Training produced no valid checkpoint.")

    overfit = detect_overfitting(history, settings.overfit_window)
    if overfit:
        log(f"[{run_dir.name}] WARNING: {overfit}")
    model.load_state_dict(best_state)
    best = history[best_epoch - 1]
    version = make_model_version(model_cfg, settings.augment, settings.seed, best_state)
    settings_dict = dict(settings.__dict__)
    ckpt = save_checkpoint(
        run_dir / "best.pt", model, prep, model_version=version,
        extra={"training": {
            "settings": settings_dict, "best_epoch": best_epoch, "epochs_run": len(history),
            "best_val_loss": float(best["val_loss"]), "best_val_acc": float(best["val_acc"]),
            "stopped_early": stopped_early, "overfitting_warning": overfit or "",
            "train_seconds": float(train_seconds), "device": str(device), "n_params": int(n_params),
        }},
    )
    with open(run_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump({"settings": settings_dict, "history": history, "best_epoch": best_epoch,
                   "stopped_early": stopped_early, "overfitting_warning": overfit,
                   "model_version": version, "n_params": n_params, "train_seconds": train_seconds}, fh, indent=2)
    log(f"[{run_dir.name}] done in {train_seconds:.1f}s: best epoch {best_epoch} "
        f"val_loss={best['val_loss']:.4f} val_acc={best['val_acc']:.3f} -> {ckpt}")
    return TrainResult(run_dir, ckpt, history, best_epoch, float(best["val_loss"]), float(best["val_acc"]),
                       stopped_early, overfit, n_params, train_seconds, version, settings_dict)
