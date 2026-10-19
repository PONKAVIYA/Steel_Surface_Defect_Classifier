"""Evaluation on a held-out split: metrics, figures, latency and model size."""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.data.dataset import SteelDefectDataset
from src.data.preprocessing import PreprocessingConfig, load_image, preprocess_pil
from src.models.checkpoints import load_checkpoint
from src.training.metrics import compute_metrics
from src.utils import plotting
from src.utils.plotting import plt


def hardware_info() -> dict:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@torch.no_grad()
def predict_tensor_batch(model, x: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(torch.softmax(model(x[i:i + batch_size]), dim=1).cpu())
    return torch.cat(out).numpy()


def measure_latency(model, prep: PreprocessingConfig, sample_path: str | Path,
                    runs: int = 200, warmup: int = 20) -> dict:
    """Per-image latency on CPU at batch size 1.

    model_ms: forward + softmax on an already-preprocessed tensor.
    pipeline_ms: read file bytes, decode, grayscale, resize, normalise, forward, softmax.
    """
    model.eval()
    data = Path(sample_path).read_bytes()
    x = torch.from_numpy(preprocess_pil(load_image(data), prep)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        for _ in range(warmup):
            torch.softmax(model(x), 1)
        model_t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            torch.softmax(model(x), 1)
            model_t.append((time.perf_counter() - t0) * 1000)
        pipe_t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            xi = torch.from_numpy(preprocess_pil(load_image(data), prep)).unsqueeze(0).unsqueeze(0)
            torch.softmax(model(xi), 1)
            pipe_t.append((time.perf_counter() - t0) * 1000)

    def stats(v):
        v = sorted(v)
        return {"median": statistics.median(v), "mean": statistics.fmean(v),
                "p95": v[int(0.95 * (len(v) - 1))], "runs": len(v)}
    return {"model_ms": stats(model_t), "pipeline_ms": stats(pipe_t), "batch_size": 1}


def plot_confusion_matrix(cm, path: Path, title: str) -> None:
    cm = np.asarray(cm)
    names = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.grid(False)
    ax.imshow(cm, cmap=plotting.SEQ_BLUE, vmin=0, vmax=max(1, cm.max()))
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] or i == j:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if cm[i, j] > thresh else plotting.INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], path: Path, best_epoch: int | None, title: str) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, key, label in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")):
        ax.plot(ep, [h[f"train_{key}"] for h in history], color=plotting.SERIES_TRAIN, label="Training")
        ax.plot(ep, [h[f"val_{key}"] for h in history], color=plotting.SERIES_VAL, label="Validation")
        if best_epoch:
            ax.axvline(best_epoch, color=plotting.INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(best_epoch, 1.01, f"selected epoch {best_epoch}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=plotting.INK_SECONDARY)
        ax.set_xlabel("Epoch")
        ax.set_title(label, loc="left", pad=14)
        ax.legend(loc="lower left" if key == "acc" else "upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=plotting.INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(rows: pd.DataFrame, probs: np.ndarray, dataset_root: Path, path: Path,
                  n_correct: int = 6, n_wrong: int = 6, seed: int = 0) -> None:
    """Grid of representative correct and incorrect test predictions."""
    y = rows["label_index"].to_numpy()
    pred = probs.argmax(1)
    rng = np.random.default_rng(seed)
    correct = np.where(pred == y)[0]
    wrong = np.where(pred != y)[0]
    # One correct example per class where possible, then fill.
    pick_c = []
    for c in range(len(CLASS_NAMES)):
        idx = correct[y[correct] == c]
        if len(idx):
            pick_c.append(int(rng.choice(idx)))
    pick_c = pick_c[:n_correct]
    pick_w = list(wrong[np.argsort(-probs[wrong].max(1))][:n_wrong]) if len(wrong) else []
    sel = [(i, True) for i in pick_c] + [(int(i), False) for i in pick_w]
    if not sel:
        return
    ncol = 6
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.75 * nrow), gridspec_kw={"hspace": 0.55})
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, (i, ok) in enumerate(sel):
        ax = axes.flat[k]
        img = load_image(dataset_root / rows.iloc[i]["rel_path"]).convert("L")
        ax.imshow(np.asarray(img), cmap="gray", vmin=0, vmax=255)
        t = DISPLAY_NAMES[CLASS_NAMES[y[i]]]
        p = DISPLAY_NAMES[CLASS_NAMES[pred[i]]]
        mark = "Correct" if ok else "Wrong"
        ax.set_title(f"{mark}\ntrue: {t}\npred: {p} ({probs[i].max():.2f})", fontsize=7.5,
                     color=plotting.GOOD if ok else plotting.CRITICAL, loc="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.08, hspace=0.45)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate_checkpoint(checkpoint: str | Path, manifest: pd.DataFrame, dataset_root: str | Path,
                        out_dir: str | Path, split: str = "test", latency_runs: int = 200,
                        latency_warmup: int = 20, history_path: str | Path | None = None,
                        split_meta: dict | None = None, make_figures: bool = True) -> dict:
    """Evaluate one checkpoint on one split and write metrics.json (+ figures) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_root)
    model, prep, meta = load_checkpoint(checkpoint)
    ds = SteelDefectDataset(manifest, dataset_root, split, prep, "none")
    probs = predict_tensor_batch(model, ds.x)
    y = ds.y.numpy()
    metrics = compute_metrics(y, probs)
    rows = ds.rows
    if "multi_defect" in rows.columns:
        md = rows["multi_defect"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        pred = probs.argmax(1)
        metrics["subsets"] = {
            "single_defect_images": {"n": int((~md).sum()),
                                     "accuracy": float((pred[~md] == y[~md]).mean()) if (~md).any() else None},
            "multi_defect_images": {"n": int(md.sum()),
                                    "accuracy": float((pred[md] == y[md]).mean()) if md.any() else None},
        }
    sample = dataset_root / rows.iloc[0]["rel_path"]
    latency = measure_latency(model, prep, sample, latency_runs, latency_warmup)
    report = {
        "split": split,
        "checkpoint": str(checkpoint),
        "model_version": meta["model_version"],
        "model_config": meta["model_config"],
        "preprocessing": meta["preprocessing"],
        "training": meta.get("training", {}),
        "split_meta": split_meta or {},
        "metrics": metrics,
        "latency": latency,
        "model_size_bytes": int(meta["file_size_bytes"]),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "hardware": hardware_info(),
        "evaluated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }
    pred_df = rows[["rel_path", "label"]].copy()
    pred_df["predicted"] = [CLASS_NAMES[i] for i in probs.argmax(1)]
    pred_df["confidence"] = probs.max(1)
    for i, c in enumerate(CLASS_NAMES):
        pred_df[f"p_{c}"] = probs[:, i]
    pred_df.to_csv(out_dir / f"predictions_{split}.csv", index=False)
    if make_figures:
        plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png",
                              f"Confusion matrix - {split} split (n={metrics['n']})")
        plot_examples(rows, probs, dataset_root, out_dir / "prediction_examples.png")
        hist_file = Path(history_path) if history_path else Path(checkpoint).parent / "history.json"
        if hist_file.is_file():
            with open(hist_file, encoding="utf-8") as fh:
                h = json.load(fh)
            plot_training_curves(h["history"], out_dir / "training_curves.png", h.get("best_epoch"),
                                 f"Training history - {meta['model_version']}")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report
