"""Classification metrics computed from actual predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import CLASS_NAMES, CLASS_TO_INDEX, DISPLAY_NAMES


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """ECE of the top-class softmax score (how far confidence is from observed accuracy)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y_true[mask]).mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted",
                                                    zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    eps = 1e-12
    nll = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))
    ps, inc = CLASS_TO_INDEX["pitted_surface"], CLASS_TO_INDEX["inclusion"]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(pm), "recall": float(rm), "f1": float(fm)},
        "weighted": {"precision": float(pw), "recall": float(rw), "f1": float(fw)},
        "per_class": {
            c: {"display_name": DISPLAY_NAMES[c], "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_labels": list(CLASS_NAMES),
        "pitted_vs_inclusion": {
            "pitted_surface_predicted_as_inclusion": int(cm[ps, inc]),
            "inclusion_predicted_as_pitted_surface": int(cm[inc, ps]),
            "pitted_surface_support": int(cm[ps].sum()),
            "inclusion_support": int(cm[inc].sum()),
        },
        "nll": nll,
        "ece": expected_calibration_error(probs, y_true),
        "mean_confidence": float(probs.max(1).mean()),
    }


def summarize_runs(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
        "values": [float(v) for v in arr],
    }
