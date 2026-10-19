"""Validation-based selection of the review-needed threshold on the top softmax score."""
from __future__ import annotations

import numpy as np


def calibrate_review_threshold(probs: np.ndarray, y_true: np.ndarray, target_accuracy: float = 0.99,
                               min_threshold: float = 0.5) -> dict:
    """Pick the lowest threshold t >= min_threshold such that validation predictions with
    top score >= t reach `target_accuracy`. Predictions below t are shown as "review needed".

    This is a heuristic on an uncalibrated softmax score: it limits how often a confident
    prediction was wrong *on the validation split*; it is not a guarantee on new data.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    conf = probs.max(1)
    correct = probs.argmax(1) == y_true
    candidates = np.unique(np.concatenate([[min_threshold], conf[conf >= min_threshold]]))
    curve = []
    chosen = None
    for t in candidates:
        acc_mask = conf >= t
        if not acc_mask.any():
            continue
        acc = float(correct[acc_mask].mean())
        curve.append({"threshold": float(t), "coverage": float(acc_mask.mean()), "accepted_accuracy": acc})
        if chosen is None and acc >= target_accuracy:
            chosen = float(t)
    reachable = chosen is not None
    if not reachable:
        chosen = float(max(min_threshold, conf.max()))
    flagged = conf < chosen
    errors = ~correct
    return {
        "threshold": chosen,
        "target_accepted_accuracy": target_accuracy,
        "min_threshold": min_threshold,
        "target_reachable": reachable,
        "n_val": int(len(y_true)),
        "val_accuracy": float(correct.mean()),
        "val_coverage": float((~flagged).mean()),
        "val_accepted_accuracy": float(correct[~flagged].mean()) if (~flagged).any() else None,
        "val_flagged": int(flagged.sum()),
        "val_errors_total": int(errors.sum()),
        "val_errors_flagged": int((errors & flagged).sum()),
        "method": ("lowest top-softmax threshold >= min_threshold whose accepted validation predictions reach "
                   "the target accuracy"),
        "curve": curve[:: max(1, len(curve) // 200)],
    }
