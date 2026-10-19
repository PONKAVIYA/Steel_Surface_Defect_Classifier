"""Deploy a trained checkpoint: calibrate the review threshold on VALIDATION, install it as the app/API
model (backing up the previous one), and write the final test-split report with figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.data.dataset import SteelDefectDataset
from src.models.checkpoints import CheckpointError, load_checkpoint, save_checkpoint
from src.training.calibration import calibrate_review_threshold
from src.training.evaluate import evaluate_checkpoint, predict_tensor_batch

_BASE_KEYS = ("format", "created_at", "torch_version", "model_config", "class_names", "display_names",
              "preprocessing", "model_version", "path", "file_size_bytes")


def find_history(checkpoint: Path) -> Path | None:
    for cand in (checkpoint.parent / "history.json", checkpoint.with_suffix(".history.json")):
        if cand.is_file():
            return cand
    return None


def deploy_checkpoint(cfg: dict, manifest: pd.DataFrame, dataset_root: Path, split_meta: dict, checkpoint: Path,
                      dest: Path, final_dir: Path, selection: dict, history: Path | None = None,
                      log=print) -> dict:
    checkpoint, dest, final_dir = Path(checkpoint), Path(dest), Path(final_dir)
    model, prep, meta = load_checkpoint(checkpoint)
    val_ds = SteelDefectDataset(manifest, dataset_root, "val", prep, "none")
    icfg = cfg["inference"]["threshold_calibration"]
    calib = calibrate_review_threshold(predict_tensor_batch(model, val_ds.x), val_ds.y.numpy(),
                                       float(icfg["target_accepted_accuracy"]), float(icfg["min_threshold"]))

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.resolve() != checkpoint.resolve():
        try:
            old_version = load_checkpoint(dest)[2]["model_version"]
        except CheckpointError:
            old_version = "unreadable"
        if old_version != meta["model_version"]:
            backup = dest.parent / "previous" / f"{dest.stem}__{old_version}{dest.suffix}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, backup)
            old_hist = dest.parent / "training_history.json"
            if old_hist.is_file():
                shutil.copy2(old_hist, backup.with_suffix(".history.json"))
            log(f"Previous model backed up to {backup}")

    extra = {k: v for k, v in meta.items() if k not in _BASE_KEYS}
    extra["review_threshold"] = {k: v for k, v in calib.items() if k != "curve"}
    extra["selection"] = {**selection, "split_seed": split_meta.get("seed"), "source_checkpoint": str(checkpoint)}
    save_checkpoint(dest, model, prep, extra=extra, model_version=meta["model_version"])
    with open(dest.parent / "threshold_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    history = history or find_history(checkpoint)
    hist_dest = dest.parent / "training_history.json"
    if history is not None:
        if Path(history).resolve() != hist_dest.resolve():
            shutil.copy2(history, hist_dest)
    elif hist_dest.is_file():
        hist_dest.unlink()  # never pair a model with another model's training curves

    if final_dir.is_dir():
        for f in final_dir.glob("*"):
            if f.is_file():
                f.unlink()  # remove figures of the previously deployed model
    rep = evaluate_checkpoint(dest, manifest, dataset_root, final_dir, "test", int(cfg["evaluation"]["latency_runs"]),
                              int(cfg["evaluation"]["latency_warmup"]),
                              history_path=hist_dest if hist_dest.is_file() else None, split_meta=split_meta)
    rep["review_threshold"] = extra["review_threshold"]
    preds = pd.read_csv(final_dir / "predictions_test.csv")
    flagged = preds["confidence"] < calib["threshold"]
    wrong = preds["predicted"] != preds["label"]
    rep["review_on_test"] = {
        "threshold": calib["threshold"], "flagged": int(flagged.sum()), "n": int(len(preds)),
        "errors_total": int(wrong.sum()), "errors_flagged": int((wrong & flagged).sum()),
        "accepted_accuracy": float((~wrong[~flagged]).mean()) if (~flagged).any() else None,
    }
    with open(final_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    log(f"Deployed {meta['model_version']} -> {dest} | review threshold {calib['threshold']:.4f} | "
        f"test accuracy {rep['metrics']['accuracy']:.4f}")
    return {"path": str(dest), "model_version": meta["model_version"], "review_threshold": calib["threshold"],
            "test_accuracy": rep["metrics"]["accuracy"], "test_macro_f1": rep["metrics"]["macro"]["f1"]}
