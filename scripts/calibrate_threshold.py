"""(Re)calibrate the review-needed threshold of a deployed checkpoint on the VALIDATION split.

run_experiments.py already does this for the model it deploys. Use this script after deploying a
model with `scripts/train_model.py --deploy`.

Usage:
    python scripts/calibrate_threshold.py [--target 0.99] [--min-threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.dataset import SteelDefectDataset  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402
from src.models.checkpoints import CheckpointError, load_checkpoint, save_checkpoint  # noqa: E402
from src.training.calibration import calibrate_review_threshold  # noqa: E402
from src.training.evaluate import predict_tensor_batch  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("calibrate_threshold")
_BASE_KEYS = ("format", "created_at", "torch_version", "model_config", "class_names", "display_names",
              "preprocessing", "model_version", "path", "file_size_bytes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--min-threshold", type=float, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    tc = cfg["inference"]["threshold_calibration"]
    target = args.target if args.target is not None else float(tc["target_accepted_accuracy"])
    min_t = args.min_threshold if args.min_threshold is not None else float(tc["min_threshold"])
    ckpt = resolve_path(args.checkpoint or cfg["inference"]["model_path"])
    try:
        model, prep, meta = load_checkpoint(ckpt)
    except CheckpointError as exc:
        log.error("%s", exc)
        return 2
    splits = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits)
    root = resolve_path(load_split_meta(splits).get("dataset_root", cfg["paths"]["dataset_root"]))
    val = SteelDefectDataset(manifest, root, "val", prep, "none")
    calib = calibrate_review_threshold(predict_tensor_batch(model, val.x), val.y.numpy(), target, min_t)
    extra = {k: v for k, v in meta.items() if k not in _BASE_KEYS}
    extra["review_threshold"] = {k: v for k, v in calib.items() if k != "curve"}
    save_checkpoint(ckpt, model, prep, extra=extra, model_version=meta["model_version"])
    with open(ckpt.parent / "threshold_calibration.json", "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    log.info("Review threshold %.4f (target %.3f reachable=%s): flags %d/%d validation images, catches %d/%d errors",
             calib["threshold"], target, calib["target_reachable"], calib["val_flagged"], calib["n_val"],
             calib["val_errors_flagged"], calib["val_errors_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
