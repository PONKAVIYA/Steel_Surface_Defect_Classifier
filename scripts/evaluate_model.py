"""Evaluate a checkpoint on the held-out test split (or validation) and write metrics + figures.

Usage:
    python scripts/evaluate_model.py                                  # deployed model, test split
    python scripts/evaluate_model.py --checkpoint artifacts/runs/cnn_64px_flip_s0/best.pt --out artifacts/reports/eval_s0
    python scripts/evaluate_model.py --split val

Only evaluate on the test split for models whose configuration is already fixed; never use test
results to choose hyperparameters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, DISPLAY_NAMES, load_config, resolve_path  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402
from src.models.checkpoints import CheckpointError  # noqa: E402
from src.training.evaluate import evaluate_checkpoint  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("evaluate_model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=None, help="report folder (default: artifacts/reports/eval_<split>)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    ckpt = resolve_path(args.checkpoint or cfg["inference"]["model_path"])
    splits = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits)
    meta = load_split_meta(splits)
    root = resolve_path(meta.get("dataset_root", cfg["paths"]["dataset_root"]))
    out = resolve_path(args.out or f"{cfg['paths']['reports_dir']}/eval_{args.split}")
    hist = ckpt.parent / "training_history.json"
    try:
        rep = evaluate_checkpoint(ckpt, manifest, root, out, args.split, int(cfg["evaluation"]["latency_runs"]),
                                  int(cfg["evaluation"]["latency_warmup"]),
                                  history_path=hist if hist.is_file() else None, split_meta=meta)
    except CheckpointError as exc:
        log.error("%s", exc)
        return 2
    m = rep["metrics"]
    print(f"\nModel {rep['model_version']} on {args.split} split ({m['n']} images)")
    print(f"accuracy {m['accuracy']:.4f} | macro F1 {m['macro']['f1']:.4f} | weighted F1 {m['weighted']['f1']:.4f}")
    print(f"{'class':<17}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for c in CLASS_NAMES:
        pc = m["per_class"][c]
        print(f"{DISPLAY_NAMES[c]:<17}{pc['precision']:>10.3f}{pc['recall']:>9.3f}{pc['f1']:>8.3f}{pc['support']:>9d}")
    print(f"median latency (pipeline, batch 1): {rep['latency']['pipeline_ms']['median']:.2f} ms | "
          f"model file {rep['model_size_bytes'] / 1024:.1f} KB")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
