"""Train one model (one configuration, one seed) on the training split, selecting on validation.

Usage:
    python scripts/train_model.py                                   # baseline from configs/default.yaml
    python scripts/train_model.py --input-size 128 --augment flip_rot90 --seed 1
    python scripts/train_model.py --deploy                          # also deploy it (see scripts/deploy_model.py)

Training and checkpoint selection never use the test split. With --deploy, the deployed model's
test report is written afterwards (artifacts/reports/final/), as in scripts/deploy_model.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.dataset import AUGMENTATION_POLICIES  # noqa: E402
from src.data.splitting import SplitError, load_manifest, load_split_meta  # noqa: E402
from src.training.deploy import deploy_checkpoint  # noqa: E402
from src.training.train import TrainingError, TrainSettings, train_model  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("train_model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--input-size", type=int, choices=[64, 100, 128], default=None)
    ap.add_argument("--augment", choices=AUGMENTATION_POLICIES, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batchnorm", action="store_true", help="BatchNorm experiment (not the baseline)")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--deploy", action="store_true",
                    help="deploy the best checkpoint: calibrate threshold, back up old model, write test report")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    try:
        manifest = load_manifest(splits_dir)
    except SplitError as exc:
        log.error(str(exc))
        return 2
    meta = load_split_meta(splits_dir)
    dataset_root = resolve_path(meta.get("dataset_root", cfg["paths"]["dataset_root"]))
    settings = TrainSettings.from_config(
        cfg, input_size=args.input_size, augment=args.augment, seed=args.seed,
        use_batchnorm=True if args.batchnorm else None, max_epochs=args.max_epochs, patience=args.patience,
    )
    try:
        result = train_model(manifest, dataset_root, settings, resolve_path(cfg["paths"]["runs_dir"]),
                             cfg["preprocessing"]["resample"], log=log.info, run_name=args.run_name)
    except (TrainingError, FileNotFoundError, ValueError) as exc:
        log.error("Training failed: %s", exc)
        return 1
    if args.deploy:
        deploy_checkpoint(cfg, manifest, dataset_root, meta, result.checkpoint_path,
                          resolve_path(cfg["inference"]["model_path"]),
                          resolve_path(cfg["paths"]["reports_dir"]) / "final",
                          {"rule": "manual: scripts/train_model.py --deploy"}, log=log.info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
