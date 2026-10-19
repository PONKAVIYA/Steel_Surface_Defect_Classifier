"""Install a trained checkpoint as the model used by the app and the API.

Steps: calibrate the review threshold on the validation split, back up the currently deployed model to
artifacts/models/previous/, install the new one, and regenerate the test report in artifacts/reports/final/.

Usage:
    python scripts/deploy_model.py --checkpoint artifacts/models/experimental/steel_cnn_bn_fp32.pt
    python scripts/deploy_model.py --checkpoint artifacts/runs/cnn_64px_flip_s1/best.pt
    python scripts/deploy_model.py --checkpoint artifacts/models/previous/<file>.pt     # roll back

Choose checkpoints on validation results, never on test results. Re-run scripts/export_model.py
afterwards if you use the INT8 model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402
from src.models.checkpoints import CheckpointError  # noqa: E402
from src.training.deploy import deploy_checkpoint  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("deploy_model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--history", default=None, help="training history JSON (default: found next to the checkpoint)")
    ap.add_argument("--reason", default="manual deployment", help="recorded in the checkpoint metadata")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    splits = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits)
    split_meta = load_split_meta(splits)
    root = resolve_path(split_meta.get("dataset_root", cfg["paths"]["dataset_root"]))
    try:
        deploy_checkpoint(cfg, manifest, root, split_meta, resolve_path(args.checkpoint),
                          resolve_path(cfg["inference"]["model_path"]),
                          resolve_path(cfg["paths"]["reports_dir"]) / "final",
                          {"rule": args.reason}, Path(args.history) if args.history else None, log=log.info)
    except CheckpointError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
