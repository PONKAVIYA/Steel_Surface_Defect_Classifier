"""Copy a few TEST-split images per class into samples/ for the app's demo mode and the hardware simulator.

These images are never used for training or model selection. Usage:
    python scripts/export_samples.py --per-class 2
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args(argv)
    cfg = load_config()
    splits = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits)
    root = resolve_path(load_split_meta(splits).get("dataset_root", cfg["paths"]["dataset_root"]))
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    test = manifest[(manifest["split"] == "test") & (~manifest["multi_defect"].astype(str).str.lower().eq("true"))]
    n = 0
    for cls in CLASS_NAMES:
        rows = test[test["label"] == cls].sort_values("rel_path").head(args.per_class)
        for k, rel in enumerate(rows["rel_path"], 1):
            src = root / rel
            dest = out / f"{cls}_test_{k:02d}{src.suffix.lower()}"
            shutil.copy2(src, dest)
            n += 1
    (out / "README.md").write_text(
        "Demo images copied from the TEST split of the NEU surface defect dataset by scripts/export_samples.py.\n"
        "They are never used for training or model selection. The file name carries the dataset label.\n"
        "NEU dataset: K. Song and Y. Yan, Northeastern University (research use).\n", encoding="utf-8")
    print(f"Copied {n} sample images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
