"""Validate the raw dataset, detect duplicates, create a reproducible split manifest.

Usage:
    python scripts/prepare_dataset.py                      # uses paths.dataset_root from the config
    python scripts/prepare_dataset.py --source /path/to/NEU --seed 42

Outputs (artifacts/splits/):
    split_manifest.csv       one row per image: path, label, split, duplicate group, stats
    split_meta.json          seed, ratios, dataset root, sizes
    dataset_report.json      counts, integrity problems, duplicates, multi-defect annotations
    quality_reference.json   training-split image statistics used for inference quality warnings
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import CLASS_NAMES, PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.data.dataset import DatasetError, discover_dataset, find_duplicates  # noqa: E402
from src.data.splitting import make_split, save_manifest, verify_split  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("prepare_dataset")


def build_quality_reference(train_df, cfg) -> dict:
    q = cfg["quality"]
    lo, hi = q["percentile_low"], q["percentile_high"]
    ref = {"source": "training split statistics", "n_images": int(len(train_df)),
           "percentile_low": lo, "percentile_high": hi, "stats": {}}
    for col in ("mean", "std", "lap_var", "clipped_frac", "channel_diff"):
        vals = train_df[col].to_numpy(dtype=float)
        ref["stats"][col] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "p_low": float(np.percentile(vals, lo)), "p_high": float(np.percentile(vals, hi)),
            "median": float(np.median(vals)),
        }
    s = ref["stats"]
    ref["thresholds"] = {
        "min_mean_brightness": s["mean"]["p_low"],
        "max_mean_brightness": s["mean"]["p_high"],
        "min_contrast_std": s["std"]["p_low"],
        "min_laplacian_var": s["lap_var"]["p_low"],
        "max_clipped_fraction": max(s["clipped_frac"]["p_high"], 0.05),
    }
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="dataset root (overrides paths.dataset_root)")
    ap.add_argument("--out", default=None, help="output folder (overrides paths.splits_dir)")
    ap.add_argument("--seed", type=int, default=None, help="split seed (overrides data.split.seed)")
    ap.add_argument("--allow-missing-classes", action="store_true",
                    help="continue when a class has no images (only for pipeline tests)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    root = resolve_path(args.source or cfg["paths"]["dataset_root"])
    out = resolve_path(args.out or cfg["paths"]["splits_dir"])
    seed = args.seed if args.seed is not None else int(cfg["data"]["split"]["seed"])
    ratios = {k: float(cfg["data"]["split"][k]) for k in ("train", "val", "test")}

    log.info("Scanning dataset at %s", root)
    dcfg = cfg["data"]["duplicates"]
    try:
        disc = discover_dataset(root, set(cfg["data"]["supported_extensions"]),
                                signature_size=int(dcfg["signature_size"]),
                                signature_sigma=float(dcfg["signature_sigma"]))
    except DatasetError as exc:
        log.error(str(exc))
        return 2
    df = disc.records
    if df.empty:
        log.error("No labelled images found under %s. See README 'Dataset layout'.", root)
        return 2
    log.info("Readable labelled images: %d", len(df))
    for c, n in disc.class_counts.items():
        log.info("  %-16s %4d", c, n)
    if disc.unreadable:
        log.warning("Unreadable/corrupted images skipped: %d (see dataset_report.json)", len(disc.unreadable))
    if disc.unlabeled:
        log.warning("Images with no recognisable label skipped: %d", len(disc.unlabeled))
    if disc.label_conflicts:
        log.warning("Folder/file-name label conflicts excluded: %d", len(disc.label_conflicts))
    if disc.missing_classes:
        msg = f"Missing classes: {disc.missing_classes}"
        if not args.allow_missing_classes:
            log.error(msg + " (use --allow-missing-classes only for pipeline tests)")
            return 3
        log.warning(msg)

    # Duplicate / near-duplicate detection.
    thr = float(dcfg["similarity_threshold"])
    groups, pairs = find_duplicates(df, disc.signatures, thr)
    df["group_id"] = groups
    cross = [p for p in pairs if p["cross_class"]]
    log.info("Duplicate pairs: %d exact, %d near (high-pass NCC >= %.2f), %d cross-class; %d groups",
             sum(p["exact"] for p in pairs), sum(not p["exact"] for p in pairs), thr, len(cross),
             int((df.groupby("group_id").size() > 1).sum()))
    # Exact duplicates with conflicting labels are unusable ground truth -> exclude both.
    conflict_paths = {p["a"] for p in cross if p["exact"]} | {p["b"] for p in cross if p["exact"]}
    if conflict_paths:
        log.warning("Excluding %d images that are exact duplicates with different labels", len(conflict_paths))
        df = df[~df["rel_path"].isin(conflict_paths)].reset_index(drop=True)

    df = make_split(df, ratios, seed)
    summary = verify_split(df)
    log.info("Split sizes: %s", summary["sizes"])

    multi = df[df["multi_defect"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "n_images": int(len(df)),
        "class_counts": {c: int((df["label"] == c).sum()) for c in CLASS_NAMES},
        "missing_classes": disc.missing_classes,
        "image_sizes": {f"{w}x{h}": int(n) for (w, h), n in df.groupby(["width", "height"]).size().items()},
        "image_modes": {f"{f}/{m}": int(n) for (f, m), n in df.groupby(["format", "mode"]).size().items()},
        "max_channel_difference": float(df["channel_diff"].max()),
        "unreadable": disc.unreadable,
        "unlabeled": disc.unlabeled,
        "unsupported_format_files": disc.unsupported,
        "label_conflicts": disc.label_conflicts,
        "duplicates": {
            "method": (f"md5 (exact) + normalised cross-correlation of high-pass {dcfg['signature_size']}px "
                       f"images over 8 orientations, threshold {thr}"),
            "n_pairs": len(pairs), "n_exact": sum(p["exact"] for p in pairs),
            "n_cross_class": len(cross), "pairs": pairs[:200],
            "n_groups_with_duplicates": int((df.groupby("group_id").size() > 1).sum()),
            "excluded_conflicting_exact_duplicates": sorted(conflict_paths),
        },
        "annotations": {
            "n_with_annotation": int(df["has_annotation"].sum()),
            "n_multi_defect": int(len(multi)),
            "multi_defect_by_label": {c: int((multi["label"] == c).sum()) for c in CLASS_NAMES},
            "n_annotation_missing_primary": int((df["annotation_has_primary"] == False).sum()),  # noqa: E712
            "note": ("Image-level label = file-name/folder label (NEU-CLS convention). Images whose XML "
                     "boxes also contain other defect classes are flagged multi_defect and kept."),
        },
        "split": {"seed": seed, "ratios": ratios, **summary},
    }
    out.mkdir(parents=True, exist_ok=True)
    try:
        rel_root = str(root.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_root = str(root)
    save_manifest(df, out, {"seed": seed, "ratios": ratios, "dataset_root": rel_root,
                            "created_at": report["created_at"], "sizes": summary["sizes"],
                            "n_images": int(len(df))})
    with open(out / "dataset_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    qref = build_quality_reference(df[df["split"] == "train"], cfg)
    with open(out / "quality_reference.json", "w", encoding="utf-8") as fh:
        json.dump(qref, fh, indent=2)
    log.info("Wrote manifest, dataset report and quality reference to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
