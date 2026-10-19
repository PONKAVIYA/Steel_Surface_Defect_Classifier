"""Reproducible, stratified, group-aware train/val/test splitting and manifest I/O."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES

MANIFEST_NAME = "split_manifest.csv"


class SplitError(RuntimeError):
    pass


def make_split(df: pd.DataFrame, ratios: dict[str, float], seed: int,
               group_col: str = "group_id") -> pd.DataFrame:
    """Assign each row to train/val/test.

    * Stratified per class: each class is split with the same ratios.
    * Group-aware: all rows sharing a group_id (near-duplicates) land in the same split.
      A group is assigned to the class of its majority label.
    * Deterministic for a given seed and input (rows are sorted by rel_path first).
    """
    total = sum(ratios.values())
    if not np.isclose(total, 1.0):
        raise SplitError(f"Split ratios must sum to 1.0 (got {total})")
    if group_col not in df.columns:
        df = df.assign(**{group_col: np.arange(len(df))})
    df = df.sort_values("rel_path").reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    group_label = df.groupby(group_col)["label"].agg(lambda s: s.value_counts().index[0])
    group_size = df.groupby(group_col).size()
    assignment: dict[int, str] = {}
    for cls in CLASS_NAMES:
        groups = np.array(sorted(group_label[group_label == cls].index))
        if len(groups) == 0:
            continue
        rng.shuffle(groups)
        n_cls = int(group_size[groups].sum())
        target_test = round(n_cls * ratios["test"])
        target_val = round(n_cls * ratios["val"])
        n_test = n_val = 0
        for g in groups:
            size = int(group_size[g])
            if n_test < target_test:
                assignment[g] = "test"
                n_test += size
            elif n_val < target_val:
                assignment[g] = "val"
                n_val += size
            else:
                assignment[g] = "train"
    df["split"] = df[group_col].map(assignment)
    if df["split"].isna().any():
        raise SplitError("Some rows were not assigned to a split.")
    return df


def verify_split(df: pd.DataFrame) -> dict:
    """Check that splits are disjoint by file, content hash and duplicate group. Raises on leakage."""
    problems = []
    for col in ("rel_path", "md5", "group_id"):
        if col not in df.columns:
            continue
        per_split = {s: set(df.loc[df["split"] == s, col]) for s in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = per_split[a] & per_split[b]
            if overlap:
                problems.append(f"{len(overlap)} shared '{col}' values between {a} and {b}")
    if problems:
        raise SplitError("Data leakage detected: " + "; ".join(problems))
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    return {
        "sizes": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "per_class": {s: {c: int(counts.loc[s].get(c, 0)) if s in counts.index else 0 for c in CLASS_NAMES}
                      for s in ("train", "val", "test")},
    }


def save_manifest(df: pd.DataFrame, splits_dir: str | Path, meta: dict) -> Path:
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    path = splits_dir / MANIFEST_NAME
    df.to_csv(path, index=False)
    with open(splits_dir / "split_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


def load_manifest(splits_dir: str | Path) -> pd.DataFrame:
    path = Path(splits_dir) / MANIFEST_NAME
    if not path.is_file():
        raise SplitError(
            f"Split manifest not found at {path}. Run: python scripts/prepare_dataset.py"
        )
    df = pd.read_csv(path, keep_default_na=False)
    required = {"rel_path", "label", "label_index", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SplitError(f"Manifest {path} is missing columns: {sorted(missing)}")
    bad = set(df["label"]) - set(CLASS_NAMES)
    if bad:
        raise SplitError(f"Manifest contains unknown labels: {sorted(bad)}")
    verify_split(df)
    return df


def load_split_meta(splits_dir: str | Path) -> dict:
    path = Path(splits_dir) / "split_meta.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
