"""Dataset discovery, validation and the PyTorch dataset used for training/evaluation.

Supported raw layouts (labels are inferred, then validated):
  * class folders            root/<class>/<any>.bmp|jpg|png       (e.g. root/crazing/x.png)
  * NEU-CLS flat             root/Cr_1.bmp, root/In_12.bmp ...
  * NEU-DET (Kaggle export)  root/train_images/crazing_1.jpg + root/train_annotations/crazing_1.xml
Folder names and file-name prefixes are mapped through LABEL_ALIASES. If both are present
and disagree, the file is reported as a label conflict and excluded.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CLASS_NAMES, CLASS_TO_INDEX, LABEL_ALIASES
from src.data.image_stats import channel_difference, gray_stats, highpass_signature, md5_bytes
from src.data.preprocessing import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    PreprocessingConfig,
    load_image,
    preprocess_pil,
)

AUGMENTATION_POLICIES = ("none", "flip", "flip_rot90")


class DatasetError(RuntimeError):
    """Raised when the dataset is missing or structurally invalid."""


@dataclass
class DiscoveryResult:
    records: pd.DataFrame
    signatures: np.ndarray | None = None  # (N, S, S) high-pass fingerprints aligned with records
    unreadable: list[dict] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    label_conflicts: list[dict] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)

    @property
    def class_counts(self) -> dict[str, int]:
        counts = Counter(self.records["label"]) if len(self.records) else Counter()
        return {c: int(counts.get(c, 0)) for c in CLASS_NAMES}

    @property
    def missing_classes(self) -> list[str]:
        return [c for c, n in self.class_counts.items() if n == 0]


def normalize_label(token: str | None) -> str | None:
    if not token:
        return None
    key = token.strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(key)


def label_from_filename(stem: str) -> str | None:
    """'crazing_10' -> crazing, 'rolled-in_scale_5' -> rolled-in_scale, 'Cr_1' -> crazing, 'PS12' -> pitted_surface."""
    m = re.match(r"^(.*?)[_\-\s]*\(?\d+\)?$", stem)
    prefix = m.group(1) if m else stem
    return normalize_label(prefix)


def _find_annotation(img_path: Path) -> Path | None:
    stem = img_path.stem
    parent = img_path.parent
    candidates = [parent / f"{stem}.xml"]
    if parent.name.endswith("_images"):
        candidates.append(parent.parent / (parent.name[: -len("_images")] + "_annotations") / f"{stem}.xml")
    candidates.append(parent.parent / "annotations" / f"{stem}.xml")
    candidates.append(parent / "annotations" / f"{stem}.xml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _annotation_labels(xml_path: Path) -> list[str]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        lab = normalize_label(name)
        labels.append(lab or f"unknown:{name}")
    return labels


def discover_dataset(root: str | Path, extensions: set[str] | None = None,
                     compute_stats: bool = True, signature_size: int = 64,
                     signature_sigma: float = 3.0) -> DiscoveryResult:
    """Walk the dataset root, infer and validate labels, check image integrity and compute statistics."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(
            f"Dataset folder not found: {root}\n"
            "Place the NEU surface defect dataset there (see README 'Dataset layout'), "
            "or pass --source to scripts/prepare_dataset.py."
        )
    exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
    rows, sigs, unreadable, unlabeled, conflicts, unsupported = [], [], [], [], [], []
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".xml" or "annotation" in path.parent.name.lower():
            continue
        if suffix not in exts:
            if suffix in {".tif", ".tiff", ".gif", ".webp"}:
                unsupported.append(str(path.relative_to(root)))
            continue
        rel = path.relative_to(root).as_posix()
        folder_label = normalize_label(path.parent.name) if path.parent != root else None
        file_label = label_from_filename(path.stem)
        if folder_label and file_label and folder_label != file_label:
            conflicts.append({"path": rel, "folder_label": folder_label, "filename_label": file_label})
            continue
        label = folder_label or file_label
        if label is None:
            unlabeled.append(rel)
            continue
        try:
            data = path.read_bytes()
            img = load_image(data)
        except (ImageLoadError, OSError) as exc:
            unreadable.append({"path": rel, "error": str(exc)})
            continue
        ann = _find_annotation(path)
        ann_labels = _annotation_labels(ann) if ann else []
        secondary = sorted(set(ann_labels) - {label})
        row = {
            "rel_path": rel,
            "label": label,
            "label_index": CLASS_TO_INDEX[label],
            "md5": md5_bytes(data),
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "has_annotation": ann is not None,
            "n_boxes": len(ann_labels),
            "annotation_has_primary": (label in ann_labels) if ann else None,
            "secondary_labels": ";".join(secondary),
            "multi_defect": bool(secondary),
        }
        if compute_stats:
            row.update(gray_stats(img))
            row["channel_diff"] = channel_difference(img)
        rows.append(row)
        sigs.append(highpass_signature(img, signature_size, signature_sigma))
    df = pd.DataFrame(rows)
    signatures = np.stack(sigs) if sigs else np.zeros((0, signature_size, signature_size), np.float32)
    return DiscoveryResult(df, signatures, unreadable, unlabeled, conflicts, unsupported)


def _dihedral(x: np.ndarray) -> list[np.ndarray]:
    """All 8 flip/rotation variants of a (N, S, S) stack, flattened to (N, S*S)."""
    out = []
    for k in range(4):
        r = np.rot90(x, k, axes=(1, 2))
        out.append(np.ascontiguousarray(r).reshape(len(x), -1))
        out.append(np.ascontiguousarray(r[:, :, ::-1]).reshape(len(x), -1))
    return out


def find_duplicates(df: pd.DataFrame, signatures: np.ndarray, threshold: float = 0.70,
                    block: int = 1024) -> tuple[np.ndarray, list[dict]]:
    """Group exact (md5) and near duplicates (high-pass NCC >= threshold over all 8 orientations).

    Orientation-invariant matching matters because training uses flips/rotations: a rotated copy
    of a test image in the training set would leak. Returns (group_id per row, duplicate pairs).
    Union-find ensures transitively connected duplicates share one group so they stay in one split.
    """
    n = len(df)
    if signatures is None or len(signatures) != n:
        raise ValueError("signatures must be aligned with df rows")
    base = signatures.reshape(n, -1).astype(np.float32)
    variants = _dihedral(signatures.astype(np.float32))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    pairs = []
    md5s = df["md5"].to_numpy()
    labels = df["label"].to_numpy()
    paths = df["rel_path"].to_numpy()
    for start in range(0, n, block):
        stop = min(n, start + block)
        sim = np.full((stop - start, n), -1.0, dtype=np.float32)
        for v in variants:
            np.maximum(sim, base[start:stop] @ v.T, out=sim)
        for i in range(start, stop):
            row = sim[i - start]
            exact = md5s == md5s[i]
            hits = np.where(((row >= threshold) | exact) & (np.arange(n) > i))[0]
            for j in hits:
                union(i, int(j))
                pairs.append({
                    "a": paths[i], "b": paths[j], "similarity": round(float(row[j]), 4),
                    "exact": bool(exact[j]), "label_a": labels[i], "label_b": labels[j],
                    "cross_class": bool(labels[i] != labels[j]),
                })
    roots = [find(i) for i in range(n)]
    remap = {r: g for g, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], dtype=int), pairs


# --------------------------------------------------------------------------------------------
# Augmentation (training split only)
# --------------------------------------------------------------------------------------------
def augment_tensor(x: torch.Tensor, policy: str, generator: torch.Generator | None = None) -> torch.Tensor:
    """Label-preserving geometric augmentation on a (C, H, W) tensor.

    * flip:        random horizontal and vertical mirror (each p=0.5)
    * flip_rot90:  flips + random rotation by k*90 degrees (all 8 dihedral orientations)
    Mirroring or rotating a surface patch does not change which defect type it shows; whether the
    orientation change helps is decided empirically on the validation split (see EXPERIMENTS.md).
    """
    if policy == "none":
        return x
    if policy not in AUGMENTATION_POLICIES:
        raise ValueError(f"Unknown augmentation policy '{policy}'. Options: {AUGMENTATION_POLICIES}")
    r = torch.rand(3, generator=generator)
    if r[0] < 0.5:
        x = torch.flip(x, dims=[-1])
    if r[1] < 0.5:
        x = torch.flip(x, dims=[-2])
    if policy == "flip_rot90":
        k = int(torch.randint(0, 4, (1,), generator=generator))
        if k:
            x = torch.rot90(x, k, dims=[-2, -1])
    return x


class SteelDefectDataset(Dataset):
    """Preloads one split into memory using the shared preprocessing pipeline."""

    def __init__(self, manifest: pd.DataFrame, dataset_root: str | Path, split: str,
                 preprocessing: PreprocessingConfig, augment: str = "none", seed: int = 0):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        if split != "train" and augment != "none":
            raise ValueError("Augmentation is only allowed on the training split.")
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            raise DatasetError(f"No rows for split '{split}' in the manifest.")
        self.rows = rows
        self.root = Path(dataset_root)
        self.preprocessing = preprocessing
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)
        arrays = []
        for rel in rows["rel_path"]:
            img = load_image(self.root / rel)
            arrays.append(preprocess_pil(img, preprocessing))
        self.x = torch.from_numpy(np.stack(arrays)).unsqueeze(1)  # (N, 1, S, S)
        self.y = torch.tensor(rows["label_index"].to_numpy(), dtype=torch.long)

    def with_augmentation(self, policy: str, seed: int) -> "SteelDefectDataset":
        """Shallow copy sharing the preloaded tensors but with its own augmentation policy and RNG."""
        if policy != "none" and set(self.rows["split"]) != {"train"}:
            raise ValueError("Augmentation is only allowed on the training split.")
        clone = object.__new__(SteelDefectDataset)
        clone.__dict__.update(self.__dict__)
        clone.augment = policy
        clone.generator = torch.Generator().manual_seed(seed)
        return clone

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.augment != "none":
            x = augment_tensor(x, self.augment, self.generator)
        return x, self.y[idx]
