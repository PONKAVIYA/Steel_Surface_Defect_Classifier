"""Dataset discovery, validation, duplicate detection and splitting."""
from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
from PIL import Image

from src.config import CLASS_NAMES
from src.data.dataset import (DatasetError, discover_dataset, find_duplicates, label_from_filename,
                              normalize_label)
from src.data.splitting import SplitError, load_manifest, make_split, save_manifest, verify_split
from src.data.synthetic import generate_synthetic_dataset
from tests.conftest import REAL_MANIFEST, REAL_SPLITS, requires_real_data, write_config

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


# ----------------------------------------------------------------------------- labels
@pytest.mark.parametrize("stem,label", [
    ("crazing_10", "crazing"), ("rolled-in_scale_5", "rolled-in_scale"), ("pitted_surface_299", "pitted_surface"),
    ("Cr_1", "crazing"), ("In_12", "inclusion"), ("PS3", "pitted_surface"), ("RS_100", "rolled-in_scale"),
    ("Sc_7", "scratches"), ("Pa_2", "patches"), ("unknown_3", None),
])
def test_label_from_filename(stem, label):
    assert label_from_filename(stem) == label


def test_normalize_label_rejects_unknown():
    assert normalize_label("rust") is None
    assert normalize_label("Rolled_In_Scale") == "rolled-in_scale"


# ----------------------------------------------------------------------------- discovery
def test_discovers_all_six_classes(synthetic_root):
    disc = discover_dataset(synthetic_root)
    assert disc.missing_classes == []
    assert set(disc.class_counts) == set(CLASS_NAMES)
    assert all(n == 14 for n in disc.class_counts.values())
    assert len(disc.signatures) == len(disc.records)
    assert not disc.unreadable and not disc.unlabeled


@pytest.mark.parametrize("layout", ["class-folders", "neu-cls"])
def test_other_layouts(tmp_path, layout):
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=3, layout=layout)
    disc = discover_dataset(root)
    assert disc.class_counts == {c: 3 for c in CLASS_NAMES}


def test_missing_dataset_raises_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="Dataset folder not found"):
        discover_dataset(tmp_path / "nope")


def test_corrupted_and_empty_images_are_reported(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=2)
    (root / "train_images" / "crazing_99.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    (root / "train_images" / "patches_98.jpg").write_bytes(b"")
    disc = discover_dataset(root)
    bad = {u["path"] for u in disc.unreadable}
    assert bad == {"train_images/crazing_99.jpg", "train_images/patches_98.jpg"}
    assert all("decode" in u["error"].lower() or "empty" in u["error"].lower() for u in disc.unreadable)
    assert disc.class_counts["crazing"] == 2


def test_label_conflict_and_unlabeled(tmp_path):
    root = tmp_path / "ds"
    (root / "crazing").mkdir(parents=True)
    Image.new("L", (200, 200), 128).save(root / "crazing" / "Sc_1.png")   # folder says crazing, name says scratches
    Image.new("L", (200, 200), 128).save(root / "crazing" / "img_1.png")  # OK
    (root / "misc").mkdir()
    Image.new("L", (200, 200), 128).save(root / "misc" / "photo.png")     # no label
    disc = discover_dataset(root)
    assert [c["path"] for c in disc.label_conflicts] == ["crazing/Sc_1.png"]
    assert disc.unlabeled == ["misc/photo.png"]
    assert len(disc.records) == 1


def test_missing_classes_detected(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=2, classes=("crazing", "patches"))
    disc = discover_dataset(root)
    assert set(disc.missing_classes) == {"inclusion", "pitted_surface", "rolled-in_scale", "scratches"}


def test_annotation_secondary_labels(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=1)
    ann = root / "train_annotations"
    ann.mkdir()
    (ann / "crazing_1.xml").write_text(
        "<annotation><object><name>crazing</name></object><object><name>patches</name></object></annotation>")
    df = discover_dataset(root).records.set_index("rel_path")
    row = df.loc["train_images/crazing_1.jpg"]
    assert row["multi_defect"] and row["secondary_labels"] == "patches" and row["annotation_has_primary"]


# ----------------------------------------------------------------------------- duplicates
def test_exact_and_rotated_duplicates_grouped(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=4)
    src = root / "train_images" / "inclusion_1.jpg"
    shutil.copy(src, root / "train_images" / "inclusion_90.jpg")  # exact copy
    Image.open(src).rotate(90).save(root / "train_images" / "inclusion_91.jpg", quality=95)  # rotated copy
    disc = discover_dataset(root)
    groups, pairs = find_duplicates(disc.records, disc.signatures, 0.70)
    df = disc.records.assign(group_id=groups).set_index("rel_path")
    g = df.loc["train_images/inclusion_1.jpg", "group_id"]
    assert df.loc["train_images/inclusion_90.jpg", "group_id"] == g
    assert df.loc["train_images/inclusion_91.jpg", "group_id"] == g
    assert any(p["exact"] for p in pairs)
    # unrelated images stay in their own groups
    assert (df["group_id"] == g).sum() == 3


# ----------------------------------------------------------------------------- splitting
def test_split_is_stratified_disjoint_and_reproducible(synthetic_manifest):
    df = synthetic_manifest
    summary = verify_split(df)
    assert sum(summary["sizes"].values()) == len(df)
    for c in CLASS_NAMES:
        assert summary["per_class"]["test"][c] == round(14 * 0.15)
        assert summary["per_class"]["val"][c] == round(14 * 0.15)
    again = make_split(df.drop(columns=["split"]), RATIOS, seed=42)
    assert again.set_index("rel_path")["split"].equals(df.set_index("rel_path")["split"].loc[again["rel_path"]])
    other = make_split(df.drop(columns=["split"]), RATIOS, seed=7)
    assert not other.set_index("rel_path")["split"].equals(again.set_index("rel_path")["split"])


def test_duplicate_groups_stay_in_one_split():
    rows = []
    for c in CLASS_NAMES:
        for i in range(20):
            rows.append({"rel_path": f"{c}_{i}.png", "label": c, "md5": f"{c}{i}",
                         "group_id": f"{c}-{i // 4}"})  # groups of 4
    df = make_split(pd.DataFrame(rows), RATIOS, seed=1)
    assert (df.groupby("group_id")["split"].nunique() == 1).all()


def test_verify_split_detects_leakage(synthetic_manifest):
    df = synthetic_manifest.copy()
    leak = df[df["split"] == "train"].iloc[0].copy()
    leak["split"] = "test"
    with pytest.raises(SplitError, match="leakage"):
        verify_split(pd.concat([df, leak.to_frame().T], ignore_index=True))


def test_manifest_roundtrip_and_missing(tmp_path, synthetic_manifest):
    with pytest.raises(SplitError, match="prepare_dataset"):
        load_manifest(tmp_path)
    save_manifest(synthetic_manifest, tmp_path, {"seed": 42})
    back = load_manifest(tmp_path)
    assert len(back) == len(synthetic_manifest)
    assert json.loads((tmp_path / "split_meta.json").read_text())["seed"] == 42


def test_split_rejects_bad_ratios(synthetic_manifest):
    with pytest.raises(SplitError):
        make_split(synthetic_manifest, {"train": 0.8, "val": 0.15, "test": 0.15}, 0)


# ----------------------------------------------------------------------------- prepare script
def test_prepare_dataset_script(tmp_path, synthetic_root):
    from scripts import prepare_dataset
    cfg_path = write_config(tmp_path, {})
    out = tmp_path / "splits"
    assert prepare_dataset.main(["--config", str(cfg_path), "--source", str(synthetic_root), "--out", str(out)]) == 0
    for f in ("split_manifest.csv", "split_meta.json", "dataset_report.json", "quality_reference.json"):
        assert (out / f).is_file(), f
    report = json.loads((out / "dataset_report.json").read_text())
    assert report["class_counts"] == {c: 14 for c in CLASS_NAMES}
    assert prepare_dataset.main(["--config", str(cfg_path), "--source", str(tmp_path / "missing"),
                                 "--out", str(out)]) == 2


def test_prepare_dataset_refuses_missing_classes(tmp_path):
    from scripts import prepare_dataset
    root = generate_synthetic_dataset(tmp_path / "ds", per_class=3, classes=("crazing", "scratches"))
    cfg_path = write_config(tmp_path, {})
    assert prepare_dataset.main(["--config", str(cfg_path), "--source", str(root), "--out",
                                 str(tmp_path / "s")]) == 3


# ----------------------------------------------------------------------------- real dataset
@pytest.mark.real_data
@requires_real_data
def test_real_dataset_manifest():
    df = load_manifest(REAL_SPLITS)
    assert len(df) == 1800
    assert df["label"].value_counts().to_dict() == {c: 300 for c in CLASS_NAMES}
    sizes = df["split"].value_counts().to_dict()
    assert sizes == {"train": 1260, "val": 270, "test": 270}
    for s in ("train", "val", "test"):
        assert df[df["split"] == s]["label"].value_counts().nunique() == 1  # perfectly stratified
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    report = json.loads((REAL_SPLITS / "dataset_report.json").read_text())
    assert report["image_sizes"] == {"200x200": 1800}
    assert report["duplicates"]["n_cross_class"] == 0
    assert REAL_MANIFEST.is_file()


@pytest.mark.real_data
@requires_real_data
def test_real_split_is_reproducible():
    df = load_manifest(REAL_SPLITS)
    again = make_split(df.drop(columns=["split"]), RATIOS, seed=42)
    a = again.set_index("rel_path")["split"]
    b = df.set_index("rel_path")["split"]
    assert a.sort_index().equals(b.sort_index())
