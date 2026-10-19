"""Shared fixtures. Tests that need the real NEU dataset or the trained model are marked `real_data`
and skipped (never silently passed) when those resources are missing. Everything else runs on
SYNTHETIC images generated on the fly, which only validates the software pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STEEL_API_NO_AUTOLOAD", "1")

from src.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from src.data.dataset import discover_dataset, find_duplicates  # noqa: E402
from src.data.splitting import make_split  # noqa: E402
from src.data.synthetic import generate_synthetic_dataset  # noqa: E402

REAL_SPLITS = ROOT / "artifacts" / "splits"
REAL_MANIFEST = REAL_SPLITS / "split_manifest.csv"
REAL_DATASET = ROOT / "data" / "NEU-DET"
DEPLOYED_MODEL = ROOT / "artifacts" / "models" / "steel_cnn_fp32.pt"
FINAL_REPORT = ROOT / "artifacts" / "reports" / "final" / "metrics.json"

real_dataset_available = REAL_MANIFEST.is_file() and REAL_DATASET.is_dir()
real_model_available = DEPLOYED_MODEL.is_file()

requires_real_data = pytest.mark.skipif(not real_dataset_available,
                                        reason="BLOCKED: NEU dataset / split manifest not present")
requires_real_model = pytest.mark.skipif(not (real_dataset_available and real_model_available),
                                         reason="BLOCKED: trained model artifacts/models/steel_cnn_fp32.pt not present")


def pytest_configure(config):
    config.addinivalue_line("markers", "real_data: needs the real NEU dataset and/or trained model")
    config.addinivalue_line("markers", "slow: trains a small model")


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return generate_synthetic_dataset(tmp_path_factory.mktemp("synthetic") / "NEU-DET", per_class=14, seed=0)


@pytest.fixture(scope="session")
def synthetic_manifest(synthetic_root):
    disc = discover_dataset(synthetic_root)
    df = disc.records
    groups, _ = find_duplicates(df, disc.signatures, 0.70)
    df["group_id"] = groups
    return make_split(df, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=42)


@pytest.fixture(scope="session")
def tiny_checkpoint(synthetic_manifest, synthetic_root, tmp_path_factory) -> Path:
    """A small model trained for a few epochs on SYNTHETIC data (software tests only)."""
    from src.training.train import TrainSettings, train_model
    s = TrainSettings(input_size=64, augment="flip", seed=0, max_epochs=6, patience=10, batch_size=16)
    res = train_model(synthetic_manifest, synthetic_root, s, tmp_path_factory.mktemp("runs"), log=lambda m: None)
    return res.checkpoint_path


def write_config(tmp_dir: Path, overrides: dict) -> Path:
    """Write a full config YAML (defaults + overrides) for scripts / app tests."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(cfg, overrides)
    path = Path(tmp_dir) / "test_config.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


@pytest.fixture
def temp_cfg(tmp_path, tiny_checkpoint) -> dict:
    """Config using temporary storage and the synthetic tiny model."""
    overrides = {
        "paths": {"inspections_dir": str(tmp_path / "inspections"), "splits_dir": str(tmp_path / "splits"),
                  "reports_dir": str(tmp_path / "reports"), "runs_dir": str(tmp_path / "runs")},
        "inference": {"model_path": str(tiny_checkpoint), "review_threshold": None},
        "hardware": {"adapter": "simulated", "drop_folder": str(tmp_path / "inbox")},
    }
    path = write_config(tmp_path, overrides)
    cfg = load_config(path)
    cfg["_test_config_path"] = str(path)
    return cfg


def png_bytes(arr) -> bytes:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
