"""Configuration loading and the fixed class mapping.

The class order below is the single source of truth for class index <-> name.
Checkpoints store this list and are rejected at load time if it does not match.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

CLASS_NAMES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
DISPLAY_NAMES: dict[str, str] = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted surface",
    "rolled-in_scale": "Rolled-in scale",
    "scratches": "Scratches",
}
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Short descriptions shown in the UI (visual appearance in NEU images).
CLASS_DESCRIPTIONS: dict[str, str] = {
    "crazing": "Fine network of thin, interconnected surface cracks; low-contrast, texture-like.",
    "inclusion": "Non-metallic particles embedded in the surface, seen as dark spots or streaks.",
    "patches": "Irregular darker or lighter regions (blotches) across the surface.",
    "pitted_surface": "Many small pits or cavities scattered over the surface.",
    "rolled-in_scale": "Oxide scale pressed into the surface during rolling; rough, flaky marks.",
    "scratches": "Long, thin linear marks, typically straight lines from mechanical contact.",
}

# Aliases accepted when discovering labels from folder names or file-name prefixes
# (NEU-CLS uses Cr_1.bmp, In_1.bmp ...; NEU-DET uses crazing_1.jpg ...).
LABEL_ALIASES: dict[str, str] = {
    "crazing": "crazing", "cr": "crazing",
    "inclusion": "inclusion", "in": "inclusion",
    "patches": "patches", "pa": "patches",
    "pitted_surface": "pitted_surface", "pitted-surface": "pitted_surface", "ps": "pitted_surface",
    "pitted": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale", "rolled_in_scale": "rolled-in_scale", "rolled-in-scale": "rolled-in_scale",
    "rs": "rolled-in_scale",
    "scratches": "scratches", "scratch": "scratches", "sc": "scratches",
}


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name)


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the YAML config, apply optional overrides and validate the class list.

    Without an explicit path, the STEEL_CONFIG environment variable (if set) selects the config file.
    """
    env_path = os.environ.get("STEEL_CONFIG")
    cfg_path = Path(path) if path else (Path(env_path) if env_path else DEFAULT_CONFIG_PATH)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if overrides:
        cfg = _deep_update(copy.deepcopy(cfg), overrides)
    classes = tuple(cfg.get("data", {}).get("class_names", CLASS_NAMES))
    if classes != CLASS_NAMES:
        raise ValueError(
            f"Config class_names {list(classes)} differ from the fixed class mapping {list(CLASS_NAMES)}. "
            "The class order is fixed so that checkpoints, reports and the database stay consistent."
        )
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
