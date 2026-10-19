"""Shared resources and helpers for the Streamlit pages."""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES, PROJECT_ROOT, load_config, resolve_path
from src.database.db import DatabaseError
from src.database.repository import InspectionRepository
from src.hardware.interface import build_adapter
from src.inference.predictor import ModelUnavailableError, Predictor
from src.inference.service import InspectionService
from src.utils.plotting import CATEGORICAL

ACCENT = "#3B5BA9"
MUTED = "#AEB7C2"
INK = "#1D2A35"
CLASS_COLORS = {c: CATEGORICAL[i] for i, c in enumerate(CLASS_NAMES)}  # fixed class order, never cycled
SAMPLES_DIR = PROJECT_ROOT / "samples"


@st.cache_resource(show_spinner=False)
def _load_config_cached(path: str | None) -> dict:
    return load_config(path)


def get_config() -> dict:
    """Config selected by STEEL_CONFIG (or the default); cached per file path."""
    return _load_config_cached(os.environ.get("STEEL_CONFIG"))


@st.cache_resource(show_spinner="Loading model ...")
def _load_predictor(path: str, mtime: float, threshold: float | None):
    try:
        return Predictor.from_checkpoint(path, threshold), None
    except ModelUnavailableError as exc:
        return None, str(exc)


def get_predictor(cfg: dict) -> tuple[Predictor | None, str | None]:
    path = resolve_path(cfg["inference"]["model_path"])
    mtime = path.stat().st_mtime if path.is_file() else 0.0
    return _load_predictor(str(path), mtime, cfg["inference"].get("review_threshold"))


@st.cache_resource(show_spinner=False)
def _load_repository(base_dir: str):
    try:
        return InspectionRepository(base_dir), None
    except DatabaseError as exc:
        return None, str(exc)


def get_repository(cfg: dict) -> tuple[InspectionRepository | None, str | None]:
    return _load_repository(str(resolve_path(cfg["paths"]["inspections_dir"])))


def get_service(cfg: dict) -> InspectionService:
    predictor, _ = get_predictor(cfg)
    repo, _ = get_repository(cfg)
    return InspectionService(cfg, predictor, repo)


def get_adapter(cfg: dict):
    """One adapter per browser session (connection state must not leak between users)."""
    key = f"adapter::{cfg['hardware'].get('adapter')}"
    if key not in st.session_state:
        st.session_state[key] = build_adapter(cfg)
    return st.session_state[key]


def read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def sample_files() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def probability_chart(probabilities: dict[str, float], highlight: str) -> alt.Chart:
    df = pd.DataFrame({
        "class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
        "score": [probabilities[c] for c in CLASS_NAMES],
        "is_top": [c == highlight for c in CLASS_NAMES],
    })
    df["label"] = df["score"].map(lambda v: f"{v * 100:.1f}%")
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK, labelFontSize=13, ticks=False,
                                                                domain=False, labelLimit=160, labelOverlap=False)),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Softmax score",
                axis=alt.Axis(format="%", grid=True, gridColor="#DDE2E8", labelColor="#5B6672", domain=False)),
        tooltip=[alt.Tooltip("class:N", title="Class"), alt.Tooltip("score:Q", title="Score", format=".4f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.condition(alt.datum.is_top, alt.value(ACCENT), alt.value(MUTED)))
    text = base.mark_text(align="left", dx=6, color=INK, fontSize=12).encode(text="label:N")
    return (bars + text).properties(height=alt.Step(30)).configure_view(strokeWidth=0)


def class_count_chart(counts: dict[str, int], title: str = "") -> alt.Chart:
    df = pd.DataFrame({"class": [DISPLAY_NAMES[c] for c in CLASS_NAMES], "key": list(CLASS_NAMES),
                       "count": [counts.get(c, 0) for c in CLASS_NAMES]})
    chart = alt.Chart(df, title=title) if title else alt.Chart(df)
    return chart.mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("class:N", sort=[DISPLAY_NAMES[c] for c in CLASS_NAMES], title=None,
                axis=alt.Axis(labelColor=INK, labelFontSize=12, ticks=False, domain=False, labelOverlap=False)),
        x=alt.X("count:Q", title="Inspections", axis=alt.Axis(tickMinStep=1, gridColor="#DDE2E8", domain=False)),
        color=alt.Color("key:N", scale=alt.Scale(domain=list(CLASS_NAMES), range=list(CLASS_COLORS.values())),
                        legend=None),
        tooltip=[alt.Tooltip("class:N", title="Predicted class"), alt.Tooltip("count:Q", title="Inspections")],
    ).properties(height=alt.Step(28)).configure_view(strokeWidth=0)
