
"""Steel Surface Defect Classifier - Streamlit application.

Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src.ui import (
    page_analytics,
    page_evaluation,
    page_history,
    page_home,
    page_inspect,
    page_weights,
)  # noqa: E402

from src.ui.common import (  # noqa: E402
    get_config,
    get_predictor,
    get_repository,
)

st.set_page_config(
    page_title="Steel defect classifier",
    page_icon=":material/texture:",
    layout="wide",
)

PAGES = {
    "Overview": page_home.render,
    "Inspect surface": page_inspect.render,
    "Inspection history": page_history.render,
    "Analytics": page_analytics.render,
    "Model evaluation": page_evaluation.render,
    "CNN Weights": page_weights.render,
}


def sidebar_status(cfg: dict) -> None:
    predictor, err = get_predictor(cfg)
    repo, repo_err = get_repository(cfg)

    st.sidebar.divider()

    if predictor:
        st.sidebar.markdown(
            f":material/check_circle: **Model loaded**  \n"
            f"`{predictor.model_version}`"
        )
    else:
        st.sidebar.markdown(
            ":material/error: **No trained model**  \n"
            "Classification is unavailable."
        )
        if err:
            st.sidebar.caption(err)

    if repo is None:
        st.sidebar.markdown(
            f":material/database: Inspection history unavailable: {repo_err}"
        )

    mode = cfg["hardware"].get("adapter", "standalone")

    mode_text = {
        "standalone": "Standalone (uploads only)",
        "simulated": "Simulated camera (no real device)",
        "folder": "Hardware drop folder",
    }.get(mode, mode)

    st.sidebar.caption(f"Acquisition mode: {mode_text}")


def main() -> None:
    cfg = get_config()

    st.sidebar.title("Steel defect classifier")

    page = st.sidebar.radio(
        "Go to",
        list(PAGES),
        key="nav",
        label_visibility="collapsed",
    )

    sidebar_status(cfg)

    PAGES[page](cfg)


if __name__ == "__main__":
    main()
