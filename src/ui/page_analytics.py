"""Analytics computed from stored inspection records only."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.database.db import DatabaseError
from src.database.repository import SOURCE_LABELS
from src.ui.common import CLASS_COLORS, class_count_chart, get_repository


def render(cfg: dict) -> None:
    st.title("Analytics")
    st.caption("Everything below is counted from saved inspection records. Classes are model predictions on the "
               "images that were inspected, not measured defect rates of a production line: that would need "
               "representative sampling and confirmed labels.")
    repo, err = get_repository(cfg)
    if repo is None:
        st.error(f"The inspection database cannot be opened: {err}")
        return
    try:
        records = repo.query()
    except DatabaseError as exc:
        st.error(str(exc))
        return
    if not records:
        st.info("No inspections saved yet. Classify and save an image on *Inspect surface* to see analytics.")
        return

    df = repo.to_dataframe(records)
    df["day"] = [r.created_local.date() for r in records]
    n = len(df)
    n_review = int(df["review_needed"].sum())
    n_quality = int((df["n_quality_warnings"] > 0).sum())
    n_verified = int(df["verified_label"].notna().sum())
    k = st.columns(4)
    k[0].metric("Inspections saved", n)
    k[1].metric("Review needed", n_review, help="Top softmax score below the validation-chosen threshold")
    k[2].metric("With image quality warnings", n_quality)
    k[3].metric("Confirmed by an operator", n_verified)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Inspections by predicted class**")
        st.altair_chart(class_count_chart(repo.class_counts(records)), width="stretch")
    with c2:
        st.markdown("**Inspections per day by predicted class**")
        daily = df.groupby(["day", "predicted_class"]).size().reset_index(name="count")
        daily["class"] = daily["predicted_class"].map(DISPLAY_NAMES)
        daily["day"] = pd.to_datetime(daily["day"])
        chart = alt.Chart(daily).mark_bar(stroke="#F4F6F8", strokeWidth=1).encode(
            x=alt.X("yearmonthdate(day):O", title=None, axis=alt.Axis(labelAngle=0, format="%d %b")),
            y=alt.Y("count:Q", title="Inspections", axis=alt.Axis(tickMinStep=1, gridColor="#DDE2E8", domain=False)),
            color=alt.Color("class:N", title="Predicted class",
                            scale=alt.Scale(domain=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
                                            range=list(CLASS_COLORS.values())),
                            legend=alt.Legend(orient="bottom", columns=3)),
            order=alt.Order("predicted_class:N"),
            tooltip=[alt.Tooltip("yearmonthdate(day):T", title="Day"), alt.Tooltip("class:N", title="Class"),
                     alt.Tooltip("count:Q", title="Inspections")],
        ).properties(height=230).configure_view(strokeWidth=0)
        st.altair_chart(chart, width="stretch")

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown("**Review needed by predicted class**")
        rv = df.groupby("predicted_class")["review_needed"].agg(["sum", "count"]).reindex(CLASS_NAMES).fillna(0)
        rv_df = pd.DataFrame({"Predicted class": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
                              "Review needed": rv["sum"].astype(int).to_numpy(),
                              "Inspections": rv["count"].astype(int).to_numpy()})
        st.dataframe(rv_df, hide_index=True, width="stretch")
    with c4:
        st.markdown("**Image sources**")
        src = df["source"].map(SOURCE_LABELS).value_counts().rename_axis("Source").reset_index(name="Inspections")
        st.dataframe(src, hide_index=True, width="stretch")

    if n_verified:
        st.markdown("**Model predictions vs operator-confirmed classes**")
        v = df[df["verified_label"].notna()]
        agree = int((v["verified_label"] == v["predicted_class"]).sum())
        st.markdown(f"On the {n_verified} confirmed inspections the model agreed {agree} times "
                    f"({agree / n_verified * 100:.0f}%). This is a small, non-random sample.")
        ct = pd.crosstab(v["verified_label"].map(DISPLAY_NAMES), v["display_class"])
        ct.index.name, ct.columns.name = "Confirmed", "Predicted"
        st.dataframe(ct, width="stretch")

    st.markdown("**Recent activity**")
    recent = df.head(10)[["inspection_id", "created_at_local", "display_class", "confidence", "review_needed",
                          "source"]].copy()
    recent["source"] = recent["source"].map(SOURCE_LABELS)
    st.dataframe(recent, hide_index=True, width="stretch", column_config={
        "inspection_id": "Inspection ID", "created_at_local": "Date and time", "display_class": "Predicted class",
        "confidence": st.column_config.NumberColumn("Score", format="percent"),
        "review_needed": st.column_config.CheckboxColumn("Review"), "source": "Source"})
