"""Inspection history: search, filter, view a record and its image, export CSV, confirm labels."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.database.db import DatabaseError
from src.database.repository import SOURCE_LABELS, RecordValidationError, format_local
from src.ui.common import class_count_chart, get_repository, probability_chart


def _detail(repo, rec) -> None:
    st.markdown(f"### {rec.inspection_id}")
    if rec.problems:
        st.warning("This record is partly damaged: " + "; ".join(rec.problems))
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        path = repo.image_file(rec)
        if path is not None:
            st.image(str(path), caption=rec.original_filename or path.name, width="stretch")
        elif rec.image_path:
            st.warning(f"The stored image is missing on disk ({rec.image_path}). The record itself is intact.")
        else:
            st.info("No image was stored with this record.")
    with c2:
        st.markdown(f"**Model prediction:** {rec.display_class} ({rec.confidence * 100:.1f}%)")
        if rec.verified_label:
            agree = "agrees" if rec.verified_label == rec.predicted_class else "differs"
            st.markdown(f"**Verified by operator:** {DISPLAY_NAMES[rec.verified_label]} ({agree} with the model)")
        if rec.review_needed:
            st.warning("Flagged as review needed (score below the review threshold).")
        if rec.probabilities and set(rec.probabilities) == set(CLASS_NAMES):
            st.altair_chart(probability_chart(rec.probabilities, rec.predicted_class), width="stretch")
        warns = [i for i in (rec.quality.get("issues") or []) if i.get("severity") == "warning"]
        if warns:
            st.markdown("**Image quality warnings at inspection time**\n" +
                        "\n".join(f"- {w['message']}" for w in warns))
    meta = pd.DataFrame([
        ("Date and time", format_local(rec.created_local)),
        ("Source", SOURCE_LABELS.get(rec.source, rec.source)),
        ("Device", rec.device_id or "-"),
        ("Model version", rec.model_version),
        ("Preprocessing", ", ".join(f"{k}={v}" for k, v in rec.preprocessing.items()) or "-"),
        ("Inference time", "-" if rec.inference_ms is None else f"{rec.inference_ms:.1f} ms"),
        ("Operator note", rec.operator_note or "-"),
    ], columns=["Field", "Value"])
    st.dataframe(meta, hide_index=True, width="stretch")

    with st.form(f"verify_{rec.inspection_id}"):
        st.markdown("**Record the confirmed class** (after a person has checked the surface)")
        options = [None] + list(CLASS_NAMES)
        current = options.index(rec.verified_label) if rec.verified_label in CLASS_NAMES else 0
        label = st.selectbox("Confirmed class", options, index=current,
                             format_func=lambda c: "Not verified" if c is None else DISPLAY_NAMES[c])
        note = st.text_area("Operator note", value=rec.operator_note, max_chars=2000)
        if st.form_submit_button("Update record"):
            try:
                repo.set_verified_label(rec.inspection_id, label)
                repo.update_note(rec.inspection_id, note)
                st.success("Record updated.")
                st.rerun()
            except (DatabaseError, RecordValidationError) as exc:
                st.error(f"Update failed: {exc}")


def render(cfg: dict) -> None:
    st.title("Inspection history")
    repo, err = get_repository(cfg)
    if repo is None:
        st.error(f"The inspection database cannot be opened: {err}")
        return
    try:
        total = repo.count()
    except DatabaseError as exc:
        st.error(str(exc))
        return
    if total == 0:
        st.info("No inspections saved yet. Classify an image on *Inspect surface* and press "
                "*Save inspection record*.")
        return

    f1, f2, f3 = st.columns([2, 3, 2])
    id_q = f1.text_input("Search inspection ID", placeholder="e.g. INS-2026")
    classes = f2.multiselect("Predicted class", CLASS_NAMES, format_func=lambda c: DISPLAY_NAMES[c])
    today = date.today()
    rng = f3.date_input("Date range", value=(today - timedelta(days=30), today), max_value=today)
    g1, g2 = st.columns([3, 2])
    sources = g1.multiselect("Source", list(SOURCE_LABELS), format_func=lambda s: SOURCE_LABELS[s])
    review_only = g2.checkbox("Only review-needed", value=False)
    if isinstance(rng, (tuple, list)):  # while picking, Streamlit returns a 1-tuple until both ends are set
        d_from = rng[0] if len(rng) > 0 else None
        d_to = rng[1] if len(rng) > 1 else d_from
    else:
        d_from = d_to = rng

    try:
        records = repo.query(id_contains=id_q or None, classes=classes or None, date_from=d_from, date_to=d_to,
                             review_only=review_only, sources=sources or None)
    except DatabaseError as exc:
        st.error(str(exc))
        return

    st.caption(f"{len(records)} of {total} inspections match. Classes are model predictions unless a person "
               "recorded a confirmed class.")
    if not records:
        st.info("No inspections match these filters. Widen the date range or clear a filter.")
        return

    df = repo.to_dataframe(records)
    view = df[["inspection_id", "created_at_local", "display_class", "confidence", "review_needed",
               "n_quality_warnings", "verified_label", "source"]].copy()
    view["source"] = view["source"].map(SOURCE_LABELS)
    view["verified_label"] = view["verified_label"].map(lambda v: DISPLAY_NAMES.get(v, "") if isinstance(v, str) else "")
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "inspection_id": "Inspection ID",
        "created_at_local": "Date and time",
        "display_class": "Predicted class",
        "confidence": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="percent"),
        "review_needed": st.column_config.CheckboxColumn("Review", width="small"),
        "n_quality_warnings": st.column_config.NumberColumn("Quality flags", width="small"),
        "verified_label": "Confirmed class",
        "source": "Source",
    })
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.download_button("Export these records to CSV", data=repo.export_csv(records),
                           file_name=f"inspections_{date.today():%Y%m%d}.csv", mime="text/csv",
                           icon=":material/download:")
    with right:
        st.markdown("**Predicted class counts (filtered)**")
        st.altair_chart(class_count_chart(repo.class_counts(records)), width="stretch")

    st.divider()
    ids = [r.inspection_id for r in records]
    sel = st.selectbox("Open inspection", ids, key="history_open")
    rec = next((r for r in records if r.inspection_id == sel), None)
    if rec is not None:
        _detail(repo, rec)
