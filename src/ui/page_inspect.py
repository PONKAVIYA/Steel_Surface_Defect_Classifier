"""Inspect page: choose an image, check its quality, classify, explain, and save a traceable record."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import CLASS_NAMES, DISPLAY_NAMES
from src.data.preprocessing import ImageLoadError, load_image
from src.database.db import DatabaseError
from src.database.repository import SOURCE_LABELS, RecordValidationError
from src.hardware.interface import HardwareError, build_result_message
from src.inference.explainability import EXPLANATION_DISCLAIMER, overlay_heatmap
from src.inference.image_quality import QUALITY_DISCLAIMER, assess_image_quality
from src.inference.predictor import PREDICTION_DISCLAIMER, ModelUnavailableError
from src.inference.service import InspectionOutcome
from src.ui.common import get_adapter, get_service, probability_chart, sample_files

DISPLAY_PX = 256
SRC_UPLOAD, SRC_DEMO, SRC_CAMERA = "Upload an image", "Use a demo sample", "Capture from camera"


def _pick_image(cfg: dict):
    """Returns (bytes, filename, source, device_id, capture) or Nones."""
    adapter_kind = cfg["hardware"].get("adapter", "standalone")
    options = [SRC_UPLOAD, SRC_DEMO] + ([SRC_CAMERA] if adapter_kind != "standalone" else [])
    choice = st.radio("Image source", options, horizontal=True, key="inspect_source")
    if choice == SRC_UPLOAD:
        f = st.file_uploader("Steel surface image (PNG, JPEG or BMP)", type=["png", "jpg", "jpeg", "bmp"],
                             key="uploader")
        if f is not None:
            return f.getvalue(), f.name, "upload", None, None
    elif choice == SRC_DEMO:
        files = sample_files()
        if not files:
            st.info("No demo samples found. Run `python scripts/export_samples.py`.")
            return None, None, None, None, None
        sel = st.selectbox("Demo sample from the held-out test split (file name shows the dataset label)", files,
                           format_func=lambda p: p.name, key="demo_sample")
        return sel.read_bytes(), sel.name, "demo_sample", None, None
    else:
        adapter = get_adapter(cfg)
        simulated = adapter.source_type == "simulated_hardware"
        if simulated:
            st.caption(":material/science: Simulated camera: replays stored images. These are not real device "
                       "captures and are recorded as *simulated hardware*.")
        if st.button("Capture frame", icon=":material/photo_camera:", key="capture_btn"):
            try:
                if not adapter.connected:
                    adapter.connect()
                st.session_state["capture"] = adapter.acquire()
            except HardwareError as exc:
                st.session_state.pop("capture", None)
                st.error(f"Capture failed ({exc.code}): {exc}")
        cap = st.session_state.get("capture")
        if cap is not None:
            return cap.image_bytes, cap.filename, cap.source, cap.device_id, cap
    return None, None, None, None, None


def _show_quality(report) -> None:
    if report.warnings:
        st.warning("**This image may give an unreliable result:**\n\n" +
                   "\n".join(f"- {i.message}" for i in report.warnings))
    else:
        st.caption(":material/check: No image quality warnings.")
    infos = [i for i in report.issues if i.severity == "info"]
    if infos:
        with st.expander("Image notes"):
            for i in infos:
                st.markdown(f"- {i.message}")
    st.caption(QUALITY_DISCLAIMER)


def _explanation_view(outcome: InspectionOutcome, service, cfg: dict) -> None:
    st.markdown("#### Why the model chose this class")
    p = outcome.prediction
    c1, c2 = st.columns(2)
    explain_cls = c1.selectbox("Explain class", CLASS_NAMES, index=p.class_index,
                               format_func=lambda c: DISPLAY_NAMES[c] + (" (predicted)" if c == p.class_name else ""),
                               key=f"explain_cls_{outcome.inspection_id}")
    layer = c2.selectbox("Feature layer", ["block4", "block3"],
                         format_func=lambda b: {"block4": "Last conv block (coarser, standard)",
                                                "block3": "Third conv block (finer grid)"}[b],
                         key=f"explain_layer_{outcome.inspection_id}")
    idx = CLASS_NAMES.index(explain_cls)
    if outcome.explanation is not None and idx == p.class_index and layer == outcome.explanation.layer:
        expl = outcome.explanation
    else:
        expl = service.predictor.explain(outcome.image_bytes, idx, layer)
    original = load_image(outcome.image_bytes).convert("L").resize((DISPLAY_PX, DISPLAY_PX), Image.Resampling.BILINEAR)
    model_in = Image.fromarray((p.input_array * 255).astype(np.uint8)).resize((DISPLAY_PX, DISPLAY_PX),
                                                                               Image.Resampling.NEAREST)
    cols = st.columns(3)
    cols[0].image(original, caption="Original image (display size)", width="stretch")
    cols[1].image(model_in, caption=f"What the model sees ({p.preprocessing['input_size']} x "
                                    f"{p.preprocessing['input_size']} px)", width="stretch")
    if expl.ok:
        cols[2].image(overlay_heatmap(p.input_array, expl.heatmap, DISPLAY_PX),
                      caption=f"Grad-CAM for {DISPLAY_NAMES[expl.target_name]} "
                              f"(native grid {expl.native_resolution[0]} x {expl.native_resolution[1]})",
                      width="stretch")
    else:
        cols[2].info(expl.message)
    st.caption(EXPLANATION_DISCLAIMER + " Bright = stronger influence (inferno colour scale).")


def _trace_table(outcome: InspectionOutcome) -> None:
    p = outcome.prediction
    prep = p.preprocessing
    rows = [
        ("Inspection ID", outcome.inspection_id),
        ("Date and time", pd.Timestamp(outcome.created_at).tz_convert(None).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Image source", SOURCE_LABELS.get(outcome.source, outcome.source)),
        ("Device", outcome.device_id or "-"),
        ("File", f"{outcome.original_filename or '-'} ({outcome.image_size[0]} x {outcome.image_size[1]} px)"),
        ("Model version", p.model_version),
        ("Preprocessing", f"grayscale, {prep['resample']} resize to {prep['input_size']} x {prep['input_size']}, "
                          "pixel / 255"),
        ("Inference time", f"{p.inference_ms:.1f} ms (preprocessing + forward pass, CPU)"),
        ("Review threshold", "-" if p.review_threshold is None else f"{p.review_threshold:.3f}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), hide_index=True, width="stretch")


def _result_view(outcome: InspectionOutcome, service, cfg: dict) -> None:
    p = outcome.prediction
    st.markdown(f"## {p.display_name}")
    st.markdown(f"Model prediction with a softmax score of **{p.confidence * 100:.1f}%**.")
    if p.review_needed:
        st.warning(f":material/visibility: **Review needed.** {p.review_reason}")
    if outcome.quality.warnings:
        st.warning(":material/photo_camera: Image quality warnings were raised for this image; treat the "
                   "prediction with extra caution.")
    st.caption(PREDICTION_DISCLAIMER)
    st.altair_chart(probability_chart(p.probabilities, p.class_name), width="stretch")
    _explanation_view(outcome, service, cfg)

    st.markdown("#### Traceability")
    _trace_table(outcome)
    if outcome.saved or st.session_state.get("saved_id") == outcome.inspection_id:
        st.success(f"Saved as inspection {outcome.inspection_id}. Find it under *Inspection history*.")
        return
    if service.repository is None:
        st.error("Inspection history is unavailable, so this result cannot be saved.")
        return
    note = st.text_area("Operator note (optional)", max_chars=2000, key=f"note_{outcome.inspection_id}",
                        placeholder="e.g. coil number, line position, what you see on the surface")
    if st.button("Save inspection record", type="primary", icon=":material/save:", key="save_btn"):
        try:
            service.save(outcome, note)
            st.session_state["saved_id"] = outcome.inspection_id
            st.rerun()
        except (DatabaseError, RecordValidationError) as exc:
            st.error(f"The inspection was not saved: {exc}")


def render(cfg: dict) -> None:
    st.title("Inspect a steel surface")
    service = get_service(cfg)
    predictor = service.predictor
    if predictor is None:
        st.error("No trained model is loaded, so real classification is unavailable. You can still check image "
                 "quality. Train and deploy a model with `python scripts/run_experiments.py`.")

    image_bytes, filename, source, device_id, capture = _pick_image(cfg)
    if image_bytes is None:
        st.info("Choose an image to inspect: upload a photo of a steel surface or pick a demo sample.")
        return

    key = hashlib.sha256(image_bytes).hexdigest() + f"|{source}|{filename}"
    if st.session_state.get("inspect_key") != key:  # new image -> forget the previous result
        st.session_state["inspect_key"] = key
        st.session_state.pop("outcome", None)
        st.session_state.pop("saved_id", None)

    try:
        img = load_image(image_bytes)
    except ImageLoadError as exc:
        st.error(f"This file cannot be used: {exc}")
        return
    input_size = predictor.input_size if predictor else int(cfg["preprocessing"]["input_size"])
    quality = assess_image_quality(img, cfg, input_size, service.quality_thresholds)

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.image(img.convert("L") if img.mode not in ("L", "RGB") else img,
                 caption=f"{filename} | {img.width} x {img.height} px | {img.format}", width="stretch")
        _show_quality(quality)
    with right:
        if st.button("Classify", type="primary", icon=":material/search:", disabled=predictor is None,
                     key="classify_btn"):
            try:
                with st.spinner("Classifying ..."):
                    outcome = service.inspect(image_bytes, source, filename, device_id, explain=True)
                st.session_state["outcome"] = outcome
                if capture is not None:
                    adapter = get_adapter(cfg)
                    adapter.send_result(capture, build_result_message(outcome.summary(), capture))
            except ImageLoadError as exc:
                st.error(f"This file cannot be classified: {exc}")
            except ModelUnavailableError as exc:
                st.error(str(exc))
        outcome = st.session_state.get("outcome")
        if outcome is not None:
            _result_view(outcome, service, cfg)
        elif predictor is not None:
            st.caption("Press Classify to run the model on this image.")
