"""Overview page: purpose, defect classes, workflow, model and dataset status."""
from __future__ import annotations

import streamlit as st

from src.config import CLASS_DESCRIPTIONS, CLASS_NAMES, DISPLAY_NAMES, resolve_path
from src.ui.common import get_predictor, pct, read_json, sample_files

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#AEB7C2", fontname="Helvetica",
        fontsize=11, fontcolor="#1D2A35", margin="0.18,0.08"];
  edge [color="#7C8794", arrowsize=0.7];
  a [label="1  Acquire image\\nupload, demo sample\\nor camera (ECE)"];
  b [label="2  Check quality\\nbrightness, contrast,\\nblur, size, colour"];
  c [label="3  Preprocess\\ngrayscale, resize,\\nscale to [0, 1]"];
  d [label="4  Classify\\ncustom CNN,\\nsix classes", color="#3B5BA9", penwidth=1.6];
  e [label="5  Explain\\nGrad-CAM heatmap"];
  f [label="6  Record\\nID, time, model,\\nnote, image"];
  a -> b -> c -> d -> e -> f;
}
"""


def render(cfg: dict) -> None:
    st.title("Steel surface defect classifier")
    st.markdown(
        "Classifies grayscale images of hot-rolled steel surfaces into six defect types with a small custom "
        "convolutional neural network trained on the NEU surface defect dataset. It is a teaching and "
        "demonstration system for an IT + ECE inspection project: every result is a model prediction to be "
        "checked by a person, not an engineering diagnosis."
    )

    predictor, err = get_predictor(cfg)
    final = read_json(resolve_path(cfg["paths"]["reports_dir"]) / "final" / "metrics.json")
    ds = read_json(resolve_path(cfg["paths"]["splits_dir"]) / "dataset_report.json")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Model status")
        if predictor is None:
            st.error(f"No trained model is available, so images cannot be classified. {err}\n\n"
                     "Train one with `python scripts/run_experiments.py` (or `scripts/train_model.py --deploy`).")
        else:
            info = predictor.info()
            thr = info["review_threshold"]
            thr_text = f"{thr:.3f} ({info['review_threshold_source']})" if thr is not None else "not set"
            st.markdown(
                f"- Version: `{info['model_version']}`\n"
                f"- Input: {info['input_size']} x {info['input_size']} grayscale, "
                f"{info['n_params']:,} trainable parameters\n"
                f"- Training augmentation: {info.get('augment') or 'n/a'}\n"
                f"- Review threshold: {thr_text}"
            )
            if final:
                m = final["metrics"]
                st.markdown(
                    f"Measured on the held-out test split ({m['n']} images never used for training or selection): "
                    f"accuracy {pct(m['accuracy'])}, macro F1 {pct(m['macro']['f1'])}. "
                    "See *Model evaluation* for per-class results and limitations."
                )
            else:
                st.info("No test-set evaluation report found yet, so no accuracy is shown.")
    with right:
        st.subheader("Dataset")
        if ds:
            counts = ds["class_counts"]
            sp = ds["split"]["sizes"]
            lo, hi = min(counts.values()), max(counts.values())
            per_class = f"{lo}" if lo == hi else f"{lo} to {hi}"
            st.markdown(
                f"- {ds['n_images']:,} images, {per_class} per class\n"
                f"- Sizes: {', '.join(ds['image_sizes'])} px\n"
                f"- Split (seed {ds['split']['seed']}): {sp['train']} train / {sp['val']} validation / "
                f"{sp['test']} test\n"
                f"- Duplicate groups kept within one split: {ds['duplicates']['n_groups_with_duplicates']}\n"
                f"- Images whose annotations also show a second defect type: {ds['annotations']['n_multi_defect']}"
            )
        else:
            st.info("Dataset not prepared yet. Run `python scripts/prepare_dataset.py`.")

    st.subheader("Inspection workflow")
    st.graphviz_chart(WORKFLOW_DOT, width="stretch")

    st.subheader("The six defect classes")
    samples = {p.name.rsplit("_test_", 1)[0]: p for p in sample_files()}
    cols = st.columns(3, gap="medium")
    for i, cls in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            if cls in samples:
                st.image(str(samples[cls]), width=170)
            st.markdown(f"**{DISPLAY_NAMES[cls]}**  \n{CLASS_DESCRIPTIONS[cls]}")

    st.subheader("Limits to keep in mind")
    st.markdown(
        "- The model only knows these six classes. A defect-free surface or an unknown defect type is still "
        "assigned to one of them; there is no unknown-class detector.\n"
        "- Trained on 200 x 200 px NEU patches under the NEU lighting. Different cameras, lighting or "
        "magnification can lower accuracy; the quality checks flag images far from the training conditions.\n"
        "- Confidence scores are softmax outputs, not calibrated probabilities of being correct.\n"
        "- Grad-CAM heatmaps are approximate explanations, not measured defect boundaries."
    )
