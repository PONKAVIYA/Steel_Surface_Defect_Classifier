
"""Model evaluation page: shows only metrics that were actually measured and saved by the scripts."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import CLASS_NAMES, DISPLAY_NAMES, resolve_path
from src.ui.common import pct, read_json


def _ms(d: dict) -> str:
    return f"{d['mean'] * 100:.2f}% ± {d['std'] * 100:.2f}%"


def render(cfg: dict) -> None:
    st.title("Model Evaluation")

    reports = resolve_path(cfg["paths"]["reports_dir"])
    final = read_json(reports / "final" / "metrics.json")
    exp = read_json(reports / "experiments.json")
    quant = read_json(reports / "quantization" / "quantization.json")

    if final is None:
        st.info(
            "No evaluation results are available yet. They appear here after running "
            "`python scripts/run_experiments.py` (or `scripts/evaluate_model.py`). "
            "Nothing is shown until real test-set predictions exist."
        )
        return

    m = final["metrics"]
    mc = final["model_config"]
    prep = final["preprocessing"]
    tr = final.get("training", {})

    st.markdown(
        f"Deployed model `{final['model_version']}` evaluated on the "
        f"**held-out test split** ({m['n']} images, "
        f"split seed {final.get('split_meta', {}).get('seed', '?')}). "
        "These images were not used for training, checkpoint selection, "
        "configuration choice or threshold calibration."
    )

    # ---------------------------------------------------------
    # Overall model performance
    # ---------------------------------------------------------
    k = st.columns(4)

    k[0].metric("Test Accuracy", pct(m["accuracy"], 2))
    k[1].metric("Macro F1", pct(m["macro"]["f1"], 2))
    k[2].metric("Weighted F1", pct(m["weighted"]["f1"], 2))
    k[3].metric(
        "Median Latency / Image",
        f"{final['latency']['pipeline_ms']['median']:.2f} ms",
        help="Decode + preprocess + forward pass at batch size 1 on the CPU listed below"
    )

    # ---------------------------------------------------------
    # Class-wise performance
    # ---------------------------------------------------------
    st.subheader("Class-wise Performance")

    per = pd.DataFrame([
        {
            "Class": DISPLAY_NAMES[c],
            "Accuracy (Recall)": m["per_class"][c]["recall"],
            "Precision": m["per_class"][c]["precision"],
            "Recall": m["per_class"][c]["recall"],
            "F1 Score": m["per_class"][c]["f1"],
            "Support": m["per_class"][c]["support"],
        }
        for c in CLASS_NAMES
    ])

    st.dataframe(
        per,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (Recall)": st.column_config.NumberColumn(
                "Accuracy (Recall)",
                format="percent"
            ),
            "Precision": st.column_config.NumberColumn(
                "Precision",
                format="percent"
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                format="percent"
            ),
            "F1 Score": st.column_config.NumberColumn(
                "F1 Score",
                format="percent"
            ),
            "Support": st.column_config.NumberColumn(
                "Support",
                format="%d"
            ),
        }
    )

    st.caption(
        "Class-wise accuracy is shown as recall (sensitivity): correctly "
        "classified images of a class divided by the total actual images "
        "belonging to that class."
    )

    # ---------------------------------------------------------
    # Class-wise accuracy chart
    # ---------------------------------------------------------
    st.subheader("Class-wise Accuracy Comparison")

    accuracy_fig = px.bar(
        per,
        x="Class",
        y="Accuracy (Recall)",
        title="Accuracy by Defect Class",
        text="Accuracy (Recall)",
        range_y=[0, 1],
        labels={
            "Class": "Defect Class",
            "Accuracy (Recall)": "Accuracy"
        },
    )

    accuracy_fig.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        cliponaxis=False
    )

    accuracy_fig.update_layout(
        yaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "title": "Class-wise Accuracy",
            "dtick": 0.1
        },
        xaxis_title="Defect Class",
        showlegend=False,
        height=480,
        margin=dict(t=70, b=80)
    )

    st.plotly_chart(accuracy_fig, use_container_width=True)

    # ---------------------------------------------------------
    # Precision, Recall and F1 comparison
    # ---------------------------------------------------------
    st.subheader("Precision, Recall and F1 Comparison")

    metric_data = per.melt(
        id_vars="Class",
        value_vars=["Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )

    metric_fig = px.bar(
        metric_data,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
        text="Score",
        title="Precision, Recall and F1 by Defect Class"
    )

    metric_fig.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside",
        cliponaxis=False
    )

    metric_fig.update_layout(
        yaxis={
            "range": [0, 1],
            "tickformat": ".0%",
            "title": "Score",
            "dtick": 0.1
        },
        xaxis_title="Defect Class",
        height=500,
        margin=dict(t=70, b=80)
    )

    st.plotly_chart(metric_fig, use_container_width=True)

    # ---------------------------------------------------------
    # Confusion between Pitted Surface and Inclusion
    # ---------------------------------------------------------
    pvi = m["pitted_vs_inclusion"]

    st.subheader("Pitted Surface vs Inclusion")

    st.markdown(
        f"Pitted surface vs inclusion: "
        f"{pvi['pitted_surface_predicted_as_inclusion']} of "
        f"{pvi['pitted_surface_support']} pitted-surface images were "
        f"predicted as inclusion, and "
        f"{pvi['inclusion_predicted_as_pitted_surface']} of "
        f"{pvi['inclusion_support']} inclusion images were predicted "
        "as pitted surface."
    )

    # ---------------------------------------------------------
    # Single-defect and multi-defect subsets
    # ---------------------------------------------------------
    if m.get("subsets"):
        s = m["subsets"]

        st.subheader("Single-defect vs Multi-defect Images")

        st.markdown(
            f"Images whose NEU annotations also contain a second defect type: "
            f"accuracy {pct(s['multi_defect_images']['accuracy'])} on "
            f"{s['multi_defect_images']['n']} images, versus "
            f"{pct(s['single_defect_images']['accuracy'])} on "
            f"{s['single_defect_images']['n']} single-defect images."
        )

    # ---------------------------------------------------------
    # Evaluation images
    # ---------------------------------------------------------
    st.subheader("Evaluation Visualizations")

    cm = reports / "final" / "confusion_matrix.png"

    if cm.is_file():
        st.image(
            str(cm),
            caption="Confusion matrix on the test split (rows: true class)",
            width=620
        )

    tc = reports / "final" / "training_curves.png"

    if tc.is_file():
        st.image(
            str(tc),
            caption=(
                "Training and validation curves of the deployed run. "
                "The checkpoint with the lowest validation loss is kept."
            ),
            use_container_width=True
        )

    ex = reports / "final" / "prediction_examples.png"

    if ex.is_file():
        st.image(
            str(ex),
            caption=(
                "Representative correct predictions and the most "
                "confident errors (test split)"
            )
        )

    # ---------------------------------------------------------
    # Stability across training seeds
    # ---------------------------------------------------------
    st.subheader("Stability Across Training Seeds")

    if exp and exp.get("test"):
        rows = []

        for name, t in exp["test"].items():
            rows.append({
                "Configuration": (
                    f"{t['input_size']} px, augmentation {t['augment']}"
                    + (
                        " (blueprint baseline)"
                        if name.startswith("baseline")
                        else " (selected on validation)"
                    )
                ),
                "Seeds": len(t["per_seed"]),
                "Test Accuracy (Mean ± Std)": _ms(t["test_acc"]),
                "Test Macro F1 (Mean ± Std)": _ms(t["test_macro_f1"]),
                "Min - Max Accuracy": (
                    f"{t['test_acc']['min'] * 100:.2f}% - "
                    f"{t['test_acc']['max'] * 100:.2f}%"
                )
            })

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Same split and test images for every run; only the training "
            "seed differs. The deployed seed was chosen by validation loss, "
            "not by test accuracy."
        )

    # ---------------------------------------------------------
    # Design experiments
    # ---------------------------------------------------------
    if exp:
        st.subheader("Design Experiments (Validation Split Only)")

        a = exp["augmentation"]

        aug = pd.DataFrame([
            {
                "Augmentation": p,
                "Val Accuracy (Mean ± Std)": _ms(r["val_acc"]),
                "Val Loss (Mean)": f"{r['val_loss']['mean']:.4f}"
            }
            for p, r in a["results"].items()
        ])

        st.markdown(
            f"**Augmentation at 64 px** (selected: `{a['selected']}`)"
        )

        st.dataframe(
            aug,
            hide_index=True,
            use_container_width=True
        )

        s = exp["input_size"]

        size = pd.DataFrame([
            {
                "Input Size": f"{k} px",
                "Parameters": r["n_params"],
                "Val Accuracy (Mean ± Std)": _ms(r["val_acc"]),
                "Crazing Recall": _ms(r["val_recall"]["crazing"]),
                "Scratches Recall": _ms(r["val_recall"]["scratches"]),
                "Mean Epochs": f"{r['epochs_run']['mean']:.0f}"
            }
            for k, r in s["results"].items()
        ])

        st.markdown(
            f"**Input Resolution** (selected: {s['selected']} px)"
        )

        st.dataframe(
            size,
            hide_index=True,
            use_container_width=True
        )

        if (reports / "val_recall_by_input_size.png").is_file():
            st.image(
                str(reports / "val_recall_by_input_size.png"),
                use_container_width=True
            )

        if (reports / "resolution_examples.png").is_file():
            st.image(
                str(reports / "resolution_examples.png"),
                caption=(
                    "What fine crazing and thin scratches look like "
                    "at each input size"
                ),
                width=820
            )

        # BatchNorm experiment
        bn = exp.get("batchnorm_experiment")

        if bn:
            st.markdown(
                f"**BatchNorm experiment** ({bn['input_size']} px): "
                f"validation accuracy {_ms(bn['with_bn']['val_acc'])} "
                f"with BatchNorm vs "
                f"{_ms(bn['without_bn']['val_acc'])} without. "
                f"{bn['note']}"
            )

            bnt = read_json(
                reports / "batchnorm_test_check" / "summary.json"
            )

            if bnt:
                st.markdown(
                    f"Test split check of the same BatchNorm runs "
                    f"({len(bnt['runs'])} seeds): accuracy "
                    f"{_ms(bnt['test_acc'])}, macro F1 "
                    f"{_ms(bnt['test_macro_f1'])}. "
                    "Measured after the deployment decision for "
                    "information; it did not change the deployed model."
                )

    # ---------------------------------------------------------
    # Review-needed threshold
    # ---------------------------------------------------------
    rt = final.get("review_threshold")

    if rt:
        st.subheader("Review-needed Threshold")

        rot = final.get("review_on_test", {})

        st.markdown(
            f"Threshold **{rt['threshold']:.3f}** on the top softmax score, "
            f"chosen on the validation split as the lowest value whose "
            f"accepted predictions reach "
            f"{pct(rt['target_accepted_accuracy'], 0)} accuracy "
            f"(validation: {rt['val_flagged']} of {rt['n_val']} images "
            f"flagged, {rt['val_errors_flagged']} of "
            f"{rt['val_errors_total']} errors caught). "
            f"On the test split it flagged {rot.get('flagged', '?')} of "
            f"{rot.get('n', '?')} images and caught "
            f"{rot.get('errors_flagged', '?')} of "
            f"{rot.get('errors_total', '?')} errors; accuracy of the "
            f"unflagged predictions was "
            f"{pct(rot.get('accepted_accuracy'), 2)}."
        )

        st.caption(
            f"Expected calibration error of the softmax scores on test: "
            f"{m['ece']:.3f}. Softmax scores are not calibrated "
            "probabilities; the threshold is a practical triage rule, "
            "not a guarantee."
        )

    # ---------------------------------------------------------
    # FP32 vs INT8 comparison
    # ---------------------------------------------------------
    st.subheader("FP32 vs INT8")

    if quant and quant.get("source_model_version") not in (
        None,
        final["model_version"]
    ):
        st.info(
            f"The saved INT8 comparison belongs to "
            f"`{quant['source_model_version']}`, not to the deployed model. "
            "Run `python scripts/export_model.py` to quantize the current model."
        )

    elif quant and quant.get("status") == "ok":
        q = pd.DataFrame([
            {
                "Model": "FP32",
                "Test Accuracy": pct(quant["fp32"]["accuracy"], 2),
                "Macro F1": pct(quant["fp32"]["macro_f1"], 2),
                "File Size": (
                    f"{quant['fp32']['size_bytes'] / 1024:.1f} KB"
                ),
                "Median Latency": (
                    f"{quant['fp32']['latency_ms']['median']:.3f} ms"
                )
            },
            {
                "Model": "INT8 (post-training static)",
                "Test Accuracy": pct(quant["int8"]["accuracy"], 2),
                "Macro F1": pct(quant["int8"]["macro_f1"], 2),
                "File Size": (
                    f"{quant['int8']['size_bytes'] / 1024:.1f} KB"
                ),
                "Median Latency": (
                    f"{quant['int8']['latency_ms']['median']:.3f} ms"
                )
            }
        ])

        st.dataframe(
            q,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            f"Same {quant['n_test']} test images. Prediction agreement "
            f"FP32 vs INT8: {pct(quant['agreement'], 2)}. "
            f"Latency: model forward pass only, batch 1, engine "
            f"`{quant['engine']}` on {quant['hardware']['cpu']}. "
            "INT8 speed depends on the device; re-measure on the target hardware."
        )

    elif quant:
        st.warning(
            f"INT8 quantization was attempted but is not available: "
            f"{quant.get('reason')}"
        )

    else:
        st.info(
            "INT8 quantization has not been run. "
            "Use `python scripts/export_model.py`."
        )

    # ---------------------------------------------------------
    # Model and measurement details
    # ---------------------------------------------------------
    st.subheader("Model and Measurement Details")

    hw = final["hardware"]

    st.markdown(
        f"- Architecture: 4 conv blocks {mc['channels']} "
        f"(3x3, ReLU, 2x2 max-pool), linear to 6 logits; "
        f"BatchNorm: {mc['use_batchnorm']}; "
        f"{final['n_params']:,} parameters; "
        f"file {final['model_size_bytes'] / 1024:.1f} KB\n"
        f"- Input: {prep['input_size']} x {prep['input_size']} grayscale, "
        f"{prep['resample']} resize, pixel / 255\n"
        f"- Training: Adam lr "
        f"{tr.get('settings', {}).get('learning_rate')}, "
        f"batch {tr.get('settings', {}).get('batch_size')}, "
        f"augmentation {tr.get('settings', {}).get('augment')}, "
        f"seed {tr.get('settings', {}).get('seed')}, best epoch "
        f"{tr.get('best_epoch')} of {tr.get('epochs_run')} "
        f"(early stopping on validation loss)\n"
        f"- Latency measured on {hw['cpu']} "
        f"({hw['torch_threads']} threads), PyTorch {hw['torch']}"
    )
