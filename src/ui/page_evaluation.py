"""Model evaluation page: shows only metrics that were actually measured and saved by the scripts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES, resolve_path
from src.ui.common import pct, read_json


def _ms(d: dict) -> str:
    return f"{d['mean'] * 100:.2f}% ± {d['std'] * 100:.2f}"


def render(cfg: dict) -> None:
    st.title("Model evaluation")
    reports = resolve_path(cfg["paths"]["reports_dir"])
    final = read_json(reports / "final" / "metrics.json")
    exp = read_json(reports / "experiments.json")
    quant = read_json(reports / "quantization" / "quantization.json")
    if final is None:
        st.info("No evaluation results are available yet. They appear here after running "
                "`python scripts/run_experiments.py` (or `scripts/evaluate_model.py`). Nothing is shown until real "
                "test-set predictions exist.")
        return

    m = final["metrics"]
    mc, prep, tr = final["model_config"], final["preprocessing"], final.get("training", {})
    st.markdown(
        f"Deployed model `{final['model_version']}` evaluated on the **held-out test split** ({m['n']} images, "
        f"split seed {final.get('split_meta', {}).get('seed', '?')}). These images were not used for training, "
        "checkpoint selection, configuration choice or threshold calibration."
    )
    k = st.columns(4)
    k[0].metric("Test accuracy", pct(m["accuracy"], 2))
    k[1].metric("Macro F1", pct(m["macro"]["f1"], 2))
    k[2].metric("Weighted F1", pct(m["weighted"]["f1"], 2))
    k[3].metric("Median latency / image", f"{final['latency']['pipeline_ms']['median']:.2f} ms",
                help="Decode + preprocess + forward pass at batch size 1 on the CPU listed below")

    st.subheader("Per-class results")
    per = pd.DataFrame([{"Class": DISPLAY_NAMES[c], "Precision": m["per_class"][c]["precision"],
                         "Recall": m["per_class"][c]["recall"], "F1": m["per_class"][c]["f1"],
                         "Support": m["per_class"][c]["support"]} for c in CLASS_NAMES])
    per.loc[len(per)] = ["Macro average", m["macro"]["precision"], m["macro"]["recall"], m["macro"]["f1"], m["n"]]
    per.loc[len(per)] = ["Weighted average", m["weighted"]["precision"], m["weighted"]["recall"],
                         m["weighted"]["f1"], m["n"]]
    st.dataframe(per, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(c, format="%.3f") for c in ("Precision", "Recall", "F1")})
    pvi = m["pitted_vs_inclusion"]
    st.markdown(
        f"Pitted surface vs inclusion: {pvi['pitted_surface_predicted_as_inclusion']} of "
        f"{pvi['pitted_surface_support']} pitted-surface images were predicted as inclusion, and "
        f"{pvi['inclusion_predicted_as_pitted_surface']} of {pvi['inclusion_support']} inclusion images as "
        "pitted surface."
    )
    if m.get("subsets"):
        s = m["subsets"]
        st.markdown(
            f"Images whose NEU annotations also contain a second defect type: accuracy "
            f"{pct(s['multi_defect_images']['accuracy'])} on {s['multi_defect_images']['n']} images, versus "
            f"{pct(s['single_defect_images']['accuracy'])} on {s['single_defect_images']['n']} single-defect images."
        )
    cm = reports / "final" / "confusion_matrix.png"
    if cm.is_file():
        st.image(str(cm), caption="Confusion matrix on the test split (rows: true class)", width=620)
    tc = reports / "final" / "training_curves.png"
    if tc.is_file():
        st.image(str(tc), caption="Training and validation curves of the deployed run. The checkpoint with the "
                 "lowest validation loss is kept.", width="stretch")
    ex = reports / "final" / "prediction_examples.png"
    if ex.is_file():
        st.image(str(ex), caption="Representative correct predictions and the most confident errors (test split)")

    st.subheader("Stability across training seeds")
    if exp and exp.get("test"):
        rows = []
        for name, t in exp["test"].items():
            rows.append({"Configuration": f"{t['input_size']} px, augmentation {t['augment']}"
                                          + (" (blueprint baseline)" if name.startswith("baseline") else
                                             " (selected on validation)"),
                         "Seeds": len(t["per_seed"]), "Test accuracy (mean ± std)": _ms(t["test_acc"]),
                         "Test macro F1 (mean ± std)": _ms(t["test_macro_f1"]),
                         "Min - max accuracy": f"{t['test_acc']['min'] * 100:.2f}% - {t['test_acc']['max'] * 100:.2f}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Same split and test images for every run; only the training seed differs. The deployed seed was "
                   "chosen by validation loss, not by test accuracy.")

    if exp:
        st.subheader("Design experiments (validation split only)")
        a = exp["augmentation"]
        aug = pd.DataFrame([{"Augmentation": p, "Val accuracy (mean ± std)": _ms(r["val_acc"]),
                             "Val loss (mean)": f"{r['val_loss']['mean']:.4f}"} for p, r in a["results"].items()])
        st.markdown(f"**Augmentation at 64 px** (selected: `{a['selected']}`)")
        st.dataframe(aug, hide_index=True, width="stretch")
        s = exp["input_size"]
        size = pd.DataFrame([{"Input size": f"{k} px", "Parameters": r["n_params"],
                              "Val accuracy (mean ± std)": _ms(r["val_acc"]),
                              "Crazing recall": _ms(r["val_recall"]["crazing"]),
                              "Scratches recall": _ms(r["val_recall"]["scratches"]),
                              "Mean epochs": f"{r['epochs_run']['mean']:.0f}"} for k, r in s["results"].items()])
        st.markdown(f"**Input resolution** (selected: {s['selected']} px)")
        st.dataframe(size, hide_index=True, width="stretch")
        if (reports / "val_recall_by_input_size.png").is_file():
            st.image(str(reports / "val_recall_by_input_size.png"), width="stretch")
        if (reports / "resolution_examples.png").is_file():
            st.image(str(reports / "resolution_examples.png"), caption="What fine crazing and thin scratches look "
                     "like at each input size", width=820)
        bn = exp.get("batchnorm_experiment")
        if bn:
            st.markdown(
                f"**BatchNorm experiment** ({bn['input_size']} px): validation accuracy {_ms(bn['with_bn']['val_acc'])} "
                f"with BatchNorm vs {_ms(bn['without_bn']['val_acc'])} without. {bn['note']}")
            bnt = read_json(reports / "batchnorm_test_check" / "summary.json")
            if bnt:
                st.markdown(
                    f"Test split check of the same BatchNorm runs ({len(bnt['runs'])} seeds): accuracy "
                    f"{_ms(bnt['test_acc'])}, macro F1 {_ms(bnt['test_macro_f1'])}. "
                    "Measured after the deployment decision for information; it did not change the deployed model.")

    rt = final.get("review_threshold")
    if rt:
        st.subheader("Review-needed threshold")
        rot = final.get("review_on_test", {})
        st.markdown(
            f"Threshold **{rt['threshold']:.3f}** on the top softmax score, chosen on the validation split as the "
            f"lowest value whose accepted predictions reach {pct(rt['target_accepted_accuracy'], 0)} accuracy "
            f"(validation: {rt['val_flagged']} of {rt['n_val']} images flagged, {rt['val_errors_flagged']} of "
            f"{rt['val_errors_total']} errors caught). On the test split it flagged {rot.get('flagged', '?')} of "
            f"{rot.get('n', '?')} images and caught {rot.get('errors_flagged', '?')} of {rot.get('errors_total', '?')} "
            f"errors; accuracy of the unflagged predictions was {pct(rot.get('accepted_accuracy'), 2)}."
        )
        st.caption(f"Expected calibration error of the softmax scores on test: {m['ece']:.3f}. Softmax scores are "
                   "not calibrated probabilities; the threshold is a practical triage rule, not a guarantee.")

    st.subheader("FP32 vs INT8")
    if quant and quant.get("source_model_version") not in (None, final["model_version"]):
        st.info(f"The saved INT8 comparison belongs to `{quant['source_model_version']}`, not to the deployed model. "
                "Run `python scripts/export_model.py` to quantize the current model.")
    elif quant and quant.get("status") == "ok":
        q = pd.DataFrame([
            {"Model": "FP32", "Test accuracy": pct(quant["fp32"]["accuracy"], 2),
             "Macro F1": pct(quant["fp32"]["macro_f1"], 2), "File size": f"{quant['fp32']['size_bytes'] / 1024:.1f} KB",
             "Median latency": f"{quant['fp32']['latency_ms']['median']:.3f} ms"},
            {"Model": "INT8 (post-training static)", "Test accuracy": pct(quant["int8"]["accuracy"], 2),
             "Macro F1": pct(quant["int8"]["macro_f1"], 2), "File size": f"{quant['int8']['size_bytes'] / 1024:.1f} KB",
             "Median latency": f"{quant['int8']['latency_ms']['median']:.3f} ms"},
        ])
        st.dataframe(q, hide_index=True, width="stretch")
        st.caption(f"Same {quant['n_test']} test images. Prediction agreement FP32 vs INT8: "
                   f"{pct(quant['agreement'], 2)}. Latency: model forward pass only, batch 1, engine "
                   f"`{quant['engine']}` on {quant['hardware']['cpu']}. INT8 speed depends on the device; "
                   "re-measure on the target hardware.")
    elif quant:
        st.warning(f"INT8 quantization was attempted but is not available: {quant.get('reason')}")
    else:
        st.info("INT8 quantization has not been run. Use `python scripts/export_model.py`.")

    st.subheader("Model and measurement details")
    hw = final["hardware"]
    st.markdown(
        f"- Architecture: 4 conv blocks {mc['channels']} (3x3, ReLU, 2x2 max-pool), linear to 6 logits; "
        f"BatchNorm: {mc['use_batchnorm']}; {final['n_params']:,} parameters; file {final['model_size_bytes'] / 1024:.1f} KB\n"
        f"- Input: {prep['input_size']} x {prep['input_size']} grayscale, {prep['resample']} resize, pixel / 255\n"
        f"- Training: Adam lr {tr.get('settings', {}).get('learning_rate')}, batch {tr.get('settings', {}).get('batch_size')}, "
        f"augmentation {tr.get('settings', {}).get('augment')}, seed {tr.get('settings', {}).get('seed')}, best epoch "
        f"{tr.get('best_epoch')} of {tr.get('epochs_run')} (early stopping on validation loss)\n"
        f"- Latency measured on {hw['cpu']} ({hw['torch_threads']} threads), PyTorch {hw['torch']}"
    )
