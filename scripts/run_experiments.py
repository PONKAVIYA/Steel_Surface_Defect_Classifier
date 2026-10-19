"""Run the full, pre-registered experiment protocol and deploy the selected model.

Protocol (selection uses the VALIDATION split only; the test split is read only in step 5):
  1. Augmentation policy at 64 px: none / flip / flip_rot90                 (seeds S)
  2. Input size with the chosen policy: 64 / 100 / 128 px                    (seeds S)
  3. BatchNorm experiment at the chosen size + policy (reported, not deployed) (seeds S)
  4. Select the configuration with the highest mean validation accuracy among non-BN runs
     (ties within one validation image broken by lower mean validation loss, then smaller input).
     Train the baseline (64 px + chosen policy) and the selected configuration on FINAL seeds.
  5. Evaluate both on the untouched test split (mean +/- std over seeds).
  6. Deploy the selected configuration's seed with the lowest validation loss, calibrate the
     review threshold on validation, and write the full test report for the deployed model.

Usage:
    python scripts/run_experiments.py                    # full protocol (~30 min on a 2-core CPU)
    python scripts/run_experiments.py --seeds 0 --final-seeds 0 1 --max-epochs 5 --tag quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.config import CLASS_NAMES, DISPLAY_NAMES, load_config, resolve_path  # noqa: E402
from src.data.dataset import SteelDefectDataset  # noqa: E402
from src.data.preprocessing import load_image  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402
from src.models.checkpoints import load_checkpoint  # noqa: E402
from src.training.deploy import deploy_checkpoint  # noqa: E402
from src.training.evaluate import evaluate_checkpoint, predict_tensor_batch  # noqa: E402
from src.training.metrics import compute_metrics, summarize_runs  # noqa: E402
from src.training.train import TrainSettings, train_model  # noqa: E402
from src.utils import plotting  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402
from src.utils.plotting import plt  # noqa: E402

log = get_logger("experiments")


class Runner:
    def __init__(self, cfg, manifest, dataset_root, runs_dir, max_epochs=None):
        self.cfg = cfg
        self.manifest = manifest
        self.root = dataset_root
        self.runs_dir = runs_dir
        self.max_epochs = max_epochs
        self.cache: dict = {}
        self.results: dict[str, dict] = {}

    def run(self, input_size: int, augment: str, seed: int, batchnorm: bool = False) -> dict:
        s = TrainSettings.from_config(self.cfg, input_size=input_size, augment=augment, seed=seed,
                                      use_batchnorm=batchnorm, max_epochs=self.max_epochs)
        name = s.run_name()
        if name in self.results:
            return self.results[name]
        run_dir = self.runs_dir / name
        hist_path = run_dir / "history.json"
        ckpt = run_dir / "best.pt"
        reuse = False
        if ckpt.is_file() and hist_path.is_file():
            with open(hist_path, encoding="utf-8") as fh:
                h = json.load(fh)
            reuse = h.get("settings") == dict(s.__dict__)
        if reuse:
            log.info("[%s] reusing existing run", name)
        else:
            train_model(self.manifest, self.root, s, self.runs_dir, self.cfg["preprocessing"]["resample"],
                        log=log.info, cached_datasets=self.cache)
            with open(hist_path, encoding="utf-8") as fh:
                h = json.load(fh)
        # Validation metrics of the selected (best-val-loss) checkpoint.
        model, prep, _ = load_checkpoint(ckpt)
        if input_size not in self.cache:
            self.cache[input_size] = (
                SteelDefectDataset(self.manifest, self.root, "train", prep, "none", seed),
                SteelDefectDataset(self.manifest, self.root, "val", prep, "none"),
            )
        val_ds = self.cache[input_size][1]
        probs = predict_tensor_batch(model, val_ds.x)
        vm = compute_metrics(val_ds.y.numpy(), probs)
        best = h["history"][h["best_epoch"] - 1]
        res = {
            "run": name, "input_size": input_size, "augment": augment, "seed": seed, "batchnorm": batchnorm,
            "checkpoint": str(ckpt), "best_epoch": h["best_epoch"], "epochs_run": len(h["history"]),
            "stopped_early": h["stopped_early"], "overfitting_warning": h.get("overfitting_warning"),
            "train_seconds": h["train_seconds"], "n_params": h["n_params"],
            "val_loss": best["val_loss"], "val_acc": vm["accuracy"], "train_acc_at_best": best["train_acc"],
            "val_macro_f1": vm["macro"]["f1"],
            "val_recall": {c: vm["per_class"][c]["recall"] for c in CLASS_NAMES},
            "val_pitted_vs_inclusion": vm["pitted_vs_inclusion"],
        }
        self.results[name] = res
        return res


def aggregate(results: list[dict]) -> dict:
    return {
        "val_acc": summarize_runs([r["val_acc"] for r in results]),
        "val_loss": summarize_runs([r["val_loss"] for r in results]),
        "val_macro_f1": summarize_runs([r["val_macro_f1"] for r in results]),
        "val_recall": {c: summarize_runs([r["val_recall"][c] for r in results]) for c in CLASS_NAMES},
        "epochs_run": summarize_runs([r["epochs_run"] for r in results]),
        "train_seconds": summarize_runs([r["train_seconds"] for r in results]),
        "n_params": results[0]["n_params"],
        "seeds": [r["seed"] for r in results],
        "overfitting_warnings": sum(1 for r in results if r["overfitting_warning"]),
    }


def pick_best(candidates: dict[str, dict], tie_tol: float) -> str:
    """Highest mean val acc; ties within tie_tol -> lowest mean val loss -> smaller input."""
    best_acc = max(c["val_acc"]["mean"] for c in candidates.values())
    tied = {k: c for k, c in candidates.items() if c["val_acc"]["mean"] >= best_acc - tie_tol}
    return min(tied, key=lambda k: (tied[k]["val_loss"]["mean"], tied[k]["input_size"]))


def resolution_figure(manifest, root: Path, out: Path) -> None:
    """Same crazing and scratches images at native 200 px and at 128 / 100 / 64 px model input."""
    rows = []
    for cls in ("crazing", "scratches"):
        r = manifest[(manifest["split"] == "train") & (manifest["label"] == cls)].sort_values("rel_path").iloc[3]
        rows.append((cls, r["rel_path"]))
    sizes = [200, 128, 100, 64]
    fig, axes = plt.subplots(len(rows), len(sizes), figsize=(2.2 * len(sizes), 2.45 * len(rows)))
    for i, (cls, rel) in enumerate(rows):
        img = load_image(root / rel).convert("L")
        for j, s in enumerate(sizes):
            ax = axes[i, j]
            ax.axis("off")
            im = img if s == 200 else img.resize((s, s), resample=2)  # 2 = bilinear (as in preprocessing)
            ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{DISPLAY_NAMES[cls]} - {s} px" + (" (native)" if s == 200 else ""), fontsize=8,
                         loc="left", color=plotting.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def size_recall_figure(size_aggs: dict[int, dict], out: Path) -> None:
    """Dot plot (mean +/- 1 std over seeds). Dots, not bars, because the axis does not start at zero."""
    sizes = sorted(size_aggs)
    x = np.arange(len(CLASS_NAMES))
    offset = 0.22
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, s in enumerate(sizes):
        means = [size_aggs[s]["val_recall"][c]["mean"] for c in CLASS_NAMES]
        stds = [size_aggs[s]["val_recall"][c]["std"] for c in CLASS_NAMES]
        ax.errorbar(x + (k - (len(sizes) - 1) / 2) * offset, means, yerr=stds, fmt="o", markersize=7,
                    color=plotting.CATEGORICAL[k], ecolor=plotting.CATEGORICAL[k], elinewidth=1.5, capsize=0,
                    label=f"{s} px", markeredgecolor=plotting.SURFACE, markeredgewidth=1.5)
    ax.set_xticks(x, [DISPLAY_NAMES[c] for c in CLASS_NAMES])
    lo = min(size_aggs[s]["val_recall"][c]["mean"] - size_aggs[s]["val_recall"][c]["std"]
             for s in sizes for c in CLASS_NAMES)
    ax.set_ylim(max(0.0, lo - 0.03), 1.01)
    ax.set_ylabel("Validation recall")
    ax.set_title("Per-class validation recall by input size (mean ± 1 std over 3 seeds)", loc="left", pad=26)
    ax.legend(ncol=len(sizes), loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds for selection experiments")
    ap.add_argument("--final-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="seeds for test reporting")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for runs/reports folders (e.g. 'quick')")
    ap.add_argument("--no-deploy", action="store_true")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    splits_dir = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits_dir)
    split_meta = load_split_meta(splits_dir)
    root = resolve_path(split_meta.get("dataset_root", cfg["paths"]["dataset_root"]))
    tag = f"_{args.tag}" if args.tag else ""
    runs_dir = resolve_path(cfg["paths"]["runs_dir"] + tag)
    reports = resolve_path(cfg["paths"]["reports_dir"] + tag)
    reports.mkdir(parents=True, exist_ok=True)
    runner = Runner(cfg, manifest, root, runs_dir, args.max_epochs)
    tie_tol = 1.0 / max(1, int((manifest["split"] == "val").sum()))
    out = {"protocol": __doc__.split("Usage:")[0].strip(), "seeds": args.seeds, "final_seeds": args.final_seeds,
           "split_meta": split_meta, "tie_tolerance": tie_tol}

    # 1. Augmentation policy at 64 px.
    aug_aggs = {}
    for policy in ("none", "flip", "flip_rot90"):
        rs = [runner.run(64, policy, s) for s in args.seeds]
        aug_aggs[policy] = {"input_size": 64, "augment": policy, **aggregate(rs)}
        log.info("AUG %-10s val_acc %.4f +/- %.4f  val_loss %.4f", policy, aug_aggs[policy]["val_acc"]["mean"],
                 aug_aggs[policy]["val_acc"]["std"], aug_aggs[policy]["val_loss"]["mean"])
    policy = pick_best(aug_aggs, tie_tol)
    log.info("Selected augmentation policy (validation): %s", policy)
    out["augmentation"] = {"results": aug_aggs, "selected": policy}

    # 2. Input size.
    size_aggs = {}
    for size in (64, 100, 128):
        rs = [runner.run(size, policy, s) for s in args.seeds]
        size_aggs[size] = {"input_size": size, "augment": policy, **aggregate(rs)}
        log.info("SIZE %3d val_acc %.4f +/- %.4f  val_loss %.4f", size, size_aggs[size]["val_acc"]["mean"],
                 size_aggs[size]["val_acc"]["std"], size_aggs[size]["val_loss"]["mean"])
    sel_key = pick_best({str(k): v for k, v in size_aggs.items()}, tie_tol)
    sel_size = int(sel_key)
    log.info("Selected input size (validation): %d px", sel_size)
    out["input_size"] = {"results": {str(k): v for k, v in size_aggs.items()}, "selected": sel_size}
    size_recall_figure(size_aggs, reports / "val_recall_by_input_size.png")
    resolution_figure(manifest, root, reports / "resolution_examples.png")

    # 3. BatchNorm experiment (reported only).
    bn_rs = [runner.run(sel_size, policy, s, batchnorm=True) for s in args.seeds]
    out["batchnorm_experiment"] = {
        "input_size": sel_size, "augment": policy, "with_bn": aggregate(bn_rs),
        "without_bn": size_aggs[sel_size],
        "note": "Reported for information; the deployed model keeps the blueprint architecture (no BatchNorm).",
    }
    log.info("BN   %3d val_acc %.4f (no BN %.4f)", sel_size, out["batchnorm_experiment"]["with_bn"]["val_acc"]["mean"],
             size_aggs[sel_size]["val_acc"]["mean"])

    # 4 + 5. Final seeds, test evaluation for the baseline and the selected configuration.
    configs = {"baseline_64px": 64}
    if sel_size != 64:
        configs[f"selected_{sel_size}px"] = sel_size
    out["test"] = {}
    for name, size in configs.items():
        rs = [runner.run(size, policy, s) for s in args.final_seeds]
        per_seed = []
        for r in rs:
            rep = evaluate_checkpoint(r["checkpoint"], manifest, root, reports / "test_runs" / r["run"], "test",
                                      latency_runs=50, latency_warmup=10, split_meta=split_meta,
                                      make_figures=False)
            m = rep["metrics"]
            per_seed.append({"seed": r["seed"], "run": r["run"], "val_loss": r["val_loss"], "val_acc": r["val_acc"],
                             "test_acc": m["accuracy"], "test_macro_f1": m["macro"]["f1"],
                             "test_recall": {c: m["per_class"][c]["recall"] for c in CLASS_NAMES},
                             "pitted_vs_inclusion": m["pitted_vs_inclusion"], "test_ece": m["ece"]})
        out["test"][name] = {
            "input_size": size, "augment": policy, "per_seed": per_seed,
            "test_acc": summarize_runs([p["test_acc"] for p in per_seed]),
            "test_macro_f1": summarize_runs([p["test_macro_f1"] for p in per_seed]),
            "test_recall": {c: summarize_runs([p["test_recall"][c] for p in per_seed]) for c in CLASS_NAMES},
            "val_acc": summarize_runs([p["val_acc"] for p in per_seed]),
        }
        log.info("TEST %-16s acc %.4f +/- %.4f  macroF1 %.4f", name, out["test"][name]["test_acc"]["mean"],
                 out["test"][name]["test_acc"]["std"], out["test"][name]["test_macro_f1"]["mean"])

    # 6. Deploy the selected configuration's lowest-validation-loss seed.
    final_name = list(configs)[-1]
    final_runs = [runner.results[p["run"]] for p in out["test"][final_name]["per_seed"]]
    chosen = min(final_runs, key=lambda r: r["val_loss"])
    out["deployment"] = {"config": final_name, "run": chosen["run"], "rule": "lowest validation loss among final seeds"}
    if not args.no_deploy:
        dest = resolve_path(cfg["inference"]["model_path"] if not tag else f"artifacts/models{tag}/steel_cnn_fp32.pt")
        info = deploy_checkpoint(cfg, manifest, root, split_meta, Path(chosen["checkpoint"]), dest, reports / "final",
                                 {**out["deployment"], "selected_augment": policy, "selected_input_size": sel_size},
                                 log=log.info)
        out["deployment"].update(info)

    out["all_runs"] = runner.results
    with open(reports / "experiments.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("Wrote %s", reports / "experiments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
