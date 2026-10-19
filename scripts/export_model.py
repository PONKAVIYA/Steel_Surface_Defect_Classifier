"""Optional INT8 post-training static quantization of the deployed FP32 model + side-by-side comparison.

Usage:
    python scripts/export_model.py                       # calibrate on 512 training images, compare on test
    python scripts/export_model.py --calibration-images 1260

Writes artifacts/models/steel_cnn_int8.pt and artifacts/reports/quantization/quantization.json.
The FP32 model is left untouched. If quantization is unsupported here, the reason is recorded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data.dataset import SteelDefectDataset  # noqa: E402
from src.data.splitting import load_manifest, load_split_meta  # noqa: E402
from src.models.checkpoints import CheckpointError, load_checkpoint  # noqa: E402
from src.models.quantization import QuantizationUnavailable, load_int8, quantize_static, save_int8  # noqa: E402
from src.training.evaluate import hardware_info, predict_tensor_batch  # noqa: E402
from src.training.metrics import compute_metrics  # noqa: E402
from src.utils.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("export_model")


def latency(model, x1: torch.Tensor, runs: int = 300, warmup: int = 30) -> dict:
    with torch.no_grad():
        for _ in range(warmup):
            model(x1)
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x1)
            t.append((time.perf_counter() - t0) * 1000)
    t.sort()
    return {"median": statistics.median(t), "mean": statistics.fmean(t), "p95": t[int(0.95 * (len(t) - 1))],
            "runs": runs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None, help="FP32 checkpoint (default: inference.model_path)")
    ap.add_argument("--calibration-images", type=int, default=512)
    ap.add_argument("--out", default=None, help="INT8 output path (default: inference.int8_model_path)")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")
    cfg = load_config(args.config)
    ckpt = resolve_path(args.checkpoint or cfg["inference"]["model_path"])
    out = resolve_path(args.out or cfg["inference"]["int8_model_path"])
    report_dir = resolve_path(args.report_dir or (cfg["paths"]["reports_dir"] + "/quantization"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "quantization.json"
    try:
        fp32, prep, meta = load_checkpoint(ckpt)
    except CheckpointError as exc:
        log.error("%s", exc)
        return 2
    splits = resolve_path(cfg["paths"]["splits_dir"])
    manifest = load_manifest(splits)
    root = resolve_path(load_split_meta(splits).get("dataset_root", cfg["paths"]["dataset_root"]))
    train = SteelDefectDataset(manifest, root, "train", prep, "none")
    test = SteelDefectDataset(manifest, root, "test", prep, "none")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(train), generator=g)[: args.calibration_images]
    calib = train.x[idx]
    try:
        qmodel, engine = quantize_static(fp32, calib)
        save_int8(out, qmodel, fp32.config, engine, {"source_checkpoint": str(ckpt),
                                                     "source_model_version": meta["model_version"],
                                                     "calibration_images": int(len(calib)),
                                                     "calibration_split": "train"})
        qmodel, _ = load_int8(out)  # verify the saved file loads and runs
    except (QuantizationUnavailable, RuntimeError, ValueError) as exc:
        log.error("INT8 quantization unavailable: %s", exc)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"status": "unavailable", "reason": str(exc), "hardware": hardware_info()}, fh, indent=2)
        return 3

    y = test.y.numpy()
    p32 = predict_tensor_batch(fp32, test.x)
    p8 = predict_tensor_batch(qmodel, test.x)
    m32, m8 = compute_metrics(y, p32), compute_metrics(y, p8)
    x1 = test.x[:1]
    report = {
        "status": "ok",
        "method": "PyTorch eager-mode post-training static quantization (conv+ReLU fused, per-channel weights, "
                  "calibrated on training images)",
        "engine": engine,
        "n_test": int(len(y)),
        "calibration_images": int(len(calib)),
        "source_model_version": meta["model_version"],
        "fp32": {"accuracy": m32["accuracy"], "macro_f1": m32["macro"]["f1"], "size_bytes": ckpt.stat().st_size,
                 "latency_ms": latency(fp32, x1), "per_class_recall": {c: v["recall"] for c, v in m32["per_class"].items()}},
        "int8": {"accuracy": m8["accuracy"], "macro_f1": m8["macro"]["f1"], "size_bytes": out.stat().st_size,
                 "latency_ms": latency(qmodel, x1), "per_class_recall": {c: v["recall"] for c, v in m8["per_class"].items()},
                 "path": str(out)},
        "agreement": float((p32.argmax(1) == p8.argmax(1)).mean()),
        "max_abs_prob_diff": float(np.abs(p32 - p8).max()),
        "hardware": hardware_info(),
        "notes": ["Latency = model forward pass only, batch size 1, same thread count for both models.",
                  "FP32 size is the full checkpoint file (weights + metadata); INT8 size is the INT8 file.",
                  "INT8 speed-ups depend on the CPU and runtime; re-measure on the target device."],
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("FP32 acc %.4f | INT8 acc %.4f | agreement %.4f | size %d -> %d bytes | latency %.3f -> %.3f ms",
             m32["accuracy"], m8["accuracy"], report["agreement"], report["fp32"]["size_bytes"],
             report["int8"]["size_bytes"], report["fp32"]["latency_ms"]["median"], report["int8"]["latency_ms"]["median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
