"""End-to-end pipeline tests.

* test_synthetic_pipeline: prepare -> train -> evaluate -> load via inference -> inspect -> save, on
  SYNTHETIC images. It validates software plumbing only; its accuracy is meaningless for NEU.
* real_data tests: verify the delivered model and reports were produced from real test predictions.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_NAMES
from src.data.splitting import load_manifest
from src.database.repository import InspectionRepository
from src.inference.predictor import Predictor
from src.inference.service import InspectionService
from src.training.evaluate import evaluate_checkpoint
from src.training.metrics import compute_metrics
from tests.conftest import (DEPLOYED_MODEL, FINAL_REPORT, REAL_DATASET, REAL_SPLITS, requires_real_model,
                            write_config)


@pytest.mark.slow
def test_synthetic_pipeline(tmp_path, synthetic_root):
    from scripts import prepare_dataset, train_model
    cfg_path = write_config(tmp_path, {
        "paths": {"dataset_root": str(synthetic_root), "splits_dir": str(tmp_path / "splits"),
                  "runs_dir": str(tmp_path / "runs"), "models_dir": str(tmp_path / "models"),
                  "reports_dir": str(tmp_path / "reports"), "inspections_dir": str(tmp_path / "insp")},
        "inference": {"model_path": str(tmp_path / "models" / "model.pt")},
        "training": {"max_epochs": 8, "early_stopping_patience": 4},
    })
    assert prepare_dataset.main(["--config", str(cfg_path)]) == 0
    assert train_model.main(["--config", str(cfg_path), "--run-name", "e2e", "--deploy"]) == 0
    model_path = tmp_path / "models" / "model.pt"
    assert model_path.is_file()

    manifest = load_manifest(tmp_path / "splits")
    rep = evaluate_checkpoint(model_path, manifest, synthetic_root, tmp_path / "reports" / "e2e", "test",
                              latency_runs=5, latency_warmup=1, history_path=tmp_path / "runs" / "e2e" / "history.json")
    assert rep["metrics"]["n"] == int((manifest["split"] == "test").sum())
    for fig in ("confusion_matrix.png", "training_curves.png", "prediction_examples.png", "metrics.json"):
        assert (tmp_path / "reports" / "e2e" / fig).is_file()

    from scripts import calibrate_threshold
    assert calibrate_threshold.main(["--config", str(cfg_path)]) == 0

    from src.config import load_config
    cfg = load_config(cfg_path)
    predictor = Predictor.from_checkpoint(model_path)
    assert predictor.review_threshold is not None
    service = InspectionService(cfg, predictor, InspectionRepository(tmp_path / "insp"))
    row = manifest[manifest["split"] == "test"].iloc[0]
    outcome = service.inspect((synthetic_root / row["rel_path"]).read_bytes(), "upload", row["rel_path"])
    rec = service.save(outcome, "e2e")
    assert service.repository.get(rec.inspection_id).predicted_class in CLASS_NAMES

    from scripts import evaluate_model
    assert evaluate_model.main(["--config", str(cfg_path), "--out", str(tmp_path / "reports" / "cli")]) == 0

    # Deploy a second model: previous one is backed up, report + history follow the new model, rollback works.
    from scripts import deploy_model
    first_version = Predictor.from_checkpoint(model_path).model_version
    assert train_model.main(["--config", str(cfg_path), "--run-name", "e2e_b", "--seed", "1"]) == 0
    new_ckpt = tmp_path / "runs" / "e2e_b" / "best.pt"
    assert deploy_model.main(["--config", str(cfg_path), "--checkpoint", str(new_ckpt)]) == 0
    deployed = Predictor.from_checkpoint(model_path)
    assert deployed.model_version != first_version and deployed.threshold_source.startswith("validation")
    backups = list((tmp_path / "models" / "previous").glob("*.pt"))
    assert len(backups) == 1 and first_version in backups[0].name
    final = json.loads((tmp_path / "reports" / "final" / "metrics.json").read_text())
    assert final["model_version"] == deployed.model_version and "review_on_test" in final
    hist = json.loads((tmp_path / "models" / "training_history.json").read_text())
    assert hist["model_version"] == deployed.model_version
    assert deploy_model.main(["--config", str(cfg_path), "--checkpoint", str(backups[0])]) == 0  # roll back
    assert Predictor.from_checkpoint(model_path).model_version == first_version
    assert deploy_model.main(["--config", str(cfg_path), "--checkpoint", str(tmp_path / "nope.pt")]) == 2


@pytest.mark.real_data
@requires_real_model
def test_reported_metrics_match_recomputed_predictions():
    """The final report was computed from actual test predictions, and re-running the model reproduces them."""
    report = json.loads(FINAL_REPORT.read_text())
    preds = pd.read_csv(FINAL_REPORT.parent / "predictions_test.csv")
    manifest = load_manifest(REAL_SPLITS)
    test = manifest[manifest["split"] == "test"]
    assert len(preds) == len(test) == report["metrics"]["n"]
    assert set(preds["rel_path"]) == set(test["rel_path"])
    y = preds["label"].map(CLASS_NAMES.index).to_numpy()
    probs = preds[[f"p_{c}" for c in CLASS_NAMES]].to_numpy()
    recomputed = compute_metrics(y, probs)
    assert recomputed["accuracy"] == pytest.approx(report["metrics"]["accuracy"])
    assert recomputed["confusion_matrix"] == report["metrics"]["confusion_matrix"]

    p = Predictor.from_checkpoint(DEPLOYED_MODEL)
    assert p.model_version == report["model_version"]
    for _, row in preds.sample(20, random_state=0).iterrows():
        r = p.predict(REAL_DATASET / row["rel_path"])
        assert r.class_name == row["predicted"]
        assert r.confidence == pytest.approx(row["confidence"], abs=1e-5)


@pytest.mark.real_data
@requires_real_model
def test_test_split_never_used_for_selection():
    exp = json.loads((FINAL_REPORT.parents[1] / "experiments.json").read_text())
    final_name = exp["deployment"]["config"]
    per_seed = exp["test"][final_name]["per_seed"]
    chosen = min(per_seed, key=lambda s: s["val_loss"])
    assert chosen["run"] == exp["deployment"]["run"]  # chosen by validation loss
    for r in exp["all_runs"].values():
        assert "test_acc" not in r  # selection records carry validation metrics only
    report = json.loads(FINAL_REPORT.read_text())
    assert report["review_threshold"]["n_val"] == 270
    assert np.isfinite(report["metrics"]["ece"])
