"""Baseline CNN architecture, training step, checkpoints and training loop."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.config import CLASS_NAMES
from src.data.preprocessing import PreprocessingConfig
from src.models.checkpoints import CheckpointError, load_checkpoint, save_checkpoint
from src.models.cnn import ModelConfig, SteelCNN, count_parameters
from src.training.train import TrainSettings, detect_overfitting, train_model

BLUEPRINT_SHAPES = [("block1", (8, 32, 32)), ("block2", (16, 16, 16)), ("block3", (32, 8, 8)),
                    ("block4", (64, 4, 4)), ("flatten", (1024,)), ("classifier", (6,))]


def test_blueprint_parameter_count_and_shapes():
    m = SteelCNN(ModelConfig(input_size=64))
    assert count_parameters(m) == 30534
    assert m.layer_shapes() == BLUEPRINT_SHAPES


def test_layer_types_match_blueprint():
    m = SteelCNN()
    for name in ("block1", "block2", "block3", "block4"):
        b = getattr(m, name)
        assert b.conv.kernel_size == (3, 3) and b.conv.stride == (1, 1) and b.conv.padding == (1, 1)
        assert isinstance(b.relu, nn.ReLU) and isinstance(b.bn, nn.Identity)
        assert b.pool.kernel_size == 2 and b.pool.stride == 2
    assert m.classifier.in_features == 1024 and m.classifier.out_features == 6
    assert not any(isinstance(mod, (nn.Softmax, nn.BatchNorm2d, nn.Dropout)) for mod in m.modules())


@pytest.mark.parametrize("size,params,flat", [(100, 38214, 2304), (128, 48966, 4096)])
def test_other_input_sizes(size, params, flat):
    m = SteelCNN(ModelConfig(input_size=size))
    assert count_parameters(m) == params
    assert m.classifier.in_features == flat
    assert m(torch.zeros(2, 1, size, size)).shape == (2, 6)


@pytest.mark.parametrize("batch", [1, 7, 32])
def test_output_shape(batch):
    out = SteelCNN()(torch.rand(batch, 1, 64, 64))
    assert out.shape == (batch, len(CLASS_NAMES)) and torch.isfinite(out).all()


def test_rejects_wrong_input_shape():
    m = SteelCNN()
    for bad in (torch.zeros(1, 3, 64, 64), torch.zeros(1, 1, 128, 128), torch.zeros(1, 64, 64)):
        with pytest.raises(ValueError, match="Expected input"):
            m(bad)


def test_logits_not_softmaxed_and_training_step():
    torch.manual_seed(0)
    m = SteelCNN()
    x, y = torch.rand(8, 1, 64, 64), torch.randint(0, 6, (8,))
    logits = m(x)
    assert not torch.allclose(logits.sum(1), torch.ones(8))  # raw logits go to CrossEntropyLoss
    loss = nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    probs = m.predict_proba(x)
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-5)


def test_batchnorm_is_an_explicit_option():
    m = SteelCNN(ModelConfig(use_batchnorm=True))
    assert isinstance(m.block1.bn, nn.BatchNorm2d)
    assert m.block1.conv.bias is None


def test_checkpoint_roundtrip(tmp_path):
    m = SteelCNN().eval()
    path = save_checkpoint(tmp_path / "m.pt", m, PreprocessingConfig(), model_version="test-v1")
    m2, prep, meta = load_checkpoint(path)
    x = torch.rand(3, 1, 64, 64)
    assert torch.equal(m(x), m2(x))
    assert meta["model_version"] == "test-v1" and meta["class_names"] == list(CLASS_NAMES)
    assert prep.input_size == 64 and not m2.training


def test_checkpoint_validation_failures(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    (tmp_path / "junk.pt").write_bytes(b"garbage")
    with pytest.raises(CheckpointError, match="Could not read"):
        load_checkpoint(tmp_path / "junk.pt")
    m = SteelCNN()
    p = save_checkpoint(tmp_path / "bad_classes.pt", m, PreprocessingConfig(), extra={"class_names": ["a"] * 6})
    with pytest.raises(CheckpointError, match="class order"):
        load_checkpoint(p)
    p = save_checkpoint(tmp_path / "bad_size.pt", m, PreprocessingConfig(input_size=128))
    with pytest.raises(CheckpointError):
        load_checkpoint(p)
    payload = torch.load(save_checkpoint(tmp_path / "ok.pt", m, PreprocessingConfig()), weights_only=True)
    payload["model_state"]["classifier.weight"] = torch.zeros(6, 99)
    torch.save(payload, tmp_path / "bad_weights.pt")
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(tmp_path / "bad_weights.pt")


def test_overfitting_detector():
    hist = [{"val_loss": 1.0 - 0.1 * i, "train_acc": 0.5 + 0.02 * i} for i in range(5)]
    hist += [{"val_loss": 0.6 + 0.1 * i, "train_acc": 0.6 + 0.02 * i} for i in range(1, 10)]
    assert "overfitting" in detect_overfitting(hist, window=8)
    assert detect_overfitting(hist[:7], window=8) is None


@pytest.mark.slow
def test_training_loop_on_synthetic_data(synthetic_manifest, synthetic_root, tmp_path):
    s = TrainSettings(input_size=64, augment="flip", seed=3, max_epochs=40, patience=2, batch_size=16)
    logs = []
    res = train_model(synthetic_manifest, synthetic_root, s, tmp_path, log=logs.append)
    assert res.checkpoint_path.is_file() and (res.run_dir / "history.json").is_file()
    assert res.n_params == 30534
    assert 1 <= res.best_epoch <= len(res.history)
    assert all(k in res.history[0] for k in ("train_loss", "train_acc", "val_loss", "val_acc"))
    assert res.best_val_loss == min(h["val_loss"] for h in res.history)
    assert res.stopped_early or len(res.history) == 40
    model, _, meta = load_checkpoint(res.checkpoint_path)
    assert meta["training"]["best_epoch"] == res.best_epoch
    assert any("early stopping" in m or "done in" in m for m in logs)


def test_training_is_reproducible(synthetic_manifest, synthetic_root, tmp_path):
    s = TrainSettings(input_size=64, augment="flip", seed=5, max_epochs=2, patience=5, batch_size=16)
    a = train_model(synthetic_manifest, synthetic_root, s, tmp_path / "a", log=lambda m: None)
    b = train_model(synthetic_manifest, synthetic_root, s, tmp_path / "b", log=lambda m: None)
    assert [h["val_loss"] for h in a.history] == [h["val_loss"] for h in b.history]
    assert a.model_version == b.model_version
