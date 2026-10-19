# Experiments

All numbers below come from `artifacts/reports/experiments.json`, `artifacts/reports/final/metrics.json`, `artifacts/reports/quantization/quantization.json` and `artifacts/reports/batchnorm_test_check/`, produced by `scripts/run_experiments.py`, `scripts/export_model.py` and `scripts/evaluate_model.py` on 2026-10-19. Hardware: 2-core Intel Xeon @ 2.10 GHz, CPU only, PyTorch 2.14.1, Python 3.13.

## Protocol (fixed before any results)

* One split for everything: seed 42, 1,260 train / 270 validation / 270 test (210 / 45 / 45 per class). Same test images in every comparison.
* Training: Adam, learning rate 0.001, batch 32, cross-entropy on raw logits, at most 100 epochs, early stopping after 15 epochs without validation-loss improvement, the checkpoint with the lowest validation loss is kept.
* **Choices are made on the validation split only**: augmentation policy, then input size (highest mean validation accuracy over 3 seeds; ties within one validation image broken by lower validation loss, then smaller input).
* BatchNorm is an experiment that is reported but **not deployed** (the blueprint baseline has none).
* The test split is read only after the configuration is fixed: 5 training seeds of the selected configuration, reported as mean ± std. The deployed seed is the one with the lowest validation loss, not the best test score.
* The review threshold is calibrated on validation.

## 1. Augmentation (64 px, validation, 3 seeds)

| Policy | Val accuracy | Val loss | Scratches recall | Pitted surface recall |
|---|---|---|---|---|
| none | 93.33% ± 1.70 | 0.225 | 0.844 | 0.896 |
| **flip** (horizontal + vertical mirror) | **94.94% ± 0.77** | **0.147** | 0.896 | 0.956 |
| flip_rot90 (all 8 orientations) | 94.57% ± 1.19 | 0.160 | 0.904 | 0.948 |

Why these transforms are label-preserving: NEU patches are crops of a strip surface with no canonical left/right or top/bottom, and a mirrored crack network, pit field or scratch is still the same defect type. 90° rotations are also label-preserving, but they turn rolling-direction scratches (mostly horizontal or vertical bands in NEU) into the perpendicular direction, which may be less realistic for a fixed camera. Validation decided: **flip**, which also roughly halved the seed-to-seed spread.

Caveat: two of the three flip_rot90 runs reached the 100-epoch limit without early stopping, so that policy may be slightly under-trained here.

## 2. Input size (flip, validation, 3 seeds)

| Input | Parameters | Val accuracy | Crazing recall | Scratches recall | Pitted surface recall | Epochs (mean) | Train time (mean) |
|---|---|---|---|---|---|---|---|
| **64 px** | 30,534 | 94.94% ± 0.77 | 0.963 ± 0.013 | 0.896 ± 0.056 | 0.956 | 82 | 77 s |
| 100 px | 38,214 | 95.06% ± 1.07 | 0.993 ± 0.013 | 0.881 ± 0.026 | 0.881 | 64 | 229 s |
| 128 px | 48,966 | 94.81% ± 0.98 | 0.978 ± 0.022 | 0.889 ± 0.044 | 0.881 | 55 | 268 s |

(The conv stack is identical; only the linear layer grows with the flattened feature map: 4×4, 6×6, 8×8 × 64. Training times for 100 and 128 px are inflated: the test suite ran on the same two CPU cores during some of those runs.)

Does 64 px lose thin scratches or fine crazing?
* **Scratches: no evidence.** Recall is about the same at all sizes (0.88-0.90) and the spread between seeds is larger than the differences. The test-set errors on scratches are faint, thin lines predicted as inclusion (see `artifacts/reports/final/prediction_examples.png`); larger inputs did not fix them on validation, which points to low contrast rather than resolution.
* **Crazing: a small, uncertain effect.** 64 px recall is 3 points lower than 100 px on validation (about 1.3 of 45 images per seed). On the test split the selected 64 px model recognises crazing well (0.987 ± 0.012 over 5 seeds; 45/45 for the deployed seed).
* **Pitted surface is better at 64 px** (0.956 vs 0.881), so the larger inputs trade one class for another.

Overall accuracy is tied (64 vs 100 px differ by a third of one validation image), so the rule picked **64 px** on validation loss. It is also 3× faster to train and the blueprint size. `artifacts/reports/resolution_examples.png` shows what crazing and scratches look like at each size; `val_recall_by_input_size.png` plots the table.

## 3. BatchNorm experiment (64 px, flip, validation, 3 seeds)

| | Val accuracy | Val loss | Parameters |
|---|---|---|---|
| Baseline (no BN) | 94.94% ± 0.77 | 0.147 | 30,534 |
| With BatchNorm after each conv | **97.90% ± 0.93** | **0.072** | 30,654 |

This is the largest effect in the study. Per the protocol it is not deployed. For information, the same three BN runs were evaluated on the test split **after** the deployment decision: **97.78% ± 0.37** accuracy (macro F1 0.978), against 94.52% ± 0.80 for the baseline. Pitted-surface/inclusion confusions fall to 0-3 per run. This is a well-supported upgrade if the team accepts the architecture change (see "Deploying the BatchNorm variant" below). The INT8 export in this project supports only the baseline.

## 4. Final evaluation on the test split

### Variability over 5 training seeds (64 px, flip, no BN)

| Seed | Val loss | Val acc | Test acc | Test macro F1 | Test ECE |
|---|---|---|---|---|---|
| 0 | 0.164 | 94.07% | 95.56% | 0.956 | 0.027 |
| **1 (deployed)** | **0.133** | 95.19% | **95.19%** | 0.952 | 0.012 |
| 2 | 0.144 | 95.56% | 94.07% | 0.941 | 0.039 |
| 3 | 0.192 | 93.70% | 94.07% | 0.941 | 0.038 |
| 4 | 0.169 | 92.59% | 93.70% | 0.937 | 0.035 |
| **Mean ± std** | | 94.22% ± 1.19 | **94.52% ± 0.80** | **0.945 ± 0.008** | |

Per-class test recall over the 5 seeds: crazing 0.987 ± 0.012, inclusion 0.902 ± 0.056, patches 1.000 ± 0.000, pitted surface 0.898 ± 0.020, rolled-in scale 0.991 ± 0.012, scratches 0.893 ± 0.019.

### Deployed model `steelcnn-64px-flip-s1-09c58bf9ba` (270 test images)

Accuracy **95.19%**, macro F1 **0.952**, weighted F1 0.952, ECE 0.012, NLL 0.129.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Crazing | 1.000 | 1.000 | 1.000 |
| Inclusion | 0.860 | 0.956 | 0.905 |
| Patches | 1.000 | 1.000 | 1.000 |
| Pitted surface | 0.951 | 0.867 | 0.907 |
| Rolled-in scale | 0.957 | 0.978 | 0.967 |
| Scratches | 0.953 | 0.911 | 0.932 |

Confusion matrix (rows = true, columns = predicted; order Cr, In, Pa, PS, RS, Sc):

```
Crazing          45  0  0  0  0  0
Inclusion         0 43  0  1  0  1
Patches           0  0 45  0  0  0
Pitted surface    0  4  0 39  1  1
Rolled-in scale   0  0  0  1 44  0
Scratches         0  3  0  0  1 41
```

* **Pitted surface vs inclusion:** 4 of 45 pitted-surface images were predicted as inclusion and 1 of 45 inclusion images as pitted surface. Across the 5 seeds: pitted surface → inclusion 2-4 per run, inclusion → pitted surface 0-6 per run.
* **Inclusion is involved in most errors.** In every seed the two most frequent confusions are inclusion ↔ pitted surface (3-8 per run, the top pair in 3 seeds) and inclusion ↔ scratches (3-7 per run, the top pair in the other 2). In the deployed model inclusion has the lowest precision (0.86): it also absorbs 3 faint scratches.
* **Multi-defect images** (annotations contain a second defect type): 85.0% accuracy on 20 test images vs 96.0% on 250 single-defect images. Some of these "errors" may be the model naming the other defect that is really present.
* Training: best epoch 62 of 77 (early stopped), 73 s on CPU. The overfitting monitor fired (validation loss rose from 0.133 to 0.187 while training accuracy kept rising); the epoch-62 checkpoint is what was kept. 8 of the 20 runs in the study triggered this warning; all keep their best-validation checkpoint.
* Size and speed: 30,534 parameters, checkpoint file 125 KB. Median latency at batch size 1: **0.47 ms** for the forward pass and **1.27 ms** for decode + preprocessing + forward (200 runs).

### Review-needed threshold

Chosen on validation as the lowest top-softmax score whose accepted predictions reach 99% accuracy: **0.901**. On validation it flags 44 of 270 images and catches 11 of 13 errors. On the untouched test split it flags **38 of 270** (14%), catches **11 of 13** errors, and the unflagged predictions are **99.14%** correct. Two errors remain above the threshold, so the flag is a triage aid, not a guarantee.

## 5. FP32 vs INT8 (deployed model, same 270 test images)

| | Test accuracy | Macro F1 | File size | Median forward latency |
|---|---|---|---|---|
| FP32 | 95.19% | 0.952 | 125 KB | 0.54 ms |
| INT8 (post-training static, x86 engine) | 94.07% | 0.941 | 43 KB | 0.41 ms |

* Method: PyTorch eager-mode post-training static quantization (`torch.ao.quantization`), conv+ReLU fused per block, per-channel weight observers, calibrated on 512 **training** images. The saved INT8 file was reloaded and checked to give bit-identical outputs to the in-memory model.
* Prediction agreement with FP32: 97.8% (6 of 270 images differ). The INT8 model makes 3 more errors, mostly rolled-in scale (recall 0.978 → 0.911).
* The 1.1-point drop is about the size of the seed-to-seed spread, so quantization-aware training was not attempted. Repeat on the target device: INT8 speed-ups depend on the CPU and runtime, and this measurement was a single run on a shared 2-core VM.
* The app uses FP32 because Grad-CAM needs gradients, which quantized models do not provide.

## Deploying the BatchNorm variant (optional)

If the team decides to adopt BatchNorm, the BN seed with the lowest validation loss (seed 2, `steelcnn-bn-64px-flip-s2-e0672032af`) is included as `artifacts/models/experimental/steel_cnn_bn_fp32.pt` together with its training history. One command installs it:

```bash
python scripts/deploy_model.py --checkpoint artifacts/models/experimental/steel_cnn_bn_fp32.pt
```

It calibrates the review threshold on validation, backs up the current model to `artifacts/models/previous/`, installs the new one and regenerates `artifacts/reports/final/`. Roll back with `--checkpoint artifacts/models/previous/<file>.pt`.

This was dry-run on real data with temporary paths: test accuracy 97.78%, and the backup, history and report all followed the new model. **Adjust the review threshold if you switch.** The BN model reaches 99% accuracy on validation almost everywhere, so the 99% target picks a threshold of only 0.573, which flags 3 of 270 test images and catches 1 of its 6 errors. Raise `inference.threshold_calibration.target_accepted_accuracy` (for example to 1.0) or set `inference.review_threshold` explicitly, keeping in mind that 270 validation images cannot measure error rates much below 0.5%.

INT8 export is implemented for the baseline only; the app's evaluation page hides the INT8 table when it refers to another model.

## Reproducing

```bash
python scripts/prepare_dataset.py     # same seed -> byte-identical manifest (verified)
python scripts/run_experiments.py     # ~45 min on 2 CPU cores; reuses finished runs
python scripts/export_model.py
```
Training uses fixed seeds and deterministic algorithms; re-running a configuration reproduces its validation history exactly (tested in `tests/test_model.py::test_training_is_reproducible`).
