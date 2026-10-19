# Steel Surface Defect Classifier

Classifies grayscale images of hot-rolled steel surfaces into six defect types (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) with the small custom CNN from the project blueprint, trained on the NEU surface defect dataset. It includes a Streamlit inspection app with Grad-CAM explanations, image-quality warnings and a traceable inspection history, a local HTTP API and adapters for the ECE team's acquisition hardware.

> Every output is a **model prediction** for a person to check, not an engineering diagnosis. The model only knows these six classes and has no "no defect" or "unknown" class.

## Results at a glance

Measured on the held-out NEU test split (270 images, 45 per class) that was never used for training or any design choice:

| | Value |
|---|---|
| Deployed model | blueprint CNN, 64 × 64 input, 30,534 parameters, 125 KB, flip augmentation |
| Test accuracy (deployed seed) | **95.19%** (macro F1 0.952) |
| Test accuracy over 5 training seeds | **94.52% ± 0.80** |
| Weakest classes | pitted surface (recall 0.87) and scratches (0.91), both mostly confused with inclusion |
| Review-needed flag (threshold 0.901 from validation) | flags 14% of test images, catches 11 of 13 errors; unflagged predictions 99.1% correct |
| Latency, CPU, batch 1 | 1.3 ms per image including decoding and preprocessing (0.5 ms forward pass) |
| INT8 (optional) | 43 KB, 94.07% accuracy, 0.41 vs 0.54 ms forward pass |
| BatchNorm variant (experiment, not deployed) | 97.78% ± 0.37 test accuracy over 3 seeds |
| Automated tests | 114 passed, 0 failed, 0 skipped (clean virtual environment) |

Larger inputs (100 and 128 px) did not beat 64 px on validation. Adding BatchNorm did, by about 3 points; it is reported separately because the blueprint baseline has no BatchNorm (switching takes one command, see below).

Full numbers, figures and test logs: [`TEST_REPORT.md`](TEST_REPORT.md), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Quick start

Requirements: Python **3.11 or newer** (tested with 3.13). No GPU or internet connection is needed at run time.

```bash
cd steel-defect-classifier
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # verified in a clean venv

streamlit run app.py                 # then open http://localhost:8501
```

The app listens on this computer only. To reach it from another machine on the lab network: `streamlit run app.py --server.address 0.0.0.0`.

On Linux, the default PyPI PyTorch wheel pulls in several GB of CUDA libraries even for CPU use. To save space you can install the CPU-only build first with `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` and then run the line above. (That index was not reachable from the build environment, so this variant is untested here. Windows and macOS wheels from PyPI are CPU builds already.)

The trained model (`artifacts/models/steel_cnn_fp32.pt`), split manifest and evaluation reports are included, so the app works immediately with the bundled demo samples. The dataset itself is not included: to retrain, re-evaluate or run the real-data tests, put it in `data/NEU-DET/` (next section).

## Dataset layout

Put the NEU dataset in `data/NEU-DET/` (the path is `paths.dataset_root` in `configs/default.yaml`). Any of these layouts works:

```
data/NEU-DET/train_images/crazing_1.jpg (+ train_annotations/*.xml, valid_images/, ...)   NEU-DET / Kaggle
data/NEU-DET/crazing/any_name.png                                                           class folders
data/NEU-DET/Cr_1.bmp, In_1.bmp, Pa_1.bmp, PS_1.bmp, RS_1.bmp, Sc_1.bmp                    NEU-CLS
```

Details, duplicate analysis and split: [`docs/DATASET.md`](docs/DATASET.md).

## Train and evaluate

```bash
python scripts/prepare_dataset.py        # validate images, find duplicates, write the 70/15/15 split manifest
python scripts/run_experiments.py        # full protocol: augmentation + input-size selection on validation,
                                         # multi-seed training, test evaluation, deployment, threshold calibration
python scripts/export_model.py           # optional: INT8 post-training quantization + FP32 vs INT8 comparison
python scripts/export_samples.py         # refresh the demo images in samples/ (taken from the test split)
```

`run_experiments.py` takes about 30-60 minutes on a 2-core CPU and reuses finished runs if interrupted. Single runs and other tools:

```bash
python scripts/train_model.py --input-size 64 --augment flip --seed 0 [--deploy]
python scripts/deploy_model.py --checkpoint <file.pt>   # install a model: validation threshold, backup of the
                                                        # old model in artifacts/models/previous/, new test report
python scripts/calibrate_threshold.py    # re-calibrate the deployed model's review threshold (e.g. after editing the target)
python scripts/evaluate_model.py         # metrics + figures for the deployed model on the test split
python scripts/evaluate_model.py --checkpoint artifacts/runs/<run>/best.pt --out artifacts/reports/<name>
```

To switch to the more accurate BatchNorm variant (97.8% test accuracy, an explicit departure from the blueprint): `python scripts/deploy_model.py --checkpoint artifacts/models/experimental/steel_cnn_bn_fp32.pt`, then raise the review-threshold target as explained in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md#deploying-the-batchnorm-variant-optional).

## Run the tests

```bash
python -m pytest                         # everything (real-data tests run when the dataset and model exist)
python -m pytest -m "not real_data"      # software tests on synthetic images only
```

Tests that need the dataset or trained model are marked `real_data` and are **skipped with a "BLOCKED" reason** when those files are missing, never silently passed.

## Using the app

| Page | What it does |
|---|---|
| Overview | Purpose, model status, dataset facts, the six classes, workflow, limitations |
| Inspect surface | Upload an image, pick a demo sample or capture from the (simulated) camera; quality warnings; classify; scores for all six classes; Grad-CAM; traceability details; save with an operator note |
| Inspection history | Search by ID; filter by class, date, source, review flag; open a record and its stored image; record the class a person confirmed; export CSV |
| Analytics | Counts by predicted class, per day, review-needed, sources, agreement with confirmed labels (only from saved records) |
| Model evaluation | Test-set metrics, confusion matrix, curves, seed variability, design experiments, threshold, FP32 vs INT8 (only measured values) |

**Review needed** appears when the top softmax score is below a threshold chosen on the validation split (the lowest score at which accepted validation predictions reach 99% accuracy). **Image quality warnings** compare the image with the training images (brightness, contrast, blur, saturation, size, aspect ratio, colour). Neither one means the steel is defective.

Inspection records live in `data/inspections/inspections.db` (SQLite); the original image bytes are stored next to it under `images/YYYY/MM/<inspection_id>.<ext>`, referenced from the record.

## Hardware integration (ECE team)

The app works without hardware. For hardware, set `hardware.adapter` in `configs/default.yaml`:

* `simulated` (default): replays `samples/`; every frame is stored as *simulated hardware*.
* `folder`: the device or its host PC drops images into `data/hardware_inbox/`; results are written back as JSON.
* `standalone`: hides the camera option.

Or let the device POST images to the local API: `uvicorn src.api.service:app --host 127.0.0.1 --port 8000` (schema at `/docs`). Headless loop: `python scripts/hardware_loop.py --adapter simulated --count 5`.

Formats, message schema, error codes, timeouts and open questions for the ECE team: [`docs/HARDWARE_INTERFACE.md`](docs/HARDWARE_INTERFACE.md). **No real device has been connected or tested.**

## Project structure

```
app.py                     Streamlit entry point (.streamlit/config.toml: theme, localhost binding)
configs/default.yaml       paths, seeds, model/training settings, thresholds, hardware mode
src/config.py              config loading + the fixed class mapping
src/data/                  preprocessing (shared by training and inference), discovery/validation,
                           duplicate detection, splitting, synthetic test images
src/models/                SteelCNN, checkpoints (validated loading), INT8 quantization
src/training/              training loop, evaluation, metrics, threshold calibration, deployment
src/inference/             predictor, Grad-CAM, image-quality checks, inspection service
src/database/              SQLite schema and inspection repository
src/hardware/              adapter interface, drop-folder adapter, simulated camera
src/api/service.py         FastAPI service
src/ui/                    Streamlit pages
scripts/                   prepare_dataset, run_experiments, train_model, deploy_model, evaluate_model,
                           calibrate_threshold, export_model (INT8), export_samples, hardware_loop
tests/                     pytest suite (synthetic + real-data tests)
artifacts/splits/          split manifest, dataset report, quality reference
artifacts/models/          deployed FP32 (+ INT8) model, threshold calibration, training history;
                           experimental/ holds the BatchNorm variant
artifacts/reports/         experiments.json, final test report + figures, quantization report
docs/                      DATASET.md, EXPERIMENTS.md, HARDWARE_INTERFACE.md
samples/                   12 demo images from the test split
```

## Configuration

Everything tunable is in `configs/default.yaml`. Set `STEEL_CONFIG=/path/to/other.yaml` to run the app, API or scripts with another config. Notable keys: `preprocessing.input_size`, `augmentation.policy`, `training.*`, `inference.model_path`, `inference.review_threshold` (null = use the validated value stored in the model), `quality.*`, `hardware.*`.

## Limitations

* Six closed classes, no unknown-class or defect-free detection.
* Trained and tested on NEU only (one dataset, one acquisition setup). Accuracy on the ECE camera must be measured with labelled images from that camera.
* Brightness and sharpness differ systematically between NEU classes, so the model can partly rely on exposure cues ([`docs/DATASET.md`](docs/DATASET.md)).
* Softmax scores are not calibrated probabilities; the review threshold is a triage rule tuned on 270 validation images.
* Grad-CAM at the last block has a coarse native grid and shows influence, not defect boundaries.
* The API has no authentication; keep it on localhost or a trusted lab network.
* Not production-ready: no long-term monitoring, no real hardware tests, no field validation.
