# Test report

Date: 2026-10-19. Everything below was executed in the build environment; nothing is estimated.

## Summary

| Area | Result |
|---|---|
| Automated tests (system environment) | **114 passed, 0 failed, 0 skipped** (20.1 s) |
| Automated tests (fresh virtual environment from `requirements.txt`) | **114 passed, 0 failed, 0 skipped, 0 warnings** (19.0 s) |
| Real-data tests with dataset/model hidden | 7 **skipped with "BLOCKED" reason**, none passed (verified behaviour) |
| Documented commands | all exited with code 0 (list below) |
| Model training on the real NEU dataset | done: 20 runs, protocol in `docs/EXPERIMENTS.md` |
| Test-set evaluation of the deployed model | done: accuracy **95.19%**, macro F1 0.952 |
| INT8 quantization | done: 94.07% accuracy, loads and runs |
| Real ECE hardware | **not available, not tested** (simulator and drop-folder adapter tested instead) |

## Environment

| | |
|---|---|
| OS | Linux 6.18 x86_64 (cloud VM), 2 logical CPUs, 7 GB RAM, no GPU |
| Python | 3.13.16 |
| torch | 2.14.1 (PyPI wheel; CUDA libraries installed but no GPU, all runs on CPU) |
| numpy / pandas / pillow | 2.5.3 / 3.0.5 / 12.3.0 |
| opencv-python-headless / scikit-learn / matplotlib | 5.0.0.93 / 1.9.1 / 3.11.2 |
| streamlit / altair | 1.66.0 / 6.3.0 |
| fastapi / pydantic / uvicorn / python-multipart / starlette | 0.146.0 / 2.13.5 / 0.53.0 / 0.0.32 / 1.9.0 (venv) |
| pytest / httpx2 | 9.1.1 / 2.13.0 |

Clean-environment check: `python -m venv <tmp>/venv && <tmp>/venv/bin/pip install -r requirements.txt` installed in 1 min 39 s from the pip cache, `pip check` reported no broken requirements, and the full test suite was run with that interpreter.

## Commands executed

```text
python scripts/prepare_dataset.py                                   exit 0  (manifest byte-identical to the previous run)
python scripts/run_experiments.py                                   exit 0  (42 min, 20 training runs)
python scripts/run_experiments.py                                   exit 0  (re-run: all 20 runs reused, identical results; regenerated figures)
python scripts/export_model.py                                      exit 0
python scripts/evaluate_model.py --checkpoint artifacts/runs/cnn_bn_64px_flip_s{0,1,2}/best.pt --out artifacts/reports/batchnorm_test_check/s{0,1,2}   exit 0
python scripts/export_samples.py                                    exit 0
python -m pytest -v                                                 114 passed
python -m pytest -m real_data -rs   (dataset + model temporarily hidden)   7 skipped (BLOCKED)
# with the clean venv interpreter:
python -m pytest -q                                                 114 passed, 0 warnings
python scripts/prepare_dataset.py                                   exit 0, manifest unchanged
python scripts/train_model.py --input-size 64 --augment flip --seed 0 --max-epochs 3 --run-name verify_cli_train   exit 0
python scripts/evaluate_model.py                                    exit 0  (accuracy 0.9519)
python scripts/evaluate_model.py --checkpoint artifacts/runs/cnn_64px_flip_s0/best.pt --out ...   exit 0 (0.9556, matches experiments.json)
python scripts/calibrate_threshold.py                               exit 0  (threshold 0.9007 before and after)
python scripts/export_samples.py                                    exit 0
python scripts/hardware_loop.py --adapter simulated --count 5       exit 0  (5 records, all marked simulated)
python scripts/<each>.py --help                                     exit 0 for all 9 scripts
python scripts/run_experiments.py   (after moving deployment into src/training/deploy.py)   exit 0, identical report
python scripts/deploy_model.py --checkpoint artifacts/models/experimental/steel_cnn_bn_fp32.pt   (temporary config/paths)
                                                                    exit 0: backup made, BN model deployed, test accuracy 0.9778
uvicorn src.api.service:app --host 127.0.0.1 --port 8000            /v1/health ok, /v1/inspect classified samples/scratches_test_01.jpg as Scratches, /docs HTTP 200
streamlit run app.py                                                /_stcore/health "ok" (HTTP 200)
```
Logs: `artifacts/logs/` (experiments, pytest, commands).

The app was also run in a headless Chromium (Playwright) and screenshotted on every page, with a model and without one; the inspect → classify → save → history → analytics flow was exercised that way as well.

## Coverage of the required tests

| Requirement (prompt §13) | Tests | Result |
|---|---|---|
| A. Python version, dependencies, imports, startup, documented commands | clean venv install, `pip check`, command list above, `test_app_smoke.py::test_all_pages_render` | pass |
| B. Six classes detected | `test_discovers_all_six_classes`, `test_real_dataset_manifest` (1,800 = 6 × 300) | pass |
| B. Loading, grayscale, resize, normalise | `test_resize_and_normalize[64,100,128]`, `test_grayscale_conversion_modes`, `test_tensor_shape` | pass |
| B. Corrupted / invalid images give clear errors | `test_corrupted_and_empty_images_are_reported`, `test_corrupted_empty_and_missing_inputs`, `test_unsupported_formats_rejected` | pass |
| B. Manifests do not overlap; counts; reproducible | `test_split_is_stratified_disjoint_and_reproducible`, `test_verify_split_detects_leakage`, `test_duplicate_groups_stay_in_one_split`, `test_real_split_is_reproducible` | pass |
| C. Instantiate, output N × 6, parameter count | `test_blueprint_parameter_count_and_shapes` (30,534; every block shape), `test_layer_types_match_blueprint`, `test_output_shape`, `test_other_input_sizes` | pass |
| C. Finite loss and gradients | `test_logits_not_softmaxed_and_training_step` | pass |
| C. Checkpoint save / load / validation | `test_checkpoint_roundtrip`, `test_checkpoint_validation_failures` | pass |
| D. Formats, dimensions, corrupt files, colour vs grayscale | `test_supported_formats[PNG,JPEG,BMP]`, `test_color_and_odd_sized_inputs`, `test_corrupted_input_raises`, `test_png_jpeg_bmp_give_near_identical_inputs` | pass |
| D. Training and inference preprocessing identical | `test_training_dataset_matches_inference_preprocessing` (exact tensor equality) | pass |
| D. Missing model never yields a prediction | `test_missing_or_invalid_model_never_predicts`, `test_api_without_model_never_predicts`, `test_app_without_model_disables_classification`, `test_hardware_loop_reports_missing_model` | pass |
| D. No state contamination between images | `test_no_state_contamination_between_images`, second-image check in `test_inspect_classify_and_save_workflow` | pass |
| E. Records created / retrieved / filtered / exported | `test_create_and_retrieve_with_image`, `test_filters`, `test_csv_export`, `test_update_note_and_verification` | pass |
| E. Missing images, damaged rows, DB errors | `test_missing_image_and_damaged_row`, `test_database_errors_are_reported`, `test_duplicate_id_fails_without_orphan_image`, `test_invalid_records_rejected` | pass |
| E. App startup and main workflow; analytics from stored records | `test_all_pages_render`, `test_inspect_classify_and_save_workflow` (classify → save → history shows 1 of 1 → analytics) | pass |
| E. Hardware simulation without a device | `test_simulator_marks_frames_as_simulated` (incl. timeout / corrupt frame / disconnect), `test_folder_drop_adapter`, `test_hardware_loop_with_simulator`, `test_simulated_camera_capture` | pass |
| F. End-to-end on real data | `run_experiments.py` (train → best checkpoint → test evaluation), `test_reported_metrics_match_recomputed_predictions` (report = recomputed from the per-image predictions; 20 test images re-predicted through the inference module match), `test_app_with_deployed_model_and_reports` | pass |
| F. Synthetic pipeline smoke test | `test_synthetic_pipeline` (prepare → train → deploy → evaluate → calibrate → inspect → save → redeploy another model → backup and rollback; synthetic images only, accuracy not reported) | pass |
| G. Grad-CAM size, finite, no fabricated maps | `test_gradcam_shape_and_range[block4,block3]`, `test_gradcam_every_class_and_bad_args`, `test_gradcam_refuses_to_fabricate_empty_map` | pass |
| G. Quality warnings on controlled examples | `test_quality_exposure`, `test_quality_flat_blurry_colour_shape_size`, `test_quality_normal_image_has_no_warnings`, `test_quality_messages_never_claim_a_defect` | pass |
| G. Review threshold from validation | `test_threshold_calibration`, `test_review_threshold_logic`, `test_deployed_model_on_demo_samples` (threshold read from checkpoint; flag ⇔ score below it) | pass |
| G. INT8 loads, runs, same test split as FP32 | `test_int8_quantization_roundtrip`, `test_int8_report_uses_same_test_split` | pass |
| Selection never used the test split | `test_test_split_never_used_for_selection` | pass |

## Failures found and fixed during development

Each was fixed in the code (not by loosening a check), then the failing test and the full suite were re-run.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | 11,310 "near-duplicate" pairs, mostly cross-class; split collapsed into giant groups | 8×8 perceptual hash is dominated by the shared lighting gradient of NEU images | Illumination-removed normalised cross-correlation over 8 orientations; validated visually (`docs/DATASET.md`) |
| 2 | Every checkpoint failed to load with `weights_only=True` | `torch.__version__` is a `TorchVersion` object, rejected by the safe unpickler | Store it as a string |
| 3 | History page crashed (`SchemaValidationError`) | Altair 6 rejects `title=None` | Only pass a title when there is one |
| 4 | CSV export labelled **every** record "verified by operator" | With pandas 3 a missing label becomes NaN, which is truthy | Explicit `notna()` check |
| 5 | `FileExistsError` instead of a clear message when the inspections path is a file | Folder creation was outside the error handling | Wrapped into `DatabaseError` |
| 6 | Grad-CAM maximum 0.97 instead of 1 | Bilinear upsampling lowers the peak after normalising | Normalise after upsampling |
| 7 | A quality warning said "stretches defect shapes" | Wording could read as a defect claim | Reworded to "surface features" |
| 8 | INT8 reload printed "must run observer" warnings | Converting the empty graph before loading the saved scales | Warning suppressed at that step; reloaded model verified bit-identical on all 270 test images |
| 9 | Starlette deprecation warning in tests | `httpx` replaced by `httpx2` for the test client | Pinned `httpx2==2.13.0` |
| 10 | Streamlit tried to contact `checkip.amazonaws.com` on start | Public-IP lookup when bound to all interfaces | App binds to `localhost` by default (`.streamlit/config.toml`) |
| 11 | Screenshots: history table cut off, chart labels dropped, time zone shown as "Asia", overlapping figure labels, bar chart with truncated axis | Layout and formatting | Full-width table, explicit chart row height, numeric UTC offset, figure margins, dot plot instead of truncated bars |
| 12 | My first instructions for switching to the BatchNorm model pointed at `artifacts/runs/` (not delivered) and would have shown the baseline's training curves next to the BN model; `train_model.py --deploy` copied a file without a threshold or history | Deployment logic duplicated in three places | One `deploy_checkpoint()` used by `run_experiments.py`, `train_model.py --deploy` and the new `deploy_model.py` (threshold, backup, matching history, fresh report); BN checkpoint shipped in `artifacts/models/experimental/`; covered by the e2e test and a real-data dry run |

## Training and evaluation status

* Dataset: 1,800 images verified (6 × 300, all 200 × 200); split 1,260 / 270 / 270, no leakage (`docs/DATASET.md`).
* Trained: 20 runs (augmentation × 3, input size × 3, BatchNorm × 3, 5 final seeds; 3 seeds per selection cell).
* Deployed: `steelcnn-64px-flip-s1-09c58bf9ba`, the blueprint baseline (no BatchNorm), 64 × 64 input, flip augmentation, chosen by validation loss.

### Real test-set metrics (270 images, never used for any choice)

| | Accuracy | Macro F1 |
|---|---|---|
| Deployed model | **95.19%** | **0.952** |
| Same configuration, 5 seeds | 94.52% ± 0.80 | 0.945 ± 0.008 |
| INT8 of the deployed model | 94.07% | 0.941 |
| BatchNorm variant, 3 seeds (informational) | 97.78% ± 0.37 | 0.978 ± 0.004 |

Per-class (deployed): crazing 1.000 / 1.000, inclusion 0.860 / 0.956, patches 1.000 / 1.000, pitted surface 0.951 / 0.867, rolled-in scale 0.957 / 0.978, scratches 0.953 / 0.911 (precision / recall). Confusion matrix, curves and examples: `artifacts/reports/final/`. Full analysis: `docs/EXPERIMENTS.md`.

### Model files

| File | Size | Notes |
|---|---|---|
| `artifacts/models/steel_cnn_fp32.pt` | 127,621 bytes | weights + class mapping + preprocessing + training settings + validation-calibrated review threshold (0.9007) |
| `artifacts/models/steel_cnn_int8.pt` | 44,291 bytes | optional; not used by the app |
| `artifacts/models/experimental/steel_cnn_bn_fp32.pt` | 132,693 bytes | BatchNorm variant (seed 2), not deployed; install with `scripts/deploy_model.py` |
| `artifacts/models/threshold_calibration.json`, `training_history.json` | small | calibration curve, epoch history |

Latency (CPU, batch 1, median of 200): 0.47 ms forward pass, 1.27 ms decode + preprocess + forward.

## Not tested / blocked

* **Real ECE hardware**: no camera or controller was available. Blocked; the simulator and drop-folder adapter were tested instead, and the API was tested over real HTTP on localhost.
* **Windows / macOS installation**: only Linux was available.
* **CPU-only PyTorch index** (`download.pytorch.org`): not reachable from the build environment; installation was verified with the PyPI wheel.
* **GPU training path**: no GPU; device selection falls back to CPU (the CUDA branch is not exercised).
* **Multi-user load on the app/API**: not tested beyond sequential requests.
* **Accuracy on images from another camera or line**: impossible without labelled data from that setup.

## Known issues and limitations

* Inclusion is the main confusion partner (inclusion ↔ pitted surface and inclusion ↔ scratches are the top two confusions in every seed); faint thin scratches tend to be called inclusion.
* Accuracy on images whose annotations contain a second defect type is lower (85% on 20 test images).
* Brightness and sharpness differ by class in NEU, so exposure changes on new hardware can shift predictions.
* Softmax scores are not calibrated probabilities; 2 of 13 test errors had scores above the review threshold.
* If the BatchNorm model is deployed, the default 99% calibration target yields a low threshold (0.573) that flags few images; raise the target (see `docs/EXPERIMENTS.md`).
* Two flip+rotation runs hit the 100-epoch cap, so that augmentation may be slightly under-trained in the comparison.
* Experiment timings for 100/128 px are inflated by concurrent test runs on the same CPU.
* The API has no authentication.
* Not production-ready.

## Steps for ECE hardware integration

1. Answer the questions in `docs/HARDWARE_INTERFACE.md` §7 (camera, host, field of view, lighting, trigger, outputs).
2. Choose a transport: drop folder (`hardware.adapter: folder`) or HTTP (`POST /v1/inspect`); serial needs a new adapter subclass.
3. Match the image requirements (square crop near 200 × 200 px at NEU-like magnification, grayscale, stable diffuse lighting).
4. Run `python scripts/hardware_loop.py --adapter folder` (or point the device at the API) and check that results and records arrive.
5. Collect and label a set of images from the real setup and evaluate on them before trusting any accuracy figure from NEU.
