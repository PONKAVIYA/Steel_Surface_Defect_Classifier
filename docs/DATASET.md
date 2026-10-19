# Dataset

## Source and layout

The project uses the **NEU surface defect database** (Northeastern University, K. Song and Y. Yan): 1,800 grayscale patches of hot-rolled steel strip, 300 per class, 200 × 200 px. The copy supplied for this project is the *NEU-DET* packaging:

```
data/NEU-DET/
  train_images/<class>_<n>.jpg        1,700 images
  train_annotations/<class>_<n>.xml   bounding boxes (Pascal VOC)
  valid_images/<class>_<n>.jpg          100 images
  valid_annotations/<class>_<n>.xml
```

The original train/valid split of this package is **not** used; all 1,800 images are pooled and re-split (below). The loader also accepts class folders (`<root>/<class>/*.png`) and the NEU-CLS naming (`Cr_1.bmp`, `In_1.bmp`, `Pa_1.bmp`, `PS_1.bmp`, `RS_1.bmp`, `Sc_1.bmp`).

Class index mapping (fixed; stored in every checkpoint and checked at load time):

| Index | Class | Display name |
|---|---|---|
| 0 | crazing | Crazing |
| 1 | inclusion | Inclusion |
| 2 | patches | Patches |
| 3 | pitted_surface | Pitted surface |
| 4 | rolled-in_scale | Rolled-in scale |
| 5 | scratches | Scratches |

## Verified facts (from `artifacts/splits/dataset_report.json`)

* 1,800 readable images, exactly 300 per class. No unreadable files, no unlabeled files, no missing classes.
* All images are 200 × 200 px, stored as **3-channel JPEG whose channels are identical** (maximum channel difference 0.0), i.e. grayscale content. They are converted to one channel.
* JPEG, not the original BMP of NEU-CLS: the pixels carry JPEG compression artefacts.
* **Multi-defect annotations:** 123 images have bounding boxes of a second defect class in their XML (scratches 50, pitted surface 39, patches 29, crazing 5). Every annotation file contains the class named in its file name. The image-level label is the file-name label (the NEU-CLS convention). These images are kept and flagged `multi_defect`; evaluation reports accuracy separately for them (85 train / 18 val / 20 test).

## Duplicate and leakage checks

The prompt asked to keep images from the same specimen or sequence in one split. NEU has no specimen metadata, so similarity is measured instead:

* **Exact duplicates:** identical file bytes (MD5). Found 1 pair: `patches_101.jpg` = `patches_105.jpg`.
* **Near duplicates:** each image is reduced to 64 × 64, its low-frequency illumination is removed (subtract a Gaussian blur, σ = 3 px), and the zero-mean, unit-norm result is compared by normalised cross-correlation, taking the maximum over all 8 flips/rotations (training uses flips, so a mirrored copy would leak).

Why not a perceptual hash: an 8 × 8 dHash flagged 11,310 "duplicate" pairs, most of them across classes, because most NEU images share a similar lighting gradient. Plain thumbnail correlation had the same problem. Removing illumination first fixed it.

Threshold 0.70 was chosen by inspecting pairs: above 0.8 the pairs were the byte copy, a **90°-rotated copy of the same scratch specimen** (`scratches_232` in the original train set, `scratches_247` in the original valid set), a brightness-shifted inclusion copy, and consecutive frames of the same scratch. One pair at 0.84 (`scratches_180` / `scratches_65`) was a false match. The threshold is deliberately conservative: grouping a false match only constrains the split slightly, while missing a real duplicate leaks test information.

Result: 43 pairs (1 exact), **0 cross-class**, 19 groups (largest 9 images, almost all scratches). Union-find puts transitively connected images in one group, and each group is assigned to a single split. Three of the pairs crossed the original NEU-DET train/valid boundary, so that original split leaked.

## Split

* Stratified per class, group-aware, seed 42, ratios 70 / 15 / 15.
* Result: **1,260 train / 270 validation / 270 test**, exactly 210 / 45 / 45 per class.
* `verify_split()` checks that no file, MD5 or duplicate group appears in two splits, and runs every time a manifest is loaded.
* Saved to `artifacts/splits/split_manifest.csv` (+ `split_meta.json`). Regenerating with the same seed reproduces it exactly (tested).

The test split is used only for final reporting: augmentation, input size, checkpoint and threshold choices use the validation split.

## Per-class image statistics (median over all images)

| Class | Brightness (0-255) | Contrast (std) | Sharpness (var. of Laplacian) |
|---|---|---|---|
| Crazing | 149 | 28.5 | 2085 |
| Inclusion | 99 | 11.2 | 31 |
| Patches | 139 | 57.1 | 1748 |
| Pitted surface | 186 | 26.4 | 181 |
| Rolled-in scale | 126 | 16.0 | 656 |
| Scratches | 93 | 20.3 | 80 |

Brightness and sharpness differ systematically between classes. A CNN can partly learn these acquisition-related cues instead of defect shape, so a camera with different exposure or focus may shift predictions even for the same defect. This is one reason accuracy on NEU does not automatically transfer to the ECE team's setup.

## Image-quality reference

`artifacts/splits/quality_reference.json` stores the 0.5th / 99.5th percentiles of these statistics on the **training split**. The inspection page warns when an image falls outside them (mean brightness 37.8-240.5, contrast std ≥ 7.6, variance of Laplacian ≥ 11.0, saturated pixels ≤ 39.6%). Some NEU classes are naturally dark or soft, which is why the bounds come from the data rather than from generic photography rules.

## Licence / use

NEU is distributed for research and education. The 12 images in `samples/` are copied from the test split for the demo; do not redistribute the full dataset with this repository.
