# Pancreas CT conversion QA

A small, reproducible project that converts **one public CT DICOM series** to NIfTI with [`dcm2niix`](https://github.com/rordenlab/dcm2niix), then checks the result against an independent DICOM read using SimpleITK. It reports geometry, physical overlap, and voxel-intensity differences, and writes a JSON report.

The project now includes DICOM-to-NIfTI conversion QA, descriptive EDA of the 3D pancreas masks, and a small-data pancreas-organ segmentation benchmark. The segmentation experiment uses the paired CT and pancreas masks; it does not classify or segment tumors.

## What the check does

- Requires a folder containing exactly one ordinary single-frame CT series.
- Runs `dcm2niix` to produce compressed NIfTI plus a JSON sidecar.
- Independently reads source DICOM with GDCM through SimpleITK.
- Compares image dimensions, voxel spacing, origin, and direction.
- Resamples the NIfTI into the source DICOM physical grid and reports overlap and absolute intensity-difference summaries.
- Writes `outputs/qa-report.json`.

The raw voxel arrays can use different axis order or signs, so this compares in physical space. Differences can still arise from resampling, rounding, modality rescale handling, or converter choices. Review the report and images; the script is not a medical or clinical validation tool.

## Run locally (including a 4080 workstation)

The conversion and QA are CPU tasks; the GPU is not required. In a terminal:

```bash
# Install dcm2niix first. Ubuntu/WSL:
sudo apt-get update && sudo apt-get install -y dcm2niix

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# DICOM_SERIES should contain one downloaded CT series, not a whole multi-series study.
ctqa run /path/to/DICOM_SERIES --out outputs
```

Windows can use WSL2 with the same Ubuntu steps. On macOS, install `dcm2niix` with Homebrew (`brew install dcm2niix`).

## Run from Chrome with Google Colab

1. Open `notebooks/colab_quickstart.ipynb` in Colab (or upload it).
2. In Colab, choose **Runtime → Run all**.
3. Upload a ZIP containing one DICOM CT series when prompted. The notebook installs the small dependencies, converts, and displays the JSON report.

Colab's CPU runtime is enough. A GPU runtime is unnecessary. Do not put protected or identifiable scans in a public notebook or repository.

## Download a concrete public IDC CT sample

The starter uses IDC's `Pancreas-CT` collection (IDC collection ID `pancreas_ct`). IDC describes 80 subjects with contrast-enhanced abdominal CT and pancreas annotations. This is a non-cancer cohort, so it is appropriate for CT conversion and anatomy handling but does not contain pancreatic cancer lesions.

Install the IDC access extra, list small CT series, choose a row, then download only its series UID:

```bash
pip install -e '.[idc]'
ctqa idc-list --limit 8
ctqa idc-download <SeriesInstanceUID> --out data/idc
ctqa run data/idc/<SeriesInstanceUID> --out outputs
```

The download uses the official [`idc-index`](https://github.com/ImagingDataCommons/idc-index) client and does not download the full collection. The collection page warns that a whole-collection download is over 1 TB, so use the one-series flow above. The dataset collection page documents the source and annotation; please include its citation/DOI if you publish results. The IDC [download guide](https://learn.canceridc.dev/data/downloading-data) describes other access methods.

## Repository layout

```text
src/ctqa/cli.py             conversion and QA command line
src/ctqa/shape_eda.py       mask shape and curvature analysis
scripts/segmentation_data.py  input audit, dataset preparation, patient-level folds
docs/segmentation-experiment.md  training protocol and Merlin caveats
notebooks/colab_quickstart.ipynb  Chrome/Colab conversion walkthrough
notebooks/pancreas_shape_eda.ipynb  Kaggle/Colab EDA walkthrough
.github/workflows/          lightweight repository checks
```

## 3D pancreas shape EDA

The EDA is intentionally useful before training. It treats each annotated patient as one observation, measures geometry in millimeters, and produces per-case shape and curvature distributions at several physical smoothing scales.

TCIA lists manual pancreas annotations as a ZIP/NIfTI package of about 949 KB for 80 cases, and the original CT series as DICOM totaling about 9.95 GB. The masks alone are enough for descriptive shape analysis. Download the annotations from the [TCIA Pancreas-CT page](https://www.cancerimagingarchive.net/collection/pancreas-ct/) and extract them under `data/masks/`. Keep downloaded clinical images out of the public repo. If you want CT-to-mask alignment checks too, convert the corresponding CT volumes to NIfTI and put them under `data/images/` with matching case basenames.

Install the project and run:

```bash
python -m pip install -e .
ctqa-shape-eda --masks data/masks --out outputs/shape-eda
# Optional: also supply paired CT NIfTI files with matching basenames
ctqa-shape-eda --masks data/masks --images data/images --out outputs/shape-eda
```

Outputs include per-case volume and PCA spread summaries, mask surface area and sphericity, curvature quantiles at 0.75, 1.5, and 3 mm smoothing, an alignment table with physical-volume change after resampling when images are supplied, and two plots. Curvatures are estimated from smoothed physical-space meshes; interpret them alongside the scale-sensitivity plot because voxel spacing and mask boundary irregularities affect local curvature. PCA spreads are four standard deviations along each principal axis, useful for comparing relative elongation but not literal pancreas lengths. These masks provide one annotation per case, so this dataset alone cannot quantify inter-reader segmentation uncertainty.

### What 80 cases can support

Eighty annotated patients are enough for descriptive EDA, data-quality checks, and a small proof-of-concept segmentation study. They are not enough to make a strong claim of broad clinical generalization. Slices and voxels within one patient are highly correlated, so they do not turn 80 patients into thousands of independent samples. If training is added later, keep every scan from a patient in the same fold; use patient-level cross-validation and report the case-level distribution, not only a pooled score. External validation would be needed for a generalization claim.

## How this connects to pancreas lesions

IDC's Pancreas-CT masks annotate the pancreas organ, not a tumor. This EDA analyzes the manual NIfTI masks; the conversion workflow remains a separate check for CT series. For a lesion-segmentation experiment, MSD Task07 Pancreas is a separate dataset distributed in NIfTI; conversion validation and lesion modeling should remain separate parts of the portfolio.

## Pancreas segmentation benchmark (80-case IDC cohort)

This experiment compares a scratch 3D U-Net baseline with a Merlin-initialized segmentation model using patient-level five-fold cross-validation. The included data audit runs before conversion/training and checks paired files, readability, volume shape and spacing, image-mask physical alignment, finite CT values, mask labels, and empty masks. The audit writes CSV and JSON reports and exits unsuccessfully if it finds a blocking issue. Downloaded scans and annotations stay outside version control.

### 1. Arrange the data

Download the CT volumes and pancreas masks from the IDC/TCIA Pancreas-CT collection. Put one matching image and mask NIfTI per patient in `data/images/` and `data/masks/`, using the same basename, e.g. `case_001.nii.gz`. Keep the original dataset's patient identifier in the basename. Do not use slices or patches as separate cases.

### 2. Install the segmentation tools and audit the inputs

Use a separate Python 3.10 environment for nnU-Net/Merlin so it does not disturb the conversion QA dependencies. On a local Ubuntu/WSL2 workstation with an RTX 4080:

```bash
cd pancreas-conversion-qa
python3.10 -m venv .venv-seg
source .venv-seg/bin/activate
python -m pip install --upgrade pip
python -m pip install nibabel numpy pandas scikit-learn

python scripts/segmentation_data.py audit \
  --images data/images --labels data/masks --out outputs/segmentation-audit
```

Inspect `outputs/segmentation-audit/summary.json` and `cases.csv`. Fix missing pairs, geometry mismatches, unexpected label values, unreadable files, or empty masks before continuing. Small spacing differences may be legitimate; the audit reports them instead of silently resampling. Images and labels must have matching voxel grids and physical affines for this workflow.

### 3. Prepare deterministic patient-level folds

```bash
python scripts/segmentation_data.py prepare \
  --images data/images --labels data/masks \
  --out data/nnunet Dataset008_PancreasIDC --folds 5 --seed 2026
```

This creates an nnU-Net dataset and a five-fold split file. Every patient appears in exactly one validation fold and in the training partition of the other four folds. No patches are split independently. The source labels must be binary pancreas masks (0 background, 1 pancreas); the preparer rejects other labels rather than guessing how to remap them.

### 4. Train the from-scratch reference

```bash
python -m pip install nnunetv2
export nnUNet_raw="$PWD/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/outputs/nnunet_results"
nnUNetv2_plan_and_preprocess -d 8 --verify_dataset_integrity
cp data/nnunet/nnUNet_raw/Dataset008_PancreasIDC/splits_final.json \
  "$nnUNet_preprocessed/Dataset008_PancreasIDC/splits_final.json"
for fold in 0 1 2 3 4; do
  nnUNetv2_train 8 3d_fullres "$fold"
done
```

This is the standard nnU-Net 3D full-resolution baseline, trained from random initialization. Record the nnU-Net version, generated plans, fold scores, and GPU memory in the experiment log. A 4080 should be suitable for a small abdominal 3D task, though nnU-Net's automatically chosen patch size and batch size determine actual memory use.

### 5. Merlin comparison and staged fine-tuning

Merlin's released segmentation route is a custom nnU-Net integration. The official Merlin repository points users to that integration and its released checkpoint. That checkpoint is a Merlin-initialized segmentation model, rather than a bare image encoder plus a freshly initialized segmentation head. It can be fine-tuned on the same folds as an informative transfer baseline, but it does not by itself implement a clean “freeze encoder, train only new head, then unfreeze encoder” experiment. See [the official Merlin repo](https://github.com/StanfordMIMI/Merlin) and its linked [Merlin nnU-Net integration](https://github.com/ashwinkumargb/Merlin-nnUNet) for the current installation and checkpoint instructions.

For the requested two-stage comparison, use the Merlin encoder weights with the same nnU-Net architecture and add a decoder/head for this binary target. In stage 1, freeze the encoder and train only the new decoder/head; in stage 2, unfreeze the last encoder block(s) at a lower learning rate, leaving earlier blocks frozen. Keep the fold split, preprocessing, loss, augmentation, stopping rule, and evaluation identical to the scratch run. Since the upstream released trainer/checkpoint architecture can change, the repository does not silently treat its segmentation-finetuned checkpoint as an encoder-only initialization; the experiment log must identify the exact Merlin code revision and the exact checkpoint used. See `docs/segmentation-experiment.md` for the comparison protocol and interpretation.

### Evaluation and limits

Report per-patient Dice and surface Dice (or a stated boundary tolerance), plus median, interquartile range, and all five fold results. Select checkpoints using only the training/validation partition for that fold; do not tune on the held-out fold. Show paired per-case differences between scratch and Merlin models. The 80 scans support a reproducible proof-of-concept comparison, not a reliable estimate of hospital-to-hospital performance. This cohort supports pancreas-organ segmentation only; tumor classification or tumor segmentation requires lesion labels from a separate dataset such as MSD Task07.

## Outputs and interpretation

`outputs/converted.nii.gz`, `outputs/converted.json`, and `outputs/qa-report.json` are generated locally and excluded from version control. The report records source and output geometry plus robust intensity-difference summaries. If geometry checks fail, stop and inspect the selected series, converter logs, and report before using the NIfTI downstream.

## Data and attribution

The project code is MIT licensed. The TCIA Pancreas-CT collection and annotation files are CC BY 3.0; credit the source collection and its DOI when sharing results. Data files are not included in this repo. For a later lesion experiment using MSD Task07 Pancreas, cite the Medical Segmentation Decathlon paper and follow that dataset’s terms separately.
