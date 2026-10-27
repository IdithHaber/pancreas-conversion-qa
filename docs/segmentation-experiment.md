# Segmentation experiment protocol

## Question

On the 80 labeled IDC Pancreas-CT patients, does a Merlin CT representation improve pancreas-organ segmentation over a standard 3D model trained from scratch?

The target is the manually labeled pancreas organ. This experiment does not train a tumor classifier or tumor segmenter.

## Data checks before training

Run `segmentation_data.py audit` first and retain both output reports with the experiment. Review at least:

- Every image has one same-basename label and vice versa.
- CT and label are readable 3D NIfTI files on the same voxel grid and affine.
- CT has finite values; inspect the reported intensity percentiles for an unexpected rescale or corrupted volume.
- The mask contains only 0 and 1 and has foreground voxels.
- All case IDs are distinct patients. If the source includes multiple scans per patient, group on the patient ID rather than scan ID before making folds.

The script intentionally does not guess at orientation fixes, resampling, label remapping, or CT intensity conversion. Correct an identified issue explicitly, regenerate the files, and rerun the audit.

## Split and comparison

Use the generated deterministic five-fold patient split for both arms. Each patient is a validation case once. Fit any data-dependent choices only within the training partition for that fold. For a publication-quality claim, add an external cohort; five-fold internal validation on a single 80-patient collection cannot establish site generalization.

### Arm A: scratch baseline

Use nnU-Net v2 `3d_fullres` with the default plans and random initialization. Record the package version, plans, fold-wise validation scores, training time, and peak GPU memory. Avoid architecture or preprocessing changes after viewing fold scores unless that change is repeated inside every fold.

### Arm B: Merlin initialization

There are two materially different experiments:

1. **Encoder transfer (preferred for the staged question):** initialize only the image encoder from a Merlin CT pretraining checkpoint; create a fresh binary segmentation decoder/head. Train the head with the encoder frozen, then optionally unfreeze the last encoder stage(s) at a lower learning rate. Record exactly which parameters are frozen in each phase and verify that the head receives gradients in phase one while the encoder does not.
2. **Released Merlin segmentation model:** start from the released Merlin + nnU-Net segmentation checkpoint and fine-tune it. This is useful as a strong pretrained segmentation comparator, but its decoder/head is already trained; it is not the “frozen encoder, new head” experiment. Label it separately in results.

The official Merlin repository directs segmentation users to a custom nnU-Net integration. It is a separate forked training stack, so pin the exact repository commit and environment rather than mixing its nnU-Net fork with another version. Follow that repository's Merlin-specific image transforms and dataset instructions. In particular, its README says not to apply a separate intensity scaling transform when nnU-Net plans already perform CT normalization. Keep all scans and derived data private/local and exclude checkpoints and datasets from Git.

Before claiming an independent test, check whether the 80 IDC cases could overlap with Merlin pretraining or any segmentation fine-tuning corpus. If that cannot be established, say so plainly; a pretrained model may have seen a scan or close derivative before evaluation.

## Metrics and reporting

Report patient-level Dice and one boundary metric (surface Dice with a prespecified millimeter tolerance, or HD95). Include per-fold mean/median and spread, and paired per-patient differences between arms. State how empty predictions are handled. Do not pool all voxels and present them as independent observations. Save out-of-fold predictions so qualitative overlays can be inspected.

Include the input audit summary, fold assignment, package/repository revisions, preprocessing settings, random seeds, model/checkpoint identifiers, and the exact metrics in the repository's experiment record. Do not commit CTs, masks, predictions, or model weights.
