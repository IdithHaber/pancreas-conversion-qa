#!/usr/bin/env python3
"""Audit paired CT/mask NIfTI data and prepare nnU-Net folds."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def nii_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def files_by_stem(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            stem = nii_stem(p)
            if stem in out:
                raise ValueError(f"Duplicate basename {stem!r} in {folder}")
            out[stem] = p
    return out


def inspect_pair(case_id: str, image_path: Path, label_path: Path, affine_tol: float) -> dict:
    row: dict = {"case_id": case_id, "image": str(image_path), "label": str(label_path), "status": "ok", "issues": ""}
    issues: list[str] = []
    try:
        im = nib.load(str(image_path))
        seg = nib.load(str(label_path))
        row["image_shape"] = "x".join(map(str, im.shape[:3]))
        row["label_shape"] = "x".join(map(str, seg.shape[:3]))
        row["spacing_mm"] = ",".join(f"{v:.5g}" for v in im.header.get_zooms()[:3])
        if len(im.shape) != 3:
            issues.append(f"image is {len(im.shape)}D; expected 3D")
        if len(seg.shape) != 3:
            issues.append(f"label is {len(seg.shape)}D; expected 3D")
        if im.shape[:3] != seg.shape[:3]:
            issues.append("image and label voxel dimensions differ")
        affine_delta = float(np.max(np.abs(im.affine - seg.affine)))
        row["max_affine_abs_diff"] = affine_delta
        if affine_delta > affine_tol:
            issues.append(f"image and label affines differ (max abs diff {affine_delta:.6g})")
        if len(im.shape) == 3:
            image = np.asanyarray(im.dataobj)
            row["image_finite_fraction"] = float(np.isfinite(image).mean())
            if not np.isfinite(image).all():
                issues.append("image contains NaN or infinite voxels")
            finite = image[np.isfinite(image)]
            if finite.size:
                row["image_min"] = float(finite.min())
                row["image_max"] = float(finite.max())
                row["image_p01"] = float(np.percentile(finite, 1))
                row["image_p99"] = float(np.percentile(finite, 99))
        if len(seg.shape) == 3:
            labels = np.asanyarray(seg.dataobj)
            if not np.isfinite(labels).all():
                issues.append("mask contains NaN or infinite voxels")
            else:
                unique, counts = np.unique(labels, return_counts=True)
                row["label_values"] = ",".join(str(x.item()) for x in unique[:100])
                row["label_voxel_counts"] = json.dumps({str(x.item()): int(n) for x, n in zip(unique[:100], counts[:100])})
                if not np.allclose(labels, np.rint(labels), atol=1e-4):
                    issues.append("mask contains non-integer label values")
                if not np.isin(np.rint(labels), [0, 1]).all():
                    issues.append("expected binary labels {0,1}; inspect/remap labels explicitly")
                foreground = int(np.count_nonzero(labels))
                row["foreground_voxels"] = foreground
                if foreground == 0:
                    issues.append("mask is empty")
        row["issues"] = "; ".join(issues)
        row["status"] = "error" if issues else "ok"
    except Exception as exc:  # Report corrupt/unreadable inputs, then continue the audit.
        row["status"] = "error"
        row["issues"] = f"unreadable input: {type(exc).__name__}: {exc}"
    return row


def audit(args: argparse.Namespace) -> int:
    images = files_by_stem(args.images)
    labels = files_by_stem(args.labels)
    image_ids, label_ids = set(images), set(labels)
    rows = [inspect_pair(k, images[k], labels[k], args.affine_tol) for k in sorted(image_ids & label_ids)]
    for case_id in sorted(image_ids - label_ids):
        rows.append({"case_id": case_id, "image": str(images[case_id]), "label": "", "status": "error", "issues": "no matching mask"})
    for case_id in sorted(label_ids - image_ids):
        rows.append({"case_id": case_id, "image": "", "label": str(labels[case_id]), "status": "error", "issues": "no matching image"})
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("case_id") if rows else pd.DataFrame(columns=["case_id", "status", "issues"])
    df.to_csv(out / "cases.csv", index=False)
    summary = {
        "image_directory": str(args.images.resolve()), "label_directory": str(args.labels.resolve()),
        "n_images": len(images), "n_labels": len(labels), "n_paired": len(image_ids & label_ids),
        "n_ok": int((df.status == "ok").sum()) if len(df) else 0,
        "n_error": int((df.status == "error").sum()) if len(df) else 0,
        "affine_tolerance": args.affine_tol,
        "blocking_issues": df.loc[df.status == "error", ["case_id", "issues"]].to_dict("records") if len(df) else [],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["n_error"] else 0


def safe_case_id(case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", case_id).strip("_")
    if not cleaned:
        raise ValueError(f"Cannot convert basename {case_id!r} to a safe nnU-Net case ID")
    return cleaned


def prepare(args: argparse.Namespace) -> int:
    images, labels = files_by_stem(args.images), files_by_stem(args.labels)
    if set(images) != set(labels):
        missing_masks = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        raise SystemExit(f"Image/mask basenames differ. Missing masks={missing_masks}; missing images={missing_images}. Run audit first.")
    ids = sorted(images)
    if len(ids) < args.folds:
        raise SystemExit(f"Need at least {args.folds} paired patients; found {len(ids)}")
    sanitized = [safe_case_id(k) for k in ids]
    if len(set(sanitized)) != len(sanitized):
        raise SystemExit("Basenames collide after nnU-Net-safe character normalization")
    for k in ids:
        row = inspect_pair(k, images[k], labels[k], args.affine_tol)
        if row["status"] != "ok":
            raise SystemExit(f"Cannot prepare data: {k}: {row['issues']}")

    dataset_name = args.dataset_name
    if not re.fullmatch(r"Dataset\d{3,}_\w+", dataset_name):
        raise SystemExit("Dataset name must look like Dataset008_PancreasIDC")
    raw = args.out / "nnUNet_raw" / dataset_name
    image_dir, label_dir = raw / "imagesTr", raw / "labelsTr"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for case_id, safe_id in zip(ids, sanitized):
        for src, dst in ((images[case_id], image_dir / f"{safe_id}_0000.nii.gz"), (labels[case_id], label_dir / f"{safe_id}.nii.gz")):
            if dst.exists():
                dst.unlink()
            if args.link:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

    dataset = {
        "channel_names": {"0": "CT"}, "labels": {"background": 0, "pancreas": 1},
        "numTraining": len(ids), "file_ending": ".nii.gz", "name": dataset_name,
        "description": "IDC Pancreas-CT binary organ segmentation; patient-level CV",
        "reference": "TCIA Pancreas-CT; cite the collection and annotation source",
        "licence": "See source dataset terms", "release": "1.0",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    folds = []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for train_idx, val_idx in splitter.split(sanitized):
        folds.append({"train": [sanitized[i] for i in train_idx], "val": [sanitized[i] for i in val_idx]})
    (raw / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n")
    manifest = pd.DataFrame({"case_id": ids, "nnunet_case_id": sanitized, "image": [str(images[k]) for k in ids], "label": [str(labels[k]) for k in ids]})
    manifest.to_csv(raw / "source_manifest.csv", index=False)
    print(f"Prepared {len(ids)} patient cases, {args.folds} folds: {raw}")
    print(f"Review {raw / 'splits_final.json'}; source NIfTIs were {'symlinked' if args.link else 'copied'}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_audit = sub.add_parser("audit", help="check paired NIfTI images and labels")
    p_audit.add_argument("--images", type=Path, required=True)
    p_audit.add_argument("--labels", type=Path, required=True)
    p_audit.add_argument("--out", type=Path, required=True)
    p_audit.add_argument("--affine-tol", type=float, default=1e-3, help="maximum absolute affine element difference")
    p_audit.set_defaults(func=audit)
    p_prep = sub.add_parser("prepare", help="copy/link validated data and create patient-level nnU-Net folds")
    p_prep.add_argument("--images", type=Path, required=True)
    p_prep.add_argument("--labels", type=Path, required=True)
    p_prep.add_argument("--out", type=Path, required=True)
    p_prep.add_argument("dataset_name")
    p_prep.add_argument("--folds", type=int, default=5)
    p_prep.add_argument("--seed", type=int, default=2026)
    p_prep.add_argument("--affine-tol", type=float, default=1e-3)
    p_prep.add_argument("--link", action="store_true", help="symlink instead of copying the input NIfTIs")
    p_prep.set_defaults(func=prepare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
