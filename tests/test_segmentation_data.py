import json
from argparse import Namespace

import nibabel as nib
import numpy as np

from scripts import segmentation_data as sd


def write_nifti(path, data, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), str(path))


def test_nii_stem_handles_supported_extensions(tmp_path):
    assert sd.nii_stem(tmp_path / "patient.nii.gz") == "patient"
    assert sd.nii_stem(tmp_path / "patient.nii") == "patient"


def test_files_by_stem_detects_duplicate_nii_variants(tmp_path):
    (tmp_path / "case.nii").touch()
    (tmp_path / "case.nii.gz").touch()
    try:
        sd.files_by_stem(tmp_path)
    except ValueError as exc:
        assert "Duplicate basename" in str(exc)
    else:
        raise AssertionError("duplicate stems should be rejected")


def test_inspect_pair_accepts_valid_aligned_binary_mask(tmp_path):
    image = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    label = np.zeros_like(image, dtype=np.uint8)
    label[1:3, 1:4, 2:5] = 1
    image_path, label_path = tmp_path / "im.nii.gz", tmp_path / "mask.nii.gz"
    write_nifti(image_path, image)
    write_nifti(label_path, label)
    row = sd.inspect_pair("case", image_path, label_path, 1e-3)
    assert row["status"] == "ok"
    assert row["foreground_voxels"] == 18
    assert row["image_finite_fraction"] == 1.0


def test_inspect_pair_reports_affine_and_label_issues(tmp_path):
    image = np.ones((3, 3, 3), dtype=np.float32)
    label = np.full((3, 3, 3), 2, dtype=np.uint8)
    affine = np.eye(4)
    affine[0, 3] = 4
    image_path, label_path = tmp_path / "im.nii.gz", tmp_path / "mask.nii.gz"
    write_nifti(image_path, image)
    write_nifti(label_path, label, affine)
    row = sd.inspect_pair("case", image_path, label_path, 1e-3)
    assert row["status"] == "error"
    assert "affines differ" in row["issues"]
    assert "expected binary labels" in row["issues"]


def test_audit_records_unpaired_inputs_and_returns_failure(tmp_path):
    images, labels, out = tmp_path / "images", tmp_path / "labels", tmp_path / "audit"
    write_nifti(images / "only-image.nii.gz", np.ones((2, 2, 2), dtype=np.float32))
    write_nifti(labels / "only-mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
    code = sd.audit(Namespace(images=images, labels=labels, out=out, affine_tol=1e-3))
    summary = json.loads((out / "summary.json").read_text())
    assert code == 1
    assert summary["n_paired"] == 0
    assert summary["n_error"] == 2


def test_prepare_generates_disjoint_patient_level_folds(tmp_path):
    images, labels, out = tmp_path / "images", tmp_path / "labels", tmp_path / "prepared"
    for i in range(6):
        image = np.full((3, 3, 3), i, dtype=np.float32)
        label = np.zeros((3, 3, 3), dtype=np.uint8)
        label[1, 1, 1] = 1
        write_nifti(images / f"patient-{i}.nii.gz", image)
        write_nifti(labels / f"patient-{i}.nii.gz", label)
    args = Namespace(images=images, labels=labels, out=out,
                     dataset_name="Dataset008_PancreasIDC", folds=3,
                     seed=17, affine_tol=1e-3, link=False)
    assert sd.prepare(args) == 0
    raw = out / "nnUNet_raw" / args.dataset_name
    folds = json.loads((raw / "splits_final.json").read_text())
    for fold in folds:
        assert set(fold["train"]).isdisjoint(fold["val"])
        assert len(fold["train"]) + len(fold["val"]) == 6
    assert len(list((raw / "imagesTr").glob("*_0000.nii.gz"))) == 6


def test_safe_case_id_rejects_empty_normalized_name():
    try:
        sd.safe_case_id("---")
    except ValueError as exc:
        assert "safe nnU-Net case ID" in str(exc)
    else:
        raise AssertionError("empty normalized ID should be rejected")
