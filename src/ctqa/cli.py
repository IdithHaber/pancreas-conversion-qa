from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def read_dicom_series(folder: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise SystemExit(f"No readable DICOM series found under {folder}")
    if len(series_ids) != 1:
        choices = "\n".join(f"  {i}: {sid}" for i, sid in enumerate(series_ids))
        raise SystemExit(
            f"Found {len(series_ids)} series. This starter expects one series per folder.\n{choices}\n"
            "Select one series into its own folder, then run again."
        )
    files = reader.GetGDCMSeriesFileNames(str(folder), series_ids[0])
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def run_conversion(dicom_dir: Path, output_dir: Path) -> Path:
    converter = shutil.which("dcm2niix")
    if not converter:
        raise SystemExit("dcm2niix is not installed or not on PATH. See README.md.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [converter, "-z", "y", "-b", "y", "-ba", "y", "-f", "converted", "-o", str(output_dir), str(dicom_dir)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    nifti = output_dir / "converted.nii.gz"
    if not nifti.exists():
        raise SystemExit(f"Expected output was not created: {nifti}; inspect dcm2niix output above.")
    return nifti


def as_sitk(image: sitk.Image, pixel_id=sitk.sitkFloat32) -> sitk.Image:
    return sitk.Cast(image, pixel_id)


def validate(dicom_dir: Path, nifti_path: Path, report_path: Path) -> dict:
    source = as_sitk(read_dicom_series(dicom_dir))
    converted = as_sitk(sitk.ReadImage(str(nifti_path)))

    # Compare in physical space, not raw array order: NIfTI and DICOM readers may
    # represent the same volume with different axis order or signs.
    aligned = sitk.Resample(
        converted, source, sitk.Transform(), sitk.sitkLinear, 0.0,
        sitk.sitkFloat32,
    )
    src = sitk.GetArrayViewFromImage(source).astype(np.float32, copy=False)
    dst = sitk.GetArrayViewFromImage(aligned).astype(np.float32, copy=False)
    finite = np.isfinite(src) & np.isfinite(dst)
    delta = np.abs(src[finite] - dst[finite])
    same_grid = (
        source.GetSize() == converted.GetSize()
        and np.allclose(source.GetSpacing(), converted.GetSpacing(), atol=1e-3, rtol=1e-3)
        and np.allclose(source.GetOrigin(), converted.GetOrigin(), atol=1e-2, rtol=0)
        and np.allclose(source.GetDirection(), converted.GetDirection(), atol=1e-4, rtol=0)
    )
    # Nearest-neighbor-independent interpolation means exact equality is not
    # expected at every voxel. Report distributions; flag gross geometry/value issues.
    report = {
        "dicom_dir": str(dicom_dir.resolve()),
        "nifti_file": str(nifti_path.resolve()),
        "source_size_xyz": list(source.GetSize()),
        "nifti_size_xyz": list(converted.GetSize()),
        "source_spacing_mm": list(source.GetSpacing()),
        "nifti_spacing_mm": list(converted.GetSpacing()),
        "source_origin": list(source.GetOrigin()),
        "nifti_origin": list(converted.GetOrigin()),
        "source_direction": list(source.GetDirection()),
        "nifti_direction": list(converted.GetDirection()),
        "same_grid_within_tolerance": bool(same_grid),
        "overlap_voxel_fraction": float(finite.mean()),
        "source_intensity_min_max": [float(np.nanmin(src)), float(np.nanmax(src))],
        "nifti_intensity_min_max": [float(np.nanmin(sitk.GetArrayViewFromImage(converted))), float(np.nanmax(sitk.GetArrayViewFromImage(converted)))],
        "abs_difference_on_source_grid": {
            "median": float(np.median(delta)) if delta.size else None,
            "p95": float(np.percentile(delta, 95)) if delta.size else None,
            "max": float(np.max(delta)) if delta.size else None,
        },
        "notes": [
            "Intensity comparison is after physical-space resampling; inspect the overlay for interpolation effects.",
            "This is a conversion QA aid, not clinical validation or proof that metadata are correct.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not same_grid or report["overlap_voxel_fraction"] < 0.95:
        raise SystemExit("QA failed: geometry differs or physical overlap is below 95%; inspect the JSON report.")
    return report


def idc_candidates(limit: int):
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise SystemExit("Install the IDC extra first: pip install -e '.[idc]'") from exc
    client = IDCClient.client()
    query = f"""
        SELECT collection_id, PatientID, SeriesInstanceUID, SeriesDescription,
               series_size_MB
        FROM index
        WHERE collection_id = 'pancreas_ct' AND Modality = 'CT'
        ORDER BY series_size_MB ASC
        LIMIT {int(limit)}
    """
    return client, client.sql_query(query)


def download_idc_series(series_uid: str, output_dir: Path) -> Path:
    try:
        from idc_index import IDCClient
    except ImportError as exc:
        raise SystemExit("Install the IDC extra first: pip install -e '.[idc]'") from exc
    client = IDCClient.client()
    output_dir.mkdir(parents=True, exist_ok=True)
    client.download_dicom_series(
        seriesInstanceUID=series_uid,
        downloadDir=str(output_dir),
        dirTemplate="%SeriesInstanceUID",
        quiet=False,
        show_progress_bar=True,
    )
    series_dir = output_dir / series_uid
    if not series_dir.is_dir():
        raise SystemExit(f"Expected downloaded series folder was not found: {series_dir}")
    return series_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one DICOM CT series and compare it with an independent DICOM read.")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Convert a single-series DICOM folder with dcm2niix.")
    convert.add_argument("dicom_dir", type=Path)
    convert.add_argument("--out", type=Path, default=Path("outputs"))
    check = sub.add_parser("check", help="Compare converted NIfTI geometry and intensities to source DICOM.")
    check.add_argument("dicom_dir", type=Path)
    check.add_argument("nifti", type=Path)
    check.add_argument("--report", type=Path, default=Path("outputs/qa-report.json"))
    listing = sub.add_parser("idc-list", help="Show small CT series in IDC Pancreas-CT.")
    listing.add_argument("--limit", type=int, default=8)
    fetch = sub.add_parser("idc-download", help="Download one IDC Pancreas-CT series by SeriesInstanceUID.")
    fetch.add_argument("series_uid")
    fetch.add_argument("--out", type=Path, default=Path("data/idc"))
    both = sub.add_parser("run", help="Convert, then check.")
    both.add_argument("dicom_dir", type=Path)
    both.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.command == "idc-list":
        _, rows = idc_candidates(args.limit)
        print(rows.to_string(index=False))
    elif args.command == "idc-download":
        print(download_idc_series(args.series_uid, args.out))
    elif args.command == "convert":
        print(run_conversion(args.dicom_dir, args.out))
    elif args.command == "check":
        validate(args.dicom_dir, args.nifti, args.report)
    else:
        nifti = run_conversion(args.dicom_dir, args.out)
        validate(args.dicom_dir, nifti, args.out / "qa-report.json")


if __name__ == "__main__":
    main()
