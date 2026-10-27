import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from ctqa import cli


def test_read_dicom_series_rejects_no_series(monkeypatch, tmp_path):
    class Reader:
        def GetGDCMSeriesIDs(self, _):
            return None

    monkeypatch.setattr(cli.sitk, "ImageSeriesReader", Reader)
    with pytest.raises(SystemExit, match="No readable DICOM"):
        cli.read_dicom_series(tmp_path)


def test_read_dicom_series_rejects_ambiguous_folder(monkeypatch, tmp_path):
    class Reader:
        def GetGDCMSeriesIDs(self, _):
            return ["one", "two"]

    monkeypatch.setattr(cli.sitk, "ImageSeriesReader", Reader)
    with pytest.raises(SystemExit, match="Found 2 series"):
        cli.read_dicom_series(tmp_path)


def test_run_conversion_calls_converter_and_checks_output(monkeypatch, tmp_path):
    source = tmp_path / "dicoms"
    source.mkdir()
    output = tmp_path / "out"
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/dcm2niix")

    def run(command, check):
        assert check is True
        assert command[0] == "/usr/bin/dcm2niix"
        (output / "converted.nii.gz").touch()

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.run_conversion(source, output) == output / "converted.nii.gz"


def test_run_conversion_reports_missing_converter(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="dcm2niix is not installed"):
        cli.run_conversion(tmp_path, tmp_path / "out")


def test_validate_writes_report_for_matching_geometry(monkeypatch, tmp_path):
    source = sitk.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    source.SetSpacing((1.0, 1.0, 2.0))
    converted = sitk.Image(source)
    monkeypatch.setattr(cli, "read_dicom_series", lambda _: source)
    monkeypatch.setattr(cli.sitk, "ReadImage", lambda _: converted)
    monkeypatch.setattr(cli.sitk, "Resample", lambda image, reference, *args: image)
    report_path = tmp_path / "nested" / "report.json"

    report = cli.validate(tmp_path / "dicoms", tmp_path / "scan.nii.gz", report_path)

    assert report["same_grid_within_tolerance"] is True
    assert report["overlap_voxel_fraction"] == 1.0
    assert json.loads(report_path.read_text())["abs_difference_on_source_grid"]["max"] == 0.0


def test_validate_fails_on_geometry_mismatch_but_keeps_report(monkeypatch, tmp_path):
    source = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
    converted = sitk.Image(source)
    converted.SetOrigin((5.0, 0.0, 0.0))
    monkeypatch.setattr(cli, "read_dicom_series", lambda _: source)
    monkeypatch.setattr(cli.sitk, "ReadImage", lambda _: converted)
    monkeypatch.setattr(cli.sitk, "Resample", lambda image, reference, *args: reference)
    report_path = tmp_path / "report.json"

    with pytest.raises(SystemExit, match="QA failed"):
        cli.validate(tmp_path / "dicoms", tmp_path / "scan.nii.gz", report_path)
    assert json.loads(report_path.read_text())["same_grid_within_tolerance"] is False


def test_idc_download_creates_output_and_checks_series_folder(monkeypatch, tmp_path):
    uid = "1.2.3"

    class Client:
        @classmethod
        def client(cls):
            return cls()

        def download_dicom_series(self, **kwargs):
            Path(kwargs["downloadDir"], uid).mkdir()

    monkeypatch.setitem(__import__("sys").modules, "idc_index", SimpleNamespace(IDCClient=Client))
    result = cli.download_idc_series(uid, tmp_path / "data" / "idc")
    assert result.is_dir()
    assert result.name == uid

