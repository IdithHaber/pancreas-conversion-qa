import numpy as np
import pytest

from ctqa.shape_eda import basic_shape_stats, mesh_curvatures, surface_from_mask


def test_basic_shape_stats_respects_voxel_spacing():
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:3, 2:5, 3:7] = True
    stats = basic_shape_stats(mask, (1.0, 2.0, 3.0))
    assert stats["voxel_count"] == 24
    assert stats["volume_mm3"] == pytest.approx(144.0)
    assert stats["bbox_x_mm"] == pytest.approx(4.0)
    assert stats["bbox_y_mm"] == pytest.approx(6.0)
    assert stats["bbox_z_mm"] == pytest.approx(6.0)


def test_basic_shape_stats_rejects_empty_mask():
    with pytest.raises(ValueError, match="Mask is empty"):
        basic_shape_stats(np.zeros((3, 3, 3), dtype=bool), (1, 1, 1))


def test_surface_and_curvature_are_finite_for_solid_cube():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[3:9, 3:9, 3:9] = True
    vertices, faces = surface_from_mask(mask, (1.0, 1.0, 1.0), 0.5)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    kmax, kmin, mean_h, gaussian_k = mesh_curvatures(vertices, faces)
    assert len(kmax) > 0
    assert np.isfinite(np.concatenate([kmax, kmin, mean_h, gaussian_k])).all()
    assert (kmax >= kmin).all()

