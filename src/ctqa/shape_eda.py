"""Descriptive 3D shape and curvature EDA for pancreas NIfTI masks."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib import cm, colors


def nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def find_niftis(folder: Path) -> dict[str, Path]:
    paths = sorted([*folder.rglob("*.nii.gz"), *folder.rglob("*.nii")])
    found = {}
    for path in paths:
        key = nifti_stem(path)
        if key in found:
            raise ValueError(f"Duplicate NIfTI basename {key!r}: {found[key]} and {path}")
        found[key] = path
    return found


def mesh_curvatures(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate vertex Gaussian/mean and principal curvature magnitudes on a triangle mesh."""
    tri = vertices[faces]
    e01 = tri[:, 1] - tri[:, 0]
    e02 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e01, e02)
    twice_area = np.linalg.norm(cross, axis=1)
    face_area = 0.5 * twice_area
    safe_twice = np.maximum(twice_area, 1e-12)
    vertex_area = np.zeros(len(vertices), dtype=np.float64)
    angle_sum = np.zeros(len(vertices), dtype=np.float64)

    def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        return np.arccos(np.clip(np.einsum("ij,ij->i", a, b) / denom, -1.0, 1.0))

    a0 = angle(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a1 = angle(tri[:, 0] - tri[:, 1], tri[:, 2] - tri[:, 1])
    a2 = np.pi - a0 - a1
    for corner, ang in enumerate((a0, a1, a2)):
        np.add.at(vertex_area, faces[:, corner], face_area / 3.0)
        np.add.at(angle_sum, faces[:, corner], ang)

    # Cotangent Laplacian gives the mean-curvature normal. Its sign depends on
    # face orientation; the output signature below sorts absolute magnitudes.
    accum = np.zeros_like(vertices, dtype=np.float64)
    cot0 = np.einsum("ij,ij->i", tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / safe_twice
    cot1 = np.einsum("ij,ij->i", tri[:, 0] - tri[:, 1], tri[:, 2] - tri[:, 1]) / safe_twice
    cot2 = np.einsum("ij,ij->i", tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 2]) / safe_twice
    for opposite, cot in enumerate((cot0, cot1, cot2)):
        j = faces[:, (opposite + 1) % 3]
        k = faces[:, (opposite + 2) % 3]
        np.add.at(accum, j, cot[:, None] * (vertices[k] - vertices[j]))
        np.add.at(accum, k, cot[:, None] * (vertices[j] - vertices[k]))

    valid = vertex_area > 1e-10
    normals = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(normals, faces.reshape(-1), np.repeat(cross, 3, axis=0))
    n_norm = np.linalg.norm(normals, axis=1)
    normals[n_norm > 1e-12] /= n_norm[n_norm > 1e-12, None]
    laplace = np.zeros_like(vertices, dtype=np.float64)
    laplace[valid] = accum[valid] / (2.0 * vertex_area[valid, None])
    mean_h = 0.5 * np.einsum("ij,ij->i", laplace, normals)
    gaussian_k = np.zeros(len(vertices), dtype=np.float64)
    gaussian_k[valid] = (2.0 * np.pi - angle_sum[valid]) / vertex_area[valid]
    discriminant = np.maximum(mean_h * mean_h - gaussian_k, 0.0)
    root = np.sqrt(discriminant)
    principal_a, principal_b = mean_h + root, mean_h - root
    kmax = np.maximum(np.abs(principal_a), np.abs(principal_b))
    kmin = np.minimum(np.abs(principal_a), np.abs(principal_b))
    valid &= np.isfinite(kmax) & np.isfinite(kmin)
    return kmax[valid], kmin[valid], mean_h[valid], gaussian_k[valid]


def surface_from_mask(mask_zyx: np.ndarray, spacing_zyx: tuple[float, float, float], sigma_mm: float):
    sigma_vox = np.asarray([sigma_mm / s for s in spacing_zyx], dtype=float)
    pad = tuple(int(np.ceil(4 * x)) + 2 for x in sigma_vox)
    padded = np.pad(mask_zyx.astype(np.float32), tuple((p, p) for p in pad), mode="constant")
    smooth = gaussian_filter(padded, sigma=sigma_vox)
    vertices, faces, _, _ = marching_cubes(smooth, level=0.5, spacing=spacing_zyx, allow_degenerate=False)
    vertices -= np.asarray(pad, dtype=float) * np.asarray(spacing_zyx, dtype=float)
    return vertices.astype(np.float32), faces.astype(np.int32)


def basic_shape_stats(mask: np.ndarray, spacing_xyz: tuple[float, float, float]) -> dict:
    zyx = np.argwhere(mask)
    if not len(zyx):
        raise ValueError("Mask is empty")
    spacing_zyx = np.asarray(spacing_xyz[::-1], dtype=float)
    voxel_volume = float(np.prod(spacing_xyz))
    coords = zyx * spacing_zyx
    # Subsample for a stable, inexpensive estimate of principal-axis proportions.
    if len(coords) > 100_000:
        rng = np.random.default_rng(0)
        coords = coords[rng.choice(len(coords), 100_000, replace=False)]
    eigvals = np.linalg.eigvalsh(np.cov(coords, rowvar=False))[::-1]
    axes = 4.0 * np.sqrt(np.maximum(eigvals, 0.0))
    return {
        "volume_mm3": float(mask.sum() * voxel_volume),
        "voxel_count": int(mask.sum()),
        "bbox_x_mm": float((zyx[:, 2].max() - zyx[:, 2].min() + 1) * spacing_xyz[0]),
        "bbox_y_mm": float((zyx[:, 1].max() - zyx[:, 1].min() + 1) * spacing_xyz[1]),
        "bbox_z_mm": float((zyx[:, 0].max() - zyx[:, 0].min() + 1) * spacing_xyz[2]),
        "pca_spread_axis1_4sd_mm": float(axes[0]),
        "pca_spread_axis2_4sd_mm": float(axes[1]),
        "pca_spread_axis3_4sd_mm": float(axes[2]),
        "pca_elongation_axis1_axis2": float(axes[0] / max(axes[1], 1e-9)),
        "pca_flatness_axis2_axis3": float(axes[1] / max(axes[2], 1e-9)),
        "voxel_spacing_x_mm": float(spacing_xyz[0]),
        "voxel_spacing_y_mm": float(spacing_xyz[1]),
        "voxel_spacing_z_mm": float(spacing_xyz[2]),
    }


def analyze_case(case_id: str, mask_path: Path, ct_path: Path | None, scales_mm: list[float]):
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img) > 0
    if not np.any(mask_arr):
        raise ValueError(f"Empty mask: {mask_path}")
    stats = {"case_id": case_id, "mask_file": str(mask_path), **basic_shape_stats(mask_arr, mask_img.GetSpacing())}
    alignment = {"case_id": case_id, "ct_file": str(ct_path) if ct_path else "", "mask_file": str(mask_path), "geometry_equal": None, "mask_volume_resample_ratio": None}

    if ct_path is not None:
        ct_img = sitk.ReadImage(str(ct_path))
        same = (
            mask_img.GetSize() == ct_img.GetSize()
            and np.allclose(mask_img.GetSpacing(), ct_img.GetSpacing(), atol=1e-3, rtol=1e-3)
            and np.allclose(mask_img.GetOrigin(), ct_img.GetOrigin(), atol=1e-2, rtol=0)
            and np.allclose(mask_img.GetDirection(), ct_img.GetDirection(), atol=1e-4, rtol=0)
        )
        alignment["geometry_equal"] = bool(same)
        if same:
            alignment["mask_volume_resample_ratio"] = 1.0
        else:
            aligned = sitk.Resample(mask_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            before = float(np.count_nonzero(mask_arr) * np.prod(mask_img.GetSpacing()))
            after = float(np.count_nonzero(sitk.GetArrayViewFromImage(aligned)) * np.prod(ct_img.GetSpacing()))
            alignment["mask_volume_resample_ratio"] = after / before if before else 0.0

    scale_rows = []
    mesh_for_plot = None
    for sigma in scales_mm:
        vertices, faces = surface_from_mask(mask_arr, tuple(mask_img.GetSpacing()[::-1]), sigma)
        tri = vertices[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        kmax, kmin, mean_h, gaussian_k = mesh_curvatures(vertices, faces)
        volume_mm3 = stats["volume_mm3"]
        row = {
            "case_id": case_id,
            "volume_mm3": volume_mm3,
            "sphericity": float(np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0) / area),
            "smoothing_sigma_mm": sigma,
            "surface_area_mm2": float(area),
            "kmax_abs_p50_per_mm": float(np.percentile(kmax, 50)),
            "kmax_abs_p90_per_mm": float(np.percentile(kmax, 90)),
            "kmin_abs_p50_per_mm": float(np.percentile(kmin, 50)),
            "kmin_abs_p90_per_mm": float(np.percentile(kmin, 90)),
            "mean_curvature_abs_p50_per_mm": float(np.percentile(np.abs(mean_h), 50)),
            "gaussian_curvature_p50_per_mm2": float(np.percentile(gaussian_k, 50)),
            "surface_vertex_count": int(len(vertices)),
        }
        scale_rows.append(row)
        if sigma == scales_mm[len(scales_mm) // 2]:
            mesh_for_plot = (vertices, faces, kmax)
    return stats, scale_rows, alignment, mesh_for_plot


def plot_representative(case_id: str, mesh_data, output: Path, sigma_mm: float):
    vertices, faces, kmax = mesh_data
    # Limit rendered polygons for a responsive static figure; calculations use the full mesh.
    if len(faces) > 30_000:
        faces_plot = faces[np.linspace(0, len(faces) - 1, 30_000, dtype=int)]
    else:
        faces_plot = faces
    xyz = vertices[:, ::-1]  # marching-cubes coordinates are z,y,x; display x,y,z
    face_values = np.mean(kmax[faces_plot], axis=1)
    norm = colors.Normalize(vmin=float(np.percentile(face_values, 2)), vmax=float(np.percentile(face_values, 98)))
    cmap = cm.get_cmap("viridis")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    poly = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=faces_plot, linewidth=0, antialiased=False, shade=False)
    poly.set_facecolors(cmap(norm(face_values)))
    ax.set_title(f"{case_id}: |κ|max at {sigma_mm:g} mm smoothing")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.65, label="Curvature magnitude (1/mm)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_eda(mask_dir: Path, output_dir: Path, ct_dir: Path | None, scales_mm: list[float]):
    masks = find_niftis(mask_dir)
    if not masks:
        raise SystemExit(f"No .nii or .nii.gz masks found under {mask_dir}")
    cts = find_niftis(ct_dir) if ct_dir else {}
    missing = sorted(set(masks) - set(cts)) if ct_dir else []
    if missing:
        raise SystemExit("CT files missing for mask IDs: " + ", ".join(missing[:10]))
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_rows, curvature_rows, alignment_rows = [], [], []
    for i, (case_id, mask_path) in enumerate(masks.items(), 1):
        print(f"[{i}/{len(masks)}] {case_id}")
        ct_path = cts.get(case_id) if ct_dir else None
        stats, scales, alignment, _ = analyze_case(case_id, mask_path, ct_path, scales_mm)
        shape_rows.append(stats); curvature_rows.extend(scales); alignment_rows.append(alignment)
    shapes = pd.DataFrame(shape_rows)
    curves = pd.DataFrame(curvature_rows)
    shapes.to_csv(output_dir / "shape_summary.csv", index=False)
    curves.to_csv(output_dir / "curvature_by_case_and_scale.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(output_dir / "image_mask_alignment.csv", index=False)
    typical_id = shapes.iloc[(shapes["volume_mm3"] - shapes["volume_mm3"].median()).abs().argsort()[:1]].iloc[0]["case_id"]
    typical_mask = masks[typical_id]
    typical_ct = cts.get(typical_id) if ct_dir else None
    _, _, _, typical_mesh = analyze_case(typical_id, typical_mask, typical_ct, scales_mm)
    plot_representative(typical_id, typical_mesh, output_dir / "median_volume_case_curvature.png", scales_mm[len(scales_mm) // 2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(shapes["volume_mm3"] / 1000.0, bins=12, color="#4378a8", edgecolor="white")
    axes[0].set_xlabel("Pancreas volume (cm³)"); axes[0].set_ylabel("Cases"); axes[0].set_title("Mask volume across cases")
    for scale, frame in curves.groupby("smoothing_sigma_mm"):
        axes[1].scatter(np.full(len(frame), scale), frame["kmax_abs_p50_per_mm"], alpha=0.65, s=20)
    axes[1].set_xlabel("Mask smoothing σ (mm)"); axes[1].set_ylabel("Per-case median |κ|max (1/mm)"); axes[1].set_title("Curvature stability across smoothing scales")
    fig.tight_layout(); fig.savefig(output_dir / "eda_summary.png", dpi=160); plt.close(fig)
    print(f"Analyzed {len(shapes)} masks. Results: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Summarize pancreas mask shape and 3D curvature across cases.")
    parser.add_argument("--masks", type=Path, required=True, help="Folder of pancreas NIfTI masks; filenames define case IDs")
    parser.add_argument("--images", type=Path, help="Optional folder of paired CT NIfTI volumes with matching filenames")
    parser.add_argument("--out", type=Path, default=Path("outputs/shape-eda"))
    parser.add_argument("--scales-mm", type=float, nargs="+", default=[0.75, 1.5, 3.0])
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.scales_mm):
        parser.error("all smoothing scales must be positive millimeters")
    run_eda(args.masks, args.out, args.images, args.scales_mm)


if __name__ == "__main__":
    main()
