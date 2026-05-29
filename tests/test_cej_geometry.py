import numpy as np

from src.datasets.cej_geometry import build_geometry_prior, check_curve_constraints
from src.datasets.heatmap import fit_curve_and_sample, generate_heatmap_from_points
from src.postprocess.skeleton import extract_curve_from_heatmap_peak


def _synthetic_tooth_mask(shape=(56, 56, 80)):
    grid = np.indices(shape).astype(np.float32)
    cx, cy = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    z = grid[2]
    # Wider crown end at positive z, narrower root end at negative z.
    radius = 8.0 + 5.5 * (z / max(1.0, shape[2] - 1))
    r_xy = np.sqrt((grid[0] - cx) ** 2 + (grid[1] - cy) ** 2)
    return (r_xy <= radius).astype(np.uint8)


def _synthetic_cej_points(center=(28.0, 28.0, 38.0), n=18):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = 13.0 + 1.4 * np.sin(3.0 * theta)
    z = center[2] + 0.15 * np.sin(theta)
    pts = np.stack(
        [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            z,
        ],
        axis=1,
    )
    # Deliberately scramble order to exercise geometric sorting.
    return pts[[0, 5, 10, 2, 7, 12, 4, 9, 14, 1, 6, 11, 3, 8, 13, 15, 16, 17]].astype(np.float32)


def test_geometry_prior_axes_are_orthogonal_on_synthetic_tooth():
    mask = _synthetic_tooth_mask()
    prior = build_geometry_prior(
        mask,
        spacing=(1.0, 1.0, 1.0),
        tooth_id=11,
        points_vox=_synthetic_cej_points(),
        full_tooth_centroids_mm={
            11: np.array([28.0, 28.0, 38.0], dtype=np.float32),
            12: np.array([45.0, 28.0, 38.0], dtype=np.float32),
        },
        arch_centroid_mm=np.array([28.0, 12.0, 38.0], dtype=np.float32),
    )
    axes = prior["axes"]
    long_axis = np.asarray(axes["long_axis_root_to_crown"], dtype=np.float32)
    md_axis = np.asarray(axes["mesial_axis"], dtype=np.float32)
    bl_axis = np.asarray(axes["buccal_axis"], dtype=np.float32)

    assert prior["confidence"]["long_axis"] > 0.05
    assert abs(float(np.dot(long_axis, md_axis))) < 1e-4
    assert abs(float(np.dot(long_axis, bl_axis))) < 1e-4
    assert abs(float(np.dot(md_axis, bl_axis))) < 1e-4


def test_constrained_cej_fit_has_mesial_distal_peaks_and_buccal_lingual_valleys():
    mask = _synthetic_tooth_mask()
    pts = _synthetic_cej_points()
    prior = build_geometry_prior(
        mask,
        spacing=(1.0, 1.0, 1.0),
        tooth_id=11,
        points_vox=pts,
        full_tooth_centroids_mm={
            11: np.array([28.0, 28.0, 38.0], dtype=np.float32),
            12: np.array([45.0, 28.0, 38.0], dtype=np.float32),
        },
        arch_centroid_mm=np.array([28.0, 12.0, 38.0], dtype=np.float32),
        constraint_params={"project_to_surface": False, "min_crown_root_amplitude_mm": 1.0},
    )

    dense, report = fit_curve_and_sample(
        pts,
        spacing=(1.0, 1.0, 1.0),
        step_mm=0.5,
        closed=True,
        geometry_prior=prior,
        return_report=True,
    )
    constraints = check_curve_constraints(dense, (1.0, 1.0, 1.0), prior)

    assert report["used_geometry_prior"]
    assert not report["fallback"]
    assert constraints["passed"]
    assert constraints["direction_values_mm"]["mesial"] > constraints["direction_values_mm"]["buccal"]
    assert constraints["direction_values_mm"]["distal"] > constraints["direction_values_mm"]["lingual"]


def test_fit_curve_without_geometry_prior_keeps_legacy_array_return():
    pts = _synthetic_cej_points()
    dense = fit_curve_and_sample(pts, spacing=(1.0, 1.0, 1.0), step_mm=0.5, closed=True)
    assert isinstance(dense, np.ndarray)
    assert dense.ndim == 2
    assert dense.shape[1] == 3
    assert dense.shape[0] >= 8


def test_heatmap_to_gt_curve_tracks_dense_fit():
    pts = _synthetic_cej_points()
    dense = fit_curve_and_sample(pts, spacing=(1.0, 1.0, 1.0), step_mm=0.5, closed=True)
    heatmap = generate_heatmap_from_points(
        (64, 64, 64),
        dense,
        spacing=(1.0, 1.0, 1.0),
        sigma_mm=1.0,
        connect_points=True,
        close_loop=True,
    )
    curve = extract_curve_from_heatmap_peak(heatmap, np.ones_like(heatmap, dtype=np.uint8), peak_threshold=0.999)

    curve_pts = np.argwhere(curve > 0).astype(np.float32)
    assert curve_pts.shape[0] > 0
    # The generated peak curve is rasterized from the dense fit, so its nearest
    # dense-point distance should stay within one voxel on this synthetic case.
    from scipy.spatial import cKDTree

    d, _ = cKDTree(dense).query(curve_pts, k=1)
    assert float(np.mean(d)) < 1.0
    assert float(np.percentile(d, 95)) < 1.5
