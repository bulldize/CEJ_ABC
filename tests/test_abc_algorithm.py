import numpy as np

from src.abc.algorithm import (
    _suppress_axis_projection_outliers,
    _suppress_ring_jump_outliers,
    extract_abc_curve,
)


def test_extract_abc_curve_on_synthetic_roi():
    shape = (48, 48, 48)
    xx, yy, zz = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    cx, cy = 24, 24

    tooth = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 6 ** 2) & (zz >= 10) & (zz <= 38)
    bone_outer = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 8 ** 2) & (zz >= 14) & (zz <= 34)
    bone_inner = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 7 ** 2) & (zz >= 14) & (zz <= 34)
    bone = np.logical_and(bone_outer, np.logical_not(bone_inner))

    A = (zz.astype(np.float32) / float(shape[2])).astype(np.float32)

    curve, points, meta = extract_abc_curve(
        A,
        tooth.astype(np.uint8),
        bone.astype(np.uint8),
        spacing_xyz=(1.0, 1.0, 1.0),
        cfg={
            "angular_bins": 72,
            "smooth_sigma_bins": 1.0,
            "gradient_snap": True,
            "snap_radius_mm": 1.5,
            "pdl_threshold_mm": 1.5,
        },
    )

    assert meta["status"] == "ok"
    assert curve.shape == shape
    assert int(curve.sum()) > 0
    assert points.shape[0] == 72
    assert meta["curve_length_mm"] > 0.0
    assert meta["n_components"] >= 1


def test_suppress_ring_jump_outliers_reduces_extreme_edges():
    theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack(
        [
            10.0 * np.cos(theta),
            10.0 * np.sin(theta),
            np.zeros_like(theta),
        ],
        axis=1,
    ).astype(np.float32)
    ring[3] = np.array([40.0, 40.0, 0.0], dtype=np.float32)

    seg_before = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
    fixed = _suppress_ring_jump_outliers(ring, ratio=2.5, max_iter=3)
    seg_after = np.linalg.norm(np.roll(fixed, -1, axis=0) - fixed, axis=1)

    assert float(np.max(seg_after)) < float(np.max(seg_before))
    assert float(np.max(seg_after)) < 2.5 * float(np.median(seg_after))


def test_suppress_axis_projection_outliers_removes_z_spikes():
    theta = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
    ring = np.stack(
        [
            10.0 * np.cos(theta),
            10.0 * np.sin(theta),
            np.full_like(theta, 12.0),
        ],
        axis=1,
    ).astype(np.float32)
    ring[8, 2] = 5.0
    ring[21, 2] = 18.0

    out = _suppress_axis_projection_outliers(
        ring,
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        window=4,
        mad_k=2.5,
        min_thr_mm=0.4,
        max_iter=2,
    )
    assert float(abs(out[8, 2] - 12.0)) < float(abs(ring[8, 2] - 12.0))
    assert float(abs(out[21, 2] - 12.0)) < float(abs(ring[21, 2] - 12.0))
