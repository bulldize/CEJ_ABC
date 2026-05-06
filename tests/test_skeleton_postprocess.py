import numpy as np

from src.postprocess.skeleton import (
    compute_curve_overlap_metrics,
    fit_curve_and_sample,
    rasterize_polyline_to_mask,
)


def _make_closed_ring_points(center=(32.0, 32.0), radius=12.0, z=20.0, num=180):
    t = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z_arr = np.full_like(x, z)
    return np.stack([x, y, z_arr], axis=1).astype(np.float32)


def test_fit_curve_and_sample_returns_closed_dense_curve():
    curve_pts = _make_closed_ring_points(num=24)
    dense = fit_curve_and_sample(
        curve_pts,
        spacing=(1.0, 1.0, 1.0),
        step_mm=0.5,
        closed=True,
        smooth=0.0,
    )

    assert dense.shape[1] == 3
    assert dense.shape[0] > curve_pts.shape[0]


def test_rasterized_dense_curve_overlaps_source_curve():
    shape = (64, 64, 48)
    curve_pts = _make_closed_ring_points()
    dense = fit_curve_and_sample(
        curve_pts,
        spacing=(1.0, 1.0, 1.0),
        step_mm=0.2,
        closed=True,
        smooth=0.0,
    )

    source = np.zeros(shape, dtype=np.uint8)
    fitted = np.zeros(shape, dtype=np.uint8)
    rasterize_polyline_to_mask(source, curve_pts, close_loop=True)
    rasterize_polyline_to_mask(fitted, dense, close_loop=True)

    metrics = compute_curve_overlap_metrics(source, fitted)
    assert metrics["iou"] >= 0.8
    assert metrics["dice"] >= 0.88
