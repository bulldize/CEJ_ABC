import numpy as np

from src.postprocess.skeleton import (
    compute_curve_overlap_metrics,
    fit_curve_and_sample,
    postprocess_prediction_curve,
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


def test_postprocess_empty_heatmap_returns_empty_curve():
    heatmap = np.zeros((32, 32, 24), dtype=np.float32)
    tooth = np.ones_like(heatmap, dtype=np.uint8)

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg={"return_postprocess_diagnostics": True},
    )

    assert int(c_pred.sum()) == 0
    assert int(c_fit.sum()) == 0
    assert dense.shape == (0, 3)


def test_postprocess_uses_fallback_threshold_for_low_peak_heatmap():
    shape = (48, 48, 32)
    heatmap = np.zeros(shape, dtype=np.float32)
    tooth = np.ones(shape, dtype=np.uint8)
    ring = _make_closed_ring_points(center=(24.0, 24.0), radius=9.0, z=16.0, num=80)
    rasterize_polyline_to_mask(heatmap, ring, close_loop=True)
    heatmap *= 0.22
    cfg = {
        "threshold_theta": 0.3,
        "fallback_min_threshold": 0.1,
        "fit_pred_curve_min_points": 8,
        "return_postprocess_diagnostics": True,
    }

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg=cfg,
    )

    assert int(c_pred.sum()) >= 8
    assert int(c_fit.sum()) > 0
    assert dense.shape[0] > 0
    assert cfg["postprocess_diagnostics"]["fallback_used"] is True


def test_postprocess_filters_small_noise_components():
    shape = (64, 64, 32)
    heatmap = np.zeros(shape, dtype=np.float32)
    tooth = np.ones(shape, dtype=np.uint8)
    ring = _make_closed_ring_points(center=(32.0, 32.0), radius=10.0, z=16.0, num=120)
    rasterize_polyline_to_mask(heatmap, ring, close_loop=True)
    heatmap[3, 3, 3] = 1.0
    heatmap[4, 3, 3] = 1.0

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg={
            "threshold_theta": 0.5,
            "min_component_voxels": 8,
            "max_curve_components": 4,
            "fit_pred_curve_min_points": 8,
        },
    )

    assert int(c_pred[:8, :8, :8].sum()) == 0
    assert int(c_fit[:8, :8, :8].sum()) == 0
    assert dense.shape[0] > 0


def test_postprocess_does_not_default_to_tooth_mask_constraint():
    shape = (48, 48, 32)
    heatmap = np.zeros(shape, dtype=np.float32)
    tooth = np.zeros(shape, dtype=np.uint8)
    tooth[18:30, 18:30, 12:20] = 1
    ring = _make_closed_ring_points(center=(24.0, 24.0), radius=13.0, z=16.0, num=100)
    rasterize_polyline_to_mask(heatmap, ring, close_loop=True)

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg={"threshold_theta": 0.3, "fit_pred_curve_min_points": 8},
    )

    assert int(c_pred.sum()) > 0
    assert int(c_fit.sum()) > 0
    assert dense.shape[0] > 0


def test_postprocess_can_opt_in_to_tooth_mask_constraint():
    shape = (40, 40, 24)
    heatmap = np.zeros(shape, dtype=np.float32)
    tooth = np.zeros(shape, dtype=np.uint8)
    tooth[12:28, 12:28, 8:16] = 1
    heatmap[1:8, 1:8, 1:8] = 1.0

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg={
            "threshold_theta": 0.3,
            "fit_pred_curve_min_points": 8,
            "constrain_curve_to_tooth_mask": True,
        },
    )

    assert int(c_pred.sum()) == 0
    assert int(c_fit.sum()) == 0
    assert dense.shape == (0, 3)


def test_broken_ring_with_noise_does_not_force_closed_bridge_to_noise():
    shape = (72, 72, 36)
    heatmap = np.zeros(shape, dtype=np.float32)
    tooth = np.ones(shape, dtype=np.uint8)
    ring = _make_closed_ring_points(center=(36.0, 36.0), radius=11.0, z=18.0, num=160)
    broken = ring[ring[:, 1] <= 36.0]
    rasterize_polyline_to_mask(heatmap, broken, close_loop=False)
    heatmap[60:64, 60:64, 28:32] = 1.0

    c_pred, c_fit, dense = postprocess_prediction_curve(
        heatmap,
        tooth,
        spacing=(1.0, 1.0, 1.0),
        cfg={
            "threshold_theta": 0.5,
            "min_component_voxels": 8,
            "max_curve_components": 4,
            "fit_pred_curve_min_points": 8,
            "fit_pred_curve_min_closed_angle_coverage": 0.7,
        },
    )

    assert int(c_pred[58:, 58:, 26:].sum()) == 0
    assert int(c_fit[58:, 58:, 26:].sum()) == 0
    assert dense.shape[0] > 0
