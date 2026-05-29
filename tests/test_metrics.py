import numpy as np
from scipy.spatial import cKDTree

from src.datasets.io import save_volume
from src.eval import compute_distances as eval_compute_distances, evaluate_predictions


def compute_distances(points_vox, curve_mask, spacing):
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    curve_mm = curve_pts * spacing
    pts_mm = np.asarray(points_vox, dtype=np.float32) * spacing
    tree = cKDTree(curve_mm)
    d, _ = tree.query(pts_mm, k=1)
    return d.astype(np.float32)


def test_point_to_curve_zero():
    shape = (10, 10, 10)
    curve = np.zeros(shape, dtype=np.uint8)
    for i in range(2, 8):
        curve[i, i, i] = 1
    pts = np.array([[2, 2, 2], [4, 4, 4], [7, 7, 7]], dtype=np.float32)
    d = compute_distances(pts, curve, spacing=(1.0, 1.0, 1.0))
    assert np.allclose(d, 0.0)


def test_eval_empty_curve_uses_finite_penalty():
    curve = np.zeros((10, 10, 10), dtype=np.uint8)
    pts = np.array([[2, 2, 2], [4, 4, 4]], dtype=np.float32)
    d = eval_compute_distances(pts, curve, spacing=(1.0, 1.0, 1.0))
    assert d.shape == (2,)
    assert np.all(np.isfinite(d))
    assert np.all(d > 1.5)


def test_eval_missing_prediction_writes_failed_row_with_penalty(tmp_path):
    processed_dir = tmp_path / "processed"
    tooth_dir = processed_dir / "case_a" / "tooth_11"
    tooth_dir.mkdir(parents=True)
    shape = (5, 5, 5)
    save_volume(str(tooth_dir / "A_t.nii.gz"), np.ones(shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    save_volume(str(tooth_dir / "T_t.nii.gz"), np.ones(shape, dtype=np.uint8), spacing=(1.0, 1.0, 1.0))
    save_volume(str(tooth_dir / "H_GT.nii.gz"), np.ones(shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    (tooth_dir / "roi_meta.json").write_text('{"case_id": "case_a", "tooth_id": 11}')
    (tooth_dir / "points.json").write_text(
        '{"case_id": "case_a", "coord_type": "voxel", "space": "roi", "points": {"11": [[1, 1, 1]]}}'
    )

    result = evaluate_predictions(
        processed_dir=str(processed_dir),
        infer_dir=str(tmp_path / "infer"),
        out_dir=str(tmp_path / "eval"),
        taus=[1.0],
        prediction_name="C_pred_fit.nii.gz",
    )

    assert result["summary"]["missing_prediction_count"] == 1
    assert result["rows"][0]["status"] == "missing_prediction"
    assert result["rows"][0]["p95_dist_mm"] > 1.0
    assert (tmp_path / "eval" / "metrics_per_tooth.csv").exists()


def test_eval_empty_existing_prediction_reports_no_curve(tmp_path):
    processed_dir = tmp_path / "processed"
    infer_dir = tmp_path / "infer"
    tooth_dir = processed_dir / "case_a" / "tooth_11"
    pred_dir = infer_dir / "case_a" / "tooth_11"
    tooth_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    shape = (5, 5, 5)
    save_volume(str(tooth_dir / "A_t.nii.gz"), np.ones(shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    save_volume(str(tooth_dir / "T_t.nii.gz"), np.ones(shape, dtype=np.uint8), spacing=(1.0, 1.0, 1.0))
    save_volume(str(tooth_dir / "H_GT.nii.gz"), np.ones(shape, dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    save_volume(str(pred_dir / "C_pred_fit.nii.gz"), np.zeros(shape, dtype=np.uint8), spacing=(1.0, 1.0, 1.0))
    (tooth_dir / "roi_meta.json").write_text('{"case_id": "case_a", "tooth_id": 11}')
    (tooth_dir / "points.json").write_text(
        '{"case_id": "case_a", "coord_type": "voxel", "space": "roi", "points": {"11": [[1, 1, 1]]}}'
    )

    result = evaluate_predictions(
        processed_dir=str(processed_dir),
        infer_dir=str(infer_dir),
        out_dir=str(tmp_path / "eval"),
        taus=[1.0],
        prediction_name="C_pred_fit.nii.gz",
    )

    assert result["summary"]["missing_prediction_count"] == 0
    assert result["summary"]["no_curve_count"] == 1
    assert result["summary"]["failed_tooth_count"] == 1
    assert result["rows"][0]["status"] == "no_curve"
