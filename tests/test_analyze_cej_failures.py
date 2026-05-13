import math

import numpy as np

from scripts.analyze_cej_failures import (
    Thresholds,
    classify_labeled_row,
    count_manual_points_data,
    curve_distance_stats,
    heatmap_diagnostics,
    issue_reason_text,
    render_model_guidance,
)


def test_points_json_empty_tooth_is_no_manual():
    data = {
        "case_id": "ToothFairy3F_041",
        "coord_type": "voxel",
        "space": "roi",
        "points": {"12": []},
    }

    assert count_manual_points_data(data, 12) == 0


def test_points_json_tooth_dict_counts_manual_points():
    data = {
        "case_id": "ToothFairy3F_052",
        "coord_type": "voxel",
        "space": "roi",
        "points": {"31": [[1, 2, 3], [4, 5, 6]]},
    }

    assert count_manual_points_data(data, "31") == 2


def test_curve_distance_stats_uses_bidirectional_nearest_distance():
    gt = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]], dtype=np.float32)
    pred = np.array([[0, 1, 0], [2, 1, 0], [4, 1, 0]], dtype=np.float32)

    stats = curve_distance_stats(gt, pred, spacing=(0.5, 1.0, 1.0))

    assert stats["n_gt_curve"] == 3
    assert stats["n_pred_curve"] == 3
    assert math.isclose(stats["gt_to_pred_mean"], 1.0)
    assert math.isclose(stats["pred_to_gt_mean"], 1.0)
    assert math.isclose(stats["sym_mean"], 1.0)


def test_labeled_large_error_rules_include_sparse_prediction():
    row = {"sym_mean": 0.2, "sym_p95": 0.4, "n_pred_curve": 5}

    flags = classify_labeled_row(row, Thresholds())

    assert flags["flag_large_error"] is True
    assert flags["flag_severe_error"] is True


def test_heatmap_diagnostics_flags_low_conf_tiny_fragmented_and_low_wrap():
    thresholds = Thresholds()
    heatmap = np.zeros((20, 20, 8), dtype=np.float32)
    heatmap[2:4, 2:4, 2] = 0.8
    heatmap[16:18, 16:18, 2] = 0.7
    tooth = np.zeros_like(heatmap, dtype=np.uint8)
    tooth[5:15, 5:15, 2:6] = 1

    stats = heatmap_diagnostics(heatmap, tooth, spacing=(1.0, 1.0, 1.0), thresholds=thresholds)

    assert stats["flag_low_conf"] is False
    assert stats["flag_tiny"] is True
    assert stats["flag_fragmented"] is True
    assert stats["flag_low_wrap"] is True
    assert "热图很小" in issue_reason_text(stats)
    assert "不连通" in issue_reason_text(stats)


def test_heatmap_diagnostics_flags_empty_prediction_low_conf():
    stats = heatmap_diagnostics(
        np.zeros((8, 8, 8), dtype=np.float32),
        np.ones((8, 8, 8), dtype=np.uint8),
        spacing=(1.0, 1.0, 1.0),
        thresholds=Thresholds(),
    )

    assert stats["flag_low_conf"] is True
    assert stats["flag_tiny"] is True
    assert stats["flag_heatmap_issue"] is True


def test_model_guidance_mentions_loss_and_annotation_priorities():
    md = render_model_guidance(
        {
            "labeled_rows": [{"flag_severe_error": True}],
            "no_manual_rows": [{"flag_low_conf": True}, {"flag_fragmented": True, "flag_low_wrap": True}],
        }
    )

    assert "DiceCELoss" in md
    assert "052" in md
    assert "31/32/41/42" in md
    assert "不要强制限制在牙体内部" in md
