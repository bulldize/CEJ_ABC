import numpy as np

from src.viz import _ensure_3d_viz_deps, save_3d_viewer


def test_fixed_process_legend_is_written_to_html(tmp_path):
    _ensure_3d_viz_deps()

    shape = (20, 20, 16)
    A = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.uint8)
    T[6:14, 6:14, 4:12] = 1
    H_gt = np.zeros(shape, dtype=np.float32)
    H_gt[10, 10, 8] = 1.0
    C_gt = np.zeros(shape, dtype=np.uint8)
    C_gt[10, 10, 8] = 1
    geometry_prior = {
        "origin_mm": [10.0, 10.0, 8.0],
        "axes": {
            "long_axis_root_to_crown": [0.0, 0.0, 1.0],
            "mesial_axis": [1.0, 0.0, 0.0],
            "buccal_axis": [0.0, 1.0, 0.0],
        },
        "confidence": {"overall": 0.8},
        "fit_summary": {"constraints_passed": True},
    }
    curve_fit_report = {
        "manual_to_curve": {"mean_mm": 0.1, "p95_mm": 0.2},
        "curve_to_gt_curve": {"mean_mm": 0.1, "p95_mm": 0.2},
        "constraints": {"passed": True, "peak_margin_mm": 1.0},
    }

    out_html = tmp_path / "viewer.html"
    cfg = {
        "viz": {
            "enable_3d": True,
            "show_dense_interp_curve": True,
            "show_pseudo_gt_skeleton": True,
            "heatmap_iso_threshold": 0.3,
            "pred_heatmap_iso_threshold": 0.3,
        }
    }

    save_3d_viewer(
        A=A,
        T=T,
        H_gt=H_gt,
        spacing=(1.0, 1.0, 1.0),
        points=np.zeros((0, 3), dtype=np.float32),
        case_id="case_x",
        tooth_id=11,
        out_html=str(out_html),
        cfg=cfg,
        C_gt=C_gt,
        dense_curve_points=np.zeros((0, 3), dtype=np.float32),
        H_pred=None,
        C_pred=None,
        R=None,
        distances=None,
        geometry_prior=geometry_prior,
        curve_fit_report=curve_fit_report,
    )

    html = out_html.read_text(encoding="utf-8")
    assert "流程图例" in html
    assert "1 标注点" in html
    assert "2 插值曲线" in html
    assert "3 伪GT热图" in html
    assert "4 伪GT骨架" in html
    assert "5 推理热图" in html
    assert "6 推理曲线" in html
    assert "约束拟合曲线" in html
    assert "GT热图" in html
    assert "GT曲线" in html
    assert "方向轴" in html
    assert "手工点到拟合曲线" in html
    assert "拟合曲线到GT曲线" in html
    assert "方向约束=通过" in html
