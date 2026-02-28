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
    )

    html = out_html.read_text(encoding="utf-8")
    assert "流程图例" in html
    assert "1 标注点" in html
    assert "2 插值曲线" in html
    assert "3 伪GT热图" in html
    assert "4 伪GT骨架" in html
    assert "5 推理热图" in html
    assert "6 推理曲线" in html
