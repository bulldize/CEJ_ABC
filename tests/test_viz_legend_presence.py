import numpy as np
from src.viz import _ensure_3d_viz_deps, save_3d_viewer


def test_fixed_process_legend_is_written_to_html(tmp_path):
    _ensure_3d_viz_deps()

    shape = (20, 20, 16)
    A = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.uint8)
    T[6:14, 6:14, 4:12] = 1
    H_pred = np.zeros(shape, dtype=np.float32)
    H_pred[10, 10, 8] = 1.0
    C_pred = np.zeros(shape, dtype=np.uint8)
    C_pred[10, 10, 8] = 1

    out_html = tmp_path / "viewer.html"
    cfg = {
        "viz": {
            "prediction_only": True,
            "enable_3d": True,
            "show_dense_interp_curve": True,
            "show_pseudo_gt_skeleton": False,
            "pred_heatmap_iso_threshold": 0.3,
        }
    }

    save_3d_viewer(
        A=A,
        T=T,
        spacing=(1.0, 1.0, 1.0),
        case_id="case_x",
        tooth_id=11,
        out_html=str(out_html),
        cfg=cfg,
        H_pred=H_pred,
        C_pred=C_pred,
        dense_curve_points=np.array([[9, 10, 8], [10, 10, 8], [11, 10, 8]], dtype=np.float32),
        R=None,
        distances=None,
    )

    html = out_html.read_text(encoding="utf-8")
    assert ("流程图例" in html) or ("\\u6d41\\u7a0b\\u56fe\\u4f8b" in html)
    assert ("1 推理热图" in html) or ("1 \\u63a8\\u7406\\u70ed\\u56fe" in html)
    assert ("2 推理曲线" in html) or ("2 \\u63a8\\u7406\\u66f2\\u7ebf" in html)
    assert ("3 拟合曲线" in html) or ("3 \\u62df\\u5408\\u66f2\\u7ebf" in html)
    assert "伪GT热图" not in html
    assert "\\u4f2aGT\\u70ed\\u56fe" not in html
