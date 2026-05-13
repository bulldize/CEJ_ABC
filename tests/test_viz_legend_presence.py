import json

import numpy as np
from src.viz import _ensure_3d_viz_deps, _load_optional_points, _load_split_case_labels, _write_3d_index, save_3d_viewer


def _html_has(html, text):
    escaped = text.encode("unicode_escape").decode("ascii")
    escaped_slash = escaped.replace("/", "\\u002f")
    raw_escaped_slash = text.replace("/", "\\u002f")
    return text in html or escaped in html or escaped_slash in html or raw_escaped_slash in html


def test_fixed_process_legend_is_written_to_html(tmp_path):
    _ensure_3d_viz_deps()

    shape = (20, 20, 16)
    A = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.uint8)
    T[6:14, 6:14, 4:12] = 1
    H_pred = np.zeros(shape, dtype=np.float32)
    H_pred[10, 10, 8] = 1.0
    H_gt = np.zeros(shape, dtype=np.float32)
    H_gt[9, 10, 8] = 1.0
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
        H_gt=H_gt,
        H_pred=H_pred,
        C_pred=C_pred,
        other_teeth_mask=T.copy(),
        manual_dense_curve_points=np.array([[9, 10, 8], [10, 10, 8], [11, 10, 8]], dtype=np.float32),
        pred_dense_curve_points=np.array([[9, 11, 8], [10, 11, 8], [11, 11, 8]], dtype=np.float32),
        R=None,
        distances=None,
    )

    html = out_html.read_text(encoding="utf-8")
    assert _html_has(html, "诊断")
    assert _html_has(html, "MPR三视图")
    assert _html_has(html, "牙体表面")
    assert _html_has(html, "推理热图")
    assert _html_has(html, "伪GT热图")
    assert _html_has(html, "方向差异热图")
    assert _html_has(html, "手工/GT插值曲线")
    assert _html_has(html, "预测拟合曲线")
    assert _html_has(html, "两个曲线对比")
    assert _html_has(html, "显示控制")
    assert _html_has(html, "推理热图")
    assert _html_has(html, "伪GT热图")
    assert _html_has(html, "插值曲线")
    assert _html_has(html, "手工标点")
    assert _html_has(html, "其他牙分割")
    assert _html_has(html, "H_pred max=")
    assert _html_has(html, "mesh3d")
    assert not _html_has(html, "置信度=%{customdata[0]:.4f}")
    assert not _html_has(html, "差值=%{customdata[0]:.4f}")
    assert _html_has(html, "\"visible\":true")
    assert _html_has(html, "手工插值曲线")
    assert _html_has(html, "预测拟合曲线")
    assert not _html_has(html, "仅自检链路")
    assert not _html_has(html, "仅推理热图")
    assert not _html_has(html, "仅伪GT热图")
    assert not _html_has(html, "仅差异热图")
    assert not _html_has(html, "隐藏曲线")


def test_viewer_writes_split_label(tmp_path):
    _ensure_3d_viz_deps()

    shape = (20, 20, 16)
    A = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.uint8)
    T[6:14, 6:14, 4:12] = 1
    H_pred = np.zeros(shape, dtype=np.float32)
    H_pred[10, 10, 8] = 1.0

    out_html = tmp_path / "viewer.html"
    save_3d_viewer(
        A=A,
        T=T,
        spacing=(1.0, 1.0, 1.0),
        case_id="case_x",
        tooth_id=11,
        out_html=str(out_html),
        cfg={"viz": {"enable_3d": True, "show_pseudo_gt_skeleton": False}},
        H_pred=H_pred,
        split_label="测试集",
    )

    html = out_html.read_text(encoding="utf-8")
    assert _html_has(html, "数据集")
    assert _html_has(html, "测试集")


def test_load_optional_points_reads_tooth_points(tmp_path):
    points_path = tmp_path / "points.json"
    points_path.write_text(
        '{"case_id":"case_x","coord_type":"voxel","space":"roi",'
        '"points":{"11":[[1,2,3],[4,5,6]],"12":[[7,8,9]]}}',
        encoding="utf-8",
    )

    pts = _load_optional_points(str(points_path), 11)

    assert pts.dtype == np.float32
    assert pts.shape == (2, 3)
    assert np.allclose(pts[1], [4, 5, 6])


def test_3d_index_groups_teeth_and_marks_manual_labels(tmp_path):
    index_path = tmp_path / "index.html"
    _write_3d_index(
        str(index_path),
        [
            {
                "case_id": "case_a",
                "tooth_id": 11,
                "rel_path": "case_a/tooth_11/viewer.html",
                "has_manual_points": True,
                "has_gt_heatmap": True,
                "has_pred_heatmap": True,
                "eval_mean_mm": 0.1234,
                "eval_p95_mm": 0.5678,
                "split_label": "训练集",
            },
            {
                "case_id": "case_a",
                "tooth_id": 38,
                "rel_path": "case_a/tooth_38/viewer.html",
                "has_manual_points": False,
                "has_gt_heatmap": False,
                "has_pred_heatmap": True,
                "eval_mean_mm": None,
                "eval_p95_mm": None,
                "split_label": "测试集",
            },
        ],
    )

    html = index_path.read_text(encoding="utf-8")
    assert _html_has(html, "CEJ 3D Viewer")
    assert _html_has(html, "case_a")
    assert _html_has(html, "切牙")
    assert _html_has(html, "智齿")
    assert _html_has(html, "单根")
    assert _html_has(html, "多根")
    assert _html_has(html, "手工标注")
    assert _html_has(html, "无手工标注")
    assert _html_has(html, "GT热图")
    assert _html_has(html, "预测热图")
    assert _html_has(html, "标点误差 mean")
    assert _html_has(html, "标点误差 p95")
    assert _html_has(html, "0.123mm")
    assert _html_has(html, "0.568mm")
    assert _html_has(html, "训练集")
    assert _html_has(html, "测试集")
    assert _html_has(html, "训练集 病例/牙")
    assert _html_has(html, "测试集 病例/牙")


def test_load_split_case_labels_reads_train_and_holdout(tmp_path):
    split_path = tmp_path / "split_summary.json"
    split_path.write_text(
        json.dumps(
            {
                "train_case_ids": ["ToothFairy3F_025"],
                "holdout_case_ids": ["ToothFairy3F_040"],
            }
        ),
        encoding="utf-8",
    )

    labels = _load_split_case_labels(str(split_path))

    assert labels["ToothFairy3F_025"] == "训练集"
    assert labels["ToothFairy3F_040"] == "测试集"
