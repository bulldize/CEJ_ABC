import numpy as np

from src.abc.viz import (
    _break_long_segments_for_view,
    _clip_axis_tail_outliers_for_view,
    _filter_points_by_curve_mask,
    _rewrite_abc_viewer_labels,
)


def test_abc_viz_rewrite_handles_escaped_and_plain_text(tmp_path):
    html = (
        '<div>"title":{"text":"CEJ 3D\\u53ef\\u89c6\\u5316 | \\u75c5\\u4f8b=TF_008 \\u7259\\u4f4d=11"}</div>'
        '<div><b>流程图例</b><br>1 \\u6807\\u6ce8\\u70b9（无）<br>2 \\u63d2\\u503c\\u66f2\\u7ebf</div>'
        '<div>"name":"\\u4f2aGT\\u70ed\\u56fe (\\u003e0.30)"</div>'
        '<div>"name":"推理热图 (>0.30)"</div>'
    )
    p = tmp_path / "viewer.html"
    p.write_text(html, encoding="utf-8")

    _rewrite_abc_viewer_labels(str(p))

    out = p.read_text(encoding="utf-8")
    assert "CEJ 3D\\u53ef\\u89c6\\u5316" not in out
    assert "ABC 3D\\u53ef\\u89c6\\u5316" in out
    assert "1 ABC\\u70b9\\u4f4d" in out
    assert "2 ABC\\u63d2\\u503c\\u66f2\\u7ebf" in out
    assert '"name":"ABC\\u5019\\u9009\\u70ed\\u56fe (\\u003e0.30)"' in out
    assert '"name":"ABC结果热图 (>0.30)"' in out


def test_break_long_segments_for_view_inserts_nan_separator():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = _break_long_segments_for_view(pts, jump_ratio=3.0, min_jump_vox=3.0)
    assert out.shape[0] == pts.shape[0] + 1
    assert np.isnan(out[3]).all()


def test_clip_axis_tail_outliers_for_view_masks_one_sided_low_tail():
    # Around z=10 ring + one-sided low-z tail.
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    ring = np.stack(
        [
            8.0 * np.cos(theta),
            8.0 * np.sin(theta),
            np.full_like(theta, 10.0),
        ],
        axis=1,
    ).astype(np.float32)
    tail = np.array(
        [
            [0.0, 0.0, 6.0],
            [0.5, 0.0, 5.5],
            [1.0, 0.0, 5.0],
            [1.5, 0.0, 4.5],
            [2.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    pts = np.concatenate([ring, tail], axis=0)

    out = _clip_axis_tail_outliers_for_view(
        pts,
        spacing=(1.0, 1.0, 1.0),
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        mad_k=2.5,
        min_outliers=3,
        asymmetry_ratio=1.4,
    )
    nan_rows = np.isnan(out).all(axis=1)
    # Low-tail points should be masked for display.
    assert int(nan_rows.sum()) >= 3


def test_filter_points_by_curve_mask_keeps_points_near_curve():
    curve = np.zeros((16, 16, 16), dtype=np.uint8)
    curve[5, 5, 5] = 1
    curve[6, 5, 5] = 1
    curve[7, 5, 5] = 1
    curve[8, 5, 5] = 1
    curve[9, 5, 5] = 1

    pts = np.array(
        [
            [5.0, 5.0, 5.0],   # keep
            [6.2, 5.1, 5.0],   # keep
            [12.0, 12.0, 12.0],  # drop
            [8.1, 5.0, 5.0],   # keep
            [0.0, 0.0, 0.0],   # drop
        ],
        dtype=np.float32,
    )
    kept = _filter_points_by_curve_mask(pts, curve, max_dist_vox=1.5)
    assert kept.shape[0] == 3
    assert np.allclose(kept[0], pts[0])
    assert np.allclose(kept[1], pts[1])
    assert np.allclose(kept[2], pts[3])
