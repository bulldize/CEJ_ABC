import argparse
import glob
from html import escape as _html_escape
import json
import os

import numpy as np
from monai.data.utils import affine_to_spacing
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from src.datasets.io import load_volume
from src.postprocess.priors import compute_geometric_prior
from src.postprocess.skeleton import extract_curve, extract_curve_from_heatmap_peak
from src.utils.config import ensure_dir, load_config
from src.utils.geometry import tooth_surface
from src.utils.log import get_logger

logger = get_logger("viz")

_plt = None
_binary_erosion = None
_go = None
_marching_cubes = None


def _ensure_spacing(spacing, affine):
    if spacing is not None:
        return spacing
    if affine is None:
        return (1.0, 1.0, 1.0)
    return tuple(affine_to_spacing(affine))


def _ensure_2d_viz_deps():
    global _plt, _binary_erosion
    if _plt is None:
        import matplotlib.pyplot as plt

        _plt = plt
    if _binary_erosion is None:
        from skimage.morphology import binary_erosion

        _binary_erosion = binary_erosion


def _ensure_3d_viz_deps():
    global _go, _marching_cubes
    if _go is None:
        import plotly.graph_objects as go

        _go = go
    if _marching_cubes is None:
        from skimage.measure import marching_cubes

        _marching_cubes = marching_cubes


def _load_split_case_labels(split_summary_path):
    if not split_summary_path:
        return {}
    if not os.path.exists(split_summary_path):
        logger.warning("split summary not found: %s", split_summary_path)
        return {}
    try:
        with open(split_summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("failed to load split summary %s: %s", split_summary_path, e)
        return {}

    labels = {}
    for case_id in data.get("train_case_ids", []) or []:
        labels[str(case_id)] = "训练集"
    for case_id in data.get("holdout_case_ids", []) or []:
        labels[str(case_id)] = "测试集"
    for case_id in data.get("test_case_ids", []) or []:
        labels[str(case_id)] = "测试集"
    return labels


def _split_badge(label):
    if not label:
        return ""
    cls = "split-test" if label == "测试集" else "split-train" if label == "训练集" else "split-unknown"
    return f"<span class='badge split {cls}'>{_html_escape(label)}</span>"


def boundary2d(mask2d):
    er = _binary_erosion(mask2d)
    return mask2d ^ er


def pick_slices(shape_z, n=4):
    if n <= 1:
        return [shape_z // 2]
    return np.linspace(0, shape_z - 1, n, dtype=int).tolist()


def save_overlay(A, overlay=None, points=None, out_path=None, title=None, num_slices=4):
    z_slices = pick_slices(A.shape[2], n=num_slices)
    fig, axes = _plt.subplots(1, len(z_slices), figsize=(4 * len(z_slices), 4))
    if len(z_slices) == 1:
        axes = [axes]

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(A[:, :, z].T, cmap="gray", origin="lower")
        if overlay is not None:
            ov = overlay[:, :, z]
            ax.imshow(np.ma.masked_where(ov <= 0, ov).T, cmap="jet", alpha=0.4, origin="lower")
        if points is not None and len(points) > 0:
            pts_z = points[np.round(points[:, 2]).astype(int) == z]
            if pts_z.size > 0:
                ax.scatter(pts_z[:, 0], pts_z[:, 1], s=8, c="yellow")
        ax.set_axis_off()
        if title:
            ax.set_title(f"{title} z={z}")

    _plt.tight_layout()
    if out_path:
        _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def save_error_map(A, points, distances, out_path, num_slices=4):
    if points is None or len(points) == 0:
        return
    z_slices = pick_slices(A.shape[2], n=num_slices)
    fig, axes = _plt.subplots(1, len(z_slices), figsize=(4 * len(z_slices), 4))
    if len(z_slices) == 1:
        axes = [axes]

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(A[:, :, z].T, cmap="gray", origin="lower")
        pts_z_idx = np.where(np.round(points[:, 2]).astype(int) == z)[0]
        if pts_z_idx.size > 0:
            pts = points[pts_z_idx]
            d = distances[pts_z_idx]
            sc = ax.scatter(pts[:, 0], pts[:, 1], s=10, c=d, cmap="hot")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_axis_off()
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def compute_distances(points_vox, curve_mask, spacing):
    if points_vox is None or len(points_vox) == 0:
        return np.array([], dtype=np.float32)
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    if curve_pts.shape[0] == 0:
        return np.full((len(points_vox),), np.inf, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    curve_mm = curve_pts * spacing
    pts_mm = np.asarray(points_vox, dtype=np.float32) * spacing
    tree = cKDTree(curve_mm)
    d, _ = tree.query(pts_mm, k=1)
    return d.astype(np.float32)


def _downsample_step(shape, max_dim):
    if max_dim <= 0:
        return 1
    largest = int(max(shape))
    if largest <= max_dim:
        return 1
    return int(np.ceil(largest / float(max_dim)))


def _downsample_volume(vol, step):
    if step <= 1:
        return vol
    return vol[::step, ::step, ::step]


def _safe_normalize_for_display(A, p_low=1.0, p_high=99.0):
    lo, hi = np.percentile(A, [p_low, p_high])
    if hi <= lo:
        lo = float(np.min(A))
        hi = float(np.max(A))
    if hi <= lo:
        return np.zeros_like(A, dtype=np.float32)
    out = (A - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _make_mesh_trace_from_mask(mask, spacing, name, color, opacity, step=1, legendgroup=None, showlegend=True):
    if mask is None:
        return None
    if mask.ndim != 3:
        return None

    arr = _downsample_volume(mask, step)
    spacing_arr = np.asarray(spacing, dtype=np.float32) * float(step)
    arr = (arr > 0).astype(np.uint8)

    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[2] < 2:
        return None
    if int(arr.max()) == int(arr.min()):
        return None

    try:
        verts, faces, _, _ = _marching_cubes(arr.astype(np.float32), level=0.5, spacing=spacing_arr)
    except Exception:
        return None

    if verts.size == 0 or faces.size == 0:
        return None

    return _go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=float(opacity),
        color=color,
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="skip",
    )


def _make_isosurface_trace(
    volume,
    spacing,
    level,
    name,
    colorscale,
    opacity,
    legendgroup=None,
    showlegend=True,
    cmin=0.0,
    cmax=1.0,
    mesh_step=1,
):
    if volume is None:
        return None
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        return None
    if arr.shape[0] < 2 or arr.shape[1] < 2 or arr.shape[2] < 2:
        return None

    level = float(level)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if not (arr_min < level < arr_max):
        return None

    try:
        verts, faces, _, values = _marching_cubes(
            arr,
            level=level,
            spacing=np.asarray(spacing, dtype=np.float32),
            step_size=max(1, int(mesh_step)),
        )
    except Exception:
        return None
    if verts.size == 0 or faces.size == 0:
        return None

    return _go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=values,
        colorscale=colorscale,
        cmin=float(cmin),
        cmax=float(cmax),
        showscale=False,
        opacity=float(opacity),
        name=name,
        legendgroup=legendgroup,
        showlegend=showlegend,
        hoverinfo="skip",
        flatshading=False,
        lighting=dict(ambient=0.62, diffuse=0.72, specular=0.12, roughness=0.78),
    )


def _heatmap_stats(vol, threshold):
    if vol is None:
        return None
    arr = np.asarray(vol, dtype=np.float32)
    if arr.size == 0:
        return {"max": 0.0, "p99": 0.0, "vox_ge_threshold": 0, "vox_gt_zero": 0}
    return {
        "max": float(np.max(arr)),
        "p99": float(np.percentile(arr, 99)),
        "vox_ge_threshold": int((arr >= float(threshold)).sum()),
        "vox_gt_zero": int((arr > 0).sum()),
    }


def _diff_heatmap_stats(H_pred, H_gt, threshold):
    if H_pred is None or H_gt is None:
        return None
    pred = np.asarray(H_pred, dtype=np.float32)
    gt = np.asarray(H_gt, dtype=np.float32)
    if pred.shape != gt.shape or pred.size == 0:
        return None
    diff = pred - gt
    abs_diff = np.abs(diff)
    mask = abs_diff >= float(threshold)
    return {
        "max": float(np.max(diff)),
        "min": float(np.min(diff)),
        "abs_max": float(np.max(abs_diff)),
        "abs_p99": float(np.percentile(abs_diff, 99)),
        "vox_ge_threshold": int(mask.sum()),
        "vox_positive": int(np.logical_and(mask, diff > 0).sum()),
        "vox_negative": int(np.logical_and(mask, diff < 0).sum()),
    }


def _curve_surface_stats(curve_mask, tooth_mask, spacing):
    if curve_mask is None or tooth_mask is None:
        return None
    curve = np.asarray(curve_mask) > 0
    tooth = np.asarray(tooth_mask) > 0
    voxels = int(curve.sum())
    if voxels == 0 or int(tooth.sum()) == 0:
        return {
            "voxels": voxels,
            "inside_tooth": 0,
            "outside_tooth": voxels,
            "mean_surface_mm": None,
            "p95_surface_mm": None,
        }

    surface = tooth_surface(tooth.astype(np.uint8))
    if int(surface.sum()) == 0:
        return {
            "voxels": voxels,
            "inside_tooth": int(np.logical_and(curve, tooth).sum()),
            "outside_tooth": int(np.logical_and(curve, ~tooth).sum()),
            "mean_surface_mm": None,
            "p95_surface_mm": None,
        }

    dist_to_surface = distance_transform_edt(~surface, sampling=spacing)
    curve_dist = dist_to_surface[curve]
    return {
        "voxels": voxels,
        "inside_tooth": int(np.logical_and(curve, tooth).sum()),
        "outside_tooth": int(np.logical_and(curve, ~tooth).sum()),
        "mean_surface_mm": float(np.mean(curve_dist)),
        "p95_surface_mm": float(np.percentile(curve_dist, 95)),
    }


def _curve_pair_distance_stats(points_a, points_b, spacing):
    if points_a is None or points_b is None:
        return None
    a = np.asarray(points_a, dtype=np.float32)
    b = np.asarray(points_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        return None
    if a.shape[0] == 0 or b.shape[0] == 0:
        return None

    spacing_arr = np.asarray(spacing, dtype=np.float32)
    a_mm = a * spacing_arr
    b_mm = b * spacing_arr
    tree_b = cKDTree(b_mm)
    d_ab, _ = tree_b.query(a_mm, k=1)
    tree_a = cKDTree(a_mm)
    d_ba, _ = tree_a.query(b_mm, k=1)
    d = np.concatenate([d_ab, d_ba]).astype(np.float32)
    if d.size == 0:
        return None
    return {
        "count_a": int(a.shape[0]),
        "count_b": int(b.shape[0]),
        "mean_mm": float(np.mean(d)),
        "p95_mm": float(np.percentile(d, 95)),
        "max_mm": float(np.max(d)),
    }


def _fmt_float(value, digits=4):
    if value is None:
        return "无"
    if not np.isfinite(value):
        return "无"
    return f"{float(value):.{digits}f}"


def _fmt_mm(value):
    if value is None or not np.isfinite(value):
        return "无"
    return f"{float(value):.3f}mm"


def _add_layered_heatmap(
    fig,
    heatmap,
    spacing,
    threshold,
    name,
    colors,
    opacity,
    step=1,
    legendgroup=None,
):
    if heatmap is None:
        return 0
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.size == 0:
        return 0
    max_val = float(np.max(arr))
    if max_val < float(threshold) or max_val <= 0.0:
        return 0

    levels = [
        (float(threshold), colors[0], max(0.08, float(opacity) * 0.55), "低"),
        (max(float(threshold), max_val * 0.65), colors[1], max(0.12, float(opacity) * 0.85), "中"),
        (max(float(threshold), max_val * 0.85), colors[2], max(0.16, float(opacity) * 1.15), "高"),
    ]

    added = 0
    last_level = None
    for level, color, layer_opacity, suffix in levels:
        if last_level is not None and level <= last_level + 1e-6:
            continue
        mesh = _make_mesh_trace_from_mask(
            (arr >= level).astype(np.uint8),
            spacing,
            name=f"{name}{suffix}热区 (>{level:.3f})",
            color=color,
            opacity=min(0.92, layer_opacity),
            step=step,
            legendgroup=legendgroup,
            showlegend=(added == 0),
        )
        if mesh is not None:
            fig.add_trace(mesh)
            added += 1
        last_level = level
    return added


def _heatmap_point_cloud(
    heatmap,
    spacing,
    threshold,
    max_points=3000,
    signed=False,
):
    if heatmap is None:
        return None, None
    arr = np.asarray(heatmap, dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        return None, None

    if signed:
        mask = np.abs(arr) >= float(threshold)
    else:
        mask = arr >= float(threshold)
    pts = np.array(np.where(mask)).T.astype(np.float32)
    if pts.shape[0] == 0:
        return None, None

    vals = arr[mask].astype(np.float32)
    if pts.shape[0] > int(max_points):
        keep = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=int)
        pts = pts[keep]
        vals = vals[keep]

    spacing_arr = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing_arr
    return pts_mm, vals


def _add_heatmap_point_trace(
    fig,
    heatmap,
    spacing,
    threshold,
    name,
    legendgroup,
    colorscale,
    max_points=3000,
    marker_size=2.2,
    opacity=0.72,
    signed=False,
    cmin=None,
    cmax=None,
    colorbar_title=None,
    colorbar_y=0.5,
    showlegend=True,
):
    pts_mm, vals = _heatmap_point_cloud(
        heatmap,
        spacing,
        threshold,
        max_points=max_points,
        signed=signed,
    )
    if pts_mm is None:
        return False

    marker = dict(
        size=float(marker_size),
        color=vals,
        colorscale=colorscale,
        opacity=float(opacity),
        showscale=True,
        colorbar=dict(title=colorbar_title or name, thickness=10, len=0.26, x=1.02, y=float(colorbar_y)),
    )
    if cmin is not None:
        marker["cmin"] = float(cmin)
    if cmax is not None:
        marker["cmax"] = float(cmax)

    value_label = "差值" if signed else "置信度"
    fig.add_trace(
        _go.Scatter3d(
            x=pts_mm[:, 0],
            y=pts_mm[:, 1],
            z=pts_mm[:, 2],
            mode="markers",
            marker=marker,
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            customdata=vals[:, None],
            hovertemplate=(
                "x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<br>"
                f"{value_label}=%{{customdata[0]:.4f}}<extra></extra>"
            ),
        )
    )
    return True


def _add_thresholded_heatmap_traces(
    fig,
    H_gt,
    H_pred,
    spacing,
    thresholds,
    opacity,
    mesh_step=1,
):
    added = {"gt_heatmap": False, "pred_heatmap": False, "diff_heatmap": False}
    diff = None
    if H_pred is not None and H_gt is not None:
        pred_arr = np.asarray(H_pred, dtype=np.float32)
        gt_arr = np.asarray(H_gt, dtype=np.float32)
        if pred_arr.shape == gt_arr.shape:
            diff = pred_arr - gt_arr

    for thr in thresholds:
        thr = float(thr)
        pred_trace = _make_isosurface_trace(
            H_pred,
            spacing,
            level=thr,
            name=f"推理热图 >= {thr:.2f}",
            colorscale=[
                [0.00, "#7F1D1D"],
                [0.35, "#EA580C"],
                [0.70, "#FACC15"],
                [1.00, "#FFF7ED"],
            ],
            cmin=0.0,
            cmax=1.0,
            opacity=opacity,
            legendgroup="pred_heatmap",
            showlegend=not added["pred_heatmap"],
            mesh_step=mesh_step,
        )
        if pred_trace is not None:
            fig.add_trace(pred_trace)
            added["pred_heatmap"] = True

        gt_trace = _make_isosurface_trace(
            H_gt,
            spacing,
            level=thr,
            name=f"伪GT热图 >= {thr:.2f}",
            colorscale=[
                [0.00, "#064E3B"],
                [0.36, "#0D9488"],
                [0.72, "#A3E635"],
                [1.00, "#ECFCCB"],
            ],
            cmin=0.0,
            cmax=1.0,
            opacity=max(0.20, opacity * 0.78),
            legendgroup="gt_heatmap",
            showlegend=not added["gt_heatmap"],
            mesh_step=mesh_step,
        )
        if gt_trace is not None:
            fig.add_trace(gt_trace)
            added["gt_heatmap"] = True

        diff_pos_trace = _make_isosurface_trace(
            diff,
            spacing,
            level=thr,
            name=f"方向差异热图 预测高 >= {thr:.2f}",
            colorscale=[
                [0.00, "#7C2D12"],
                [0.45, "#F97316"],
                [0.78, "#FDBA74"],
                [1.00, "#DC2626"],
            ],
            cmin=0.0,
            cmax=1.0,
            opacity=min(0.45, opacity * 1.05),
            legendgroup="diff_heatmap",
            showlegend=not added["diff_heatmap"],
            mesh_step=mesh_step,
        )
        if diff_pos_trace is not None:
            fig.add_trace(diff_pos_trace)
            added["diff_heatmap"] = True

        diff_neg_trace = _make_isosurface_trace(
            -diff if diff is not None else None,
            spacing,
            level=thr,
            name=f"方向差异热图 预测低 >= {thr:.2f}",
            colorscale=[
                [0.00, "#312E81"],
                [0.45, "#2563EB"],
                [0.78, "#93C5FD"],
                [1.00, "#6D28D9"],
            ],
            cmin=0.0,
            cmax=1.0,
            opacity=min(0.45, opacity * 1.05),
            legendgroup="diff_heatmap",
            showlegend=not added["diff_heatmap"],
            mesh_step=mesh_step,
        )
        if diff_neg_trace is not None:
            fig.add_trace(diff_neg_trace)
            added["diff_heatmap"] = True
    return added


def _add_mpr_slices(fig, A, spacing, opacity=0.85):
    A_vis = _safe_normalize_for_display(A)
    nx, ny, nz = A_vis.shape
    sx, sy, sz = [float(v) for v in spacing]

    x_axis = np.arange(nx, dtype=np.float32) * sx
    y_axis = np.arange(ny, dtype=np.float32) * sy
    z_axis = np.arange(nz, dtype=np.float32) * sz

    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Axial (XY)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    zz = np.full_like(xx, z_axis[cz], dtype=np.float32)
    fig.add_trace(
        _go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=A_vis[:, :, cz],
            colorscale="Gray",
            opacity=float(opacity),
            showscale=False,
            name="轴位切片",
            legendgroup="mpr",
            hoverinfo="skip",
        )
    )

    # Coronal (XZ)
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    yy = np.full_like(xx, y_axis[cy], dtype=np.float32)
    fig.add_trace(
        _go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=A_vis[:, cy, :],
            colorscale="Gray",
            opacity=float(opacity),
            showscale=False,
            name="冠状切片",
            legendgroup="mpr",
            hoverinfo="skip",
        )
    )

    # Sagittal (YZ)
    yy, zz = np.meshgrid(y_axis, z_axis, indexing="ij")
    xx = np.full_like(yy, x_axis[cx], dtype=np.float32)
    fig.add_trace(
        _go.Surface(
            x=xx,
            y=yy,
            z=zz,
            surfacecolor=A_vis[cx, :, :],
            colorscale="Gray",
            opacity=float(opacity),
            showscale=False,
            name="矢状切片",
            legendgroup="mpr",
            hoverinfo="skip",
        )
    )


def _add_curve_points(
    fig,
    curve_mask,
    spacing,
    max_points=12000,
    color="#00E5FF",
    name="推理曲线",
    marker_size=2,
    legendgroup=None,
):
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    if curve_pts.shape[0] == 0:
        return
    if curve_pts.shape[0] > max_points:
        keep = np.linspace(0, curve_pts.shape[0] - 1, max_points, dtype=int)
        curve_pts = curve_pts[keep]
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    curve_mm = curve_pts * spacing_arr
    fig.add_trace(
        _go.Scatter3d(
            x=curve_mm[:, 0],
            y=curve_mm[:, 1],
            z=curve_mm[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color, opacity=0.9),
            name=name,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )


def _rasterize_polyline_to_mask(mask, points_xyz, close_loop=False):
    if points_xyz is None:
        return
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return

    shape = np.array(mask.shape, dtype=np.int32)

    def _mark(p):
        q = np.round(p).astype(np.int32)
        if np.all(q >= 0) and np.all(q < shape):
            mask[q[0], q[1], q[2]] = 1

    _mark(pts[0])
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        seg = np.linspace(p0, p1, n)
        for p in seg:
            _mark(p)
    if close_loop and pts.shape[0] > 2:
        p0 = pts[-1]
        p1 = pts[0]
        delta = np.abs(p1 - p0)
        n = int(np.ceil(np.max(delta))) + 1
        n = max(2, n)
        seg = np.linspace(p0, p1, n)
        for p in seg:
            _mark(p)


def _add_dense_interp_curve(
    fig,
    points_vox,
    spacing,
    max_points=20000,
    color="#F500FF",
    name="插值曲线",
    close_loop=True,
    legendgroup=None,
):
    if points_vox is None:
        return
    pts = np.asarray(points_vox, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 3:
        return
    if pts.shape[0] > max_points:
        keep = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[keep]
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing_arr
    if close_loop and pts_mm.shape[0] > 2:
        pts_mm = np.vstack([pts_mm, pts_mm[0:1]])
    fig.add_trace(
        _go.Scatter3d(
            x=pts_mm[:, 0],
            y=pts_mm[:, 1],
            z=pts_mm[:, 2],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=2, color=color, opacity=0.9),
            name=name,
            legendgroup=legendgroup,
            hoverinfo="skip",
        )
    )


def _add_gt_points(
    fig,
    points,
    spacing,
    distances=None,
    color="#FFD600",
    name="标注点",
    use_error_colormap=False,
    legendgroup=None,
):
    if points is None or len(points) == 0:
        return
    pts = np.asarray(points, dtype=np.float32)
    spacing_arr = np.asarray(spacing, dtype=np.float32)
    pts_mm = pts * spacing_arr

    has_valid_dist = distances is not None and len(distances) == len(points)
    if not has_valid_dist or not use_error_colormap:
        customdata = None
        hovertemplate = "x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>"
        if has_valid_dist:
            d = np.asarray(distances, dtype=np.float32)
            customdata = d[:, None]
            hovertemplate = (
                "x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<br>"
                "误差=%{customdata[0]:.3f}mm<extra></extra>"
            )
        fig.add_trace(
            _go.Scatter3d(
                x=pts_mm[:, 0],
                y=pts_mm[:, 1],
                z=pts_mm[:, 2],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.98),
                name=name,
                legendgroup=legendgroup,
                customdata=customdata,
                hovertemplate=hovertemplate,
            )
        )
        return

    d = np.asarray(distances, dtype=np.float32)
    d_vis = np.where(np.isfinite(d), d, np.nan)
    fig.add_trace(
        _go.Scatter3d(
            x=pts_mm[:, 0],
            y=pts_mm[:, 1],
            z=pts_mm[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=d_vis,
                colorscale="Turbo",
                cmin=0.0,
                cmax=float(np.nanpercentile(d_vis, 95)) if np.isfinite(d_vis).any() else 1.0,
                colorbar=dict(title="误差 (mm)"),
                opacity=0.95,
            ),
            name=f"{name}（误差着色）",
            legendgroup=legendgroup,
            hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<br>误差=%{marker.color:.3f}mm<extra></extra>",
        )
    )


def _heatmap_compare_panel_text(rows, diagnostics=None):
    lines = ["<b>诊断</b>"]
    if diagnostics:
        lines.extend(diagnostics)
    return "<br>".join(lines)


def _add_fixed_process_legend(fig, rows, diagnostics=None):
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.995,
        y=0.985,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        align="left",
        text=_heatmap_compare_panel_text(rows, diagnostics),
        bgcolor="rgba(15,23,42,0.72)",
        bordercolor="rgba(148,163,184,0.34)",
        borderwidth=1,
        borderpad=8,
        font=dict(size=11, color="#E5E7EB"),
    )


def _trace_group_from_name(name):
    name = name or ""
    if name in {"轴位切片", "冠状切片", "矢状切片"}:
        return "mpr"
    if name == "牙体表面":
        return "tooth"
    if name == "其他牙分割":
        return "other_teeth"
    if name.startswith("伪GT热图"):
        return "gt_heatmap"
    if name.startswith("推理热图"):
        return "pred_heatmap"
    if name.startswith("方向差异热图"):
        return "diff_heatmap"
    if name.startswith("几何先验"):
        return "prior"
    if name == "标注点" or name.startswith("标注点"):
        return "gt_points"
    if name == "手工插值曲线":
        return "manual_interp_curve"
    if name == "伪GT骨架":
        return "pseudo_gt_skeleton"
    if name == "推理曲线":
        return "pred_curve"
    if name == "预测拟合曲线":
        return "pred_fit_curve"
    return "other"


def _trace_threshold_from_name(name):
    name = name or ""
    marker = ">= "
    if marker not in name:
        return None
    try:
        return float(name.rsplit(marker, 1)[1])
    except ValueError:
        return None


def _visibility_for_groups(trace_groups, mode, trace_thresholds=None, active_threshold=None):
    heatmap_groups = {"gt_heatmap", "pred_heatmap", "diff_heatmap"}
    thresholded_modes = {
        "heatmap_compare": heatmap_groups,
        "pred_heatmap_only": {"pred_heatmap"},
        "gt_heatmap_only": {"gt_heatmap"},
        "diff_heatmap_only": {"diff_heatmap"},
    }
    if trace_thresholds is None:
        trace_thresholds = [None] * len(trace_groups)

    def threshold_ok(idx):
        if trace_groups[idx] not in heatmap_groups:
            return True
        if active_threshold is None:
            return True
        thr = trace_thresholds[idx]
        return thr is not None and abs(float(thr) - float(active_threshold)) <= 1e-6

    if mode in thresholded_modes:
        heatmap_keep = thresholded_modes[mode]
        keep_base = {"mpr", "tooth"}
        visible = []
        for idx, group in enumerate(trace_groups):
            if group in keep_base:
                visible.append(True)
            elif group in heatmap_groups:
                visible.append(group in heatmap_keep and threshold_ok(idx))
            else:
                visible.append(False)
        return visible
    if mode == "curve_compare":
        keep = {"mpr", "tooth", "other_teeth", "manual_interp_curve", "pred_fit_curve"}
        return [g in keep for g in trace_groups]
    if mode == "hide_curves":
        keep_hidden = {"gt_points", "manual_interp_curve", "pseudo_gt_skeleton", "pred_curve", "pred_fit_curve", "prior"}
        visible = []
        for idx, group in enumerate(trace_groups):
            if group in keep_hidden:
                visible.append(False)
            elif group in heatmap_groups:
                visible.append(threshold_ok(idx))
            else:
                visible.append(True)
        return visible
    return [True] * len(trace_groups)


def save_3d_viewer(
    A,
    T,
    spacing,
    case_id,
    tooth_id,
    out_html,
    cfg,
    H_gt=None,
    points=None,
    H_pred=None,
    C_pred=None,
    C_gt=None,
    other_teeth_mask=None,
    dense_curve_points=None,
    manual_dense_curve_points=None,
    pred_dense_curve_points=None,
    R=None,
    distances=None,
    split_label=None,
):
    viz_cfg = cfg.get("viz", {})
    max_dim = int(viz_cfg.get("max_render_dim", 96))
    mesh_step = int(viz_cfg.get("mesh_step", 1))
    slice_opacity = float(viz_cfg.get("slice_opacity", 0.32))
    slice_opacity = min(slice_opacity, float(viz_cfg.get("slice_opacity_cap", 0.36)))
    tooth_opacity = float(viz_cfg.get("tooth_opacity", 0.18))
    heatmap_opacity = float(viz_cfg.get("heatmap_opacity", 0.35))
    prior_opacity = float(viz_cfg.get("prior_opacity", 0.2))
    curve_max_points = int(viz_cfg.get("curve_max_points", 12000))
    dense_interp_curve_max_points = int(viz_cfg.get("dense_interp_curve_max_points", 20000))
    show_dense_interp_curve = bool(viz_cfg.get("show_dense_interp_curve", True))
    dense_interp_curve_closed = bool(viz_cfg.get("dense_interp_curve_closed", True))
    pseudo_gt_skeleton_max_points = int(viz_cfg.get("pseudo_gt_skeleton_max_points", 12000))
    show_pseudo_gt_skeleton = bool(viz_cfg.get("show_pseudo_gt_skeleton", True))
    use_error_colormap_for_gt_points = bool(viz_cfg.get("use_error_colormap_for_gt_points", False))
    heat_thr = float(viz_cfg.get("heatmap_iso_threshold", 0.30))
    pred_thr = float(viz_cfg.get("pred_heatmap_iso_threshold", 0.30))
    heatmap_thresholds = viz_cfg.get("heatmap_thresholds", [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90])
    heatmap_thresholds = sorted({float(v) for v in heatmap_thresholds})
    if not heatmap_thresholds:
        heatmap_thresholds = [0.30]
    heatmap_default_threshold = float(viz_cfg.get("heatmap_compare_default_threshold", pred_thr))
    heatmap_default_threshold = min(heatmap_thresholds, key=lambda v: abs(v - heatmap_default_threshold))
    heatmap_surface_opacity = float(viz_cfg.get("heatmap_surface_opacity", heatmap_opacity))
    prior_thr = float(viz_cfg.get("prior_iso_threshold", 0.50))
    color_tooth_surface = viz_cfg.get("color_tooth_surface", "#64B5F6")
    color_gt_heatmap = viz_cfg.get("color_gt_heatmap", "#FFA726")
    color_pred_heatmap = viz_cfg.get("color_pred_heatmap", "#FF5252")
    color_diff_heatmap = viz_cfg.get("color_diff_heatmap", "#F97316")
    color_prior = viz_cfg.get("color_prior", "#7E57C2")
    color_pred_curve = viz_cfg.get("color_pred_curve", "#00E5FF")
    color_dense_interp_curve = viz_cfg.get("color_dense_interp_curve", "#F500FF")
    color_pred_fit_curve = viz_cfg.get("color_pred_fit_curve", "#00B0FF")
    color_pseudo_gt_skeleton = viz_cfg.get("color_pseudo_gt_skeleton", "#76FF03")
    color_gt_points = viz_cfg.get("color_gt_points", "#FFD600")

    if manual_dense_curve_points is None:
        manual_dense_curve_points = dense_curve_points
    if pred_dense_curve_points is None and bool(viz_cfg.get("treat_legacy_dense_curve_as_pred_fit", False)):
        pred_dense_curve_points = dense_curve_points

    step = _downsample_step(A.shape, max_dim)
    A_ds = _downsample_volume(A, step)
    T_ds = _downsample_volume(T, step)
    H_gt_ds = _downsample_volume(H_gt, step) if H_gt is not None else None
    H_pred_ds = _downsample_volume(H_pred, step) if H_pred is not None else None
    other_teeth_ds = _downsample_volume(other_teeth_mask, step) if other_teeth_mask is not None else None
    R_ds = _downsample_volume(R, step) if R is not None else None
    spacing_ds = tuple(np.asarray(spacing, dtype=np.float32) * float(step))

    fig = _go.Figure()
    _add_mpr_slices(fig, A_ds, spacing_ds, opacity=slice_opacity)

    tooth_mesh = _make_mesh_trace_from_mask(
        T_ds,
        spacing_ds,
        name="牙体表面",
        color=color_tooth_surface,
        opacity=tooth_opacity,
        step=mesh_step,
        legendgroup="tooth",
    )
    if tooth_mesh is not None:
        fig.add_trace(tooth_mesh)

    other_teeth_mesh = _make_mesh_trace_from_mask(
        other_teeth_ds,
        spacing_ds,
        name="其他牙分割",
        color=viz_cfg.get("color_other_teeth", "#64748B"),
        opacity=float(viz_cfg.get("other_teeth_opacity", 0.10)),
        step=mesh_step,
        legendgroup="other_teeth",
    )
    if other_teeth_mesh is not None:
        fig.add_trace(other_teeth_mesh)

    heatmap_presence = _add_thresholded_heatmap_traces(
        fig,
        H_gt_ds,
        H_pred_ds,
        spacing_ds,
        thresholds=heatmap_thresholds,
        opacity=heatmap_surface_opacity,
        mesh_step=mesh_step,
    )

    if R_ds is not None:
        prior_mesh = _make_mesh_trace_from_mask(
            (R_ds >= prior_thr).astype(np.uint8),
            spacing_ds,
            name=f"几何先验 (>{prior_thr:.2f})",
            color=color_prior,
            opacity=prior_opacity,
            step=mesh_step,
            legendgroup="prior",
        )
        if prior_mesh is not None:
            fig.add_trace(prior_mesh)

    if C_pred is not None:
        _add_curve_points(
            fig,
            C_pred,
            spacing,
            max_points=curve_max_points,
            color=color_pred_curve,
            name="推理曲线",
            marker_size=3,
            legendgroup="pred_curve",
        )

    if show_dense_interp_curve and manual_dense_curve_points is not None:
        _add_dense_interp_curve(
            fig,
            manual_dense_curve_points,
            spacing,
            max_points=dense_interp_curve_max_points,
            color=color_dense_interp_curve,
            name="手工插值曲线",
            close_loop=dense_interp_curve_closed,
            legendgroup="manual_interp_curve",
        )

    if pred_dense_curve_points is not None:
        _add_dense_interp_curve(
            fig,
            pred_dense_curve_points,
            spacing,
            max_points=dense_interp_curve_max_points,
            color=color_pred_fit_curve,
            name="预测拟合曲线",
            close_loop=False,
            legendgroup="pred_fit_curve",
        )

    if show_pseudo_gt_skeleton and C_gt is not None:
        _add_curve_points(
            fig,
            C_gt,
            spacing,
            max_points=pseudo_gt_skeleton_max_points,
            color=color_pseudo_gt_skeleton,
            name="伪GT骨架",
            legendgroup="pseudo_gt_skeleton",
        )

    has_gt_points = points is not None and len(points) > 0
    has_interp_curve = bool(
        show_dense_interp_curve
        and manual_dense_curve_points is not None
        and len(manual_dense_curve_points) > 0
    )
    has_pred_fit_curve = bool(pred_dense_curve_points is not None and len(pred_dense_curve_points) > 0)
    has_gt_heatmap = bool(heatmap_presence.get("gt_heatmap", False))
    has_gt_skeleton = bool(show_pseudo_gt_skeleton and C_gt is not None and np.any(C_gt > 0))
    has_pred_heatmap = bool(heatmap_presence.get("pred_heatmap", False))
    has_diff_heatmap = bool(heatmap_presence.get("diff_heatmap", False))
    has_pred_curve = bool(C_pred is not None and np.any(C_pred > 0))
    has_prior = bool(R_ds is not None and np.any(R_ds >= prior_thr))
    has_tooth_surface = tooth_mesh is not None
    has_other_teeth = other_teeth_mesh is not None

    _add_gt_points(
        fig,
        points,
        spacing,
        distances=distances,
        color=color_gt_points,
        name="标注点",
        use_error_colormap=use_error_colormap_for_gt_points,
        legendgroup="gt_points",
    )

    process_legend_rows = [
        {"label": "MPR三视图", "color": "#BDBDBD", "present": True},
        {"label": "牙体表面", "color": color_tooth_surface, "present": has_tooth_surface},
        {"label": "其他牙分割", "color": viz_cfg.get("color_other_teeth", "#64748B"), "present": has_other_teeth},
        {"label": "推理热图", "color": color_pred_heatmap, "present": has_pred_heatmap},
        {"label": "伪GT热图", "color": color_gt_heatmap, "present": has_gt_heatmap},
        {"label": "方向差异热图", "color": color_diff_heatmap, "present": has_diff_heatmap},
        {"label": "手工/GT插值曲线", "color": color_dense_interp_curve, "present": has_interp_curve},
        {"label": "预测拟合曲线", "color": color_pred_fit_curve, "present": has_pred_fit_curve},
        {"label": "几何先验", "color": color_prior, "present": has_prior},
    ]

    gt_stats = _heatmap_stats(H_gt, heatmap_default_threshold)
    pred_stats = _heatmap_stats(H_pred, heatmap_default_threshold)
    diff_stats = _diff_heatmap_stats(H_pred, H_gt, heatmap_default_threshold)
    curve_stats = _curve_surface_stats(C_pred, T, spacing)
    curve_pair_stats = _curve_pair_distance_stats(manual_dense_curve_points, pred_dense_curve_points, spacing)

    def _diagnostics_for_threshold(thr):
        gt_stats_local = _heatmap_stats(H_gt, thr)
        pred_stats_local = _heatmap_stats(H_pred, thr)
        diff_stats_local = _diff_heatmap_stats(H_pred, H_gt, thr)
        lines = [f"置信度阈值={float(thr):.2f}"]
        if pred_stats_local is not None:
            lines.append(
                "H_pred max="
                f"{_fmt_float(pred_stats_local['max'])} p99={_fmt_float(pred_stats_local['p99'])} "
                f">={float(thr):.2f}体素={pred_stats_local['vox_ge_threshold']}"
            )
        else:
            lines.append("H_pred：无")
        if gt_stats_local is not None:
            lines.append(
                "H_GT max="
                f"{_fmt_float(gt_stats_local['max'])} p99={_fmt_float(gt_stats_local['p99'])} "
                f">={float(thr):.2f}体素={gt_stats_local['vox_ge_threshold']}"
            )
        else:
            lines.append("H_GT：无")
        if diff_stats_local is not None:
            lines.append(
                "|H_pred-H_GT| max="
                f"{_fmt_float(diff_stats_local['abs_max'])} p99={_fmt_float(diff_stats_local['abs_p99'])} "
                f">={float(thr):.2f}体素={diff_stats_local['vox_ge_threshold']}"
            )
            lines.append(
                "差异方向 预测高="
                f"{diff_stats_local['vox_positive']} 预测低={diff_stats_local['vox_negative']}"
            )
        else:
            lines.append("方向差异热图：无")
        if curve_stats is not None:
            lines.append(
                "推理曲线体素="
                f"{curve_stats['voxels']} 牙体内={curve_stats['inside_tooth']} "
                f"外={curve_stats['outside_tooth']}"
            )
            lines.append(
                "曲线到牙体表面 mean="
                f"{_fmt_mm(curve_stats['mean_surface_mm'])} p95={_fmt_mm(curve_stats['p95_surface_mm'])}"
            )
        else:
            lines.append("推理曲线表面距离：无")
        if curve_pair_stats is not None:
            lines.append(
                "两曲线距离 mean="
                f"{_fmt_mm(curve_pair_stats['mean_mm'])} "
                f"p95={_fmt_mm(curve_pair_stats['p95_mm'])} "
                f"max={_fmt_mm(curve_pair_stats['max_mm'])}"
            )
        else:
            lines.append("两曲线距离：无")
        return lines

    diagnostics = _diagnostics_for_threshold(heatmap_default_threshold)

    _add_fixed_process_legend(fig, process_legend_rows, diagnostics=diagnostics)

    trace_groups = [_trace_group_from_name(getattr(trace, "name", "")) for trace in fig.data]
    trace_thresholds = [_trace_threshold_from_name(getattr(trace, "name", "")) for trace in fig.data]
    panel_text_by_threshold = {
        float(thr): _heatmap_compare_panel_text(process_legend_rows, _diagnostics_for_threshold(thr))
        for thr in heatmap_thresholds
    }
    default_panel_text = panel_text_by_threshold[float(heatmap_default_threshold)]
    default_visible = _visibility_for_groups(
        trace_groups,
        "curve_compare",
        trace_thresholds=trace_thresholds,
        active_threshold=heatmap_default_threshold,
    )
    for trace, visible in zip(fig.data, default_visible):
        trace.visible = bool(visible)

    scene_camera = dict(eye=dict(x=1.45, y=1.45, z=1.15))

    title_text = f"<b>CEJ 3D Viewer</b> · {case_id} · 牙位 {tooth_id}"
    if split_label:
        title_text += f" · {split_label}"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.018,
            y=0.975,
            xanchor="left",
            yanchor="top",
            font=dict(size=18, color="#F8FAFC"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#0B1120",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            dragmode="turntable",
            camera=scene_camera,
            bgcolor="#0B1120",
            xaxis=dict(
                backgroundcolor="#0F172A",
                gridcolor="rgba(148,163,184,0.18)",
                zerolinecolor="rgba(148,163,184,0.22)",
                color="#CBD5E1",
                showspikes=False,
            ),
            yaxis=dict(
                backgroundcolor="#0F172A",
                gridcolor="rgba(148,163,184,0.18)",
                zerolinecolor="rgba(148,163,184,0.22)",
                color="#CBD5E1",
                showspikes=False,
            ),
            zaxis=dict(
                backgroundcolor="#0F172A",
                gridcolor="rgba(148,163,184,0.18)",
                zerolinecolor="rgba(148,163,184,0.22)",
                color="#CBD5E1",
                showspikes=False,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.0,
            xanchor="right",
            x=0.995,
            bgcolor="rgba(15,23,42,0.64)",
            bordercolor="rgba(148,163,184,0.22)",
            borderwidth=1,
            font=dict(size=11, color="#E5E7EB"),
        ),
        margin=dict(l=0, r=0, t=78, b=52),
    )
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)
    _inject_viewer_controls(
        out_html,
        rows=process_legend_rows,
        default_threshold=heatmap_default_threshold,
        split_label=split_label,
    )


def _inject_viewer_controls(out_html, rows, default_threshold, split_label=None):
    try:
        with open(out_html, "r", encoding="utf-8") as f:
            html = f.read()
    except OSError:
        return

    toggle_specs = [
        ("pred_heatmap", "推理热图", False),
        ("gt_heatmap", "伪GT热图", False),
        ("manual_interp_curve", "插值曲线", True),
        ("gt_points", "手工标点", False),
    ]
    controls = [
        "<div id='cej-controls' class='cej-controls'>",
        "<div class='cej-controls-title'>显示控制</div>",
    ]
    if split_label:
        controls.append(
            "<div class='cej-split-badge'>"
            f"<span>数据集</span><b>{_html_escape(split_label)}</b>"
            "</div>"
        )
    controls.extend(
        [
            "<button type='button' class='cej-primary' data-cej-curve>两个曲线对比</button>",
            "<div class='cej-toggle-list'>",
        ]
    )
    for group, label, checked in toggle_specs:
        check_attr = " checked" if checked else ""
        controls.append(
            "<label class='cej-toggle'>"
            f"<input type='checkbox' data-cej-toggle='{_html_escape(group)}'{check_attr}>"
            f"<span>{_html_escape(label)}</span>"
            "</label>"
        )
    controls.extend(["</div>", "<div id='cej-dynamic-legend' class='cej-dynamic-legend'></div>", "</div>"])
    controls_html = "\n".join(controls)

    panel_items = []
    for row in rows:
        status = "" if row["present"] else "（无）"
        panel_items.append(
            {
                "label": row["label"] + status,
                "color": row["color"],
                "group": _legend_group_for_label(row["label"]),
                "base": row["label"] in {"MPR三视图", "牙体表面", "其他牙分割", "预测拟合曲线"},
            }
        )
    legend_json = json.dumps(panel_items, ensure_ascii=False)

    style = """
<style id="cej-controls-style">
.cej-controls{position:fixed;left:18px;top:78px;z-index:20;width:210px;background:rgba(15,23,42,.78);border:1px solid rgba(148,163,184,.30);border-radius:8px;padding:10px 10px 12px;color:#e5e7eb;font:12px/1.35 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;backdrop-filter:blur(8px)}
.cej-controls-title{font-size:12px;color:#94a3b8;margin-bottom:8px;font-weight:700}
.cej-split-badge{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;border:1px solid rgba(56,189,248,.28);background:rgba(8,47,73,.32);border-radius:6px;padding:7px 8px}.cej-split-badge span{color:#94a3b8}.cej-split-badge b{color:#e0f2fe;font-size:13px}
.cej-primary{width:100%;border:1px solid rgba(56,189,248,.42);background:rgba(14,116,144,.30);color:#e0f2fe;border-radius:6px;padding:7px 8px;font-weight:700;cursor:pointer;margin-bottom:8px}
.cej-toggle-list{display:grid;gap:6px;margin-bottom:10px}.cej-toggle{display:flex;align-items:center;gap:7px;cursor:pointer;color:#cbd5e1}.cej-toggle input{accent-color:#38bdf8}
.cej-dynamic-legend{border-top:1px solid rgba(148,163,184,.22);padding-top:8px;display:grid;gap:5px}.cej-legend-row{display:flex;align-items:center;gap:7px;color:#cbd5e1}.cej-swatch{width:9px;height:9px;border-radius:2px;display:inline-block;box-shadow:0 0 0 1px rgba(255,255,255,.16)}
@media(max-width:760px){.cej-controls{left:10px;right:10px;top:auto;bottom:10px;width:auto}}
</style>
"""
    script = f"""
<script id="cej-controls-script">
(function(){{
  const legendItems = {legend_json};
  const defaultThreshold = {float(default_threshold):.2f};
  const baseGroups = new Set(['mpr','tooth','other_teeth','pred_fit_curve']);
  const controlled = new Set(['pred_heatmap','gt_heatmap','manual_interp_curve','gt_points']);
  function parseGroup(name){{
    name = name || '';
    if (['轴位切片','冠状切片','矢状切片'].includes(name)) return 'mpr';
    if (name === '牙体表面') return 'tooth';
    if (name === '其他牙分割') return 'other_teeth';
    if (name.startsWith('推理热图')) return 'pred_heatmap';
    if (name.startsWith('伪GT热图')) return 'gt_heatmap';
    if (name === '手工插值曲线') return 'manual_interp_curve';
    if (name === '预测拟合曲线') return 'pred_fit_curve';
    if (name === '标注点' || name.startsWith('标注点')) return 'gt_points';
    return 'other';
  }}
  function parseThreshold(name){{
    const m = String(name || '').match(/>=\\s*([0-9.]+)/);
    return m ? Number(m[1]) : null;
  }}
  function currentState(){{
    const state = {{}};
    document.querySelectorAll('[data-cej-toggle]').forEach(input => state[input.dataset.cejToggle] = input.checked);
    return state;
  }}
  function isTraceVisible(trace, state){{
    const group = parseGroup(trace.name);
    if (baseGroups.has(group)) return true;
    if (controlled.has(group)) {{
      if (!state[group]) return false;
      if (group === 'pred_heatmap' || group === 'gt_heatmap') {{
        const thr = parseThreshold(trace.name);
        return thr !== null && Math.abs(thr - defaultThreshold) < 1e-6;
      }}
      return true;
    }}
    return false;
  }}
  function renderLegend(state){{
    const el = document.getElementById('cej-dynamic-legend');
    if (!el) return;
    el.innerHTML = '';
    legendItems.forEach(item => {{
      if (!item.base && item.group && controlled.has(item.group) && !state[item.group]) return;
      const row = document.createElement('div');
      row.className = 'cej-legend-row';
      row.innerHTML = '<span class="cej-swatch" style="background:'+item.color+'"></span><span>'+item.label+'</span>';
      el.appendChild(row);
    }});
  }}
  function applyState() {{
    const gd = document.querySelector('.js-plotly-plot');
    if (!gd || !gd.data) return;
    const state = currentState();
    const visible = gd.data.map(trace => isTraceVisible(trace, state));
    Plotly.restyle(gd, {{visible}});
    renderLegend(state);
  }}
  function setCurveCompare() {{
    document.querySelectorAll('[data-cej-toggle]').forEach(input => {{
      input.checked = input.dataset.cejToggle === 'manual_interp_curve';
    }});
    applyState();
  }}
  function init() {{
    const controls = `{controls_html}`;
    document.body.insertAdjacentHTML('beforeend', controls);
    document.querySelectorAll('[data-cej-toggle]').forEach(input => input.addEventListener('change', applyState));
    const btn = document.querySelector('[data-cej-curve]');
    if (btn) btn.addEventListener('click', setCurveCompare);
    setTimeout(applyState, 0);
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}})();
</script>
"""
    if "</head>" in html:
        html = html.replace("</head>", style + "\n</head>", 1)
    if "</body>" in html:
        html = html.replace("</body>", script + "\n</body>", 1)
    try:
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(html)
    except OSError:
        return


def _legend_group_for_label(label):
    if label == "MPR三视图":
        return "mpr"
    if label == "牙体表面":
        return "tooth"
    if label == "其他牙分割":
        return "other_teeth"
    if label == "推理热图":
        return "pred_heatmap"
    if label == "伪GT热图":
        return "gt_heatmap"
    if label == "手工/GT插值曲线":
        return "manual_interp_curve"
    if label == "预测拟合曲线":
        return "pred_fit_curve"
    if label == "标注点":
        return "gt_points"
    return None


def _tooth_category(tooth_id):
    tid = int(tooth_id)
    pos = tid % 10
    if pos in {1, 2}:
        return "切牙"
    if pos == 3:
        return "尖牙"
    if pos in {4, 5}:
        return "前磨牙"
    if pos in {6, 7}:
        return "磨牙"
    if pos == 8:
        return "智齿"
    return "其他"


def _tooth_root_category(tooth_id):
    tid = int(tooth_id)
    quadrant = tid // 10
    pos = tid % 10
    if pos <= 5:
        return "单根"
    if quadrant in {1, 2} and pos in {6, 7, 8}:
        return "多根"
    if quadrant in {3, 4} and pos in {6, 7, 8}:
        return "多根"
    return "根型未知"


def _status_badge(label, present, positive_text=None, negative_text=None):
    text = positive_text if present else negative_text
    if text is None:
        text = label
    cls = "badge ok" if present else "badge muted"
    return f"<span class='{cls}'>{_html_escape(text)}</span>"


def _write_3d_index(index_path, records):
    by_case = {}
    for rec in records:
        by_case.setdefault(rec["case_id"], []).append(rec)

    total_teeth = len(records)
    total_cases = len(by_case)
    total_manual = sum(1 for rec in records if rec.get("has_manual_points"))
    train_cases = {rec["case_id"] for rec in records if rec.get("split_label") == "训练集"}
    test_cases = {rec["case_id"] for rec in records if rec.get("split_label") == "测试集"}
    train_teeth = sum(1 for rec in records if rec.get("split_label") == "训练集")
    test_teeth = sum(1 for rec in records if rec.get("split_label") == "测试集")
    category_order = ["切牙", "尖牙", "前磨牙", "磨牙", "智齿", "其他"]
    category_counts = {
        category: sum(1 for rec in records if _tooth_category(rec["tooth_id"]) == category)
        for category in category_order
    }

    lines = [
        "<!doctype html>",
        "<html lang='zh-CN'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>CEJ 3D Viewer</title>",
        "<style>",
        ":root{color-scheme:dark;--bg:#0b1120;--panel:#111827;--panel2:#0f172a;--line:#263244;--text:#e5e7eb;--muted:#94a3b8;--accent:#38bdf8;--ok:#34d399;--warn:#fbbf24;--test:#f472b6;}",
        "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}",
        "a{color:inherit;text-decoration:none}.shell{max-width:1480px;margin:0 auto;padding:28px 28px 40px;}",
        ".top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:22px;}",
        "h1{margin:0;font-size:30px;letter-spacing:0;font-weight:760}.sub{margin-top:6px;color:var(--muted);font-size:13px;}",
        ".metrics{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:10px 12px;min-width:92px}.metric b{display:block;font-size:18px}.metric span{color:var(--muted);font-size:12px}",
        ".case{margin-top:24px;background:rgba(15,23,42,.72);border:1px solid var(--line);border-radius:8px;overflow:hidden}.case-head{display:flex;justify-content:space-between;align-items:center;padding:14px 16px;background:#111827;border-bottom:1px solid var(--line)}",
        ".case-title{font-size:17px;font-weight:700;display:flex;align-items:center;gap:8px;flex-wrap:wrap}.case-meta{color:var(--muted);font-size:12px}.groups{padding:14px;display:grid;gap:14px}",
        ".group-title{display:flex;align-items:center;gap:8px;margin:2px 0 8px;color:#cbd5e1;font-weight:700}.count{color:var(--muted);font-weight:500;font-size:12px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}.tooth{display:block;background:#0b1220;border:1px solid #223047;border-radius:8px;padding:12px;transition:border-color .15s,background .15s}.tooth:hover{border-color:#38bdf8;background:#101a2d}",
        ".tooth-main{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}.tooth-id{font-size:22px;font-weight:760;line-height:1}.tooth-label{color:var(--muted);font-size:12px;margin-top:3px}.badges{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}",
        ".eval{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px}.eval-cell{border:1px solid #223047;background:#0f172a;border-radius:6px;padding:6px}.eval-cell b{display:block;font-size:13px;color:#e5e7eb}.eval-cell span{display:block;color:#94a3b8;font-size:11px}",
        ".badge{display:inline-flex;align-items:center;border-radius:999px;border:1px solid #334155;padding:3px 8px;font-size:12px;color:#cbd5e1;background:#111827}.badge.ok{border-color:rgba(52,211,153,.42);background:rgba(6,78,59,.34);color:#a7f3d0}.badge.warn{border-color:rgba(251,191,36,.42);background:rgba(120,53,15,.32);color:#fde68a}.badge.muted{color:#94a3b8;background:#0f172a}.badge.split{font-weight:700}.badge.split-train{border-color:rgba(56,189,248,.45);background:rgba(14,116,144,.30);color:#bae6fd}.badge.split-test{border-color:rgba(244,114,182,.45);background:rgba(131,24,67,.30);color:#fbcfe8}",
        ".open{color:#7dd3fc;font-size:12px;font-weight:650;white-space:nowrap}.empty{color:var(--muted);font-size:13px;padding:8px 0}",
        "@media(max-width:720px){.shell{padding:18px}.top{display:block}.metrics{justify-content:flex-start;margin-top:14px}.grid{grid-template-columns:1fr}}",
        "</style>",
        "</head>",
        "<body>",
        "<main class='shell'>",
        "<section class='top'>",
        "<div>",
        "<h1>CEJ 3D Viewer</h1>",
        f"<div class='sub'>输出目录：{_html_escape(os.path.dirname(index_path))}</div>",
        "</div>",
        "<div class='metrics'>",
        f"<div class='metric'><b>{total_cases}</b><span>病例</span></div>",
        f"<div class='metric'><b>{total_teeth}</b><span>牙齿</span></div>",
        f"<div class='metric'><b>{total_manual}</b><span>有手工标注</span></div>",
        f"<div class='metric'><b>{len(train_cases)}/{train_teeth}</b><span>训练集 病例/牙</span></div>",
        f"<div class='metric'><b>{len(test_cases)}/{test_teeth}</b><span>测试集 病例/牙</span></div>",
        "</div>",
        "</section>",
    ]

    summary_parts = [
        f"{category} {count}" for category, count in category_counts.items() if count > 0
    ]
    if summary_parts:
        lines.append("<div class='case-meta'>分类汇总：" + " · ".join(_html_escape(v) for v in summary_parts) + "</div>")

    for case_id in sorted(by_case):
        case_records = sorted(by_case[case_id], key=lambda rec: int(rec["tooth_id"]))
        manual_count = sum(1 for rec in case_records if rec.get("has_manual_points"))
        split_label = case_records[0].get("split_label")
        lines.extend(
            [
                "<section class='case'>",
                "<div class='case-head'>",
                f"<div class='case-title'>{_html_escape(case_id)} {_split_badge(split_label)}</div>",
                f"<div class='case-meta'>{len(case_records)} 颗牙 · {manual_count} 颗有手工标注</div>",
                "</div>",
                "<div class='groups'>",
            ]
        )
        for category in category_order:
            group_records = [rec for rec in case_records if _tooth_category(rec["tooth_id"]) == category]
            if not group_records:
                continue
            lines.append(
                f"<section><div class='group-title'>{_html_escape(category)} "
                f"<span class='count'>{len(group_records)} 颗</span></div><div class='grid'>"
            )
            for rec in group_records:
                rel = rec["rel_path"].replace(os.sep, "/")
                tooth_id = int(rec["tooth_id"])
                root_category = _tooth_root_category(tooth_id)
                mean_eval = rec.get("eval_mean_mm")
                p95_eval = rec.get("eval_p95_mm")
                mean_text = f"{float(mean_eval):.3f}mm" if mean_eval is not None and np.isfinite(mean_eval) else "无"
                p95_text = f"{float(p95_eval):.3f}mm" if p95_eval is not None and np.isfinite(p95_eval) else "无"
                lines.extend(
                    [
                        f"<a class='tooth' href='{_html_escape(rel)}'>",
                        "<div class='tooth-main'>",
                        "<div>",
                        f"<div class='tooth-id'>{tooth_id}</div>",
                        f"<div class='tooth-label'>{_html_escape(category)} · {_html_escape(root_category)}</div>",
                        "</div>",
                        "<div class='open'>打开 viewer</div>",
                        "</div>",
                        "<div class='badges'>",
                        _status_badge("手工标注", bool(rec.get("has_manual_points")), "手工标注", "无手工标注"),
                        _status_badge("GT热图", bool(rec.get("has_gt_heatmap")), "GT热图", "无GT"),
                        _status_badge("预测热图", bool(rec.get("has_pred_heatmap")), "预测热图", "无预测"),
                        _split_badge(rec.get("split_label")),
                        f"<span class='badge'>{_html_escape(root_category)}</span>",
                        "</div>",
                        "<div class='eval'>",
                        f"<div class='eval-cell'><b>{_html_escape(mean_text)}</b><span>标点误差 mean</span></div>",
                        f"<div class='eval-cell'><b>{_html_escape(p95_text)}</b><span>标点误差 p95</span></div>",
                        "</div>",
                        "</a>",
                    ]
                )
            lines.append("</div></section>")
        lines.extend(["</div>", "</section>"])

    lines.extend(["</main>", "</body>", "</html>"])
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_optional_volume(path, dtype):
    if not os.path.exists(path):
        return None
    arr, _, _ = load_volume(path, dtype=dtype)
    return arr


def _load_optional_points(path, tooth_id):
    if not os.path.exists(path):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        points_obj = data.get("points", {}) or {}
        if isinstance(points_obj, dict):
            pts = points_obj.get(str(tooth_id), [])
        elif isinstance(points_obj, list):
            pts = points_obj
        else:
            pts = []
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return np.zeros((0, 3), dtype=np.float32)
        return pts
    except Exception as e:
        logger.warning("failed to load points %s: %s", path, e)
        return np.zeros((0, 3), dtype=np.float32)


def _load_optional_dense_points(path):
    if not os.path.exists(path):
        return None
    try:
        return np.load(path).astype(np.float32)
    except Exception as e:
        logger.warning("failed to load dense curve points %s: %s", path, e)
        return None


def _roi_overlap_slices(origin_a, shape_a, origin_b, shape_b):
    origin_a = np.asarray(origin_a, dtype=np.int32)
    shape_a = np.asarray(shape_a, dtype=np.int32)
    origin_b = np.asarray(origin_b, dtype=np.int32)
    shape_b = np.asarray(shape_b, dtype=np.int32)
    start = np.maximum(origin_a, origin_b)
    end = np.minimum(origin_a + shape_a, origin_b + shape_b)
    if np.any(end <= start):
        return None
    dst_start = start - origin_a
    dst_end = end - origin_a
    src_start = start - origin_b
    src_end = end - origin_b
    dst = tuple(slice(int(dst_start[i]), int(dst_end[i])) for i in range(3))
    src = tuple(slice(int(src_start[i]), int(src_end[i])) for i in range(3))
    return dst, src


def _build_other_teeth_mask(case_tooth_items, current_tdir, current_meta, fmt):
    if current_meta is None:
        return None
    origin = current_meta.get("roi_origin_in_full")
    shape = current_meta.get("roi_shape_full") or current_meta.get("roi_shape")
    if origin is None or shape is None:
        return None
    other = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
    current_path = os.path.abspath(current_tdir)
    for other_tdir, other_meta in case_tooth_items:
        if os.path.abspath(other_tdir) == current_path:
            continue
        other_origin = other_meta.get("roi_origin_in_full")
        other_shape = other_meta.get("roi_shape_full") or other_meta.get("roi_shape")
        if other_origin is None or other_shape is None:
            continue
        overlap = _roi_overlap_slices(origin, shape, other_origin, other_shape)
        if overlap is None:
            continue
        dst, src = overlap
        t_path = os.path.join(other_tdir, f"T_t.{fmt}")
        if not os.path.exists(t_path):
            continue
        try:
            T_other, _, _ = load_volume(t_path, dtype=np.uint8)
        except Exception as e:
            logger.warning("failed to load other tooth mask %s: %s", t_path, e)
            continue
        other[dst] |= (T_other[src] > 0).astype(np.uint8)
    return other if int(other.sum()) > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz_cfg = cfg.get("viz", {})
    enable_2d = bool(viz_cfg.get("enable_2d", False))
    enable_3d = bool(viz_cfg.get("enable_3d", True))

    if not enable_2d and not enable_3d:
        logger.info("viz disabled (enable_2d=false and enable_3d=false); skipping.")
        return

    if enable_2d:
        _ensure_2d_viz_deps()
    if enable_3d:
        _ensure_3d_viz_deps()

    processed_dir = cfg["data"]["processed_dir"]
    infer_dir = os.path.join(cfg["data"]["output_dir"], "infer")
    out_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "viz"))

    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no processed teeth found")
        return

    max_cases = int(viz_cfg.get("max_cases", 5))
    max_teeth = int(viz_cfg.get("max_teeth_per_case", 8))
    num_slices = int(viz_cfg.get("num_slices", 4))
    show_prior_3d = bool(viz_cfg.get("show_prior_3d", True))
    prediction_only = bool(viz_cfg.get("prediction_only", False))
    split_case_labels = _load_split_case_labels(viz_cfg.get("split_summary_path"))
    pseudo_gt_skeleton_from_interp = bool(viz_cfg.get("pseudo_gt_skeleton_from_interp", False))
    pseudo_gt_skeleton_from_heatmap_peak = bool(viz_cfg.get("pseudo_gt_skeleton_from_heatmap_peak", True))
    pseudo_gt_peak_threshold = float(viz_cfg.get("pseudo_gt_peak_threshold", 0.999))
    curve_closed = bool(cfg.get("preprocess", {}).get("curve_closed", True))
    pseudo_gt_skeleton_threshold = float(
        viz_cfg.get("pseudo_gt_skeleton_threshold", cfg["infer"]["threshold_theta"])
    )

    selected = []
    case_counter = {}
    selected_cases = set()
    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]
        split_label = split_case_labels.get(case_id)

        if case_id not in selected_cases:
            if len(selected_cases) >= max_cases:
                continue
            selected_cases.add(case_id)
            case_counter[case_id] = 0

        if case_counter[case_id] >= max_teeth:
            continue

        case_counter[case_id] += 1
        selected.append((tdir, roi_meta, case_id, tooth_id, split_label))

    viewer_records = []
    case_tooth_items = {}
    for tdir, roi_meta, case_id, tooth_id, split_label in selected:
        case_tooth_items.setdefault(case_id, []).append((tdir, roi_meta))

    for tdir, roi_meta, case_id, tooth_id, split_label in selected:
        fmt = cfg["data"]["processed_format"]
        A, spacing, affine = load_volume(os.path.join(tdir, f"A_t.{fmt}"), dtype=np.float32)
        spacing = _ensure_spacing(spacing, affine)
        T, _, _ = load_volume(os.path.join(tdir, f"T_t.{fmt}"), dtype=np.uint8)
        other_teeth_mask = _build_other_teeth_mask(case_tooth_items.get(case_id, []), tdir, roi_meta, fmt)
        H = _load_optional_volume(os.path.join(tdir, f"H_GT.{fmt}"), dtype=np.float32)
        pts = _load_optional_points(os.path.join(tdir, "points.json"), tooth_id)
        manual_dense_curve_points = _load_optional_dense_points(os.path.join(tdir, "curve_dense_points.npy"))

        R = None
        if enable_2d or (enable_3d and show_prior_3d):
            R = compute_geometric_prior(
                A,
                T,
                spacing,
                sigma_z_mm=cfg["infer"]["prior_sigma_z_mm"],
                sigma_s_mm=cfg["infer"]["prior_sigma_s_mm"],
                delta_s_mm=cfg["infer"]["prior_delta_s_mm"],
                z_ignore_ratio=cfg["preprocess"]["z_ignore_ratio"],
                use_gradient=cfg["infer"]["prior_use_gradient"],
                gradient_weight=cfg["infer"]["prior_gradient_weight"],
            )

        h_pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "H_pred.nii.gz")
        c_pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "C_pred.nii.gz")
        c_pred_fit_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "C_pred_fit.nii.gz")
        H_pred = None
        C_pred = None
        C_gt = None
        d = None
        if os.path.exists(h_pred_path) and (os.path.exists(c_pred_fit_path) or os.path.exists(c_pred_path)):
            H_pred, _, _ = load_volume(h_pred_path, dtype=np.float32)
            if os.path.exists(c_pred_fit_path):
                C_pred, _, _ = load_volume(c_pred_fit_path, dtype=np.uint8)
            else:
                C_pred, _, _ = load_volume(c_pred_path, dtype=np.uint8)
            d = compute_distances(pts, C_pred, spacing)

        pred_dense_curve_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "curve_pred_dense_points.npy")
        pred_dense_curve_points = _load_optional_dense_points(pred_dense_curve_path)

        if enable_3d and H is not None and not prediction_only and bool(viz_cfg.get("show_pseudo_gt_skeleton", True)):
            try:
                if (
                    pseudo_gt_skeleton_from_interp
                    and manual_dense_curve_points is not None
                    and len(manual_dense_curve_points) > 0
                ):
                    C_gt = np.zeros_like(T, dtype=np.uint8)
                    _rasterize_polyline_to_mask(C_gt, manual_dense_curve_points, close_loop=curve_closed)
                elif pseudo_gt_skeleton_from_heatmap_peak:
                    C_gt = extract_curve_from_heatmap_peak(
                        H,
                        T,
                        peak_threshold=pseudo_gt_peak_threshold,
                        keep_lcc=False,
                    )
                else:
                    C_gt = extract_curve(H, T, threshold=pseudo_gt_skeleton_threshold)
            except Exception as e:
                logger.warning(
                    "failed to compute pseudo GT skeleton case=%s tooth=%s: %s",
                    case_id,
                    tooth_id,
                    e,
                )
                C_gt = None

        if enable_2d:
            boundary = np.zeros_like(T)
            for z in range(T.shape[2]):
                boundary[:, :, z] = boundary2d(T[:, :, z])
            out_dir_roi = ensure_dir(os.path.join(out_root, "roi", case_id, f"tooth_{tooth_id}"))
            save_overlay(
                A,
                overlay=boundary,
                out_path=os.path.join(out_dir_roi, "roi.png"),
                title="roi",
                num_slices=num_slices,
            )

            out_dir_prior = ensure_dir(os.path.join(out_root, "prior", case_id, f"tooth_{tooth_id}"))
            save_overlay(
                A,
                overlay=R,
                out_path=os.path.join(out_dir_prior, "prior.png"),
                title="prior",
                num_slices=num_slices,
            )

            out_dir_pgt = ensure_dir(os.path.join(out_root, "pseudo_gt", case_id, f"tooth_{tooth_id}"))
            if H is not None:
                save_overlay(
                    A,
                    overlay=H,
                    points=pts,
                    out_path=os.path.join(out_dir_pgt, "pseudo_gt.png"),
                    title="pseudo_gt",
                    num_slices=num_slices,
                )

            if H_pred is not None and C_pred is not None:
                out_dir_inf = ensure_dir(os.path.join(out_root, "infer", case_id, f"tooth_{tooth_id}"))
                save_overlay(
                    A,
                    overlay=H_pred + C_pred,
                    out_path=os.path.join(out_dir_inf, "infer.png"),
                    title="infer",
                    num_slices=num_slices,
                )

                out_dir_err = ensure_dir(os.path.join(out_root, "error", case_id, f"tooth_{tooth_id}"))
                save_error_map(A, pts, d, os.path.join(out_dir_err, "error.png"), num_slices=num_slices)

        if enable_3d:
            out_dir_3d = ensure_dir(os.path.join(out_root, "3d", case_id, f"tooth_{tooth_id}"))
            out_html = os.path.join(out_dir_3d, "viewer.html")
            save_3d_viewer(
                A=A,
                T=T,
                spacing=spacing,
                case_id=case_id,
                tooth_id=tooth_id,
                out_html=out_html,
                cfg=cfg,
                H_gt=None if prediction_only else H,
                points=np.zeros((0, 3), dtype=np.float32) if prediction_only else pts,
                H_pred=H_pred,
                C_pred=C_pred,
                C_gt=C_gt,
                other_teeth_mask=other_teeth_mask,
                manual_dense_curve_points=None if prediction_only else manual_dense_curve_points,
                pred_dense_curve_points=pred_dense_curve_points,
                R=R if show_prior_3d else None,
                distances=d,
                split_label=split_label,
            )
            viewer_records.append(
                {
                    "case_id": case_id,
                    "tooth_id": tooth_id,
                    "rel_path": os.path.relpath(out_html, os.path.join(out_root, "3d")),
                    "has_manual_points": bool(pts is not None and len(pts) > 0),
                    "has_gt_heatmap": bool(H is not None and np.any(H > 0)),
                    "has_pred_heatmap": bool(H_pred is not None and np.any(H_pred > 0)),
                    "eval_mean_mm": float(np.mean(d)) if d is not None and len(d) > 0 and np.isfinite(d).any() else None,
                    "eval_p95_mm": float(np.percentile(d[np.isfinite(d)], 95))
                    if d is not None and len(d) > 0 and np.isfinite(d).any()
                    else None,
                    "split_label": split_label,
                }
            )

        logger.info("viz case=%s tooth=%s", case_id, tooth_id)

    if enable_3d and viewer_records:
        out_3d_root = ensure_dir(os.path.join(out_root, "3d"))
        _write_3d_index(os.path.join(out_3d_root, "index.html"), viewer_records)
        logger.info("3D viewers written to %s", out_3d_root)


if __name__ == "__main__":
    main()
