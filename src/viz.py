import argparse
import glob
import json
import os

import numpy as np
from monai.data.utils import affine_to_spacing
from scipy.spatial import cKDTree

from src.datasets.io import load_volume
from src.datasets.points import get_points_for_tooth, load_points
from src.postprocess.priors import compute_geometric_prior
from src.postprocess.skeleton import extract_curve, extract_curve_from_heatmap_peak
from src.utils.config import ensure_dir, load_config
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


def _make_mesh_trace_from_mask(mask, spacing, name, color, opacity, step=1):
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
        hoverinfo="skip",
    )


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
            hoverinfo="skip",
        )
    )


def _add_curve_points(fig, curve_mask, spacing, max_points=12000, color="#00E5FF", name="推理曲线", marker_size=2):
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
    fig, points_vox, spacing, max_points=20000, color="#F500FF", name="插值曲线", close_loop=True
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
            hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<br>误差=%{marker.color:.3f}mm<extra></extra>",
        )
    )


def _add_fixed_process_legend(fig, rows):
    lines = ["<b>流程图例</b>"]
    for row in rows:
        label = row["label"] if row["present"] else f"{row['label']}（无）"
        lines.append(f"<span style='color:{row['color']};font-weight:700'>■</span> {label}")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        align="left",
        text="<br>".join(lines),
        bgcolor="rgba(0,0,0,0.60)",
        bordercolor="rgba(255,255,255,0.30)",
        borderwidth=1,
        borderpad=6,
        font=dict(size=12, color="#F5F5F5"),
    )


def save_3d_viewer(
    A,
    T,
    H_gt,
    spacing,
    points,
    case_id,
    tooth_id,
    out_html,
    cfg,
    H_pred=None,
    C_pred=None,
    C_gt=None,
    dense_curve_points=None,
    R=None,
    distances=None,
):
    viz_cfg = cfg.get("viz", {})
    max_dim = int(viz_cfg.get("max_render_dim", 96))
    mesh_step = int(viz_cfg.get("mesh_step", 1))
    slice_opacity = float(viz_cfg.get("slice_opacity", 0.85))
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
    prior_thr = float(viz_cfg.get("prior_iso_threshold", 0.50))
    color_tooth_surface = viz_cfg.get("color_tooth_surface", "#64B5F6")
    color_gt_heatmap = viz_cfg.get("color_gt_heatmap", "#FFA726")
    color_pred_heatmap = viz_cfg.get("color_pred_heatmap", "#FF5252")
    color_prior = viz_cfg.get("color_prior", "#7E57C2")
    color_pred_curve = viz_cfg.get("color_pred_curve", "#00E5FF")
    color_dense_interp_curve = viz_cfg.get("color_dense_interp_curve", "#F500FF")
    color_pseudo_gt_skeleton = viz_cfg.get("color_pseudo_gt_skeleton", "#76FF03")
    color_gt_points = viz_cfg.get("color_gt_points", "#FFD600")

    step = _downsample_step(A.shape, max_dim)
    A_ds = _downsample_volume(A, step)
    T_ds = _downsample_volume(T, step)
    H_gt_ds = _downsample_volume(H_gt, step)
    H_pred_ds = _downsample_volume(H_pred, step) if H_pred is not None else None
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
    )
    if tooth_mesh is not None:
        fig.add_trace(tooth_mesh)

    gt_mesh = _make_mesh_trace_from_mask(
        (H_gt_ds >= heat_thr).astype(np.uint8),
        spacing_ds,
        name=f"伪GT热图 (>{heat_thr:.2f})",
        color=color_gt_heatmap,
        opacity=heatmap_opacity,
        step=mesh_step,
    )
    if gt_mesh is not None:
        fig.add_trace(gt_mesh)

    if H_pred_ds is not None:
        pred_mesh = _make_mesh_trace_from_mask(
            (H_pred_ds >= pred_thr).astype(np.uint8),
            spacing_ds,
            name=f"推理热图 (>{pred_thr:.2f})",
            color=color_pred_heatmap,
            opacity=heatmap_opacity,
            step=mesh_step,
        )
        if pred_mesh is not None:
            fig.add_trace(pred_mesh)

    if R_ds is not None:
        prior_mesh = _make_mesh_trace_from_mask(
            (R_ds >= prior_thr).astype(np.uint8),
            spacing_ds,
            name=f"几何先验 (>{prior_thr:.2f})",
            color=color_prior,
            opacity=prior_opacity,
            step=mesh_step,
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
        )

    if show_dense_interp_curve and dense_curve_points is not None:
        _add_dense_interp_curve(
            fig,
            dense_curve_points,
            spacing,
            max_points=dense_interp_curve_max_points,
            color=color_dense_interp_curve,
            name="插值曲线",
            close_loop=dense_interp_curve_closed,
        )

    if show_pseudo_gt_skeleton and C_gt is not None:
        _add_curve_points(
            fig,
            C_gt,
            spacing,
            max_points=pseudo_gt_skeleton_max_points,
            color=color_pseudo_gt_skeleton,
            name="伪GT骨架",
        )

    has_gt_points = points is not None and len(points) > 0
    has_interp_curve = bool(show_dense_interp_curve and dense_curve_points is not None and len(dense_curve_points) > 0)
    has_gt_heatmap = bool(np.any(H_gt_ds >= heat_thr))
    has_gt_skeleton = bool(show_pseudo_gt_skeleton and C_gt is not None and np.any(C_gt > 0))
    has_pred_heatmap = bool(H_pred_ds is not None and np.any(H_pred_ds >= pred_thr))
    has_pred_curve = bool(C_pred is not None and np.any(C_pred > 0))

    _add_gt_points(
        fig,
        points,
        spacing,
        distances=distances,
        color=color_gt_points,
        name="标注点",
        use_error_colormap=use_error_colormap_for_gt_points,
    )

    process_legend_rows = [
        {"label": "1 标注点", "color": color_gt_points, "present": has_gt_points},
        {"label": "2 插值曲线", "color": color_dense_interp_curve, "present": has_interp_curve},
        {"label": "3 伪GT热图", "color": color_gt_heatmap, "present": has_gt_heatmap},
        {"label": "4 伪GT骨架", "color": color_pseudo_gt_skeleton, "present": has_gt_skeleton},
        {"label": "5 推理热图", "color": color_pred_heatmap, "present": has_pred_heatmap},
        {"label": "6 推理曲线", "color": color_pred_curve, "present": has_pred_curve},
    ]
    _add_fixed_process_legend(fig, process_legend_rows)

    fig.update_layout(
        title=f"CEJ 3D可视化 | 病例={case_id} 牙位={tooth_id}",
        template="plotly_dark",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            dragmode="turntable",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=42, b=0),
    )
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def _write_3d_index(index_path, records):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>CEJ 3D可视化索引</title></head><body>",
        "<h2>CEJ 3D可视化索引</h2>",
        "<ul>",
    ]
    for rec in records:
        rel = rec["rel_path"].replace(os.sep, "/")
        lines.append(
            f"<li>病例={rec['case_id']} 牙位={rec['tooth_id']} "
            f"<a href='{rel}'>打开viewer</a></li>"
        )
    lines.extend(["</ul>", "</body></html>"])
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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

        if case_id not in selected_cases:
            if len(selected_cases) >= max_cases:
                continue
            selected_cases.add(case_id)
            case_counter[case_id] = 0

        if case_counter[case_id] >= max_teeth:
            continue

        case_counter[case_id] += 1
        selected.append((tdir, roi_meta, case_id, tooth_id))

    viewer_records = []
    for tdir, roi_meta, case_id, tooth_id in selected:
        fmt = cfg["data"]["processed_format"]
        A, spacing, affine = load_volume(os.path.join(tdir, f"A_t.{fmt}"), dtype=np.float32)
        spacing = _ensure_spacing(spacing, affine)
        T, _, _ = load_volume(os.path.join(tdir, f"T_t.{fmt}"), dtype=np.uint8)
        H, _, _ = load_volume(os.path.join(tdir, f"H_GT.{fmt}"), dtype=np.float32)

        points_data = load_points(os.path.join(tdir, "points.json"))
        pts = get_points_for_tooth(points_data, tooth_id)
        dense_curve_points = None
        dense_curve_path = os.path.join(tdir, "curve_dense_points.npy")
        if os.path.exists(dense_curve_path):
            try:
                dense_curve_points = np.load(dense_curve_path).astype(np.float32)
            except Exception as e:
                logger.warning(
                    "failed to load dense curve points case=%s tooth=%s: %s",
                    case_id,
                    tooth_id,
                    e,
                )

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

        if enable_3d and bool(viz_cfg.get("show_pseudo_gt_skeleton", True)):
            try:
                if pseudo_gt_skeleton_from_interp and dense_curve_points is not None and len(dense_curve_points) > 0:
                    C_gt = np.zeros_like(T, dtype=np.uint8)
                    _rasterize_polyline_to_mask(C_gt, dense_curve_points, close_loop=curve_closed)
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
                H_gt=H,
                spacing=spacing,
                points=pts,
                case_id=case_id,
                tooth_id=tooth_id,
                out_html=out_html,
                cfg=cfg,
                H_pred=H_pred,
                C_pred=C_pred,
                C_gt=C_gt,
                dense_curve_points=dense_curve_points,
                R=R if show_prior_3d else None,
                distances=d,
            )
            viewer_records.append(
                {
                    "case_id": case_id,
                    "tooth_id": tooth_id,
                    "rel_path": os.path.relpath(out_html, os.path.join(out_root, "3d")),
                }
            )

        logger.info("viz case=%s tooth=%s", case_id, tooth_id)

    if enable_3d and viewer_records:
        out_3d_root = ensure_dir(os.path.join(out_root, "3d"))
        _write_3d_index(os.path.join(out_3d_root, "index.html"), viewer_records)
        logger.info("3D viewers written to %s", out_3d_root)


if __name__ == "__main__":
    main()
