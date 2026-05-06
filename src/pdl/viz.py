import argparse
import html
import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.pdl.algorithm import STATUS_OK, trimesh
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("pdl_viz")

DEFAULT_CONFIG = "/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/pdl/configs/pdl_runtime.yaml"


STATE_LABELS = {
    0: "other",
    1: "attached",
    2: "boundary_candidate",
    3: "primary_cycle_member",
}
STATE_COLORS = {
    0: "#bdbdbd",
    1: "#1f77b4",
    2: "#d62728",
    3: "#2ca02c",
}


def _ensure_plotly():
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(f"plotly is required for pdl html visualization: {exc}")
    return go


def _format_float(value, digits=4):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_json(path, label, required=False):
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"missing required {label}: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        if required:
            raise RuntimeError(f"failed to parse {label} at {path}: {exc}")
        return None


def _load_points(path, label, required=False):
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"missing required {label}: {path}")
        return np.zeros((0, 3), dtype=np.float32)
    try:
        pts = np.load(path).astype(np.float32)
    except Exception as exc:
        if required:
            raise RuntimeError(f"failed to load {label} at {path}: {exc}")
        return np.zeros((0, 3), dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        if required:
            raise RuntimeError(f"invalid shape for {label}: expected [N,3], got {tuple(pts.shape)} at {path}")
        return np.zeros((0, 3), dtype=np.float32)
    return pts


def _load_curve_bundle(path):
    if not os.path.exists(path):
        return []
    try:
        curves = np.load(path)
    except Exception:
        return []

    out = []
    for key in sorted(curves.files):
        pts = np.asarray(curves[key], dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            out.append(pts)
    return out


def _load_mesh(path, label, required=False):
    if trimesh is None:
        raise RuntimeError("trimesh is required for pdl visualization")
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"missing required {label}: {path}")
        return None
    try:
        mesh = trimesh.load(path, process=False)
    except Exception as exc:
        if required:
            raise RuntimeError(f"failed to load {label} at {path}: {exc}")
        return None

    if not hasattr(mesh, "vertices"):
        if required:
            raise RuntimeError(f"invalid {label}: no vertices at {path}")
        return None

    vertices = np.asarray(mesh.vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        if required:
            raise RuntimeError(f"invalid {label}: empty or malformed vertices at {path}")
        return None
    return mesh


def _load_vertex_state(path, expected_n, required=False):
    if not os.path.exists(path):
        if required:
            raise RuntimeError(f"missing required boundary vertex_state: {path}")
        return np.zeros((0,), dtype=np.uint8)
    try:
        state = np.load(path).astype(np.uint8)
    except Exception as exc:
        if required:
            raise RuntimeError(f"failed to load boundary vertex_state at {path}: {exc}")
        return np.zeros((0,), dtype=np.uint8)

    if state.ndim != 1:
        raise RuntimeError(f"invalid boundary vertex_state shape {tuple(state.shape)} at {path}, expected [N]")
    if int(expected_n) > 0 and state.shape[0] != int(expected_n):
        raise RuntimeError(
            f"boundary vertex_state size mismatch at {path}: expected {int(expected_n)}, got {int(state.shape[0])}"
        )
    return state


def _mesh_trace(go, mesh, name, color, opacity=0.30):
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)

    if faces.shape[0] == 0:
        return go.Scatter3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            mode="markers",
            marker=dict(size=2.0, color=color, opacity=opacity),
            name=name,
        )
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=float(opacity),
        name=name,
        hoverinfo="skip",
    )


def _point_trace(go, points, name, color, size=3.0, mode="markers"):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode=mode,
        marker=dict(size=float(size), color=color, opacity=0.95),
        line=dict(color=color, width=5),
        name=name,
    )


def _state_traces(go, tooth_mesh, vertex_state, max_points=10000):
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    state = np.asarray(vertex_state, dtype=np.uint8)

    traces = []
    for sid in [3, 2, 1, 0]:
        idx = np.where(state == sid)[0]
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > int(max_points):
            pick = np.linspace(0, idx.shape[0] - 1, int(max_points), dtype=np.int64)
            idx = idx[pick]
        pts = verts[idx]
        trace = _point_trace(
            go,
            pts,
            name=f"Boundary state: {STATE_LABELS.get(sid, sid)}",
            color=STATE_COLORS.get(sid, "#999999"),
            size=2.6 if sid in (0, 1) else 3.2,
            mode="markers",
        )
        if trace is not None:
            traces.append(trace)
    return traces


def _state_summary(vertex_state):
    counts = Counter(np.asarray(vertex_state, dtype=np.uint8).tolist())
    rows = []
    for sid in [3, 2, 1, 0]:
        rows.append(f"<li>{STATE_LABELS.get(sid, sid)}: {int(counts.get(sid, 0))}</li>")
    return "\n".join(rows)


def _sidebar_html(case_id, tooth_id, area_payload, meta, vertex_state):
    area_payload = area_payload or {}
    metrics = area_payload.get("metrics", {})
    status = str(area_payload.get("status", "unknown"))
    area = area_payload.get("anchorage_area_mm2", None)

    thresholds = meta.get("thresholds", {}) if isinstance(meta, dict) else {}
    boundary_stats = meta.get("boundary_stats", {}) if isinstance(meta, dict) else {}
    if not boundary_stats and isinstance(meta, dict):
        boundary_stats = meta.get("bfs_stats", {})
    curve_stats = meta.get("curve_stats", {}) if isinstance(meta, dict) else {}

    lines = [
        f"<p><b>case</b>: {html.escape(str(case_id))}<br/><b>tooth</b>: {html.escape(str(tooth_id))}</p>",
        f"<p><b>status</b>: {html.escape(status)}<br/><b>anchorage_area_mm2</b>: {_format_float(area)}</p>",
        "<h4>Thresholds</h4>",
        (
            f"<p><b>epsilon_mm</b>: {_format_float(thresholds.get('epsilon_mm', metrics.get('epsilon_mm')))}<br/>"
            f"<b>attach_threshold</b>: {_format_float(thresholds.get('attach_threshold', metrics.get('attach_threshold')))}</p>"
        ),
        "<h4>Boundary Metrics</h4>",
        (
            f"<p><b>n_attached_vertices</b>: {boundary_stats.get('n_attached_vertices', metrics.get('n_attached_vertices', 'N/A'))}<br/>"
            f"<b>n_boundary_candidates</b>: {boundary_stats.get('n_boundary_candidates', metrics.get('n_boundary_candidates', 'N/A'))}<br/>"
            f"<b>n_boundary_points</b>: {metrics.get('n_boundary_points', 'N/A')}</p>"
        ),
        "<h4>Cycle Metrics</h4>",
        (
            f"<p><b>n_cycles</b>: {curve_stats.get('n_cycles', metrics.get('n_cycles', 'N/A'))}<br/>"
            f"<b>primary_cycle_index</b>: {curve_stats.get('primary_cycle_index', 'N/A')}</p>"
        ),
        "<h4>Attached Field</h4>",
        (
            f"<p><b>mean_distance_mm_attached</b>: {_format_float(metrics.get('mean_distance_mm_attached'))}<br/>"
            f"<b>p95_distance_mm_attached</b>: {_format_float(metrics.get('p95_distance_mm_attached'))}<br/>"
            f"<b>mean_parallelism_attached</b>: {_format_float(metrics.get('mean_parallelism_attached'))}</p>"
        ),
        "<h4>State Counts</h4>",
        f"<ul>{_state_summary(vertex_state)}</ul>",
    ]

    err = None
    if isinstance(meta, dict):
        err = meta.get("error")
    if err:
        lines.append(f"<p><b>error</b>: {html.escape(str(err))}</p>")

    lines.append(
        "<h4>Default Layers</h4>"
        "<ul>"
        "<li>tooth mesh</li>"
        "<li>bone mesh</li>"
        "<li>PDL boundary</li>"
        "<li>PCA axis</li>"
        "<li>boundary state coloring</li>"
        "</ul>"
    )

    return "\n".join(lines)


def _write_viewer(path, fig_html, sidebar_html, title):
    html_body = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d9e2ec;
      --text: #102a43;
      --muted: #627d98;
      --accent: #0b7285;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); }}
    .wrap {{ display: grid; grid-template-columns: minmax(420px, 1fr) 360px; min-height: 100vh; gap: 0; }}
    .main {{ padding: 10px 10px 10px 14px; }}
    .side {{ border-left: 1px solid var(--line); padding: 14px 14px 18px 14px; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%); overflow: auto; }}
    .title {{ margin: 0 0 8px 0; font-size: 18px; color: var(--accent); letter-spacing: 0.2px; }}
    h4 {{ margin: 12px 0 6px 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--muted); }}
    p, li {{ line-height: 1.4; font-size: 13px; }}
    ul {{ margin: 6px 0 8px 18px; padding: 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; box-shadow: 0 3px 12px rgba(16, 42, 67, 0.06); }}
    @media (max-width: 980px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      .side {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"main\">{fig_html}</div>
    <aside class=\"side\">
      <div class=\"card\">
        <h3 class=\"title\">{html.escape(title)}</h3>
        {sidebar_html}
      </div>
    </aside>
  </div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)


def _write_index(path, records):
    total = len(records)
    ok = sum(1 for r in records if r.get("status") == STATUS_OK)
    area_values = [float(r["anchorage_area_mm2"]) for r in records if r.get("anchorage_area_mm2") is not None]
    area_mean = _format_float(float(np.mean(np.asarray(area_values, dtype=np.float32))) if area_values else None)

    rows = []
    for r in records:
        rel = html.escape(r["relpath"].replace("\\", "/"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['case_id']))}</td>"
            f"<td>{html.escape(str(r['tooth_id']))}</td>"
            f"<td>{html.escape(str(r.get('status', 'unknown')))}</td>"
            f"<td>{_format_float(r.get('anchorage_area_mm2'))}</td>"
            f"<td>{html.escape(str(r.get('n_boundary_points', 'N/A')))}</td>"
            f"<td><a href='{rel}'>viewer.html</a></td>"
            "</tr>"
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html_body = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>PDL Boundary 3D Index</title>
  <style>
    :root {{
      --bg: #f2f5f9;
      --panel: #ffffff;
      --line: #d9e2ec;
      --text: #102a43;
      --muted: #627d98;
      --accent: #0b7285;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: radial-gradient(circle at top right, #ebf7ff 0%, var(--bg) 45%); color: var(--text); font-family: "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif; }}
    .container {{ max-width: 1080px; margin: 0 auto; padding: 28px 20px 40px 20px; }}
    h1 {{ margin: 0 0 8px 0; color: var(--accent); font-size: 28px; letter-spacing: 0.2px; }}
    .meta {{ color: var(--muted); margin-bottom: 16px; font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; box-shadow: 0 6px 18px rgba(16, 42, 67, 0.06); }}
    .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
    .v {{ font-size: 24px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; box-shadow: 0 6px 18px rgba(16, 42, 67, 0.06); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; font-size: 14px; }}
    th {{ background: #f8fbff; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; font-size: 12px; }}
    tr:last-child td {{ border-bottom: 0; }}
    a {{ color: #0b5ed7; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 760px) {{
      .stats {{ grid-template-columns: 1fr; }}
      th, td {{ padding: 9px; font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>PDL Boundary 3D Index</h1>
    <p class=\"meta\">Generated at {generated_at}</p>
    <section class=\"stats\">
      <article class=\"card\"><div class=\"k\">Viewers</div><div class=\"v\">{total}</div></article>
      <article class=\"card\"><div class=\"k\">Status OK</div><div class=\"v\">{ok}</div></article>
      <article class=\"card\"><div class=\"k\">Mean Area (mm2)</div><div class=\"v\">{area_mean}</div></article>
    </section>
    <table>
      <thead>
        <tr>
          <th>Case</th>
          <th>Tooth</th>
          <th>Status</th>
          <th>Anchorage Area (mm2)</th>
          <th>Boundary Points</th>
          <th>Viewer</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)


def _load_tooth_artifacts(tooth_dir):
    tooth_mesh = _load_mesh(os.path.join(tooth_dir, "tooth_mesh.ply"), "tooth mesh", required=True)
    bone_mesh = _load_mesh(os.path.join(tooth_dir, "bone_mesh.ply"), "bone mesh", required=True)
    axis_cloud = _load_mesh(os.path.join(tooth_dir, "tooth_axis.ply"), "tooth axis", required=True)
    area_payload = _load_json(os.path.join(tooth_dir, "anchorage_area.json"), "anchorage_area", required=True)
    meta = _load_json(os.path.join(tooth_dir, "PDL_meta.json"), "PDL_meta", required=True)
    smooth_curves = _load_curve_bundle(os.path.join(tooth_dir, "smooth_curves_3d.npz"))
    curve_stats = meta.get("curve_stats", {}) if isinstance(meta, dict) else {}
    primary_cycle_index = int(curve_stats.get("primary_cycle_index", 0)) if smooth_curves else 0

    boundary_curve = _load_points(
        os.path.join(tooth_dir, "pdl_boundary_curve.npy"),
        "pdl boundary curve",
        required=not bool(smooth_curves),
    )
    if boundary_curve.shape[0] == 0 and smooth_curves:
        if 0 <= primary_cycle_index < len(smooth_curves):
            boundary_curve = np.asarray(smooth_curves[primary_cycle_index], dtype=np.float32)
        else:
            boundary_curve = np.asarray(smooth_curves[0], dtype=np.float32)

    tooth_vertices = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    vertex_state = _load_vertex_state(
        os.path.join(tooth_dir, "vertex_state.npy"),
        expected_n=tooth_vertices.shape[0],
        required=True,
    )

    if str(area_payload.get("status", "unknown")) != STATUS_OK:
        raise RuntimeError(
            "anchorage_area status is not ok, cannot build default-layer viewer: "
            f"status={area_payload.get('status')}"
        )

    pdl_region_mesh = _load_mesh(os.path.join(tooth_dir, "pdl_region_mesh.ply"), "pdl region mesh", required=False)
    seed_pts = _load_points(os.path.join(tooth_dir, "seed_points_mm.npy"), "seed points", required=False)
    cand_pts = _load_points(
        os.path.join(tooth_dir, "boundary_candidate_points_mm.npy"),
        "boundary candidate points",
        required=False,
    )

    return {
        "tooth_mesh": tooth_mesh,
        "bone_mesh": bone_mesh,
        "axis_points": np.asarray(axis_cloud.vertices, dtype=np.float32),
        "boundary_curve": boundary_curve,
        "smooth_curves": smooth_curves,
        "primary_cycle_index": primary_cycle_index,
        "vertex_state": vertex_state,
        "area_payload": area_payload,
        "meta": meta,
        "pdl_region_mesh": pdl_region_mesh,
        "seed_pts": seed_pts,
        "cand_pts": cand_pts,
    }


def _build_figure(go, artifacts, max_state_points):
    fig = go.Figure()

    smooth_curves = list(artifacts.get("smooth_curves", []))
    primary_cycle_index = int(artifacts.get("primary_cycle_index", 0))
    if smooth_curves and 0 <= primary_cycle_index < len(smooth_curves):
        primary_curve = np.asarray(smooth_curves[primary_cycle_index], dtype=np.float32)
    else:
        primary_curve = np.asarray(artifacts["boundary_curve"], dtype=np.float32)

    traces = [
        _mesh_trace(go, artifacts["tooth_mesh"], "tooth mesh", "#d8b365", opacity=0.33),
        _mesh_trace(go, artifacts["bone_mesh"], "bone mesh", "#5ab4ac", opacity=0.22),
        _point_trace(go, primary_curve, "PDL boundary", "#d62728", size=3.6, mode="lines+markers"),
        _point_trace(go, artifacts["axis_points"], "PCA axis", "#1f77b4", size=4.0, mode="lines+markers"),
    ]

    if smooth_curves:
        for curve_idx, curve in enumerate(smooth_curves):
            if curve_idx == primary_cycle_index:
                continue
            traces.append(
                _point_trace(
                    go,
                    curve,
                    f"PDL boundary extra {curve_idx}",
                    "#ff7f50",
                    size=2.6,
                    mode="lines",
                )
            )

    if artifacts["pdl_region_mesh"] is not None:
        traces.append(_mesh_trace(go, artifacts["pdl_region_mesh"], "pdl region mesh", "#f46d43", opacity=0.38))

    seed_trace = _point_trace(go, artifacts["seed_pts"], "seed points", "#2ca02c", size=3.0)
    if seed_trace is not None:
        traces.append(seed_trace)
    cand_trace = _point_trace(go, artifacts["cand_pts"], "boundary candidates", "#ff7f0e", size=2.8)
    if cand_trace is not None:
        traces.append(cand_trace)

    traces.extend(
        _state_traces(
            go,
            artifacts["tooth_mesh"],
            artifacts["vertex_state"],
            max_points=max_state_points,
        )
    )

    for trace in traces:
        if trace is not None:
            fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
            aspectmode="data",
            bgcolor="#f8fbff",
        ),
        paper_bgcolor="#f6f8fb",
        plot_bgcolor="#f6f8fb",
        legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.75)"),
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    go = _ensure_plotly()
    cfg = load_config(args.config)
    viz_cfg = cfg.get("pdl_viz", {})
    if not bool(viz_cfg.get("enable", True)):
        logger.info("pdl_viz.enable=false, skip html visualization generation")
        return

    infer_root = os.path.join(cfg["data"]["output_dir"], "infer")
    viz_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "viz", "3d"))

    if not os.path.isdir(infer_root):
        raise RuntimeError(f"pdl infer dir not found: {infer_root}")

    max_cases = int(viz_cfg.get("max_cases", 5))
    max_teeth = int(viz_cfg.get("max_teeth_per_case", 8))
    max_state_points = int(viz_cfg.get("max_state_points", 8000))
    fail_fast = bool(viz_cfg.get("fail_fast", True))

    case_ids = sorted([d for d in os.listdir(infer_root) if os.path.isdir(os.path.join(infer_root, d))])
    if not case_ids:
        raise RuntimeError(f"no cases found under infer root: {infer_root}")
    if max_cases > 0:
        case_ids = case_ids[:max_cases]

    records = []
    errors = []

    for case_id in case_ids:
        tooth_root = os.path.join(infer_root, case_id)
        tooth_dirs = sorted(
            [
                d
                for d in os.listdir(tooth_root)
                if d.startswith("tooth_") and os.path.isdir(os.path.join(tooth_root, d))
            ]
        )
        if max_teeth > 0:
            tooth_dirs = tooth_dirs[:max_teeth]

        for tooth_name in tooth_dirs:
            tooth_id = tooth_name.replace("tooth_", "")
            tooth_dir = os.path.join(tooth_root, tooth_name)

            try:
                artifacts = _load_tooth_artifacts(tooth_dir)
                fig = _build_figure(go, artifacts, max_state_points=max_state_points)
                fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

                sidebar_html = _sidebar_html(
                    case_id=case_id,
                    tooth_id=tooth_id,
                    area_payload=artifacts["area_payload"],
                    meta=artifacts["meta"],
                    vertex_state=artifacts["vertex_state"],
                )

                out_dir = ensure_dir(os.path.join(viz_root, case_id, tooth_name))
                viewer_path = os.path.join(out_dir, "viewer.html")
                _write_viewer(viewer_path, fig_html, sidebar_html, f"PDL Viewer | case={case_id} tooth={tooth_id}")

                rel = os.path.relpath(viewer_path, viz_root)
                records.append(
                    {
                        "case_id": case_id,
                        "tooth_id": tooth_id,
                        "relpath": rel,
                        "status": artifacts["area_payload"].get("status"),
                        "anchorage_area_mm2": artifacts["area_payload"].get("anchorage_area_mm2"),
                        "n_boundary_points": artifacts["area_payload"].get("metrics", {}).get("n_boundary_points"),
                    }
                )
                logger.info("pdl viz generated case=%s tooth=%s", case_id, tooth_id)
            except Exception as exc:
                message = f"pdl viz failed case={case_id} tooth={tooth_id}: {exc}"
                logger.error(message)
                errors.append(message)
                if fail_fast:
                    raise RuntimeError(message) from exc

    if not records:
        raise RuntimeError(f"no pdl viewers generated under {infer_root}")

    index_path = os.path.join(viz_root, "index.html")
    _write_index(index_path, records)
    logger.info("pdl viz index -> %s", index_path)

    if errors:
        err_path = os.path.join(viz_root, "viz_generation_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump({"errors": errors}, f, ensure_ascii=False, indent=2)
        logger.warning("pdl viz generated with %d errors, details -> %s", len(errors), err_path)


if __name__ == "__main__":
    main()
