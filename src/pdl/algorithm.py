import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from skimage import measure, morphology

try:
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


DEFAULT_PDL_CFG = {
    "min_component_voxels": 64,
    "taubin_iterations": 15,
    "taubin_lambda": 0.5,
    "taubin_mu": -0.53,
    "epsilon_mm": 0.35,
    "attach_threshold": 0.85,
    "seed_z_quantile_low": 0.05,
    "seed_z_quantile_high": 0.45,
    "bfs_upward_tolerance_mm": 0.0,
    "min_seed_vertices": 24,
    "min_boundary_vertices": 24,
    "boundary_resample_points": 1000,
    "min_cycle_vertices": 10,
    "spline_smooth_s": 2.0,
    "axis_length_mm": 15.0,
    "fail_on_geometry_error": True,
}


STATUS_OK = "ok"
STATUS_GEOMETRY_FAILED = "geometry_failed"
STATUS_SMOOTH_FAILED = "smooth_failed"
STATUS_NORMAL_FAILED = "normal_failed"
STATUS_NO_SEED = "no_seed"
STATUS_NO_BOUNDARY = "no_boundary"
STATUS_AREA_FAILED = "area_failed"


@dataclass
class PDLGeometry:
    tooth_mesh_raw: "trimesh.Trimesh"
    bone_mesh_raw: "trimesh.Trimesh"
    tooth_mesh: "trimesh.Trimesh"
    bone_mesh: "trimesh.Trimesh"
    frame: Dict[str, np.ndarray]
    axis_points_mm: np.ndarray


def _require_deps():
    if trimesh is None:
        raise RuntimeError("trimesh is required for pdl boundary pipeline")
    if o3d is None:
        raise RuntimeError("open3d is required for Taubin smoothing")


def _safe_unit(vec: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(vec))
    if n <= 1e-8:
        return None
    return vec / n


def _require_trimesh():
    if trimesh is None:
        raise RuntimeError("trimesh is required for pdl boundary pipeline")


def _orient_axis_root_to_crown(centered_mm: np.ndarray, axis: np.ndarray) -> np.ndarray:
    proj = centered_mm @ axis
    if proj.size < 16:
        return axis

    q_low = np.percentile(proj, 15)
    q_high = np.percentile(proj, 85)
    low_mask = proj <= q_low
    high_mask = proj >= q_high
    if int(low_mask.sum()) < 4 or int(high_mask.sum()) < 4:
        return axis

    radial_vec = centered_mm - np.outer(proj, axis)
    radial = np.linalg.norm(radial_vec, axis=1)
    low_w = float(np.mean(radial[low_mask]))
    high_w = float(np.mean(radial[high_mask]))

    if high_w < low_w:
        return -axis
    return axis


def compute_local_frame(vertices_mm: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    verts = np.asarray(vertices_mm, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[0] < 8 or verts.shape[1] != 3:
        return None

    centroid = np.mean(verts, axis=0)
    centered = verts - centroid[None, :]
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = _safe_unit(axis)
    if axis is None:
        return None

    axis = _orient_axis_root_to_crown(centered, axis)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    u = ref - np.dot(ref, axis) * axis
    u = _safe_unit(u)
    if u is None:
        return None

    v = np.cross(axis, u)
    v = _safe_unit(v)
    if v is None:
        return None

    return {
        "origin_mm": centroid.astype(np.float32),
        "a_t": axis.astype(np.float32),
        "u_t": u.astype(np.float32),
        "v_t": v.astype(np.float32),
    }


def mask_to_mesh(mask: np.ndarray, spacing_xyz: Tuple[float, float, float], min_component_voxels: int = 64):
    _require_trimesh()

    m = np.asarray(mask) > 0
    if int(m.sum()) < int(min_component_voxels):
        return None

    m = morphology.remove_small_objects(m, min_size=int(min_component_voxels))
    if int(m.sum()) < int(min_component_voxels):
        return None

    if min(m.shape) < 2:
        return None

    try:
        verts, faces, _, _ = measure.marching_cubes(
            m.astype(np.float32),
            level=0.5,
            spacing=tuple(float(v) for v in spacing_xyz),
        )
    except Exception:
        return None

    if verts.size == 0 or faces.size == 0:
        return None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def build_axis_points(frame: Dict[str, np.ndarray], axis_length_mm: float) -> np.ndarray:
    origin = np.asarray(frame["origin_mm"], dtype=np.float32)
    axis = np.asarray(frame["a_t"], dtype=np.float32)
    axis = _safe_unit(axis)
    if axis is None:
        return np.zeros((0, 3), dtype=np.float32)
    half = float(axis_length_mm) * 0.5
    p1 = origin - axis * half
    p2 = origin + axis * half
    return np.stack([p1, p2], axis=0).astype(np.float32)


def _mesh_stats(mesh_obj) -> Dict[str, object]:
    verts = np.asarray(getattr(mesh_obj, "vertices", np.zeros((0, 3))), dtype=np.float32)
    faces = np.asarray(getattr(mesh_obj, "faces", np.zeros((0, 3))), dtype=np.int32)
    vol = None
    try:
        vol = float(abs(mesh_obj.volume))
    except Exception:
        vol = None
    return {
        "n_vertices": int(verts.shape[0]),
        "n_faces": int(faces.shape[0]),
        "is_watertight": bool(getattr(mesh_obj, "is_watertight", False)),
        "volume_mm3": vol,
    }


def _trimesh_to_o3d(mesh_obj):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_obj.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_obj.faces, dtype=np.int32))
    m.compute_vertex_normals()
    return m


def _o3d_to_trimesh(mesh_obj):
    verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
    faces = np.asarray(mesh_obj.triangles, dtype=np.int32)
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out.remove_unreferenced_vertices()
    try:
        out.remove_degenerate_faces()
        out.remove_duplicate_faces()
    except Exception:
        pass
    try:
        out.fix_normals()
    except Exception:
        pass
    return out


def taubin_smooth_trimesh(mesh_obj, iterations: int, lam: float, mu: float):
    _require_deps()
    mesh_o3d = _trimesh_to_o3d(mesh_obj)
    smoothed = mesh_o3d.filter_smooth_taubin(
        number_of_iterations=int(iterations),
        lambda_filter=float(lam),
        mu=float(mu),
    )
    smoothed.compute_vertex_normals()
    return _o3d_to_trimesh(smoothed)


def _compute_vertex_normals(mesh_obj):
    verts = np.asarray(mesh_obj.vertices, dtype=np.float32)
    normals = np.asarray(mesh_obj.vertex_normals, dtype=np.float32)
    if normals.shape != verts.shape:
        raise RuntimeError("vertex normals shape mismatch")
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(n <= 1e-8):
        raise RuntimeError("invalid zero normals")
    return normals / n


def _build_adjacency(n_vertices: int, faces: np.ndarray):
    adj = [set() for _ in range(int(n_vertices))]
    for tri in np.asarray(faces, dtype=np.int32):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adj[i].add(j)
        adj[i].add(k)
        adj[j].add(i)
        adj[j].add(k)
        adj[k].add(i)
        adj[k].add(j)
    return adj


def _periodic_interp(theta_q, theta_ref, values_ref):
    theta_q = np.mod(np.asarray(theta_q, dtype=np.float32), 2.0 * np.pi)
    theta_ref = np.mod(np.asarray(theta_ref, dtype=np.float32), 2.0 * np.pi)
    values_ref = np.asarray(values_ref, dtype=np.float32)

    order = np.argsort(theta_ref)
    theta_s = theta_ref[order]
    val_s = values_ref[order]

    if theta_s.shape[0] < 2:
        return np.full_like(theta_q, float(val_s[0]) if val_s.shape[0] > 0 else 0.0)

    theta_ext = np.concatenate([theta_s, theta_s[:1] + 2.0 * np.pi], axis=0)
    val_ext = np.concatenate([val_s, val_s[:1]], axis=0)
    return np.interp(theta_q, theta_ext, val_ext).astype(np.float32)


def _curve_length_mm(points_mm: np.ndarray) -> float:
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-6) -> bool:
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)

    if abs(o1) <= eps and _on_segment2d(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment2d(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment2d(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment2d(c, d, b, eps):
        return True

    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _has_self_intersection_2d(poly_xy: np.ndarray) -> bool:
    pts = np.asarray(poly_xy, dtype=np.float32)
    n = int(pts.shape[0])
    if n < 4:
        return False

    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        for j in range(i + 1, n):
            # Adjacent segments share vertices and are expected to meet.
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == n - 1:
                continue

            c = pts[j]
            d = pts[(j + 1) % n]
            if _segments_intersect2d(a, b, c, d):
                return True
    return False


def _validate_closed_curve(points_mm: np.ndarray, frame: Dict[str, np.ndarray]) -> bool:
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 8:
        return False

    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    if np.any(seg_len <= 1e-6):
        return False

    close_dist = float(np.linalg.norm(pts[0] - pts[-1]))
    med_step = float(np.median(seg_len))
    if close_dist > max(1e-3, 3.0 * med_step):
        return False

    rel = pts - np.asarray(frame["origin_mm"], dtype=np.float32)[None, :]
    u = _safe_unit(np.asarray(frame["u_t"], dtype=np.float32))
    v = _safe_unit(np.asarray(frame["v_t"], dtype=np.float32))
    if u is None or v is None:
        return False
    poly_xy = np.stack([rel @ u, rel @ v], axis=1)
    if _has_self_intersection_2d(poly_xy):
        return False

    return True


def _compute_feature_fields(tooth_mesh, bone_mesh):
    tooth_v = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    bone_v = np.asarray(bone_mesh.vertices, dtype=np.float32)

    if tooth_v.shape[0] == 0 or bone_v.shape[0] == 0:
        raise RuntimeError("empty vertices in smoothed mesh")

    tooth_n = _compute_vertex_normals(tooth_mesh)
    bone_n = _compute_vertex_normals(bone_mesh)

    tree = cKDTree(bone_v)
    dist, nearest_idx = tree.query(tooth_v, k=1)
    nearest_idx = nearest_idx.astype(np.int32)

    parallel = np.abs(np.sum(tooth_n * bone_n[nearest_idx], axis=1))
    parallel = np.clip(parallel, 0.0, 1.0)

    return {
        "distance_mm": dist.astype(np.float32),
        "parallelism": parallel.astype(np.float32),
        "nearest_bone_index": nearest_idx,
        "tooth_normals": tooth_n.astype(np.float32),
        "bone_normals": bone_n.astype(np.float32),
    }


def _vertex_neighbors(tooth_mesh, n_vertices: int, faces: np.ndarray) -> List[List[int]]:
    try:
        raw = getattr(tooth_mesh, "vertex_neighbors", None)
        if raw is not None and len(raw) == int(n_vertices):
            return [list(map(int, nb)) for nb in raw]
    except Exception:
        pass
    return [sorted(list(nb)) for nb in _build_adjacency(n_vertices, faces)]


def _trace_boundary_bfs(tooth_mesh, frame, feature_fields, cfg):
    # Kept under the old name to preserve the downstream result contract.
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    faces = np.asarray(tooth_mesh.faces, dtype=np.int32)
    n = int(verts.shape[0])

    origin = np.asarray(frame["origin_mm"], dtype=np.float32)
    axis = _safe_unit(np.asarray(frame["a_t"], dtype=np.float32))
    if axis is None:
        raise RuntimeError("invalid axis for boundary extraction")
    z = (verts - origin[None, :]) @ axis

    dist = np.asarray(feature_fields["distance_mm"], dtype=np.float32)
    para = np.asarray(feature_fields["parallelism"], dtype=np.float32)
    eps = float(cfg.get("epsilon_mm", 0.35))
    thr = float(cfg.get("attach_threshold", 0.85))
    attached_mask = np.logical_and(dist <= eps, para >= thr)

    state = np.zeros((n,), dtype=np.uint8)
    state[attached_mask] = 1

    if int(np.sum(attached_mask)) == 0:
        return {
            "status": STATUS_NO_BOUNDARY,
            "seed_indices": np.zeros((0,), dtype=np.int32),
            "boundary_candidate_indices": np.zeros((0,), dtype=np.int32),
            "vertex_state": state,
            "visited_mask": attached_mask.copy(),
            "attached_mask": attached_mask,
            "z_projection_mm": z.astype(np.float32),
        }

    neighbors = _vertex_neighbors(tooth_mesh, n_vertices=n, faces=faces)
    boundary_candidates: List[int] = []
    for idx in np.where(attached_mask)[0]:
        nb = np.asarray(neighbors[int(idx)], dtype=np.int32)
        if nb.size == 0:
            continue
        if not np.all(attached_mask[nb]):
            boundary_candidates.append(int(idx))

    boundary_idx = np.asarray(sorted(set(boundary_candidates)), dtype=np.int32)
    state[boundary_idx] = 2

    return {
        "status": STATUS_OK if boundary_idx.shape[0] >= int(cfg.get("min_boundary_vertices", 24)) else STATUS_NO_BOUNDARY,
        "seed_indices": np.zeros((0,), dtype=np.int32),
        "boundary_candidate_indices": boundary_idx,
        "vertex_state": state,
        "visited_mask": attached_mask.copy(),
        "attached_mask": attached_mask,
        "z_projection_mm": z.astype(np.float32),
    }


def _build_boundary_graph(tooth_mesh, boundary_candidate_indices, faces: np.ndarray):
    graph = nx.Graph()
    idx = np.asarray(boundary_candidate_indices, dtype=np.int32)
    if idx.size == 0:
        return graph

    candidate_set = {int(v) for v in idx.tolist()}
    neighbors = _vertex_neighbors(tooth_mesh, n_vertices=int(np.asarray(tooth_mesh.vertices).shape[0]), faces=faces)
    graph.add_nodes_from(candidate_set)
    for node in candidate_set:
        for nb in neighbors[node]:
            if int(nb) in candidate_set:
                graph.add_edge(int(node), int(nb))
    return graph


def _is_valid_cycle_sequence(graph: nx.Graph, sequence: List[int]) -> bool:
    seq = [int(v) for v in sequence]
    if len(seq) < 3 or len(set(seq)) != len(seq):
        return False
    return all(graph.has_edge(seq[i], seq[(i + 1) % len(seq)]) for i in range(len(seq)))


def _walk_degree_two_cycle(graph: nx.Graph) -> List[int]:
    nodes = list(graph.nodes())
    if len(nodes) < 3:
        return []
    if any(graph.degree(v) != 2 for v in nodes):
        return []

    start = min(nodes)
    ordered = [start]
    prev = None
    cur = start

    while True:
        nbs = sorted(int(v) for v in graph.neighbors(cur))
        nxt_choices = [v for v in nbs if v != prev]
        if not nxt_choices:
            return []
        nxt = nxt_choices[0]
        if nxt == start:
            return ordered if len(ordered) == len(nodes) else []
        if nxt in ordered:
            return []
        ordered.append(int(nxt))
        prev, cur = cur, nxt


def _order_cycle_nodes(graph: nx.Graph, cycle_nodes: List[int]) -> List[int]:
    cycle = [int(v) for v in cycle_nodes]
    if _is_valid_cycle_sequence(graph, cycle):
        return cycle

    induced = graph.subgraph(cycle).copy()
    ordered = _walk_degree_two_cycle(induced)
    if ordered:
        return ordered

    for basis_cycle in nx.cycle_basis(induced):
        if set(map(int, basis_cycle)) == set(cycle) and _is_valid_cycle_sequence(induced, basis_cycle):
            return [int(v) for v in basis_cycle]
    return []


def _dedupe_consecutive_points(points_mm: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    pts = np.asarray(points_mm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    keep = [0]
    for idx in range(1, int(pts.shape[0])):
        if float(np.linalg.norm(pts[idx] - pts[keep[-1]])) > float(atol):
            keep.append(idx)
    out = pts[np.asarray(keep, dtype=np.int32)]
    if out.shape[0] > 1 and float(np.linalg.norm(out[0] - out[-1])) <= float(atol):
        out = out[:-1]
    return out.astype(np.float32)


def _fit_closed_spline(points_mm: np.ndarray, target_n: int, smooth_s: float) -> np.ndarray:
    pts = _dedupe_consecutive_points(points_mm)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return np.zeros((0, 3), dtype=np.float32)

    try:
        tck, _ = splprep(pts.T, s=float(smooth_s), per=True)
        u_new = np.linspace(0.0, 1.0, int(target_n), endpoint=False)
        smooth_xyz = np.column_stack(splev(u_new, tck)).astype(np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    return smooth_xyz


def _build_boundary_curves(tooth_mesh, frame, boundary_candidate_indices, cfg):
    idx = np.asarray(boundary_candidate_indices, dtype=np.int32)
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    faces = np.asarray(tooth_mesh.faces, dtype=np.int32)
    min_cycle_vertices = int(cfg.get("min_cycle_vertices", 10))
    target_n = int(cfg.get("boundary_resample_points", 1000))
    smooth_s = float(cfg.get("spline_smooth_s", 2.0))

    if idx.shape[0] < int(cfg.get("min_boundary_vertices", 24)):
        return []

    graph = _build_boundary_graph(tooth_mesh, idx, faces=faces)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return []

    curves = []
    seen_node_sets = set()
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes).copy()
        if component.number_of_nodes() < min_cycle_vertices:
            continue

        try:
            core = nx.k_core(component, k=2)
        except Exception:
            core = component
        if core.number_of_nodes() < min_cycle_vertices:
            core = component

        basis_cycles = nx.cycle_basis(core)
        if not basis_cycles and core.number_of_nodes() >= min_cycle_vertices and all(core.degree(v) == 2 for v in core.nodes()):
            basis_cycles = [list(core.nodes())]

        for basis_cycle in basis_cycles:
            cycle_nodes = [int(v) for v in basis_cycle]
            node_key = frozenset(cycle_nodes)
            if len(node_key) < min_cycle_vertices or node_key in seen_node_sets:
                continue

            ordered_nodes = _order_cycle_nodes(core, cycle_nodes)
            if len(ordered_nodes) < min_cycle_vertices:
                continue

            smooth_curve = _fit_closed_spline(verts[np.asarray(ordered_nodes, dtype=np.int32)], target_n=target_n, smooth_s=smooth_s)
            if smooth_curve.shape[0] < min_cycle_vertices:
                continue
            if not _validate_closed_curve(smooth_curve, frame):
                continue

            seen_node_sets.add(node_key)
            curves.append(
                {
                    "node_indices": np.asarray(ordered_nodes, dtype=np.int32),
                    "curve_mm": smooth_curve.astype(np.float32),
                    "length_mm": _curve_length_mm(smooth_curve),
                    "component_size": int(component.number_of_nodes()),
                }
            )

    curves.sort(key=lambda item: float(item["length_mm"]), reverse=True)
    return curves


def _compute_anchorage_area(tooth_mesh, frame, boundary_curve_mm):
    curve = np.asarray(boundary_curve_mm, dtype=np.float32)
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float32)
    faces = np.asarray(tooth_mesh.faces, dtype=np.int32)

    if curve.shape[0] < 8 or verts.shape[0] == 0 or faces.shape[0] == 0:
        return {
            "status": STATUS_AREA_FAILED,
            "anchorage_area_mm2": 0.0,
            "root_face_mask": np.zeros((faces.shape[0],), dtype=bool),
            "root_vertex_mask": np.zeros((verts.shape[0],), dtype=bool),
        }

    origin = np.asarray(frame["origin_mm"], dtype=np.float32)
    axis = _safe_unit(np.asarray(frame["a_t"], dtype=np.float32))
    u = _safe_unit(np.asarray(frame["u_t"], dtype=np.float32))
    v = _safe_unit(np.asarray(frame["v_t"], dtype=np.float32))
    if axis is None or u is None or v is None:
        return {
            "status": STATUS_AREA_FAILED,
            "anchorage_area_mm2": 0.0,
            "root_face_mask": np.zeros((faces.shape[0],), dtype=bool),
            "root_vertex_mask": np.zeros((verts.shape[0],), dtype=bool),
        }

    rel_b = curve - origin[None, :]
    theta_b = np.mod(np.arctan2(rel_b @ v, rel_b @ u), 2.0 * np.pi)
    t_b = rel_b @ axis

    rel_v = verts - origin[None, :]
    theta_v = np.mod(np.arctan2(rel_v @ v, rel_v @ u), 2.0 * np.pi)
    t_v = rel_v @ axis
    t_cut_v = _periodic_interp(theta_v, theta_b, t_b)
    root_vertex_mask = t_v <= t_cut_v

    tri = verts[faces]
    centroid = np.mean(tri, axis=1)
    rel_c = centroid - origin[None, :]
    theta_c = np.mod(np.arctan2(rel_c @ v, rel_c @ u), 2.0 * np.pi)
    t_c = rel_c @ axis
    t_cut_c = _periodic_interp(theta_c, theta_b, t_b)
    root_face_mask = t_c <= t_cut_c

    vec1 = tri[:, 1] - tri[:, 0]
    vec2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)
    anchorage_area = float(np.sum(areas[root_face_mask]))

    return {
        "status": STATUS_OK if anchorage_area >= 0.0 else STATUS_AREA_FAILED,
        "anchorage_area_mm2": max(0.0, anchorage_area),
        "root_face_mask": root_face_mask,
        "root_vertex_mask": root_vertex_mask,
    }


def extract_pdl_boundary(
    T_roi: np.ndarray,
    B_roi: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    cfg: Optional[Dict] = None,
):
    params = dict(DEFAULT_PDL_CFG)
    if cfg:
        params.update(cfg)

    try:
        _require_deps()
    except Exception as exc:
        return {
            "status": STATUS_GEOMETRY_FAILED,
            "error": str(exc),
        }

    tooth_mesh_raw = mask_to_mesh(
        T_roi,
        spacing_xyz,
        min_component_voxels=int(params.get("min_component_voxels", 64)),
    )
    bone_mesh_raw = mask_to_mesh(
        B_roi,
        spacing_xyz,
        min_component_voxels=int(params.get("min_component_voxels", 64)),
    )
    if tooth_mesh_raw is None or bone_mesh_raw is None:
        return {
            "status": STATUS_GEOMETRY_FAILED,
            "error": "mask->mesh failed for tooth or bone",
        }

    try:
        tooth_mesh = taubin_smooth_trimesh(
            tooth_mesh_raw,
            iterations=int(params.get("taubin_iterations", 15)),
            lam=float(params.get("taubin_lambda", 0.5)),
            mu=float(params.get("taubin_mu", -0.53)),
        )
        bone_mesh = taubin_smooth_trimesh(
            bone_mesh_raw,
            iterations=int(params.get("taubin_iterations", 15)),
            lam=float(params.get("taubin_lambda", 0.5)),
            mu=float(params.get("taubin_mu", -0.53)),
        )
    except Exception as exc:
        return {
            "status": STATUS_SMOOTH_FAILED,
            "error": str(exc),
        }

    frame = compute_local_frame(np.asarray(tooth_mesh.vertices, dtype=np.float32))
    if frame is None:
        return {
            "status": STATUS_GEOMETRY_FAILED,
            "error": "failed to compute PCA tooth axis",
        }

    axis_points = build_axis_points(frame, axis_length_mm=float(params.get("axis_length_mm", 15.0)))

    try:
        feature_fields = _compute_feature_fields(tooth_mesh, bone_mesh)
    except Exception as exc:
        return {
            "status": STATUS_NORMAL_FAILED,
            "error": str(exc),
        }

    bfs = _trace_boundary_bfs(
        tooth_mesh=tooth_mesh,
        frame=frame,
        feature_fields=feature_fields,
        cfg=params,
    )
    if bfs["status"] != STATUS_OK:
        return {
            "status": bfs["status"],
            "error": "boundary extraction failed to produce valid attachment/candidate vertices",
            "tooth_mesh_raw": tooth_mesh_raw,
            "bone_mesh_raw": bone_mesh_raw,
            "tooth_mesh": tooth_mesh,
            "bone_mesh": bone_mesh,
            "frame": frame,
            "axis_points_mm": axis_points,
            "feature_fields": feature_fields,
            "bfs": bfs,
        }

    smooth_cycles = _build_boundary_curves(
        tooth_mesh=tooth_mesh,
        frame=frame,
        boundary_candidate_indices=bfs["boundary_candidate_indices"],
        cfg=params,
    )
    if len(smooth_cycles) == 0:
        return {
            "status": STATUS_NO_BOUNDARY,
            "error": "boundary postprocess failed to build valid smooth cycles",
            "tooth_mesh_raw": tooth_mesh_raw,
            "bone_mesh_raw": bone_mesh_raw,
            "tooth_mesh": tooth_mesh,
            "bone_mesh": bone_mesh,
            "frame": frame,
            "axis_points_mm": axis_points,
            "feature_fields": feature_fields,
            "bfs": bfs,
        }

    boundary_curve = np.asarray(smooth_cycles[0]["curve_mm"], dtype=np.float32)
    primary_cycle_indices = np.asarray(smooth_cycles[0]["node_indices"], dtype=np.int32)
    bfs["vertex_state"][primary_cycle_indices] = 3

    curve_stats = {
        "n_cycles": int(len(smooth_cycles)),
        "primary_cycle_index": 0,
        "cycles": [
            {
                "curve_index": int(curve_idx),
                "n_points": int(np.asarray(item["curve_mm"]).shape[0]),
                "n_nodes": int(np.asarray(item["node_indices"]).shape[0]),
                "length_mm": float(item["length_mm"]),
                "component_size": int(item["component_size"]),
                "is_primary": bool(curve_idx == 0),
            }
            for curve_idx, item in enumerate(smooth_cycles)
        ],
    }

    area = _compute_anchorage_area(tooth_mesh, frame, boundary_curve)
    if area["status"] != STATUS_OK:
        return {
            "status": STATUS_AREA_FAILED,
            "error": "failed to compute anchorage area",
            "tooth_mesh_raw": tooth_mesh_raw,
            "bone_mesh_raw": bone_mesh_raw,
            "tooth_mesh": tooth_mesh,
            "bone_mesh": bone_mesh,
            "frame": frame,
            "axis_points_mm": axis_points,
            "feature_fields": feature_fields,
            "bfs": bfs,
            "boundary_curve_mm": boundary_curve,
            "smooth_curves_3d": [np.asarray(item["curve_mm"], dtype=np.float32) for item in smooth_cycles],
            "curve_stats": curve_stats,
            "area": area,
        }

    mesh_stats = {
        "tooth_raw": _mesh_stats(tooth_mesh_raw),
        "bone_raw": _mesh_stats(bone_mesh_raw),
        "tooth_smooth": _mesh_stats(tooth_mesh),
        "bone_smooth": _mesh_stats(bone_mesh),
    }

    distance_mm = np.asarray(feature_fields["distance_mm"], dtype=np.float32)
    parallelism = np.asarray(feature_fields["parallelism"], dtype=np.float32)
    attached_mask = np.asarray(bfs["attached_mask"], dtype=bool)
    if int(np.sum(attached_mask)) > 0:
        d_att = distance_mm[attached_mask]
        p_att = parallelism[attached_mask]
        mean_d = float(np.mean(d_att))
        p95_d = float(np.percentile(d_att, 95))
        mean_p = float(np.mean(p_att))
    else:
        mean_d = None
        p95_d = None
        mean_p = None

    metrics = {
        "status": STATUS_OK,
        "n_tooth_vertices": int(np.asarray(tooth_mesh.vertices).shape[0]),
        "n_bone_vertices": int(np.asarray(bone_mesh.vertices).shape[0]),
        "n_seed_vertices": 0,
        "n_attached_vertices": int(np.sum(attached_mask)),
        "n_boundary_candidates": int(np.asarray(bfs["boundary_candidate_indices"]).shape[0]),
        "n_boundary_points": int(boundary_curve.shape[0]),
        "n_cycles": int(curve_stats["n_cycles"]),
        "anchorage_area_mm2": float(area["anchorage_area_mm2"]),
        "mean_distance_mm_attached": mean_d,
        "p95_distance_mm_attached": p95_d,
        "mean_parallelism_attached": mean_p,
        "epsilon_mm": float(params.get("epsilon_mm", 0.35)),
        "attach_threshold": float(params.get("attach_threshold", 0.85)),
    }

    return {
        "status": STATUS_OK,
        "tooth_mesh_raw": tooth_mesh_raw,
        "bone_mesh_raw": bone_mesh_raw,
        "tooth_mesh": tooth_mesh,
        "bone_mesh": bone_mesh,
        "frame": frame,
        "axis_points_mm": axis_points,
        "feature_fields": feature_fields,
        "bfs": bfs,
        "boundary_curve_mm": boundary_curve,
        "smooth_curves_3d": [np.asarray(item["curve_mm"], dtype=np.float32) for item in smooth_cycles],
        "curve_stats": curve_stats,
        "area": area,
        "metrics": metrics,
        "mesh_stats": mesh_stats,
    }
