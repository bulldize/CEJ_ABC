import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from src.pdl.algorithm import (  # noqa: E402
    STATUS_OK,
    _build_boundary_curves,
    _compute_anchorage_area,
    _compute_feature_fields,
    _trace_boundary_bfs,
    _validate_closed_curve,
)


def _frame_z():
    return {
        "origin_mm": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "a_t": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "u_t": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "v_t": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }


def test_feature_fields_distance_parallelism_bounds():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    fields = _compute_feature_fields(mesh, mesh.copy())

    dist = np.asarray(fields["distance_mm"], dtype=np.float32)
    para = np.asarray(fields["parallelism"], dtype=np.float32)

    assert dist.shape[0] == np.asarray(mesh.vertices).shape[0]
    assert np.all(dist >= 0.0)
    assert float(np.max(dist)) <= 1e-4
    assert np.all((para >= 0.0) & (para <= 1.0))
    assert float(np.min(para)) >= 0.999


def test_attached_boundary_candidates_follow_one_ring_transition():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [2, 3, 4],
            [3, 5, 4],
        ],
        dtype=np.int32,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.fix_normals()

    out = _trace_boundary_bfs(
        tooth_mesh=mesh,
        frame=_frame_z(),
        feature_fields={
            "distance_mm": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            "parallelism": np.ones((verts.shape[0],), dtype=np.float32),
        },
        cfg={
            "epsilon_mm": 0.10,
            "attach_threshold": 0.85,
            "min_boundary_vertices": 1,
        },
    )

    attached = np.asarray(out["attached_mask"], dtype=bool)
    boundary = np.asarray(out["boundary_candidate_indices"], dtype=np.int32)
    assert int(np.sum(attached)) == 4
    assert set(boundary.tolist()) == {2, 3}


def _ring_strip_mesh(n=16, radius_outer=3.0, radius_inner=2.6, z=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False, dtype=np.float32)
    outer = np.stack(
        [radius_outer * np.cos(theta), radius_outer * np.sin(theta), np.full_like(theta, z)],
        axis=1,
    )
    inner = np.stack(
        [radius_inner * np.cos(theta), radius_inner * np.sin(theta), np.full_like(theta, z)],
        axis=1,
    )
    verts = np.vstack([outer, inner]).astype(np.float32)

    faces = []
    for idx in range(int(n)):
        nxt = (idx + 1) % int(n)
        faces.append([idx, nxt, int(n) + idx])
        faces.append([nxt, int(n) + nxt, int(n) + idx])
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int32), process=False)


def test_boundary_graph_cycles_generate_smooth_closed_curve():
    mesh = _ring_strip_mesh(n=20)
    mesh.fix_normals()
    boundary_idx = np.arange(20, dtype=np.int32)

    curves = _build_boundary_curves(
        tooth_mesh=mesh,
        frame=_frame_z(),
        boundary_candidate_indices=boundary_idx,
        cfg={
            "min_boundary_vertices": 8,
            "min_cycle_vertices": 8,
            "boundary_resample_points": 120,
            "spline_smooth_s": 0.5,
        },
    )

    assert len(curves) >= 1
    primary = curves[0]
    smooth_curve = np.asarray(primary["curve_mm"], dtype=np.float32)
    assert smooth_curve.shape == (120, 3)
    assert _validate_closed_curve(smooth_curve, _frame_z())
    assert float(primary["length_mm"]) > 0.0


def test_closed_curve_validation_and_self_intersection_check():
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    circle = np.stack([2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros_like(theta)], axis=1).astype(np.float32)
    assert _validate_closed_curve(circle, _frame_z())

    # Figure-eight in XY plane: intersects near origin.
    eight = np.stack([2.0 * np.sin(theta), 2.0 * np.sin(theta) * np.cos(theta), np.zeros_like(theta)], axis=1).astype(
        np.float32
    )
    assert not _validate_closed_curve(eight, _frame_z())


def test_anchorage_area_positive_and_stable_range():
    tooth_mesh = trimesh.creation.cylinder(radius=2.0, height=10.0, sections=96)
    frame = _frame_z()

    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    boundary_1 = np.stack([2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros_like(theta)], axis=1).astype(np.float32)
    out_1 = _compute_anchorage_area(tooth_mesh, frame, boundary_1)

    assert out_1["status"] == STATUS_OK
    area_1 = float(out_1["anchorage_area_mm2"])
    assert area_1 > 0.0
    assert 60.0 <= area_1 <= 90.0

    boundary_2 = boundary_1.copy()
    boundary_2[:, 2] = 0.1 * np.cos(2.0 * theta)
    out_2 = _compute_anchorage_area(tooth_mesh, frame, boundary_2)

    assert out_2["status"] == STATUS_OK
    area_2 = float(out_2["anchorage_area_mm2"])
    rel = abs(area_2 - area_1) / max(1e-6, area_1)
    assert rel <= 0.20
