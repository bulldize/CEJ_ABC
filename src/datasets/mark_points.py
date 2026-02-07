import json
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.datasets.points import save_points
from src.utils.log import get_logger


logger = get_logger("mark_points")


def _find_xyz_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}

    def pick_by_patterns(patterns):
        for p in patterns:
            for c, lc in lower.items():
                if re.search(p, lc):
                    return c
        return None

    x_col = pick_by_patterns([r"position\\s*\\[0\\]", r"\\bx\\b"])
    y_col = pick_by_patterns([r"position\\s*\\[1\\]", r"\\by\\b"])
    z_col = pick_by_patterns([r"position\\s*\\[2\\]", r"\\bz\\b"])
    if x_col and y_col and z_col:
        return x_col, y_col, z_col

    num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols) >= 3:
        return num_cols[0], num_cols[1], num_cols[2]
    raise ValueError("cannot find x/y/z columns in excel file")


def _find_tooth_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in cols:
        if "牙位" in str(c):
            return c
    for c in cols:
        if "tooth" in str(c).lower():
            return c
    raise ValueError("cannot find tooth id column in cej_points_ras.xlsx")


def _read_scan_boundary(scan_path: str) -> np.ndarray:
    df = pd.read_excel(scan_path)
    x_col, y_col, z_col = _find_xyz_cols(df)
    pts = df[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
    if pts.shape[0] < 8:
        logger.warning("scan boundary has %d points, expected 8", pts.shape[0])
    return pts


def compute_mark_affines(scan_points: np.ndarray, shape: Tuple[int, int, int]):
    shape = np.asarray(shape, dtype=np.float32)
    if np.any(shape < 2):
        raise ValueError(f"invalid shape for affine compute: {shape}")
    minv = scan_points.min(axis=0)
    maxv = scan_points.max(axis=0)
    spacing = (maxv - minv) / (shape - 1.0)

    aff_v2m = np.eye(4, dtype=np.float32)
    aff_v2m[0, 0] = spacing[0]
    aff_v2m[1, 1] = spacing[1]
    aff_v2m[2, 2] = spacing[2]
    aff_v2m[:3, 3] = minv

    aff_m2v = np.linalg.inv(aff_v2m).astype(np.float32)
    return minv, maxv, spacing, aff_v2m, aff_m2v


def _points_mark_to_vox(points_mark: np.ndarray, aff_m2v: np.ndarray) -> np.ndarray:
    if points_mark.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points_mark, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    pts_v = (aff_m2v @ pts_h.T).T
    return pts_v[:, :3]


def _read_cej_points(cej_path: str):
    df = pd.read_excel(cej_path)
    tooth_col = _find_tooth_col(df)
    x_col, y_col, z_col = _find_xyz_cols(df)
    order_col = None
    for c in df.columns:
        if "点位" in str(c):
            order_col = c
            break
    return df, tooth_col, x_col, y_col, z_col, order_col


def _group_points_by_tooth(df: pd.DataFrame, tooth_col: str, x_col: str, y_col: str, z_col: str, order_col=None):
    df = df.copy()
    tooth_ids = df[tooth_col].astype(str).str.extract(r"(\\d+)")[0]
    df["_tooth_id"] = tooth_ids
    df = df.dropna(subset=["_tooth_id"])
    if order_col is not None:
        df = df.sort_values(["_tooth_id", order_col])

    points_by_tooth: Dict[str, np.ndarray] = {}
    for tooth_id, sub in df.groupby("_tooth_id"):
        pts = sub[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
        points_by_tooth[str(int(tooth_id))] = pts
    return points_by_tooth


def ensure_mark_points(case_dir: str, case_id: str, shape, affine_world: np.ndarray, cfg: dict):
    cej_name = cfg["data"].get("cej_points_name", "cej_points_ras.xlsx")
    scan_path = cfg["data"].get("scan_boundary_path", os.path.join("data", "scan_boundary_ras.xlsx"))
    mark_meta_name = cfg["data"].get("mark_meta_name", "mark_meta.json")
    raw_points_name = cfg["data"].get("raw_points_name", "points.json")

    cej_path = os.path.join(case_dir, cej_name)
    if not os.path.exists(cej_path):
        return False
    if not os.path.exists(scan_path):
        raise FileNotFoundError(f"scan boundary file not found: {scan_path}")

    scan_pts = _read_scan_boundary(scan_path)
    minv, maxv, spacing, aff_v2m, aff_m2v = compute_mark_affines(scan_pts, shape)
    affine_spacing = np.linalg.norm(np.asarray(affine_world, dtype=np.float32)[:3, :3], axis=0)
    if not np.allclose(spacing, affine_spacing, atol=1e-3):
        logger.warning(
            "case=%s mark spacing %s differs from A affine spacing %s",
            case_id, np.round(spacing, 6).tolist(), np.round(affine_spacing, 6).tolist()
        )

    df, tooth_col, x_col, y_col, z_col, order_col = _read_cej_points(cej_path)
    points_by_tooth_mark = _group_points_by_tooth(df, tooth_col, x_col, y_col, z_col, order_col)

    points_by_tooth_vox: Dict[str, list] = {}
    n_outside = 0
    for tooth_id, pts_mark in points_by_tooth_mark.items():
        pts_vox = _points_mark_to_vox(pts_mark, aff_m2v)
        if pts_vox.size > 0:
            in_bounds = np.all((pts_vox >= 0) & (pts_vox <= (np.asarray(shape) - 1)), axis=1)
            n_outside += int((~in_bounds).sum())
        points_by_tooth_vox[tooth_id] = pts_vox.tolist()

    if n_outside > 0:
        logger.warning("case=%s mark->vox has %d points outside volume bounds", case_id, n_outside)

    raw_points_path = os.path.join(case_dir, raw_points_name)
    save_points(raw_points_path, case_id, "voxel", "full", points_by_tooth_vox)

    aff_mark_to_world = (affine_world @ aff_m2v).astype(np.float32)

    mark_meta = {
        "case_id": case_id,
        "scan_boundary_path": scan_path,
        "cej_points_path": cej_path,
        "shape": [int(v) for v in shape],
        "mark_min": [float(v) for v in minv],
        "mark_max": [float(v) for v in maxv],
        "spacing_mark": [float(v) for v in spacing],
        "affine_vox_to_mark": aff_v2m.tolist(),
        "affine_mark_to_vox": aff_m2v.tolist(),
        "affine_mark_to_world": aff_mark_to_world.tolist(),
        "affine_world": affine_world.tolist(),
    }
    with open(os.path.join(case_dir, mark_meta_name), "w") as f:
        json.dump(mark_meta, f)

    # quick validation on corners
    corners = np.array(
        [
            [0, 0, 0],
            [shape[0] - 1, 0, 0],
            [0, shape[1] - 1, 0],
            [0, 0, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, 0],
            [shape[0] - 1, 0, shape[2] - 1],
            [0, shape[1] - 1, shape[2] - 1],
            [shape[0] - 1, shape[1] - 1, shape[2] - 1],
        ],
        dtype=np.float32,
    )
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    corners_h = np.concatenate([corners, ones], axis=1)
    corners_mark = (aff_v2m @ corners_h.T).T[:, :3]
    min_err = float(np.max(np.abs(corners_mark.min(axis=0) - minv)))
    max_err = float(np.max(np.abs(corners_mark.max(axis=0) - maxv)))
    logger.info("mark affine check case=%s min_err=%.6f max_err=%.6f", case_id, min_err, max_err)
    logger.info("wrote %s and %s", raw_points_path, os.path.join(case_dir, mark_meta_name))

    return True
