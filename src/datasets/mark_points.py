import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.datasets.manual_points_source import find_xyz_cols, find_tooth_col, load_manual_points_table
from src.datasets.points import save_points
from src.utils.log import get_logger
from src.utils.geometry import world_to_vox


logger = get_logger("mark_points")

def _read_scan_boundary(scan_path: str) -> np.ndarray:
    df = pd.read_excel(scan_path)
    x_col, y_col, z_col = find_xyz_cols(df)
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


def _points_world_to_vox(points_world: np.ndarray, affine_world: np.ndarray, flip_xy: bool = False) -> np.ndarray:
    if points_world.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(points_world, dtype=np.float32)
    if flip_xy:
        pts = pts.copy()
        pts[:, 0] *= -1.0
        pts[:, 1] *= -1.0
    return world_to_vox(pts, affine_world)


def _count_in_bounds(points_by_tooth: Dict[str, np.ndarray], shape: Tuple[int, int, int]) -> Tuple[int, int]:
    shape = np.asarray(shape, dtype=np.float32) - 1.0
    total = 0
    inb = 0
    for pts in points_by_tooth.values():
        pts = np.asarray(pts, dtype=np.float32)
        if pts.size == 0:
            continue
        mask = np.all((pts >= 0) & (pts <= shape), axis=1)
        inb += int(mask.sum())
        total += int(mask.size)
    return total, inb


def _read_cej_points(cej_path: str):
    return load_manual_points_table(Path(cej_path))


def _group_points_by_tooth(df: pd.DataFrame, tooth_col: str, x_col: str, y_col: str, z_col: str, order_col=None):
    df = df.copy()
    tooth_ids = df[tooth_col].astype(str).str.extract(r"(\d+)")[0]
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
        fallback = cfg["data"].get("cej_points_path")
        if fallback and os.path.exists(fallback):
            cej_path = fallback
        else:
            fallback = os.path.join(os.path.dirname(scan_path), cej_name)
            if os.path.exists(fallback):
                cej_path = fallback
            else:
                mrk_json_candidates = [
                    os.path.join(case_dir, name)
                    for name in os.listdir(case_dir)
                    if name.endswith(".mrk.json")
                ]
                if mrk_json_candidates:
                    cej_path = case_dir
                else:
                    # no cej points available -> ensure empty points.json for unsupervised cases
                    raw_points_path = os.path.join(case_dir, raw_points_name)
                    if not os.path.exists(raw_points_path):
                        save_points(raw_points_path, case_id, "voxel", "full", {})
                    return False
    df, tooth_col, x_col, y_col, z_col, order_col, source_info = _read_cej_points(cej_path)
    points_by_tooth_mark = _group_points_by_tooth(df, tooth_col, x_col, y_col, z_col, order_col)

    scan_pts = _read_scan_boundary(scan_path) if os.path.exists(scan_path) else None
    minv = maxv = spacing = aff_v2m = aff_m2v = None
    if scan_pts is not None:
        minv, maxv, spacing, aff_v2m, aff_m2v = compute_mark_affines(scan_pts, shape)
        affine_spacing = np.linalg.norm(np.asarray(affine_world, dtype=np.float32)[:3, :3], axis=0)
        if not np.allclose(spacing, affine_spacing, atol=1e-3):
            logger.warning(
                "case=%s mark spacing %s differs from A affine spacing %s",
                case_id, np.round(spacing, 6).tolist(), np.round(affine_spacing, 6).tolist()
            )

    # try interpreting CEJ points in world (RAS or LPS) using the real affine
    world_ras = {
        tooth_id: _points_world_to_vox(pts_mark, affine_world, flip_xy=False)
        for tooth_id, pts_mark in points_by_tooth_mark.items()
    }
    total_ras, in_ras = _count_in_bounds(world_ras, shape)
    world_lps = {
        tooth_id: _points_world_to_vox(pts_mark, affine_world, flip_xy=True)
        for tooth_id, pts_mark in points_by_tooth_mark.items()
    }
    total_lps, in_lps = _count_in_bounds(world_lps, shape)

    world_candidates = [
        ("world_ras", world_ras, total_ras, in_ras),
        ("world_lps", world_lps, total_lps, in_lps),
    ]
    best_world = max(world_candidates, key=lambda x: (x[3] / max(x[2], 1), x[3]))
    best_world_ratio = best_world[3] / max(best_world[2], 1)

    preferred_world_mode = None
    if source_info.get("source_format") == "slicer_mrk_json":
        coord_values = {
            str(v).upper()
            for v in df.get("coord_system", pd.Series(dtype=str)).dropna().astype(str).tolist()
        }
        if coord_values == {"LPS"}:
            preferred_world_mode = "world_lps"
        elif coord_values == {"RAS"}:
            preferred_world_mode = "world_ras"

    conversion_mode = None
    points_by_tooth_vox: Dict[str, np.ndarray] = {}
    total = inb = 0
    if preferred_world_mode in {"world_ras", "world_lps"}:
        for mode_name, mode_points, mode_total, mode_inb in world_candidates:
            if mode_name == preferred_world_mode:
                conversion_mode = mode_name
                points_by_tooth_vox = mode_points
                total = mode_total
                inb = mode_inb
                logger.info(
                    "case=%s using %s from manual source hint (in-bounds %d/%d)",
                    case_id, conversion_mode, inb, total
                )
                break
    elif best_world_ratio >= 0.5:
        conversion_mode, points_by_tooth_vox, total, inb = best_world
        logger.info(
            "case=%s using %s with A affine (in-bounds %d/%d)",
            case_id, conversion_mode, inb, total
        )
    elif scan_pts is not None and aff_m2v is not None:
        points_by_tooth_vox = {
            tooth_id: _points_mark_to_vox(pts_mark, aff_m2v)
            for tooth_id, pts_mark in points_by_tooth_mark.items()
        }
        total, inb = _count_in_bounds(points_by_tooth_vox, shape)
        conversion_mode = "mark"
        logger.info(
            "case=%s using mark->vox from scan boundary (in-bounds %d/%d)",
            case_id, inb, total
        )
    else:
        conversion_mode, points_by_tooth_vox, total, inb = best_world
        logger.warning(
            "case=%s scan boundary missing; fallback to %s (in-bounds %d/%d)",
            case_id, conversion_mode, inb, total
        )

    n_outside = int(total - inb)
    if n_outside > 0:
        logger.warning("case=%s %s has %d points outside volume bounds", case_id, conversion_mode, n_outside)

    raw_points_path = os.path.join(case_dir, raw_points_name)
    points_by_tooth_vox_list = {k: v.tolist() for k, v in points_by_tooth_vox.items()}
    save_points(raw_points_path, case_id, "voxel", "full", points_by_tooth_vox_list)

    mark_meta = {
        "case_id": case_id,
        "source_format": source_info.get("source_format"),
        "scan_boundary_path": scan_path if scan_pts is not None else None,
        "cej_points_path": cej_path,
        "shape": [int(v) for v in shape],
        "points_coord": conversion_mode,
        "points_in_bounds": {"total": int(total), "in_bounds": int(inb)},
        "affine_world": affine_world.tolist(),
    }
    if scan_pts is not None and aff_v2m is not None and aff_m2v is not None:
        aff_mark_to_world = (affine_world @ aff_m2v).astype(np.float32)
        mark_meta.update(
            {
                "mark_min": [float(v) for v in minv],
                "mark_max": [float(v) for v in maxv],
                "spacing_mark": [float(v) for v in spacing],
                "affine_vox_to_mark": aff_v2m.tolist(),
                "affine_mark_to_vox": aff_m2v.tolist(),
                "affine_mark_to_world": aff_mark_to_world.tolist(),
            }
        )
    with open(os.path.join(case_dir, mark_meta_name), "w") as f:
        json.dump(mark_meta, f)

    # quick validation on corners
    if scan_pts is not None and aff_v2m is not None:
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
