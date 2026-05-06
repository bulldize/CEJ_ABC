import argparse
import json
import os
from collections import defaultdict

import numpy as np

from src.datasets.io import load_volume
from src.pdl.algorithm import DEFAULT_PDL_CFG, STATUS_OK, extract_pdl_boundary, trimesh
from src.utils.config import ensure_dir, load_config
from src.utils.log import get_logger

logger = get_logger("pdl_extract")

DEFAULT_CONFIG = "/Users/bulldize/Desktop/华西口腔_CEJ_ABC/runtime/pdl/configs/pdl_runtime.yaml"


def _save_mesh(mesh_obj, path):
    if trimesh is None:
        raise RuntimeError("trimesh is required to export pdl mesh artifacts")
    mesh_obj.export(path)


def _save_points_ply(points_mm, path):
    if trimesh is None:
        raise RuntimeError("trimesh is required to export pdl point artifacts")
    pts = np.asarray(points_mm, dtype=np.float32)
    cloud = trimesh.points.PointCloud(pts)
    cloud.export(path)


def _save_curve_bundle_npz(curves, path):
    payload = {}
    for idx, curve in enumerate(curves or []):
        pts = np.asarray(curve, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            payload[f"curve_{idx:03d}"] = pts
    if payload:
        np.savez(path, **payload)


def _frame_json(frame):
    return {
        "origin_mm": np.asarray(frame["origin_mm"], dtype=np.float32).tolist(),
        "a_t": np.asarray(frame["a_t"], dtype=np.float32).tolist(),
        "u_t": np.asarray(frame["u_t"], dtype=np.float32).tolist(),
        "v_t": np.asarray(frame["v_t"], dtype=np.float32).tolist(),
    }


def _save_root_region_mesh(tooth_mesh, root_face_mask, out_path):
    if trimesh is None:
        return
    faces = np.asarray(tooth_mesh.faces, dtype=np.int32)
    mask = np.asarray(root_face_mask, dtype=bool)
    if faces.shape[0] == 0 or int(np.sum(mask)) == 0:
        return

    root_mesh = trimesh.Trimesh(
        vertices=np.asarray(tooth_mesh.vertices, dtype=np.float32),
        faces=faces[mask],
        process=False,
    )
    root_mesh.remove_unreferenced_vertices()
    try:
        root_mesh.fix_normals()
    except Exception:
        pass
    root_mesh.export(out_path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    params = dict(DEFAULT_PDL_CFG)
    params.update(cfg.get("pdl_extract", {}))

    fmt = cfg["data"]["processed_format"]
    processed_dir = cfg["data"]["processed_dir"]
    infer_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "infer"))

    if not os.path.isdir(processed_dir):
        logger.warning("processed_dir not found: %s", processed_dir)
        return

    tooth_dirs = sorted(
        [
            os.path.join(processed_dir, c, t)
            for c in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, c))
            for t in os.listdir(os.path.join(processed_dir, c))
            if t.startswith("tooth_") and os.path.isdir(os.path.join(processed_dir, c, t))
        ]
    )
    if not tooth_dirs:
        logger.warning("no pdl processed teeth found under %s", processed_dir)
        return

    case_status = defaultdict(lambda: {"ok": 0, "failed": 0})

    for tooth_dir in tooth_dirs:
        roi_meta_path = os.path.join(tooth_dir, "roi_meta.json")
        if not os.path.exists(roi_meta_path):
            continue
        with open(roi_meta_path, "r", encoding="utf-8") as f:
            roi_meta = json.load(f)

        case_id = str(roi_meta["case_id"])
        tooth_id = int(roi_meta["tooth_id"])

        T_path = os.path.join(tooth_dir, f"T_t.{fmt}")
        B_path = os.path.join(tooth_dir, f"B_t.{fmt}")
        if not (os.path.exists(T_path) and os.path.exists(B_path)):
            logger.warning("skip missing ROI files: %s", tooth_dir)
            continue

        T_roi, spacing, _ = load_volume(T_path, dtype=np.uint8)
        B_roi, _, _ = load_volume(B_path, dtype=np.uint8)

        spacing_from_meta = roi_meta.get("spacing", None)
        if isinstance(spacing_from_meta, (list, tuple)) and len(spacing_from_meta) == 3:
            spacing = tuple(float(v) for v in spacing_from_meta)

        out_tooth_dir = ensure_dir(os.path.join(infer_root, case_id, f"tooth_{tooth_id}"))

        result = extract_pdl_boundary(
            T_roi=T_roi,
            B_roi=B_roi,
            spacing_xyz=spacing,
            cfg=params,
        )

        status = str(result.get("status", "unknown"))
        error = result.get("error", None)

        area_payload = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "status": status,
            "anchorage_area_mm2": float(result.get("metrics", {}).get("anchorage_area_mm2", 0.0)) if status == STATUS_OK else 0.0,
            "metrics": result.get("metrics", {}),
            "error": error,
        }

        meta = {
            "case_id": case_id,
            "tooth_id": tooth_id,
            "status": status,
            "roi_origin_in_full": roi_meta.get("roi_origin_in_full", [0, 0, 0]),
            "mesh_stats": result.get("mesh_stats", {}),
            "error": error,
        }

        if status == STATUS_OK:
            _save_mesh(result["tooth_mesh"], os.path.join(out_tooth_dir, "tooth_mesh.ply"))
            _save_mesh(result["bone_mesh"], os.path.join(out_tooth_dir, "bone_mesh.ply"))
            _save_points_ply(result["axis_points_mm"], os.path.join(out_tooth_dir, "tooth_axis.ply"))
            _save_points_ply(result["boundary_curve_mm"], os.path.join(out_tooth_dir, "pdl_boundary_curve.ply"))
            np.save(os.path.join(out_tooth_dir, "pdl_boundary_curve.npy"), np.asarray(result["boundary_curve_mm"], dtype=np.float32))
            _save_curve_bundle_npz(result.get("smooth_curves_3d", []), os.path.join(out_tooth_dir, "smooth_curves_3d.npz"))
            np.save(
                os.path.join(out_tooth_dir, "feature_distance_mm.npy"),
                np.asarray(result["feature_fields"]["distance_mm"], dtype=np.float32),
            )
            np.save(
                os.path.join(out_tooth_dir, "feature_parallelism.npy"),
                np.asarray(result["feature_fields"]["parallelism"], dtype=np.float32),
            )
            np.save(
                os.path.join(out_tooth_dir, "vertex_state.npy"),
                np.asarray(result["bfs"]["vertex_state"], dtype=np.uint8),
            )

            tooth_v = np.asarray(result["tooth_mesh"].vertices, dtype=np.float32)
            seed_idx = np.asarray(result["bfs"]["seed_indices"], dtype=np.int32)
            boundary_idx = np.asarray(result["bfs"]["boundary_candidate_indices"], dtype=np.int32)
            np.save(os.path.join(out_tooth_dir, "seed_points_mm.npy"), tooth_v[seed_idx])
            np.save(os.path.join(out_tooth_dir, "boundary_candidate_points_mm.npy"), tooth_v[boundary_idx])

            _save_root_region_mesh(
                tooth_mesh=result["tooth_mesh"],
                root_face_mask=result["area"]["root_face_mask"],
                out_path=os.path.join(out_tooth_dir, "pdl_region_mesh.ply"),
            )

            meta.update(
                {
                    "F_t": _frame_json(result["frame"]),
                    "taubin": {
                        "iterations": int(params.get("taubin_iterations", 15)),
                        "lambda": float(params.get("taubin_lambda", 0.5)),
                        "mu": float(params.get("taubin_mu", -0.53)),
                    },
                    "thresholds": {
                        "epsilon_mm": float(params.get("epsilon_mm", 0.35)),
                        "attach_threshold": float(params.get("attach_threshold", 0.85)),
                    },
                    "bfs_stats": {
                        "n_seed_vertices": int(seed_idx.shape[0]),
                        "n_attached_vertices": int(np.sum(np.asarray(result["bfs"]["attached_mask"], dtype=bool))),
                        "n_boundary_candidates": int(boundary_idx.shape[0]),
                    },
                    "boundary_stats": {
                        "n_boundary_points": int(np.asarray(result["boundary_curve_mm"]).shape[0]),
                    },
                    "curve_stats": result.get("curve_stats", {}),
                }
            )
            case_status[case_id]["ok"] += 1
        else:
            case_status[case_id]["failed"] += 1
            if bool(params.get("fail_on_geometry_error", True)):
                with open(os.path.join(out_tooth_dir, "anchorage_area.json"), "w", encoding="utf-8") as f:
                    json.dump(area_payload, f, ensure_ascii=False, indent=2)
                with open(os.path.join(out_tooth_dir, "PDL_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                raise RuntimeError(f"PDL extraction failed case={case_id} tooth={tooth_id}: {error}")

        with open(os.path.join(out_tooth_dir, "anchorage_area.json"), "w", encoding="utf-8") as f:
            json.dump(area_payload, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_tooth_dir, "PDL_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("pdl extract case=%s tooth=%s status=%s", case_id, tooth_id, status)

    for case_id, stat in case_status.items():
        out_case_dir = ensure_dir(os.path.join(infer_root, case_id))
        with open(os.path.join(out_case_dir, "quality_report.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case_id": case_id,
                    "ok_teeth": int(stat["ok"]),
                    "failed_teeth": int(stat["failed"]),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
