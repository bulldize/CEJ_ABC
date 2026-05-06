import json
import os

import numpy as np
import pytest
import yaml

pytest.importorskip("trimesh")
pytest.importorskip("open3d")

from src.datasets.io import save_volume
from src.pdl.eval import main as pdl_eval_main
from src.pdl.extract import main as pdl_extract_main
from src.pdl.preprocess import main as pdl_preprocess_main
from src.pdl.viz import main as pdl_viz_main


def _synthetic_case(shape=(48, 48, 48)):
    xx, yy, zz = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )
    cx, cy = shape[0] // 2, shape[1] // 2

    tooth = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 6 ** 2) & (zz >= 10) & (zz <= 38)
    bone_outer = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 8 ** 2) & (zz >= 12) & (zz <= 36)
    bone_inner = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 5 ** 2) & (zz >= 12) & (zz <= 36)
    bone = np.logical_and(bone_outer, np.logical_not(bone_inner))

    A = (zz.astype(np.float32) / float(shape[2])).astype(np.float32)
    B = np.zeros(shape, dtype=np.int16)
    B[bone] = 1
    B[tooth] = 11
    return A, B


def _write_raw_case(tmp_path):
    case_dir = tmp_path / "raw" / "case_001"
    case_dir.mkdir(parents=True, exist_ok=True)
    A, B = _synthetic_case()
    save_volume(str(case_dir / "A.nii.gz"), A.astype(np.float32), spacing=(0.5, 0.5, 0.5), dtype=np.float32)
    save_volume(str(case_dir / "B.nii.gz"), B.astype(np.int16), spacing=(0.5, 0.5, 0.5), dtype=np.int16)


def _write_cfg(tmp_path):
    cfg = {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "raw_layout": "case_dirs",
            "processed_dir": str(tmp_path / "processed_pdl"),
            "output_dir": str(tmp_path / "outputs_pdl"),
            "raw_a_name": "A.nii.gz",
            "raw_b_name": "B.nii.gz",
            "raw_meta_name": "meta.json",
            "processed_format": "nii.gz",
        },
        "pdl_preprocess": {
            "roi_padding_mm": 8.0,
            "target_spacing_mm": [0.5, 0.5, 0.5],
            "resample_to_target": False,
        },
        "pdl_extract": {
            "taubin_iterations": 10,
            "taubin_lambda": 0.5,
            "taubin_mu": -0.53,
            "epsilon_mm": 0.35,
            "attach_threshold": 0.85,
            "seed_z_quantile_low": 0.05,
            "seed_z_quantile_high": 0.45,
            "min_boundary_vertices": 8,
            "boundary_resample_points": 120,
            "min_cycle_vertices": 8,
            "spline_smooth_s": 1.0,
            "fail_on_geometry_error": True,
        },
        "pdl_viz": {
            "max_cases": 1,
            "max_teeth_per_case": 1,
            "max_state_points": 3000,
        },
    }
    cfg_path = tmp_path / "pdl_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def test_pdl_pipeline_end_to_end(tmp_path):
    _write_raw_case(tmp_path)
    cfg_path = _write_cfg(tmp_path)

    pdl_preprocess_main(["--config", str(cfg_path)])
    pdl_extract_main(["--config", str(cfg_path)])
    pdl_viz_main(["--config", str(cfg_path)])
    pdl_eval_main(["--config", str(cfg_path)])

    out_tooth = tmp_path / "outputs_pdl" / "infer" / "case_001" / "tooth_11"
    out_case = tmp_path / "outputs_pdl" / "infer" / "case_001"
    out_eval = tmp_path / "outputs_pdl" / "eval"

    assert (out_tooth / "tooth_mesh.ply").exists()
    assert (out_tooth / "bone_mesh.ply").exists()
    assert (out_tooth / "tooth_axis.ply").exists()
    assert (out_tooth / "pdl_boundary_curve.ply").exists()
    assert (out_tooth / "pdl_boundary_curve.npy").exists()
    assert (out_tooth / "smooth_curves_3d.npz").exists()
    assert (out_tooth / "anchorage_area.json").exists()
    assert (out_tooth / "PDL_meta.json").exists()

    payload = json.loads((out_tooth / "anchorage_area.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert float(payload["anchorage_area_mm2"]) > 0.0

    assert (out_case / "metrics_per_tooth.csv").exists()
    assert (out_case / "metrics_summary.json").exists()
    assert (out_case / "quality_report.json").exists()

    assert (out_eval / "metrics_per_tooth.csv").exists()
    assert (out_eval / "metrics_summary.json").exists()
    assert (out_eval / "quality_report.json").exists()

    assert (tmp_path / "outputs_pdl" / "viz" / "3d" / "index.html").exists()
    assert (tmp_path / "outputs_pdl" / "viz" / "3d" / "case_001" / "tooth_11" / "viewer.html").exists()


def test_pdl_extract_works_without_A_t_dependency(tmp_path):
    _write_raw_case(tmp_path)
    cfg_path = _write_cfg(tmp_path)

    pdl_preprocess_main(["--config", str(cfg_path)])

    a_t_path = tmp_path / "processed_pdl" / "case_001" / "tooth_11" / "A_t.nii.gz"
    if a_t_path.exists():
        os.remove(a_t_path)

    pdl_extract_main(["--config", str(cfg_path)])

    out_tooth = tmp_path / "outputs_pdl" / "infer" / "case_001" / "tooth_11"
    assert (out_tooth / "anchorage_area.json").exists()


def test_pdl_viz_falls_back_to_primary_curve_without_curve_bundle(tmp_path):
    _write_raw_case(tmp_path)
    cfg_path = _write_cfg(tmp_path)

    pdl_preprocess_main(["--config", str(cfg_path)])
    pdl_extract_main(["--config", str(cfg_path)])

    curve_bundle = tmp_path / "outputs_pdl" / "infer" / "case_001" / "tooth_11" / "smooth_curves_3d.npz"
    if curve_bundle.exists():
        os.remove(curve_bundle)

    pdl_viz_main(["--config", str(cfg_path)])

    viewer = tmp_path / "outputs_pdl" / "viz" / "3d" / "case_001" / "tooth_11" / "viewer.html"
    assert viewer.exists()
