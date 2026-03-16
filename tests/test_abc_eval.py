import csv
import json

import yaml
import numpy as np

from src.abc.eval import main as eval_main
from src.datasets.io import save_volume


def test_abc_eval_generates_csv_and_summary(tmp_path):
    output_dir = tmp_path / "outputs_abc"
    infer_tooth_dir = output_dir / "infer" / "case_001" / "tooth_11"
    infer_tooth_dir.mkdir(parents=True, exist_ok=True)

    curve = np.zeros((8, 8, 8), dtype=np.uint8)
    curve[2, 2, 2] = 1
    curve[3, 2, 2] = 1
    curve[4, 2, 2] = 1
    save_volume(
        str(infer_tooth_dir / "C_ABC.nii.gz"),
        curve,
        spacing=(0.5, 0.5, 1.0),
        dtype=np.uint8,
    )

    with open(infer_tooth_dir / "abc_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "ok",
                "curve_length_mm": 1.5,
                "n_components": 1,
                "n_inner_wall_candidates": 12,
                "n_curve_points": 180,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cfg = {"data": {"output_dir": str(output_dir)}}
    cfg_path = tmp_path / "abc_eval_test.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    eval_main(["--config", str(cfg_path)])

    eval_dir = output_dir / "eval"
    csv_path = eval_dir / "metrics_per_tooth.csv"
    summary_path = eval_dir / "metrics_summary.json"

    assert csv_path.exists()
    assert summary_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case_001"
    assert rows[0]["tooth_id"] == "11"
    assert rows[0]["status"] == "ok"
    assert int(rows[0]["curve_voxels"]) == 3

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_cases"] == 1
    assert summary["total_teeth"] == 1
    assert summary["teeth_with_curve"] == 1
    assert summary["status_counts"]["ok"] == 1
