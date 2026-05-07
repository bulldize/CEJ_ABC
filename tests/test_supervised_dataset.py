import json

import numpy as np

from src.datasets.dataset import ToothDataset, build_supervised_audit
from src.datasets.io import save_volume


def _write_tooth(root, case_id, tooth_id, points, h_gt):
    tooth_dir = root / case_id / f"tooth_{tooth_id}"
    tooth_dir.mkdir(parents=True)
    shape = h_gt.shape
    save_volume(str(tooth_dir / "A_t.nii.gz"), np.ones(shape, dtype=np.float32))
    save_volume(str(tooth_dir / "T_t.nii.gz"), np.ones(shape, dtype=np.uint8))
    save_volume(str(tooth_dir / "H_GT.nii.gz"), h_gt.astype(np.float32))
    (tooth_dir / "points.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "coord_type": "voxel",
                "space": "roi",
                "points": {str(tooth_id): points},
            }
        )
    )
    (tooth_dir / "roi_meta.json").write_text(
        json.dumps({"case_id": case_id, "tooth_id": tooth_id})
    )
    return tooth_dir


def test_supervised_dataset_skips_unlabeled_teeth(tmp_path):
    processed_dir = tmp_path / "processed"
    h_labeled = np.zeros((4, 4, 4), dtype=np.float32)
    h_labeled[1, 1, 1] = 1.0
    _write_tooth(
        processed_dir,
        "case_a",
        11,
        points=[[1.0, 1.0, 1.0]],
        h_gt=h_labeled,
    )
    skipped_dir = _write_tooth(
        processed_dir,
        "case_a",
        12,
        points=[],
        h_gt=np.zeros((4, 4, 4), dtype=np.float32),
    )

    audit = build_supervised_audit(str(processed_dir))
    assert audit["total_tooth_dirs"] == 2
    assert audit["used_labeled_tooth_dirs"] == 1
    assert audit["skipped_unlabeled_tooth_dirs"] == 1
    assert audit["skipped_unlabeled_teeth"][0]["tooth_dir"] == str(skipped_dir)

    ds = ToothDataset(str(processed_dir), cache_rate=0.0)
    assert len(ds) == 1
    assert ds.items[0]["tooth_dir"].endswith("tooth_11")
