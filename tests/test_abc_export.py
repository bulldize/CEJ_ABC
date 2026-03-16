import numpy as np

from src.abc.export import _build_three_segment_compare


def test_abc_compare_label_priority_and_counts():
    tooth = np.zeros((12, 12, 12), dtype=np.uint8)
    tooth[3:9, 3:9, 3:9] = 1

    seg2 = np.zeros_like(tooth)
    seg2[4:6, 4:6, 4:6] = 1

    seg3 = np.zeros_like(tooth)
    seg3[5:8, 5:8, 5:8] = 1

    out, stats = _build_three_segment_compare(tooth, seg2, seg3)

    assert out.shape == tooth.shape
    assert int(out.max()) == 3
    assert int(stats["label_counts"].get(1, 0)) > 0
    assert int(stats["label_counts"].get(3, 0)) > 0
    assert int(stats["segment2_segment3_overlap_voxels"]) > 0
