import numpy as np
from src.datasets.roi import crop_roi, stitch_back


def test_crop_and_stitch():
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros_like(vol, dtype=np.uint8)
    mask[2:5, 3:7, 1:4] = 1
    vol[mask > 0] = 1.0

    out = crop_roi(vol, mask, padding_vox=np.array([1, 1, 1]))
    assert out is not None
    vol_roi, mask_roi, origin, _, _ = out

    stitched = stitch_back(vol.shape, mask_roi, origin)
    assert stitched.shape == mask.shape
    assert np.array_equal(stitched > 0, mask > 0)
