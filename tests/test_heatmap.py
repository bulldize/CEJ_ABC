import numpy as np
from src.datasets.heatmap import generate_heatmap_from_points


def test_heatmap_peak():
    shape = (20, 20, 20)
    pts = np.array([[10, 10, 10]], dtype=np.float32)
    h = generate_heatmap_from_points(shape, pts, spacing=(1.0, 1.0, 1.0), sigma_mm=1.0)
    assert h.shape == shape
    assert h[10, 10, 10] > 0.95
