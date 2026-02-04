import numpy as np
from scipy.ndimage import distance_transform_edt
from src.utils.geometry import tooth_surface


def compute_geometric_prior(A, T, spacing, sigma_z_mm=2.0, sigma_s_mm=1.0, delta_s_mm=1.0,
                            z_ignore_ratio=0.1, use_gradient=False, gradient_weight=1.0):
    if T.sum() == 0:
        return np.zeros_like(T, dtype=np.float32)

    spacing = np.asarray(spacing, dtype=np.float32)
    # cross-sectional area per z
    S = T.sum(axis=(0, 1))
    z_idxs = np.where(S > 0)[0]
    if len(z_idxs) == 0:
        return np.zeros_like(T, dtype=np.float32)

    # ignore extremes
    z_min = z_idxs.min()
    z_max = z_idxs.max()
    z_len = z_max - z_min + 1
    trim = int(z_len * z_ignore_ratio)
    z_start = z_min + trim
    z_end = z_max - trim
    z_range = np.arange(z_start, z_end + 1) if z_end >= z_start else z_idxs

    S_sub = S[z_range]
    z_star = z_range[np.argmin(S_sub)]

    # Rz
    z_grid = np.arange(T.shape[2])[None, None, :]
    Rz = np.exp(-((z_grid - z_star) ** 2) * (spacing[2] ** 2) / (2.0 * sigma_z_mm ** 2))

    # Rs
    surface = tooth_surface(T)
    if surface.sum() == 0:
        Rs = np.zeros_like(T, dtype=np.float32)
    else:
        dist = distance_transform_edt(~surface, sampling=spacing)
        if delta_s_mm > 0:
            Rs = (dist <= delta_s_mm).astype(np.float32)
        else:
            Rs = np.exp(-(dist ** 2) / (2.0 * sigma_s_mm ** 2))

    R = Rz * Rs * T.astype(np.float32)

    if use_gradient:
        gx, gy, gz = np.gradient(A.astype(np.float32))
        g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        R = R * (g ** gradient_weight)

    return R.astype(np.float32)
