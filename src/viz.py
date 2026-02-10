import argparse
import glob
import json
import os
import numpy as np
from scipy.spatial import cKDTree
from monai.data.utils import affine_to_spacing

from src.datasets.io import load_volume
from src.datasets.points import load_points, get_points_for_tooth
from src.postprocess.priors import compute_geometric_prior
from src.utils.config import load_config, ensure_dir
from src.utils.log import get_logger

logger = get_logger("viz")

_plt = None
_binary_erosion = None


def _ensure_viz_deps():
    global _plt, _binary_erosion
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    if _binary_erosion is None:
        from skimage.morphology import binary_erosion
        _binary_erosion = binary_erosion


def boundary2d(mask2d):
    er = _binary_erosion(mask2d)
    return mask2d ^ er


def pick_slices(shape_z, n=4):
    if n <= 1:
        return [shape_z // 2]
    return np.linspace(0, shape_z - 1, n, dtype=int).tolist()


def save_overlay(A, overlay=None, points=None, out_path=None, title=None, num_slices=4):
    z_slices = pick_slices(A.shape[2], n=num_slices)
    fig, axes = _plt.subplots(1, len(z_slices), figsize=(4 * len(z_slices), 4))
    if len(z_slices) == 1:
        axes = [axes]

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(A[:, :, z].T, cmap="gray", origin="lower")
        if overlay is not None:
            ov = overlay[:, :, z]
            ax.imshow(np.ma.masked_where(ov <= 0, ov).T, cmap="jet", alpha=0.4, origin="lower")
        if points is not None and len(points) > 0:
            pts_z = points[np.round(points[:, 2]).astype(int) == z]
            if pts_z.size > 0:
                ax.scatter(pts_z[:, 0], pts_z[:, 1], s=8, c="yellow")
        ax.set_axis_off()
        if title:
            ax.set_title(f"{title} z={z}")

    _plt.tight_layout()
    if out_path:
        _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def save_error_map(A, points, distances, out_path, num_slices=4):
    if points is None or len(points) == 0:
        return
    z_slices = pick_slices(A.shape[2], n=num_slices)
    fig, axes = _plt.subplots(1, len(z_slices), figsize=(4 * len(z_slices), 4))
    if len(z_slices) == 1:
        axes = [axes]

    for i, z in enumerate(z_slices):
        ax = axes[i]
        ax.imshow(A[:, :, z].T, cmap="gray", origin="lower")
        pts_z_idx = np.where(np.round(points[:, 2]).astype(int) == z)[0]
        if pts_z_idx.size > 0:
            pts = points[pts_z_idx]
            d = distances[pts_z_idx]
            sc = ax.scatter(pts[:, 0], pts[:, 1], s=10, c=d, cmap="hot")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_axis_off()
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def compute_distances(points_vox, curve_mask, spacing):
    if points_vox is None or len(points_vox) == 0:
        return np.array([], dtype=np.float32)
    curve_pts = np.array(np.where(curve_mask > 0)).T.astype(np.float32)
    if curve_pts.shape[0] == 0:
        return np.full((len(points_vox),), np.inf, dtype=np.float32)
    spacing = np.asarray(spacing, dtype=np.float32)
    curve_mm = curve_pts * spacing
    pts_mm = np.asarray(points_vox, dtype=np.float32) * spacing
    tree = cKDTree(curve_mm)
    d, _ = tree.query(pts_mm, k=1)
    return d.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    enable_2d = bool(cfg.get("viz", {}).get("enable_2d", False))
    if not enable_2d:
        logger.info("2D viz disabled (viz.enable_2d=false); skipping.")
        return

    _ensure_viz_deps()
    processed_dir = cfg["data"]["processed_dir"]
    infer_dir = os.path.join(cfg["data"]["output_dir"], "infer")
    out_root = ensure_dir(os.path.join(cfg["data"]["output_dir"], "viz"))

    tooth_dirs = sorted(glob.glob(os.path.join(processed_dir, "*", "tooth_*")))
    if not tooth_dirs:
        logger.warning("no processed teeth found")
        return

    num_slices = int(cfg["viz"].get("num_slices", 4))
    for tdir in tooth_dirs:
        roi_meta_path = os.path.join(tdir, "roi_meta.json")
        with open(roi_meta_path, "r") as f:
            roi_meta = json.load(f)
        case_id = roi_meta["case_id"]
        tooth_id = roi_meta["tooth_id"]

        fmt = cfg["data"]["processed_format"]
        A, spacing, affine = load_volume(os.path.join(tdir, f"A_t.{fmt}"), dtype=np.float32)
        spacing = _ensure_spacing(spacing, affine)
        T, _, _ = load_volume(os.path.join(tdir, f"T_t.{fmt}"), dtype=np.uint8)
        H, _, _ = load_volume(os.path.join(tdir, f"H_GT.{fmt}"), dtype=np.float32)

        points_data = load_points(os.path.join(tdir, "points.json"))
        pts = get_points_for_tooth(points_data, tooth_id)

        # ROI check: boundary overlay
        boundary = np.zeros_like(T)
        for z in range(T.shape[2]):
            boundary[:, :, z] = boundary2d(T[:, :, z])
        out_dir_roi = ensure_dir(os.path.join(out_root, "roi", case_id, f"tooth_{tooth_id}"))
        save_overlay(
            A,
            overlay=boundary,
            out_path=os.path.join(out_dir_roi, "roi.png"),
            title="roi",
            num_slices=num_slices,
        )

        # Prior check (R_t)
        R = compute_geometric_prior(
            A,
            T,
            spacing,
            sigma_z_mm=cfg["infer"]["prior_sigma_z_mm"],
            sigma_s_mm=cfg["infer"]["prior_sigma_s_mm"],
            delta_s_mm=cfg["infer"]["prior_delta_s_mm"],
            z_ignore_ratio=cfg["preprocess"]["z_ignore_ratio"],
            use_gradient=cfg["infer"]["prior_use_gradient"],
            gradient_weight=cfg["infer"]["prior_gradient_weight"],
        )
        out_dir_prior = ensure_dir(os.path.join(out_root, "prior", case_id, f"tooth_{tooth_id}"))
        save_overlay(
            A,
            overlay=R,
            out_path=os.path.join(out_dir_prior, "prior.png"),
            title="prior",
            num_slices=num_slices,
        )

        # Pseudo-GT check
        out_dir_pgt = ensure_dir(os.path.join(out_root, "pseudo_gt", case_id, f"tooth_{tooth_id}"))
        save_overlay(
            A,
            overlay=H,
            points=pts,
            out_path=os.path.join(out_dir_pgt, "pseudo_gt.png"),
            title="pseudo_gt",
            num_slices=num_slices,
        )

        # Inference check
        h_pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "H_pred.nii.gz")
        c_pred_path = os.path.join(infer_dir, case_id, f"tooth_{tooth_id}", "C_pred.nii.gz")
        if os.path.exists(h_pred_path) and os.path.exists(c_pred_path):
            H_pred, _, _ = load_volume(h_pred_path, dtype=np.float32)
            C_pred, _, _ = load_volume(c_pred_path, dtype=np.uint8)
            out_dir_inf = ensure_dir(os.path.join(out_root, "infer", case_id, f"tooth_{tooth_id}"))
            save_overlay(
                A,
                overlay=H_pred + C_pred,
                out_path=os.path.join(out_dir_inf, "infer.png"),
                title="infer",
                num_slices=num_slices,
            )

            # Error map
            d = compute_distances(pts, C_pred, spacing)
            out_dir_err = ensure_dir(os.path.join(out_root, "error", case_id, f"tooth_{tooth_id}"))
            save_error_map(A, pts, d, os.path.join(out_dir_err, "error.png"), num_slices=num_slices)

        logger.info("viz case=%s tooth=%s", case_id, tooth_id)


if __name__ == "__main__":
    main()
def _ensure_spacing(spacing, affine):
    if spacing is not None:
        return spacing
    if affine is None:
        return (1.0, 1.0, 1.0)
    return tuple(affine_to_spacing(affine))
