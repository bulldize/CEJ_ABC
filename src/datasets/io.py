import json
import os
import numpy as np
import nibabel as nib


def _default_affine(spacing):
    aff = np.eye(4, dtype=np.float32)
    aff[0, 0] = spacing[0]
    aff[1, 1] = spacing[1]
    aff[2, 2] = spacing[2]
    return aff


def load_volume(path, meta_path=None, dtype=None):
    if path.endswith(".npy"):
        arr = np.load(path)
        spacing = (1.0, 1.0, 1.0)
        affine = _default_affine(spacing)
        sidecar = path.replace(".npy", ".meta.json")
        meta_candidate = meta_path if meta_path and os.path.exists(meta_path) else sidecar
        if meta_candidate and os.path.exists(meta_candidate):
            with open(meta_candidate, "r") as f:
                meta = json.load(f)
            spacing = tuple(meta.get("spacing", spacing))
            affine = np.array(meta.get("affine", affine)).astype(np.float32)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr, spacing, affine

    img = nib.load(path)
    arr = img.get_fdata()
    if dtype is not None:
        arr = arr.astype(dtype)
    spacing = img.header.get_zooms()[:3]
    affine = img.affine.astype(np.float32)
    return arr, spacing, affine


def save_volume(path, arr, affine=None, spacing=None, dtype=None):
    if dtype is not None:
        arr = arr.astype(dtype)
    if path.endswith(".npy"):
        np.save(path, arr)
        if spacing is not None:
            meta_path = path.replace(".npy", ".meta.json")
            with open(meta_path, "w") as f:
                json.dump({"spacing": list(spacing), "affine": affine.tolist() if affine is not None else None}, f)
        return

    if affine is None:
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        affine = _default_affine(spacing)
    img = nib.Nifti1Image(arr, affine)
    nib.save(img, path)
