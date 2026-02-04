import json
import os
import numpy as np
from src.utils.geometry import world_to_vox


def load_points(path):
    if not os.path.exists(path):
        return {"case_id": None, "coord_type": "voxel", "space": "full", "points": {}}
    with open(path, "r") as f:
        return json.load(f)


def save_points(path, case_id, coord_type, space, points_dict):
    data = {
        "case_id": case_id,
        "coord_type": coord_type,
        "space": space,
        "points": points_dict,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def get_points_for_tooth(points_data, tooth_id):
    pts = points_data.get("points", {}).get(str(tooth_id), [])
    return np.asarray(pts, dtype=np.float32)


def ensure_voxel_points(points, coord_type, affine):
    if coord_type == "world":
        return world_to_vox(points, affine)
    return np.asarray(points, dtype=np.float32)
