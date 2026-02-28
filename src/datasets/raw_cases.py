import glob
import os
from typing import Dict, List, Optional


def resolve_nifti_path(path: str) -> str:
    if not os.path.isdir(path):
        return path
    base = os.path.basename(path)
    candidate = os.path.join(path, base)
    if os.path.exists(candidate):
        return candidate
    for name in os.listdir(path):
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return os.path.join(path, name)
    return path


def _strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _to_case_id_from_image_name(name: str) -> str:
    stem = _strip_nii_ext(name)
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def _to_layout(cfg_data: dict) -> str:
    layout = str(cfg_data.get("raw_layout", "auto")).lower()
    if layout in {"auto", "case_dirs", "toothfairy3"}:
        return layout
    return "auto"


def _maybe_limit_cases(cases: List[Dict], cfg_data: dict) -> List[Dict]:
    max_cases = cfg_data.get("max_cases", None)
    if max_cases is not None:
        try:
            max_cases = int(max_cases)
            if max_cases > 0:
                cases = cases[:max_cases]
        except Exception:
            pass
    return cases


def _collect_cases_from_case_dirs(cfg_data: dict) -> List[Dict]:
    raw_dir = cfg_data["raw_dir"]
    raw_a_name = cfg_data["raw_a_name"]
    raw_b_name = cfg_data["raw_b_name"]
    raw_points_name = cfg_data.get("raw_points_name", "points.json")
    raw_meta_name = cfg_data.get("raw_meta_name", "meta.json")

    out = []
    case_dirs = sorted([d for d in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(d)])
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        a_path = resolve_nifti_path(os.path.join(case_dir, raw_a_name))
        b_path = resolve_nifti_path(os.path.join(case_dir, raw_b_name))
        out.append(
            {
                "case_id": case_id,
                "case_dir": case_dir,
                "a_path": a_path,
                "b_path": b_path,
                "points_path": os.path.join(case_dir, raw_points_name),
                "meta_path": os.path.join(case_dir, raw_meta_name),
                "layout": "case_dirs",
            }
        )
    return out


def _collect_cases_from_toothfairy3(cfg_data: dict) -> List[Dict]:
    raw_dir = cfg_data["raw_dir"]
    images_tr = cfg_data.get("images_tr_dir") or os.path.join(raw_dir, "imagesTr")
    labels_tr = cfg_data.get("labels_tr_dir") or os.path.join(raw_dir, "labelsTr")
    points_dir = cfg_data.get("points_dir") or os.path.join(raw_dir, "pointsTr")
    meta_dir = cfg_data.get("meta_dir", None)

    if not os.path.isdir(images_tr) or not os.path.isdir(labels_tr):
        return []

    out = []
    image_paths = sorted(glob.glob(os.path.join(images_tr, "*.nii")) + glob.glob(os.path.join(images_tr, "*.nii.gz")))
    for a_path in image_paths:
        image_name = os.path.basename(a_path)
        case_id = _to_case_id_from_image_name(image_name)
        b_candidates = [
            os.path.join(labels_tr, f"{case_id}.nii.gz"),
            os.path.join(labels_tr, f"{case_id}.nii"),
        ]
        b_path = None
        for p in b_candidates:
            if os.path.exists(p):
                b_path = p
                break
        if b_path is None:
            continue

        points_path = os.path.join(points_dir, f"{case_id}.json")
        meta_path = None
        if meta_dir:
            meta_path = os.path.join(meta_dir, f"{case_id}.json")

        out.append(
            {
                "case_id": case_id,
                "case_dir": None,
                "a_path": a_path,
                "b_path": b_path,
                "points_path": points_path,
                "meta_path": meta_path,
                "layout": "toothfairy3",
                "images_tr_dir": images_tr,
                "labels_tr_dir": labels_tr,
            }
        )
    return out


def collect_raw_cases(cfg_data: dict) -> List[Dict]:
    layout = _to_layout(cfg_data)
    if layout == "case_dirs":
        cases = _collect_cases_from_case_dirs(cfg_data)
        return _maybe_limit_cases(cases, cfg_data)
    if layout == "toothfairy3":
        cases = _collect_cases_from_toothfairy3(cfg_data)
        return _maybe_limit_cases(cases, cfg_data)

    # auto mode
    cases = _collect_cases_from_toothfairy3(cfg_data)
    if not cases:
        cases = _collect_cases_from_case_dirs(cfg_data)
    return _maybe_limit_cases(cases, cfg_data)


def collect_raw_case_map(cfg_data: dict) -> Dict[str, Dict]:
    cases = collect_raw_cases(cfg_data)
    return {c["case_id"]: c for c in cases}


def get_raw_case_record(cfg_data: dict, case_id: str) -> Optional[Dict]:
    return collect_raw_case_map(cfg_data).get(case_id)
