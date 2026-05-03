import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


MRK_JSON_SUFFIX = ".mrk.json"
IGNORE_FILES_NAME = "cej_ignore_files.txt"


def choose_existing_path(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _ignored_names(path: Path) -> set:
    base_dir = path.parent if path.is_file() else path
    ignore_path = base_dir / IGNORE_FILES_NAME
    if not ignore_path.exists():
        return set()
    names = set()
    for line in ignore_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            names.add(item)
    return names


def list_mrk_json_files(path: Path) -> List[Path]:
    ignored = _ignored_names(path)
    if path.is_file() and path.name.endswith(MRK_JSON_SUFFIX):
        if path.name in ignored:
            return []
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        [
            p
            for p in path.iterdir()
            if p.is_file() and p.name.endswith(MRK_JSON_SUFFIX) and p.name not in ignored
        ]
    )


def has_mrk_json_files(path: Path) -> bool:
    return len(list_mrk_json_files(path)) > 0


def detect_points_source(manual_case_dir: Path) -> Optional[Path]:
    excel_path = choose_existing_path(
        [
            manual_case_dir / "cej_points_ras.xlsx",
            manual_case_dir / "points.xlsx",
        ]
    )
    if excel_path is not None:
        return excel_path
    if has_mrk_json_files(manual_case_dir):
        return manual_case_dir
    return None


def is_mrk_source(path: Path) -> bool:
    return path.name.endswith(MRK_JSON_SUFFIX) or has_mrk_json_files(path)


def _extract_tooth_id(text: str) -> Optional[str]:
    match = re.search(r"(\d{2})", str(text))
    if match is None:
        return None
    return str(int(match.group(1)))


def _extract_prefix(text: str) -> Optional[str]:
    match = re.search(r"\b([FP])[_\-]?", str(text), flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper()


def find_tooth_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if "牙位" in str(c):
            return c
    for c in df.columns:
        if "tooth" in str(c).lower():
            return c
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"group", "grp"} or "group" in lc:
            return c
    return None


def find_xyz_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}

    def pick_by_patterns(patterns: List[str]) -> Optional[str]:
        for pattern in patterns:
            for c, lc in lower.items():
                if re.search(pattern, lc):
                    return c
        return None

    x_col = pick_by_patterns([r"position\s*\[0\]", r"\bx\b"])
    y_col = pick_by_patterns([r"position\s*\[1\]", r"\by\b"])
    z_col = pick_by_patterns([r"position\s*\[2\]", r"\bz\b"])
    if x_col and y_col and z_col:
        return x_col, y_col, z_col

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 3:
        return num_cols[0], num_cols[1], num_cols[2]
    raise ValueError("cannot find x/y/z columns")


def infer_prefix_from_source(points_source: Path) -> Optional[str]:
    if is_mrk_source(points_source):
        counts = {"F": 0, "P": 0}
        for mrk_path in list_mrk_json_files(points_source):
            prefix = _extract_prefix(mrk_path.name)
            if prefix in counts:
                counts[prefix] += 1
        if counts["F"] == 0 and counts["P"] == 0:
            try:
                df = load_manual_points_dataframe(points_source)
            except Exception:
                return None
            for value in df.get("label", pd.Series(dtype=str)).dropna().astype(str).head(256):
                prefix = _extract_prefix(value)
                if prefix in counts:
                    counts[prefix] += 1
        if counts["F"] == 0 and counts["P"] == 0:
            return None
        return "F" if counts["F"] >= counts["P"] else "P"

    try:
        df = pd.read_excel(points_source, nrows=128)
    except Exception:
        return None
    tooth_col = find_tooth_col(df)
    if tooth_col is None:
        return None

    counts = {"F": 0, "P": 0}
    for value in df[tooth_col].dropna().astype(str).head(128):
        prefix = _extract_prefix(value)
        if prefix in counts:
            counts[prefix] += 1
    if counts["F"] == 0 and counts["P"] == 0:
        return None
    return "F" if counts["F"] >= counts["P"] else "P"


def load_manual_points_dataframe(points_source: Path) -> pd.DataFrame:
    if not is_mrk_source(points_source):
        return pd.read_excel(points_source)

    rows: List[Dict[str, Any]] = []
    for mrk_path in list_mrk_json_files(points_source):
        with mrk_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        markups = data.get("markups", [])
        file_tooth = _extract_tooth_id(mrk_path.name.replace(MRK_JSON_SUFFIX, ""))
        for markup in markups:
            coord_system = str(markup.get("coordinateSystem", "")).upper()
            control_points = markup.get("controlPoints", []) or []
            for idx, control_point in enumerate(control_points):
                label = str(control_point.get("label", ""))
                tooth_id = file_tooth or _extract_tooth_id(label)
                position = control_point.get("position", None)
                x = y = z = None
                if isinstance(position, list) and len(position) >= 3:
                    x, y, z = position[:3]
                rows.append(
                    {
                        "tooth": tooth_id,
                        "x": x,
                        "y": y,
                        "z": z,
                        "order": idx,
                        "label": label,
                        "coord_system": coord_system,
                        "source_file": str(mrk_path),
                    }
                )
    return pd.DataFrame(rows, columns=["tooth", "x", "y", "z", "order", "label", "coord_system", "source_file"])


def load_manual_points_table(points_source: Path):
    df = load_manual_points_dataframe(points_source)
    if is_mrk_source(points_source):
        return df, "tooth", "x", "y", "z", "order", {"source_format": "slicer_mrk_json"}

    tooth_col = find_tooth_col(df)
    if tooth_col is None:
        raise ValueError(f"cannot find tooth/group column in {points_source}")
    x_col, y_col, z_col = find_xyz_cols(df)
    order_col = None
    for c in df.columns:
        if "点位" in str(c):
            order_col = c
            break
    return df, tooth_col, x_col, y_col, z_col, order_col, {"source_format": "excel"}


def source_meta(points_source: Path) -> Dict[str, Any]:
    if is_mrk_source(points_source):
        return {
            "kind": "mrk_dir",
            "path": str(points_source),
            "files": [
                {
                    "name": p.name,
                    "size": int(p.stat().st_size),
                    "mtime_ns": int(p.stat().st_mtime_ns),
                }
                for p in list_mrk_json_files(points_source)
            ],
            "ignored_files": sorted(_ignored_names(points_source)),
        }

    st = points_source.stat()
    return {
        "kind": "file",
        "path": str(points_source),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
