#!/usr/bin/env python3
"""Analyze CEJ curve failures and low-confidence unlabeled predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import cKDTree


DEFAULT_PROCESSED_DIR = Path("/Users/bulldize/cej_runtime/run_manual_inc_041055_001/processed")
DEFAULT_INFER_DIR = Path(
    "/Users/bulldize/cej_runtime/run_sup_inc_041055_001/outputs_current_postprocess/infer"
)
DEFAULT_EVAL_CSV = Path("/Users/bulldize/cej_runtime/run_sup_inc_041055_001/outputs/eval_quick_pred_fit.csv")
DEFAULT_METRICS_CSV = Path("/Users/bulldize/cej_runtime/run_sup_inc_041055_001/outputs/train/metrics.csv")
DEFAULT_MANUAL_REPORT = Path("/Users/bulldize/cej_runtime/run_manual_inc_041055_001/manual_points_usage_report.json")
DEFAULT_OUTPUT_DIR = Path(
    "/Users/bulldize/cej_runtime/run_sup_inc_041055_001/outputs_current_postprocess/analysis"
)


@dataclass(frozen=True)
class Thresholds:
    heatmap_threshold: float = 0.30
    large_sym_p95_mm: float = 1.50
    large_sym_mean_mm: float = 0.80
    severe_sym_p95_mm: float = 3.00
    min_pred_curve_points: int = 16
    severe_min_pred_curve_points: int = 8
    tiny_heatmap_voxels: int = 100
    fragmented_lcc_ratio: float = 0.80
    low_wrap_coverage: float = 0.45


def tooth_category(tooth_id: str) -> str:
    digit = str(tooth_id)[-1]
    return {
        "1": "切牙",
        "2": "切牙",
        "3": "尖牙",
        "4": "前磨牙",
        "5": "前磨牙",
        "6": "磨牙",
        "7": "磨牙",
        "8": "智齿",
    }.get(digit, "未知")


def tooth_root_group(tooth_id: str) -> str:
    return "多根" if str(tooth_id)[-1] in {"6", "7", "8"} else "单根"


def format_mm(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "inf"
    return f"{val:.3f}"


def normalize_tooth_id(tooth_id: Any) -> str:
    return str(int(tooth_id)) if str(tooth_id).strip().isdigit() else str(tooth_id)


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_manual_points_data(data: Any, tooth_id: Any) -> int:
    tooth_key = normalize_tooth_id(tooth_id)
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return 0

    points = data.get("points")
    if isinstance(points, dict):
        candidates = [tooth_key, str(tooth_id), int(tooth_key) if tooth_key.isdigit() else tooth_key]
        for key in candidates:
            if key in points and isinstance(points[key], list):
                return len(points[key])
        return 0
    if isinstance(points, list):
        return len(points)

    control_points = data.get("controlPoints")
    if isinstance(control_points, list):
        return len(control_points)

    markups = data.get("markups")
    if isinstance(markups, list):
        return sum(
            len(markup.get("controlPoints", []) or [])
            for markup in markups
            if isinstance(markup, dict)
        )
    return 0


def count_manual_points(points_path: Path, tooth_id: str) -> int:
    data = load_json(points_path)
    if data is None:
        return 0
    return count_manual_points_data(data, tooth_id)


def load_nifti_array(path: Path, dtype: Optional[np.dtype] = None) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.asarray(nib.load(str(path)).dataobj)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def load_spacing(tooth_dir: Path) -> np.ndarray:
    for name in ("T_t.nii.gz", "A_t.nii.gz", "H_GT.nii.gz"):
        path = tooth_dir / name
        if path.exists():
            return np.asarray(nib.load(str(path)).header.get_zooms()[:3], dtype=np.float32)
    return np.ones(3, dtype=np.float32)


def load_curve(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(np.load(path), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    return pts


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return math.inf
    return float(np.percentile(values, q))


def curve_distance_stats(
    gt_curve_vox: np.ndarray,
    pred_curve_vox: np.ndarray,
    spacing: Sequence[float],
) -> Dict[str, Any]:
    gt_curve_vox = np.asarray(gt_curve_vox, dtype=np.float32)
    pred_curve_vox = np.asarray(pred_curve_vox, dtype=np.float32)
    spacing_arr = np.asarray(spacing, dtype=np.float32)

    if gt_curve_vox.size == 0 or pred_curve_vox.size == 0:
        return {
            "n_gt_curve": int(gt_curve_vox.shape[0]) if gt_curve_vox.ndim == 2 else 0,
            "n_pred_curve": int(pred_curve_vox.shape[0]) if pred_curve_vox.ndim == 2 else 0,
            "gt_to_pred_mean": math.inf,
            "gt_to_pred_p95": math.inf,
            "gt_to_pred_max": math.inf,
            "pred_to_gt_mean": math.inf,
            "pred_to_gt_p95": math.inf,
            "pred_to_gt_max": math.inf,
            "sym_mean": math.inf,
            "sym_p95": math.inf,
            "sym_max": math.inf,
        }

    gt_mm = gt_curve_vox * spacing_arr
    pred_mm = pred_curve_vox * spacing_arr
    pred_tree = cKDTree(pred_mm)
    gt_tree = cKDTree(gt_mm)
    gt_to_pred = pred_tree.query(gt_mm, k=1)[0]
    pred_to_gt = gt_tree.query(pred_mm, k=1)[0]
    sym = np.concatenate([gt_to_pred, pred_to_gt])
    return {
        "n_gt_curve": int(gt_curve_vox.shape[0]),
        "n_pred_curve": int(pred_curve_vox.shape[0]),
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "gt_to_pred_p95": percentile(gt_to_pred, 95),
        "gt_to_pred_max": float(gt_to_pred.max()),
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "pred_to_gt_p95": percentile(pred_to_gt, 95),
        "pred_to_gt_max": float(pred_to_gt.max()),
        "sym_mean": float(sym.mean()),
        "sym_p95": percentile(sym, 95),
        "sym_max": float(sym.max()),
    }


def angular_coverage(mask: Optional[np.ndarray], coords: np.ndarray, spacing: Sequence[float], bins: int = 36) -> float:
    if mask is None or coords.shape[0] < 3:
        return 0.0
    tooth_coords = np.argwhere(mask > 0)
    if tooth_coords.shape[0] < 10:
        return 0.0

    spacing_arr = np.asarray(spacing, dtype=np.float32)
    center = tooth_coords.mean(axis=0)
    centered_tooth = (tooth_coords - center) * spacing_arr
    try:
        _, _, vt = np.linalg.svd(centered_tooth, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if vt.shape[0] < 3:
        return 0.0

    plane_basis = vt[1:3].T
    projected = ((coords - center) * spacing_arr) @ plane_basis
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    bucket = np.floor((angles + np.pi) / (2.0 * np.pi) * bins).astype(np.int32)
    bucket = np.clip(bucket, 0, bins - 1)
    return float(len(set(bucket.tolist())) / bins)


def heatmap_diagnostics(
    heatmap: Optional[np.ndarray],
    tooth_mask: Optional[np.ndarray],
    spacing: Sequence[float],
    thresholds: Thresholds,
) -> Dict[str, Any]:
    if heatmap is None or heatmap.size == 0:
        return {
            "hmax": 0.0,
            "p99": 0.0,
            "vox03": 0,
            "cc_count": 0,
            "lcc_ratio": 0.0,
            "wrap_coverage": 0.0,
            "flag_low_conf": True,
            "flag_tiny": True,
            "flag_fragmented": False,
            "flag_low_wrap": True,
            "flag_heatmap_issue": True,
        }

    hmax = float(np.max(heatmap))
    p99 = float(np.percentile(heatmap, 99))
    mask03 = heatmap >= thresholds.heatmap_threshold
    vox03 = int(mask03.sum())
    cc_count = 0
    lcc_ratio = 0.0
    wrap_coverage = 0.0
    if vox03 > 0:
        labeled, cc_count = label(mask03, structure=generate_binary_structure(3, 3))
        sizes = np.bincount(labeled.ravel())[1:]
        if sizes.size:
            lcc_ratio = float(sizes.max() / vox03)
        wrap_coverage = angular_coverage(tooth_mask, np.argwhere(mask03), spacing)

    flag_low_conf = hmax < thresholds.heatmap_threshold or vox03 == 0
    flag_tiny = vox03 < thresholds.tiny_heatmap_voxels
    flag_fragmented = cc_count > 1 and lcc_ratio < thresholds.fragmented_lcc_ratio
    flag_low_wrap = wrap_coverage < thresholds.low_wrap_coverage
    return {
        "hmax": hmax,
        "p99": p99,
        "vox03": vox03,
        "cc_count": int(cc_count),
        "lcc_ratio": lcc_ratio,
        "wrap_coverage": wrap_coverage,
        "flag_low_conf": bool(flag_low_conf),
        "flag_tiny": bool(flag_tiny),
        "flag_fragmented": bool(flag_fragmented),
        "flag_low_wrap": bool(flag_low_wrap),
        "flag_heatmap_issue": bool(flag_low_conf or flag_tiny or flag_fragmented or flag_low_wrap),
    }


def load_eval_rows(eval_csv: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not eval_csv.exists():
        return {}
    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        return {
            (row.get("case_id", ""), normalize_tooth_id(row.get("tooth_id", ""))): row
            for row in csv.DictReader(f)
        }


def summarize_metrics(metrics_csv: Path) -> Dict[str, Any]:
    if not metrics_csv.exists():
        return {"exists": False}
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    losses = []
    for row in rows:
        try:
            losses.append(float(row.get("loss", "")))
        except ValueError:
            pass
    if not losses:
        return {"exists": True, "epochs": len(rows)}
    return {
        "exists": True,
        "epochs": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "delta_loss": losses[-1] - losses[0],
        "last5": losses[-5:],
    }


def load_manual_report_summary(manual_report: Path) -> Dict[str, Any]:
    data = load_json(manual_report)
    if not isinstance(data, dict):
        return {"exists": False}
    summary = data.get("summary", {})
    cases = data.get("cases", [])
    return {
        "exists": True,
        "summary": summary,
        "case_count": len(cases) if isinstance(cases, list) else 0,
    }


def has_gt_heatmap(tooth_dir: Path) -> bool:
    hgt = load_nifti_array(tooth_dir / "H_GT.nii.gz", dtype=np.float32)
    return bool(hgt is not None and hgt.size > 0 and float(np.max(hgt)) > 0)


def classify_labeled_row(row: Dict[str, Any], thresholds: Thresholds) -> Dict[str, bool]:
    n_pred = int(row.get("n_pred_curve", 0) or 0)
    sym_mean = float(row.get("sym_mean", math.inf))
    sym_p95 = float(row.get("sym_p95", math.inf))
    large = (
        sym_p95 > thresholds.large_sym_p95_mm
        or sym_mean > thresholds.large_sym_mean_mm
        or n_pred < thresholds.min_pred_curve_points
    )
    severe = sym_p95 > thresholds.severe_sym_p95_mm or n_pred < thresholds.severe_min_pred_curve_points
    return {"flag_large_error": bool(large), "flag_severe_error": bool(severe)}


def analyze_run(
    processed_dir: Path,
    infer_dir: Path,
    eval_csv: Path,
    metrics_csv: Path,
    manual_report: Path,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    eval_rows = load_eval_rows(eval_csv)
    labeled_rows: List[Dict[str, Any]] = []
    no_manual_rows: List[Dict[str, Any]] = []

    for case_dir in sorted(processed_dir.glob("*")):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        for tooth_dir in sorted(case_dir.glob("tooth_*")):
            if not tooth_dir.is_dir():
                continue
            tooth_id = normalize_tooth_id(tooth_dir.name.split("_")[-1])
            infer_tooth_dir = infer_dir / case_id / tooth_dir.name
            spacing = load_spacing(tooth_dir)
            h_pred = load_nifti_array(infer_tooth_dir / "H_pred.nii.gz", dtype=np.float32)
            tooth_mask = load_nifti_array(tooth_dir / "T_t.nii.gz", dtype=np.uint8)
            heat = heatmap_diagnostics(h_pred, tooth_mask, spacing, thresholds)
            n_manual_points = count_manual_points(tooth_dir / "points.json", tooth_id)
            has_manual = n_manual_points > 0 or has_gt_heatmap(tooth_dir)
            base = {
                "case_id": case_id,
                "tooth_id": tooth_id,
                "tooth_key": f"{case_id}-{tooth_id}",
                "category": tooth_category(tooth_id),
                "root_group": tooth_root_group(tooth_id),
                "n_manual_points": n_manual_points,
                **heat,
            }

            if has_manual:
                gt_curve = load_curve(tooth_dir / "curve_dense_points.npy")
                pred_curve = load_curve(infer_tooth_dir / "curve_pred_dense_points.npy")
                dist = curve_distance_stats(gt_curve, pred_curve, spacing)
                row = {**base, **dist}
                eval_row = eval_rows.get((case_id, tooth_id), {})
                for key in ("fit_mean", "fit_p95", "fit_sr05", "fit_sr10", "fit_sr15", "n_fit"):
                    row[f"eval_{key}"] = eval_row.get(key, "")
                row.update(classify_labeled_row(row, thresholds))
                labeled_rows.append(row)
            else:
                row = {**base}
                row["issue_reasons"] = issue_reason_text(row)
                no_manual_rows.append(row)

    labeled_rows.sort(key=lambda r: (float(r.get("sym_p95", -1)), float(r.get("sym_mean", -1))), reverse=True)
    no_manual_rows.sort(
        key=lambda r: (
            0 if r["flag_low_conf"] else 1,
            float(r["hmax"]),
            int(r["vox03"]),
            float(r["wrap_coverage"]),
        )
    )
    return {
        "thresholds": thresholds,
        "labeled_rows": labeled_rows,
        "no_manual_rows": no_manual_rows,
        "metrics_summary": summarize_metrics(metrics_csv),
        "manual_report_summary": load_manual_report_summary(manual_report),
    }


def issue_reason_text(row: Dict[str, Any]) -> str:
    reasons = []
    if row.get("flag_low_conf"):
        reasons.append("低置信")
    if row.get("flag_tiny"):
        reasons.append("热图很小")
    if row.get("flag_fragmented"):
        reasons.append("不连通")
    if row.get("flag_low_wrap"):
        reasons.append("不包裹")
    return "、".join(reasons) if reasons else "正常"


def labeled_issue_text(row: Dict[str, Any]) -> str:
    reasons = []
    if row.get("flag_severe_error"):
        reasons.append("严重")
    if row.get("flag_large_error"):
        reasons.append("大误差")
    if int(row.get("n_pred_curve", 0) or 0) < 16:
        reasons.append("预测曲线点少")
    if row.get("flag_fragmented"):
        reasons.append("热图碎裂")
    if row.get("flag_low_wrap"):
        reasons.append("不包裹")
    return "、".join(dict.fromkeys(reasons)) if reasons else "可接受"


def build_category_summary(
    labeled_rows: Sequence[Dict[str, Any]],
    no_manual_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    for row in labeled_rows:
        g = grouped[row["category"]]
        g["category"] = row["category"]
        g["total"] += 1
        g["labeled"] += 1
        g["large_error"] += int(bool(row.get("flag_large_error")))
        g["severe_error"] += int(bool(row.get("flag_severe_error")))
        g.setdefault("sym_p95_values", []).append(float(row.get("sym_p95", math.inf)))
        g.setdefault("sym_mean_values", []).append(float(row.get("sym_mean", math.inf)))
    for row in no_manual_rows:
        g = grouped[row["category"]]
        g["category"] = row["category"]
        g["total"] += 1
        g["no_manual"] += 1
        g["no_manual_flagged"] += int(bool(row.get("flag_heatmap_issue")))
        g["low_conf"] += int(bool(row.get("flag_low_conf")))
        g["fragmented"] += int(bool(row.get("flag_fragmented")))
        g["low_wrap"] += int(bool(row.get("flag_low_wrap")))

    order = {"切牙": 0, "尖牙": 1, "前磨牙": 2, "磨牙": 3, "智齿": 4, "未知": 5}
    out = []
    for category, data in grouped.items():
        p95_values = [v for v in data.get("sym_p95_values", []) if math.isfinite(v)]
        mean_values = [v for v in data.get("sym_mean_values", []) if math.isfinite(v)]
        out.append(
            {
                "category": category,
                "total": int(data.get("total", 0)),
                "labeled": int(data.get("labeled", 0)),
                "large_error": int(data.get("large_error", 0)),
                "severe_error": int(data.get("severe_error", 0)),
                "no_manual": int(data.get("no_manual", 0)),
                "no_manual_flagged": int(data.get("no_manual_flagged", 0)),
                "low_conf": int(data.get("low_conf", 0)),
                "fragmented": int(data.get("fragmented", 0)),
                "low_wrap": int(data.get("low_wrap", 0)),
                "median_sym_mean": float(np.median(mean_values)) if mean_values else "",
                "median_sym_p95": float(np.median(p95_values)) if p95_values else "",
            }
        )
    return sorted(out, key=lambda r: order.get(r["category"], 99))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]], limit: int = 20) -> str:
    if not rows:
        return "无。\n"
    clipped = list(rows[:limit])
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in clipped:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if key in {"hmax", "p99", "lcc_ratio", "wrap_coverage"}:
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(format_mm(value))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def annotation_priority(labeled_rows: Sequence[Dict[str, Any]], no_manual_rows: Sequence[Dict[str, Any]]) -> List[str]:
    preset = {"052-31", "052-32", "052-41", "052-42", "052-33", "041-31", "041-41", "041-42", "052-22"}
    severe_labeled = [r for r in labeled_rows if r.get("flag_severe_error")]
    preset_labeled = [
        r
        for r in labeled_rows
        if f"{r['case_id'].split('_')[-1]}-{r['tooth_id']}" in preset and r not in severe_labeled
    ]
    labeled_priority = (severe_labeled + preset_labeled)[:12]
    low_conf = [r for r in no_manual_rows if r.get("flag_low_conf")][:12]
    fragmented = [
        r
        for r in no_manual_rows
        if not r.get("flag_low_conf") and (r.get("flag_fragmented") or r.get("flag_low_wrap"))
    ][:12]
    lines = []
    if labeled_priority:
        lines.append(
            "1. 先复核或重标有标注但曲线大误差牙："
            + "、".join(f"{r['case_id'].split('_')[-1]}-{r['tooth_id']}" for r in labeled_priority[:10])
            + "。"
        )
    if low_conf:
        lines.append(
            "2. 补无标注且 0.30 阈值下几乎无热图的牙："
            + "、".join(f"{r['case_id'].split('_')[-1]}-{r['tooth_id']}" for r in low_conf[:10])
            + "。"
        )
    if fragmented:
        lines.append(
            "3. 再补高 max 但碎裂/不包裹的无标注牙："
            + "、".join(f"{r['case_id'].split('_')[-1]}-{r['tooth_id']}" for r in fragmented[:10])
            + "。"
        )
    lines.append("4. 类别上优先下前牙 31/32/41/42，其次智齿 18/28/38/48，再补前磨牙中的低置信样本。")
    return lines


def render_failure_report(analysis: Dict[str, Any], args: argparse.Namespace) -> str:
    labeled_rows = analysis["labeled_rows"]
    no_manual_rows = analysis["no_manual_rows"]
    category_rows = build_category_summary(labeled_rows, no_manual_rows)
    large_rows = [r for r in labeled_rows if r.get("flag_large_error")]
    severe_rows = [r for r in labeled_rows if r.get("flag_severe_error")]
    preset_review_keys = {"052-31", "052-32", "052-41", "052-42", "052-33", "041-31", "041-41", "041-42", "052-22"}
    preset_review_rows = [
        r
        for r in labeled_rows
        if f"{r['case_id'].split('_')[-1]}-{r['tooth_id']}" in preset_review_keys and not r.get("flag_large_error")
    ]
    no_manual_flagged = [r for r in no_manual_rows if r.get("flag_heatmap_issue")]
    low_conf = [r for r in no_manual_rows if r.get("flag_low_conf")]
    metrics = analysis["metrics_summary"]
    manual_report = analysis["manual_report_summary"]

    tooth_counts = Counter(r["tooth_id"] for r in severe_rows)
    category_counts = Counter(r["category"] for r in no_manual_flagged)
    loss_line = "未找到训练 metrics。"
    if metrics.get("exists") and "first_loss" in metrics:
        loss_line = (
            f"当前增量训练 loss: {metrics['first_loss']:.4f} -> {metrics['last_loss']:.4f}, "
            f"min={metrics['min_loss']:.4f}, delta={metrics['delta_loss']:.4f}。"
        )
    manual_line = "未找到 manual points usage report。"
    if manual_report.get("exists"):
        summary = manual_report.get("summary", {})
        manual_line = (
            f"manual point 使用报告: cases={summary.get('total_cases', manual_report.get('case_count'))}, "
            f"pass={summary.get('pass_cases')}, all_used={summary.get('all_used')}。"
        )

    for row in labeled_rows:
        row["labeled_issue"] = labeled_issue_text(row)

    labeled_table = markdown_table(
        large_rows,
        [
            ("牙", "tooth_key"),
            ("类别", "category"),
            ("问题", "labeled_issue"),
            ("sym mean", "sym_mean"),
            ("sym p95", "sym_p95"),
            ("sym max", "sym_max"),
            ("pred点数", "n_pred_curve"),
            ("H max", "hmax"),
            ("vox>=0.30", "vox03"),
        ],
        limit=25,
    )
    preset_review_table = markdown_table(
        preset_review_rows,
        [
            ("牙", "tooth_key"),
            ("类别", "category"),
            ("问题", "labeled_issue"),
            ("sym mean", "sym_mean"),
            ("sym p95", "sym_p95"),
            ("sym max", "sym_max"),
            ("pred点数", "n_pred_curve"),
            ("H max", "hmax"),
            ("vox>=0.30", "vox03"),
        ],
        limit=20,
    )
    no_manual_low_table = markdown_table(
        low_conf,
        [
            ("牙", "tooth_key"),
            ("类别", "category"),
            ("原因", "issue_reasons"),
            ("H max", "hmax"),
            ("p99", "p99"),
            ("vox>=0.30", "vox03"),
            ("连通域", "cc_count"),
            ("包裹率", "wrap_coverage"),
        ],
        limit=25,
    )
    no_manual_frag_table = markdown_table(
        [r for r in no_manual_flagged if not r.get("flag_low_conf")],
        [
            ("牙", "tooth_key"),
            ("类别", "category"),
            ("原因", "issue_reasons"),
            ("H max", "hmax"),
            ("vox>=0.30", "vox03"),
            ("连通域", "cc_count"),
            ("LCC占比", "lcc_ratio"),
            ("包裹率", "wrap_coverage"),
        ],
        limit=25,
    )
    category_table = markdown_table(
        category_rows,
        [
            ("类别", "category"),
            ("总数", "total"),
            ("有标注", "labeled"),
            ("大误差", "large_error"),
            ("严重", "severe_error"),
            ("无标注", "no_manual"),
            ("无标注异常", "no_manual_flagged"),
            ("低置信", "low_conf"),
            ("碎裂", "fragmented"),
            ("不包裹", "low_wrap"),
            ("median mean", "median_sym_mean"),
            ("median p95", "median_sym_p95"),
        ],
        limit=20,
    )

    lines = [
        "# CEJ 预测失败分析 041055",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- processed：`{args.processed_dir}`",
        f"- infer：`{args.infer_dir}`",
        f"- 0.30 热图阈值：`{analysis['thresholds'].heatmap_threshold}`",
        "",
        "## 结论摘要",
        "",
        f"- 有标注牙 `{len(labeled_rows)}` 颗，其中大误差 `{len(large_rows)}` 颗，严重 `{len(severe_rows)}` 颗。",
        f"- 无标注牙 `{len(no_manual_rows)}` 颗，其中低置信/碎裂/不包裹 `{len(no_manual_flagged)}` 颗，低置信 `{len(low_conf)}` 颗。",
        f"- {loss_line}",
        f"- {manual_line}",
        "- 主要问题不是单一原因：052 前牙集中大误差需要先复核；无标注低置信说明标注覆盖不足；高 max 但热图碎裂/不包裹说明仅热图 DiceCE 训练目标不够。",
        "",
        "## 有标注但曲线误差大",
        "",
        labeled_table,
        "### 预置复核/边界牙",
        "",
        preset_review_table,
        "重点解读：052 的 31/32/41/42/33 与 041 的 31/41/42 是优先复核对象。若这些牙的手工点本身没问题，就说明模型在下前牙局部形态上泛化不足。",
        "",
        "## 无标注但热图低置信或不包裹",
        "",
        "### 极低置信",
        "",
        no_manual_low_table,
        "### 高置信但碎裂/不包裹",
        "",
        no_manual_frag_table,
        "",
        "## 按类别汇总",
        "",
        category_table,
        "",
        "## 补标优先级",
        "",
        *annotation_priority(labeled_rows, no_manual_rows),
        "",
        "## 原因判断",
        "",
        "- 标注不足：无标注牙里存在大量 `H_pred max < 0.30` 或 `vox>=0.30` 很少的情况，尤其智齿和部分切牙/前磨牙。",
        "- 病例或标注异常：052 前牙成簇失败，且同类牙在其他病例并不都失败，应先打开 viewer 逐颗复核手工点和预测曲线。",
        "- 模型目标不足：当前训练只监督 Gaussian 热图，没有显式曲线连续性、闭环、表面邻域或包裹牙体的结构约束，所以高 max 不等于形成完整 CEJ 曲线。",
        "- 后处理会放大问题：当阈值后只有局部热点或多个碎片时，后处理只能拟合少量点或错误闭环，应把这种情况标记为低可信，而不是当作有效曲线。",
        "",
        "## 输出文件",
        "",
        "- `labeled_curve_error_rank.csv`：有标注牙双向曲线误差排序。",
        "- `no_manual_lowconf_rank.csv`：无标注低置信/碎裂/不包裹排序。",
        "- `tooth_category_summary.csv`：按牙类统计。",
        "- `model_improvement_guidance.md`：模型和训练改进指导。",
    ]
    if tooth_counts:
        lines.insert(
            lines.index("## 原因判断") - 1,
            f"严重误差牙位计数：`{dict(tooth_counts)}`；无标注异常类别计数：`{dict(category_counts)}`。",
        )
    return "\n".join(lines) + "\n"


def render_model_guidance(analysis: Dict[str, Any]) -> str:
    labeled_rows = analysis["labeled_rows"]
    no_manual_rows = analysis["no_manual_rows"]
    severe = [r for r in labeled_rows if r.get("flag_severe_error")]
    low_conf = [r for r in no_manual_rows if r.get("flag_low_conf")]
    fragmented = [r for r in no_manual_rows if r.get("flag_fragmented") or r.get("flag_low_wrap")]
    return f"""# CEJ 模型改进指导

## 当前判断

- 当前训练来自 `codex/manual-points-group-col` 链路：3D UNet 输入 CT ROI + tooth mask，输出单通道 CEJ 热图。
- 本机只负责使用已训练模型做推理、可视化和分析；下面所有训练/模型修改建议均面向云端训练端，不要求在本机训练。
- loss 使用 `DiceCELoss(sigmoid=True)` / `train.loss: dice_bce`，监督目标是由手工点插值曲线生成的 Gaussian `H_GT`。
- 当前 run 的有标注严重误差牙 `{len(severe)}` 颗，无标注低置信牙 `{len(low_conf)}` 颗，无标注碎裂/不包裹牙 `{len(fragmented)}` 颗。
- loss 仍下降，说明继续加入高质量标注大概率有帮助；但只看 train loss 不能判断泛化，因为当前没有 val loss、按牙类指标、曲线连续性指标。
- 需要重点核实云端训练数据语义：如果无标注牙的 `H_GT=0` 也进入 supervised loss，模型会被训练成“这些牙没有 CEJ”，这会放大无标注牙低置信问题。

## 本机使用边界

1. 本机不继续训练模型，不改 checkpoint，只使用已转移的 `last.pt` 做推理、后处理、viewer 和 failure analysis。
2. 本机可以做质量门控：标记低置信、碎裂、不包裹、曲线点太少的牙，避免把低可信曲线当成有效结果。
3. 本机产出的补标清单和 hard case 列表用于指导下一轮云端训练，不在本机闭环训练。

## 第一阶段：先补评价指标

1. 每个 epoch 或每次训练后输出验证表：`case_id/tooth_id/category/sym_mean/sym_p95/H_pred_max/p99/vox03/cc_count/lcc_ratio/wrap_coverage`。
2. 固定 hard case 验证集：`052-31/32/41/42/33`、`041-31/41/42`，以及无标注低置信复核后加入的牙。
3. 报告按牙类分层：切牙、尖牙、前磨牙、磨牙、智齿；不要再只保留一个整体 train loss。
4. 后处理前增加可信度门控：如果 `vox>=0.30` 太少、连通域碎、包裹率低，输出低可信状态，不把拟合曲线当有效预测。
5. supervised 训练集只纳入有非空 `H_GT` / 有 manual points 的牙；无标注牙应被 ignore，而不是作为全背景负样本训练。

## 第二阶段：改训练目标

1. 保留热图 DiceCE 作为主 loss，但加入前景加权 focal/BCE，避免模型只产生局部高亮点。
2. 增加骨架/中心线辅助头或辅助 loss：用 `curve_dense_points.npy` 栅格化的细骨架作为目标，鼓励连续闭环而不是孤立热区。
3. 增加 CEJ 表面邻域软约束：根据牙体 mask 的表面距离生成软权重，惩罚预测概率大面积远离牙体表面；不要强制限制在牙体内部，因为 GT 曲线本身不一定在 mask 内。
4. 增加曲线结构指标驱动的训练/选择：优先优化 angular coverage、连通性、闭环完整度，而不只优化 voxel overlap。
5. 对 hard case 过采样：训练 batch 中提高 `31/32/41/42`、智齿、低置信无标注补标牙的采样概率。

## 第三阶段：补标策略

1. 先复核/重标 052 前牙：`31, 32, 41, 42, 33`。这些牙成簇失败，先排除坐标/标注异常。
2. 补无标注极低置信牙：`044-22, 044-12, 044-25, 051-18, 041-12, 044-24, 055-41, 051-28, 041-22, 041-14`。
3. 系统性补智齿 `18/28/38/48`，因为无标注异常里智齿占比高，且常见高 max 但碎裂。
4. 再补高置信但不包裹/碎裂的前磨牙与智齿，用来教模型完整 CEJ 环而不是只学局部热点。

## 推荐实验顺序

1. Baseline：不改模型，只补上述标注并继续微调，记录新验证指标。
2. Loss v1：DiceCE + foreground focal/BCE + hard-case oversampling。
3. Loss v2：在 v1 上加骨架/中心线辅助目标。
4. Geometry v1：在 v2 上加表面邻域软约束和低可信后处理门控。
5. 每个实验都用同一批 hard case viewer 做视觉验收，重点看热图是否包裹牙齿、曲线是否连续、是否出现被强制拉进牙体内部的问题。
"""


def write_outputs(analysis: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_rows = []
    for row in analysis["labeled_rows"]:
        out = dict(row)
        out["labeled_issue"] = labeled_issue_text(row)
        labeled_rows.append(out)
    no_manual_rows = []
    for row in analysis["no_manual_rows"]:
        out = dict(row)
        out["issue_reasons"] = issue_reason_text(row)
        no_manual_rows.append(out)
    category_rows = build_category_summary(analysis["labeled_rows"], analysis["no_manual_rows"])

    write_csv(
        output_dir / "labeled_curve_error_rank.csv",
        labeled_rows,
        [
            "case_id",
            "tooth_id",
            "tooth_key",
            "category",
            "root_group",
            "n_manual_points",
            "labeled_issue",
            "flag_large_error",
            "flag_severe_error",
            "sym_mean",
            "sym_p95",
            "sym_max",
            "gt_to_pred_mean",
            "gt_to_pred_p95",
            "gt_to_pred_max",
            "pred_to_gt_mean",
            "pred_to_gt_p95",
            "pred_to_gt_max",
            "n_gt_curve",
            "n_pred_curve",
            "hmax",
            "p99",
            "vox03",
            "cc_count",
            "lcc_ratio",
            "wrap_coverage",
            "eval_fit_mean",
            "eval_fit_p95",
            "eval_fit_sr10",
        ],
    )
    write_csv(
        output_dir / "no_manual_lowconf_rank.csv",
        no_manual_rows,
        [
            "case_id",
            "tooth_id",
            "tooth_key",
            "category",
            "root_group",
            "issue_reasons",
            "flag_heatmap_issue",
            "flag_low_conf",
            "flag_tiny",
            "flag_fragmented",
            "flag_low_wrap",
            "hmax",
            "p99",
            "vox03",
            "cc_count",
            "lcc_ratio",
            "wrap_coverage",
        ],
    )
    write_csv(
        output_dir / "tooth_category_summary.csv",
        category_rows,
        [
            "category",
            "total",
            "labeled",
            "large_error",
            "severe_error",
            "no_manual",
            "no_manual_flagged",
            "low_conf",
            "fragmented",
            "low_wrap",
            "median_sym_mean",
            "median_sym_p95",
        ],
    )

    report_path = output_dir / "cej_failure_analysis_041055.md"
    guidance_path = output_dir / "model_improvement_guidance.md"
    report_path.write_text(render_failure_report(analysis, args), encoding="utf-8")
    guidance_path.write_text(render_model_guidance(analysis), encoding="utf-8")
    return {
        "report": report_path,
        "guidance": guidance_path,
        "labeled_csv": output_dir / "labeled_curve_error_rank.csv",
        "no_manual_csv": output_dir / "no_manual_lowconf_rank.csv",
        "category_csv": output_dir / "tooth_category_summary.csv",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--infer-dir", type=Path, default=DEFAULT_INFER_DIR)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--manual-report", type=Path, default=DEFAULT_MANUAL_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heatmap-threshold", type=float, default=0.30)
    parser.add_argument("--large-sym-p95-mm", type=float, default=1.50)
    parser.add_argument("--large-sym-mean-mm", type=float, default=0.80)
    parser.add_argument("--severe-sym-p95-mm", type=float, default=3.00)
    parser.add_argument("--min-pred-curve-points", type=int, default=16)
    parser.add_argument("--tiny-heatmap-voxels", type=int, default=100)
    parser.add_argument("--low-wrap-coverage", type=float, default=0.45)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    thresholds = Thresholds(
        heatmap_threshold=args.heatmap_threshold,
        large_sym_p95_mm=args.large_sym_p95_mm,
        large_sym_mean_mm=args.large_sym_mean_mm,
        severe_sym_p95_mm=args.severe_sym_p95_mm,
        min_pred_curve_points=args.min_pred_curve_points,
        tiny_heatmap_voxels=args.tiny_heatmap_voxels,
        low_wrap_coverage=args.low_wrap_coverage,
    )
    analysis = analyze_run(
        processed_dir=args.processed_dir,
        infer_dir=args.infer_dir,
        eval_csv=args.eval_csv,
        metrics_csv=args.metrics_csv,
        manual_report=args.manual_report,
        thresholds=thresholds,
    )
    paths = write_outputs(analysis, args)
    print(f"labeled teeth: {len(analysis['labeled_rows'])}")
    print(f"no manual teeth: {len(analysis['no_manual_rows'])}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
