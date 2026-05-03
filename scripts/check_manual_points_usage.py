#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.manual_points_source import detect_points_source, find_tooth_col, find_xyz_cols, infer_prefix_from_source, load_manual_points_dataframe


def choose_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def normalize_sid(raw: str) -> str:
    sid = raw.strip()
    if sid.isdigit() and len(sid) < 3:
        return sid.zfill(3)
    return sid


def parse_dir_case_hint(dir_name: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.fullmatch(r"ToothFairy3([FP])_(\d+)", dir_name)
    if m:
        return m.group(1), normalize_sid(m.group(2))

    m = re.search(r"(\d+)$", dir_name)
    if m:
        return None, normalize_sid(m.group(1))

    return None, None


def build_case_ids(
    sid: str, prefix_hint: Optional[str], inferred_prefix: Optional[str], case_prefix: str
) -> List[str]:
    if prefix_hint in {"F", "P"}:
        return [f"ToothFairy3{prefix_hint}_{sid}"]

    if case_prefix == "F":
        order = ["F"]
    elif case_prefix == "P":
        order = ["P"]
    elif case_prefix == "both":
        order = ["F", "P"]
    else:  # auto
        if inferred_prefix in {"F", "P"}:
            order = [inferred_prefix, "P" if inferred_prefix == "F" else "F"]
        else:
            order = ["F", "P"]
    return [f"ToothFairy3{p}_{sid}" for p in order]


def case_files_exist(tf_root: Path, case_id: str) -> bool:
    image = choose_existing_path(
        [
            tf_root / "imagesTr" / f"{case_id}_0000.nii.gz",
            tf_root / "imagesTr" / f"{case_id}_0000.nii",
        ]
    )
    label = choose_existing_path(
        [
            tf_root / "labelsTr" / f"{case_id}.nii.gz",
            tf_root / "labelsTr" / f"{case_id}.nii",
        ]
    )
    return image is not None and label is not None


def count_processed_points(case_processed_dir: Path) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_tooth: Dict[str, int] = {}
    if not case_processed_dir.is_dir():
        return total, by_tooth

    tooth_dirs = sorted([p for p in case_processed_dir.iterdir() if p.is_dir() and p.name.startswith("tooth_")])
    for td in tooth_dirs:
        tooth_id = td.name.split("_", 1)[1] if "_" in td.name else td.name
        points_path = td / "points.json"
        n = 0
        if points_path.exists():
            try:
                data = json.loads(points_path.read_text())
                pts = data.get("points", {})
                if isinstance(pts, dict):
                    for v in pts.values():
                        if isinstance(v, list):
                            n += len(v)
                elif isinstance(pts, list):
                    n += len(pts)
            except Exception:
                n = 0
        by_tooth[tooth_id] = n
        total += n
    return total, by_tooth


def collect_filter_stats(case_processed_dir: Path) -> Dict[str, int]:
    stats = {
        "filter_input_points": 0,
        "filter_in_bounds": 0,
        "filter_removed": 0,
        "filter_invalid": 0,
        "filter_report_count": 0,
    }
    if not case_processed_dir.is_dir():
        return stats

    for td in case_processed_dir.iterdir():
        if not (td.is_dir() and td.name.startswith("tooth_")):
            continue
        dist_path = td / "point_surface_dist.json"
        if not dist_path.exists():
            continue
        try:
            d = json.loads(dist_path.read_text())
        except Exception:
            continue
        stats["filter_report_count"] += 1
        stats["filter_input_points"] += int(d.get("n_points", 0) or 0)
        stats["filter_in_bounds"] += int(d.get("n_in_bounds", 0) or 0)
        stats["filter_removed"] += int(d.get("n_removed", 0) or 0)
        stats["filter_invalid"] += int(d.get("n_invalid", 0) or 0)
    return stats


def read_manual_counts(points_file: Path) -> Tuple[int, int, int, Dict[str, int]]:
    df = load_manual_points_dataframe(points_file)
    total_rows = int(len(df))
    tooth_col = find_tooth_col(df)
    x_col, y_col, z_col = find_xyz_cols(df)
    if tooth_col is None:
        raise ValueError(f"cannot find tooth/group column in {points_file}")

    tooth_ids = df[tooth_col].astype(str).str.extract(r"(\d+)")[0]
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    z = pd.to_numeric(df[z_col], errors="coerce")
    valid_mask = tooth_ids.notna() & x.notna() & y.notna() & z.notna()

    valid_df = df.loc[valid_mask].copy()
    valid_df["_tooth_id"] = tooth_ids.loc[valid_mask].astype(int).astype(str)
    valid_rows = int(valid_mask.sum())
    invalid_rows = int(total_rows - valid_rows)

    by_tooth = {str(k): int(v) for k, v in valid_df["_tooth_id"].value_counts().sort_index().items()}
    return total_rows, valid_rows, invalid_rows, by_tooth


def run_check(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    manual_root = Path(args.manual_root)
    tf_root = Path(args.toothfairy_root)
    processed_root = Path(args.processed_root)

    results: List[Dict[str, object]] = []
    manual_dirs = sorted([p for p in manual_root.glob("*") if p.is_dir()])
    if args.max_cases is not None:
        manual_dirs = manual_dirs[: int(args.max_cases)]

    for d in manual_dirs:
        points_file = detect_points_source(d)
        if points_file is None:
            continue

        prefix_hint, sid = parse_dir_case_hint(d.name)
        inferred_prefix = infer_prefix_from_source(points_file)
        if sid is None:
            results.append(
                {
                    "manual_dir": d.name,
                    "case_id": "",
                    "status": "FAIL_PARSE_CASE_ID",
                    "reason": "cannot parse sid from directory name",
                }
            )
            continue

        candidates = build_case_ids(sid, prefix_hint, inferred_prefix, args.case_prefix)
        chosen_case_id = None
        for cid in candidates:
            if (processed_root / cid).is_dir():
                chosen_case_id = cid
                break
        if chosen_case_id is None:
            for cid in candidates:
                if case_files_exist(tf_root, cid):
                    chosen_case_id = cid
                    break

        if chosen_case_id is None:
            results.append(
                {
                    "manual_dir": d.name,
                    "case_id": "",
                    "status": "FAIL_CASE_MAPPING",
                    "reason": f"no candidate found: {','.join(candidates)}",
                }
            )
            continue

        excel_total, excel_valid, excel_invalid, excel_by_tooth = read_manual_counts(points_file)
        case_processed_dir = processed_root / chosen_case_id
        processed_kept, processed_by_tooth = count_processed_points(case_processed_dir)
        filter_stats = collect_filter_stats(case_processed_dir)
        filter_expected_kept = (
            filter_stats["filter_input_points"] - filter_stats["filter_removed"] - filter_stats["filter_invalid"]
        )

        missing_teeth = sorted(
            [
                t
                for t, n in excel_by_tooth.items()
                if int(n) > 0 and int(processed_by_tooth.get(str(t), 0)) == 0
            ],
            key=lambda x: int(x),
        )
        partial_teeth = sorted(
            [
                t
                for t, n in excel_by_tooth.items()
                if int(n) > int(processed_by_tooth.get(str(t), 0)) > 0
            ],
            key=lambda x: int(x),
        )
        dropped_points_before_tooth = max(0, excel_valid - int(filter_stats["filter_input_points"]))

        rows_match_processed = excel_valid == processed_kept
        rows_match_filter_input = (
            filter_stats["filter_report_count"] > 0 and excel_valid == filter_stats["filter_input_points"]
        )
        filter_kept_match_processed = (
            filter_stats["filter_report_count"] > 0 and filter_expected_kept == processed_kept
        )

        if rows_match_processed:
            status = "PASS_FULLY_USED"
            reason = ""
        elif rows_match_filter_input and filter_kept_match_processed:
            status = "PASS_WITH_FILTER_DROP"
            reason = "all excel points reached preprocessing, then part dropped by distance/out-of-bounds filter"
        elif filter_stats["filter_report_count"] == 0 and processed_kept == 0 and excel_valid > 0:
            status = "FAIL_NO_POINTS_AFTER_PREPROCESS"
            reason = "processed case has zero usable points; rerun preprocess for this case"
        elif dropped_points_before_tooth > 0:
            status = "FAIL_POINTS_NOT_REACHING_TOOTH_STAGE"
            reason = (
                f"{dropped_points_before_tooth} points are absent before tooth-level filtering "
                f"(likely missing tooth masks/labels for some tooth ids)"
            )
        else:
            status = "FAIL_COUNT_MISMATCH"
            reason = "excel valid points and processed points are inconsistent"

        results.append(
            {
                "manual_dir": d.name,
                "case_id": chosen_case_id,
                "points_file": str(points_file),
                "status": status,
                "excel_total_rows": excel_total,
                "excel_valid_rows": excel_valid,
                "excel_invalid_rows": excel_invalid,
                "processed_kept_points": processed_kept,
                "delta_excel_valid_minus_processed": excel_valid - processed_kept,
                "filter_input_points": filter_stats["filter_input_points"],
                "filter_removed_points": filter_stats["filter_removed"],
                "filter_invalid_points": filter_stats["filter_invalid"],
                "filter_expected_kept": filter_expected_kept,
                "filter_report_count": filter_stats["filter_report_count"],
                "check_rows_eq_processed": rows_match_processed,
                "check_rows_eq_filter_input": rows_match_filter_input,
                "check_filter_kept_eq_processed": filter_kept_match_processed,
                "missing_teeth": "|".join(missing_teeth),
                "partial_teeth": "|".join(partial_teeth),
                "excel_points_by_tooth": json.dumps(excel_by_tooth, ensure_ascii=False, sort_keys=True),
                "processed_points_by_tooth": json.dumps(processed_by_tooth, ensure_ascii=False, sort_keys=True),
                "reason": reason,
            }
        )

    n_total = len(results)
    n_pass = sum(1 for r in results if str(r.get("status", "")).startswith("PASS"))
    n_fail = n_total - n_pass
    summary = {
        "manual_root": str(manual_root),
        "processed_root": str(processed_root),
        "total_cases": n_total,
        "pass_cases": n_pass,
        "fail_cases": n_fail,
        "all_used": bool(n_fail == 0),
    }
    return results, summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "manual_dir",
        "case_id",
        "points_file",
        "status",
        "excel_total_rows",
        "excel_valid_rows",
        "excel_invalid_rows",
        "processed_kept_points",
        "delta_excel_valid_minus_processed",
        "filter_input_points",
        "filter_removed_points",
        "filter_invalid_points",
        "filter_expected_kept",
        "filter_report_count",
        "check_rows_eq_processed",
        "check_rows_eq_filter_input",
        "check_filter_kept_eq_processed",
        "missing_teeth",
        "partial_teeth",
        "excel_points_by_tooth",
        "processed_points_by_tooth",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)


def write_json(path: Path, summary: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "cases": rows}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def print_console_report(rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    print(
        "manual_dir,case_id,status,excel_valid_rows,processed_kept_points,"
        "filter_input_points,filter_removed_points,filter_invalid_points,missing_teeth,reason"
    )
    for r in rows:
        print(
            f"{r.get('manual_dir','')},{r.get('case_id','')},{r.get('status','')},"
            f"{r.get('excel_valid_rows','')},{r.get('processed_kept_points','')},"
            f"{r.get('filter_input_points','')},{r.get('filter_removed_points','')},"
            f"{r.get('filter_invalid_points','')},{r.get('missing_teeth','')},{r.get('reason','')}"
        )
    print(
        f"SUMMARY total={summary['total_cases']} pass={summary['pass_cases']} "
        f"fail={summary['fail_cases']} all_used={summary['all_used']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether manual Excel CEJ points are fully used in processed tooth outputs."
    )
    parser.add_argument("--manual-root", default="/root/手工标注1")
    parser.add_argument("--toothfairy-root", default="/root/ToothFairy3")
    parser.add_argument("--processed-root", default="/root/cej_runs/run_unsup_001/processed")
    parser.add_argument("--case-prefix", default="auto", choices=["auto", "F", "P", "both"])
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--output-csv",
        default="/root/cej_runs/run_unsup_001/manual_points_usage_report.csv",
    )
    parser.add_argument(
        "--output-json",
        default="/root/cej_runs/run_unsup_001/manual_points_usage_report.json",
    )
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    rows, summary = run_check(args)
    print_console_report(rows, summary)

    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
        print(f"[saved] csv: {args.output_csv}")
    if args.output_json:
        write_json(Path(args.output_json), summary, rows)
        print(f"[saved] json: {args.output_json}")

    if args.fail_on_mismatch and summary["fail_cases"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
