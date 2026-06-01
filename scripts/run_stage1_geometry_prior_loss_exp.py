#!/usr/bin/env python3
import argparse
from pathlib import Path

from stage1_geometry_prior_common import (
    DEFAULT_RUN_PARENT,
    add_common_args,
    find_latest_exp1,
    run_experiment,
    write_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage1-only experiment 2: geometry-prior preprocessing plus weak shape-prior loss."
    )
    add_common_args(parser, default_prefix="stage1_geometry_prior_loss")
    parser.add_argument("--lambda-shape-prior", type=float, default=0.02)
    parser.add_argument("--shape-prior-sigma-mm", type=float, default=2.0)
    parser.add_argument("--exp1-root", default=None)
    args = parser.parse_args()
    report = run_experiment(
        args,
        experiment_key="geometry_prior_loss",
        experiment_label="Stage1 geometry-prior preprocessing plus weak shape-prior loss",
        default_prefix="stage1_geometry_prior_loss",
        shape_prior=True,
    )
    if args.setup_only or args.train_only:
        return 0

    exp2_root = Path(report["run_root"]).resolve()
    exp1_root = Path(args.exp1_root).resolve() if args.exp1_root else find_latest_exp1(DEFAULT_RUN_PARENT)
    if exp1_root:
        comparison = write_comparison(exp1_root, exp2_root, exp2_root / "summary")
        if comparison:
            print(f"[COMPARISON] {comparison['comparison_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
