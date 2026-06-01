#!/usr/bin/env python3
import argparse

from stage1_geometry_prior_common import add_common_args, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage1-only experiment 1: geometry-prior GT preprocessing without a new loss."
    )
    add_common_args(parser, default_prefix="stage1_geometry_prior_preprocess")
    parser.add_argument("--shape-prior-sigma-mm", type=float, default=2.0)
    parser.set_defaults(lambda_shape_prior=0.0)
    args = parser.parse_args()
    run_experiment(
        args,
        experiment_key="geometry_prior_preprocess",
        experiment_label="Stage1 geometry-prior preprocessing",
        default_prefix="stage1_geometry_prior_preprocess",
        shape_prior=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
