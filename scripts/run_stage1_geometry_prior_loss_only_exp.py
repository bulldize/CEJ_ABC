#!/usr/bin/env python3
import argparse

from stage1_geometry_prior_common import add_common_args, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage1-only geometry-prior loss experiment without modifying GT labels. "
            "The runner copies the fixed 16/3 processed split, preserves H_GT/C_GT/curve_dense_points.npy, "
            "writes only H_SHAPE_PRIOR.nii.gz plus loss-only audit files, then trains with the weak prior loss."
        )
    )
    add_common_args(parser, default_prefix="stage1_geometry_prior_loss_only")
    parser.add_argument("--lambda-shape-prior", type=float, default=0.02)
    parser.add_argument("--shape-prior-sigma-mm", type=float, default=2.0)
    args = parser.parse_args()
    run_experiment(
        args,
        experiment_key="geometry_prior_loss_only",
        experiment_label="Stage1 geometry-prior weak loss only, original GT preserved",
        default_prefix="stage1_geometry_prior_loss_only",
        shape_prior=True,
        loss_only_geometry=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
