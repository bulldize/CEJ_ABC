# Geometry Prior Exp1/Exp2 Comparison

| Experiment | threshold | mean_dist_mm | p95_dist_mm | sr@1.0mm | failed/bad | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline Stage1-only | 0.25 | 0.628000 | 1.417000 | 0.846000 | 3 | baseline |
| exp1 preprocess prior | 0.35 | 2.868568 | 4.035110 | 0.563922 | 25 | FAIL |
| exp2 shape-prior loss | 0.25 | 3.504990 | 18.751021 | 0.617255 | 24 | FAIL |

Conclusion: `do_not_merge_geometry_prior_by_default`
