# Architecture

Modules:
- datasets: IO, ROI crop/stitch, points, heatmap, dataset
- models: 3D U-Net
- train: training loop and checkpoints
- infer: prediction, postprocess, stitch-back
- eval: metrics (Point-to-Curve, SR@tau)
- viz: ROI check, pseudo-GT, inference, error maps
- postprocess: threshold, intersection, skeletonize, largest component, priors

Coordinate Convention:
- The world coordinate system defined by `A.nii.gz` sform/qform (affine) is the canonical reference.
- External annotations must align to this world space before converting to voxel coordinates.
