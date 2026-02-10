# Architecture

Modules:
- datasets: IO, ROI crop/stitch, points, heatmap, dataset
- models: 3D U-Net
- train: training loop and checkpoints
- pretrain_mae: masked reconstruction pretrain (unsupervised)
- preprocess_unsup: TF_008 unsupervised ROI preprocessing
- infer: prediction, postprocess, stitch-back
- eval: metrics (Point-to-Curve, SR@tau)
- viz: ROI check, pseudo-GT, inference, error maps
- postprocess: threshold, intersection, skeletonize, largest component, priors
- utils.mask: patch-wise mask generator for masked reconstruction

Coordinate Convention:
- The world coordinate system defined by `A.nii.gz` sform/qform (affine) is the canonical reference.
- External annotations must align to this world space before converting to voxel coordinates.
