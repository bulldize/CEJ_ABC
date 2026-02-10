# Architecture

Modules:
- datasets: IO, ROI crop/stitch, points, heatmap, dataset (MONAI transforms/CacheDataset)
- models: MONAI 3D UNet wrapper
- train: training loop and checkpoints (MONAI DiceCE loss)
- pretrain_mae: masked reconstruction pretrain (unsupervised)
- preprocess_unsup: TF_008 unsupervised ROI preprocessing
- infer: prediction, postprocess, stitch-back (optional sliding window, divisible padding)
- eval: metrics (Point-to-Curve, SR@tau) with affine spacing fallback
- viz: ROI check, pseudo-GT, inference, error maps
- postprocess: threshold, intersection, skeletonize, largest component (MONAI LCC), priors
- utils.mask: patch-wise mask generator for masked reconstruction

Coordinate Convention:
- The world coordinate system defined by `A.nii.gz` sform/qform (affine) is the canonical reference.
- External annotations must align to this world space before converting to voxel coordinates.

MONAI Integration:
- Local MONAI repo is staged under `MONAI/` and injected via `src/__init__.py` for imports.
