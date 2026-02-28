# Architecture

Modules:
- datasets: IO, ROI crop/stitch, points, heatmap, dataset (MONAI transforms/CacheDataset)
- datasets.raw_cases: unified raw-case discovery for `case_dirs` and `toothfairy3` layouts (`raw_layout=auto` supported)
- models: MONAI 3D UNet wrapper
- train: training loop and checkpoints (MONAI DiceCE loss)
- pretrain_mae: masked reconstruction pretrain (unsupervised)
- preprocess_unsup: TF_008 unsupervised ROI preprocessing
- infer: prediction, postprocess, stitch-back (optional sliding window, divisible padding)
- eval: metrics (Point-to-Curve, SR@tau) with affine spacing fallback
- viz: interactive 3D HTML viewer (Slicer-like MPR slices + surfaces) and optional 2D overlays
- export_pseudo_gt_full: full-mouth pseudo-GT export for medical QA (NIfTI masks/labelmaps for 3D Slicer)
- postprocess: threshold, intersection, skeletonize, largest component (MONAI LCC), priors
- utils.mask: patch-wise mask generator for masked reconstruction

Coordinate Convention:
- The world coordinate system defined by `A.nii.gz` sform/qform (affine) is the canonical reference.
- External annotations must align to this world space before converting to voxel coordinates.
- NIfTI save path keeps qform/sform consistent with affine so external tools read orientation reliably.

MONAI Integration:
- Use installed `monai` package from environment (`requirements.txt`), no in-repo MONAI source checkout.
