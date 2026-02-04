import argparse
import json
import os
import numpy as np
import nibabel as nib


def make_synth_case(out_dir, case_id="case_0001", shape=(64, 64, 64), spacing=(0.5, 0.5, 0.5)):
    os.makedirs(out_dir, exist_ok=True)
    X, Y, Z = shape
    A = np.random.normal(0, 0.1, size=shape).astype(np.float32)
    B = np.zeros(shape, dtype=np.int16)

    cx, cy = X // 2, Y // 2
    radius = min(X, Y) // 6
    z0, z1 = Z // 4, 3 * Z // 4

    for x in range(X):
        for y in range(Y):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                B[x, y, z0:z1] = 11
                A[x, y, z0:z1] += 1.0

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    nib.save(nib.Nifti1Image(A, affine), os.path.join(out_dir, "A.nii.gz"))
    nib.save(nib.Nifti1Image(B, affine), os.path.join(out_dir, "B.nii.gz"))

    # sparse points along a ring
    points = []
    for z in range(z0 + 2, z1 - 2, 3):
        for angle in [0, 90, 180, 270]:
            rad = np.deg2rad(angle)
            x = cx + int(radius * np.cos(rad))
            y = cy + int(radius * np.sin(rad))
            points.append([x, y, z])

    points_data = {
        "case_id": case_id,
        "coord_type": "voxel",
        "space": "full",
        "points": {"11": points},
    }
    with open(os.path.join(out_dir, "points.json"), "w") as f:
        json.dump(points_data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_id", default="case_0001")
    args = parser.parse_args()

    case_dir = os.path.join(args.output_dir, args.case_id)
    make_synth_case(case_dir, case_id=args.case_id)


if __name__ == "__main__":
    main()
