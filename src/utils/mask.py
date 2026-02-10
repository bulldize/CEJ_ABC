import torch


def _to_3tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) != 3:
            raise ValueError("patch_size must be int or length-3 tuple")
        return int(val[0]), int(val[1]), int(val[2])
    return int(val), int(val), int(val)


def create_patch_mask(shape, patch_size, mask_ratio, device=None, dtype=torch.float32):
    """Create a voxel-level mask with patch-wise random masking.

    Returns mask with 1 for masked voxels, 0 for visible voxels.
    """
    sx, sy, sz = [int(s) for s in shape]
    px, py, pz = _to_3tuple(patch_size)
    if sx <= 0 or sy <= 0 or sz <= 0:
        return torch.zeros((sx, sy, sz), device=device, dtype=dtype)

    gx = max(1, sx // px)
    gy = max(1, sy // py)
    gz = max(1, sz // pz)
    total = gx * gy * gz
    mask_count = int(total * float(mask_ratio))
    if mask_ratio > 0 and mask_count == 0:
        mask_count = 1

    grid = torch.zeros(total, device=device, dtype=dtype)
    if mask_count > 0:
        idx = torch.randperm(total, device=device)[:mask_count]
        grid[idx] = 1.0

    grid = grid.view(gx, gy, gz)
    mask = grid.repeat_interleave(px, dim=0).repeat_interleave(py, dim=1).repeat_interleave(pz, dim=2)
    # pad to match shape if needed
    if mask.shape[0] < sx or mask.shape[1] < sy or mask.shape[2] < sz:
        pad_x = max(0, sx - mask.shape[0])
        pad_y = max(0, sy - mask.shape[1])
        pad_z = max(0, sz - mask.shape[2])
        mask = torch.nn.functional.pad(mask, (0, pad_z, 0, pad_y, 0, pad_x))
    return mask[:sx, :sy, :sz]


def create_patch_mask_batch(batch_size, shape, patch_size, mask_ratio, device=None, dtype=torch.float32):
    masks = [create_patch_mask(shape, patch_size, mask_ratio, device=device, dtype=dtype) for _ in range(batch_size)]
    return torch.stack(masks, dim=0).unsqueeze(1)
