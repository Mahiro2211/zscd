"""Coordinated bi-temporal augmented views for dense TPT.

Same crop/flip parameters are applied to both dates so the change signal
stays spatially aligned across the pair (dense analog of TPT's
RandomResizedCrop views, data/datautils.py:62-66).
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop


@torch.no_grad()
def bi_temporal_views(im1, im2, n_views, crop_size=224, scale=(0.3, 1.0),
                      ratio=(3. / 4., 4. / 3.), clean=True):
    """im1/im2: [3,H,W] normalized tensors. Returns (views1 [N,3,s,s], views2)."""
    views1, views2 = [], []
    rrc = RandomResizedCrop(crop_size, scale=scale, ratio=ratio)
    for _ in range(n_views):
        i, j, h, w = rrc.get_params(im1, scale, ratio)
        v1 = im1[:, i:i + h, j:j + w].unsqueeze(0)
        v2 = im2[:, i:i + h, j:j + w].unsqueeze(0)
        v1 = F.interpolate(v1, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        v2 = F.interpolate(v2, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        if torch.rand(1).item() < 0.5:                       # same flip for both dates
            v1, v2 = torch.flip(v1, dims=[3]), torch.flip(v2, dims=[3])
        views1.append(v1)
        views2.append(v2)
    if clean:                                                # clean view = full frame resized
        c1 = F.interpolate(im1.unsqueeze(0), size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        c2 = F.interpolate(im2.unsqueeze(0), size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        views1.insert(0, c1)
        views2.insert(0, c2)
    return torch.cat(views1, 0), torch.cat(views2, 0)
