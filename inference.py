"""Sliding-window native-resolution inference.

Protocol alignment with Zan et al. 2025: their supervised CD networks predict
at the full 512x512 tile. Instead of downsampling the whole tile to CLIP's
224 (which loses detail and upsamples a coarse 16x16 map back to 512), we
slide a native 224x224 window across the tile and average overlapping change
probabilities -> a full-resolution 512x512 prediction built only from
native-resolution CLIP forwards.
"""
import torch
import torch.nn.functional as F

from zscd.models.cd_model import change_score


def window_positions(size, win, stride):
    """Start offsets covering [0, size) with windows of `win`, stepping by
    `stride`; the last window is pinned to the right/bottom edge."""
    if size <= win:
        return [0]
    pos = list(range(0, size - win + 1, stride))
    if pos[-1] != size - win:
        pos.append(size - win)
    return pos


@torch.no_grad()
def slide_change_prob(model, im1, im2, win=224, stride=112,
                      baseline='semantic', batch_windows=8):
    """im1/im2: [3,H,W] normalized tensors (any H,W). Returns full-resolution
    change probability map [H,W] (or score map for the no-text baseline)."""
    dev = im1.device
    H, W = im1.shape[-2:]
    pos_h = window_positions(H, win, stride)
    pos_w = window_positions(W, win, stride)

    crops1, crops2, coords = [], [], []
    for i in pos_h:
        for j in pos_w:
            crops1.append(im1[:, i:i + win, j:j + win])
            crops2.append(im2[:, i:i + win, j:j + win])
            coords.append((i, j))

    acc = torch.zeros(H, W, device=dev)
    cnt = torch.zeros(H, W, device=dev)
    for s in range(0, len(coords), batch_windows):
        c1 = torch.stack(crops1[s:s + batch_windows])
        c2 = torch.stack(crops2[s:s + batch_windows])
        with torch.autocast('cuda', dtype=torch.float16):
            prob = change_score(model, c1, c2, baseline=baseline).float()  # [B,h,w]
        prob = F.interpolate(prob.unsqueeze(1), size=(win, win), mode='bilinear',
                             align_corners=False).squeeze(1)     # window-res
        for b, (i, j) in enumerate(coords[s:s + batch_windows]):
            acc[i:i + win, j:j + win] += prob[b]
            cnt[i:i + win, j:j + win] += 1
    return acc / cnt
