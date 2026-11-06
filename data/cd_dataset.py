"""Paired bi-temporal change-detection dataset for WHU-GCD test set II.

Layout: <root>/{im1,im2,label}/<domain>_<id>.png ; label is binary 0/255.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DOMAINS = ['levir', 'second', 'cnam', 'dsifn', 'clcd']


def clip_normalize(t):
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (t - mean) / std


def clip_denormalize(t):
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    return t * std + mean


class PairedChangeDataset(Dataset):
    """Returns (im1 [3,H,W] normalized, im2, label [H,W] {0,1} long, domain, name)."""

    def __init__(self, root, domains=None, limit_per_domain=0, seed=0):
        self.root = root
        names = sorted(os.listdir(os.path.join(root, 'im1')))
        names = [n for n in names if n.endswith('.png')]
        domains = domains or DOMAINS
        self.items = []
        for n in names:
            dom = n.split('_')[0]
            if dom in domains:
                self.items.append((dom, n))
        if limit_per_domain > 0:
            rng = np.random.RandomState(seed)
            by_dom = {}
            for dom, n in self.items:
                by_dom.setdefault(dom, []).append((dom, n))
            self.items = []
            for dom, lst in by_dom.items():
                idx = rng.permutation(len(lst))[:limit_per_domain]
                self.items.extend(lst[i] for i in sorted(idx))
            self.items.sort()
        self.domains = sorted({d for d, _ in self.items})

    def __len__(self):
        return len(self.items)

    def item_index(self, dom, name):
        """Index of item (dom, name) — built lazily on first call."""
        if not hasattr(self, '_index'):
            self._index = {it: i for i, it in enumerate(self.items)}
        return self._index[(dom, name)]

    def __getitem__(self, i):
        dom, name = self.items[i]
        im1 = torch.from_numpy(np.array(Image.open(os.path.join(self.root, 'im1', name)).convert('RGB'))).permute(2, 0, 1).float() / 255.
        im2 = torch.from_numpy(np.array(Image.open(os.path.join(self.root, 'im2', name)).convert('RGB'))).permute(2, 0, 1).float() / 255.
        lab = torch.from_numpy(np.array(Image.open(os.path.join(self.root, 'label', name)).convert('L'))).long()
        lab = (lab > 127).long()
        return clip_normalize(im1), clip_normalize(im2), lab, dom, name
