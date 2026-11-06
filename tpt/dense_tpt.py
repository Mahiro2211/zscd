"""Dense Test-Time Prompt Tuning (TPT) for change detection.

Per test pair (adapted from TPT/tpt_classification.py):
  1. reset the learnable prompt context to its hand-crafted initialization
  2. build N coordinated bi-temporal crop views (1 clean + N-1 augmented)
  3. forward views (visual tower under no_grad, features cached — only the text
     side changes between optimization steps)
  4. select the top `selection_p` fraction of views by mean per-pixel entropy
     of the change map (TPT's select_confident_samples, dense version)
  5. minimize the marginal entropy of the view-averaged change probability map
     (dense version of TPT's avg_entropy) w.r.t. the prompt context, AdamW
  6. final prediction on the clean full-resolution pair ('context' mode)
"""
import torch
import torch.nn.functional as F

from ..data.viewgen import bi_temporal_views


def _pixel_entropy(p, eps=1e-6):
    return -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))


@torch.no_grad()
def _visual_feats(model, views, chunk=8):
    feats = []
    for i in range(0, views.shape[0], chunk):
        with torch.autocast('cuda', dtype=torch.float16):
            _, d = model.visual(views[i:i + chunk])
        feats.append(d.float())
    return torch.cat(feats, 0)                                     # [N, D, h, w]


def _change_from_feats(model, feats1, feats2, contexts):
    tf = F.normalize(model.text.forward_with_context(model.texts, contexts).float(), dim=-1)
    scale = 1.0 / model.tau
    p1 = model.class_logits(feats1, tf, scale).softmax(dim=1)
    p2 = model.class_logits(feats2, tf, scale).softmax(dim=1)
    return 1.0 - (p1 * p2).sum(dim=1)                             # [N, h, w], keeps grad to ctx


def tpt_predict_pair(model, im1, im2, n_views=16, tta_steps=1, lr=5e-3,
                     selection_p=0.1, crop_size=224, crop_scale=(0.3, 1.0)):
    """im1/im2: [3,H,W] normalized, on model device. Runs the TPT loop and
    returns (losses, tuned contexts snapshot). Final prediction is left to the
    caller (sliding-window, native resolution) with mode='context'."""
    model.reset()
    ctx = model.contexts
    ctx.requires_grad_(True)
    opt = torch.optim.AdamW([ctx], lr=lr)

    views1, views2 = bi_temporal_views(im1, im2, n_views - 1,
                                       crop_size=crop_size, scale=crop_scale, clean=True)
    feats1 = _visual_feats(model, views1)
    feats2 = _visual_feats(model, views2)
    n_sel = max(1, int(views1.shape[0] * selection_p))

    losses = []
    for _ in range(tta_steps):
        change = _change_from_feats(model, feats1, feats2, ctx)
        with torch.no_grad():
            score = _pixel_entropy(change.detach()).mean(dim=(1, 2))
            idx = torch.argsort(score)[:n_sel]
        p_bar = change[idx].mean(dim=0)                            # marginal over confident views
        loss = _pixel_entropy(p_bar).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    ctx.requires_grad_(False)
    return losses, ctx.detach().clone()
