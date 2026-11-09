"""CPU smoke test for the TTA adapters (zscd.tta.tent_cd).

Run:  python -m zscd.scripts.smoke_tent    (minutes on CPU; ViT-L @ 224)

Checks — all must pass before anything is trusted on the GPU box:
  1. collect_params_ln opens grads on LayerNorm affines only, plausible count
  2. TentCD.forward actually moves those params (continual: drift accumulates)
  3. episodic mode: params restored bitwise after every batch; two calls on
     the same input give identical outputs
  4. losses finite and sane with div_weight=0 (plain TENT) vs 1
  5. QuantileCalibrator._offset math: identity during warmup, anchored after
"""
import torch
import torch.nn.functional as F

from zscd.data.cd_dataset import PairedChangeDataset
from zscd.models.dino_tower import DenseChangeDINO
from zscd.paths import DATA_ROOT
from zscd.tta import QuantileCalibrator, TentCD, collect_params_ln


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    print(f'device: {device}')

    ds = PairedChangeDataset(DATA_ROOT, ['levir'], 2)
    im1, im2, lab, _, _ = ds[0]
    im1 = F.interpolate(im1.unsqueeze(0), size=(224, 224), mode='bilinear').to(device)
    im2 = F.interpolate(im2.unsqueeze(0), size=(224, 224), mode='bilinear').to(device)
    print(f'sample: levir 224x224, label change ratio {lab.float().mean():.4f}')

    base = DenseChangeDINO('dinov2').to(device).eval()

    # -- 1. param collection ------------------------------------------------
    params, names = collect_params_ln(base)
    n_adapt = sum(p.numel() for p in params)
    print(f'[1] LN tensors: {len(params)}, params: {n_adapt}, '
          f'e.g. {names[:2]} ... {names[-2:]}')
    assert 0 < len(params) < 800, 'implausible LN count'
    with_grad = [p for p in base.parameters() if p.requires_grad]
    assert not with_grad, 'frozen tower must have no grads before wrapping'

    # -- 2+4. continual mode: params move, losses finite ---------------------
    tent_c = TentCD(base, baseline='absdiff_norm', lr=1e-4, steps=1, alpha=8.0,
                    div_weight=1.0, mode='continual', debug_every=1)
    grad_ids = {id(p) for p in base.parameters() if p.requires_grad}
    ln_ids = {id(p) for p in params}
    assert grad_ids == ln_ids, 'grad set must equal the LN affine set exactly'
    src = [p.detach().clone() for p in tent_c.params]
    c1 = tent_c(im1, im2)
    d1 = tent_c._delta_w()
    assert torch.isfinite(c1).all(), 'NaN/Inf in adapted score map'
    assert d1 > 0, 'params did not move'
    _ = tent_c(im1, im2)
    d2 = tent_c._delta_w()
    print(f'[2] continual drift: {d1:.3e} -> {d2:.3e} (accumulates)')
    assert d2 > d1, 'continual drift should accumulate'

    # plain-TENT loss values (div=0) on the same score map, both finite
    tent_c.mode = 'episodic'                       # stop drifting for the check
    with torch.no_grad():
        c_probe = tent_c._score(im1, im2)
    for dw in (0.0, 1.0):
        tent_c.div_weight = dw
        loss, ent, div = tent_c.tent_loss(c_probe)
        print(f'[4] div_weight={dw}: loss={loss:.4f} ent={ent:.4f} div={div:.4f}')
        assert torch.isfinite(loss), 'loss not finite'

    # -- 3. episodic mode: bitwise restore + determinism ---------------------
    tent_e = TentCD(base, baseline='absdiff_norm', lr=1e-4, steps=1, alpha=8.0,
                    div_weight=1.0, mode='episodic')
    snap = [p.detach().clone() for p in tent_e.params]
    o1 = tent_e(im1, im2)
    assert all(torch.equal(p, s) for p, s in zip(tent_e.params, snap)), \
        'episodic did not restore params bitwise'
    o2 = tent_e(im1, im2)
    # CUDA LayerNorm-backward kernels are atomics-based and non-bitwise
    # deterministic; identical logic must still agree to float noise.
    odiff = (o1 - o2).abs().max().item()
    assert torch.allclose(o1, o2, rtol=1e-4, atol=1e-5), \
        f'episodic outputs differ beyond float noise (max {odiff:.2e})'
    drift_after = tent_e._delta_w()
    assert drift_after == 0.0, 'episodic left residual drift'
    print(f'[3] episodic: bitwise restore OK, outputs repeat to {odiff:.2e} '
          f'(score range [{o1.min():.4f}, {o1.max():.4f}])')
    # adapted prediction differs from the frozen baseline (params did move)
    with torch.no_grad():
        c_base = tent_e._score(im1, im2)
    diff = (o1 - c_base).abs().max().item()
    print(f'[3] adapted-vs-baseline max |delta score| within batch: {diff:.2e} '
          '(>0 while adapted; snapshot baseline computed post-restore)')

    # -- 5. quantile offset math ---------------------------------------------
    qc = QuantileCalibrator(base, baseline='absdiff_norm', q=0.5, warmup=8)
    assert qc._offset() == 0.0, 'warmup must be identity'
    qc.n_maps = 8
    qc.hist.zero_()
    qc.hist[77] = 1e6                              # all mass in bin 77
    expect = (77 + 0.5) / 256 - 0.5
    assert abs(qc._offset() - expect) < 1e-9, 'median anchor math wrong'
    print(f'[5] quantile offset: warmup identity OK, anchored offset '
          f'{expect:+.4f} OK')

    print('\nALL SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
