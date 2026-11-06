"""Binary change-detection metrics following Zan et al. 2025 (Sec 5.2).

mF1 = (F1_nochange + F1_change) / 2 ; IoU1 = IoU of the change class;
plus OA / precision / recall. Probabilities are quantized to 256 bins so a
per-domain 2x256 histogram supports metrics at ANY threshold, including the
oracle best-F1 threshold, without keeping pixel-level data.
"""
import numpy as np

N_BINS = 256


class HistAccumulator:
    """hist[label, bin] counts; label in {0,1}, bin = round(prob*255)."""

    def __init__(self, domains):
        self.domains = list(domains)
        self.hist = {d: np.zeros((2, N_BINS), dtype=np.int64) for d in self.domains}
        self.total = {d: np.zeros(2, dtype=np.int64) for d in self.domains}

    def update(self, prob, label, domain):
        """prob: HxW float tensor in [0,1]; label: HxW long {0,1}."""
        import torch
        p = prob.detach().cpu().numpy()
        l = label.detach().cpu().numpy()
        bins = np.clip(np.rint(p * (N_BINS - 1)).astype(np.int64), 0, N_BINS - 1)
        h = np.zeros((2, N_BINS), dtype=np.int64)
        np.add.at(h, (l.ravel(), bins.ravel()), 1)
        self.hist[domain] += h
        self.total[domain] += np.bincount(l.ravel(), minlength=2)

    # ------------------------------------------------------------------
    def select_threshold(self, domain='overall', criterion='F1_change'):
        """Pick the bin index maximizing `criterion` on this accumulator's
        histogram. Used on the CALIBRATION split only — never on eval."""
        h = self.hist[domain] if domain in self.hist else sum(self.hist.values())
        best_t, best_v = 0, -1.0
        for t in range(N_BINS):
            m = self._metrics_from_hist(h, t)
            if m[criterion] > best_v:
                best_v, best_t = m[criterion], t
        return best_t

    def report_with_calib(self, calib, domain='overall'):
        """Eval-side report at three declared protocols:
        fixed (0.5) / calibrated (threshold from `calib` accumulator) / oracle
        (best threshold on eval — analysis only, NOT paper-comparable)."""
        t_calib = calib.select_threshold(domain) if domain in calib.hist \
            else calib.select_threshold('overall')
        r = self.domain_report(domain) if domain in self.hist else None
        if r is None:                                   # merged accumulator
            merged = HistAccumulator([domain])
            merged.hist[domain] = sum(self.hist.values())
            r = merged.domain_report(domain)
        out = dict(fixed=r['fixed'], best=r['best'])
        out['calibrated'] = self._metrics_from_hist(
            self.hist.get(domain) if domain in self.hist else sum(self.hist.values()),
            t_calib)
        out['calibrated']['threshold'] = t_calib / (N_BINS - 1)
        return out

    @staticmethod
    def _metrics_from_hist(h, t):
        """t: threshold bin index; predict change iff bin >= t.
        h[label, bin]: rows = ground truth {0:no-change, 1:change}."""
        tp0 = h[0, :t].sum()          # no-change correctly predicted
        fn0 = h[0, t:].sum()          # no-change predicted as change
        tp1 = h[1, t:].sum()          # change correctly predicted
        fn1 = h[1, :t].sum()          # change predicted as no-change
        fp0 = fn1                     # for class 0: actual change predicted as no-change
        fp1 = fn0                     # for class 1: actual no-change predicted as change
        n = h.sum()
        pr1 = tp1 / max(tp1 + fp1, 1)
        rc1 = tp1 / max(tp1 + fn1, 1)
        f1_1 = 2 * pr1 * rc1 / max(pr1 + rc1, 1e-9)
        pr0 = tp0 / max(tp0 + fp0, 1)
        rc0 = tp0 / max(tp0 + fn0, 1)
        f1_0 = 2 * pr0 * rc0 / max(pr0 + rc0, 1e-9)
        oa = (tp1 + tp0) / max(n, 1)
        iou1 = tp1 / max(tp1 + fn1 + fp1, 1)
        return dict(mF1=(f1_0 + f1_1) / 2, F1_change=f1_1, F1_nochange=f1_0,
                    IoU1=iou1, OA=oa, Pr_change=pr1, Rc_change=rc1,
                    Pr_nochange=pr0, Rc_nochange=rc0)

    def domain_report(self, domain):
        h = self.hist[domain]
        fixed = self._metrics_from_hist(h, N_BINS // 2)             # threshold 0.5
        # oracle best-F1_change threshold sweep
        best, best_t = fixed, N_BINS // 2
        for t in range(0, N_BINS, 1):
            m = self._metrics_from_hist(h, t)
            if m['F1_change'] > best['F1_change']:
                best, best_t = m, t
        best = dict(best)
        best['threshold'] = best_t / (N_BINS - 1)
        return dict(n_pairs_pixels=int(h.sum()), fixed=fixed, best=best)

    def report(self):
        out = {d: self.domain_report(d) for d in self.domains}
        tot = sum(self.hist.values())
        merged = HistAccumulator(['overall'])
        merged.hist['overall'] = tot
        out['overall'] = merged.domain_report('overall')
        return out


def format_table(report, calibrated=True):
    """report: {domain: {fixed, calibrated, best}} (calibrated may be None)."""
    rows = []
    if calibrated:
        hdr = (f"{'domain':<9}|{'mF1@.5':>8} {'IoU1@.5':>8}|{'mF1c':>7} {'IoU1c':>7} {'F1c':>6} {'thr':>5}"
               f"|{'mF1*':>7} {'IoU1*':>7} {'F1*':>6}")
        rows.append(hdr)
        for d, r in report.items():
            f, c, b = r['fixed'], r.get('calibrated') or {}, r['best']
            rows.append(f"{d:<9}|{f['mF1']:>8.4f} {f['IoU1']:>8.4f}"
                        f"|{c.get('mF1', float('nan')):>7.4f} {c.get('IoU1', float('nan')):>7.4f} "
                        f"{c.get('F1_change', float('nan')):>6.4f} {c.get('threshold', float('nan')):>5.3f}"
                        f"|{b['mF1']:>7.4f} {b['IoU1']:>7.4f} {b['F1_change']:>6.4f}")
    else:
        hdr = (f"{'domain':<9} {'mF1':>7} {'IoU1':>7} {'F1_ch':>7} {'OA':>7} "
               f"{'mF1*':>7} {'IoU1*':>7} {'F1_ch*':>7} {'thr*':>6}")
        rows.append(hdr)
        for d, r in report.items():
            f, b = r['fixed'], r['best']
            rows.append(f"{d:<9} {f['mF1']:>7.4f} {f['IoU1']:>7.4f} {f['F1_change']:>7.4f} {f['OA']:>7.4f} "
                        f"{b['mF1']:>7.4f} {b['IoU1']:>7.4f} {b['F1_change']:>7.4f} {b['threshold']:>6.3f}")
    return '\n'.join(rows)
