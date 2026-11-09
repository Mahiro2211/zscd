#!/usr/bin/env python
"""Aggregate results/per_sample/*.json into results/sample_analysis.md.

Reads the per-tile records produced by scripts/analyze_samples.py (one JSON
per weights x baseline) and writes a Chinese markdown report:

  1. per-config sections: good/mid/bad bucket proportions per domain,
     no-change-tile false-alarm proportions, per-domain means (IoU1/AUC/
     GT-vs-pred positive rate), change-size vs performance, and the
     best/worst/worst-FP sample rankings with embedded canvases
  2. semantic vs notext cross-comparison (bucket migration + paired IoU1
     deltas on the shared sample set)
  3. auto-generated factual observations (computed numbers only)

Usage:
  python scripts/make_sample_report.py            # all configs found
  python scripts/make_sample_report.py --embed 0  # tables only, no images
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(REPO_DIR))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from zscd.paths import RESULTS_DIR

WEIGHTS_ORDER = ['openai', 'remoteclip', 'georsclip']
WEIGHTS_CN = {'openai': 'OpenAI CLIP', 'remoteclip': 'RemoteCLIP', 'georsclip': 'GeoRSCLIP'}
BASE_CN = {'semantic': '语义一致性', 'notext': '无文本 feat-cos'}
BUCKET_CN = {'good': '好', 'mid': '中', 'bad': '差',
             'nc_clean': '干净', 'nc_mild': '轻微误报', 'nc_severe': '严重误报'}
SIZE_BINS = [('<1%', 0.0, 0.01), ('1–5%', 0.01, 0.05), ('>5%', 0.05, 1.1)]


def pct(c, n):
    return f'{c} ({100 * c / n:.1f}%)' if n else '—'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float('nan')


def median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float('nan')


def fmt(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def load_configs():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, 'per_sample', '*.json'))):
        with open(p) as f:
            d = json.load(f)
        d.setdefault('sample_frac', 1.0)
        out.append(d)
    out.sort(key=lambda d: (WEIGHTS_ORDER.index(d['weights']), d['baseline'] != 'semantic'))
    return out


def cfg_title(d):
    frac = '全量' if d['sample_frac'] >= 1.0 else f'抽样{d["sample_frac"]:.0%}'
    n = d.get('n_pairs', d.get('n_eval'))
    return (f'{WEIGHTS_CN[d["weights"]]} · {BASE_CN[d["baseline"]]}'
            f'(thr={d["threshold"]:.3f}, {frac}, n={n})')


def config_section(d, embed):
    """Markdown block for one weights x baseline config."""
    rs = d['records']
    change = [r for r in rs if r['gt_pos'] > 0]
    nochange = [r for r in rs if r['gt_pos'] == 0]
    bd = d['bucket_def']
    lines = [f'### {cfg_title(d)}', '']

    # --- bucket proportions, change tiles --------------------------------
    lines += [f'**有变化样本分桶**(IoU1@thr:好 ≥{bd["iou_good"]} / 中 {bd["iou_mid"]}–{bd["iou_good"]}'
              f' / 差 <{bd["iou_mid"]};n={len(change)}):', '',
              '| domain | 好 | 中 | 差 | 合计 |', '|---|---|---|---|---|']
    for dom in sorted({r['domain'] for r in change}) + ['overall']:
        rs_d = change if dom == 'overall' else [r for r in change if r['domain'] == dom]
        if not rs_d:
            continue
        c = [sum(r['bucket'] == b for r in rs_d) for b in ('good', 'mid', 'bad')]
        lines.append(f'| {dom} | {pct(c[0], len(rs_d))} | {pct(c[1], len(rs_d))} | '
                     f'{pct(c[2], len(rs_d))} | {len(rs_d)} |')

    # --- no-change tiles --------------------------------------------------
    lines += ['', f'**无变化样本**(GT 全背景;按预测正例率分桶;n={len(nochange)}):', '',
              '| domain | 干净 | 轻微误报 | 严重误报 | 合计 |', '|---|---|---|---|---|']
    for dom in sorted({r['domain'] for r in nochange}) + ['overall']:
        rs_d = nochange if dom == 'overall' else [r for r in nochange if r['domain'] == dom]
        if not rs_d:
            continue
        c = [sum(r['bucket'] == b for r in rs_d) for b in ('nc_clean', 'nc_mild', 'nc_severe')]
        lines.append(f'| {dom} | {pct(c[0], len(rs_d))} | {pct(c[1], len(rs_d))} | '
                     f'{pct(c[2], len(rs_d))} | {len(rs_d)} |')

    # --- per-domain means --------------------------------------------------
    lines += ['', '**分域诊断**(均值;「正例率比」= 预测正例率 / GT 正例率,>1.2 过报、<0.8 漏报):', '',
              '| domain | IoU1 均值 | IoU1 中位 | AUC 均值 | GT 正例率 | 预测正例率 | 正例率比 |',
              '|---|---|---|---|---|---|---|']
    for dom in sorted({r['domain'] for r in change}) + ['overall']:
        rs_d = change if dom == 'overall' else [r for r in change if r['domain'] == dom]
        if not rs_d:
            continue
        gt, pr = mean(r['gt_pos'] for r in rs_d), mean(r['pred_pos'] for r in rs_d)
        lines.append(f'| {dom} | {fmt(mean(r["iou1"] for r in rs_d))} | {fmt(median(r["iou1"] for r in rs_d))} '
                     f'| {fmt(mean(r["auc"] for r in rs_d))} | {fmt(gt)} | {fmt(pr)} '
                     f'| {fmt(pr / gt if gt else float("nan"), 2)} |')

    # --- change size vs performance ---------------------------------------
    lines += ['', '**变化区域大小 vs 性能**(按 GT 正例率分组,仅限有变化样本):', '',
              '| GT 正例率 | n | IoU1 均值 | AUC 均值 | 漏检(召回<0.1)占比 |', '|---|---|---|---|---|']
    for label, lo, hi in SIZE_BINS:
        grp = [r for r in change if lo <= r['gt_pos'] < hi]
        if not grp:
            lines.append(f'| {label} | 0 | — | — | — |')
            continue
        miss = sum(r['rc'] < 0.1 for r in grp) / len(grp)
        lines.append(f'| {label} | {len(grp)} | {fmt(mean(r["iou1"] for r in grp))} '
                     f'| {fmt(mean(r["auc"] for r in grp))} | {miss:.1%} |')

    # --- rankings ----------------------------------------------------------
    for key, title in (('best', f'最佳样本(IoU1 top-{len(d["rankings"]["best"])})'),
                       ('worst', f'最差样本(IoU1 bottom-{len(d["rankings"]["worst"])})'),
                       ('worst_fp', f'无变化·误报最重 top-{len(d["rankings"]["worst_fp"])}')):
        lst = d['rankings'][key]
        if not lst:
            continue
        lines += ['', f'**{title}**:', '',
                  '| # | 样本 | 域 | IoU1 | F1c | OA | 召回 | GT 正例率 | 预测正例率 |',
                  '|---|---|---|---|---|---|---|---|---|']
        for i, r in enumerate(lst, 1):
            lines.append(f'| {i} | {r["name"][:-4]} | {r["domain"]} | {r["iou1"]:.3f} | {r["f1c"]:.3f} '
                         f'| {r["oa"]:.3f} | {r["rc"]:.3f} | {r["gt_pos"]:.3%} | {r["pred_pos"]:.3%} |')
        for r in lst[:embed]:
            lines += [f'![{r["name"][:-4]}]({r["img"]})', f'*{key} · {r["name"][:-4]} '
                     f'({r["domain"]}, IoU1={r["iou1"]:.3f}, GT 正例率={r["gt_pos"]:.2%})*', '']
    return lines


def cross_section(pairs):
    """semantic vs notext comparison for each weights having both baselines."""
    lines = ['## 2. semantic vs notext 交叉对比(同样本集合)', '']
    for sem, note in pairs:
        w = WEIGHTS_CN[sem['weights']]
        sem_r = {r['name']: r for r in sem['records']}
        note_r = {r['name']: r for r in note['records']}
        common = sorted(set(sem_r) & set(note_r))
        change = [n for n in common if sem_r[n]['gt_pos'] > 0]
        if not change:
            continue
        deltas = [note_r[n]['iou1'] - sem_r[n]['iou1'] for n in change]
        better = sum(d > 0 for d in deltas) / len(deltas)
        flips = sum(sem_r[n]['bucket'] == 'bad' and note_r[n]['bucket'] == 'good' for n in change)
        back = sum(sem_r[n]['bucket'] == 'good' and note_r[n]['bucket'] == 'bad' for n in change)
        lines += [f'### {w}(共有变化样本 n={len(change)})', '',
                  f'- 逐样本 IoU1 差(notext − semantic):均值 **{mean(deltas):+.3f}**,中位 {median(deltas):+.3f};'
                  f'notext 更好的样本占 **{better:.1%}**',
                  f'- 桶级翻转:semantic 差 → notext 好 **{flips}** 个;semantic 好 → notext 差 {back} 个', '']
        top = sorted(change, key=lambda n: note_r[n]['iou1'] - sem_r[n]['iou1'], reverse=True)[:3]
        lines += ['| 样本 | 域 | semantic IoU1 | notext IoU1 | Δ |', '|---|---|---|---|---|']
        for n in top:
            ds_ = note_r[n]['iou1'] - sem_r[n]['iou1']
            lines.append(f'| {n[:-4]} | {sem_r[n]["domain"]} | {sem_r[n]["iou1"]:.3f} '
                         f'| {note_r[n]["iou1"]:.3f} | {ds_:+.3f} |')
        lines.append('')
    return lines


def observations(configs):
    """Factual auto-observations computed across all configs."""
    lines = ['## 3. 自动观察要点(纯计算事实,解读请人工补充)', '']
    stats = []
    for d in configs:
        ch = [r for r in d['records'] if r['gt_pos'] > 0]
        nc = [r for r in d['records'] if r['gt_pos'] == 0]
        if not ch:
            continue
        good = sum(r['bucket'] == 'good' for r in ch) / len(ch)
        bad = sum(r['bucket'] == 'bad' for r in ch) / len(ch)
        iou, auc = mean(r['iou1'] for r in ch), mean(r['auc'] for r in ch)
        gt, pr = mean(r['gt_pos'] for r in ch), mean(r['pred_pos'] for r in ch)
        sev = sum(r['bucket'] == 'nc_severe' for r in nc) / len(nc) if nc else float('nan')
        # worst bad-concentration domain
        by_dom = {}
        for r in ch:
            by_dom.setdefault(r['domain'], []).append(r)
        worst_dom = max(by_dom, key=lambda dm: sum(r['bucket'] == 'bad' for r in by_dom[dm]) / len(by_dom[dm]))
        stats.append(dict(d=d, tag=f'{WEIGHTS_CN[d["weights"]]}/{BASE_CN[d["baseline"]]}',
                          good=good, bad=bad, iou=iou, auc=auc, gt=gt, pr=pr, sev=sev,
                          worst_dom=worst_dom, worst_dom_bad=max(
                              sum(r['bucket'] == 'bad' for r in by_dom[worst_dom]) / len(by_dom[worst_dom]), 0.0)))
    if not stats:
        return lines + ['(无数据)']

    best = max(stats, key=lambda s: (s['good'], s['iou']))
    lines.append(f'- 好样本占比最高:**{best["tag"]}** {best["good"]:.1%}'
                 f'(逐样本 IoU1 均值 {best["iou"]:.3f},n={sum(r["gt_pos"] > 0 for r in best["d"]["records"])})')
    for s in stats:
        bias = '过报' if s['pr'] / s['gt'] > 1.2 else ('漏报' if s['pr'] / s['gt'] < 0.8 else '基本平衡')
        lines.append(f'- {s["tag"]}:差样本 {s["bad"]:.1%},集中于 **{s["worst_dom"]}** 域'
                     f'({s["worst_dom_bad"]:.0%});整体预测正例率/GT 正例率 = {s["pr"] / s["gt"]:.2f}({bias});'
                     f'无变化样本严重误报 {s["sev"]:.1%}')
    gap = max(stats, key=lambda s: s['auc'] - s['iou'])
    lines.append(f'- 排序 vs 阈值化:AUC−IoU1 差距最大的是 **{gap["tag"]}**({gap["auc"]:.3f} − {gap["iou"]:.3f} '
                 f'= {gap["auc"] - gap["iou"]:+.3f})——像素排序能力明显高于阈值化后的产出,固定阈值是主要损失来源')
    d0 = best['d']
    ch = [r for r in d0['records'] if r['gt_pos'] > 0]
    small = [r['iou1'] for r in ch if r['gt_pos'] < 0.01]
    large = [r['iou1'] for r in ch if r['gt_pos'] >= 0.05]
    if small and large:
        lines.append(f'- 变化区域大小的影响({best["tag"]}):GT 正例率 <1% 组 IoU1 均值 {mean(small):.3f}'
                     f'(n={len(small)}),≥5% 组 {mean(large):.3f}(n={len(large)})')
    lines.append('')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--embed', type=int, default=4,
                    help='images embedded per ranking list (0 = tables only)')
    args = ap.parse_args()

    configs = load_configs()
    if not configs:
        sys.exit(f'no JSONs under {os.path.join(RESULTS_DIR, "per_sample")} — run analyze_samples.py first')

    pairs = []
    for w in WEIGHTS_ORDER:
        sem = next((d for d in configs if d['weights'] == w and d['baseline'] == 'semantic'), None)
        note = next((d for d in configs if d['weights'] == w and d['baseline'] == 'notext'), None)
        if sem and note:
            pairs.append((sem, note))

    fracs = {d['sample_frac'] for d in configs}
    if len(fracs) == 1:
        frac_txt = '全量' if fracs == {1.0} else f'每域随机抽样 {min(fracs):.0%}'
    else:
        frac_txt = (' / '.join(f'{f:.0%}' for f in sorted(fracs))
                    + '(逐配置见各节标题;比例不同的配置样本集合不同,不可直接对比)')

    lines = ['# 样本级预测质量分析(好/坏样本占比)', '',
             f'**协议**:与 report.md 主实验同源的推理协议(滑窗原生分辨率、冻结 CLIP、单一固定全局阈值 0.5,'
             f'评测全程不使用任何标签);评测集 = {frac_txt},种子固定。'
             f'**分桶**:有变化样本按固定阈值 0.5 下 IoU1:好 ≥0.5 / 中 0.2–0.5 / 差 <0.2;'
             f'无变化样本(GT 全背景)按预测正例率:干净 ≤5% / 轻微 5–20% / 严重 >20%。'
             f'**AUC** 为逐样本阈值无关的像素排序能力,与 IoU1 的差距反映「排序尚可但固定阈值吃亏」。'
             f'抽样子集的绝对数值与全量主实验存在抽样误差,分域方向性结论可用。'
             f'TPT 变体逐样本代价高,未纳入本分析。', '',
             f'生成时间:{time.strftime("%Y-%m-%d %H:%M")}', '']

    lines += [f'共 {len(configs)} 个配置:']
    for d in configs:
        ch = [r for r in d['records'] if r['gt_pos'] > 0]
        good = sum(r['bucket'] == 'good' for r in ch) / len(ch) if ch else float('nan')
        lines.append(f'- {cfg_title(d)} — 好 {good:.1%},像素级 mF1 {d["aggregate"]["mF1"]:.3f}')
    lines.append('')

    lines += ['## 1. 逐配置分析', '']
    for d in configs:
        lines += config_section(d, args.embed)

    if pairs:
        lines += cross_section(pairs)
    else:
        lines += ['## 2. semantic vs notext 交叉对比', '',
                  '(需同一 weights 的 semantic 与 notext 两个配置都有数据,当前缺失)', '']

    lines += observations(configs)

    out = os.path.join(RESULTS_DIR, 'sample_analysis.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('saved ->', out)


if __name__ == '__main__':
    main()
