"""Aggregate results/zs_*_slide.json into results/report.md.

Primary (paper-comparable) column: `fixed` — the single FIXED global threshold
0.5 (same operating convention as the supervised baselines' built-in 0.5/argmax
decision; no labels are used anywhere), sliding-window native-resolution
predictions. oracle* (eval-label threshold sweep, analysis only) is auxiliary.

Supervised reference: Zan et al. 2025 Table 14 (test set II, models trained on
the WHU-GCD training set) — the bar zero-shot runs are measured against.
"""
import glob
import json

from zscd.paths import RESULTS_DIR as RES
WNAMES = {'openai': 'OpenAI CLIP', 'remoteclip': 'RemoteCLIP', 'georsclip': 'GeoRSCLIP'}
BLABELS = [('semantic', '语义一致性 1−ΣP1P2'),
           ('notext', '无文本 (1−cos)/2'),
           ('absdiff_raw', '|d1−d2| rel-L1 (raw)'),
           ('absdiff_norm', '|d1−d2| rel-L1 (norm)')]
DOMS = ['levir', 'second', 'cnam', 'dsifn', 'clcd', 'overall']

# Zan et al. 2025 Table 14, test set II (mF1/IoU1, %): supervised, trained on
# WHU-GCD train set.
SUPERVISED_TEST2 = [
    ('FC-Siam-Conc (监督)', 63.66, 26.88),
    ('FC-Siam-Diff (监督)', 65.11, 27.96),
    ('BIT_S3 (监督)', 69.70, 32.71),
    ('ChangeFormer (监督, 最强)', 70.95, 35.74),
    ('Changer (监督)', 67.07, 27.65),
    ('TinyCD (监督)', 68.60, 29.72),
]


def load_all():
    for f in sorted(glob.glob(f'{RES}/zs_*_slide.json')):
        with open(f) as fp:
            yield json.load(fp)


def main():
    runs = {}
    for r in load_all():
        runs[(r['weights'], r.get('baseline', 'semantic'))] = r

    L = []
    L.append('# 零样本域泛化变化检测实验报告\n')
    L.append('**协议(与 Zan et al. 2025 对齐)**:WHU-GCD test set II(test2,3,906 对)逐像素评测;'
             '指标 mF1=(F1₀+F1₁)/2、IoU1(变化类);滑窗原生分辨率预测(224 窗口拼成完整 512×512);'
             '单一**固定**全局阈值 0.5——与监督基线内置 0.5/argmax 决策同精神,评测全程不使用任何标签。'
             '**fixed 列为与论文同口径的数字**;oracle*(用评测标签扫阈值,仅分析)为辅助。\n')
    L.append('全部实验零训练:冻结 CLIP ViT-L/14,四轨差分构造并列(semantic / notext / absdiff_raw / absdiff_norm)。\n')

    L.append('\n## 1. 总体对比(overall,fixed@0.5 口径,%)vs Zan Table 14 全监督基线\n')
    L.append('| 权重 | 方法 | mF1 | IoU1 | F1_change |')
    L.append('|---|---|---|---|---|')
    for w in ['openai', 'remoteclip', 'georsclip']:
        for b, label in BLABELS:
            r = runs.get((w, b))
            if r:
                c = r['report']['overall']['fixed']
                L.append(f"| {WNAMES[w]} | {label} (零样本) | {c['mF1'] * 100:.2f} | {c['IoU1'] * 100:.2f} "
                         f"| {c['F1_change'] * 100:.2f} |")
    for name, mf1, iou1 in SUPERVISED_TEST2:
        L.append(f'| — | {name} | {mf1:.2f} | {iou1:.2f} | — |')

    L.append('\n## 2. 分域明细(fixed@0.5 mF1/IoU1,%)\n')
    L.append('| 权重 | 方法 | ' + ' | '.join(DOMS) + ' |')
    L.append('|' + '---|' * (len(DOMS) + 2))
    for w in ['openai', 'remoteclip', 'georsclip']:
        for b, label in BLABELS:
            r = runs.get((w, b))
            if not r:
                continue
            cells = [f"{r['report'][d]['fixed']['mF1'] * 100:.1f}/"
                     f"{r['report'][d]['fixed']['IoU1'] * 100:.1f}" for d in DOMS]
            L.append(f'| {WNAMES[w]} | {label} | ' + ' | '.join(cells) + ' |')

    L.append('\n## 3. Oracle 口径(仅分析:用评测标签扫最优阈值,overall,%)\n')
    L.append('| 权重 | 方法 | mF1* | IoU1* | thr* |')
    L.append('|---|---|---|---|---|')
    for (w, b), r in sorted(runs.items()):
        o = r['report']['overall']
        label = dict(BLABELS).get(b, b)
        L.append(f"| {WNAMES[w]} | {label} | {o['best']['mF1'] * 100:.2f} | {o['best']['IoU1'] * 100:.2f} "
                 f"| {o['best']['threshold']:.3f} |")

    out = f'{RES}/report.md'
    with open(out, 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    print('\nsaved ->', out)


if __name__ == '__main__':
    main()
