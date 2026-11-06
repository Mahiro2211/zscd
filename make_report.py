"""Aggregate results/zs_*_slide.json and results/tpt_*_slide.json into results/report.md.

Primary (paper-comparable) column: `calibrated` — single global threshold from
the stratified 5% calibration split, frozen, evaluated on the remaining 95%,
sliding-window native-resolution predictions. fixed@0.5 and oracle* are
auxiliary (oracle uses eval labels — analysis only).
"""
import glob
import json

from zscd.paths import RESULTS_DIR as RES
WNAMES = {'openai': 'OpenAI CLIP', 'remoteclip': 'RemoteCLIP', 'georsclip': 'GeoRSCLIP'}
DOMS = ['levir', 'second', 'cnam', 'dsifn', 'clcd', 'overall']


def load_all():
    runs = []
    for f in sorted(glob.glob(f'{RES}/zs_*_slide.json') + glob.glob(f'{RES}/tpt_*_slide.json')):
        with open(f) as fp:
            runs.append(json.load(fp))
    return runs


def main():
    runs = {}
    for r in load_all():
        k = (r['weights'], r.get('baseline', 'semantic'), 'tpt' if 'n_views' in r else 'zs')
        runs[k] = r

    L = []
    L.append('# 零样本域泛化变化检测实验报告\n')
    L.append('**协议(与 Zan et al. 2025 对齐)**:WHU-GCD test set II(test2,3,906 对)逐像素评测;'
             '指标 mF1=(F1₀+F1₁)/2、IoU1(变化类);滑窗原生分辨率预测(224 窗口拼成完整 512×512);'
             '单一全局阈值取自按域分层 5% 校准子集(种子 0,不接触评测像素)后冻结。'
             '**calibrated 列为与论文同口径的数字**;fixed@0.5 与 oracle*(用评测标签扫阈值,仅分析)为辅助。\n')
    L.append('全部实验零训练:冻结 CLIP ViT-L/14。TPT 仅逐样本调 8 个 prompt 上下文向量后重置。\n')

    L.append('\n## 1. 总体对比(calibrated 口径,eval 95% 子集)\n')
    L.append('| 权重 | 方法 | mF1 | IoU1 | F1_change | thr |')
    L.append('|---|---|---|---|---|---|')
    for w in ['openai', 'remoteclip', 'georsclip']:
        for mode, label in [('semantic', '语义一致性'), ('notext', '无文本 feat-cos')]:
            r = runs.get((w, mode, 'zs'))
            if r:
                c = r['report']['overall']['calibrated']
                L.append(f"| {WNAMES[w]} | {label} (零样本) | {c['mF1']:.4f} | {c['IoU1']:.4f} "
                         f"| {c['F1_change']:.4f} | {c['threshold']:.3f} |")
        t = runs.get((w, 'semantic', 'tpt'))
        if t:
            c = t['report']['overall']['calibrated']
            L.append(f"| {WNAMES[w]} | 语义一致性 + Dense-TPT | {c['mF1']:.4f} | {c['IoU1']:.4f} "
                     f"| {c['F1_change']:.4f} | {c['threshold']:.3f} |")

    L.append('\n## 2. 分域明细(calibrated mF1 / IoU1)\n')
    L.append('| 权重 | 方法 | ' + ' | '.join(DOMS) + ' |')
    L.append('|' + '---|' * (len(DOMS) + 2))
    for w in ['openai', 'remoteclip', 'georsclip']:
        for mode, label in [('semantic', '语义'), ('notext', '无文本')]:
            r = runs.get((w, mode, 'zs'))
            if not r:
                continue
            cells = []
            for d in DOMS:
                c = r['report'][d]['calibrated']
                cells.append(f"{c['mF1']:.3f}/{c['IoU1']:.3f}")
            L.append(f'| {WNAMES[w]} | {label} (zs) | ' + ' | '.join(cells) + ' |')
        t = runs.get((w, 'semantic', 'tpt'))
        if t:
            cells = []
            for d in DOMS:
                c = t['report'][d]['calibrated']
                cells.append(f"{c['mF1']:.3f}/{c['IoU1']:.3f}")
            L.append(f'| {WNAMES[w]} | 语义 + TPT | ' + ' | '.join(cells) + ' |')

    L.append('\n## 3. 辅助口径(overall)\n')
    L.append('| 权重 | 方法 | mF1@0.5 | IoU1@0.5 | mF1* | IoU1* |')
    L.append('|---|---|---|---|---|---|')
    for (w, mode, kind), r in sorted(runs.items()):
        o = r['report']['overall']
        label = WNAMES[w] + ' ' + ('语义' if mode == 'semantic' else '无文本') + ('+TPT' if kind == 'tpt' else '')
        L.append(f"| {WNAMES[w]} | {label} | {o['fixed']['mF1']:.4f} | {o['fixed']['IoU1']:.4f} "
                 f"| {o['best']['mF1']:.4f} | {o['best']['IoU1']:.4f} |")

    out = f'{RES}/report.md'
    with open(out, 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    print('\nsaved ->', out)


if __name__ == '__main__':
    main()
