# zscd — Zero-Shot Domain-Generalization Change Detection

冻结 CLIP 在 WHU-GCD test set II(`~/Dataset/test2`,3,906 对,5 域)上的零样本变化检测,
四轨差分构造并列(semantic / notext / absdiff_raw / absdiff_norm)。详见 `notes_paper.md`(论文笔记)。

## 结构
```
vendor/                自包含 CLIP 构件(OpenAI 原版 Transformer/tokenizer,逐字搬运,
                       与 denseclip 数值一致)— 无需外部 denseclip 目录
paths.py               可移植路径(RESULTS/CKPT/DATA,支持 ZSCD_CKPT_DIR/ZSCD_DATA_ROOT 覆盖)
data/cd_dataset.py    PairedChangeDataset(im1/im2/label,按前缀分域,CLIP 归一化)
models/weights.py     三权重统一加载:OpenAI JIT / open_clip(RemoteCLIP, GeoRSCLIP)
models/clip_tower.py  DenseViT:任意 patch/任意输入尺寸,输出 (global, dense[1,D,h,w])
models/text_prompts.py 6 地类 + 模板(Zan et al. 2025 类别体系)
models/cd_model.py    SemanticChangeCLIP:pixel-text 相似度 → P1/P2 → change=1−ΣP1P2;
                      ensemble 文本特征(零样本);change_score() 统一四轨分发
eval/metrics.py       mF1/IoU1/OA(论文口径,fixed@0.5)+ 256-bin 直方图任意阈值/最优阈值
run_zeroshot.py       零样本(--baseline semantic|notext|absdiff_raw|absdiff_norm)
make_report.py        汇总 report.md(四轨 × 三权重,附 Zan Table 14 监督参照)
sanity_check.py       权重加载健全性检查
```

## 权重(ViT-L/14;三者唯一公共规格——RemoteCLIP/GeoRSCLIP 均无 ViT-B/16)
| 名称 | 来源 | 格式 |
|---|---|---|
| openai | OpenAI ViT-L/14(~/.cache/clip) | JIT .pt |
| remoteclip | chendelong/RemoteCLIP HF | open_clip |
| georsclip | Zilun/GeoRSCLIP RS5M_ViT-L-14-336 | open_clip |

## 评测协议(与 Zan et al. 2025 对齐)
1. **数据**:test set II 原样(3,906 对,512×512,二值标签),逐像素混淆矩阵。
2. **指标**:mF1=(F1₀+F1₁)/2、IoU1=TP₁/(TP₁+FN₁+FP₁)(论文式 4-8 同口径),另附 OA/P/R。
3. **原生分辨率预测**:滑窗推理(`inference.py`)——CLIP 只看原生 224×224 窗口
   (stride 112,重叠平均),预测图拼成完整 512×512,与论文方法全分辨率直测同粒度。
4. **单一固定全局阈值 0.5(论文可比口径)**:与监督基线内置 0.5/argmax 决策同精神,
   评测全程不使用任何标签(`fixed` 列)。oracle\*(评测标签扫描,仅分析)为辅助口径。
   已知代价(georsclip 全量实测,fixed@0.5 vs oracle):notext 分数整体偏低
   (oracle thr≈0.17),fixed@0.5 下近乎漏检全部变化;semantic 分数整体偏高
   (oracle thr≈0.83),0.5 处过报、mF1 塌缩;absdiff 相对 L1 天然有界居中
   (oracle thr≈0.53),fixed@0.5 与 oracle 几乎持平(IoU1 23.4 vs 25.8)——
   免标签口径下 absdiff 双轨反而最稳。
5. **监督参照**:Zan Table 14 test set II 六个全监督基线(ChangeFormer 最强 70.95 mF1/35.74 IoU1)
   编入 `make_report.py` 对比表,零样本目标即对照这组掉下来的域外数字。

## 关键设计决策(实验驱动)
1. **CLIP 窗口 224**:整图降采样到 224 会损失细节;504(pos-embed 插值)会退化零样本特征
   (早期对比结论,AUC 与 mF1/IoU 双验证;**原始产物未入库**,仅存结论)。滑窗 224
   兼顾原生输入与全分辨率输出。
2. **温度 τ=0.07**(DenseCLIP identity-loss 先例,Zhou et al., ECCV 2022, arXiv:2112.01071;
   数值依据为本地 denseclip 移植代码,论文原文待亲验):CLIP 原生 logit_scale=100 使
   6 类后验近均匀,change 概率饱和在 1−1/6≈0.833(本仓库推理);τ=0.07 是合理的折中。
3. **诊断结论**(早期子集探测,levir+second+clcd 24 对;**原始日志未保留**,仅供设计参考):
   - feat_cos(1−cos(F1,F2),无文本)AUC 0.69-0.78,**最强零样本信号**
   - semantic(1−ΣP1P2)AUC 0.49-0.69
   - 概率余弦/JSD ≈ 随机;simcos(文本子空间余弦)0.57-0.67
   - 权重排序:GeoRSCLIP > OpenAI ≈ RemoteCLIP(feat_cos 口径)
4. **absdiff 双轨消融**(`--baseline absdiff_raw|absdiff_norm`):监督式 |d1−d2| 做差的
   零样本版(类比,无具体文献出处),以相对 L1 距离 Σ|d1−d2|/(Σ|d1|+Σ|d2|) 实现
   三角不等式天然有界 [0,1],
   满足 256-bin 直方图协议(裸 L1 无界会被 clip 到顶 bin)。raw 保留双期范数漂移
   (norm drift 记为变化,含光照/季节信息,未测过的新变体);norm 先逐像素 L2 归一化
   (仅方向变化,近似与 feat_cos 单调,固定阈值 0.5 下预期≈notext,作 sanity check)。
   完全绕开 softmax 与 τ 超参。
5. **模板平均 prompt ensemble**:文本特征取多模板平均,为 CLIP 原论文标准做法
   (Radford et al., 2021, prompt ensembling);模板本身为本仓库构造(见 notes_paper.md)。

## 运行
```bash
python -m zscd.run_zeroshot --weights georsclip                       # 语义一致性(滑窗+固定阈值0.5)
python -m zscd.run_zeroshot --weights georsclip --baseline notext      # 无文本消融
python -m zscd.run_zeroshot --weights georsclip --baseline absdiff_raw  # |d1-d2| 相对L1,原始特征
python -m zscd.run_zeroshot --weights georsclip --baseline absdiff_norm # |d1-d2| 相对L1,归一化特征
```
结果 JSON(`*_slide.json`)与可视化落在 `results/`,汇总见 `results/report.md`。
