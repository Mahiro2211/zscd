# zscd — Zero-Shot Domain-Generalization Change Detection

冻结 CLIP 在 WHU-GCD test set II(`~/Dataset/test2`,3,906 对,5 域)上的零样本变化检测,
含 Dense-TPT(测试时提示调优)稠密化改造。详见 `notes_paper.md`(论文笔记)与仓库根 `AGENT.md`。

## 结构
```
vendor/                自包含 CLIP 构件(OpenAI 原版 Transformer/tokenizer,逐字搬运,
                       与 denseclip 数值一致)— 无需外部 denseclip 目录
paths.py               可移植路径(RESULTS/CKPT/DATA,支持 ZSCD_CKPT_DIR/ZSCD_DATA_ROOT 覆盖)
data/cd_dataset.py    PairedChangeDataset(im1/im2/label,按前缀分域,CLIP 归一化)
data/viewgen.py       双时相同参数随机裁剪视图(TPT 用)
models/weights.py     三权重统一加载:OpenAI JIT / open_clip(RemoteCLIP, GeoRSCLIP)
models/clip_tower.py  DenseViT:任意 patch/任意输入尺寸,输出 (global, dense[1,D,h,w])
models/text_prompts.py 6 地类 + 模板(Zan et al. 2025 类别体系)
models/cd_model.py    SemanticChangeCLIP:pixel-text 相似度 → P1/P2 → change=1−ΣP1P2;
                      contexts 可学习(TPT), ensemble 文本特征(零样本)
tpt/dense_tpt.py      Dense TPT:视图熵筛选 top10% → 边际熵最小化(逐像素)→ 每对重置
eval/metrics.py       mF1/IoU1/OA(论文口径)+ 256-bin 直方图任意阈值/最优阈值
run_zeroshot.py       零样本(--baseline semantic|notext)
run_tpt.py            Dense TPT
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
4. **单一全局阈值(论文可比口径)**:按域分层 5% 校准子集(种子 0)选阈值后冻结,
   在其余 95% 上评测(`calibrated` 列)。fixed@0.5 与 oracle\*(评测标签扫描,仅分析)为辅助口径。
5. **TPT 单列**:逐样本测试时自适应(不用标签、调后重置)与论文"固定模型直测"分开报告。

## 关键设计决策(实验驱动)
1. **CLIP 窗口 224**:整图降采样到 224 会损失细节;504(pos-embed 插值)会退化零样本特征
   (AUC 与 mF1/IoU 双验证)。滑窗 224 兼顾原生输入与全分辨率输出。
2. **温度 τ=0.07**(DenseCLIP identity-loss 先例):CLIP 原生 logit_scale=100 使 6 类后验近均匀,
   change 概率饱和在 1−1/6≈0.833;τ=0.07 是合理的折中。
3. **诊断结论**(子集探测,levir+second+clcd 24 对):
   - feat_cos(1−cos(F1,F2),无文本)AUC 0.69-0.78,**最强零样本信号**
   - semantic(1−ΣP1P2)AUC 0.49-0.69,弱但可调(是 TPT 的作用对象)
   - 概率余弦/JSD ≈ 随机;simcos(文本子空间余弦)0.57-0.67
   - 权重排序:GeoRSCLIP > OpenAI ≈ RemoteCLIP(feat_cos 口径)
4. **TPT 仅调 prompt 上下文**(8 token 初始化 "a satellite image of a"),忠实原版;
   视觉特征 no_grad 缓存,仅文本侧反向;最终预测同样用滑窗(context 模式)。

## 运行
```bash
python -m zscd.run_zeroshot --weights georsclip                       # 语义一致性(滑窗+校准阈值)
python -m zscd.run_zeroshot --weights georsclip --baseline notext      # 无文本消融
python -m zscd.run_tpt --weights georsclip --n-views 32               # Dense TPT(同协议)
```
结果 JSON(`*_slide.json`)与可视化落在 `results/`,汇总见 `results/report.md`。
