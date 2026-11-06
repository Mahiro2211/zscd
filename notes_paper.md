# Zan et al. 2025 论文阅读笔记(ISPRS J. P&RS 225: 275-290, DOI: 10.1016/j.isprsjprs.2025.04.023)

> 论文:Open-vocabulary generative vision-language models for creating a large-scale remote sensing change detection dataset
> 全文提取:`paper_fulltext.txt`(16 页,96K 字符)

## 1. 核心贡献
用 Stable Diffusion(开放词表生成式 VLM)合成大规模变化检测数据集 **WHU-GCD**:
- **SD-T2I-RS**:在场景分类(AID/Million-AID)+ 语义分割数据(LoveDA/Evlab-SS/LandCover.ai/Berlin/Chicago)上微调 SD 文生图
- **SD-Inpainting-RS**:局部重绘模型,用于在真实影像指定掩码内生成指定类别变化
- CLIP 用于学习遥感词汇(bird's-eye view 风格 prompt);SAM 点提示提取同质区域掩码
- 核心结论:**纯生成式数据集训练的模型域外泛化优于真实数据集训练的模型**

## 2. 类别体系(本项目语义一致性法的文本 prompt 依据)
6 个土地覆盖类别,加 "RS" 前缀作为鸟瞰域标识(DreamBooth 风格 "[domain identifier] [class noun]"):

| 类别 | 文本 caption | AID 对应场景 |
|---|---|---|
| Building | RS building | Center, church, commercial, dense/sparse/medium residential, industrial, resort, storage tanks |
| Road | RS road | Bridge, viaduct, intersection, road, roundabout |
| Water | RS water | Pond, river |
| Barren | RS barren | Bare land |
| Forest | RS forest | Forest |
| Agriculture | RS agriculture | Farmland |

**25 个可行 from-to 变化方向**(Table 3,基于圆度/面积约束):
- Building → {Barren, Forest, Agriculture};Road(圆度>e_road) → {Barren, Forest};Water(圆度<e_river) → {Barren},(圆度>e_pond) → {Forest, Agriculture};Barren(面积>S_large 且圆度>e_square) → {Road, Water};Forest → {Building, Water};Agriculture → {Building, Road}

## 3. WHU-GCD 数据集结构
- 训练数据:7 个公开 RS 数据集 → 6,212 场景级样本 + 17,037 分割级 tile(512×512,GSD 0.3~0.5m)
- 变化样本:Part1(分割数据生成,24,577 个)+ Part2(Google Earth 双时相 22 景 513 km²,SAM+inpainting 生成)
- 划分:训练 24,167 / 验证 600 / **test set I** 3,300(生成式,域内)
- **test set II(域外)= 本项目 ~/Dataset/test2**:合并 5 个真实 CD 数据集测试集,统一裁剪 512×512,共 **3,906 对**:

| 来源 | 原始图像 | 裁剪后对数 | 关注地物 |
|---|---|---|---|
| DSIFN | 48 张 1024² | 192 | Building, road, agriculture |
| LEVIR-CD | 348 张 1024² | 1,392 | Building |
| SECOND | — | 1,694 | Building, water, low vegetation, tree, playground, ground |
| CLCD | — | 120 | Building, road, lake, bare land |
| CNAM-CD | 随机选 508 对 | 508 | Impervious surface, water, bare land, vegetation, other |
| **合计** | | **3,906** | |

与本地数据前缀完全对应:`dsifn_/levir_/second_/clcd_/cnam_`。二值标签(0=不变, 255=变化),14.8% 标签全空。

## 4. 实验设置与指标(本项目对齐)
- 硬件:2×RTX A6000 48GB;CE 损失,AdamW(wd=0.05, β=0.9/0.999),poly 衰减 lr=0.001,batch 32,10k~40k 迭代
- 训练增强:180° 旋转、裁剪、翻转
- **主指标**:
  - **mF1** = (F1_0 + F1_1)/2(不变类与变化类 F1 平均)
  - **IoU1** = TP1/(TP1+FP1+FN1)(变化类 IoU)
  - SCD 另用 Kappa、mIoU_cls、SeK
- 评测方法(监督基线):FC-EF/FC-Siam-diff/FC-Siam-conc(Daudt 2018)、IFNet、TinyCD、BIT_S3、ChangeFormer、Changer

## 5. 参考性能水平(供零样本结果对照)
- 互泛化实验(Table 13,ChangeFormer 训练于 WHU-GCD,测试于其他域):mF1 62~75,IoU1 21~41
- 真实数据集训练的模型域外普遍差:过度假阴性(FN)是泛化差主因
- Transformer 方法(BIT/ChangeFormer)泛化优于 CNN(Changer 在 test II 上落后 ChangeFormer 8.1 IoU1)
- WHU-GCD 训练的模型在 test set II 上:ChangeFormer 最优,BIT_S3 次优

## 6. 对本项目的启示
1. 6 地类 + "RS" 前缀 caption 直接用作逐像素分类的 CLIP 文本 prompt
2. bird's-eye view 语境:prompt 模板可扩展为 "a satellite image of a building" 等,与 "RS building" 两种风格都值得测试
3. 25 个 from-to 方向可作可选消融(限制可行变化方向 vs 任意类别不一致即变化);但 test2 各域地物体系不严格遵循 6 类,默认不限制
4. 零样本(无训练)结果预期显著低于监督方法(IoU1 20-40),重点在于三种预训练权重的**相对**泛化差异与 TPT 的增益
