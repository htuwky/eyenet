# Content-Agnostic Eye-Movement Event Representation Learning for Psychiatric Screening

阶段一论文初稿，导师讨论版  
Last updated: 2026-05-27

## 摘要

眼动行为为精神疾病风险筛查提供了一种低负担、非侵入式的行为表征来源。然而，现有眼动建模方法常依赖特定刺激内容、实验范式、AOI 定义或设备采样条件，限制了跨任务和跨数据集迁移。本研究提出一种内容无关的 fixation/event 级眼动表征学习框架 EyeNet。该框架将不同来源的眼动数据统一转换为 fixation-event 序列，并使用 masked event modeling (MEM) 进行自监督预训练，再在 EMS 精神分裂症与健康对照二分类任务上进行下游微调验证。

我们整合了 EMS、GazeBase、CRCNS eye-1、OneStop 和 HBN 等数据源，并建立严格的 EMS-anchor aligned split 协议，保证 EMS downstream test subjects 不进入自监督预训练的 train/validation split。基于五个 EMS subject-level split 的评估显示，BiGRU attention encoder 的 MEM 预训练主要通过 fine-tuning 初始化带来收益。EMS-only MEM fine-tune 在 balanced accuracy 上表现最好，达到 0.750 +/- 0.058；EMS+GazeBase+CRCNS+OneStop 在 AUC 上表现最好，达到 0.825 +/- 0.044，但 balanced accuracy 为 0.731 +/- 0.047。

结果表明，公共眼动数据对表征学习具有迁移价值，但并非简单地“数据越多越好”。Frozen probing 明显弱于 fine-tuning，提示当前预训练表征仍需要任务适配。旧双流和新 subject-summary 双流均未超过 encoder-only 主线，因此阶段一论文主线应聚焦内容无关 fixation/event encoder。

关键词：眼动；自监督学习；masked event modeling；精神分裂症筛查；内容无关表征；BiGRU

## 1. 引言

眼动数据能够反映个体的注意分配、视觉搜索、凝视稳定性和扫视动态等行为特征，因此长期被用于认知状态、神经发育障碍和精神疾病相关研究。对于精神分裂症等疾病，眼动异常可能表现为 fixation duration、saccade amplitude、scanpath organization 和视觉探索策略的改变。相比问卷或访谈，眼动采集具有非侵入、低负担和客观量化的潜力。

但眼动建模面临一个核心问题：许多方法依赖具体观看内容或实验范式。例如，基于 AOI、文字区域、图像语义或任务阶段的特征可以在单一实验中有效，但难以跨数据集复用。不同数据集之间还存在设备采样率、屏幕尺寸、实验任务和数据清洗流程差异。如果模型直接依赖这些因素，最终得到的可能是任务分类器或数据集分类器，而不是可迁移的眼动行为表征。

本项目的目标是构建一种尽量内容无关、范式弱相关、设备采样率弱相关的眼动事件表征模型。我们不使用图像 ID、视频语义、阅读文本、AOI 或设备型号作为模型输入，而是将所有数据源统一到 fixation/event 级序列，再通过自监督 masked event modeling 学习通用眼动动态模式。

本阶段研究问题包括：

1. fixation/event 级 MEM 预训练是否能提升 EMS SZ/HC 下游分类？
2. 公共眼动数据是否能作为额外自监督预训练来源提升迁移表现？
3. 不同公共数据组合是否存在稳定收益，还是存在负迁移？
4. frozen encoder probing 和 supervised fine-tuning 哪一种更适合作为下游策略？
5. 双流融合是否能稳定超过 encoder-only 主线？

## 2. 数据集

### 2.1 EMS

EMS 是当前主下游数据集，用于精神分裂症与健康对照二分类。当前进入 encoder-ready 表的数据包含 160 名有标签受试者和 225,159 个 fixation events。EMS 既可用于自监督 MEM 预训练，也用于 downstream supervised fine-tuning，但模型选择阶段必须保证 downstream test subjects 不进入 MEM train/validation。

### 2.2 GazeBase

GazeBase 是公共眼动数据集，当前使用视频任务 `VD1/VD2` 转换后的 fixation events，共 322 名受试者和 843,517 个 events。GazeBase 不提供目标精神疾病标签，因此仅作为自监督预训练来源。

### 2.3 CRCNS eye-1

CRCNS eye-1 是自然电影观看眼动数据，当前包含 16 名受试者和 67,172 个 fixation events。由于其自然观看范式与视频类范式更接近，它是当前重要的公共迁移候选之一。

### 2.4 OneStop

OneStop 是阅读场景眼动数据，当前包含 360 名受试者和 2,042,834 个 fixation events。它提供大量序列数据，但阅读范式与 EMS 任务可能存在差异，因此其收益需要通过实验验证。

### 2.5 HBN

HBN 当前有 1,244 名 usable subjects 和 1,684,382 个 events。HBN 具有丰富临床和行为标签，但标签体系复杂，本阶段不将其标签并入 EMS SZ/HC 二分类，只作为自监督预训练候选。

## 3. 方法

### 3.1 内容无关事件表征

所有数据集首先被转换到共享 fixation/event schema。当前 encoder 使用 `encoder_no_position_core`，共 13 个特征：

```text
x_norm
y_norm
duration_ms
log_duration_ms
saccade_dx_norm
saccade_dy_norm
saccade_amplitude_norm
saccade_angle_sin
saccade_angle_cos
transition_missing
is_first_event_in_segment
is_last_event_in_segment
event_index_in_segment_norm
```

该 schema 不包含刺激物 ID、图像内容、视频语义、AOI、文字位置、任务名称、设备型号或采样率 one-hot。所有连续特征只在训练 split 上拟合标准化器，避免 test 信息泄漏。

### 3.2 Encoder 架构

阶段一主模型为轻量 BiGRU attention encoder：

```text
input: [batch, sequence_length, 13]
projection: Linear(13 -> 64) + LayerNorm + ReLU + Dropout
temporal encoder: 1-layer bidirectional GRU, hidden_dim=64
event embedding: 128
pooling: masked attention pooling
```

自监督预训练使用 MEM head 重建被遮盖的事件特征；下游任务使用 supervised binary head 输出 EMS SZ/HC 概率。

### 3.3 Masked Event Modeling

MEM 采用 span masking，而不是随机独立遮盖单点事件。当前主设置为：

```text
mask_probability: 0.45
mask_span_events: 2-8
batch_size: 8
max_seq_len: 1500
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
gradient_clip_norm: 5.0
```

实现上使用 learnable mask token，避免用 0 替换 masked values 引入数值含义偏差。重建目标位于 dataloader-standardized feature space，避免原始量纲差异主导损失。

### 3.4 Aligned Split 协议

为避免目标数据集 test subjects 在自监督预训练中被看到，我们设计 EMS-anchor aligned split：

```text
对于每个 seed:
EMS downstream train subjects -> MEM train
EMS downstream valid subjects -> MEM valid
EMS downstream test subjects -> MEM test
非 EMS 数据集按 subject 独立划分
```

泄漏审计结果：

```text
checked split rows: 35
passed rows: 35
max overlap between downstream test subjects and MEM train: 0
max overlap between downstream test subjects and MEM valid: 0
expected overlap with MEM test: 32 per seed
```

这保证了阶段一主结果具有明确的 test subject 隔离。

### 3.5 下游评估

EMS 下游评估使用五个 subject-level 60/20/20 split，seeds 为 `0,1,2,3,4`。阈值只在 validation split 上选择，主阈值策略为 validation-selected best balanced accuracy。最终报告 test split 的 AUC、accuracy、balanced accuracy、sensitivity、specificity 和 F1。

## 4. 实验设计

阶段一主表包含以下模型：

1. Supervised-only BiGRU encoder，不使用 MEM 预训练。
2. EMS-only MEM BiGRU，fine-tune 和 frozen。
3. EMS+CRCNS MEM BiGRU，fine-tune 和 frozen。
4. EMS+GazeBase+CRCNS MEM BiGRU，fine-tune 和 frozen。
5. EMS+GazeBase+CRCNS+OneStop MEM BiGRU，fine-tune 和 frozen。
6. EMS+GazeBase+CRCNS+HBN MEM BiGRU，fine-tune 和 frozen。
7. EMS+All-public MEM BiGRU，fine-tune 和 frozen。

本阶段最终表排除：

```text
smoke tests
single-split screening
non-aligned exploratory runs
Transformer exploratory runs
old dual-stream exploratory runs
new summary dual-stream design work
```

## 5. 结果

### 5.1 阶段一主结果

| Experiment | Mode | Pretraining Data | AUC Mean | AUC Std | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU | fine-tune | EMS | 0.798 | 0.064 | **0.750** | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + OneStop | **0.825** | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| EMS+CRCNS BiGRU | fine-tune | EMS + CRCNS eye-1 | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 | 0.736 |
| EMS+GazeBase+CRCNS BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 | 0.800 | 0.049 | 0.725 | 0.026 | 0.725 | 0.725 | 0.722 |
| EMS+CRCNS aligned BiGRU, seq3000 | fine-tune | EMS + CRCNS eye-1 | 0.809 | 0.038 | 0.706 | 0.042 | 0.738 | 0.675 | 0.711 |
| EMS+GazeBase+CRCNS+HBN BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + HBN | 0.795 | 0.071 | 0.700 | 0.087 | 0.750 | 0.650 | 0.716 |
| Supervised-only BiGRU | supervised | none | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 | 0.732 |
| EMS+All-public BiGRU | fine-tune | EMS + GazeBase + CRCNS eye-1 + OneStop + HBN | 0.782 | 0.074 | 0.694 | 0.051 | 0.788 | 0.600 | 0.720 |

### 5.2 Frozen Encoder 结果

Frozen probing 整体明显弱于 fine-tuning：

| Experiment | Mode | AUC Mean | Balanced Accuracy Mean | Sensitivity Mean | Specificity Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| EMS+GazeBase+CRCNS+OneStop BiGRU | frozen | 0.740 | 0.644 | 0.800 | 0.488 |
| EMS+All-public BiGRU | frozen | 0.741 | 0.631 | 0.800 | 0.463 |
| EMS-only MEM BiGRU | frozen | 0.722 | 0.631 | 0.863 | 0.400 |
| EMS+GazeBase+CRCNS BiGRU | frozen | 0.721 | 0.613 | 0.788 | 0.438 |
| EMS+CRCNS BiGRU | frozen | 0.719 | 0.613 | 0.813 | 0.413 |
| EMS+CRCNS aligned BiGRU, seq3000 | frozen | 0.717 | 0.613 | 0.800 | 0.425 |
| EMS+GazeBase+CRCNS+HBN BiGRU | frozen | 0.715 | 0.613 | 0.788 | 0.438 |

### 5.3 双流探索结果

旧双流使用 pretrained encoder stream 加 segment-GRU macro stream，并比较 concat/gated fusion。新双流使用 pretrained encoder stream 加 strict subject-summary stream，并比较 concat/gated/residual-logit fusion。

| Model | AUC Mean | Balanced Accuracy Mean | Balanced Accuracy Std | Sensitivity Mean | Specificity Mean | F1 Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0.798 | **0.750** | 0.058 | 0.713 | 0.788 | 0.738 |
| EMS+GazeBase+CRCNS+OneStop BiGRU fine-tune | **0.825** | 0.731 | 0.047 | 0.775 | 0.688 | 0.740 |
| Strict summary-only logistic regression | 0.805 | 0.731 | 0.084 | 0.763 | 0.700 | 0.738 |
| Strict summary + encoder dual-stream gated | 0.806 | 0.725 | 0.046 | 0.675 | 0.775 | 0.699 |
| Strict summary-only SVM-RBF | 0.840 | 0.719 | 0.070 | 0.750 | 0.688 | 0.720 |
| Strict summary + encoder dual-stream residual-logit | 0.795 | 0.713 | 0.041 | 0.775 | 0.650 | 0.727 |
| Strict summary + encoder dual-stream concat | 0.809 | 0.706 | 0.065 | 0.738 | 0.675 | 0.701 |
| Old encoder + segment-GRU dual-stream gated | 0.811 | 0.725 | 0.105 | 0.838 | 0.613 | 0.756 |
| Old encoder + segment-GRU dual-stream concat | 0.799 | 0.694 | 0.078 | 0.675 | 0.713 | 0.688 |

## 6. 讨论

### 6.1 MEM 预训练的作用

相比 supervised-only BiGRU，最强 MEM fine-tune 设置在 balanced accuracy 上从 0.700 提升到 0.750。这说明 fixation/event 级自监督预训练能够为 EMS 下游任务提供有用初始化。但 frozen probing 整体较弱，说明当前预训练表征尚不足以直接线性迁移，需要 supervised fine-tuning 适配 EMS 标签。

### 6.2 公共数据融合不是越多越好

EMS+GazeBase+CRCNS+OneStop 获得最高 AUC，但 balanced accuracy 未超过 EMS-only MEM。加入 HBN 或全部 public 后，balanced accuracy 反而下降。这提示公共数据之间存在范式差异、年龄结构差异或噪声差异，简单合并并不能保证迁移收益。

### 6.3 指标选择影响主模型定义

如果主指标是 balanced accuracy，当前阶段一主模型应选择 EMS-only MEM BiGRU fine-tune。如果强调公共数据迁移和排序能力，EMS+GazeBase+CRCNS+OneStop 是最强 AUC 候选。因此论文中应明确区分“balanced accuracy 最优模型”和“public-data AUC 最优候选”。

### 6.4 双流模型的定位

双流实验验证了融合工程链路，也证明 strict subject-summary features 存在独立预测信号。但在当前 EMS 标签规模下，concat、gated 和 residual-logit 都未稳定超过 encoder-only balanced accuracy。因此双流不进入当前主线，只作为探索性负结果和下一阶段设计动机。

## 7. 局限性

第一，EMS 标签数据规模仍较小，只有 160 名有标签受试者。因此当前结果应被视为方法学验证和初步筛查证据，而不是临床诊断性能。

第二，公共数据集虽然已经转换到统一 event schema，但不同数据集之间仍存在任务范式、年龄分布、设备条件和数据采集流程差异。当前结果说明这些差异可能导致负迁移。

第三，当前阶段只系统收敛了 BiGRU encoder。Transformer 和更长序列模型仍属于探索性结果，尚未形成最终对照。

第四，下游标签当前只有 EMS SZ/HC。若未来加入其他临床标签，应采用 dataset-specific heads、multi-task learning 或 external validation，而不是直接混合不同疾病标签。

## 8. 下一步工作

短期工作：

1. 固化阶段一 encoder 结果表和泄漏审计。
2. 将 smoke、single-split、exploratory 和 final_aligned_5seed 明确分层。
3. 完成论文方法、实验和讨论部分初稿。
4. 更新 README、脚本索引和工程规范，保证后续复现实验入口清晰。

中期工作：

1. 在阶段一论文主线稳定后，再重新评估 Transformer 消融。
2. 如果继续做双流，应将 summary stream 作为解释性或辅助校正模块，而不是与 encoder stream 等容量竞争。
3. 在有新的同类临床标签数据后，再讨论跨数据集 supervised validation。

## 9. 阶段一结论

本阶段最重要的结论是：

```text
EyeNet 的 fixation/event 级自监督 encoder 主线已经形成可复现的阶段一结果。
MEM 预训练配合 supervised fine-tuning 优于 supervised-only baseline。
EMS-only MEM 是 balanced accuracy 最强结果。
EMS+GazeBase+CRCNS+OneStop 是公共数据融合 AUC 最强候选。
公共数据不是越多越好，HBN/all-public 当前没有提升。
双流没有超过 encoder-only 主线，暂作为探索性 evidence 收尾。
```

因此，下一阶段应优先完成论文和报告叙事，再谨慎开展新的 Transformer 或双流扩展。
