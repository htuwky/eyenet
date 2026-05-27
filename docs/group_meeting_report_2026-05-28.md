# EyeNet 阶段一组会汇报提纲

汇报日期：2026-05-28  
当前版本：阶段一 encoder 主线收尾版

## 1. 一句话结论

过去两周的核心工作已经把 EyeNet 从“模型和数据源探索”收敛到一个可复现的阶段一结论：

```text
以内容无关 fixation/event 序列作为统一输入，
通过 masked event modeling 做自监督预训练，
再在 EMS SZ/HC 下游任务上做严格 aligned five-seed 评估。
当前 balanced accuracy 最强的是 EMS-only MEM fine-tune；
公共数据融合中 AUC 最强的是 EMS+GazeBase+CRCNS+OneStop。
```

关键判断：

```text
公共眼动数据有迁移价值，但不是数据越多越好。
MEM 预训练的价值主要体现在 fine-tuning 初始化上，frozen probing 不适合作为主下游策略。
旧双流和新 summary 双流都没有超过 encoder-only 主线，当前应作为探索性证据归档。
```

## 2. 项目目标

本项目不是构建依赖特定图片、视频、阅读材料、AOI 或眼动仪型号的分类器，而是构建：

```text
尽量内容无关、观看范式弱相关、设备采样率弱相关的眼动事件表征模型。
```

因此当前主输入被固定为 fixation/event 级特征，而不是原始 gaze 采样点或刺激内容特征。

当前 encoder 输入 schema 为 `encoder_no_position_core`，共 13 个特征：

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

不使用：

```text
图像 ID、视频语义、AOI、文字位置、任务名称、设备型号、采样率 one-hot。
```

## 3. 数据集状态

| 数据集 | 当前状态 | 受试者数 | fixation events | 当前角色 |
| --- | ---: | ---: | ---: | --- |
| EMS | 完成 | 160 | 225,159 | SZ/HC 下游验证主数据集 |
| GazeBase | 完成 | 322 | 843,517 | 公共自监督预训练 |
| CRCNS eye-1 | 完成 | 16 | 67,172 | 自然观看/视频域公共预训练 |
| OneStop | 完成 | 360 | 2,042,834 | 阅读场景公共预训练 |
| HBN | 完成 | 1,244 usable | 1,684,382 | 公共预训练候选，但当前收益不稳定 |
| Saliency4ASD | 暂缓 | TBD | TBD | ASD 标签不能和 EMS SZ/HC 混合 |

关键约束：

```text
EMS 是当前唯一主下游标签数据集。
GazeBase / CRCNS / OneStop / HBN 只用于自监督预训练或辅助分析。
不同疾病标签不能混成一个二分类标签。
```

## 4. 当前主模型

阶段一主模型是 BiGRU attention encoder：

```text
input: [batch, sequence_length, 13]
projection: Linear(13 -> 64) + LayerNorm + ReLU + Dropout
temporal encoder: 1-layer BiGRU, hidden_dim=64
event embedding: 128
pooling: masked attention pooling
pretraining head: reconstruct masked 13-dim event features
downstream head: EMS SZ/HC binary classifier
```

当前训练设置：

```text
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 8
max_seq_len: 1500
mask_probability: 0.45
mask_strategy: span
span length: 2-8 events
gradient_clip_norm: 5.0
```

## 5. 实验协议收敛

项目现在明确区分四类结果：

| 类型 | 含义 | 是否进入主结论 |
| --- | --- | --- |
| smoke | 工程跑通检查 | 否 |
| single_split | 单 split 快速筛选 | 否 |
| exploratory | 探索性实验，例如早期双流和 Transformer | 否 |
| final_aligned_5seed | 严格 aligned 五 seed | 是 |

阶段一主表只使用：

```text
strict EMS-anchor aligned five-seed results
seeds: 0,1,2,3,4
threshold: validation-selected best balanced accuracy
```

## 6. 泄漏审计

为了避免 EMS test subject 在自监督预训练阶段被看到，我们做了 aligned split 审计。

审计结果：

```text
checked split rows: 35
passed rows: 35
max overlap between downstream test subjects and MEM train: 0
max overlap between downstream test subjects and MEM valid: 0
expected overlap with MEM test: 32 per seed
```

结论：

```text
阶段一 aligned encoder 结果没有发现 downstream test subject 泄漏进 MEM train/valid。
```

## 7. 阶段一主结果

主指标：test balanced accuracy mean across five seeds。  
阈值选择：validation-selected best balanced accuracy。

| 预训练数据 | Mode | AUC Mean | AUC Std | Balanced Acc Mean | Balanced Acc Std | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EMS-only | fine-tune | 0.798 | 0.064 | **0.750** | 0.058 | 0.713 | 0.788 |
| EMS+GazeBase+CRCNS+OneStop | fine-tune | **0.825** | 0.044 | 0.731 | 0.047 | 0.775 | 0.688 |
| EMS+CRCNS | fine-tune | 0.813 | 0.062 | 0.725 | 0.060 | 0.788 | 0.663 |
| EMS+GazeBase+CRCNS | fine-tune | 0.800 | 0.049 | 0.725 | 0.026 | 0.725 | 0.725 |
| EMS+CRCNS seq3000 | fine-tune | 0.809 | 0.038 | 0.706 | 0.042 | 0.738 | 0.675 |
| EMS+GazeBase+CRCNS+HBN | fine-tune | 0.795 | 0.071 | 0.700 | 0.087 | 0.750 | 0.650 |
| Supervised-only BiGRU | supervised | 0.784 | 0.069 | 0.700 | 0.114 | 0.800 | 0.600 |
| EMS+All-public | fine-tune | 0.782 | 0.074 | 0.694 | 0.051 | 0.788 | 0.600 |

## 8. 结果解释

第一，MEM 预训练有价值，但主要体现在 fine-tuning 初始化上。

```text
Supervised-only balanced acc: 0.700
EMS-only MEM fine-tune balanced acc: 0.750
```

第二，frozen probing 不适合做主下游策略。

```text
多数 frozen balanced acc 在 0.61-0.64 左右，明显弱于 fine-tune。
```

第三，公共数据不是越多越好。

```text
EMS+GazeBase+CRCNS+OneStop AUC 最高：0.825
EMS+All-public balanced acc 下降到：0.694
HBN 融合没有带来稳定收益
```

第四，当前模型选择要按指标区分。

```text
如果主指标是 balanced accuracy：EMS-only MEM 最好。
如果强调公共数据迁移和 AUC：EMS+GazeBase+CRCNS+OneStop 最好。
```

## 9. 双流模型收尾

旧双流结构：

```text
pretrained encoder stream + segment-GRU macro stream + gated/concat fusion
```

新双流结构：

```text
pretrained encoder stream + strict subject-summary stream
fusion: concat / gated / residual-logit
```

五 seed 结果：

| 模型 | AUC Mean | Balanced Acc Mean | Balanced Acc Std | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMS-only MEM BiGRU fine-tune | 0.798 | **0.750** | 0.058 | 0.713 | 0.788 |
| EMS+GazeBase+CRCNS+OneStop BiGRU | **0.825** | 0.731 | 0.047 | 0.775 | 0.688 |
| Strict summary-only logistic regression | 0.805 | 0.731 | 0.084 | 0.763 | 0.700 |
| New summary+encoder gated | 0.806 | 0.725 | 0.046 | 0.675 | 0.775 |
| New summary+encoder residual-logit | 0.795 | 0.713 | 0.041 | 0.775 | 0.650 |
| New summary+encoder concat | 0.809 | 0.706 | 0.065 | 0.738 | 0.675 |
| Old dual gated | 0.811 | 0.725 | 0.105 | 0.838 | 0.613 |
| Old dual concat | 0.799 | 0.694 | 0.078 | 0.675 | 0.713 |

结论：

```text
summary-only 有独立信号，但当前融合方式没有稳定超过 encoder-only。
旧双流和新双流都应作为 exploratory negative evidence 收尾。
当前主模型仍然是 encoder-only MEM BiGRU fine-tuning。
```

## 10. 当前项目状态

```text
阶段一 encoder：已收尾，可用于组会和论文初稿
泄漏审计：通过
旧双流：探索性收尾
新 summary 双流：探索性收尾
Transformer：暂不继续打开主线
下一步：整理报告、论文初稿、工程规范和可复现入口
```

## 11. 汇报时建议强调的三句话

```text
第一，我们已经把项目从模型搜索收敛成严格 aligned five-seed encoder 评估。
第二，MEM 预训练有价值，但公共数据融合不是越多越好。
第三，双流没有超过 encoder-only，因此当前论文主线应该聚焦内容无关 event encoder。
```
