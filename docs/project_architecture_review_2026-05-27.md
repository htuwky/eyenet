# EyeNet 工程审计与整理记录

日期：2026-05-27  
范围：项目结构、实验协议、文档一致性、脚本入口、最小工程规范

## 1. 当前结论

项目已经从早期 EMS 建模原型推进到一个可复现的研究工程系统。当前应停止继续扩展长训练任务，优先完成阶段一 encoder 主线的文档、报告、论文初稿和工程规范整理。

当前主线：

```text
fixation/event schema
-> masked event modeling encoder pretraining
-> EMS aligned five-seed fine-tuning
-> phase-1 encoder result
```

当前不作为主线继续推进：

```text
old segment-GRU dual-stream
new strict summary+encoder dual-stream
Transformer reruns
Saliency4ASD adapter work
```

这些方向可以保留为 exploratory 或 future work。

## 2. 工程结构状态

当前仓库结构基本合理：

```text
configs/       数据集、实验和特征 schema 配置
data/splits/   小型可复现 subject split 文件
docs/          协议、实验总结、汇报和论文草稿
scripts/       CLI 入口
src/eyenet/    可复用包代码
tests/         最小工程测试
```

数据和大文件策略合理：

```text
data/raw/
data/processed/
experiments/
outputs/
checkpoints/
Hospital_Data/
```

这些目录由 `.gitignore` 排除，避免把原始数据、处理后数据、实验输出和 checkpoint 提交到 Git。

## 3. 当前源码检查

只读审计结果：

```text
src/ + scripts/ + tests/ Python files: 119
AST parse errors: 0
script entrypoints checked: 72
missing main guard: 0
```

关键模块导入检查通过：

```text
eyenet.data.subject_summary
eyenet.models.dual_stream
eyenet.training.summary_encoder_dual_stream
```

注意：`pytest` 当前没有安装在 base 或 `eyenet` conda 环境中，因此新增测试文件尚未通过 `pytest` 正式执行。已用直接函数调用方式做了轻量验证。

## 4. 文档一致性问题

已发现并修复的主要问题：

1. `README.md` 仍把 Transformer、dual-stream 和 Saliency4ASD 写成当前下一步。
2. `scripts/README.md` 没有列出新 summary/dual-stream/phase-1 汇总脚本。
3. 组会报告 markdown 是 mojibake 乱码。
4. 论文初稿 markdown 是 mojibake 乱码。
5. `create_phase1_docx_deliverables.py` 有乱码 footer 和语法风险。
6. 本工程复盘文档本身也是 mojibake 乱码。

整理后的文档原则：

```text
phase-1 encoder summary 是当前主线；
dual-stream 是探索性收尾；
Transformer 是 future exploratory；
组会报告和论文初稿只引用 strict aligned five-seed 主结果。
```

## 5. 当前实验源文件

阶段一 encoder source of truth：

```text
experiments/encoder_downstream/phase1_encoder_summary.csv
experiments/encoder_downstream/phase1_encoder_split_leakage_audit.csv
docs/current_experiment_summary.md
docs/encoder_model_selection_summary.md
```

双流探索性收尾：

```text
experiments/ems_encoder_dual_stream/old_encoder_dual_stream_summary.csv
experiments/ems_subject_summary_baseline_strict/summary.csv
experiments/ems_summary_encoder_dual_stream/summary.csv
docs/old_encoder_dual_stream_closure.md
docs/new_summary_encoder_dual_stream_closure.md
```

## 6. 工程规范补齐

本次补齐了最小规范：

```text
pyproject.toml
  [project.optional-dependencies].dev = pytest, ruff
  [tool.pytest.ini_options]
  [tool.ruff]
  [tool.ruff.lint]

tests/
  test_imports.py
  test_subject_summary.py
```

这不是完整测试体系，只是最低限度的工程护栏，用于防止核心模块不可导入和 strict summary feature selection 逻辑被误改。

## 7. Docx 交付物状态

已重新生成：

```text
docs/group_meeting_report_2026-05-28.docx
docs/paper_drafts/content_agnostic_eye_movement_screening_phase1_draft.docx
```

结构性检查通过：

```text
group report: 58 paragraphs, 4 tables
paper draft: 95 paragraphs, 3 tables
```

限制：

```text
本机缺少 LibreOffice/soffice，Documents renderer 无法输出 PNG。
因此本次不能声明 docx 已完成视觉渲染 QA，只能声明 docx 可打开、可提取文本、标题和表格结构存在。
```

## 8. 建议提交分组

建议不要把全部变更一次性混提交。推荐分三批：

1. 文档修复和阶段一报告：
   - README
   - scripts/README
   - group meeting report md/docx
   - phase-1 paper draft md/docx
   - architecture review

2. 双流和 summary exploratory 代码：
   - subject_summary.py
   - encoder_dual_stream.py
   - summary_encoder_dual_stream.py
   - old/new dual-stream scripts
   - closure docs

3. 工程规范：
   - pyproject dev config
   - tests/

如果需要快速提交到 GitHub，也可以先合并为一个“project cleanup and phase-1 deliverables”提交，但后续维护性会差一些。

## 9. 下一步

建议顺序：

1. 本地检查两份 `.docx` 是否排版可接受。
2. 若排版需要细调，优先调 markdown 内容长度和表格列，而不是重写生成脚本。
3. 确认后做一次 `git status` 分组提交。
4. 之后再决定是否继续 Transformer 或新模型探索。
