from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT = Path("docs/paper_drafts/content_agnostic_eye_movement_screening_draft.docx")
RESULT_TABLE = Path("experiments/ems_fixed_split/summary/fixed_split_publication_table.csv")
PRIMARY_METRICS = Path("experiments/ems_fixed_split/summary/fixed_split_primary_test_metrics.csv")
GATE_SUMMARY = Path("experiments/ems_fixed_split/summary/dual_stream_fusion/gated_gate_summary_by_split.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin_name, margin_value in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{margin_name}"))
        if node is None:
            node = OxmlElement(f"w:{margin_name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(margin_value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "0")
    tbl_ind.set(qn("w:type"), "dxa")

    tbl_grid = tbl.tblGrid
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, widths[idx])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_doc_defaults(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(5)

    title = styles["Title"]
    title.font.name = "Calibri"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title.font.size = Pt(21)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0x0B, 0x25, 0x45)
    title.paragraph_format.space_after = Pt(8)

    for style_name, size, color, before, after in [
        ("Heading 1", 15, RGBColor(0x1F, 0x4D, 0x78), 13, 6),
        ("Heading 2", 12.5, RGBColor(0x2E, 0x74, 0xB5), 9, 4),
        ("Heading 3", 11.5, RGBColor(0x1F, 0x4D, 0x78), 7, 3),
    ]:
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)


def add_footer(doc: Document) -> None:
    paragraph = doc.sections[0].footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Content-Agnostic Eye Movement Screening Model | Working Draft")
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def add_para(doc: Document, text: str, style: str | None = None) -> None:
    paragraph = doc.add_paragraph(style=style)
    paragraph.add_run(text)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run(item)


def add_numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(style="List Number")
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.add_run(item)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, text in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.text = text
        set_cell_shading(cell, "EAF2F8")
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(8.5)
    for row in rows:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = text
            for paragraph in cells[idx].paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                if idx > 0 and len(text) < 12:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8.5)
    set_table_geometry(table, widths)
    doc.add_paragraph()


def metric_rows_for_doc() -> list[list[str]]:
    rows = read_csv(RESULT_TABLE)
    name_map = {
        "dual_stream_concat": "Dual concat",
        "macro_behavior_stream": "Macro only",
        "ml_hist_gradient_boosting": "ML HistGB",
        "event_temporal_stream": "Event only",
        "dual_stream_gated": "Dual gated",
        "ml_svm_rbf": "ML SVM-RBF",
        "ml_random_forest": "ML RandomForest",
        "ml_logistic_regression": "ML Logistic",
        "ml_mlp": "MLP baseline",
    }
    ordered = [
        "ML HistGB",
        "ML SVM-RBF",
        "Macro only",
        "Event only",
        "Dual concat",
        "Dual gated",
    ]
    converted = []
    for row in rows:
        display = name_map.get(row["display_name"], row["display_name"])
        if display not in ordered:
            continue
        converted.append(
            [
                display,
                row["auc"],
                row["accuracy"],
                row["balanced_accuracy"],
                row["sensitivity"],
                row["specificity"],
                row["f1"],
                row["confusion_matrix"],
            ]
        )
    converted.sort(key=lambda row: ordered.index(row[0]))
    return converted


def primary_metric_lookup() -> dict[str, dict[str, str]]:
    rows = read_csv(PRIMARY_METRICS)
    return {row["display_name"]: row for row in rows}


def add_title_block(doc: Document) -> None:
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("基于内容解耦眼动特征的精神疾病初步筛查模型研究")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("EMS 数据集固定划分实验与双流模型工作草稿 | 2026-05-20")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    add_para(
        doc,
        "本文档是当前项目的论文雏形，记录截至目前已经完成的数据处理、模型设计、固定划分实验、双流融合实验和阶段性结论。后续接入跨数据集预训练和自采数据微调后，可在此基础上继续扩展为完整论文。",
    )


def build_doc() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics = primary_metric_lookup()
    gate_rows = read_csv(GATE_SUMMARY)

    doc = Document()
    set_doc_defaults(doc)
    add_footer(doc)
    add_title_block(doc)

    doc.add_heading("摘要草稿", level=1)
    add_para(
        doc,
        "精神疾病筛查通常依赖量表、临床访谈和专门范式，客观、低负担、可迁移的初筛工具仍然不足。本研究围绕内容解耦的眼动行为建模展开，目标是构建一种不依赖具体图片或视频语义、可适配不同采集设备和观看范式的初步筛查模型。研究首先基于 EMS 数据集完成统一预处理、fixation/saccade 事件建模、窗口级宏观行为特征提取和事件级动态序列构建；随后在 subject-level 60/20/20 固定划分下比较传统机器学习、Macro-Behavior Stream、Event-Temporal Stream、双流 concat 融合和双流 gated 融合。当前结果显示，Dual concat 获得最高 AUC 0.898，Macro only 获得 AUC 0.891，HistGradientBoosting 获得最高 balanced accuracy 0.844。门控融合在小样本条件下出现向 Macro 单流塌缩，test macro gate 平均值为 0.963，因而不作为最终主模型。本阶段结果支持采用 Macro-Behavior 主干和 Event-Temporal 辅助流的内容解耦双流建模路线，后续将接入更多公开眼动数据集进行跨数据集预训练和泛化验证。",
    )

    doc.add_heading("1. 研究目标与核心假设", level=1)
    add_para(
        doc,
        "本项目的核心目标不是建立只适用于 EMS 数据集或某一套图片刺激的分类器，而是建立一套面向医院初步筛查场景的内容解耦眼动模型。该模型应尽量只依赖跨设备、跨范式可获得的公共眼动信息，例如 gaze 坐标、timestamp、validity、采样率、屏幕参数和观看距离。图片编号、视频内容、AOI 语义、任务答案和瞳孔直径不作为核心输入。",
    )
    add_bullets(
        doc,
        [
            "临床目标：用于青少年或一般人群的初步风险筛查，而不是替代临床诊断。",
            "工程目标：支持后续接入不同硬件、不同采样率和不同观看范式的数据。",
            "建模目标：通过宏观行为流和微观事件时序流形成互补表征，提高模型的排序能力和筛查灵敏度。",
            "论文目标：证明内容解耦、跨数据集可扩展的眼动建模路线具有可行性，并为后续多数据集预训练和医院部署建立实验基础。",
        ],
    )

    doc.add_heading("2. 数据处理路线", level=1)
    add_para(
        doc,
        "当前阶段已完成 EMS 数据集的完整处理。处理流程强调统一模态和内容解耦：原始 fixation 数据被转化为事件表、窗口级宏观行为特征表和事件级时序表。IMAGE 字段仅作为原始试次边界或记录来源，不作为模型输入特征；模型不读取图片内容，也不使用图片语义。",
    )
    add_numbered(
        doc,
        [
            "统一 subject_id、label、fold、trial/image 边界等基础字段。",
            "基于屏幕物理参数和观看距离完成坐标归一化和 DVA 转换。",
            "构建 event table，保留 fixation 坐标、duration、saccade 转移距离、角度和速度等公共眼动事件特征。",
            "构建 Macro-Behavior segment features，在固定窗口中统计 fixation duration、scanpath length、BCEA、空间覆盖率、transition velocity 和 entropy 等宏观行为指标。",
            "构建 Event-Temporal sequence，将每个受试者的 fixation/saccade event 组织为事件序列，用于事件级动态建模。",
        ],
    )
    add_table(
        doc,
        ["处理产物", "文件", "用途"],
        [
            ["EMS event table", "data/processed/EMS/ems_events.csv", "统一 fixation/saccade 事件层数据"],
            ["Macro features", "ems_segment_features_no_pupil.csv", "Macro-Behavior Stream 输入"],
            ["Subject aggregate", "ems_subject_features_segment_agg_no_pupil.csv", "传统 ML baseline 输入"],
            ["Event sequence", "ems_event_temporal_sequences_no_pupil.csv", "Event-Temporal Stream 输入"],
            ["Fixed split", "ems_subject_split_60_20_20_seed42.csv", "主实验 train/valid/test 划分"],
        ],
        [1800, 3600, 3960],
    )

    doc.add_heading("3. 当前模型设计", level=1)
    add_para(
        doc,
        "当前模型从最初的 GCN + 微观时频设想调整为更适合小样本和跨数据集泛化的双流时序结构。Macro-Behavior Stream 使用窗口级统计序列，Event-Temporal Stream 使用 fixation/saccade 事件序列。两条流均采用 1-layer BiGRU + attention pooling，再在 subject-level 输出风险概率。",
    )
    add_table(
        doc,
        ["模型/分支", "输入", "结构", "当前作用"],
        [
            ["Traditional ML", "subject-level 聚合统计特征", "Logistic/SVM/RF/HistGB/MLP", "强 baseline，判断非深度方法上限"],
            ["Macro-Behavior Stream", "100 个窗口级宏观行为特征序列，37 维特征", "Linear projection + 1-layer BiGRU + attention", "当前最稳定的深度学习主干"],
            ["Event-Temporal Stream", "fixation/saccade 事件序列，22 维特征，最长 2020 events", "Linear projection + 1-layer BiGRU + attention", "捕捉更细粒度的事件动态，作为辅助流"],
            ["Dual concat", "Macro embedding + Event embedding", "两个编码器后直接 concat，再 MLP 分类", "当前主双流候选，AUC 最优"],
            ["Dual gated", "Macro embedding + Event embedding", "gate * Macro + (1-gate) * Event", "验证自适应融合，但出现 Macro 单流塌缩"],
        ],
        [1700, 3100, 2900, 1660],
    )

    doc.add_heading("4. 实验协议", level=1)
    add_para(
        doc,
        "为避免官方 4-fold 在部分模型中造成概率尺度不一致和 fold 分布问题，当前主实验采用 subject-level 60/20/20 固定划分。训练集只用于模型拟合，验证集用于 early stopping、超参数选择和阈值选择，测试集只用于最终评估。所有主结果均报告 validation-selected best balanced accuracy threshold 下的 test 指标。",
    )
    add_table(
        doc,
        ["项目", "当前设置"],
        [
            ["数据划分", "subject-level 60/20/20，train 96，valid 32，test 32，HC/SZ 均衡"],
            ["核心输入", "x/y/timestamp/validity/采样率/屏幕参数/观看距离推导出的公共眼动特征"],
            ["不使用信息", "图片内容、视频语义、AOI、IMAGE 语义、任务答案、瞳孔核心特征"],
            ["深度模型默认结构", "projection_dim=64，hidden_dim=64，attention_dim=64，dropout=0.3"],
            ["优化器", "AdamW，learning_rate=1e-3，weight_decay=1e-4"],
            ["训练控制", "max_epochs=100，patience=15，valid AUC early stopping"],
            ["分类阈值", "在 validation set 选择，test set 只评估一次"],
        ],
        [2600, 6760],
    )

    doc.add_heading("5. 固定划分实验结果", level=1)
    add_para(
        doc,
        "当前固定划分结果表明，传统 HistGradientBoosting 仍是强 baseline；Macro-Behavior Stream 与 Dual concat 在 AUC 上表现更好，但固定阈值下的 balanced accuracy 尚未全面超过 HistGradientBoosting。因此论文表述应保持克制：双流模型提升了风险排序能力，但分类阈值泛化仍需在更多数据集和更大样本中验证。",
    )
    add_table(
        doc,
        ["模型", "AUC", "Acc", "Bal Acc", "Sens", "Spec", "F1", "Confusion"],
        metric_rows_for_doc(),
        [1700, 780, 780, 850, 780, 780, 780, 2910],
    )
    add_bullets(
        doc,
        [
            f"Dual concat 当前 AUC 最高：{metrics['dual_stream_concat']['auc']}，说明 Event-Temporal Stream 对 Macro 表征存在一定互补排序信息。",
            f"Macro only AUC 为 {metrics['macro_behavior_stream']['auc']}，与 Dual concat 的固定阈值分类结果相同，说明 Macro 是当前最稳定主干。",
            f"Event only AUC 为 {metrics['event_temporal_stream']['auc']}，sensitivity 为 {metrics['event_temporal_stream']['sensitivity']}，提示事件动态流更偏向发现阳性风险，但 specificity 较低。",
            f"HistGradientBoosting balanced accuracy 为 {metrics['ml_hist_gradient_boosting']['balanced_accuracy']}，是目前固定阈值分类最强 baseline。",
        ],
    )

    doc.add_heading("6. 双流融合分析", level=1)
    add_para(
        doc,
        "双流实验的关键问题是 Event-Temporal Stream 是否真的补充 Macro-Behavior Stream。concat fusion 相比 Macro only 将 test AUC 从 0.891 提升至 0.898，但固定阈值下 accuracy、balanced accuracy、sensitivity 和 specificity 与 Macro only 相同。gated fusion 原本用于验证自适应权重分配能否改善 sensitivity/specificity 平衡，但当前结果显示其泛化明显变差。",
    )
    gate_test = next(row for row in gate_rows if row["split"] == "test")
    gate_valid = next(row for row in gate_rows if row["split"] == "valid")
    add_table(
        doc,
        ["融合方式", "Test AUC", "Bal Acc", "Sensitivity", "Specificity", "结论"],
        [
            [
                "Dual concat",
                metrics["dual_stream_concat"]["auc"],
                metrics["dual_stream_concat"]["balanced_accuracy"],
                metrics["dual_stream_concat"]["sensitivity"],
                metrics["dual_stream_concat"]["specificity"],
                "AUC 最优，作为当前主双流模型",
            ],
            [
                "Dual gated",
                metrics["dual_stream_gated"]["auc"],
                metrics["dual_stream_gated"]["balanced_accuracy"],
                metrics["dual_stream_gated"]["sensitivity"],
                metrics["dual_stream_gated"]["specificity"],
                "出现单流塌缩，作为补充/负结果",
            ],
        ],
        [1700, 900, 900, 1000, 1000, 3860],
    )
    add_para(
        doc,
        f"门控权重分析显示，gated fusion 在 test set 的 macro_gate 平均值为 {float(gate_test['macro_gate_mean']):.3f}，event_gate 平均值为 {float(gate_test['event_gate_mean']):.3f}；在 valid set 的 macro_gate 平均值为 {float(gate_valid['macro_gate_mean']):.3f}，event_gate 平均值为 {float(gate_valid['event_gate_mean']):.3f}。test set 中 93.75% 的受试者 macro_gate 超过 0.90，说明门控几乎完全压制 Event 流，没有形成真正的自适应双流融合。",
    )

    doc.add_heading("7. 阶段性结论", level=1)
    add_numbered(
        doc,
        [
            "内容解耦路线成立：当前所有主模型均不依赖图片或视频语义，只使用公共眼动行为特征。",
            "Macro-Behavior Stream 是当前最稳定的深度学习主干，适合作为后续跨数据集预训练的核心 encoder。",
            "Event-Temporal Stream 单独性能较弱，但对 Dual concat 的 AUC 有增益，说明其具有辅助价值。",
            "Dual concat 是当前主双流方案；gated fusion 在当前小样本条件下出现单流塌缩，不作为最终主模型。",
            "传统 ML baseline 仍然非常强，后续论文不能简单声称深度学习全面优于机器学习，而应强调序列建模、可解释性和跨数据集扩展潜力。",
        ],
    )

    doc.add_heading("8. 下一阶段计划：跨数据集预训练", level=1)
    add_para(
        doc,
        "下一阶段将从 EMS 单数据集实验转向跨数据集建模。计划优先接入 GazeBase、CRCNS eye-1、HBN、Saliency4ASD 等公开数据源，并根据各数据集是否包含疾病标签、年龄段、观看范式和原始采样质量，分别用于自监督预训练、正常人眼动动态建模、辅助表型学习或异常注意模式辅助任务。",
    )
    add_table(
        doc,
        ["数据集", "预期作用", "训练任务", "注意事项"],
        [
            ["GazeBase", "大规模正常人眼动 encoder 预训练", "自监督/对比学习/重建任务", "无精神疾病标签，不能直接作为 SZ 分类监督"],
            ["CRCNS eye-1", "补充自然视频观看域", "视频观看下的眼动动态自监督", "需统一采样率、坐标、validity 和观看距离"],
            ["HBN", "儿童/青少年自然视频眼动辅助学习", "自监督或表型辅助任务", "年龄段贴近应用，但标签体系复杂"],
            ["Saliency4ASD", "异常视觉注意模式辅助任务", "ASD/TD 辅助分类或表征学习", "不能直接并入 SZ 标签，只能做辅助迁移"],
        ],
        [1500, 2700, 2600, 2560],
    )

    doc.add_heading("参考文献草稿", level=1)
    references = [
        "Song et al., 2024. EMS: A Large-Scale Eye Movement Dataset, Benchmark and New Model for Schizophrenia Recognition. IEEE Transactions on Neural Networks and Learning Systems.",
        "Liu & Deubel, 2018. An elaborate algorithm for automatic processing of eye movement data and identifying fixations in eye-tracking experiments. Biomedical Engineering Online.",
        "Niehorster et al., 2020. The accuracy and precision of position and velocity in eye tracking / preprocessing evaluation. Behavior Research Methods.",
        "Holmqvist et al., 2011. Eye Tracking: A Comprehensive Guide to Methods and Measures. Oxford University Press.",
        "Bengio, Courville & Vincent, 2013. Representation learning: A review and new perspectives. IEEE TPAMI.",
        "Srivastava et al., 2014. Dropout: A simple way to prevent neural networks from overfitting. JMLR.",
        "Kornblith et al., 2019. Do Better ImageNet Models Transfer Better? CVPR.",
        "Wang et al., 2020. Graph Convolutional Networks for gaze relation learning / eye movement analysis. Pattern Recognition Letters.",
        "Glaholt et al., 2019. Deep learning analysis of eye movement dynamics in schizophrenia. Schizophrenia Bulletin.",
    ]
    add_bullets(doc, references)

    doc.save(OUTPUT)


if __name__ == "__main__":
    build_doc()
    print(OUTPUT)
