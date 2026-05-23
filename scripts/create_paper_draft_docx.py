from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT = Path("docs/paper_drafts/content_agnostic_eye_movement_screening_draft.docx")


EMS_MULTI_SEED_ROWS = [
    ["From scratch", "0.784 +/- 0.069", "0.700 +/- 0.114", "0.800", "0.600", "Supervised encoder baseline."],
    ["EMS MEM frozen", "0.716 +/- 0.059", "0.619 +/- 0.081", "0.813", "0.425", "Pretrained encoder alone is insufficient."],
    ["EMS MEM fine-tune", "0.832 +/- 0.065", "0.763 +/- 0.052", "0.800", "0.725", "Best multi-seed encoder route so far."],
]


PUBLIC_PRETRAIN_ROWS = [
    ["None", "from scratch", "0.828", "0.719", "0.813", "0.625", "Seed42 supervised encoder baseline."],
    ["EMS-only MEM", "fine-tune", "0.891", "0.750", "0.938", "0.563", "Highest seed42 AUC."],
    ["EMS-only MEM", "frozen", "0.848", "0.688", "0.938", "0.438", "Frozen probing underperforms fine-tuning."],
    ["HBN+EMS MEM", "fine-tune", "0.859", "0.719", "0.875", "0.563", "Pipeline works, but no gain over EMS-only."],
    ["HBN+EMS MEM", "frozen", "0.824", "0.719", "0.938", "0.500", "Some ranking transfer, weak specificity."],
    ["GazeBase+EMS MEM", "fine-tune", "0.863", "0.813", "0.750", "0.875", "More conservative, high specificity."],
    ["GazeBase+EMS MEM", "frozen", "0.863", "0.688", "0.938", "0.438", "Ranking transfers; threshold boundary needs adaptation."],
]


DATASET_ROWS = [
    ["EMS", "Complete", "160", "225,159", "Main SZ/HC downstream benchmark."],
    ["HBN", "Adapter complete", "1,244 usable", "1,684,382 after QC", "Public unlabeled pretraining source; did not improve over EMS-only in seed42."],
    ["GazeBase", "Adapter complete", "322", "843,517", "Video tasks VD1/VD2; high-specificity downstream operating point."],
    ["Saliency4ASD", "Pending", "TBD", "TBD", "Use cautiously; ASD labels must not be merged with SZ/HC."],
    ["CRCNS eye-1", "Pending", "TBD", "TBD", "Local raw-file status must be verified."],
]


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


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110) -> None:
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
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.82)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.2)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(5)

    title = styles["Title"]
    title.font.name = "Calibri"
    title._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    title.font.size = Pt(20)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0x0B, 0x25, 0x45)
    title.paragraph_format.space_after = Pt(8)

    for style_name, size, color, before, after in [
        ("Heading 1", 15, RGBColor(0x1F, 0x4D, 0x78), 12, 5),
        ("Heading 2", 12.5, RGBColor(0x2E, 0x74, 0xB5), 8, 3),
        ("Heading 3", 11.5, RGBColor(0x1F, 0x4D, 0x78), 6, 2),
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
    run = paragraph.add_run("EyeNet content-agnostic screening model | working draft | 2026-05-23")
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
                run.font.size = Pt(8.2)
    for row in rows:
        cells = table.add_row().cells
        for idx, text in enumerate(row):
            cells[idx].text = text
            for paragraph in cells[idx].paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                if idx > 0 and len(text) < 16:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(8.1)
    set_table_geometry(table, widths)
    doc.add_paragraph()


def add_title_block(doc: Document) -> None:
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("基于内容解耦眼动表征的精神疾病初步筛查模型研究")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("EMS 下游验证、公共数据自监督预训练与工程化阶段总结 | 2026-05-23")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def build_doc() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    set_doc_defaults(doc)
    add_footer(doc)
    add_title_block(doc)

    doc.add_heading("摘要草稿", level=1)
    add_para(
        doc,
        "本研究面向精神疾病风险的低负担初步筛查，构建一种不依赖图片、视频语义或任务内容的眼动行为建模流程。当前工程已完成 EMS 下游 SZ/HC 基准、HBN 与 GazeBase 公共眼动数据适配、统一 fixation-event schema、13 维 no-position encoder-ready 表征，以及 masked event modeling (MEM) 自监督预训练。五个 EMS stratified subject split 的结果显示，EMS-only MEM 预训练后进行监督 fine-tuning，相比从零训练 encoder，将平均 test AUC 从 0.784 提升到 0.832，将平均 balanced accuracy 从 0.700 提升到 0.763。HBN 已完成 pipeline 接入，但在当前固定 split 上未超过 EMS-only MEM；GazeBase+EMS 在单 split 上提供更高 specificity 的保守操作点。当前结果支持内容解耦眼动表征路线，但仍属于小样本初步证据，后续需要完成 Saliency4ASD/CRCNS 数据源筛选并进行受控消融。",
    )

    doc.add_heading("1. 研究目标与边界", level=1)
    add_para(
        doc,
        "本项目的目标不是训练一个只适用于 EMS 图片范式的分类器，而是建立一套可迁移的眼动表征学习流程。模型输入限制在跨设备、跨范式可获得的公共眼动信息，例如 fixation 坐标、duration、transition/saccade 特征、局部事件顺序和 missingness 标记。图片编号、视频内容、AOI 语义、任务答案和疾病无关标签不作为核心 encoder 输入。",
    )
    add_bullets(
        doc,
        [
            "临床定位：初步筛查和风险排序，不替代临床诊断。",
            "监督边界：EMS 的 SZ/HC 标签是当前下游基准；ASD、HBN 表型或其他疾病标签不能合并成一个通用疾病标签。",
            "预训练策略：公共数据优先用于自监督 MEM，不直接定义 SZ/HC 决策边界。",
            "评估原则：模型选择依赖 validation；test 只用于最终报告；小样本结果必须报告 multi-seed mean/std。",
        ],
    )

    doc.add_heading("2. 数据处理与统一输入", level=1)
    add_para(
        doc,
        "当前统一输入单位是 fixation event。HBN 和 GazeBase 这类 raw gaze 数据只作为 adapter 输入，必须先经 I-DT fixation detection 转成共享事件表，再进入 QC、encoder-ready 转换和 MEM 训练。GazeBase 的原始 x/y 已经是 DVA；HBN 是像素坐标。两者原始格式不同，但最终都统一到同一套 fixation-event schema 和 13 维 encoder feature schema。",
    )
    add_table(
        doc,
        ["Dataset", "Status", "Subjects", "Fixation events", "Notes"],
        DATASET_ROWS,
        [1350, 1500, 1050, 1450, 4010],
    )

    doc.add_heading("3. 当前 encoder 和 MEM 配置", level=1)
    add_para(
        doc,
        "当前 encoder 是保守的小模型基线：13 维事件特征先投影到 64 维，再经 1-layer bidirectional GRU 和 masked attention pooling 得到 subject embedding。MEM 预训练在 masked fixation spans 上重建原始标准化事件特征。该设置的目的是先建立稳定、可解释、可复现实验基线，而不是一次性追求最大模型容量。",
    )
    add_table(
        doc,
        ["Component", "Current setting"],
        [
            ["Input", "[batch, sequence_length, 13]"],
            ["Projection", "Linear(13 -> 64) + LayerNorm + ReLU + Dropout(0.3)"],
            ["Temporal encoder", "1-layer BiGRU, hidden_dim=64, event embedding=128"],
            ["Pooling", "Masked attention pooling"],
            ["MEM objective", "Masked MSE reconstruction on span-masked fixation features"],
            ["Optimizer", "AdamW, learning_rate=1e-3, weight_decay=1e-4"],
            ["Training defaults", "batch_size=8, max_seq_len=1500, mask_probability=0.30, span length 2-8"],
        ],
        [2500, 6860],
    )

    doc.add_heading("4. EMS 多 seed 基准结果", level=1)
    add_para(
        doc,
        "为避免单个 32-subject test split 的偶然性，当前已完成五个 EMS stratified subject splits 的 encoder 对照。主要结论是：EMS-only MEM 作为初始化再进行下游 supervised fine-tuning，优于从零训练；冻结 encoder 的 probing 表现较差，说明自监督表征需要任务标签适配才能转化为 SZ/HC 判别性能。",
    )
    add_table(
        doc,
        ["Experiment", "AUC", "Bal Acc", "Sens", "Spec", "Interpretation"],
        EMS_MULTI_SEED_ROWS,
        [1650, 1250, 1350, 900, 900, 3310],
    )

    doc.add_heading("5. 公共数据预训练初步结果", level=1)
    add_para(
        doc,
        "HBN 与 GazeBase 已经完成同一套处理流程：raw gaze -> fixation events -> schema validation -> QC -> encoder-ready -> MEM -> EMS downstream。当前固定 split 结果显示，公共数据不是越多越好；是否有益取决于分布匹配和下游操作点。HBN+EMS 没有超过 EMS-only；GazeBase+EMS 的 AUC 低于 EMS-only seed42，但 specificity 和 balanced accuracy 更高，可能形成更保守的筛查阈值候选。",
    )
    add_table(
        doc,
        ["Pretraining", "Mode", "AUC", "Bal Acc", "Sens", "Spec", "Notes"],
        PUBLIC_PRETRAIN_ROWS,
        [1450, 1000, 700, 850, 700, 700, 3960],
    )

    doc.add_heading("6. 阶段性判断", level=1)
    add_numbered(
        doc,
        [
            "内容解耦 fixation-event pipeline 已经成立，EMS、HBN、GazeBase 可以进入同一 encoder-ready schema。",
            "当前最稳的 encoder 证据来自 EMS-only MEM fine-tune，多 seed 下平均 AUC 和 balanced accuracy 均优于 from scratch。",
            "HBN 技术链路已完成，但目前不作为优先优化路线。",
            "GazeBase 值得保留，因为它提供了与 EMS-only 不同的错误模式：更高 specificity、更低 sensitivity。",
            "正式超参数消融应推迟到 Saliency4ASD 和 CRCNS 至少完成基础数据源筛选之后。",
        ],
    )

    doc.add_heading("7. 下一阶段计划", level=1)
    add_para(
        doc,
        "下一阶段不应直接进入大规模刷分，而应先完成剩余数据源筛选，再选择一到两条最有希望的预训练路线做受控消融。建议顺序如下：",
    )
    add_numbered(
        doc,
        [
            "接入 Saliency4ASD：确认文件结构，解析 fixation/scanpath，统一到 EyeNet fixation-event schema。ASD 标签只可作为辅助任务候选，不能与 EMS SZ/HC 混合。",
            "核查 CRCNS eye-1：确认本地数据是否完整，再决定是否实现 adapter。",
            "对每个新数据源只使用固定 baseline 超参数，不做调参，先跑 source+EMS MEM fine-tune/frozen。",
            "完成数据源筛选后再做小范围消融：hidden_dim 32/64/128，dropout 0.1/0.3/0.5，max_seq_len 1000/1500/2500。",
            "最终模型选择必须基于 multi-seed mean/std，而不是某一个 split 的最高 AUC。",
        ],
    )

    doc.add_heading("8. 当前论文表述限制", level=1)
    add_bullets(
        doc,
        [
            "不能声称模型已经达到临床诊断水平；当前只支持初步筛查和风险排序表述。",
            "不能声称公共数据必然提升；HBN 已经显示公共数据可能无增益。",
            "不能只报告最高单 split；EMS 样本小，必须报告 multi-seed 统计。",
            "不能混合不同疾病标签形成统一二分类标签。",
            "不能把 task_id、video_id、image_id 或刺激语义作为 universal encoder 输入。",
        ],
    )

    doc.add_heading("参考文献草稿", level=1)
    add_bullets(
        doc,
        [
            "Song et al., 2024. EMS: A Large-Scale Eye Movement Dataset, Benchmark and New Model for Schizophrenia Recognition. IEEE Transactions on Neural Networks and Learning Systems.",
            "Holmqvist et al., 2011. Eye Tracking: A Comprehensive Guide to Methods and Measures. Oxford University Press.",
            "Liu & Deubel, 2018. An elaborate algorithm for automatic processing of eye movement data and identifying fixations in eye-tracking experiments. Biomedical Engineering Online.",
            "Bengio, Courville & Vincent, 2013. Representation learning: A review and new perspectives. IEEE TPAMI.",
            "Srivastava et al., 2014. Dropout: A simple way to prevent neural networks from overfitting. JMLR.",
            "Kornblith et al., 2019. Do Better ImageNet Models Transfer Better? CVPR.",
        ],
    )

    doc.save(OUTPUT)


if __name__ == "__main__":
    build_doc()
    print(OUTPUT)
