"""HTML and PDF report exporting for LitVerify AI."""
from __future__ import annotations

import io
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import pandas as pd
from jinja2 import Template

from utils.dataframe import strip_report_json


REPORT_TEMPLATE = Template(
    """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{{ title }}</title>
  <style>
    body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 32px; color: #172033; }
    h1, h2 { color: #1f5f75; }
    .meta { color: #607089; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card { border: 1px solid #d9e2ef; border-radius: 8px; padding: 14px; background: #f7fbfc; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d9e2ef; padding: 7px 9px; text-align: left; vertical-align: top; }
    th { background: #eef6f7; }
    .finding { border-left: 4px solid #1f8a70; padding-left: 12px; margin: 8px 0; }
    .real { color: #18864b; font-weight: 700; }
    .suspicious { color: #b26b00; font-weight: 700; }
    .fake { color: #b42318; font-weight: 700; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <p class="meta">生成时间：{{ created_at }} · 样本数：{{ total }}</p>

  <div class="grid">
    <div class="card"><strong>总数</strong><br>{{ total }}</div>
    <div class="card"><strong>真实</strong><br><span class="real">{{ real }}</span></div>
    <div class="card"><strong>可疑</strong><br><span class="suspicious">{{ suspicious }}</span></div>
    <div class="card"><strong>虚假</strong><br><span class="fake">{{ fake }}</span></div>
  </div>

  <h2>验证结论</h2>
  {% for item in findings %}
    <div class="finding">{{ item }}</div>
  {% endfor %}

  <h2>规则体系</h2>
  <p>LitVerify AI 使用 DOI 格式、DOI 解析、标题匹配、作者重合、来源匹配、年份合理性、卷期页一致性、arXiv 解析、跨库一致性，以及作者格式 / 期刊名称 / 标题质量等本地元数据规则进行加权评分。</p>

  <h2>明细记录</h2>
  {{ table_html }}
</body>
</html>
"""
)


def _counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["verdict"].value_counts().to_dict() if "verdict" in df.columns else {}
    return {
        "real": int(counts.get("REAL", 0)),
        "suspicious": int(counts.get("SUSPICIOUS", 0)),
        "fake": int(counts.get("FAKE", 0)),
    }


def _default_findings(total: int, counts: dict[str, int]) -> list[str]:
    return [
        (
            f"共核验 {total} 条引用，其中真实 {counts['real']} 条、"
            f"可疑 {counts['suspicious']} 条、虚假 {counts['fake']} 条。"
        ),
        "报告保留每条引用的判定、得分、DOI 和主要证据，便于答辩复现。",
    ]


def build_verification_report_html(
    df: pd.DataFrame,
    title: str = "LitVerify AI 学术引用验证报告",
    findings: list[str] | None = None,
) -> str:
    """Build a standalone HTML verification report."""
    safe_df = strip_report_json(df)
    counts = _counts(safe_df)
    total = int(len(safe_df))
    return REPORT_TEMPLATE.render(
        title=title,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total=total,
        **counts,
        findings=findings or _default_findings(total, counts),
        table_html=safe_df.to_html(index=False, escape=True),
    )


def build_verification_report_pdf(
    df: pd.DataFrame,
    title: str = "LitVerify AI 学术引用验证报告",
    findings: list[str] | None = None,
) -> bytes:
    """Build a downloadable PDF verification report with ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    safe_df = strip_report_json(df)
    counts = _counts(safe_df)
    total = int(len(safe_df))
    rendered_findings = findings or _default_findings(total, counts)
    font_name = _register_pdf_font()

    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = font_name
    title_style = ParagraphStyle(
        "DWTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor("#1f5f75"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    meta_style = ParagraphStyle(
        "DWMeta",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#607089"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    normal_style = ParagraphStyle(
        "DWNormal",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#172033"),
    )
    heading_style = ParagraphStyle(
        "DWHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1f5f75"),
        spaceBefore=8,
        spaceAfter=6,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=title,
    )

    story: list = [
        Paragraph(title, title_style),
        Paragraph(
            f"生成时间：{datetime.now():%Y-%m-%d %H:%M} · 样本数：{total}",
            meta_style,
        ),
    ]
    story.extend(_summary_table(total, counts, font_name))
    story.append(Paragraph("验证结论", heading_style))
    for item in rendered_findings:
        story.append(Paragraph(f"• {html_escape(str(item))}", normal_style))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("明细记录", heading_style))
    story.append(_detail_table(safe_df, normal_style, doc.width, font_name))

    doc.build(story)
    return buffer.getvalue()


def _register_pdf_font() -> str:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:  # noqa: BLE001
        return "Helvetica"

    candidates = [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("DWChinese", str(path)))
            return "DWChinese"
        except Exception:  # noqa: BLE001
            continue
    return "Helvetica"


def _summary_table(total: int, counts: dict[str, int], font_name: str) -> list:
    from reportlab.lib import colors
    from reportlab.platypus import Spacer, Table, TableStyle

    data = [
        ["总数", "真实", "可疑", "虚假"],
        [str(total), str(counts["real"]), str(counts["suspicious"]), str(counts["fake"])],
    ]
    table = Table(data, colWidths=[70, 70, 70, 70])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef6f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f5f75")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f7fbfc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ef")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return [table, Spacer(1, 6)]


def _detail_table(df: pd.DataFrame, style, available_width: float, font_name: str):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    table_df = _display_columns(df)
    if table_df.empty:
        return Paragraph("暂无明细记录。", style)

    columns = list(table_df.columns)
    widths = _column_widths(columns, available_width)
    data = [[Paragraph(html_escape(str(col)), style) for col in columns]]
    for _, row in table_df.iterrows():
        data.append([Paragraph(_pdf_cell(row.get(col)), style) for col in columns])

    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef6f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f5f75")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e2ef")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _display_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "citation", "citation_text", "raw_citation", "title",
        "matched_title", "doi", "matched_doi", "year",
        "venue", "verdict", "score", "reasons", "suggestions",
    ]
    selected = [col for col in preferred if col in df.columns]
    for col in df.columns:
        if col not in selected and len(selected) < 8:
            selected.append(col)
    selected = selected[:9]
    return df.loc[:, selected] if selected else df


def _column_widths(columns: list[str], available_width: float) -> list[float]:
    if not columns:
        return []
    weights = []
    for col in columns:
        name = col.lower()
        if "citation" in name or "title" in name or "reason" in name or "suggestion" in name:
            weights.append(2.2)
        elif "score" in name or "year" in name or "verdict" in name:
            weights.append(0.85)
        else:
            weights.append(1.2)
    total = sum(weights)
    return [available_width * weight / total for weight in weights]


def _pdf_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).replace("\n", " ").strip()
    if len(text) > 260:
        text = f"{text[:257]}..."
    return html_escape(text)
