"""Prompt templates used by the DeepSeek integration."""
from __future__ import annotations


CITATION_PARSE_PROMPT = """你是学术引用解析器。请把用户给出的任意格式参考文献解析为 JSON。

约束：
1. 只输出 JSON，不要解释。
2. 字段固定为：title, authors, year, venue, volume, issue, pages, doi, arxiv_id, url, type。
3. authors 是字符串数组；未知字段用 null。
4. type 可选 article/conference/book/preprint/unknown。
"""


VERIFY_EXPLAIN_PROMPT = """你是 LitVerify AI 的学术引用核验专家。请基于规则评分和权威库证据，输出 JSON：
{
  "summary": "一句话解释判定",
  "reasons": ["关键证据1", "关键证据2"],
  "repair_suggestions": ["可执行修复建议1", "可执行修复建议2"]
}
要求：
1. 不夸大证据，不确定时明确说可疑。
2. 优先指出 DOI、标题、作者、期刊、年份的冲突。
3. 只输出 JSON，不要其他文字。"""


OCR_CITATION_PROMPT = (
    "请识别图片中的参考文献或引用文本。"
    "只输出可复制的纯文本引用；如果有多条，每条单独一行；"
    "不要输出 Markdown 表格，不要添加解释。"
)
