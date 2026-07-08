"""Agent router — single entry point that turns a (mode, text, files)
triple into a structured assistant message payload.

Each return dict has the same shape so ``ui.chat_shell.render_message``
can replay it after any Streamlit rerun::

    {
        "role": "assistant",
        "kind":  "verify_single" | "verify_batch" | "fake_analysis"
               | "chart" | "export" | "chat" | "error",
        "text":  "...markdown summary...",
        "data":  {...kind-specific serialisable payload...},
        "mode":  the inferred tool,
    }
"""
from __future__ import annotations

import base64
import io
import json
import re
from datetime import datetime
from typing import Any, Iterator

import pandas as pd

from config.prompts import (
    BATCH_NARRATE_PROMPT,
    CHART_CODE_PROMPT,
    CHAT_SYSTEM_PROMPT,
    FAKE_NARRATE_PROMPT,
)
from config.settings import settings
from db.history import list_history, history_summary
from llm.deepseek_client import DeepSeekClient, runtime_api_key
from llm.text_utils import strip_fenced_block
from services.citation_verifier import CitationVerifier
from services.data_loader import load_dataframe
from services.data_processor import batch_verify, batch_verify_structured
from services.exporter import build_verification_report_html, build_verification_report_pdf
from services.fake_analyzer import build_fake_pattern_report
from services.rule_engine import VerificationReport
from utils.dataframe import df_to_json_safe_records


# Mode metadata — labels/icons shown on the user bubble + error messages.
# Note: there is no "ocr" tool — an image upload routes to ``_run_ocr`` which
# explains that screenshots must be converted to text first.
MODES: list[dict[str, str]] = [
    {"id": "chat",           "label": "智能问答", "icon": "💬"},
    {"id": "verify_single",  "label": "单条验证", "icon": "📋"},
    {"id": "verify_batch",   "label": "批量验证", "icon": "📊"},
    {"id": "fake_analysis",  "label": "虚假特征", "icon": "🔬"},
    {"id": "chart",          "label": "数据画图", "icon": "📈"},
    {"id": "export",         "label": "导出报告", "icon": "📄"},
]

MODES_BY_ID = {m["id"]: m for m in MODES}

# Shown when a DeepSeek-dependent feature is used without an API key. Single /
# batch verification stay usable because they fall back to local explanations.
_NO_KEY_HINT = (
    "**{feature}** 依赖 DeepSeek，需要先在右上角「⚙ 设置」里配置 API Key。\n\n"
    "在此之前，你仍可使用 **📋 单条验证** 和 **📊 批量验证**——它们用本地规则引擎 + "
    "CrossRef/OpenAlex 即可工作，无需 API Key。"
)


# --------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------- #
# File extensions that mean "a table of citations" → batch verification.
_TABLE_EXTS = (".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json")

# Text-only intents the LLM router is allowed to pick. Batch / OCR never come
# from text — they require an uploaded table / image and are decided by file
# type, which is unambiguous and needs no model call.
_TEXT_INTENTS = ("verify_single", "fake_analysis", "chart", "export", "chat")


def infer_mode(
    text: str,
    files: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    """Decide which tool to run for a (text, files) input.

    Files win because their type is unambiguous; only genuinely text-only
    inputs reach the LLM intent classifier.
    """
    tables = [f for f in files if _has_ext(f.get("name", ""), _TABLE_EXTS)]
    images = [f for f in files if _looks_like_image(f.get("name", ""))]
    if tables:
        return "verify_batch"
    if images:
        return "ocr"
    # A non-table, non-image attachment (e.g. .txt/.md holding one citation)
    # with no typed text → treat the file content as a single citation.
    if files and not (text or "").strip():
        return "verify_single"

    # An explicit charting ask (柱状图/统计图表/可视化…) is a high-precision,
    # deterministic signal — route it straight to the chart tool *before* the
    # LLM classifier, which otherwise gets distracted by words like 报告/分析 in
    # the same sentence and mislabels it as chat. Guarded against a pasted
    # citation that merely happens to contain a chart-ish word.
    if _looks_like_chart_request(text) and not _looks_like_reference(text):
        return "chart"

    if _is_obvious_chat(text):
        return "chat"

    intent = _classify_text_intent_llm(text, history)
    if intent not in _TEXT_INTENTS:
        intent = _classify_text_intent_heuristic(text)
    return intent


def _classify_text_intent_llm(text: str, history: list[dict[str, Any]]) -> str | None:
    """Ask DeepSeek for a single intent label. Returns None on any failure
    so the caller can fall back to heuristics."""
    msg = (text or "").strip()
    if not msg or not runtime_api_key():
        return None
    has_results = any(m.get("kind") in {"verify_batch", "verify_single"} for m in history)
    system = (
        "你是学术引用助手的意图分类器。只能输出下列标签之一，禁止任何解释或标点：\n"
        "verify_single —— 用户给出了一条具体的文献/引用条目（含标题、作者、DOI、年份、期刊等），希望核验其真伪。\n"
        "fake_analysis —— 用户想对已有的一批验证结果做「虚假特征/造假模式」的统计或聚类分析。\n"
        "chart —— 用户想把已有的验证结果画成图表/可视化（柱状图、饼图、折线、散点、分布图、趋势等）。\n"
        "export —— 用户想导出或下载验证报告（HTML/PDF）。\n"
        "chat —— 其它所有情况：提问、解释概念、对结果追问、闲聊、寒暄等。\n"
        f"当前对话{'已有' if has_results else '尚无'}验证结果。拿不准时优先选 chat。"
    )
    try:
        client = DeepSeekClient(timeout=15)
        # deepseek-v4 系列是推理模型：思考 token 也计入 max_tokens，且思考先于
        # 正文输出。预算给太小（此前是 8）会被思考全部吃光，content 恒为空、
        # 分类器永远失效。这里用 Flash（更快更便宜）并留足思考余量。
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": msg},
            ],
            model=settings.deepseek_flash_model,
            temperature=0.0,
            max_tokens=512,
            retries=0,
        )
    except Exception:  # noqa: BLE001 - any failure → heuristic fallback
        return None
    label = (raw or "").strip().lower()
    for intent in _TEXT_INTENTS:
        if intent in label:
            return intent
    return None


def _classify_text_intent_heuristic(text: str) -> str:
    """Keyword/pattern fallback used when the LLM router is unavailable."""
    from utils.doi_utils import extract_arxiv, extract_doi

    msg = (text or "").strip()
    if not msg:
        return "chat"
    low = msg.lower()
    if any(k in low for k in ("导出", "下载报告", "生成报告", "export", "下载 pdf", "下载pdf")):
        return "export"
    if any(k in msg for k in ("虚假特征", "造假模式", "虚假模式", "造假分析", "虚假分析", "模式分析")):
        return "fake_analysis"
    if _looks_like_chart_request(msg):
        return "chart"
    # A concrete citation usually carries a DOI / arXiv id, or reads like a
    # reference line (a 4-digit year + a quoted/《》 title) rather than a question.
    is_question = msg.endswith(("?", "？")) or any(
        low.startswith(q) for q in ("什么", "如何", "为什么", "怎么", "能不能", "可以", "how", "what", "why", "is ", "are ")
    )
    has_doi = bool(extract_doi(msg) or extract_arxiv(msg))
    looks_reference = bool(re.search(r"(18|19|20)\d{2}", msg)) and (
        '"' in msg or "《" in msg or "et al" in low or msg.count(",") >= 2
    )
    if has_doi or (looks_reference and not is_question):
        return "verify_single"
    return "chat"


# Words that signal "draw me a chart" — used both by the heuristic intent
# fallback and by ``_is_obvious_chat`` so a charting ask phrased as a question
# ("能不能画个饼图？") still reaches the chart tool instead of plain chat.
# Kept broad on purpose (recall matters more than precision here): a citation
# the user wants *verified* almost never contains these, and the chart
# short-circuit additionally guards with ``_looks_like_reference`` anyway.
_CHART_KEYWORDS = (
    # draw / make verbs
    "画图", "画个", "画一", "画张", "画成", "画出", "作图", "绘图", "绘制",
    "出图", "出张", "成图", "看图", "用图",
    # visualise
    "可视化", "可视", "visualize", "visualise",
    # chart-type nouns
    "图表", "统计图", "图形", "柱状图", "条形图", "饼图", "饼状", "折线",
    "散点", "直方图", "分布图", "趋势图", "占比图", "环形图",
    # english
    "chart", "plot", "histogram", "bar chart", "pie chart", "graph", "diagram",
)


def _looks_like_chart_request(text: str) -> bool:
    msg = (text or "").strip()
    if not msg:
        return False
    low = msg.lower()
    return any(k in low for k in _CHART_KEYWORDS)


def _looks_like_reference(text: str) -> bool:
    """True when the text reads like a concrete citation to verify (a 4-digit
    year plus reference-ish punctuation), as opposed to a request/question.

    Used to keep a pasted citation that merely *mentions* a chart word
    (e.g. a paper titled 《…图像识别…》) out of the chart tool."""
    msg = (text or "").strip()
    if not msg:
        return False
    low = msg.lower()
    if re.search(r"\b(10\.\d{4,9}/[-._;()/:a-z0-9]+|arxiv:\s*\d{4}\.\d{4,5})\b", low):
        return True
    return bool(re.search(r"(18|19|20)\d{2}", msg)) and (
        '"' in msg or "《" in msg or "et al" in low or msg.count(",") >= 2
    )


def _is_obvious_chat(text: str) -> bool:
    """Keep natural questions out of the tool router unless they are clearly citations."""
    msg = (text or "").strip()
    if not msg:
        return False
    low = msg.lower()
    # A charting ask is a tool request, not chat — let it fall through to the
    # classifier even when it ends with a question mark.
    if _looks_like_chart_request(msg):
        return False
    if _looks_like_reference(msg):
        return False
    if msg.endswith(("?", "？")):
        return True
    chat_starts = (
        "你好", "您好", "嗨", "哈喽", "介绍", "解释", "说明", "总结", "概括",
        "为什么", "怎么", "如何", "能不能", "可以", "帮我看看", "你觉得",
        "what", "why", "how", "can you", "could you", "please explain",
    )
    return any(low.startswith(prefix) for prefix in chat_starts)


def _has_ext(name: str, exts: tuple[str, ...]) -> bool:
    return (name or "").lower().endswith(exts)


def dispatch(
    mode: str,
    text: str,
    files: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the right service for ``mode``.

    ``files`` is a list of {"name": str, "bytes": bytes} — the chat shell
    materialises ``UploadedFile`` into bytes before storing them in
    session history so this function is pure.

    ``on_progress`` is an optional ``(done, total, payload) -> None``
    callback，目前只有批量验证会用它（在 UI 上画进度条——200 条要跑几分钟，
    没有进度反馈时用户会以为应用卡死）。
    """
    files = files or []
    history = history or []
    try:
        if mode == "verify_single":
            return _run_verify_single(text, files)
        if mode == "verify_batch":
            return _run_verify_batch(text, files, on_progress=on_progress)
        if mode == "fake_analysis":
            return _run_fake_analysis(text, files, history)
        if mode == "chart":
            return _run_chart(text, files, history)
        if mode == "ocr":
            return _run_ocr(text, files)
        if mode == "export":
            return _run_export(history)
        return _run_chat(text, history, files)
    except Exception as exc:  # noqa: BLE001
        return _error(f"运行 `{MODES_BY_ID.get(mode, {}).get('label', mode)}` 时出错：{exc}", mode)


# --------------------------------------------------------------------- #
# verify_single
# --------------------------------------------------------------------- #
def _run_verify_single(text: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw and files:
        raw = _read_text_file(files[0])
    if not raw:
        return _error("请在对话框输入或附带一条引用文本。", "verify_single")

    verifier = CitationVerifier()
    report = verifier.verify(raw, with_llm_explain=True, save=True)
    return {
        "role": "assistant",
        "kind": "verify_single",
        "mode": "verify_single",
        "text": _verdict_summary(report),
        "data": {"report": report.to_dict()},
    }


# --------------------------------------------------------------------- #
# verify_batch
# --------------------------------------------------------------------- #
def _run_verify_batch(
    text: str,
    files: list[dict[str, Any]],
    on_progress: Any = None,
) -> dict[str, Any]:
    if not files:
        return _error("请上传 CSV / TSV / Excel 引用清单后再点发送。", "verify_batch")

    target = files[0]
    try:
        df = load_dataframe(io.BytesIO(target["bytes"]), filename=target["name"])
    except Exception as exc:  # noqa: BLE001
        return _error(f"无法读取文件 `{target['name']}`：{exc}", "verify_batch")

    structured = _detect_structured_columns(df)
    if structured:
        result_df = batch_verify_structured(
            df, structured, max_workers=8, on_progress=on_progress
        )
        detected = "、".join(f"{role}=`{col}`" for role, col in structured.items())
        source_label = f"结构化列：{detected}"
    else:
        column = _pick_citation_column(df, hint=text)
        if column is None:
            return _error(
                f"未在 `{target['name']}` 中找到引用列。文件包含的列：{list(df.columns)}。"
                "请在对话框补充列名提示，例如：「列名 citation_text」。",
                "verify_batch",
            )
        result_df = batch_verify(df, citation_col=column, max_workers=8, on_progress=on_progress)
        source_label = f"列：`{column}`"

    counts = result_df["verdict"].value_counts().to_dict() if "verdict" in result_df.columns else {}
    fallback = (
        f"已验证 **{len(result_df)}** 条引用（{source_label}）。\n\n"
        f"- ✅ 真实 {int(counts.get('REAL', 0))}\n"
        f"- ⚠️ 可疑 {int(counts.get('SUSPICIOUS', 0))}\n"
        f"- ❌ 虚假 {int(counts.get('FAKE', 0))}\n"
        f"- 🛑 错误 {int(counts.get('ERROR', 0))}"
    )
    # Let DeepSeek read the real comparison results and report them in natural
    # language; fall back to the deterministic count summary if AI is off.
    summary = _narrate_batch_results(result_df, counts, source_label, text) or fallback

    return {
        "role": "assistant",
        "kind": "verify_batch",
        "mode": "verify_batch",
        "text": summary,
        "data": {
            "rows": df_to_json_safe_records(result_df),
            "columns": list(result_df.columns),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "filename": target["name"],
            "citation_col": (structured.get("title") if structured else column),
            "structured_columns": structured or None,
        },
    }


# Column names (中/英) that may carry a human-readable identifier / problem
# description for a verified row — checked in order, first non-empty wins.
_NAME_COLS = ("命中标题", "matched_title", "title", "标题", "题名", "题目", "citation", "引用")
_PROBLEM_COLS = ("虚假特征", "reasons", "suggestions")


def _narrate_batch_results(
    result_df,
    counts: dict[str, Any],
    source_label: str,
    user_text: str,
) -> str | None:
    """Ask DeepSeek to report the real comparison results in natural language.

    Returns None when AI is unavailable so the caller keeps the deterministic
    count summary. Only aggregate stats + the problematic rows are sent, so the
    payload stays small even for a few-hundred-row batch.
    """
    if not runtime_api_key():
        return None

    name_col = next((c for c in _NAME_COLS if c in result_df.columns), None)
    problem_col = next((c for c in _PROBLEM_COLS if c in result_df.columns), None)

    problems: list[dict[str, Any]] = []
    if "verdict" in result_df.columns:
        flagged = result_df[result_df["verdict"].isin(["FAKE", "SUSPICIOUS", "ERROR"])]
        for pos, (_, row) in enumerate(flagged.iterrows()):
            if pos >= 20:  # cap payload — narrate the rest as "等若干条"
                break
            name = str(row.get(name_col, "") or "").strip() if name_col else ""
            problem = str(row.get(problem_col, "") or "").strip() if problem_col else ""
            problems.append(
                {
                    "条目": name[:120] or "（无标题）",
                    "判定": row.get("verdict"),
                    "分数": row.get("score"),
                    "问题": problem[:200],
                }
            )

    # 分组统计（按模型/领域/主题的虚假率 + 每类特征条数），让"统计各学科
    # 虚假率"这类随文件附带的问题有数据可答，而不是被固定结构的总结盖掉。
    try:
        pattern_report = build_fake_pattern_report(result_df)
        group_stats: dict[str, Any] | None = {
            "分组虚假率": pattern_report.get("groups"),
            "每类特征条数": pattern_report.get("pattern_counts"),
        }
    except Exception:  # noqa: BLE001 - grouping is best-effort grounding
        group_stats = None

    payload = {
        "来源": source_label,
        "总数": int(len(result_df)),
        "统计": {str(k): int(v) for k, v in counts.items()},
        "用户说明": (user_text or "").strip() or None,
        "分组统计": group_stats,
        "问题条目": problems,
        "问题条目是否截断": bool(
            "verdict" in result_df.columns
            and len(result_df[result_df["verdict"].isin(["FAKE", "SUSPICIOUS", "ERROR"])]) > 20
        ),
    }
    try:
        # 90s：思考型模型读完 200 条的分组统计再写总结，30s 在云端经常超时，
        # 一超时就静默回退成呆板的计数模板，用户的问题又没人答了。
        client = DeepSeekClient(timeout=90)
        return client.chat(
            messages=[
                {"role": "system", "content": BATCH_NARRATE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=0.45,
            top_p=0.9,
            # 推理模型的思考也占 max_tokens，预算需覆盖 思考+正文
            max_tokens=3000,
            retries=0,
        ).strip() or None
    except Exception:  # noqa: BLE001 - any failure → deterministic fallback
        return None


# --------------------------------------------------------------------- #
# fake_analysis
# --------------------------------------------------------------------- #
def _run_fake_analysis(
    text: str,
    files: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run fake-pattern analysis on either an uploaded result CSV or the
    most recent batch result still in chat history. Falls back to the
    persistent SQLite history if neither is present.

    回复文案不再是写死的模板：带着用户的原话让 DeepSeek 针对问题作答
    （此前不管问"特征条数"还是"学科分布"都返回同一份固定摘要）。AI 不可用
    时回退到确定性模板，模板里也带上每类特征的条数。"""
    df = _df_from_files_or_history(files, history)
    if df is None or df.empty:
        return _error(
            "需要一个含 `verdict` / `score` / `reasons` 列的结果数据。"
            "先做一次「批量验证」或上传上次的结果 CSV 即可。",
            "fake_analysis",
        )

    stats = build_fake_pattern_report(df)
    summary = _narrate_fake_analysis(stats, text) or _fake_analysis_fallback(stats)
    return {
        "role": "assistant",
        "kind": "fake_analysis",
        "mode": "fake_analysis",
        "text": summary,
        "data": {
            "stats": stats,
            "rows": df_to_json_safe_records(df.head(200)),
        },
    }


def _narrate_fake_analysis(stats: dict[str, Any], user_text: str) -> str | None:
    """Ask DeepSeek to answer the user's actual question over the stats.

    Returns None when AI is unavailable/fails so the caller falls back to the
    deterministic template."""
    if not runtime_api_key():
        return None
    payload = {
        "用户问题": (user_text or "").strip() or "总结这批结果的虚假特征规律。",
        "统计数据": stats,
    }
    try:
        client = DeepSeekClient(timeout=90)
        return client.chat(
            messages=[
                {"role": "system", "content": FAKE_NARRATE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=0.4,
            top_p=0.9,
            max_tokens=3000,
            retries=0,
        ).strip() or None
    except Exception:  # noqa: BLE001 - any failure → deterministic fallback
        return None


def _fake_analysis_fallback(stats: dict[str, Any]) -> str:
    """Deterministic summary used when AI is off — now with per-feature counts."""
    pattern_counts = stats.get("pattern_counts") or []
    if pattern_counts:
        bullets = "\n".join(
            f"- {p['pattern']}：{p['count']} 条" for p in pattern_counts[:8]
        )
    else:
        bullets = "- 暂未发现强模式特征。"
    summary = (
        f"对 **{stats['total']}** 条引用做了虚假特征聚类，"
        f"高风险（可疑/虚假）占比 **{stats['fake_like_ratio']:.0%}**。\n\n"
        f"**各类虚假特征命中条数：**\n{bullets}"
    )
    group_md = _format_group_rates(stats.get("groups") or {})
    if group_md:
        summary += f"\n\n{group_md}"
    return summary


# --------------------------------------------------------------------- #
# chart — natural-language → DeepSeek-generated Python → sandboxed exec
# --------------------------------------------------------------------- #
def _run_chart(
    text: str,
    files: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Turn a natural-language request into a chart over the latest result set.

    With an API key, DeepSeek writes real plotting code (any plotly chart
    type), which runs once here in a restricted namespace — no imports, no
    file/network access, safe-builtins only, bounded by a timeout. The
    resulting Figure is serialised to JSON so replays never re-execute the
    code. A failed run is retried once with the traceback; if that also
    fails (or AI is off) we fall back to the deterministic heuristic chart.
    """
    df = _df_from_files_or_history(files, history)
    if df is None or df.empty:
        return _error(
            "还没有可画图的数据。先做一次「批量验证」（上传 CSV/Excel）或上传一份结果表格，"
            "再让我画图就行。",
            "chart",
        )

    if runtime_api_key():
        generated = _chart_from_llm_code(text, df)
        if generated is not None:
            fig, code = generated
            title = _fig_title(fig) or "数据图表"
            return {
                "role": "assistant",
                "kind": "chart",
                "mode": "chart",
                "text": f"已按你的需求生成 **{title}**。",
                "data": {
                    "fig_json": fig.to_json(),
                    "code": code,
                    "rows": df_to_json_safe_records(df),
                },
            }

    spec = _heuristic_chart_spec(df)
    if spec is None:
        return _error("这份数据里没有适合画图的列（需要至少一个分类或数值列）。", "chart")

    title = spec.get("title") or "数据图表"
    chart_label = _CHART_TYPE_LABELS.get(spec["chart_type"], spec["chart_type"])
    summary = f"已生成 **{title}**（{chart_label}）。"
    if runtime_api_key():
        summary += "\n\n_（AI 代码生成未成功，已回退为默认图表；换个说法再试一次通常就好。）_"
    reason = (spec.get("reason") or "").strip()
    if reason:
        summary += f"\n\n_{reason}_"

    return {
        "role": "assistant",
        "kind": "chart",
        "mode": "chart",
        "text": summary,
        "data": {
            "spec": spec,
            "rows": df_to_json_safe_records(df),
        },
    }


def _fig_title(fig: Any) -> str | None:
    try:
        title = fig.layout.title.text
        return str(title).strip() or None
    except Exception:  # noqa: BLE001
        return None


_CHART_TYPE_LABELS = {
    "bar": "柱状图", "pie": "饼图", "line": "折线图",
    "histogram": "直方图", "scatter": "散点图", "box": "箱线图",
}


def _column_profile(df: pd.DataFrame, max_cols: int = 30, sample: int = 5) -> list[dict[str, Any]]:
    """Compact, model-friendly description of each column: name, kind, samples.

    Sent to DeepSeek so it can pick sensible x/y/color without ever seeing the
    full data. Numeric vs. categorical is surfaced because it drives chart-type
    selection (histogram/scatter want numbers; pie/bar want categories)."""
    profile: list[dict[str, Any]] = []
    for col in list(df.columns)[:max_cols]:
        series = df[col]
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        values = series.dropna().unique()[:sample]
        profile.append(
            {
                "列名": str(col),
                "类型": "数值" if is_numeric else "分类/文本",
                "唯一值数": int(series.nunique(dropna=True)),
                "示例": [str(v)[:40] for v in values],
            }
        )
    return profile


# How many generation rounds the model gets: the first attempt plus one
# repair attempt that sees the failed code + its error message.
_CHART_CODE_ATTEMPTS = 2
# Wall-clock budget for one exec of generated code. Generous for a groupby +
# plotly call; mostly a backstop against an accidental heavy computation.
_CHART_EXEC_TIMEOUT = 20.0

# Names the generated code may use. Everything else (open/__import__/eval/…)
# simply doesn't exist inside the exec namespace.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "len", "list", "map",
    "max", "min", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "ValueError", "TypeError",
    "KeyError", "Exception",
)

# Cheap static screen on top of the namespace sandbox: these tokens have no
# legitimate use in a df→plotly snippet, so reject early with a message the
# repair round can act on. ``while`` is banned as the easiest infinite-loop
# vector — the chart code never needs it.
_CHART_FORBIDDEN = re.compile(
    r"\b(import|open|exec|eval|compile|__import__|globals|locals|breakpoint"
    r"|input|while|getattr|setattr|delattr|vars|dir|type)\b"
)


def _chart_from_llm_code(text: str, df: pd.DataFrame) -> tuple[Any, str] | None:
    """Ask DeepSeek for plotting code and execute it in the sandbox.

    Returns ``(fig, code)`` on success, None when generation/exec failed after
    the repair attempt — the caller then falls back to the heuristic chart.
    """
    payload = {
        "可用列": _column_profile(df),
        "总行数": int(len(df)),
        "用户请求": (text or "").strip() or "请挑一个最能反映这批结果的图。",
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CHART_CODE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
    ]
    for _ in range(_CHART_CODE_ATTEMPTS):
        try:
            client = DeepSeekClient(timeout=45)
            # 思考型模型：1000 会被思考吃掉大半、代码被截断，预算放宽
            raw = client.chat(messages=messages, temperature=0.2, max_tokens=2600, retries=0)
        except Exception:  # noqa: BLE001 - model unavailable → heuristic fallback
            return None
        code = strip_fenced_block(raw, "python")
        try:
            fig = _exec_chart_code(code, df)
            return fig, code
        except Exception as exc:  # noqa: BLE001 - feed the error back for one repair
            messages.append({"role": "assistant", "content": code})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"这段代码执行失败：{type(exc).__name__}: {exc}\n"
                        "请修正后重新输出完整代码（仍然只输出代码本身）。"
                    ),
                }
            )
    return None


def _exec_chart_code(code: str, df: pd.DataFrame) -> Any:
    """Run generated plotting code in a restricted namespace; return its ``fig``.

    The sandbox is two layers: a static token screen (`_CHART_FORBIDDEN`) and
    an exec namespace whose ``__builtins__`` only contains `_SAFE_BUILTIN_NAMES`
    — so even tokens the screen misses can't reach import machinery or the
    filesystem. The code gets a *copy* of ``df`` plus pd/np/px/go, runs inside
    a worker thread with a timeout, and must leave a plotly Figure in ``fig``.
    """
    import builtins as _builtins
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    if not (code or "").strip():
        raise ValueError("模型没有返回代码")
    match = _CHART_FORBIDDEN.search(code)
    if match:
        raise ValueError(f"代码包含被禁止的用法：{match.group(0)}")

    namespace: dict[str, Any] = {
        "__builtins__": {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES},
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }

    def _run() -> Any:
        exec(compile(code, "<chart_code>", "exec"), namespace)  # noqa: S102
        return namespace.get("fig")

    # One-shot worker so a timeout abandons the thread instead of blocking the
    # Streamlit script; shutdown(wait=False) lets the stuck thread die with it.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lv-chart")
    try:
        fig = pool.submit(_run).result(timeout=_CHART_EXEC_TIMEOUT)
    finally:
        pool.shutdown(wait=False)
    if fig is None:
        raise ValueError("代码没有把图表赋值给变量 fig")
    if not isinstance(fig, go.Figure):
        raise TypeError(f"fig 不是 plotly Figure，而是 {type(fig).__name__}")
    return fig


def _heuristic_chart_spec(df: pd.DataFrame) -> dict[str, Any] | None:
    """Sensible default when AI is off or code generation failed.

    Prefers the verdict distribution (the most meaningful view of a result
    set); otherwise pies the first low-cardinality category, else histograms
    the first numeric column."""
    cols = list(df.columns)
    if "verdict" in cols:
        return {
            "chart_type": "pie", "x": "verdict", "y": None, "color": None,
            "agg": "count", "title": "判定结果分布", "reason": "默认展示各判定结果的占比。",
        }
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]) and 1 < df[col].nunique(dropna=True) <= 30:
            return {
                "chart_type": "bar", "x": str(col), "y": None, "color": None,
                "agg": "count", "title": f"{col} 分布", "reason": f"默认按「{col}」统计各类别数量。",
            }
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            return {
                "chart_type": "histogram", "x": str(col), "y": None, "color": None,
                "agg": "none", "title": f"{col} 分布", "reason": f"默认展示「{col}」的数值分布。",
            }
    return None


# --------------------------------------------------------------------- #
# ocr
# --------------------------------------------------------------------- #
def _run_ocr(_text: str, _files: list[dict[str, Any]]) -> dict[str, Any]:
    return _error(
        "当前 DeepSeek 调用只接收文本化数据，不能直接读取截图或文件。"
        "请先用本地 OCR 工具把截图中的参考文献转成文本，粘贴到对话框后我再核验。",
        "ocr",
    )


# --------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------- #
def _run_export(history: list[dict[str, Any]]) -> dict[str, Any]:
    df = _latest_batch_df(history)
    if df is None or df.empty:
        df = list_history(limit=500)
        source_note = "对话内尚无批量结果，已基于本机持久化历史导出。"
    else:
        source_note = "已基于当前对话最近一次批量结果导出。"

    if df.empty:
        return _error("还没有任何验证历史可导出。先做一次单条或批量验证试试。", "export")

    html = build_verification_report_html(df)
    pdf = build_verification_report_pdf(df)
    summary = (
        f"{source_note}\n\n"
        f"包含 **{len(df)}** 条记录，点击下方按钮下载 HTML 或 PDF 报告。"
    )
    filename_stem = f"litverify_report_{datetime.now():%Y%m%d_%H%M}"
    return {
        "role": "assistant",
        "kind": "export",
        "mode": "export",
        "text": summary,
        "data": {
            "html_b64": base64.b64encode(html.encode("utf-8")).decode("ascii"),
            "pdf_b64": base64.b64encode(pdf).decode("ascii"),
            "filename": f"{filename_stem}.html",
            "pdf_filename": f"{filename_stem}.pdf",
        },
    }


# --------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------- #
def _build_chat_messages(
    msg: str,
    history: list[dict[str, Any]],
    files: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Assemble the chat message list — system prompt + grounding context.

    Shared by the (non-streaming) ``_run_chat`` and the streaming
    ``stream_chat`` so both feed the model identical context: aggregate history
    stats, the most recent single-citation report (full rule + evidence
    detail), and a compact view of the most recent batch (counts + flagged
    rows) so follow-ups like「第几条为什么可疑」are answered from data.
    """
    context = {
        "history_summary": history_summary(),
        "last_single_report": _latest_report(history),
        "last_batch": _latest_batch_context(history),
        "uploaded_text": _attachment_text_context(files or []),
    }
    messages: list[dict[str, Any]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    # Replay recent turns for conversational grounding. File-only user turns
    # (an upload with no typed text) are kept as a synthetic note so the model
    # still knows which file was just processed.
    messages.extend(_replay_turns(history, limit=12))
    messages.append(
        {
            "role": "user",
            "content": (
                f"【验证数据（仅供你引用，不要原样回显）】\n"
                f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                f"【用户问题】\n{msg}"
            ),
        }
    )
    return messages


def stream_chat(
    text: str,
    history: list[dict[str, Any]] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(channel, text)`` deltas for live (streaming) chat rendering.

    ``channel`` is ``"reasoning"`` or ``"answer"`` (see
    ``DeepSeekClient.chat_stream``). Never raises: empty input, a missing API
    key and mid-stream failures are all surfaced as ``("answer", text)`` so the
    UI renders them as the reply without special-casing. The concatenated
    *answer* text is what the caller stores back into history.
    """
    history = history or []
    msg = (text or "").strip()
    if not msg:
        yield ("answer", "发点什么我才好回应你哦。")
        return
    if not runtime_api_key():
        yield ("answer", _NO_KEY_HINT.format(feature="智能问答"))
        return

    messages = _build_chat_messages(msg, history, files)
    try:
        # With streaming, the timeout is a per-read budget rather than a
        # whole-answer budget — the model's thinking phase keeps the stream fed,
        # so the slow first token no longer trips "Request timed out".
        client = DeepSeekClient(timeout=120)
        produced_answer = False
        # max_tokens 同时约束「思考 + 正文」：给 1800 时长答案会在句中被硬
        # 截断，思考较长时正文甚至完全为空（表现为"没有返回内容"）。
        for channel, delta in client.chat_stream(
            messages=messages, temperature=0.65, top_p=0.9, max_tokens=6000
        ):
            if channel == "answer":
                produced_answer = True
            yield (channel, delta)
        if not produced_answer:
            yield ("answer", "（模型没有返回内容，请重试或换个问法。）")
    except Exception as exc:  # noqa: BLE001
        hint = (
            "（提示：当前用的是 deepseek-v4-pro，思考模式生成较慢；"
            "在 ⚙ 设置里把模型切到 Flash 通常会快很多。）"
            if "timed out" in str(exc).lower() else ""
        )
        yield (
            "answer",
            f"\n\nAI 暂不可用，但你仍可使用「单条验证 / 批量验证」等本地能力。\n"
            f"具体错误：`{exc}`{hint}",
        )


def _run_chat(
    text: str,
    history: list[dict[str, Any]],
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Non-streaming chat — fallback path (e.g. a caller using ``dispatch``
    directly). The live UI streams via ``stream_chat`` instead."""
    msg = (text or "").strip()
    if not msg:
        return _error("发点什么我才好回应你哦。", "chat")
    if not runtime_api_key():
        return _error(_NO_KEY_HINT.format(feature="智能问答"), "chat")

    messages = _build_chat_messages(msg, history, files)
    try:
        client = DeepSeekClient(timeout=60)
        answer = client.chat(
            messages=messages, temperature=0.65, top_p=0.9, max_tokens=6000, retries=0
        )
    except Exception as exc:  # noqa: BLE001
        hint = (
            "（提示：当前用的是 deepseek-v4-pro，思考模式生成较慢；"
            "在 ⚙ 设置里把模型切到 Flash 通常会快很多。）"
            if "timed out" in str(exc).lower() else ""
        )
        answer = (
            "AI 暂不可用，但你仍可使用「单条验证 / 批量验证」等本地能力。\n"
            f"具体错误：`{exc}`{hint}"
        )

    return {
        "role": "assistant",
        "kind": "chat",
        "mode": "chat",
        "text": answer,
        "data": {},
    }


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _error(message: str, mode: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "kind": "error",
        "mode": mode,
        "text": message,
        "data": {},
    }


def _format_group_rates(groups: dict[str, list[dict[str, Any]]]) -> str:
    """Render the per-dimension (模型/领域/主题) fake-rate breakdown as markdown."""
    blocks: list[str] = []
    for label, rows in groups.items():
        if not rows:
            continue
        lines = "\n".join(
            f"- {r['group']}：虚假率 {r['fake_rate']:.0%}（{r['fake_like']}/{r['total']}），"
            f"主要问题 {r['top_failure']}"
            for r in rows[:6]
        )
        blocks.append(f"**按{label}分组的虚假率：**\n{lines}")
    return "\n\n".join(blocks)


def _verdict_summary(report: VerificationReport) -> str:
    cite = report.user_citation
    bits = []
    if cite.title:
        bits.append(f"《{cite.title}》")
    if cite.doi:
        bits.append(f"DOI `{cite.doi}`")
    head = " · ".join(bits) if bits else (cite.raw or "")[:80]

    # DeepSeek's natural-language read of the evidence (explain_verification),
    # so single-citation results are *told*, not just scored. reasons/summary
    # come straight from the model; fall back gracefully when AI is off.
    explanation = report.explanation or {}
    ai_summary = str(explanation.get("summary") or "").strip()
    reasons = [str(r).strip() for r in (explanation.get("reasons") or []) if str(r).strip()]
    reasons_md = "\n".join(f"- {r}" for r in reasons[:4])

    suggestions = report.suggestions or []
    suggestion_md = "\n".join(f"- {s}" for s in suggestions[:4]) if suggestions else "_暂无修复建议。_"

    parts = [
        f"**{head}**",
        f"判定 **{report.verdict}**，综合得分 **{report.overall_score}/100**。",
    ]
    if ai_summary:
        parts.append(ai_summary)
    if reasons_md:
        parts.append(f"**关键证据**：\n{reasons_md}")
    parts.append(f"**修复建议**：\n{suggestion_md}")
    return "\n\n".join(parts)


def _read_text_file(file: dict[str, Any]) -> str:
    """Best-effort text decode for short .txt-ish attachments."""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return file["bytes"].decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _pick_citation_column(df: pd.DataFrame, hint: str = "") -> str | None:
    if df.empty:
        return None
    hint = (hint or "").lower()
    for col in df.columns:
        if str(col).lower() in hint or str(col) in hint:
            return col
    for col in df.columns:
        name = str(col).lower()
        if "citation" in name or "引用" in name or "reference" in name or "ref" in name:
            return col
    # Fallback: pick the column with the longest average string length
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        return None
    return max(
        object_cols,
        key=lambda c: df[c].astype(str).str.len().mean(),
    )


# Keyword fingerprints used to recognise "this column holds X" for the
# structured-batch path. Each entry maps a citation role to a list of
# case-insensitive substrings; the first matching column wins. Both the
# contest test data (中文表头) and common English exports are covered.
_STRUCTURED_HINTS: dict[str, tuple[str, ...]] = {
    "doi": ("doi", "数字对象标识"),
    "title": ("title", "标题", "题目", "题名"),
    "authors": ("author", "作者", "署名"),
    "year": ("year", "年份", "发表年", "出版年"),
    "venue": ("venue", "journal", "conference", "期刊", "会议", "刊物", "来源"),
}


def _detect_structured_columns(df: pd.DataFrame) -> dict[str, str] | None:
    """Return a {role: column_name} map when the spreadsheet clearly carries
    one citation field per column. Returns None when the file looks like a
    single-column "raw citation" list, which keeps the legacy parser path
    active for backward compatibility.
    """
    if df.empty:
        return None
    columns = [str(c) for c in df.columns]
    matched: dict[str, str] = {}
    for role, hints in _STRUCTURED_HINTS.items():
        for col in columns:
            lowered = col.lower()
            if any(hint in lowered for hint in hints):
                matched[role] = col
                break
    # We treat the input as structured when we have at least two of the
    # high-signal roles (title / authors / doi). Year and venue alone are
    # too generic to flip the path.
    high_signal = sum(1 for role in ("title", "authors", "doi") if role in matched)
    if high_signal >= 2:
        return matched
    return None


def _df_from_files_or_history(
    files: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> pd.DataFrame | None:
    if files:
        try:
            return load_dataframe(io.BytesIO(files[0]["bytes"]), filename=files[0]["name"])
        except Exception:  # noqa: BLE001
            return None
    df = _latest_batch_df(history)
    if df is not None:
        return df
    persisted = list_history(limit=2000)
    return persisted if not persisted.empty else None


def _latest_batch_df(history: list[dict[str, Any]]) -> pd.DataFrame | None:
    for msg in reversed(history):
        if msg.get("kind") == "verify_batch" and msg.get("data", {}).get("rows"):
            return pd.DataFrame(msg["data"]["rows"])
    return None


def _latest_report(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for msg in reversed(history):
        if msg.get("kind") == "verify_single" and msg.get("data", {}).get("report"):
            return msg["data"]["report"]
    return None


# Verbose / redundant row columns dropped from the chat *detail* payload to
# keep the token budget sane. The concise 虚假特征 column already names which
# rules failed, so the raw rule-score dump (reasons) and suggestions add bulk
# without new signal. They are still used when computing 分组分析 below.
_DETAIL_DROP_COLS = {"reasons", "suggestions"}
# Cap on raw rows sent to the model. The contest set is 200; group analysis is
# always computed over *all* rows, so even a truncated detail list stays useful.
_DETAIL_ROW_CAP = 300
_ATTACHMENT_TEXT_EXTS = (".txt", ".md", ".csv", ".tsv", ".json", ".log")
_ATTACHMENT_TEXT_CAP = 12000


def _latest_batch_context(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Full grounding for chat questions about the most recent batch.

    The model gets BOTH halves of what it needs to answer like it ran the
    verification itself:

    * **原始明细** — every verified row with all of its original columns
      (生成模型 / 学术领域 / 有关主题 / 作者 / 发表年份 / DOI ...) plus the
      verdict, score and 虚假特征. So custom asks like「按模型/学科分组统计」or
      「第几条为什么可疑」can be answered straight from the data.
    * **分组分析** — the pre-computed per-model / per-field / per-topic
      fake-rate breakdown (the task-one statistics), so grouped questions are
      answerable even when the raw detail list is truncated.
    """
    for msg in reversed(history):
        if msg.get("kind") != "verify_batch":
            continue
        data = msg.get("data") or {}
        rows = data.get("rows") or []

        analysis: dict[str, Any] | None = None
        if rows:
            try:
                report = build_fake_pattern_report(pd.DataFrame(rows))
                analysis = {
                    "总数": report.get("total"),
                    "高风险占比": round(float(report.get("fake_like_ratio") or 0), 4),
                    "判定计数": report.get("verdict_counts"),
                    "Top失效特征": report.get("top_patterns"),
                    "分组虚假率": report.get("groups"),  # 按 模型 / 领域 / 主题
                }
            except Exception:  # noqa: BLE001 - analysis is best-effort grounding
                analysis = None

        detail = [
            {k: v for k, v in row.items() if k not in _DETAIL_DROP_COLS}
            for row in rows[:_DETAIL_ROW_CAP]
        ]
        return {
            "文件": data.get("filename"),
            "总数": len(rows),
            "统计": data.get("counts") or {},
            "可用字段": list(rows[0].keys()) if rows else [],
            "分组分析": analysis,
            "原始明细": detail,
            "原始明细是否截断": len(rows) > _DETAIL_ROW_CAP,
        }
    return None


def _attachment_text_context(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert readable attachments into bounded text snippets for DeepSeek.

    The model never receives file objects. It only sees text that the app has
    decoded here, plus flags telling it when content was skipped or truncated.
    """
    payload: list[dict[str, Any]] = []
    remaining = _ATTACHMENT_TEXT_CAP
    for file in files[:6]:
        name = str(file.get("name") or "uploaded")
        item: dict[str, Any] = {
            "filename": name,
            "size": file.get("size"),
        }
        lower = name.lower()
        if _looks_like_image(name):
            item["status"] = "skipped_image"
            item["note"] = "图片未传给 DeepSeek；需要先转成文本。"
            payload.append(item)
            continue
        if not lower.endswith(_ATTACHMENT_TEXT_EXTS):
            item["status"] = "skipped_unreadable"
            item["note"] = "该附件类型没有被解码为文本，DeepSeek 看不到其正文。"
            payload.append(item)
            continue
        text = _read_text_file(file).strip()
        if not text:
            item["status"] = "empty_or_binary"
            payload.append(item)
            continue
        if remaining <= 0:
            item["status"] = "skipped_context_limit"
            payload.append(item)
            continue
        snippet = text[:remaining]
        remaining -= len(snippet)
        item.update(
            {
                "status": "included_text",
                "text": snippet,
                "truncated": len(snippet) < len(text),
            }
        )
        payload.append(item)
    return payload


def _replay_turns(
    history: list[dict[str, Any]], limit: int = 12
) -> list[dict[str, Any]]:
    """Replay the last ``limit`` user/assistant turns as chat messages.

    A user turn that only carried an upload (no typed text) is replayed as a
    synthetic note so the model still knows a file was processed.
    """
    out: list[dict[str, Any]] = []
    for prior in history[-limit:]:
        role = prior.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = (prior.get("text") or "").strip()
        if role == "user" and not text:
            names = "、".join(
                str(f.get("name", "")) for f in (prior.get("files") or []) if f.get("name")
            )
            if names:
                text = f"（上传了文件：{names}）"
        if not text:
            continue
        out.append({"role": role, "content": text})
    return out


def _looks_like_image(name: str) -> bool:
    name = (name or "").lower()
    return any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))


# Optional helper used by chat_shell for verify_single rendering
def report_from_dict(payload: dict[str, Any]) -> VerificationReport:
    from services.rule_engine import (
        Citation,
        RuleResult,
        VerificationEvidence,
    )
    user_payload = payload.get("user_citation") or {}
    evidence_payload = payload.get("evidence") or {}
    citation_fields = Citation.__dataclass_fields__
    cite = Citation(**{k: v for k, v in user_payload.items() if k in citation_fields})
    evidence_kwargs: dict[str, Any] = {}
    for field_name in VerificationEvidence.__dataclass_fields__:
        if field_name == "lookup_failed":
            evidence_kwargs[field_name] = bool(evidence_payload.get(field_name, False))
            continue
        record_payload = evidence_payload.get(field_name)
        if isinstance(record_payload, dict):
            evidence_kwargs[field_name] = Citation(
                **{k: v for k, v in record_payload.items() if k in citation_fields}
            )
        else:
            evidence_kwargs[field_name] = None
    evidence = VerificationEvidence(**evidence_kwargs)
    rule_results = [RuleResult(**rr) for rr in payload.get("rule_results", [])]
    return VerificationReport(
        user_citation=cite,
        evidence=evidence,
        rule_results=rule_results,
        overall_score=float(payload.get("overall_score", 0)),
        verdict=str(payload.get("verdict", "ERROR")),
        suggestions=list(payload.get("suggestions") or []),
        explanation=dict(payload.get("explanation") or {}),
    )
