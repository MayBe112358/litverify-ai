"""Agent router — single entry point that turns a (mode, text, files)
triple into a structured assistant message payload.

Each return dict has the same shape so ``ui.chat_shell.render_message``
can replay it after any Streamlit rerun::

    {
        "role": "assistant",
        "kind":  "verify_single" | "verify_batch" | "fake_analysis"
               | "ocr" | "export" | "chat" | "error",
        "text":  "...markdown summary...",
        "data":  {...kind-specific serialisable payload...},
        "mode":  the chip the user picked,
    }
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any

import pandas as pd
from PIL import Image

from config.prompts import BATCH_NARRATE_PROMPT, CHAT_SYSTEM_PROMPT
from db.history import list_history, history_summary
from llm.deepseek_client import DeepSeekClient, runtime_api_key
from llm.ocr_extractor import image_to_citation_text, split_citations
from services.citation_verifier import CitationVerifier
from services.data_loader import load_dataframe
from services.data_processor import batch_verify, batch_verify_structured
from services.exporter import build_verification_report_html, build_verification_report_pdf
from services.fake_analyzer import build_fake_pattern_report
from services.rule_engine import VerificationReport
from utils.dataframe import df_to_json_safe_records


# Mode metadata — labels/icons shown on the user bubble + intent classifier.
MODES: list[dict[str, str]] = [
    {"id": "chat",           "label": "智能问答", "icon": "💬",
     "hint": "围绕验证结果或学术引用自由提问"},
    {"id": "verify_single",  "label": "单条验证", "icon": "📋",
     "hint": "粘贴或上传一条引用，多源比对后给出判定"},
    {"id": "verify_batch",   "label": "批量验证", "icon": "📊",
     "hint": "上传 CSV / Excel，每行一条引用，自动批量核验"},
    {"id": "fake_analysis",  "label": "虚假特征", "icon": "🔬",
     "hint": "对历史或批量结果做虚假模式聚类分析"},
    {"id": "ocr",            "label": "截图识别", "icon": "🖼️",
     "hint": "上传文献综述截图，OCR 提取引用并核验"},
    {"id": "export",         "label": "导出报告", "icon": "📄",
     "hint": "把最近一批结果导出为 HTML 验证报告"},
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
_TEXT_INTENTS = ("verify_single", "fake_analysis", "export", "chat")


def dispatch_auto(
    text: str,
    files: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Agent entry point — infer what the user wants, then run it.

    Replaces the old manual tool chips: file type decides batch/OCR
    deterministically; for plain text we ask DeepSeek to classify the
    intent (with a heuristic fallback when the model is unavailable).
    """
    files = files or []
    history = history or []
    mode = infer_mode(text, files, history)
    return dispatch(mode, text, files, history)


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
        "export —— 用户想导出或下载验证报告（HTML/PDF）。\n"
        "chat —— 其它所有情况：提问、解释概念、对结果追问、闲聊、寒暄等。\n"
        f"当前对话{'已有' if has_results else '尚无'}验证结果。拿不准时优先选 chat。"
    )
    try:
        client = DeepSeekClient(timeout=15)
        raw = client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": msg},
            ],
            temperature=0.0,
            max_tokens=8,
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


def _has_ext(name: str, exts: tuple[str, ...]) -> bool:
    return (name or "").lower().endswith(exts)


def dispatch(
    mode: str,
    text: str,
    files: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the right service for ``mode``.

    ``files`` is a list of {"name": str, "bytes": bytes} — the chat shell
    materialises ``UploadedFile`` into bytes before storing them in
    session history so this function is pure.
    """
    files = files or []
    history = history or []
    try:
        if mode == "verify_single":
            return _run_verify_single(text, files)
        if mode == "verify_batch":
            return _run_verify_batch(text, files)
        if mode == "fake_analysis":
            return _run_fake_analysis(files, history)
        if mode == "ocr":
            return _run_ocr(text, files)
        if mode == "export":
            return _run_export(history)
        return _run_chat(text, history)
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
def _run_verify_batch(text: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    if not files:
        return _error("请上传 CSV / TSV / Excel 引用清单后再点发送。", "verify_batch")

    target = files[0]
    try:
        df = load_dataframe(io.BytesIO(target["bytes"]), filename=target["name"])
    except Exception as exc:  # noqa: BLE001
        return _error(f"无法读取文件 `{target['name']}`：{exc}", "verify_batch")

    structured = _detect_structured_columns(df)
    if structured:
        result_df = batch_verify_structured(df, structured, max_workers=8)
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
        result_df = batch_verify(df, citation_col=column, max_workers=8)
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
            "rows": _df_to_records(result_df),
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

    payload = {
        "来源": source_label,
        "总数": int(len(result_df)),
        "统计": {str(k): int(v) for k, v in counts.items()},
        "用户说明": (user_text or "").strip() or None,
        "问题条目": problems,
        "问题条目是否截断": bool(
            "verdict" in result_df.columns
            and len(result_df[result_df["verdict"].isin(["FAKE", "SUSPICIOUS", "ERROR"])]) > 20
        ),
    }
    try:
        client = DeepSeekClient(timeout=30)
        return client.chat(
            messages=[
                {"role": "system", "content": BATCH_NARRATE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            temperature=0.3,
            max_tokens=1600,
            retries=0,
        ).strip() or None
    except Exception:  # noqa: BLE001 - any failure → deterministic fallback
        return None


# --------------------------------------------------------------------- #
# fake_analysis
# --------------------------------------------------------------------- #
def _run_fake_analysis(
    files: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run fake-pattern analysis on either an uploaded result CSV or the
    most recent batch result still in chat history. Falls back to the
    persistent SQLite history if neither is present."""
    df = _df_from_files_or_history(files, history)
    if df is None or df.empty:
        return _error(
            "需要一个含 `verdict` / `score` / `reasons` 列的结果数据。"
            "先做一次「批量验证」或上传上次的结果 CSV 即可。",
            "fake_analysis",
        )

    stats = build_fake_pattern_report(df)
    bullets = "\n".join(f"- {p}" for p in stats.get("top_patterns", []) or ["暂未发现强模式特征。"])
    summary = (
        f"对 **{stats['total']}** 条引用做了虚假特征聚类，"
        f"高风险（可疑/虚假）占比 **{stats['fake_like_ratio']:.0%}**。\n\n"
        f"**Top 失效规则：**\n{bullets}"
    )
    group_md = _format_group_rates(stats.get("groups") or {})
    if group_md:
        summary += f"\n\n{group_md}"
    return {
        "role": "assistant",
        "kind": "fake_analysis",
        "mode": "fake_analysis",
        "text": summary,
        "data": {
            "stats": stats,
            "rows": _df_to_records(df.head(200)),
        },
    }


# --------------------------------------------------------------------- #
# ocr
# --------------------------------------------------------------------- #
def _run_ocr(text: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    if not runtime_api_key():
        return _error(_NO_KEY_HINT.format(feature="截图识别（OCR）"), "ocr")
    image_files = [f for f in files if _looks_like_image(f["name"])]
    if not image_files:
        return _error("请上传一张文献综述/参考文献截图（jpg / png / webp）。", "ocr")

    target = image_files[0]
    try:
        image = Image.open(io.BytesIO(target["bytes"]))
    except Exception as exc:  # noqa: BLE001
        return _error(f"无法解析图片 `{target['name']}`：{exc}", "ocr")

    extracted = image_to_citation_text(image)
    citations = split_citations(extracted)

    reports: list[dict[str, Any]] = []
    if citations:
        verifier = CitationVerifier()
        for line in citations[:5]:  # cap so the chat bubble stays digestible
            try:
                report = verifier.verify(line, with_llm_explain=False, save=True)
                reports.append(report.to_dict())
            except Exception as exc:  # noqa: BLE001
                reports.append({"error": str(exc), "raw": line})

    extra = (
        f"\n\n_用户附注：{text.strip()}_" if text.strip() else ""
    )
    summary = (
        f"在 `{target['name']}` 中识别到 **{len(citations)}** 条候选引用，"
        f"已抽样核验前 {len(reports)} 条。{extra}"
    )
    return {
        "role": "assistant",
        "kind": "ocr",
        "mode": "ocr",
        "text": summary,
        "data": {
            "raw_text": extracted,
            "citations": citations,
            "reports": reports,
        },
    }


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
            "html_b64": _b64encode(html.encode("utf-8")),
            "pdf_b64": _b64encode(pdf),
            "filename": f"{filename_stem}.html",
            "pdf_filename": f"{filename_stem}.pdf",
        },
    }


# --------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------- #
def _run_chat(text: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    msg = (text or "").strip()
    if not msg:
        return _error("发点什么我才好回应你哦。", "chat")
    if not runtime_api_key():
        return _error(_NO_KEY_HINT.format(feature="智能问答"), "chat")

    # Ground the model in everything we actually verified this session: the
    # aggregate history stats, the most recent single-citation report (full
    # rule + evidence detail), and a compact view of the most recent batch
    # (counts + the flagged rows) so follow-ups like「第几条为什么可疑」can be
    # answered from data instead of guesswork.
    context = {
        "history_summary": history_summary(),
        "last_single_report": _latest_report(history),
        "last_batch": _latest_batch_context(history),
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT}
    ]
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

    try:
        client = DeepSeekClient(timeout=30)
        answer = client.chat(messages=messages, temperature=0.3, max_tokens=1800)
    except Exception as exc:  # noqa: BLE001
        answer = (
            "AI 暂不可用，但你仍可使用「单条验证 / 批量验证」等本地能力。\n"
            f"具体错误：`{exc}`"
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


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records — see
    :func:`utils.dataframe.df_to_json_safe_records` for the details. Kept
    as a thin alias so the dispatch flow reads top-to-bottom."""
    return df_to_json_safe_records(df)


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


def _b64encode(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


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
