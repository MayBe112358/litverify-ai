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
from datetime import datetime
from typing import Any

import pandas as pd
from PIL import Image

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


# Mode metadata — also drives ui/tools_menu.py chip list
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
    summary = (
        f"已验证 **{len(result_df)}** 条引用（{source_label}）。\n\n"
        f"- ✅ 真实 {int(counts.get('REAL', 0))}\n"
        f"- ⚠️ 可疑 {int(counts.get('SUSPICIOUS', 0))}\n"
        f"- ❌ 虚假 {int(counts.get('FAKE', 0))}\n"
        f"- 🛑 错误 {int(counts.get('ERROR', 0))}"
    )

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

    summary = history_summary()
    last_report = _latest_report(history)
    context = {
        "history_summary": summary,
        "last_report": last_report,
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是 LitVerify AI 的学术引用验证助手。回答要基于已给证据，"
                "不能编造检索结果。鼓励用户在需要时切换聊天框上方的功能 chip："
                "📋 单条验证 / 📊 批量验证 / 🔬 虚假特征 / 🖼️ 截图识别 / 📄 导出报告。"
            ),
        }
    ]
    # Replay last few assistant/user turns for grounding
    for prior in history[-8:]:
        if prior.get("role") not in {"user", "assistant"}:
            continue
        if not prior.get("text"):
            continue
        messages.append({"role": prior["role"], "content": prior["text"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"对话上下文 JSON：{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
                f"用户问题：{msg}"
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
    suggestions = report.suggestions or []
    suggestion_md = "\n".join(f"- {s}" for s in suggestions[:4]) if suggestions else "_暂无修复建议。_"
    return (
        f"**{head}**\n\n"
        f"判定 **{report.verdict}**，综合得分 **{report.overall_score}/100**。\n\n"
        f"**修复建议**：\n{suggestion_md}"
    )


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
    cite = Citation(**{k: v for k, v in user_payload.items() if k in Citation.__dataclass_fields__})
    evidence = VerificationEvidence(
        crossref=Citation(**evidence_payload["crossref"]) if evidence_payload.get("crossref") else None,
        openalex=Citation(**evidence_payload["openalex"]) if evidence_payload.get("openalex") else None,
        arxiv=Citation(**evidence_payload["arxiv"]) if evidence_payload.get("arxiv") else None,
    )
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
