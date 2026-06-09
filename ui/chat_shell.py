"""Main chat shell for the LitVerify AI agent.

Layout:
- Empty state → centered logo + welcome line + composer (no chip row).
- Conversation state → message stream + composer fixed at bottom.

Composer (single component used in both states):
- Left: a "＋" button that opens the native file picker directly
  (no popover menu — the agent infers the action from message + file).
- Middle: `st.chat_input` with native Enter-to-send + multi-line.
- Above the input row (only when something is staged): chips showing
  the currently selected mode and pending file attachments, each with
  an × to clear.
"""
from __future__ import annotations

import base64
from html import escape
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from services.agent_router import (
    MODES_BY_ID,
    dispatch,
    infer_mode,
    report_from_dict,
    stream_chat,
)
from ui._scripts import DRAG_BRIDGE_SCRIPT
from ui.verdict_card import render_verdict_card
from utils.session import active_session, append_message


def _set_empty_flag(is_empty: bool) -> None:
    """Push html[data-dw-empty="1|0"] so CSS swaps composer layout."""
    flag = "1" if is_empty else "0"
    components.html(
        f"""
        <script>
        (function () {{
            var doc = window.parent && window.parent.document
                ? window.parent.document : document;
            doc.documentElement.setAttribute("data-dw-empty", "{flag}");
        }})();
        </script>
        """,
        height=0,
    )


_ACCEPTED_TYPES = [
    "csv", "tsv", "xlsx", "xls", "json", "parquet",
    "png", "jpg", "jpeg", "webp", "bmp",
    "txt", "md",
]


# --------------------------------------------------------------------- #
# Welcome (empty) state — single welcome line; composer renders below
# --------------------------------------------------------------------- #
def _render_welcome() -> None:
    st.markdown(
        """
        <div class="dw-welcome">
            <h1 class="dw-welcome-title">我能帮你验证哪条引用？</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- #
# Per-message renderers
# --------------------------------------------------------------------- #
def _render_user(message: dict[str, Any]) -> None:
    with st.chat_message("user"):
        mode = message.get("mode") or "chat"
        mode_meta = MODES_BY_ID.get(mode, {})
        chips: list[str] = []
        if mode != "chat":
            chips.append(
                f'<span style="display:inline-block;padding:0.12rem 0.55rem;'
                f'border-radius:999px;background:var(--dw-primary-soft);'
                f'color:var(--dw-primary);-webkit-text-fill-color:var(--dw-primary);'
                f'font-size:0.72rem;font-weight:600;margin-right:0.35rem;">'
                f'{mode_meta.get("icon", "")} {mode_meta.get("label", mode)}</span>'
            )
        for f in message.get("files") or []:
            chips.append(
                f'<span style="display:inline-block;padding:0.12rem 0.55rem;'
                f'border-radius:999px;background:var(--dw-chip-bg);'
                f'color:var(--dw-text-soft);font-size:0.72rem;'
                f'margin-right:0.35rem;">📎 {escape(f["name"])}</span>'
            )
        if chips:
            st.markdown(
                f'<div style="margin-bottom:0.4rem;">{"".join(chips)}</div>',
                unsafe_allow_html=True,
            )
        if message.get("text"):
            st.write(message["text"])


# {assistant message kind → renderer for the inner payload}. The renderers
# all share the (data, idx) signature so ``_render_assistant`` can dispatch
# in two lines instead of a five-branch ``if/elif`` chain. The dict itself
# is populated at the bottom of this module once each renderer is defined.
_KIND_RENDERERS: dict[str, Callable[[dict[str, Any], int], None]] = {}

_EVIDENCE_SOURCE_LABELS = {
    "crossref": "CrossRef",
    "openalex": "OpenAlex",
    "pubmed": "PubMed",
    "semantic_scholar": "Semantic Scholar",
    "dblp": "DBLP",
    "wanfang": "万方",
    "datacite": "DataCite",
    "arxiv": "arXiv",
    "doidb": "DOIDB",
}


def _render_assistant(message: dict[str, Any], idx: int) -> None:
    kind = message.get("kind", "chat")
    with st.chat_message("assistant"):
        if message.get("text"):
            st.markdown(message["text"])
        renderer = _KIND_RENDERERS.get(kind)
        if renderer:
            renderer(message.get("data") or {}, idx)


def _render_verify_single(data: dict[str, Any], idx: int) -> None:
    payload = data.get("report")
    if not payload:
        return
    try:
        report = report_from_dict(payload)
        render_verdict_card(report)
    except Exception as exc:  # noqa: BLE001
        st.error(f"渲染验证卡片失败：{exc}")
        return

    with st.expander("规则评分明细", expanded=False):
        rule_rows = [
            {
                "规则": r.name,
                "得分": round(r.score, 2),
                "权重": r.weight,
                "通过": "✓" if r.passed else "✕",
                "说明": r.reason,
            }
            for r in report.rule_results
        ]
        if rule_rows:
            st.dataframe(
                pd.DataFrame(rule_rows),
                use_container_width=True,
                hide_index=True,
                key=f"vs_rules_{idx}",
            )

    evidence_payload = payload.get("evidence") or {}
    evidence_rows = []
    for source, label in _EVIDENCE_SOURCE_LABELS.items():
        rec = evidence_payload.get(source) or {}
        if rec:
            evidence_rows.append(
                {
                    "来源": label,
                    "标题": rec.get("title") or "—",
                    "DOI": rec.get("doi") or "—",
                    "年份": rec.get("year") or "—",
                    "期刊/来源": rec.get("venue") or "—",
                }
            )
    if evidence_rows:
        with st.expander("外部证据", expanded=False):
            st.dataframe(
                pd.DataFrame(evidence_rows),
                use_container_width=True,
                hide_index=True,
                key=f"vs_evidence_{idx}",
            )


def _render_verify_batch(data: dict[str, Any], idx: int) -> None:
    rows = data.get("rows") or []
    if not rows:
        return
    df = pd.DataFrame(rows)

    counts = data.get("counts") or {}
    if counts:
        chip_html = "".join(
            f'<div style="flex:1;background:var(--dw-surface-alt);'
            f'border:1px solid var(--dw-border);border-radius:12px;'
            f'padding:0.65rem 0.9rem;text-align:center;">'
            f'<div style="font-size:0.7rem;color:var(--dw-muted);'
            f'font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">{k}</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:var(--dw-text);'
            f'margin-top:0.15rem;">{v}</div></div>'
            for k, v in counts.items()
        )
        st.markdown(
            f'<div style="display:flex;gap:0.5rem;margin:0.4rem 0 0.8rem;">{chip_html}</div>',
            unsafe_allow_html=True,
        )

    if "verdict" in df.columns and "score" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="verdict", hole=0.55, title=None)
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"batch_pie_{idx}")
        with c2:
            fig = px.histogram(df, x="score", color="verdict", nbins=20, title=None)
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"batch_hist_{idx}")

    with st.expander(f"详细结果（{len(df)} 行）", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True, key=f"batch_df_{idx}")

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 下载结果 CSV",
        csv,
        file_name=f"litverify_batch_{idx}.csv",
        mime="text/csv",
        key=f"download_batch_{idx}",
    )


def _render_fake_analysis(data: dict[str, Any], idx: int) -> None:
    stats = data.get("stats") or {}
    rows = data.get("rows") or []
    counts = stats.get("verdict_counts") or {}
    if counts:
        df = pd.DataFrame([{"verdict": k, "count": v} for k, v in counts.items()])
        fig = px.bar(df, x="verdict", y="count", title=None)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, key=f"fake_bar_{idx}")
    if rows:
        with st.expander(f"参与分析的样本（{len(rows)} 行）", expanded=False):
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                key=f"fake_df_{idx}",
            )


def _render_chart(data: dict[str, Any], idx: int) -> None:
    """Render a chart from a validated spec + the result rows.

    The spec was already sanitised in ``agent_router._validate_chart_spec``
    (every column verified against the DataFrame), so here we only do the
    pandas aggregation + Plotly call. No ``eval``/``exec``, no model-supplied
    code — just a fixed mapping from chart_type to a ``plotly.express`` call."""
    spec = data.get("spec") or {}
    rows = data.get("rows") or []
    if not spec or not rows:
        return
    df = pd.DataFrame(rows)

    try:
        fig = _build_chart_fig(df, spec)
    except Exception as exc:  # noqa: BLE001 - never let a bad spec break the stream
        st.warning(f"这张图没能画出来：{exc}")
        return
    if fig is None:
        st.warning("这份数据不足以生成该图表。")
        return

    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

    with st.expander(f"作图所用数据（{len(df)} 行）", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True, key=f"chart_df_{idx}")


def _build_chart_fig(df: pd.DataFrame, spec: dict[str, Any]):
    """Map a validated spec to a Plotly figure. Returns None if unplottable."""
    chart_type = spec.get("chart_type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    agg = spec.get("agg", "count")
    title = spec.get("title") or "数据图表"

    if x not in df.columns:
        return None
    if color is not None and color not in df.columns:
        color = None

    # Aggregate first when asked, so bar/line/pie show one value per category
    # instead of Plotly's implicit stacking of raw rows.
    plot_df = df
    if agg in ("count", "sum", "mean") and chart_type in ("bar", "line", "pie"):
        group_cols = [c for c in (x, color) if c]
        if agg == "count":
            plot_df = df.groupby(group_cols, dropna=False).size().reset_index(name="计数")
            value_col = "计数"
        else:
            if y not in df.columns:
                return None
            grouped = df.groupby(group_cols, dropna=False)[y]
            plot_df = (grouped.sum() if agg == "sum" else grouped.mean()).reset_index()
            value_col = y
    else:
        value_col = y

    if chart_type == "pie":
        return px.pie(plot_df, names=x, values=value_col, color=color, hole=0.45, title=title)
    if chart_type == "histogram":
        return px.histogram(df, x=x, color=color, nbins=20, title=title)
    if chart_type == "scatter":
        if y not in df.columns:
            return None
        return px.scatter(df, x=x, y=y, color=color, title=title)
    if chart_type == "box":
        if y not in df.columns:
            return None
        return px.box(df, x=x, y=y, color=color, title=title)
    if chart_type == "line":
        return px.line(plot_df, x=x, y=value_col, color=color, markers=True, title=title)
    # default: bar
    return px.bar(plot_df, x=x, y=value_col, color=color, barmode="group", title=title)


def _render_ocr(data: dict[str, Any], idx: int) -> None:
    raw_text = data.get("raw_text", "")
    citations = data.get("citations", []) or []
    reports = data.get("reports", []) or []

    if citations:
        with st.expander(f"识别到的引用 · {len(citations)} 条", expanded=False):
            for line in citations:
                st.markdown(f"- {line}")

    if reports:
        with st.expander(f"抽样核验结果 · {len(reports)} 条", expanded=True):
            for rep in reports:
                if rep.get("error"):
                    st.warning(f"`{rep.get('raw', '')[:80]}` → {rep['error']}")
                    continue
                try:
                    render_verdict_card(report_from_dict(rep))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"卡片渲染失败：{exc}")

    if raw_text:
        with st.expander("OCR 原始文本", expanded=False):
            st.code(raw_text, language=None)


def _render_export(data: dict[str, Any], idx: int) -> None:
    html_b64 = data.get("html_b64")
    if not html_b64:
        return
    html_bytes = base64.b64decode(html_b64.encode("ascii"))
    pdf_b64 = data.get("pdf_b64")
    if pdf_b64:
        col_pdf, col_html = st.columns(2)
        with col_pdf:
            st.download_button(
                "⬇️ 下载 PDF 报告",
                base64.b64decode(pdf_b64.encode("ascii")),
                file_name=data.get("pdf_filename") or "litverify_report.pdf",
                mime="application/pdf",
                key=f"download_export_pdf_{idx}",
                type="primary",
                use_container_width=True,
            )
        with col_html:
            st.download_button(
                "⬇️ 下载 HTML 报告",
                html_bytes,
                file_name=data.get("filename") or "litverify_report.html",
                mime="text/html",
                key=f"download_export_html_{idx}",
                use_container_width=True,
            )
        return

    st.download_button(
        "⬇️ 下载 HTML 报告",
        html_bytes,
        file_name=data.get("filename") or "litverify_report.html",
        mime="text/html",
        key=f"download_export_{idx}",
        type="primary",
    )


# --------------------------------------------------------------------- #
# Composer  (＋ upload button  |  chat_input)
# --------------------------------------------------------------------- #
def _ensure_composer_state() -> None:
    st.session_state.setdefault("pending_mode", "chat")
    st.session_state.setdefault("pending_files", [])
    st.session_state.setdefault("file_uploader_nonce", 0)
    st.session_state.setdefault("home_uploader_nonce", 0)


def _clear_pending_files() -> None:
    st.session_state["pending_files"] = []
    _bump_uploaders()


def _bump_uploaders() -> None:
    """Force upload widgets to render fresh after staging/removing files."""
    st.session_state["file_uploader_nonce"] = (
        int(st.session_state.get("file_uploader_nonce", 0)) + 1
    )
    st.session_state["home_uploader_nonce"] = (
        int(st.session_state.get("home_uploader_nonce", 0)) + 1
    )


def _materialise_files(uploaded_files) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in uploaded_files or []:
        try:
            data = f.read()
        except Exception:  # noqa: BLE001
            data = b""
        out.append(
            {
                "name": getattr(f, "name", "uploaded"),
                "size": len(data),
                "type": getattr(f, "type", ""),
                "bytes": data,
            }
        )
    return out


def _stage_uploaded_files(uploaded_files) -> bool:
    """Append newly uploaded files to the composer staging area."""
    existing_names = {f["name"] for f in st.session_state.get("pending_files", [])}
    new_files = []
    seen_names = set(existing_names)
    for uploaded in uploaded_files or []:
        name = getattr(uploaded, "name", "")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        new_files.append(uploaded)
    if not new_files:
        return False
    st.session_state["pending_files"] = (
        st.session_state.get("pending_files", [])
        + _materialise_files(new_files)
    )
    _bump_uploaders()
    return True


def _render_home_drop_upload() -> None:
    """Invisible full-home drop target that appears only while dragging files."""
    components.html(DRAG_BRIDGE_SCRIPT, height=0)
    st.markdown(
        '<div class="dw-drop-preview"><div class="dw-drop-card">📎 松开即可上传文件</div></div>',
        unsafe_allow_html=True,
    )
    nonce = int(st.session_state.get("home_uploader_nonce", 0))
    with st.container(key="home_drop_upload"):
        uploaded = st.file_uploader(
            "拖拽文件到主页上传",
            accept_multiple_files=True,
            key=f"home_uploader_{nonce}",
            label_visibility="collapsed",
        )
    if _stage_uploaded_files(uploaded):
        st.rerun()


def _render_pending_chips() -> None:
    """Above the input: chips for pending file attachments."""
    files = st.session_state.get("pending_files") or []
    if not files:
        return

    with st.container(key="stage_row"):
        for idx, f in enumerate(files):
            filename = _compact_filename(str(f.get("name", "uploaded")))
            file_key = f"file_{idx}_{_key_part(str(f.get('name', 'uploaded')))}"

            def remove_file(index: int = idx) -> None:
                pending = list(st.session_state.get("pending_files") or [])
                if 0 <= index < len(pending):
                    pending.pop(index)
                    st.session_state["pending_files"] = pending
                    _bump_uploaders()

            _render_stage_chip(
                key=file_key,
                label=f"📎 {filename}",
                variant="file",
                help_text="移除此附件",
                on_remove=remove_file,
            )


def _render_stage_chip(
    key: str,
    label: str,
    variant: str,
    help_text: str,
    on_remove,
) -> None:
    css_variant = "mode" if variant == "mode" else "file"
    with st.container(key=f"stage_chip_{key}"):
        st.markdown(
            (
                f'<span class="dw-stage-chip-label dw-stage-chip-{css_variant}">'
                f'{escape(label)}</span>'
            ),
            unsafe_allow_html=True,
        )
        if st.button("×", key=f"stage_remove_{key}", help=help_text):
            on_remove()
            st.rerun()


def _compact_filename(name: str, limit: int = 28) -> str:
    if len(name) <= limit:
        return name
    stem, dot, suffix = name.rpartition(".")
    if not dot or len(suffix) > 8:
        return f"{name[:limit - 1]}…"
    keep = max(8, limit - len(suffix) - 4)
    return f"{stem[:keep]}….{suffix}"


def _key_part(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")[:28]


def _render_plus_uploader() -> None:
    """The composer "＋" button — clicking it opens the native file picker
    directly (no popover menu).

    Implemented as an ``st.file_uploader`` whose dropzone is restyled into
    a compact circular "＋" via the ``.st-key-composer_plus`` CSS hook in
    ``theme.css``. The agent then infers what to do from the message + the
    uploaded file type (see ``services.agent_router.dispatch_auto``).
    """
    nonce = int(st.session_state.get("file_uploader_nonce", 0))
    with st.container(key="composer_plus"):
        uploaded = st.file_uploader(
            "上传文件",
            accept_multiple_files=True,
            key=f"plus_uploader_{nonce}",
            label_visibility="collapsed",
        )
    if _stage_uploaded_files(uploaded):
        st.rerun()


def _composer(is_empty: bool) -> None:
    """Render the composer pill: stage chips + ＋ upload button + chat_input.

    Layout (empty vs docked) is driven by ``html[data-dw-empty]`` and the
    ``.st-key-composer_shell`` CSS hook — no Python branching needed.
    """
    _ensure_composer_state()

    shell = st.container(key="composer_shell")
    with shell:
        _render_pending_chips()
        col_plus, col_input = st.columns([0.07, 0.93], gap="small")
        with col_plus:
            _render_plus_uploader()
        with col_input:
            prompt = st.chat_input(
                "粘贴一条引用、上传文献表格/截图，或直接提问…",
                key=f"composer_input_{st.session_state.get('active_session_id', 'x')}",
            )

    if not prompt:
        return

    if isinstance(prompt, str):
        text = (prompt or "").strip()
        input_files = []
    else:
        text = (getattr(prompt, "text", "") or "").strip()
        input_files = getattr(prompt, "files", []) or []

    files = list(st.session_state.get("pending_files") or [])
    if input_files:
        existing_names = {f["name"] for f in files}
        files.extend(
            f for f in _materialise_files(input_files)
            if f["name"] not in existing_names
        )

    if not text and not files:
        return

    user_msg = {
        "role": "user",
        "text": text,
        # No manual mode anymore — the agent infers the action. The user
        # bubble just shows their text + any attachments.
        "mode": "chat",
        "files": [{"name": f["name"], "size": f["size"], "type": f.get("type", "")} for f in files],
    }
    append_message(user_msg)

    sess = active_session()
    history = sess["messages"][:-1]
    # Decide the action here, where the file *bytes* are still in hand. Chat is
    # deferred to a streaming render on the next run (so the answer types out
    # live); every other tool keeps the spinner + full-render path.
    mode = infer_mode(text, files, history)
    if mode == "chat":
        st.session_state["_pending_chat"] = {"text": text, "files": files}
        _clear_pending_files()
        st.rerun()

    with st.spinner("🤖  正在理解你的需求…"):
        assistant_msg = dispatch(mode, text, files, history)
    append_message(assistant_msg)

    _clear_pending_files()
    st.rerun()


# --------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------- #
def render_chat_shell() -> None:
    sess = active_session()
    messages = sess.get("messages") or []
    is_empty = not messages

    _set_empty_flag(is_empty)
    _ensure_composer_state()
    _render_home_drop_upload()

    if is_empty:
        _render_welcome()
    else:
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                _render_user(msg)
            else:
                _render_assistant(msg, idx=i)
        # A pending chat reply streams in here — right after the last stored
        # message, before the spacer — so it lands in the correct position.
        _stream_pending_chat()
        # Spacer so the last message isn't hidden behind the fixed composer
        st.markdown('<div style="height:140px;"></div>', unsafe_allow_html=True)

    _composer(is_empty=is_empty)


def _stream_pending_chat() -> None:
    """Render a deferred chat reply token-by-token, then persist it.

    ``_composer`` stashes ``_pending_chat`` (with the file bytes) and reruns;
    we pick it up here, stream the answer into a live ``st.empty`` placeholder,
    store the final text as a normal assistant message, and rerun so later
    renders replay it from history like any other message."""
    pending = st.session_state.pop("_pending_chat", None)
    if not pending:
        return
    sess = active_session()
    history = sess["messages"][:-1]  # exclude the user turn we're answering

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 正在思考…")
        full = ""
        try:
            for delta in stream_chat(pending["text"], history, pending["files"]):
                full += delta
                placeholder.markdown(full + " ▌")
        except Exception as exc:  # noqa: BLE001 - backstop; stream_chat rarely raises
            full = full or f"AI 暂不可用：`{exc}`"
        placeholder.markdown(full or "（无回复）")

    append_message(
        {"role": "assistant", "kind": "chat", "mode": "chat", "text": full or "（无回复）", "data": {}}
    )
    st.rerun()


# Register the per-kind renderers now that they're all defined above.
_KIND_RENDERERS.update(
    {
        "verify_single": _render_verify_single,
        "verify_batch": _render_verify_batch,
        "fake_analysis": _render_fake_analysis,
        "chart": _render_chart,
        "ocr": _render_ocr,
        "export": _render_export,
    }
)
