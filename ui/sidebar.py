"""Sidebar — Gemini-style minimal nav.

Layout:
- Brand: logo + "LitVerify AI" wordmark
- Primary actions: 发起新对话 (with pencil icon)
- "最近" section: list of recent chat sessions
- Bottom: connection pill + settings cog
"""
from __future__ import annotations

import streamlit as st

from ui.components import logo_data_uri
from ui.settings_dialog import open_settings_dialog
from utils.session import (
    active_session,
    delete_session,
    new_session,
    switch_session,
)


def _brand_block() -> None:
    uri = logo_data_uri()
    img_html = (
        f'<img src="{uri}" alt="LitVerify" '
        f'style="width:30px;height:30px;object-fit:contain;flex-shrink:0;" />'
        if uri
        else (
            '<div style="width:30px;height:30px;border-radius:9px;'
            'background:linear-gradient(135deg,var(--dw-primary) 0%,var(--dw-accent) 100%);'
            'color:#fff;font-weight:800;font-size:0.78rem;'
            'display:flex;align-items:center;justify-content:center;'
            'letter-spacing:-0.01em;flex-shrink:0;">LV</div>'
        )
    )
    st.markdown(
        f"""
        <div class="dw-sidebar-brand">
            {img_html}
            <div class="dw-sidebar-brand-text">LitVerify
                <span class="dw-sidebar-brand-grad">AI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        _brand_block()

        # ----- New chat (style hook: .st-key-sidebar_new_chat) -----
        if st.button("✎  发起新对话", use_container_width=True, key="sidebar_new_chat"):
            new_session(activate=True)
            st.rerun()

        st.markdown(
            '<div class="dw-sidebar-section">最近</div>',
            unsafe_allow_html=True,
        )

        sessions: dict[str, dict] = st.session_state.get("sessions", {})
        active_id = st.session_state.get("active_session_id")
        ordered = sorted(
            sessions.values(),
            key=lambda s: s.get("created_at", ""),
            reverse=True,
        )

        # Session list. Each row = a click-to-switch button + a × button.
        for sess in ordered:
            sid = sess["id"]
            title = sess.get("title") or "新对话"
            is_active = (sid == active_id)
            row_l, row_r = st.columns([6, 1])
            with row_l:
                if st.button(
                    title,
                    key=f"sess_pick_{sid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    switch_session(sid)
                    st.rerun()
            with row_r:
                if st.button(
                    "×",
                    key=f"sess_del_{sid}",
                    help="删除此会话",
                    use_container_width=True,
                ):
                    delete_session(sid)
                    st.rerun()

        # ----- Footer: real container so CSS can pin it to sidebar bottom -----
        # st.markdown HTML divs can't wrap subsequent Streamlit widgets;
        # only st.container(key=...) creates a real wrapping element that
        # we can target via .st-key-sidebar_footer and push to flex-end.
        with st.container(key="sidebar_footer"):
            is_online = bool(st.session_state.get("deepseek_api_key"))
            dot_color = "var(--dw-success)" if is_online else "var(--dw-muted)"
            label = "DeepSeek 已连接" if is_online else "未配置 API Key"
            st.markdown(
                f"""
                <div class="dw-sidebar-status">
                    <span class="dw-sidebar-dot" style="background:{dot_color};"></span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("⚙  设置", use_container_width=True, key="sidebar_open_settings"):
                open_settings_dialog()

        # Side-effect only — ensures the current sessions dict always has at
        # least one active session even after delete operations above.
        active_session()
