"""Floating left-rail shortcuts shown when the sidebar is collapsed.

Four persistent items pinned to the viewport-left edge — the same
vertical order as the expanded sidebar so the layout stays consistent
across both states:

  1. logo            ← rendered by skinning [data-testid="stExpandSidebarButton"]
  2. ✎ 新对话         ← rail_new_chat
  3. ● API key dot   ← rail_status  (just above settings)
  4. ⚙ 设置          ← rail_settings (bottom)

CSS in ``ui/theme.py`` positions each via ``position: fixed`` and hides
the entire rail whenever ``[data-testid="stSidebar"][aria-expanded="true"]``
matches (i.e. the user has pulled the sidebar open).
"""
from __future__ import annotations

import streamlit as st

from ui.settings_dialog import open_settings_dialog
from utils.session import new_session


def render_collapsed_rail() -> None:
    # Custom logo element — shown only when sidebar is EXPANDED (Streamlit
    # removes its own stExpandSidebarButton from the DOM in that case, so
    # we render our own logo so the top-left corner never feels empty).
    # CSS hides this in collapsed state where the skinned expand button
    # takes its place.
    st.markdown(
        '<div class="dw-rail-logo" title="LitVerify AI"></div>',
        unsafe_allow_html=True,
    )

    if st.button("✎", key="rail_new_chat", help="新对话"):
        new_session(activate=True)
        st.rerun()

    # Connection status dot — purely informational, sits just above
    # the settings cog. Hover to read the label.
    is_online = bool(st.session_state.get("deepseek_api_key"))
    dot_color = "var(--dw-success)" if is_online else "var(--dw-muted)"
    label = "DeepSeek 已连接" if is_online else "未配置 API Key"
    st.markdown(
        f"""
        <div class="dw-rail-status" title="{label}">
            <span class="dw-rail-status-dot" style="background:{dot_color};"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("⚙", key="rail_settings", help="设置"):
        open_settings_dialog()
