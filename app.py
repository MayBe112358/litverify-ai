"""LitVerify AI — single-page agent shell.

The old multi-page navigation is gone. The entire experience is a chat
window with tool-chip routing (single / batch / fake-pattern / OCR /
export / chat) and a slim sidebar for session management + settings.
"""
from __future__ import annotations

import streamlit as st

from config.settings import settings
from ui.chat_shell import render_chat_shell
from ui.components import set_tab_chrome
from ui.sidebar import render_sidebar
from ui.theme import apply_theme
from utils.session import init_session


st.set_page_config(
    page_title=settings.app_name,
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session()
apply_theme()
set_tab_chrome(title=settings.app_name, icon="🔎")
render_sidebar()

# Floating shortcuts that only show when the sidebar is collapsed:
# logo (handled via CSS skin on stExpandSidebarButton), + new chat, ⚙ settings.
from ui.rail import render_collapsed_rail  # noqa: E402 — must run after init_session
render_collapsed_rail()

render_chat_shell()
