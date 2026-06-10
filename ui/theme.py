"""LitVerify AI — agent-first theme.

Pared-down design language inspired by ChatGPT / Claude / Gemini:
- One flat surface for the canvas, sidebar slightly darker.
- Soft indigo accent, used sparingly (active states, send button).
- Generous padding, mid radii (10/14/22).
- Components: chat bubble, composer with tool-chip strip, sidebar list rows.

Light + Dark only — the old "政务蓝" palette is gone; settings dialog
keeps the toggle minimal.
"""
from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from ui.components import logo_data_uri


# ----------------------------- palettes ------------------------------- #
THEMES: dict[str, dict[str, str]] = {
    "浅色": {
        "scheme":         "light",
        "bg":             "#FFFFFF",
        "canvas":         "#FFFFFF",
        "surface":        "#FFFFFF",
        "surface_alt":    "#F7F7F8",
        "sidebar":        "#F9F9F9",
        "bubble_user":    "#F4F4F5",
        "bubble_ai":      "#FFFFFF",
        "composer_bg":    "#FFFFFF",
        "chip_bg":        "#F4F4F5",
        "chip_bg_active": "#E8EAFE",
        "text":           "#1F2128",
        "text_soft":      "#3F4250",
        "muted":          "#8E94A3",
        "primary":        "#5654E5",
        "primary_strong": "#4845D2",
        "primary_soft":   "rgba(86, 84, 229, 0.10)",
        "accent":         "#7E7BFA",
        "success":        "#10B981",
        "warning":        "#F59E0B",
        "danger":         "#EF4444",
        "border":         "#E5E5EA",
        "border_strong":  "#D1D1D8",
        "input_bg":       "#FFFFFF",
        "code_bg":        "#F4F4F5",
    },
    "深色": {
        "scheme":         "dark",
        "bg":             "#1A1B1F",
        "canvas":         "#212327",
        "surface":        "#2A2C32",
        "surface_alt":    "#26282D",
        "sidebar":        "#181A1D",
        "bubble_user":    "#33353B",
        "bubble_ai":      "#212327",
        "composer_bg":    "#2A2C32",
        "chip_bg":        "#33353B",
        "chip_bg_active": "#3A3A60",
        "text":           "#ECECF0",
        "text_soft":      "#C9CAD1",
        "muted":          "#8A8E99",
        "primary":        "#8A88FF",
        "primary_strong": "#A6A4FF",
        "primary_soft":   "rgba(138, 136, 255, 0.16)",
        "accent":         "#A6A4FF",
        "success":        "#34D399",
        "warning":        "#FBBF24",
        "danger":         "#F87171",
        "border":         "#3A3C42",
        "border_strong":  "#4A4C54",
        "input_bg":       "#2A2C32",
        "code_bg":        "#1F2024",
    },
}

DEFAULT_THEME = "浅色"


def current_theme_name() -> str:
    stored = st.session_state.get("theme_name", DEFAULT_THEME)
    return stored if stored in THEMES else DEFAULT_THEME


def get_theme(name: str | None = None) -> dict[str, str]:
    return THEMES.get(name or current_theme_name(), THEMES[DEFAULT_THEME])


# ----------------------- Plotly template per theme -------------------- #
def _register_plotly_templates() -> None:
    for name, theme in THEMES.items():
        tpl = go.layout.Template()
        tpl.layout = go.Layout(
            paper_bgcolor=theme["surface"],
            plot_bgcolor=theme["surface"],
            font=dict(
                color=theme["text"],
                family="'Inter','PingFang SC','Microsoft YaHei',sans-serif",
            ),
            colorway=[
                theme["primary"], theme["accent"], theme["success"],
                theme["warning"], theme["danger"], "#A78BFA", "#F472B6",
            ],
            xaxis=dict(gridcolor=theme["border"], linecolor=theme["border_strong"],
                       zerolinecolor=theme["border_strong"], tickcolor=theme["border_strong"]),
            yaxis=dict(gridcolor=theme["border"], linecolor=theme["border_strong"],
                       zerolinecolor=theme["border_strong"], tickcolor=theme["border_strong"]),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=48, b=20),
        )
        pio.templates[f"litverify_{name}"] = tpl


_register_plotly_templates()



# ----------------------------- CSS ------------------------------------ #
# The bulk of the stylesheet lives in ``ui/theme.css`` as plain CSS so it's
# easy to edit with proper syntax highlighting. ``_build_css`` only needs
# to produce the small ``:root {...}`` block whose values come from the
# active palette — everything else in the file references those CSS vars.
_CSS_PATH = Path(__file__).resolve().parent / "theme.css"


def _build_root_block(t: dict[str, str]) -> str:
    return f"""
:root {{
    color-scheme: {t["scheme"]};
    --dw-bg: {t["bg"]};
    --dw-canvas: {t["canvas"]};
    --dw-surface: {t["surface"]};
    --dw-surface-alt: {t["surface_alt"]};
    --dw-sidebar: {t["sidebar"]};
    --dw-bubble-user: {t["bubble_user"]};
    --dw-bubble-ai: {t["bubble_ai"]};
    --dw-composer-bg: {t["composer_bg"]};
    --dw-chip-bg: {t["chip_bg"]};
    --dw-chip-bg-active: {t["chip_bg_active"]};
    --dw-text: {t["text"]};
    --dw-text-soft: {t["text_soft"]};
    --dw-muted: {t["muted"]};
    --dw-primary: {t["primary"]};
    --dw-primary-strong: {t["primary_strong"]};
    --dw-primary-soft: {t["primary_soft"]};
    --dw-accent: {t["accent"]};
    --dw-success: {t["success"]};
    --dw-warning: {t["warning"]};
    --dw-danger: {t["danger"]};
    --dw-border: {t["border"]};
    --dw-border-strong: {t["border_strong"]};
    --dw-input-bg: {t["input_bg"]};
    --dw-code-bg: {t["code_bg"]};

    --dw-radius-sm: 8px;
    --dw-radius-md: 12px;
    --dw-radius-lg: 16px;
    --dw-radius-xl: 22px;

    --dw-shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
    --dw-shadow-sm: 0 2px 8px rgba(15, 23, 42, 0.06);
    --dw-shadow-md: 0 12px 28px rgba(15, 23, 42, 0.10);
    --dw-glow: 0 0 0 3px color-mix(in srgb, {t["primary"]} 22%, transparent);

    --dw-trans: 160ms cubic-bezier(0.22, 1, 0.36, 1);

    --background-color: {t["canvas"]} !important;
    --secondary-background-color: {t["surface"]} !important;
    --primary-color: {t["primary"]} !important;
    --text-color: {t["text"]} !important;
}}
"""


def _build_css(t: dict[str, str]) -> str:
    """Build the full stylesheet — :root vars + the static theme.css body."""
    return _build_root_block(t) + "\n" + _CSS_PATH.read_text(encoding="utf-8")


_THEME_MARKER_JS = """
<script>
(function () {
    var theme = "__THEME__";
    var doc = window.parent && window.parent.document ? window.parent.document : document;
    doc.documentElement.setAttribute("data-dw-theme", theme);
    if (doc.body) doc.body.setAttribute("data-dw-theme", theme);
})();
</script>
"""


def apply_theme(name: str | None = None) -> None:
    """Inject the active theme's CSS + matching Plotly template."""
    active = name or current_theme_name()
    theme = get_theme(active)
    pio.templates.default = f"litverify_{active}"

    css = _build_css(theme)
    # Inject the logo data URI into the CSS for the collapsed-state
    # sidebar-expand button skin. If no logo file is found, fall back
    # to a 1px transparent gif so the rule remains valid.
    logo_uri = logo_data_uri() or (
        "data:image/gif;base64,"
        "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    css = css.replace("__LOGO_URI__", logo_uri)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    marker = "dark" if active == "深色" else "light"
    st.iframe(
        _THEME_MARKER_JS.replace("__THEME__", marker),
        height=1,
    )
