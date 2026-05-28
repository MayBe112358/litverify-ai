"""Tiny shared UI helpers for the agent shell.

Old hero / page_header / KPI strip / status pill are gone — the agent
UI has no page chrome other than the chat shell.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
_LOGO_CANDIDATES = (
    PROJECT_ROOT / "assets" / "logo_sm.png",
    PROJECT_ROOT / "assets" / "logo.png",
    WORKSPACE_ROOT / "logo_sm.png",
    WORKSPACE_ROOT / "logo.png",
)


@lru_cache(maxsize=1)
def logo_data_uri() -> str | None:
    """Return the project logo as a data URI, or None if missing."""
    for path in _LOGO_CANDIDATES:
        if path.exists():
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except Exception:  # noqa: BLE001
                continue
    return None


def safe_html(s: str) -> str:
    """Collapse blank lines so Streamlit's markdown parser doesn't bail out."""
    return "\n".join(line for line in s.splitlines() if line.strip())


def set_tab_chrome(title: str = "LitVerify AI", icon: str = "🔎") -> None:
    """Push the browser tab title + emoji favicon (called once from app.py)."""
    import json
    import streamlit.components.v1 as components

    safe_title = json.dumps(title)
    safe_icon = json.dumps(icon)
    components.html(
        f"""
        <script>
        (function () {{
            function applyChrome() {{
                const doc = window.parent && window.parent.document ? window.parent.document : document;
                doc.title = {safe_title};
                const c = doc.createElement('canvas');
                c.width = 64; c.height = 64;
                const ctx = c.getContext('2d');
                ctx.font = '52px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText({safe_icon}, 32, 36);
                let link = doc.querySelector("link[rel*='icon']");
                if (!link) {{
                    link = doc.createElement('link');
                    link.rel = 'icon';
                    doc.head.appendChild(link);
                }}
                link.href = c.toDataURL('image/png');
            }}
            try {{
                applyChrome();
                setTimeout(applyChrome, 60);
                setTimeout(applyChrome, 240);
            }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )
