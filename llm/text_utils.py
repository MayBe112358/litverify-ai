"""Small text helpers for model responses."""
from __future__ import annotations


def strip_fenced_block(text: str, language: str | None = None) -> str:
    """Return model output without a surrounding Markdown code fence."""
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if language and content.lstrip().startswith(language):
            content = content.lstrip()[len(language) :]
    return content.strip(" \n`")
