"""Extract citation text from screenshots with DeepSeek-VL."""
from __future__ import annotations

from PIL import Image

from config.prompts import OCR_CITATION_PROMPT
from llm.deepseek_client import DeepSeekClient


def image_to_citation_text(image: Image.Image, client: DeepSeekClient | None = None) -> str:
    """Extract plain citation text from an image."""
    active_client = client or DeepSeekClient()
    return active_client.vision_extract_table(image, prompt=OCR_CITATION_PROMPT).strip()


def split_citations(text: str) -> list[str]:
    """Split OCR output into individual citation-like lines."""
    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    return [line for line in lines if len(line) >= 12]
