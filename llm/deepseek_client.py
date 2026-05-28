"""DeepSeek client using the OpenAI-compatible API."""
from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any

from PIL import Image

from config.prompts import OCR_CITATION_PROMPT
from config.settings import settings

try:  # pragma: no cover - import availability depends on environment
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def _session_str(key: str) -> str:
    """Read a string from Streamlit session_state outside the script context."""
    try:
        import streamlit as st

        return str(st.session_state.get(key) or "").strip()
    except Exception:
        return ""


def runtime_api_key() -> str:
    """Prefer sidebar API key, then environment API key."""
    return _session_str("deepseek_api_key") or settings.deepseek_api_key


def runtime_chat_model() -> str:
    """Prefer the chat-model name set in the sidebar, then the env default."""
    return _session_str("deepseek_chat_model") or settings.chat_model


def runtime_vl_model() -> str:
    """Prefer the vision-model name set in the sidebar, then the env default."""
    return _session_str("deepseek_vl_model") or settings.vl_model


class DeepSeekClient:
    """Wrap chat and vision calls with consistent retry behavior."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        if OpenAI is None:
            raise RuntimeError("未安装 openai SDK，请先安装 requirements.txt 中的依赖。")
        key = api_key or runtime_api_key()
        if not key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 或侧边栏中填写。")
        # ``timeout`` caps total request time so the UI never hangs forever
        # if DeepSeek is slow or unreachable; combined with low retries below,
        # worst-case wait is bounded to ~ (timeout * (retries + 1)).
        self.client = OpenAI(
            api_key=key,
            base_url=settings.deepseek_base_url,
            timeout=timeout,
            max_retries=0,  # we handle retries ourselves with backoff
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        retries: int = 1,
    ) -> str:
        """Call a DeepSeek text or multimodal model.

        Default ``retries=1`` keeps the worst-case latency under ~60s
        (timeout * 2) so the Streamlit UI doesn't stay frozen on slow APIs.
        """
        model_name = model or runtime_chat_model()
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"DeepSeek 调用失败：{last_error}")

    @staticmethod
    def encode_image(image: Image.Image, max_side: int = 1600) -> str:
        """Compress and base64-encode an image."""
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            image = image.resize((int(w * scale), int(h * scale)))
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def vision_extract_table(self, image: Image.Image, prompt: str | None = None) -> str:
        """Ask DeepSeek-VL to extract text from an image.

        The method name is kept for compatibility with the original base,
        while LitVerify passes citation-specific prompts by default.
        """
        b64 = self.encode_image(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt or OCR_CITATION_PROMPT},
                ],
            }
        ]
        return self.chat(messages=messages, model=runtime_vl_model(), temperature=0.1, max_tokens=4096)
