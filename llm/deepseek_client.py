"""DeepSeek client using the OpenAI-compatible API."""
from __future__ import annotations

import time
from typing import Any, Iterator

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
        top_p: float | None = None,
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
                request: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if top_p is not None:
                    request["top_p"] = top_p
                response = self.client.chat.completions.create(
                    **request,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"DeepSeek 调用失败：{last_error}")

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        top_p: float | None = None,
        max_tokens: int = 2048,
    ) -> Iterator[tuple[str, str]]:
        """Stream a DeepSeek chat completion as ``(channel, text)`` deltas.

        ``channel`` is ``"reasoning"`` (the model's thinking, from
        ``reasoning_content``) or ``"answer"`` (the final reply, from
        ``content``). Both are surfaced so the UI can show live motion during
        the long thinking phase of deepseek-v4-pro instead of a frozen
        "thinking…" placeholder — without that, the answer only appears once
        reasoning finishes and *looks* like a non-streamed dump.

        Streaming is also what fixes the "Request timed out" failures: a
        non-streaming call must produce the whole answer inside one timeout
        window, but here every chunk resets the per-read budget, so the slow
        thinking phase keeps the connection alive.
        """
        model_name = model or runtime_chat_model()
        request: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if top_p is not None:
            request["top_p"] = top_p
        stream = self.client.chat.completions.create(**request)
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ("reasoning", reasoning)
            content = getattr(delta, "content", None)
            if content:
                yield ("answer", content)
