"""Disk-cache helpers for the external API clients.

`make_disk_cache` is used by CrossRef / OpenAlex / arXiv clients to memoize
slow HTTP lookups. When `diskcache` is unavailable we fall back to a tiny
in-memory dict-like cache so the app still works offline.
"""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SimpleTTLCache:
    """Tiny dict-like fallback used when diskcache is unavailable."""

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __setitem__(self, key: str, value: object) -> None:
        self._data[key] = value

    def set(self, key: str, value: object, expire: int | None = None) -> None:
        _ = expire
        self._data[key] = value


def make_disk_cache(path: str):
    """Create a diskcache Cache, falling back to an in-memory cache."""
    cache_path = Path(path)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    try:
        from diskcache import Cache

        return Cache(cache_path)
    except Exception:  # pragma: no cover - depends on optional dependency
        return SimpleTTLCache()
