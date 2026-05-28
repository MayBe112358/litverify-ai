"""Shared helpers for telling 'database said no' apart from 'API unreachable'.

A genuine *miss* (HTTP 404 / empty result) is real evidence and gets cached.
A *transient* failure (timeout, connection reset, 5xx) is NOT evidence — we
must not cache it, and the rule engine should stay neutral instead of pushing
a real citation toward FAKE just because the network blipped.
"""
from __future__ import annotations

import requests


class LookupUnavailable(RuntimeError):
    """Raised when an external source can't be reached (vs. returning a miss)."""


def is_transient_error(exc: BaseException) -> bool:
    """True for network/5xx errors worth retrying; False for a real 404 miss."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return isinstance(exc, requests.RequestException)
