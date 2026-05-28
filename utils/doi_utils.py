"""DOI, arXiv and URL extraction helpers."""
from __future__ import annotations

import re


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
# An arXiv id is either new-style (1706.03762, 2306.12345v2) or old-style
# (astro-ph/0211000); this fragment matches either, used inside the patterns below.
_ARXIV_ID = r"(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
# Labeled forms always carry an explicit "arXiv" token, so they're safe to match
# even inside a DOI: "arXiv:1706.03762", "arXiv.1706.03762" (the 10.48550/arXiv.*
# DOI), or an arxiv.org/abs|pdf URL.
ARXIV_LABELED_PATTERN = re.compile(rf"arxiv[:.\s/]+({_ARXIV_ID})", re.IGNORECASE)
ARXIV_URL_PATTERN = re.compile(rf"arxiv\.org/(?:abs|pdf)/({_ARXIV_ID})", re.IGNORECASE)
# Bare forms have no label, so they need guards to avoid swallowing the numeric
# tail of a DOI (e.g. ``2025.0012`` inside ``10.11897/COMPJ.2025.0012``).
ARXIV_NEW_PATTERN = re.compile(
    r"(?<![./0-9A-Za-z])\d{4}\.\d{4,5}(?:v\d+)?(?![./0-9A-Za-z])",
    re.IGNORECASE,
)
ARXIV_OLD_PATTERN = re.compile(
    r"(?<![./0-9A-Za-z])[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?(?![./0-9A-Za-z])",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s\])>]+", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI text by stripping common URL prefixes and punctuation."""
    if not doi:
        return None
    value = doi.strip().rstrip(".,;)")
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:\s*", "", value, flags=re.IGNORECASE)
    match = DOI_PATTERN.search(value)
    return match.group(0).lower() if match else value.lower()


def extract_doi(text: str | None) -> str | None:
    """Extract the first DOI from free text."""
    if not text:
        return None
    match = DOI_PATTERN.search(text)
    return normalize_doi(match.group(0)) if match else None


def extract_arxiv(text: str | None) -> str | None:
    """Extract an arXiv id from free text.

    Handles ``arXiv:1706.03762``, ``arXiv.1706.03762`` (incl. the
    ``10.48550/arXiv.*`` DOI), ``arxiv.org/abs/...`` URLs, and bare ids —
    while the guarded bare patterns avoid mistaking a DOI tail for an id.
    """
    if not text:
        return None
    for pattern in (ARXIV_LABELED_PATTERN, ARXIV_URL_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(1).lower()
    for pattern in (ARXIV_NEW_PATTERN, ARXIV_OLD_PATTERN):
        match = pattern.search(text)
        if match:
            return match.group(0).lower()
    return None


def extract_url(text: str | None) -> str | None:
    """Extract the first URL from free text."""
    if not text:
        return None
    match = URL_PATTERN.search(text)
    return match.group(0).rstrip(".,;") if match else None


def looks_like_doi(value: str | None) -> bool:
    """Return whether the given value is syntactically a DOI."""
    normalized = normalize_doi(value)
    return bool(normalized and DOI_PATTERN.fullmatch(normalized))
