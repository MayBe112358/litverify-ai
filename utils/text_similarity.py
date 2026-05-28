"""Text similarity helpers for citation field matching."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

try:  # pragma: no cover - optional acceleration
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None  # type: ignore[assignment]


def normalize_text(value: str | None) -> str:
    """Lowercase text and remove punctuation-like noise."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def title_similarity(a: str | None, b: str | None) -> float:
    """Return a 0-1 title similarity score."""
    left = normalize_text(a)
    right = normalize_text(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if fuzz is not None:
        return fuzz.token_sort_ratio(left, right) / 100
    return SequenceMatcher(None, " ".join(sorted(left.split())), " ".join(sorted(right.split()))).ratio()


def _surname_token(name: str | None) -> str:
    """Return the most-likely surname token (last token with length >= 2).

    Works across styles:
    - "Harris C R"            → "harris"
    - "Charles R. Harris"     → "harris"
    - "van der Walt S J"      → "walt"
    - "Stéfan J. van der Walt"→ "walt"
    """
    tokens = [t for t in normalize_text(name).split() if len(t) >= 2]
    return tokens[-1] if tokens else ""


def author_overlap(a_list: list[str] | None, b_list: list[str] | None) -> float:
    """How many of the user-listed surnames also appear in the authority record.

    Uses *coverage of the user's surnames* instead of Jaccard, because users
    routinely write "X, Y, Z, et al." while CrossRef returns the full author
    list (often 10+). A pure Jaccard would always score low even on
    perfectly-matched references.
    """
    a_surnames = {_surname_token(n) for n in a_list or [] if _surname_token(n)}
    b_surnames = {_surname_token(n) for n in b_list or [] if _surname_token(n)}
    a_surnames.discard("")
    b_surnames.discard("")
    if not a_surnames or not b_surnames:
        return 0.0
    return len(a_surnames & b_surnames) / len(a_surnames)


def compact_match(a: str | None, b: str | None) -> bool:
    """Loose equality for venue, volume, issue and page values."""
    left = normalize_text(a)
    right = normalize_text(b)
    return bool(left and right and (left == right or left in right or right in left))
