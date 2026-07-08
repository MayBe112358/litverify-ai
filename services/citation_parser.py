"""Citation parser for APA, GB/T 7714, IEEE and free-form references."""
from __future__ import annotations

import json
import re
from typing import Any

from config.prompts import CITATION_PARSE_PROMPT
from llm.deepseek_client import DeepSeekClient
from llm.text_utils import strip_fenced_block
from services.rule_engine import Citation
from utils.doi_utils import extract_arxiv, extract_doi, extract_url


YEAR_PATTERN = re.compile(r"\b(18|19|20|21)\d{2}\b")
QUOTED_TITLE_PATTERN = re.compile(r"[\"“](.*?)[\"”]")
# "Harris C R", "Millman K J", "Vaswani A" 这种"姓 + 首字母缩写" 出现两次以上的句段判定为作者块
_AUTHOR_INITIALS_PATTERN = re.compile(r"\b[A-Z][A-Za-z\-]+(?:\s+[A-Z]\.?){1,3}\b")
_AUTHOR_HINT_TOKENS = ("et al", " et al.", "等,", "等.", "等，")
# Chinese punctuation splitters used for both authors and title/venue chunks.
_AUTHOR_SPLIT_PATTERN = re.compile(r"\s*,\s*|\s+and\s+|、|；|;|，")
_TITLE_SPLIT_PATTERN = re.compile(r"\.\s+|\s-\s|。\s*|\.\s*(?=[A-Z一-鿿])")
# A 4-digit year stripping pattern guarded so it doesn't eat the numeric tail
# of a DOI like "10.11897/COMPJ.2025.0012" — only strip when surrounded by
# whitespace or sentence-style punctuation, never when preceded by ``.`` or ``/``.
_PREFIX_YEAR_STRIP = re.compile(r"(?<![./])\b(18|19|20|21)\d{2}\b(?![./])")


class CitationParser:
    """Parse raw citation strings into a normalized Citation dataclass."""

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self.use_llm_fallback = use_llm_fallback

    def parse(self, raw_text: str) -> Citation:
        """Parse citation text with deterministic heuristics and optional LLM fallback."""
        text = " ".join(str(raw_text or "").split())
        citation = self._heuristic_parse(text)
        if (
            self.use_llm_fallback
            and self._needs_llm(citation)
            and self._has_parseable_text(text)
        ):
            llm = self._llm_parse(text)
            if llm:
                citation = self._merge(citation, llm)
        return citation

    @staticmethod
    def _has_parseable_text(text: str) -> bool:
        """False when the input is essentially just a DOI / arXiv id / URL.

        此时文本里没有任何可供提取的标题、作者信息，再调 LLM 只会诱导模型
        凭记忆"补全"元数据——幻觉出来的标题会污染后续规则比对，把一条真实
        DOI 拉成可疑。这类输入直接走 DOI/ID 检索即可。"""
        residue = re.sub(r"(?i)https?://\S+", " ", text)
        residue = re.sub(r"(?i)\b10\.\d{4,9}/\S+", " ", residue)
        residue = re.sub(r"(?i)\barxiv\s*[:：]?\s*\d{4}\.\d{4,5}(v\d+)?", " ", residue)
        residue = re.sub(r"(?i)\b(doi|arxiv|url)\b\s*[:：]?", " ", residue)
        return len(re.findall(r"[A-Za-z一-鿿]", residue)) >= 4

    def _heuristic_parse(self, text: str) -> Citation:
        doi = extract_doi(text)
        # ``extract_arxiv`` is self-guarding: labeled forms (arXiv:/arXiv./URL,
        # incl. the 10.48550 DOI) are matched first, and the bare-number
        # fallback won't grab a DOI tail like ``2025.0012``.
        arxiv_id = extract_arxiv(text)
        url = extract_url(text)
        year_match = YEAR_PATTERN.search(text)
        year = int(year_match.group(0)) if year_match else None
        title = self._extract_title(text, year)
        authors = self._extract_authors(text, title)
        venue = self._extract_venue(text, title, year)
        volume, issue, pages = self._extract_biblio_numbers(text)
        ctype = "preprint" if arxiv_id else "article"
        return Citation(
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            arxiv_id=arxiv_id,
            url=url,
            type=ctype,
            raw=text,
        )

    @staticmethod
    def _looks_like_author_block(candidate: str) -> bool:
        """Heuristics to detect a chunk that's actually the author list, not the title.

        - Contains explicit hints ("et al"，"等,"，"等.")
        - Contains 2+ "Surname X Y" patterns (GB/T 7714 风格的姓+首字母)
        - Is short (<= 10 words) and comma-heavy
        - Single-author short string like "Smith J" / "Vaswani A B"
        - 2+ short Chinese name segments separated by 、 / ，
        """
        lower = candidate.lower()
        if any(hint in lower for hint in _AUTHOR_HINT_TOKENS):
            return True
        if len(_AUTHOR_INITIALS_PATTERN.findall(candidate)) >= 2:
            return True
        if "," in candidate and len(candidate.split()) <= 10 and candidate.count(",") >= 2:
            return True
        # Single-author latin pattern: "Surname X" / "Surname X Y" / "Surname X. Y."
        # — a short fragment (<= 4 tokens) that fully matches the initials regex.
        if (
            len(candidate.split()) <= 4
            and _AUTHOR_INITIALS_PATTERN.fullmatch(candidate.strip())
        ):
            return True
        # Chinese: "王宇轩，李泽峰，陈星燃" — 2+ segments of 2-4 CJK chars
        cn_segments = re.findall(r"[一-龥]{2,4}(?=[，、,；;]|$)", candidate)
        if len(cn_segments) >= 2 and len(candidate) <= 40:
            return True
        return False

    @staticmethod
    def _title_word_count(text: str) -> int:
        """Approximate token count that also handles CJK-only titles."""
        latin = len(re.findall(r"[A-Za-z][A-Za-z\-]*", text))
        cjk = len(re.findall(r"[一-龥]", text))
        # Each CJK character roughly equals one word for length checks.
        return latin + cjk

    @classmethod
    def _extract_title(cls, text: str, year: int | None) -> str | None:
        quoted = QUOTED_TITLE_PATTERN.search(text)
        if quoted and quoted.group(1).strip():
            return quoted.group(1).strip(" .")
        cleaned = re.sub(r"(doi|DOI)\s*:\s*\S+", "", text)
        if year:
            cleaned = cleaned.replace(str(year), f"{year}.")
        parts = [p.strip(" .[]") for p in _TITLE_SPLIT_PATTERN.split(cleaned) if p.strip(" .[]")]
        candidates: list[str] = []
        for p in parts:
            if cls._title_word_count(p) < 2:
                continue
            if p.lower().startswith(("doi", "http")):
                continue
            if year and str(year) in p:
                continue
            if cls._looks_like_author_block(p):
                continue
            candidates.append(p)
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _extract_authors(text: str, title: str | None) -> list[str]:
        prefix = text.split(title, 1)[0] if title and title in text else text[:120]
        # Strip parenthesized year "(2024)" and bare years; DOI fragments like
        # "10.1038/nature14539" stay intact thanks to ``_PREFIX_YEAR_STRIP``.
        prefix = re.sub(r"\(\d{4}\)", "", prefix)
        prefix = _PREFIX_YEAR_STRIP.sub("", prefix)
        prefix = prefix.strip(" .[]")
        if not prefix:
            return []
        prefix = re.sub(r"\bet al\.?", "", prefix, flags=re.IGNORECASE)
        prefix = re.sub(r"\b等[,\.，。]?$", "", prefix)
        raw_parts = _AUTHOR_SPLIT_PATTERN.split(prefix)
        cleaned = []
        for part in raw_parts:
            value = part.strip(" .")
            if not value or len(value) > 60:
                continue
            # Drop fragments that obviously aren't names — DOIs, URLs, anything
            # that already starts a metadata key.
            if value.lower().startswith(("doi", "http", "arxiv", "pp", "vol", "no.")):
                continue
            if "/" in value or value.count(".") >= 2:
                continue
            cleaned.append(value)
        return cleaned[:12]

    @staticmethod
    def _extract_venue(text: str, title: str | None, year: int | None) -> str | None:
        after = text
        if title and title in text:
            after = text.split(title, 1)[1]
        if year and str(year) in after:
            after = after.split(str(year), 1)[0] + after.split(str(year), 1)[-1]
        after = re.sub(r"(doi|DOI)\s*:\s*\S+|https?://\S+", "", after).strip(" .")
        parts = [p.strip(" .,，。") for p in re.split(r"\.\s+|,\s+|，|。", after) if p.strip(" .,，。")]
        venue_candidates = [
            p
            for p in parts
            if not re.fullmatch(r"\d+(\(\d+\))?(:\s*)?\d+[-–]\d+", p)
            and len(p) >= 2
            and not p.lower().startswith(("vol", "no", "pp"))
        ]
        return venue_candidates[0] if venue_candidates else None

    @staticmethod
    def _extract_biblio_numbers(text: str) -> tuple[str | None, str | None, str | None]:
        volume = issue = pages = None
        vol_issue = re.search(r"\b(\d{1,4})\s*\((\d{1,4})\)", text)
        if vol_issue:
            volume, issue = vol_issue.group(1), vol_issue.group(2)
        if not volume:
            m = re.search(r"vol(?:ume)?\.?\s*(\d{1,4})", text, re.IGNORECASE)
            if m:
                volume = m.group(1)
            else:
                m = re.search(r"卷\s*(\d{1,4})", text)
                if m:
                    volume = m.group(1)
        if not issue:
            m = re.search(r"\b(?:no\.?|issue)\s*(\d{1,4})", text, re.IGNORECASE)
            if m:
                issue = m.group(1)
            else:
                m = re.search(r"期\s*(\d{1,4})", text)
                if m:
                    issue = m.group(1)
        pages_match = re.search(r"\b(?:pp\.?\s*)?(\d{1,6}\s*[-–]\s*\d{1,6})\b", text, re.IGNORECASE)
        if pages_match:
            pages = pages_match.group(1).replace(" ", "").replace("–", "-")
        return volume, issue, pages

    @staticmethod
    def _needs_llm(citation: Citation) -> bool:
        return not citation.title or (not citation.authors and not citation.doi and not citation.arxiv_id)

    @staticmethod
    def _merge(primary: Citation, fallback: Citation) -> Citation:
        for field_name in (
            "title",
            "year",
            "venue",
            "volume",
            "issue",
            "pages",
            "doi",
            "arxiv_id",
            "url",
            "type",
        ):
            if not getattr(primary, field_name) and getattr(fallback, field_name):
                setattr(primary, field_name, getattr(fallback, field_name))
        if not primary.authors and fallback.authors:
            primary.authors = fallback.authors
        return primary

    @staticmethod
    def _grounded(text: str, value: Any) -> str | None:
        """Return ``value`` only when it literally appears in the raw input.

        LLM 解析只被允许"摘录"，不允许"补全"——凡是原文里找不到的标题/
        作者/期刊一律丢弃，防止模型根据 DOI 或记忆联想出不存在的字段。
        比对时去掉大小写与全部标点/空白差异。"""
        if not value:
            return None
        norm = lambda s: re.sub(r"[\W_]+", "", str(s)).lower()  # noqa: E731
        candidate = norm(value)
        return str(value) if candidate and candidate in norm(text) else None

    @classmethod
    def _llm_parse(cls, text: str) -> Citation | None:
        try:
            client = DeepSeekClient(timeout=20)
            raw = client.chat(
                messages=[
                    {"role": "system", "content": CITATION_PARSE_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                # 思考型模型的思考 token 也计入 max_tokens，预算需留足
                max_tokens=2200,
                retries=0,
            )
            data: dict[str, Any] = json.loads(strip_fenced_block(raw, "json"))
            return Citation(
                title=cls._grounded(text, data.get("title")),
                authors=[
                    str(x) for x in data.get("authors") or [] if cls._grounded(text, x)
                ],
                year=int(data["year"]) if data.get("year") else None,
                venue=cls._grounded(text, data.get("venue")),
                volume=data.get("volume"),
                issue=data.get("issue"),
                pages=data.get("pages"),
                doi=extract_doi(data.get("doi") or ""),
                arxiv_id=data.get("arxiv_id"),
                url=data.get("url"),
                type=data.get("type") or "unknown",
                raw=text,
            )
        except Exception:
            return None
