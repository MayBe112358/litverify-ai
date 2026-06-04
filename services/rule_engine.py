"""Weighted citation verification rule engine."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT, settings
from utils.doi_utils import looks_like_doi, normalize_doi
from utils.text_similarity import author_overlap, compact_match, title_similarity

try:  # pragma: no cover - optional config dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass
class Citation:
    """Structured citation flowing through the whole LitVerify pipeline."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    type: str = "unknown"
    citation_count: int | None = None
    source: str | None = None
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize citation for UI, DB and prompts."""
        return asdict(self)


@dataclass
class RuleConfig:
    """Configurable rule metadata."""

    id: str
    name: str
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    """Single rule result."""

    id: str
    name: str
    passed: bool
    score: float
    weight: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationEvidence:
    """External evidence gathered for a citation.

    ``lookup_failed`` is True when at least one source couldn't be reached
    (timeout / 5xx) rather than returning a clean miss — the rule engine
    uses it to stay neutral instead of penalising a possible outage.
    """

    crossref: Citation | None = None
    openalex: Citation | None = None
    arxiv: Citation | None = None
    pubmed: Citation | None = None
    semantic_scholar: Citation | None = None
    dblp: Citation | None = None
    wanfang: Citation | None = None
    datacite: Citation | None = None
    doidb: Citation | None = None
    lookup_failed: bool = False

    def records(self) -> list[Citation]:
        """Return all available external records in source-priority order."""
        return [
            record
            for record in (
                self.crossref,
                self.openalex,
                self.pubmed,
                self.semantic_scholar,
                self.dblp,
                self.wanfang,
                self.datacite,
                self.arxiv,
                self.doidb,
            )
            if record
        ]

    def best_record(self) -> Citation | None:
        """Pick the strongest available authority record.

        Metadata-rich records are preferred over resolver-only evidence such
        as DOIDB, so title/author/venue rules see the best comparison target.
        """
        for record in self.records():
            if record.title or record.authors or record.venue:
                return record
        return self.records()[0] if self.records() else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "crossref": self.crossref.to_dict() if self.crossref else None,
            "openalex": self.openalex.to_dict() if self.openalex else None,
            "arxiv": self.arxiv.to_dict() if self.arxiv else None,
            "pubmed": self.pubmed.to_dict() if self.pubmed else None,
            "semantic_scholar": (
                self.semantic_scholar.to_dict() if self.semantic_scholar else None
            ),
            "dblp": self.dblp.to_dict() if self.dblp else None,
            "wanfang": self.wanfang.to_dict() if self.wanfang else None,
            "datacite": self.datacite.to_dict() if self.datacite else None,
            "doidb": self.doidb.to_dict() if self.doidb else None,
            "lookup_failed": self.lookup_failed,
        }


@dataclass
class VerificationReport:
    """End-to-end verification report."""

    user_citation: Citation
    evidence: VerificationEvidence
    rule_results: list[RuleResult]
    overall_score: float
    verdict: str
    suggestions: list[str] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_citation": self.user_citation.to_dict(),
            "evidence": self.evidence.to_dict(),
            "rule_results": [r.to_dict() for r in self.rule_results],
            "overall_score": self.overall_score,
            "verdict": self.verdict,
            "suggestions": self.suggestions,
            "explanation": self.explanation,
        }


DEFAULT_RULES = [
    RuleConfig("doi_format", "DOI 格式校验", True, 10, {}),
    RuleConfig("doi_resolve", "DOI 在外部 DOI/学术库可解析", True, 20, {}),
    RuleConfig("title_match", "标题与权威库一致", True, 18, {"min_similarity": 0.85}),
    RuleConfig("author_match", "作者列表一致", True, 12, {"min_overlap": 0.70}),
    RuleConfig("venue_match", "期刊/会议存在且一致", True, 10, {}),
    RuleConfig("year_reasonable", "年份合理", True, 6, {"min_year": 1900, "max_year": 2026}),
    RuleConfig("page_volume_consistent", "卷/期/页一致", True, 6, {}),
    RuleConfig("arxiv_resolve", "ArXiv ID 可解析", True, 8, {}),
    RuleConfig("cross_db_consistent", "多源外部库一致", True, 6, {}),
    # 纯本地元数据校验（无需外部 API，对齐赛题任务二的"元数据格式 / 语义一致性"维度）
    RuleConfig("author_format", "作者姓名格式合理", True, 4, {}),
    RuleConfig("venue_format", "期刊/会议名称合理", True, 4, {}),
    RuleConfig("title_quality", "标题长度与字符合理", True, 4, {"min_len": 6, "max_len": 300}),
]

# Academic venue fingerprints for the local venue check (zh + en).
_VENUE_KEYWORDS = (
    "journal", "conference", "proceedings", "transactions", "review",
    "letters", "annals", "bulletin", "symposium", "advances", "nature",
    "science", "学报", "大学", "会议", "研究", "通报", "科学", "学刊", "论坛",
)


def _no_record(evidence: VerificationEvidence) -> tuple[float, str]:
    """Score + note for rules that need an authority record but have none.

    Distinguishes a real miss (harsh) from a transient outage (neutral) so
    a flaky network never pushes a genuine citation toward FAKE."""
    if evidence.lookup_failed:
        return 0.4, "外部库暂时不可用"
    return 0.1, "权威库未命中"


def _looks_like_name(value: str) -> bool:
    value = value.strip()
    if not (2 <= len(value) <= 60):
        return False
    return not re.search(r"https?://|@|\d|/", value)


def _looks_like_venue(value: str) -> bool:
    value = value.strip()
    return bool(2 <= len(value) <= 100 and not re.search(r"https?://|@|\d{5,}", value))


def load_rule_config(path: str | Path | None = None) -> tuple[list[RuleConfig], dict[str, int]]:
    """Load rules from YAML; fallback to bundled defaults."""
    config_path = Path(path or PROJECT_ROOT / "config" / "rules_default.yaml")
    thresholds = settings.default_thresholds
    if yaml is None or not config_path.exists():
        return DEFAULT_RULES, thresholds
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    rules = [
        RuleConfig(
            id=item["id"],
            name=item.get("name", item["id"]),
            enabled=bool(item.get("enabled", True)),
            weight=float(item.get("weight", 1)),
            params=dict(item.get("params") or {}),
        )
        for item in data.get("rules", [])
    ]
    return rules or DEFAULT_RULES, dict(data.get("thresholds") or thresholds)


class RuleEngine:
    """Apply the weighted rule set (DOI / 外部库 / 本地元数据) and produce a transparent verdict."""

    def __init__(
        self,
        rules: list[RuleConfig] | None = None,
        thresholds: dict[str, int] | None = None,
    ) -> None:
        loaded_rules, loaded_thresholds = load_rule_config()
        self.rules = rules or loaded_rules
        self.thresholds = thresholds or loaded_thresholds

    def evaluate(self, user: Citation, evidence: VerificationEvidence) -> VerificationReport:
        """Evaluate all enabled rules.

        When the user-supplied DOI resolves to an authoritative record we
        backfill any *missing* fields on a derived citation so the
        downstream rules see a complete record — values the user actually
        typed are never overwritten. The report still echoes the original
        ``user`` citation so the UI keeps showing what the user submitted.
        """
        effective = self._effective_user(user, evidence)
        results = [self._evaluate_rule(rule, effective, evidence) for rule in self.rules if rule.enabled]
        weight_sum = sum(max(result.weight, 0) for result in results) or 1
        overall = round(sum(result.score * result.weight for result in results) / weight_sum * 100, 2)
        if overall >= self.thresholds.get("real", 80):
            verdict = "REAL"
        elif overall >= self.thresholds.get("suspicious", 40):
            verdict = "SUSPICIOUS"
        else:
            verdict = "FAKE"
        return VerificationReport(user, evidence, results, overall, verdict)

    @staticmethod
    def _effective_user(user: Citation, evidence: VerificationEvidence) -> Citation:
        """Return ``user`` (or a backfilled copy) for rule evaluation."""
        if not user.doi:
            return user
        target = normalize_doi(user.doi)
        record: Citation | None = None
        for cand in evidence.records():
            if cand and cand.doi and normalize_doi(cand.doi) == target:
                record = cand
                break
        if record is None:
            return user
        overrides: dict[str, Any] = {}
        for field_name in ("title", "year", "venue", "volume", "issue", "pages"):
            if not getattr(user, field_name) and getattr(record, field_name):
                overrides[field_name] = getattr(record, field_name)
        if not user.authors and record.authors:
            overrides["authors"] = list(record.authors)
        return replace(user, **overrides) if overrides else user

    def _evaluate_rule(
        self,
        rule: RuleConfig,
        user: Citation,
        evidence: VerificationEvidence,
    ) -> RuleResult:
        method = getattr(self, f"_rule_{rule.id}", None)
        if method is None:
            return RuleResult(rule.id, rule.name, True, 1.0, rule.weight, "规则未实现，按通过处理。")
        score, reason = method(user, evidence, rule.params)
        score = max(0.0, min(1.0, float(score)))
        return RuleResult(rule.id, rule.name, score >= 0.8, score, rule.weight, reason)

    @staticmethod
    def _rule_doi_format(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = evidence, params
        if not user.doi:
            return 0.45, "未提供 DOI，无法进行 DOI 格式核验。"
        if looks_like_doi(user.doi):
            return 1.0, f"DOI 格式有效：{normalize_doi(user.doi)}。"
        return 0.0, f"DOI 格式不符合规范：{user.doi}。"

    @staticmethod
    def _rule_doi_resolve(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = params
        if not user.doi:
            return 0.45, "未提供 DOI，降级使用标题和字段匹配。"
        normalized = normalize_doi(user.doi)
        hits = [source for source in evidence.records() if normalize_doi(source.doi) == normalized]
        if hits:
            names = "、".join(source.source or "权威库" for source in hits)
            return 1.0, f"DOI 可在 {names} 解析。"
        return 0.0, "给定 DOI 未在已启用外部 DOI/学术库命中。"

    @staticmethod
    def _rule_title_match(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        record = evidence.best_record()
        if not record:
            score, note = _no_record(evidence)
            return score, f"{note}，无法比对标题。"
        if not user.title or not record.title:
            return 0.3, "缺少标题字段，标题匹配证据不足。"
        score = title_similarity(user.title, record.title)
        threshold = float(params.get("min_similarity", 0.85))
        return score, f"标题相似度 {score:.2f}（阈值 {threshold:.2f}）。"

    @staticmethod
    def _rule_author_match(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        record = evidence.best_record()
        if not record:
            score, note = _no_record(evidence)
            return score, f"{note}，无法比对作者。"
        if not user.authors or not record.authors:
            return 0.35, "缺少作者列表，作者匹配证据不足。"
        score = author_overlap(user.authors, record.authors)
        threshold = float(params.get("min_overlap", 0.7))
        return score, f"作者重合度 {score:.2f}（阈值 {threshold:.2f}）。"

    @staticmethod
    def _rule_venue_match(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = params
        record = evidence.best_record()
        if not record:
            score, note = _no_record(evidence)
            return score, f"{note}，无法核验来源期刊/会议。"
        if not record.venue:
            return 0.35, "权威库未返回期刊/会议来源。"
        if not user.venue:
            return 0.6, f"权威库来源存在：{record.venue}；用户引用未写明来源。"
        if compact_match(user.venue, record.venue):
            return 1.0, f"来源匹配：{record.venue}。"
        return 0.2, f"来源不一致：用户为 {user.venue}，权威库为 {record.venue}。"

    @staticmethod
    def _rule_year_reasonable(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        record = evidence.best_record()
        lo = int(params.get("min_year", 1900))
        hi = int(params.get("max_year", 2026))
        if user.year and (lo <= user.year <= hi):
            if record and record.year and user.year != record.year:
                return 0.35, f"年份合理但与权威库不一致：用户 {user.year}，权威库 {record.year}。"
            return 1.0, f"年份 {user.year} 在合理范围（{lo}-{hi}）内。"
        return 0.2, f"年份缺失或超出合理范围（{lo}-{hi}）。"

    @staticmethod
    def _rule_page_volume_consistent(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = params
        record = evidence.best_record()
        if not record:
            score, note = _no_record(evidence)
            return score, f"{note}，无法核验卷/期/页。"
        checks: list[float] = []
        labels = []
        for field_name, label in (("volume", "卷"), ("issue", "期"), ("pages", "页码")):
            left = getattr(user, field_name)
            right = getattr(record, field_name)
            if left and right:
                checks.append(1.0 if compact_match(left, right) else 0.0)
                labels.append(f"{label}:{left}/{right}")
        if not checks:
            return 0.55, "卷/期/页信息不足，未发现直接冲突。"
        score = sum(checks) / len(checks)
        return score, "；".join(labels)

    @staticmethod
    def _rule_arxiv_resolve(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = params
        if not user.arxiv_id:
            return 0.75, "未提供 arXiv ID，本规则不作为强负证据。"
        if evidence.arxiv:
            return 1.0, f"arXiv ID 可解析：{user.arxiv_id}。"
        return 0.0, f"arXiv ID 未解析成功：{user.arxiv_id}。"

    @staticmethod
    def _rule_cross_db_consistent(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = user, params
        records = [record for record in evidence.records() if record.title or record.doi]
        if not records:
            return _no_record(evidence)[0], "所有已启用外部库均未命中，缺少跨库证据。"
        if len(records) == 1:
            record = records[0]
            if record.source == "arXiv":
                return 0.7, "arXiv 预印本，跨库比对证据有限。"
            return 0.6, f"仅 {record.source or '一个外部库'} 命中，无法做强交叉比对。"

        title_scores: list[float] = []
        doi_checks: list[bool] = []
        for i, left in enumerate(records):
            for right in records[i + 1:]:
                if left.title and right.title:
                    title_scores.append(title_similarity(left.title, right.title))
                if left.doi and right.doi:
                    doi_checks.append(normalize_doi(left.doi) == normalize_doi(right.doi))
        avg_title = sum(title_scores) / len(title_scores) if title_scores else 0.6
        doi_score = 1.0 if doi_checks and all(doi_checks) else (0.6 if not doi_checks else 0.0)
        score = max(avg_title, doi_score) if doi_checks else avg_title
        sources = "、".join(record.source or "外部库" for record in records[:5])
        return score, f"命中来源：{sources}；跨库标题均值 {avg_title:.2f}，DOI {'一致' if doi_score == 1.0 else '证据不足或不一致'}。"

    # ----- local metadata rules (no external API) ----- #
    @staticmethod
    def _rule_author_format(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = evidence, params
        if not user.authors:
            return 0.5, "未提供作者，无法校验作者格式。"
        bad = [a for a in user.authors if not _looks_like_name(a)]
        if not bad:
            return 1.0, f"作者格式正常（{len(user.authors)} 位）。"
        return max(0.0, 1 - len(bad) / len(user.authors)), f"疑似异常作者：{'、'.join(bad[:3])}。"

    @staticmethod
    def _rule_venue_format(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = evidence, params
        if not user.venue:
            return 0.5, "未提供期刊/会议名称。"
        venue = user.venue.strip()
        if any(k in venue.lower() for k in _VENUE_KEYWORDS):
            return 1.0, f"含学术期刊/会议关键词：{venue}。"
        if _looks_like_venue(venue):
            return 0.8, f"期刊/会议名格式合理：{venue}。"
        return 0.2, f"期刊/会议名可疑：{venue}。"

    @staticmethod
    def _rule_title_quality(user: Citation, evidence: VerificationEvidence, params: dict[str, Any]) -> tuple[float, str]:
        _ = evidence
        if not user.title:
            return 0.4, "未提供标题。"
        title = user.title.strip()
        n = len(title)
        if n < int(params.get("min_len", 6)) or n > int(params.get("max_len", 300)):
            return 0.2, f"标题长度异常（{n} 字符）。"
        allowed = set("-:,.()'\"，。：、（）【】《》 ")
        junk = sum(1 for ch in title if not (ch.isalnum() or ch in allowed))
        if junk / n > 0.3:
            return 0.2, "标题包含较多无意义字符。"
        return 1.0, "标题长度与字符正常。"
