"""Generate natural-language explanations for citation verification."""
from __future__ import annotations

import json
from typing import Any

from config.prompts import VERIFY_EXPLAIN_PROMPT
from llm.deepseek_client import DeepSeekClient
from llm.text_utils import strip_fenced_block
from services.rule_engine import VerificationReport


def local_explanation(report: VerificationReport, error: Exception | None = None) -> dict[str, Any]:
    """Deterministic fallback when DeepSeek is unavailable."""
    failed = [item for item in report.rule_results if not item.passed]
    if report.verdict == "REAL":
        summary = "多源证据和字段规则基本一致，引用可信度较高。"
    elif report.verdict == "SUSPICIOUS":
        summary = "部分关键字段证据不足或存在冲突，建议人工复核。"
    else:
        summary = "关键字段与权威库不一致或无法解析，极可能为虚假引用。"
    reasons = [item.reason for item in failed[:4]] or ["未发现明显冲突。"]
    suggestions = [
        "优先用 DOI 在 CrossRef 或 OpenAlex 重新检索。",
        "核对标题、作者、年份、期刊/会议名是否来自同一条权威记录。",
    ]
    if error:
        reasons.append(f"AI 解读暂不可用，已使用本地解释：{error}")
    return {"summary": summary, "reasons": reasons, "repair_suggestions": suggestions}


def explain_verification(report: VerificationReport) -> dict[str, Any]:
    """Generate a structured explanation for one verification report."""
    try:
        client = DeepSeekClient(timeout=25)
        raw = client.chat(
            messages=[
                {"role": "system", "content": VERIFY_EXPLAIN_PROMPT},
                {"role": "user", "content": json.dumps(report.to_dict(), ensure_ascii=False, default=str)},
            ],
            temperature=0.2,
            # 思考型模型的思考 token 也计入 max_tokens，预算需留足
            max_tokens=2500,
            retries=0,
        )
        return json.loads(strip_fenced_block(raw, "json"))
    except Exception as exc:  # noqa: BLE001
        return local_explanation(report, exc)
