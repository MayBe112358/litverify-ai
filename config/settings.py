"""Global configuration loading for environment and runtime settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - depends on environment
    load_dotenv = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Application settings read from environment variables."""

    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # 默认走 DeepSeek V4 Pro。
    # 侧边栏「设置」里可在 Pro / Flash 之间切换：
    #   Pro   = deepseek-v4-pro   （更聪明，深度推理 / 智能洞察）
    #   Flash = deepseek-v4-flash （更快、更便宜，演示流畅）
    chat_model: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-v4-pro")
    vl_model: str = os.getenv("DEEPSEEK_VL_MODEL", "deepseek-v4-pro")

    # Pro / Flash 别名 → 实际模型名的映射。.env 可覆盖。
    deepseek_pro_model: str = os.getenv("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
    deepseek_flash_model: str = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")

    app_name: str = os.getenv("APP_NAME", "LitVerify AI")
    # Files larger than this switch from pandas to the DuckDB streaming reader.
    duckdb_threshold_mb: int = int(os.getenv("DUCKDB_THRESHOLD_MB", "100"))
    crossref_email: str = os.getenv("CROSSREF_EMAIL", "")
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "")
    api_timeout: int = int(os.getenv("API_TIMEOUT", "15"))
    real_threshold: int = int(os.getenv("REAL_THRESHOLD", "80"))
    suspicious_threshold: int = int(os.getenv("SUSPICIOUS_THRESHOLD", "40"))
    history_db_path: str = os.getenv(
        "HISTORY_DB_PATH",
        str(PROJECT_ROOT / "data" / "litverify_history.sqlite3"),
    )

    @property
    def default_thresholds(self) -> dict[str, int]:
        """Default verdict thresholds used by the rule engine."""
        return {"real": self.real_threshold, "suspicious": self.suspicious_threshold}


settings = Settings()
