"""Application settings, loaded from `SFC_*` environment variables and an optional `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SFC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    sqlite_filename: str = "sfc.sqlite3"

    # Experiment tracking (evaluation only; serving never reads it). None: local store in mlruns/
    mlflow_tracking_uri: str | None = None

    # Promotion gate for single-transaction categorization latency (p95, cold start). 5 ms is the
    # FR-3 target on a laptop CPU; slower machines (CI runners) set their own, like the generator's
    # SFC_RUNTIME_BUDGET_SECONDS. Check it on the serving hardware before relying on it.
    latency_gate_ms: float = 5.0

    # LLM (provider TBD, see Technical Design open questions)
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = Field(default=None, repr=False)

    log_level: str = "INFO"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
