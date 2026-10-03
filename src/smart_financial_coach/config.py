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
    # GitHub repository ("owner/name") whose releases hold promoted model files. None: the
    # repository of the current checkout
    model_release_repo: str | None = None

    # Experiment tracking (evaluation only; serving never reads it). None: local store in mlruns/
    mlflow_tracking_uri: str | None = None

    # LLM for the coach (Anthropic; Web App UI decision 6). No key: chat says it's unavailable
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "low"  # chat is a path the user waits on (NFR-5: answer < 8 s at p95)
    llm_api_key: SecretStr | None = Field(default=None, repr=False)

    # Web app (Web App UI note). The demo bundle holds the dataset, predictions and accounts the
    # app serves, read-only (`sfc-web build-demo`)
    demo_dir: Path = PROJECT_ROOT / "build" / "demo"
    # The shared demo password and the session-cookie signing key. Both are required to serve;
    # `sfc-web serve --dev` makes throwaway ones for a local run
    demo_password: SecretStr | None = Field(default=None, repr=False)
    session_secret: SecretStr | None = Field(default=None, repr=False)
    quick_signin: bool = False  # one-click "Continue as …": local only, never the hosted demo
    secure_cookies: bool = True  # HTTPS-only session cookie; off only for local http
    coach_name: str = "Wren"
    chat_messages_per_hour: int = 30  # per signed-in session; caps LLM spend (NFR-9)
    signin_attempts_per_minute: int = 10  # per client address

    log_level: str = "INFO"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
