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
    # Essentials: the bare minimum to live on, shown apart from everything else in the money-flow
    # chart (Web App UI, decision 3). A display grouping only; names must be taxonomy categories,
    # checked at startup. Env: a JSON list, e.g. SFC_ESSENTIALS='["Housing", "Groceries"]'
    essentials: tuple[str, ...] = (
        "Housing",
        "Utilities",
        "Groceries",
        "Insurance & Fees",
        "Childcare & Education",
    )
    # Rate limits per client address, and across all visitors as a backstop. The chat total is the
    # bound on LLM spend (NFR-9): size it to the API key's spending cap
    chat_messages_per_hour: int = 30
    chat_messages_per_hour_total: int = 200
    signin_attempts_per_minute: int = 10
    signin_attempts_per_minute_total: int = 60
    # The app's public address, for the MCP server's URL and auth metadata (FR-19), and the
    # lifetime of the personal access tokens the "Connect an assistant" page hands out
    public_url: str = "http://127.0.0.1:8000"
    mcp_token_days: int = 7
    # Proxies in front of the app that append to X-Forwarded-For: 1 behind Azure Container Apps'
    # ingress, 0 when serving directly (the header is then ignored)
    trusted_proxy_hops: int = 0

    log_level: str = "INFO"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / self.sqlite_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
