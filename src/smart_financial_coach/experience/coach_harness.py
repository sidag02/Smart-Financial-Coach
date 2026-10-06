"""The app's MCP server in process, for asking the coach as demo users off the web app.

`sfc-coach ask` and the coach evaluation suite both use it. The coach reaches its tools exactly
as in the web app: `/mcp` with a token for one user. Category corrections, goals and alert
settings go to a throwaway directory, never the local app's stores, and each `session` gets its
own feedback subject, so writes in one conversation never show up in another.

    with LocalApp(settings) as local:
        tools = local.tools("u_te_yp_0030")      # a fresh session's tools for the user
        coach.answer(tools, Conversation(), "How much did I spend in August?")
"""

import secrets
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

import anyio.from_thread
from anyio.from_thread import BlockingPortal
from anyio.lowlevel import EventLoopToken, current_token
from pydantic import SecretStr

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.accounts import Account, load_accounts
from smart_financial_coach.experience.demo import ACCOUNTS_FILE

TOKEN_LIFETIME = timedelta(hours=2)


def local_settings(settings: Settings, **overrides: Any) -> Settings:
    """`settings` for a local, throwaway run: loopback address, fresh secrets and stores."""
    scratch = Path(tempfile.mkdtemp(prefix="sfc-coach-"))
    return settings.model_copy(
        update={
            "demo_password": SecretStr(secrets.token_hex(16)),
            "session_secret": SecretStr(secrets.token_hex(32)),
            "public_url": "http://127.0.0.1:8000",
            "feedback_db": scratch / "feedback.sqlite3",
            "goals_db": scratch / "goals.sqlite3",
            **overrides,
        }
    )


class LocalApp:
    def __init__(self, settings: Settings) -> None:
        from smart_financial_coach.experience.web.app import create_app

        self.settings = settings
        self.accounts: list[Account] = load_accounts(settings.demo_dir / ACCOUNTS_FILE)
        self.sources = DataSources.from_dir(settings.demo_dir)
        self.as_of: date = self.sources.as_of()
        self.app = create_app(settings, sources=self.sources, accounts=self.accounts)
        secret = settings.session_secret.get_secret_value()  # type: ignore[union-attr]
        self._tokens = AccessTokens(secret, [a.user_id for a in self.accounts])
        self._portal_cm: Any = None
        self._lifespan_cm: Any = None
        self._loop: EventLoopToken | None = None

    @property
    def users(self) -> list[str]:
        return [a.user_id for a in self.accounts]

    def tools(self, user_id: str, session: str | None = None) -> McpTools:
        """The coach's tools for `user_id`, in `session` (a fresh one by default)."""
        if self._loop is None:
            raise RuntimeError("use LocalApp in a with-block")
        token = self._tokens.issue(
            user_id,
            TOKEN_LIFETIME,
            client="coach",
            feedback_subject=session or secrets.token_urlsafe(12),
        )
        return McpTools(self.app, token, as_of=self.as_of, loop=self._loop)

    def __enter__(self) -> "LocalApp":
        # The app's lifespan (the MCP session manager) runs on a portal's loop, as under uvicorn
        self._portal_cm = anyio.from_thread.start_blocking_portal()
        portal: BlockingPortal = self._portal_cm.__enter__()
        self._lifespan_cm = portal.wrap_async_context_manager(
            self.app.router.lifespan_context(self.app)
        )
        self._lifespan_cm.__enter__()
        self._loop = portal.call(current_token)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._loop = None
        try:
            self._lifespan_cm.__exit__(kind, error, traceback)
        finally:
            self._portal_cm.__exit__(kind, error, traceback)
