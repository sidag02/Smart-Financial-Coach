"""`sfc-coach`: ask the coach questions from the command line, as one demo user.

sfc-coach ask --user u_te_yp_0030 "Why was August so high?"
sfc-coach ask --backend subscription --user u_te_yp_0030 "Am I on track?" "And by how much?"

Questions after the first continue the same conversation. The app's MCP server runs in process
with the demo bundle (`sfc-web build-demo`), and the coach calls it with a token for the user, as
in the web app. Category corrections, goals and alert settings the coach changes go to a
throwaway directory, never the local app's stores. `--backend subscription` runs on the owner's
Claude login (FR-13 to FR-15 design, §2): unset any API key first.
"""

import argparse
import logging
import secrets
import tempfile
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import anyio.from_thread
from anyio.lowlevel import current_token
from pydantic import SecretStr

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.config import get_settings
from smart_financial_coach.experience.accounts import load_accounts
from smart_financial_coach.experience.coach import CoachUnavailableError, Conversation, make_coach
from smart_financial_coach.experience.demo import ACCOUNTS_FILE

TOKEN_LIFETIME = timedelta(hours=1)


def _ask(args: argparse.Namespace) -> int:
    from smart_financial_coach.experience.web.app import create_app

    settings = get_settings()
    scratch = Path(tempfile.mkdtemp(prefix="sfc-coach-"))
    settings = settings.model_copy(
        update={
            "coach_backend": args.backend or settings.coach_backend,
            "demo_password": SecretStr(secrets.token_hex(16)),
            "session_secret": SecretStr(secrets.token_hex(32)),
            "public_url": "http://127.0.0.1:8000",
            "feedback_db": scratch / "feedback.sqlite3",
            "goals_db": scratch / "goals.sqlite3",
        }
    )
    coach = make_coach(settings)
    if coach is None:
        print("No API key (SFC_LLM_API_KEY or ANTHROPIC_API_KEY): try --backend subscription")
        return 1
    accounts = load_accounts(settings.demo_dir / ACCOUNTS_FILE)
    if args.user not in {a.user_id for a in accounts}:
        print(f"{args.user} isn't a demo account: {', '.join(a.user_id for a in accounts)}")
        return 1
    sources = DataSources.from_dir(settings.demo_dir)
    app = create_app(settings, sources=sources, accounts=accounts)
    secret = settings.session_secret.get_secret_value()  # type: ignore[union-attr]
    token = AccessTokens(secret, [a.user_id for a in accounts]).issue(
        args.user, TOKEN_LIFETIME, client="coach"
    )
    conversation = Conversation()
    # The app's lifespan (the MCP session manager) runs on a portal's loop, as under uvicorn
    with (
        anyio.from_thread.start_blocking_portal() as portal,
        portal.wrap_async_context_manager(app.router.lifespan_context(app)),
    ):
        tools = McpTools(app, token, as_of=sources.as_of(), loop=portal.call(current_token))
        for question in args.questions:
            print(f"> {question}")
            try:
                reply = coach.answer(tools, conversation, question)
            except CoachUnavailableError as error:
                print(f"The coach is unavailable: {error}")
                return 1
            print(reply.text)
            for source_id, source in reply.cited.items():
                print(f"  [{source_id}] {source.title} · {source.detail}")
            print(
                f"  ({reply.seconds:.1f} s, {coach.backend} backend, {coach.model}, "
                f"effort {coach.effort}, credential {coach.credential_source})\n"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfc-coach", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="ask the coach one or more questions as a demo user")
    ask.add_argument("--user", required=True, help="a demo account's user id")
    ask.add_argument(
        "--backend",
        choices=("auto", "api", "subscription"),
        help="where the model runs (default: SFC_COACH_BACKEND)",
    )
    ask.add_argument("questions", nargs="+", help="questions, asked in order in one conversation")
    ask.set_defaults(handler=_ask)
    args = parser.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s %(message)s")
    return int(args.handler(args))
