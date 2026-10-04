"""`sfc-web`: build the demo bundle and serve the web app.

sfc-web build-demo --data data/synthetic/default.sqlite      # -> build/demo
sfc-web serve --dev                                           # http://127.0.0.1:8000
sfc-web serve --host 0.0.0.0 --port 8000                      # needs SFC_DEMO_PASSWORD,
                                                              # SFC_SESSION_SECRET
"""

import argparse
import logging
import secrets
from collections.abc import Sequence
from pathlib import Path

from pydantic import SecretStr

from smart_financial_coach.config import PROJECT_ROOT, get_settings

DEV_PASSWORD = "demo-password"


def _build(args: argparse.Namespace) -> int:
    from smart_financial_coach.experience.demo import build_demo

    bundle = build_demo(args.data, args.accounts, args.out)
    print(
        f"{bundle.root}: {bundle.users} users, {bundle.transactions} transactions, "
        f"categorized by {bundle.model_version}"
    )
    if bundle.flag_model_version:
        print(f"  {bundle.flags} unusual charges flagged by {bundle.flag_model_version}")
    else:
        print("  no unusual-transaction model promoted: unusual charges stay not available")
    if bundle.forecast_model_version:
        print(f"  goal forecast states from {bundle.forecast_model_version}")
    else:
        print("  no goal-forecasting model promoted: goal forecasts stay not available")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from smart_financial_coach.experience.coach import Coach
    from smart_financial_coach.experience.web.app import create_app

    settings = get_settings()
    if args.dev:  # local http: throwaway secrets, one-click sign-in, plain cookies
        settings = settings.model_copy(
            update={
                "demo_password": settings.demo_password or SecretStr(DEV_PASSWORD),
                "session_secret": settings.session_secret or SecretStr(secrets.token_hex(32)),
                "quick_signin": True,
                "secure_cookies": False,
            }
        )
        if settings.demo_password.get_secret_value() == DEV_PASSWORD:  # type: ignore[union-attr]
            print(f"Demo password: {DEV_PASSWORD}")
    coach = Coach.from_settings(settings)
    if coach is None:
        print("No Anthropic API key (SFC_LLM_API_KEY or ANTHROPIC_API_KEY): chat is unavailable")
    app = create_app(settings, coach=coach)
    # The app reads the client address itself (SFC_TRUSTED_PROXY_HOPS): uvicorn's proxy headers
    # would trust the left-most X-Forwarded-For entry, which the visitor writes
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        proxy_headers=False,
        log_level=settings.log_level.lower(),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfc-web", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-demo", help="package the demo accounts' data, categorized")
    build.add_argument("--data", type=Path, required=True, help="a generated dataset")
    build.add_argument(
        "--accounts", type=Path, default=PROJECT_ROOT / "configs" / "web" / "demo_accounts.yaml"
    )
    build.add_argument("--out", type=Path, default=get_settings().demo_dir)
    build.set_defaults(handler=_build)
    serve = sub.add_parser("serve", help="serve the web app from the demo bundle")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--dev", action="store_true", help="local http with throwaway secrets")
    serve.set_defaults(handler=_serve)
    args = parser.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s %(message)s")
    return int(args.handler(args))
