"""The web app: sign-in, overview, transactions, coach chat (mockups 1a, 1d, 1e, 1i).

Server-rendered with Jinja templates and htmx (Web App UI, decision 7). Identity lives only in
the signed session cookie; every page builds its `Tools` from that session's user, never from a
request parameter, so no URL or form field can name another user's data (NFR-2).

    app = create_app(get_settings())
"""

import html
import json
import logging
import re
import secrets
from collections import Counter, OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from anyio.lowlevel import EventLoopToken, current_token
from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from starlette.middleware.sessions import SessionMiddleware

from smart_financial_coach.access.alerts import AlertStore
from smart_financial_coach.access.feedback import FeedbackStore
from smart_financial_coach.access.goals import GoalStore
from smart_financial_coach.access.ledger import INCOME, DataSources, Ledger
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.mcp_server import PATH as MCP_PATH
from smart_financial_coach.access.mcp_server import build_mcp_server
from smart_financial_coach.access.review_items import alternatives, open_review_items
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.access.tools import (
    NOT_ME_GUIDANCE,
    NOT_ME_LIMITS,
    AlertAccess,
    Feedback,
    GoalAccess,
    GoalProblemsError,
    Source,
    ToolError,
    Tools,
    span_label,
)
from smart_financial_coach.config import PROJECT_ROOT, Settings
from smart_financial_coach.data.flags import load_flag_presets
from smart_financial_coach.experience.accounts import Account, SharedPassword, load_accounts
from smart_financial_coach.experience.coach import Coach, CoachUnavailableError, Conversation
from smart_financial_coach.experience.demo import ACCOUNTS_FILE, REPLAY_FILE
from smart_financial_coach.experience.web import charts
from smart_financial_coach.experience.web.limits import RateLimit
from smart_financial_coach.intelligence.categorization.agreement import (
    AgreementRule,
    current_votes,
    global_labels,
)
from smart_financial_coach.intelligence.forecasting.contract import OFF_TRACK, ON_TRACK

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
MOCKUPS_DIR = PROJECT_ROOT / "docs" / "design" / "mockups"
MOCKUP_PAGE = "Smart Financial Coach - Light & Dark.dc.html"
SESSION_HOURS = 12
MAX_QUESTION = 500
MAX_CONVERSATIONS = 1_000
MINUS = "\u2212"  # a real minus sign for amounts
COACH_TOKEN_LIFETIME = timedelta(minutes=5)  # one question's worth of tool calls
FLAG_WINDOW_DAYS = 60  # "Worth a look" shows a fixed window in v1 (owner, Oct 3, 2026, on #30)
OVERVIEW_FLAGS = 3  # the newest flags shown on the overview
HORIZONS = {"week": "Week", "month": "Month", "quarter": "Quarter", "year": "Year"}
REVIEW_SHOWN = 5  # review items in the transactions page's panel (FR-5)
ALTERNATIVES = 2  # quick-pick categories offered next to an item's suggestion
CHANGES_SHOWN = 5  # recent corrections listed with an undo button (FR-6)
MIN_VISIBLE = 2  # sessions that must vote on a merchant before others see its tally
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
}


class NotSignedInError(Exception):
    pass


@dataclass(frozen=True)
class Period:
    start: date
    end: date

    @property
    def label(self) -> str:
        return span_label(self.start, self.end)


def month_key(day: date) -> str:
    return f"{day:%Y-%m}"


def add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def month_period(key: str | None, as_of: date) -> Period:
    """The calendar month `key` (YYYY-MM), cut at `as_of`; the as-of month when absent or bad."""
    try:
        start = date.fromisoformat(f"{key}-01") if key else as_of.replace(day=1)
    except ValueError:
        start = as_of.replace(day=1)
    if start > as_of:
        start = as_of.replace(day=1)
    end = min(add_months(start, 1) - timedelta(days=1), as_of)
    return Period(start, end)


def horizon_period(month: Period, horizon: str) -> Period:
    if horizon == "week":
        return Period(month.end - timedelta(days=6), month.end)
    if horizon == "quarter":
        return Period(add_months(month.start, -2), month.end)
    if horizon == "year":
        return Period(add_months(month.start, -11), month.end)
    return month


def day(iso: str) -> str:
    """ "2027-04-30" -> "Apr 30, 2027"."""
    d = date.fromisoformat(str(iso)[:10])
    return f"{d:%b} {d.day}, {d.year}"


def month_name(key: str, long: bool = False) -> str:
    """ "2026-09" -> "Sep", or "Sep 2026" when long."""
    d = date.fromisoformat(f"{key}-01")
    return f"{d:%b %Y}" if long else f"{d:%b}"


def chance(p: float) -> str:
    """A probability as people say it: 0.72 -> "about a 7 in 10 chance". It stays inside the
    status band the badge shows (review on #56): "could go either way" (0.3 to 0.7) says 3 to 6
    in 10, "off track" at most 2, "on track" at least 7; and the ends never round to
    certainty."""
    if p >= 0.95:
        return "better than a 9 in 10 chance"
    if p < 0.05:
        return "less than a 1 in 10 chance"
    tenths = round(p * 10)
    if p < OFF_TRACK:
        tenths = min(tenths, 2)
    elif p < ON_TRACK:
        tenths = min(max(tenths, 3), 6)
    else:
        tenths = max(tenths, 7)
    return f"about a {max(1, min(9, tenths))} in 10 chance"


def _mean(values: list[Any]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def money(value: float, cents: bool = False, sign: bool = False) -> str:
    text = f"${abs(value):,.2f}" if cents else f"${abs(round(value)):,.0f}"
    if value < 0 and (sign or not cents):
        return MINUS + text
    if value > 0 and sign:
        return "+" + text
    return text


def goal_form_data(
    name: Annotated[str, Form()] = "",
    target_amount: Annotated[str, Form()] = "",
    target_month: Annotated[str, Form()] = "",
    saved: Annotated[str, Form()] = "",
    # The edit form's hidden goal_id; named apart from the /goals/{goal_id} path parameter
    editing: Annotated[str, Form(alias="goal_id")] = "",
) -> dict[str, str]:
    """The goal form as posted: the dependency every goal handler shares. No length limits here:
    over-long values get the form's own messages from the tools, not a 422 (review on #45)."""
    return {
        "name": name,
        "target_amount": target_amount,
        "target_month": target_month,
        "saved": saved,
        "goal_id": editing,
    }


GoalForm = Annotated[dict[str, str], Depends(goal_form_data)]


_SOURCE_TAG = re.compile(r"\s?\[(S\d+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def render_answer(text: str, cited: dict[str, Source]) -> Markup:
    """The coach's plain text as HTML: paragraphs, bullet lists, bold, and source chips."""

    def inline(line: str) -> str:
        escaped = _BOLD.sub(r"<b>\1</b>", html.escape(line))

        def chip(match: re.Match[str]) -> str:
            sid = match.group(1)
            if sid not in cited:
                return ""
            title = html.escape(cited[sid].title, quote=True)
            return f'<span class="src-chip" title="{title}" data-src="{sid}">{sid[1:]}</span>'

        return _SOURCE_TAG.sub(chip, escaped)

    blocks: list[str] = []
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(("- ", "* ", "• ")):
            items.append(f"<li>{inline(line[2:])}</li>")
            continue
        if items:
            blocks.append(f"<ul>{''.join(items)}</ul>")
            items = []
        if line:
            blocks.append(f"<p>{inline(line)}</p>")
    if items:
        blocks.append(f"<ul>{''.join(items)}</ul>")
    return Markup("".join(blocks))


def greeting(timezone: str) -> str:
    try:
        hour = datetime.now(ZoneInfo(timezone)).hour
    except (KeyError, ValueError):
        hour = datetime.now().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"


def create_app(
    settings: Settings,
    *,
    sources: DataSources | None = None,
    accounts: list[Account] | None = None,
    coach: Coach | None = None,
) -> FastAPI:
    if settings.demo_password is None or settings.session_secret is None:
        raise RuntimeError("set SFC_DEMO_PASSWORD and SFC_SESSION_SECRET to serve the web app")
    sources = sources or DataSources.from_dir(settings.demo_dir)
    accounts = (
        accounts if accounts is not None else load_accounts(settings.demo_dir / ACCOUNTS_FILE)
    )
    by_email = {a.email: a for a in accounts}
    by_user = {a.user_id: a for a in accounts}
    password = SharedPassword(settings.demo_password.get_secret_value())
    as_of = sources.as_of()
    flags_live = sources.flags is not None  # an FR-7 model is promoted and its flags are here
    spikes_live = sources.spikes is not None  # a spikes file: the FR-8 model or the simple rule
    state = sources.spike_state()
    spike_method = state.method if state is not None else None  # "model" or "simple_rule"
    # Which halves have sensitivity presets (FR-9 §2): the switch changes only those
    presets_live = {
        "unusual_charges": flags_live
        and sources.flags is not None
        and load_flag_presets(sources.flags) is not None,
        "spending_spikes": state is not None and state.presets is not None,
    }
    alerts_live = flags_live or spikes_live
    essentials = frozenset(settings.essentials)
    unknown = sorted(essentials - set(sources.categories()) | essentials & {INCOME})
    if unknown:
        raise ValueError(f"SFC_ESSENTIALS names categories that can't be essentials: {unknown}")
    conversations: OrderedDict[str, Conversation] = OrderedDict()
    # Limits are keyed by client address, which a visitor can't reset the way they can a session,
    # with a cap across everyone as the backstop: the real bound on guesses and on LLM spend
    chat_limit = RateLimit(settings.chat_messages_per_hour, 3600)
    chat_total = RateLimit(settings.chat_messages_per_hour_total, 3600)
    signin_limit = RateLimit(settings.signin_attempts_per_minute, 60)
    signin_total = RateLimit(settings.signin_attempts_per_minute_total, 60)

    def client_address(request: Request) -> str:
        """The visitor's address: behind a proxy, the X-Forwarded-For entry the proxy appended.

        Entries further left are whatever the visitor sent, so they're never trusted.
        """
        hops = settings.trusted_proxy_hops
        forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",")]
        forwarded = [part for part in forwarded if part]
        if hops > 0 and len(forwarded) >= hops:
            return forwarded[-hops]
        return request.client.host if request.client else "unknown"

    # The MCP server (FR-19): the coach and outside assistants call the same tools, as the user
    # their bearer token names
    public_url = settings.public_url.rstrip("/")
    tokens = AccessTokens(settings.session_secret.get_secret_value(), by_user)
    # Category feedback (FR-5, FR-6), keyed by browser session: visitors share demo accounts
    feedback = FeedbackStore(settings.feedback_db, sources.categories())
    # Savings goals (FR-10), keyed by browser session like feedback
    goal_store = GoalStore(settings.goals_db)
    # Alert sensitivity and flag actions (FR-9), in the feedback store's file, keyed the same way
    alert_store = AlertStore(settings.feedback_db)
    mcp_server, mcp_asgi = build_mcp_server(
        sources,
        tokens,
        public_url=public_url,
        feedback=feedback,
        goals=goal_store,
        alerts=alert_store,
    )

    loop: dict[str, EventLoopToken] = {}  # the app's event loop, for the coach's MCP calls

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        loop["token"] = current_token()
        async with mcp_server.session_manager.run():
            yield

    app = FastAPI(
        title="Smart Financial Coach",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret.get_secret_value(),
        session_cookie="sfc_session",
        max_age=SESSION_HOURS * 3600,
        same_site="lax",
        https_only=settings.secure_cookies,
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    mockups = MOCKUPS_DIR.is_dir()
    if mockups:
        app.mount("/mockups", StaticFiles(directory=MOCKUPS_DIR), name="mockups")

    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters.update(money=money, day=day, month_name=month_name, chance=chance)
    templates.env.globals.update(
        coach_name=settings.coach_name,
        quick_signin=settings.quick_signin,
        horizons=HORIZONS,
        colors=charts.colors,
        as_of=as_of.isoformat(),
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    # Cross-site form posts (review on #38). SameSite=Lax isn't enough on the demo's address:
    # azurecontainerapps.io isn't on the Public Suffix List, so other tenants' apps there count as
    # the same site and their pages could post forms with a visitor's cookie. Browsers send
    # Origin on every POST, so a state-changing request from a browser must come from our own
    # origin; the MCP endpoint is exempt (bearer tokens, no cookies)
    own = urlsplit(public_url)
    allowed = {f"{own.scheme}://{own.netloc}"}
    if own.hostname in ("127.0.0.1", "localhost"):  # a local run, reached by either name
        allowed |= {f"{own.scheme}://{h}:{own.port}" for h in ("127.0.0.1", "localhost")}

    @app.middleware("http")
    async def same_origin_posts(request: Request, call_next: Any) -> Response:
        path = request.url.path
        mcp = path == MCP_PATH or path.startswith(f"{MCP_PATH}/")  # not /mcp-anything (#38)
        if request.method not in ("GET", "HEAD", "OPTIONS") and not mcp:
            origin = request.headers.get("origin")
            if origin is None and (referer := request.headers.get("referer")):
                parts = urlsplit(referer)
                origin = f"{parts.scheme}://{parts.netloc}"
            if origin is not None and origin not in allowed:
                log.warning("refused a %s to %s from %s", request.method, request.url.path, origin)
                return Response("Cross-site request refused", status_code=403)
        response: Response = await call_next(request)
        return response

    @app.exception_handler(NotSignedInError)
    async def to_signin(request: Request, _: NotSignedInError) -> Response:
        if request.headers.get("HX-Request"):
            return Response(status_code=204, headers={"HX-Redirect": "/signin"})
        return RedirectResponse("/signin", status_code=303)

    def signed_in(request: Request) -> Account:
        account = by_user.get(request.session.get("uid", ""))
        if account is None:
            raise NotSignedInError
        return account

    def feedback_id(request: Request) -> str:
        """Whose category feedback this is: the browser session, not the shared account, so two
        visitors on one demo account never see each other's corrections (owner, Oct 3, 2026).
        Kept apart from the chat's id, which a new chat replaces."""
        if not request.session.get("fid"):
            request.session["fid"] = secrets.token_urlsafe(16)
        return str(request.session["fid"])

    def tools_for(account: Account, request: Request) -> Tools:
        subject = feedback_id(request)  # the same session subject for corrections and goals
        return Tools(
            Ledger.load(sources, account.user_id),
            Feedback(feedback, subject),
            GoalAccess(goal_store, subject),
            alerts=AlertAccess(alert_store, subject),
        )

    def page(request: Request, name: str, account: Account | None, **context: Any) -> Response:
        theme = request.cookies.get("theme")
        return templates.TemplateResponse(
            request,
            name,
            {
                "account": account,
                "theme": theme if theme in ("light", "dark") else None,
                "mockups": mockups,
                "mockup_page": MOCKUP_PAGE,
                "flags_live": alerts_live,
                **context,
            },
        )

    def month_options() -> list[tuple[str, str]]:
        start = as_of.replace(day=1)
        return [(month_key(m), f"{m:%B %Y}") for m in (add_months(start, -i) for i in range(12))]

    def conversation(request: Request) -> Conversation:
        sid = request.session.get("sid", "")
        if sid not in conversations:
            conversations[sid] = Conversation()
            while len(conversations) > MAX_CONVERSATIONS:
                conversations.popitem(last=False)
        conversations.move_to_end(sid)
        return conversations[sid]

    def start_session(request: Request, account: Account) -> Response:
        request.session.clear()  # a new session id at every sign-in (no fixation)
        request.session.update(
            uid=account.user_id, sid=secrets.token_urlsafe(16), fid=secrets.token_urlsafe(16)
        )
        return RedirectResponse("/", status_code=303)

    # Health and sign-in

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """Healthy only if every demo account's data loads: the deploy's smoke test and the
        container's health check then catch an unreadable or incomplete bundle."""
        try:
            for account in accounts:
                ledger = Ledger.load(sources, account.user_id)  # cached after the first check
                if ledger.spikes is not None:  # scoring reads the user's data, so check it runs
                    ledger.spikes.score(ledger.user_id, ledger.transactions, ledger.transactions)
        except Exception:
            log.exception("health check: the demo data doesn't load")
            return JSONResponse({"status": "error", "as_of": as_of.isoformat()}, 503)
        return JSONResponse(
            {
                "status": "ok",
                "as_of": as_of.isoformat(),
                "users": len(accounts),
                "spikes": spike_method,
                "presets": presets_live,
            }
        )

    @app.get("/signin")
    def signin_page(request: Request) -> Response:
        return page(request, "signin.html", None, accounts=accounts, error=None, email="")

    @app.post("/signin")
    def signin(
        request: Request,
        email: Annotated[str, Form()] = "",
        password_: Annotated[str, Form(alias="password")] = "",
    ) -> Response:
        if not (signin_limit.allow(client_address(request)) and signin_total.allow("all")):
            error = "Too many attempts. Wait a minute, then try again."
            return page(request, "signin.html", None, accounts=accounts, error=error, email=email)
        account = by_email.get(email.strip().lower())
        if account is None or not password.matches(password_):
            error = "That email and password don't match a demo account."
            return page(request, "signin.html", None, accounts=accounts, error=error, email=email)
        return start_session(request, account)

    @app.post("/signin/quick")
    def quick_signin(request: Request, user_id: Annotated[str, Form()] = "") -> Response:
        account = by_user.get(user_id)
        if not settings.quick_signin or account is None:
            return RedirectResponse("/signin", status_code=303)
        return start_session(request, account)

    @app.post("/signout")
    def signout(request: Request) -> Response:
        conversations.pop(request.session.get("sid", ""), None)
        request.session.clear()
        return RedirectResponse("/signin", status_code=303)

    # Overview (mockup 1a)

    def flow_context(tools: Tools, month: Period, horizon: str) -> dict[str, Any]:
        horizon = horizon if horizon in HORIZONS else "month"
        period = horizon_period(month, horizon)
        summary = tools.get_spending_summary(period.start.isoformat(), period.end.isoformat())
        data = summary.data
        by_category = [(c["category"], c["amount"]) for c in data["by_category"]]
        return {
            "horizon": horizon,
            "month_key": month_key(month.start),
            "flow_period": period,
            "flow_summary": data,
            "flow": charts.money_flow(data["income"], by_category, essentials),
        }

    @app.get("/")
    def overview(request: Request, month: str | None = None, horizon: str = "month") -> Response:
        account = signed_in(request)
        tools = tools_for(account, request)
        period = month_period(month, as_of)
        summary = tools.get_spending_summary(period.start.isoformat(), period.end.isoformat())
        year = tools.get_spending_summary(
            add_months(period.start, -11).isoformat(), period.end.isoformat()
        )
        months = [(m["month"], m["spending"]) for m in year.data["by_month"]]
        essential_spend = sum(
            c["amount"] for c in summary.data["by_category"] if c["category"] in essentials
        )
        return page(
            request,
            "overview.html",
            account,
            active="overview",
            greeting=greeting(tools.ledger.timezone),
            month=period,
            month_options=month_options(),
            summary=summary.data,
            essentials=essential_spend,
            goal=featured_goal(tools),
            trend=charts.trend(months, month_key(period.start)),
            flags=recent_flags(tools) if flags_live else [],
            spikes=recent_spikes(tools) if spikes_live else [],
            spike_method=spike_method,
            overview_flags=OVERVIEW_FLAGS,
            window_days=FLAG_WINDOW_DAYS,
            **flow_context(tools, period, horizon),
        )

    @app.get("/flow")
    def flow(request: Request, month: str | None = None, horizon: str = "month") -> Response:
        tools = tools_for(signed_in(request), request)
        context = flow_context(tools, month_period(month, as_of), horizon)
        return page(request, "_flow.html", None, **context)

    @app.get("/drill")
    def drill(
        request: Request, category: str, month: str | None = None, horizon: str = "month"
    ) -> Response:
        tools = tools_for(signed_in(request), request)
        period = horizon_period(month_period(month, as_of), horizon)
        if category not in tools.categories:
            return Response(status_code=404)
        args = {"start_date": period.start.isoformat(), "end_date": period.end.isoformat()}
        rows = tools.get_transactions(**args, category=category, sort="largest", limit=8).data
        spent = tools.get_spending_summary(**args).data["spending"]
        return page(
            request,
            "_drill.html",
            None,
            category=category,
            period=period,
            month_key=month_key(month_period(month, as_of).start),
            rows=rows,
            share=(-rows["total"] / spent) if spent else 0.0,
        )

    # Transactions, with review and corrections (mockup 1e; FR-5, FR-6)

    @app.get("/transactions")
    def transactions(
        request: Request,
        month: str | None = None,
        category: str | None = None,
        q: Annotated[str, Query(max_length=80)] = "",
        review: bool = False,
    ) -> Response:
        account = signed_in(request)
        tools = tools_for(account, request)
        period = month_period(month, as_of)
        category = category if category in tools.categories else None
        rows = tools.ledger.between(period.start, period.end)
        not_sure = int(rows["needs_review"].sum())
        if category:
            rows = rows[rows["category"] == category]
        if q.strip():
            text = q.strip()
            rows = rows[
                rows["merchant"].str.contains(text, case=False, regex=False)
                | rows["merchant_raw"].str.contains(text, case=False, regex=False)
            ]
        if review:
            rows = rows[rows["needs_review"]]
        queue = tools.list_review_items(limit=REVIEW_SHOWN).data
        keys: dict[str, str] = dict(
            open_review_items(tools.ledger.transactions, account.user_id)[
                ["item_id", "merchant_key"]
            ].itertuples(index=False)
        )
        picks = {
            i["item_id"]: alternatives(
                tools.ledger.transactions, keys[i["item_id"]], i["suggested_category"], ALTERNATIVES
            )
            for i in queue["items"]
        }
        return page(
            request,
            "transactions.html",
            account,
            active="transactions",
            month=period,
            month_options=month_options(),
            rows=list(rows.itertuples()),
            count=len(tools.ledger.between(period.start, period.end)),
            not_sure=not_sure,
            category=category,
            categories=[c for c in tools.categories if c != INCOME] + [INCOME],
            q=q,
            review=review,
            # The review policy the bundle was flagged with, read here rather than at start-up,
            # so an unreadable bundle shows in /healthz instead of stopping the app
            review_below=dict(
                zip(("familiar", "unfamiliar"), sources.review_thresholds(), strict=True)
            ),
            queue=queue,
            picks=picks,
            changes=tools.list_corrections(limit=CHANGES_SHOWN).data["corrections"],
            flash=request.session.pop("flash", None),
            back=str(request.url.path) + (f"?{request.url.query}" if request.url.query else ""),
            # Unusual charges are marked in the list (FR-7 §8), one per same-minute pair as in
            # "Worth a look"
            flagged=flagged_ids(tools, period),
        )

    def back_to(back: str) -> RedirectResponse:
        """Back to the page a form came from: only a transactions page of this app."""
        ok = back.startswith("/transactions") and not back.startswith("//")
        return RedirectResponse(back if ok else "/transactions", status_code=303)

    def changed(result: dict[str, Any]) -> str:
        n = result["transactions_changed"]
        noun = "transaction" if n == 1 else "transactions"
        c = result["correction"]
        if c["action"] == "confirm":
            # An ambiguous merchant's other rows move too: say so (review on #38)
            moved = f" (moved {n} {noun})" if n else ""
            return f"Confirmed {c['merchant']} as {c['to_category']}{moved}."
        return f"Moved {n} {noun} at {c['merchant']} to {c['to_category']}."

    def feedback_action(request: Request, back: str, call: Callable[[Tools], str]) -> Response:
        account = signed_in(request)
        try:
            request.session["flash"] = call(tools_for(account, request))
        except ToolError as error:
            request.session["flash"] = f"Couldn't change that: {error}"
        return back_to(back)

    @app.post("/review/{item}")
    def resolve(
        request: Request,
        item: str,
        action: Annotated[str, Form()],
        category: Annotated[str | None, Form()] = None,
        back: Annotated[str, Form()] = "/transactions",
    ) -> Response:
        args = {"item_id": item, "action": action, **({"category": category} if category else {})}
        return feedback_action(
            request, back, lambda t: changed(t.call("resolve_review_item", args).data)
        )

    @app.post("/transactions/{transaction_id}/category")
    def recategorize(
        request: Request,
        transaction_id: str,
        category: Annotated[str, Form()],
        scope: Annotated[str, Form()] = "merchant",
        back: Annotated[str, Form()] = "/transactions",
    ) -> Response:
        args = {"transaction_id": transaction_id, "category": category, "scope": scope}
        return feedback_action(
            request, back, lambda t: changed(t.call("correct_category", args).data)
        )

    @app.post("/corrections/{correction_id}/undo")
    def undo(
        request: Request, correction_id: str, back: Annotated[str, Form()] = "/transactions"
    ) -> Response:
        def run(t: Tools) -> str:
            result = t.call("undo_correction", {"correction_id": correction_id}).data
            return f"Undone: {result['undone']['merchant']} is back to how it was."

        return feedback_action(request, back, run)

    # How it learns (FR-5/FR-6 §7): the feedback replay, step by step, and visitors' agreement

    @app.get("/learning")
    def learning(request: Request) -> Response:
        account = signed_in(request)
        path = sources.dataset.parent / REPLAY_FILE
        replay = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        context: dict[str, Any] = {"replay": replay}
        if replay:
            months = [m["month"] for m in replay["months"]]

            def share(m: dict[str, Any], group: str) -> float | None:
                value = m[group].get("accuracy_view")
                return float(value) if value is not None else None

            evaluation = [share(m, "evaluation") for m in replay["months"]]
            personal = [share(m, "personal") for m in replay["months"]]
            promoted = [s["month"] for s in replay["retrainings"] if s["decision"] == "promoted"]
            context |= {
                "chart": charts.lines(
                    {"evaluation": evaluation, "personal": personal}, months, promoted
                ),
                "first_q": _mean(evaluation[:3]),
                "last_q": _mean(evaluation[-3:]),
                # New merchants against the true categories: model errors fixed, comparable with
                # the PRD's 0.80 goal for v1.1, which feedback is meant to reach
                "unseen_first": _mean(
                    [m["evaluation_unseen"].get("macro_f1_truth") for m in replay["months"][:3]]
                ),
                "unseen_last": _mean(
                    [m["evaluation_unseen"].get("macro_f1_truth") for m in replay["months"][-3:]]
                ),
                "labels": sorted(replay["labels"], key=lambda x: -x["voters"])[:15],
            }
        # Visitors' agreement, live and illustrative: counts across sessions, never who
        votes = current_votes(feedback.all_events())
        rule = AgreementRule()
        tallies: dict[str, Counter[str]] = {}
        for v in votes:
            tallies.setdefault(v.merchant_key, Counter())[v.category] += 1
        labels_now = global_labels(votes, rule)
        context["visitors"] = [
            {
                "merchant": key.title(),
                "category": counts.most_common(1)[0][0],
                "agreeing": counts.most_common(1)[0][1],
                "voters": sum(counts.values()),
                "needed": rule.n,
                "label": key in labels_now,
            }
            for key, counts in sorted(tallies.items(), key=lambda kv: -sum(kv[1].values()))[:8]
            if sum(counts.values()) >= MIN_VISIBLE  # one person's choice stays theirs (§4)
        ]
        context["single_votes"] = sum(1 for c in tallies.values() if sum(c.values()) < MIN_VISIBLE)
        return page(request, "learning.html", account, active="learning", **context)

    # Chat (mockup 1d)

    @app.get("/chat")
    def chat_page(
        request: Request, q: Annotated[str, Query(max_length=MAX_QUESTION)] = ""
    ) -> Response:
        account = signed_in(request)
        convo = conversation(request)
        return page(
            request,
            "chat.html",
            account,
            active="chat",
            turns=convo.turns,
            sources=list(enumerate(convo.sources, start=1)),
            ask=q.strip(),
            available=coach is not None,
            suggestions=suggestions(),
        )

    def suggestions() -> list[str]:
        this, last = f"{as_of:%B}", f"{add_months(as_of.replace(day=1), -1):%B}"
        return [
            f"Where did most of my money go in {this}?",
            f"How does {this} compare to {last}?",
            "What were my biggest purchases this month?",
        ]

    @app.post("/chat")
    def chat(request: Request, question: Annotated[str, Form()] = "") -> Response:
        account = signed_in(request)
        question = question.strip()[:MAX_QUESTION]
        convo = conversation(request)
        if not question:
            return Response(status_code=204)
        context: dict[str, Any] = {"question": question, "reply": None, "error": None}
        if not convo.lock.acquire(blocking=False):  # a chip or another tab, mid-answer
            context["error"] = "Still answering your last question. Ask again in a moment."
            return page(request, "_chat_turn.html", None, turn=context, sources=None)
        try:
            if coach is None:
                context["error"] = (
                    f"{settings.coach_name} isn't available right now. Your overview and "
                    "transactions still work."
                )
            elif not (chat_limit.allow(client_address(request)) and chat_total.allow("all")):
                context["error"] = (
                    "That's a lot of questions for one hour. Try again a little later."
                )
            else:
                try:
                    token = tokens.issue(
                        account.user_id,
                        COACH_TOKEN_LIFETIME,
                        client="coach",
                        feedback_subject=feedback_id(request),
                    )
                    tools = McpTools(app, token, as_of=as_of, loop=loop["token"])
                    reply = coach.answer(tools, convo, question)
                    context["reply"] = reply
                    context["answer"] = render_answer(reply.text, reply.cited)
                    log.info("coach answered in %.1f s", reply.seconds)
                except CoachUnavailableError:
                    context["error"] = (
                        f"{settings.coach_name} couldn't be reached just now. "
                        "Try again in a moment."
                    )
            convo.turns.append(context)
        finally:
            convo.lock.release()
        return page(
            request,
            "_chat_turn.html",
            None,
            turn=context,
            sources=list(enumerate(convo.sources, start=1)),
        )

    @app.post("/chat/new")
    def new_chat(request: Request) -> Response:
        signed_in(request)
        conversations.pop(request.session.get("sid", ""), None)
        request.session["sid"] = secrets.token_urlsafe(16)
        return RedirectResponse("/chat", status_code=303)

    # Coming next (Delivery Plan sync rule): real data where it exists, mockups for the rest

    def flagged_ids(tools: Tools, period: Period) -> set[str]:
        if not flags_live:
            return set()
        found = tools.detect_anomalies(period.start.isoformat(), period.end.isoformat()).data
        return {f["transaction_id"] for f in found.get("unusual_transactions", [])}

    def recent_flags(tools: Tools, include_hidden: bool = False) -> list[dict[str, Any]]:
        """The last `FLAG_WINDOW_DAYS` days' unusual charges at the session's sensitivity, newest
        first (FR-7 §8, FR-9); the ones the person hid only with `include_hidden`."""
        start = as_of - timedelta(days=FLAG_WINDOW_DAYS - 1)
        found = tools.detect_anomalies(
            start.isoformat(), as_of.isoformat(), include_hidden=include_hidden
        ).data
        flags: list[dict[str, Any]] = found.get("unusual_transactions", [])
        if not isinstance(flags, list):  # "not available"
            return []
        return sorted(flags, key=lambda f: (f["date"], f["transaction_id"]), reverse=True)

    def recent_spikes(tools: Tools, include_hidden: bool = False) -> list[dict[str, Any]]:
        """Spikes in months that ended in the last `FLAG_WINDOW_DAYS` days at the session's
        sensitivity, newest first (FR-8 §8, decision 9; FR-9). Only complete months are judged;
        the ones the person hid only with `include_hidden`."""
        start = as_of - timedelta(days=FLAG_WINDOW_DAYS - 1)
        first = start.replace(day=1)
        found = tools.detect_anomalies(
            first.isoformat(), as_of.isoformat(), include_hidden=include_hidden
        ).data
        spiking = found.get("spending_spikes", {})
        if spiking.get("status") != "ok":
            return []
        return [s for s in spiking["spikes"] if s["period_end"] >= start.isoformat()]

    @app.get("/worth-a-look")
    def worth_a_look(request: Request) -> Response:
        account = signed_in(request)
        if not alerts_live:
            return page(request, "coming.html", account, active="flags")
        tools = tools_for(account, request)
        # Hidden alerts (FR-9) are counted from the same lists the page shows, so a footer's
        # number is what "Show them" brings back
        show_hidden = request.query_params.get("hidden") == "1"
        flags = recent_flags(tools, include_hidden=True) if flags_live else []
        spikes = recent_spikes(tools, include_hidden=True) if spikes_live else []
        hidden = {
            "recognized": sum(1 for f in flags if f.get("hidden_by")),
            "expected": sum(1 for x in spikes if x.get("hidden_by")),
        }
        if not show_hidden:
            flags = [f for f in flags if not f.get("hidden_by")]
            spikes = [x for x in spikes if not x.get("hidden_by")]
        return page(
            request,
            "worth_a_look.html",
            account,
            active="flags",
            unusual_live=flags_live,
            spikes_live=spikes_live,
            flags=flags,
            spikes=spikes,
            hidden=hidden,
            show_hidden=show_hidden,
            settings=tools.get_alert_settings().data,
            can_act=True,
            guidance=NOT_ME_GUIDANCE,
            limits=NOT_ME_LIMITS,
            toast=request.session.pop("alert_toast", None),
            spike_method=spike_method,
            start=(as_of - timedelta(days=FLAG_WINDOW_DAYS - 1)).isoformat(),
            end=as_of.isoformat(),
            window_days=FLAG_WINDOW_DAYS,
        )

    # Alert sensitivity and flag actions (FR-9 §4): plain posts that redirect back with a toast,
    # so the page works without JavaScript, as Goals does

    def alert_back(back: str) -> RedirectResponse:
        """Back to the page a form came from: only "Worth a look" or the Overview."""
        ok = back in ("/", "/worth-a-look") or back.startswith("/worth-a-look?")
        return RedirectResponse(back if ok else "/worth-a-look", status_code=303)

    def alert_toast(request: Request, text: str, action_id: str | None = None) -> None:
        request.session["alert_toast"] = {"text": text, "action_id": action_id}

    @app.post("/worth-a-look/sensitivity")
    def set_sensitivity(request: Request, level: Annotated[str, Form()] = "") -> Response:
        account = signed_in(request)
        try:
            data = tools_for(account, request).call("set_alert_sensitivity", {"level": level}).data
        except ToolError as error:
            alert_toast(request, f"Couldn't change that: {error}")
        else:
            alert_toast(request, f"Alerts: {data['label']}. {data['description']}")
        return alert_back("/worth-a-look")

    @app.post("/alerts/act")
    def act_on_alert(
        request: Request,
        flag_id: Annotated[str, Form()] = "",
        action: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/worth-a-look",
    ) -> Response:
        account = signed_in(request)
        args = {"flag_id": flag_id, "action": action}
        try:
            data = tools_for(account, request).call("act_on_flag", args).data
        except ToolError as error:
            alert_toast(request, f"Couldn't do that: {error}")
        else:
            alert_toast(request, data["effect"], data["action"]["action_id"])
        return alert_back(back)

    @app.post("/alerts/undo")
    def undo_alert_action(
        request: Request,
        action_id: Annotated[str, Form()] = "",
        back: Annotated[str, Form()] = "/worth-a-look",
    ) -> Response:
        account = signed_in(request)
        try:
            tools_for(account, request).call("undo_flag_action", {"action_id": action_id})
        except ToolError as error:
            alert_toast(request, f"Couldn't undo that: {error}")
        else:
            alert_toast(request, "Undone: the alert shows again.")
        return alert_back(back)

    # Goals (FR-10; mockups 1g, 1h). The page writes through the same tools as the coach, as
    # "edit", which applies on submit. Writes are plain posts that redirect back to the list with
    # an Undo toast, so the form works without JavaScript; htmx adds the live check and opens the
    # form in the drawer. The handlers are plain `def`s, so their SQLite and pandas work runs in
    # the thread pool, never on the event loop (review on #45)

    def featured_goal(tools: Tools) -> dict[str, Any] | None:
        """The overview's goal: the active one due soonest, else a reached one."""
        listed = tools.list_goals(include_ended=False).data["goals"]
        by_status = {s: [g for g in listed if g["status"] == s] for s in ("active", "reached")}
        return next(iter(by_status["active"] or by_status["reached"]), None)

    def goal_fields(form: dict[str, str]) -> dict[str, Any]:
        """The form's fields as the tools take them: amounts without "$" or commas, and the month
        picker's "2027-06" as a day in that month. A blank name, amount or month stays blank, so
        it's reported rather than left unchanged; a blank saved amount means 0, or unchanged. An
        amount must be a plain decimal ("50", "50.", ".50", "50.25"; no exponent: "1e999999"
        would be costly to read), and anything else is reported as not an amount (review on
        #45)."""

        def text(key: str) -> str:
            return form.get(key, "").strip()

        def amount(key: str) -> str:
            value = text(key).replace("$", "").replace(",", "")
            plain = re.fullmatch(r"(?=\.?\d)\d{0,16}(\.\d{0,16})?", value)
            return value if not value or plain else "?"

        month = text("target_month")
        return {
            "name": text("name"),
            "target_amount": amount("target_amount"),
            "target_date": f"{month}-01" if re.fullmatch(r"\d{4}-\d{2}", month) else month,
            "saved": amount("saved") or None,
        }

    def goal_form(
        request: Request,
        account: Account,
        *,
        goal: dict[str, Any] | None,
        values: dict[str, str],
        problems: tuple[Any, ...] = (),
    ) -> Response:
        """The setup or edit form: in the drawer for htmx, a page of its own otherwise."""
        drawer = bool(request.headers.get("HX-Request"))
        errors = {p.field or "goal": form_message(p.code, p.message) for p in problems}
        return page(
            request,
            "_goal_form.html" if drawer else "goal_form.html",
            None if drawer else account,
            active="goals",
            drawer=drawer,
            goal=goal,
            values=values,
            errors=errors,
        )

    def form_message(code: str, message: str) -> str:
        """The tools' messages, except where the form's month picker reads better."""
        return "Pick a month." if code == "date_invalid" else message

    def form_values(form: dict[str, str]) -> dict[str, str]:
        return {k: form.get(k, "") for k in ("name", "target_amount", "target_month", "saved")}

    def form_amount(dollars: float) -> str:
        """An amount for the edit form, exactly: "12345.67", or "3000" for whole dollars. Never
        `:g`, which keeps 6 significant digits (review on #45)."""
        text = f"{dollars:.2f}"
        return text.removesuffix(".00")

    def goal_values(g: dict[str, Any]) -> dict[str, str]:
        return {
            "name": g["name"],
            "target_amount": form_amount(g["target_amount"]),
            "target_month": g["target_date"][:7],
            "saved": form_amount(g["saved"]),
        }

    def toast(request: Request, text: str, revision_id: str | None = None) -> None:
        request.session["goal_toast"] = {"text": text, "revision_id": revision_id}

    def back_to_goals(request: Request) -> Response:
        if request.headers.get("HX-Request"):
            return Response(status_code=204, headers={"HX-Redirect": "/goals"})
        return RedirectResponse("/goals", status_code=303)

    def editable_goal(tools: Tools, goal_id: str) -> dict[str, Any] | None:
        listed = tools.list_goals().data["goals"]
        return next((g for g in listed if g["goal_id"] == goal_id and g["status"] != "ended"), None)

    @app.get("/goals")
    def goals(request: Request) -> Response:
        account = signed_in(request)
        listed = tools_for(account, request).list_goals(include_archived=True).data
        by_status: dict[str, list[dict[str, Any]]] = {}
        for g in listed["goals"]:
            by_status.setdefault(g["status"], []).append(g)
        running = [g for g in listed["goals"] if g["status"] in ("active", "reached")]
        return page(
            request,
            "goals.html",
            account,
            active="goals",
            running=running,
            ended=by_status.get("ended", []),
            removed=by_status.get("archived", []),
            savings=listed,
            toast=request.session.pop("goal_toast", None),
        )

    @app.get("/goals/new")
    def new_goal(request: Request) -> Response:
        return goal_form(request, signed_in(request), goal=None, values={})

    @app.get("/goals/{goal_id}")
    def goal_detail(request: Request, goal_id: str) -> Response:
        """A goal's forecast (mockups 1g, 1l): rendered from `forecast_goal`, as the coach reads
        it, so the page and the coach can't disagree (FR-14)."""
        account = signed_in(request)
        tools = tools_for(account, request)
        goal = next((g for g in tools.list_goals().data["goals"] if g["goal_id"] == goal_id), None)
        if goal is None:
            return Response(status_code=404)
        forecast = tools.forecast_goal(goal_id).data
        live = forecast.get("status") not in ("not_available", "ended")
        projection = (
            charts.goal_projection(
                goal["saved"], goal["target_amount"], forecast["monthly"], forecast["history"]
            )
            if live and (forecast["monthly"] or forecast["history"])
            else None
        )
        return page(
            request,
            "goal.html",
            account,
            active="goals",
            goal=goal,
            forecast=forecast if live else None,
            projection=projection,
        )

    @app.get("/goals/{goal_id}/edit")
    def edit_goal(request: Request, goal_id: str) -> Response:
        account = signed_in(request)
        goal = editable_goal(tools_for(account, request), goal_id)
        if goal is None:
            return Response(status_code=404)
        return goal_form(request, account, goal=goal, values=goal_values(goal))

    @app.post("/goals/check")
    def check_goal(request: Request, form: GoalForm) -> Response:
        """The live "How it fits" box (FR-10 design, "The setup check"): facts from check_goal,
        and the problems for fields the person has filled in."""
        tools = tools_for(signed_in(request), request)
        fields = goal_fields(form)
        goal_id = form["goal_id"] or None
        try:
            data = tools.check_goal(**fields, goal_id=goal_id).data
        except ToolError:
            return Response(status_code=404)
        filled = {f for f, v in form_values(form).items() if v.strip()}
        filled |= {"target_date"} if "target_month" in filled else set()
        shown = [
            {**p, "message": form_message(p["code"], p["message"])}
            for p in data["problems"]
            if p["field"] is None or p["field"] in filled
        ]
        return page(request, "_goal_fit.html", None, check=data, problems=shown)

    @app.post("/goals")
    def create_goal(request: Request, form: GoalForm) -> Response:
        account = signed_in(request)
        tools = tools_for(account, request)
        try:
            data = tools.create_goal(**goal_fields(form)).data
        except GoalProblemsError as error:
            values = form_values(form)
            return goal_form(request, account, goal=None, values=values, problems=error.problems)
        toast(request, f"Created {data['goal']['name']}.", data["revision_id"])
        return back_to_goals(request)

    @app.post("/goals/undo")
    def undo_goal(request: Request, revision_id: Annotated[str, Form()] = "") -> Response:
        tools = tools_for(signed_in(request), request)
        try:
            data = tools.undo_goal_change(revision_id).data
        except ToolError as error:
            toast(request, f"Couldn't undo that: {error}")
        else:
            toast(request, "Undone." if data["removed"] else f"Undone: {data['goal']['name']}.")
        return back_to_goals(request)

    @app.post("/goals/{goal_id}")
    def update_goal(request: Request, goal_id: str, form: GoalForm) -> Response:
        account = signed_in(request)
        tools = tools_for(account, request)
        goal = editable_goal(tools, goal_id)
        if goal is None:
            return Response(status_code=404)
        try:
            data = tools.update_goal(goal_id, **goal_fields(form)).data
        except GoalProblemsError as error:
            values = form_values(form)
            return goal_form(request, account, goal=goal, values=values, problems=error.problems)
        toast(request, f"Saved {data['goal']['name']}.", data["revision_id"])
        return back_to_goals(request)

    @app.post("/goals/{goal_id}/archive")
    def archive_goal(request: Request, goal_id: str) -> Response:
        tools = tools_for(signed_in(request), request)
        try:
            data = tools.archive_goal(goal_id).data
        except ToolError:  # removed already (a double click, another tab), or not this session's
            toast(request, "That goal isn't there any more.")
            return back_to_goals(request)
        toast(request, f"Removed {data['goal']['name']}.", data["revision_id"])
        return back_to_goals(request)

    # Connect an assistant (FR-19): a personal access token for this user's MCP access

    @app.get("/connect")
    def connect(request: Request) -> Response:
        account = signed_in(request)
        lifetime = timedelta(days=settings.mcp_token_days)
        response = page(
            request,
            "connect.html",
            account,
            active="connect",
            token=tokens.issue(account.user_id, lifetime, feedback_subject=feedback_id(request)),
            expires=(datetime.now() + lifetime).date(),
            mcp_url=f"{public_url}{MCP_PATH}",
        )
        response.headers["Cache-Control"] = "no-store"  # the page renders a credential
        return response

    app.mount("/", mcp_asgi)  # last, so the app's own routes win: it serves only /mcp
    return app
