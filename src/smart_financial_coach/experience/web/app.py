"""The web app: sign-in, overview, transactions, coach chat (mockups 1a, 1d, 1e, 1i).

Server-rendered with Jinja templates and htmx (Web App UI, decision 7). Identity lives only in
the signed session cookie; every page builds its `Tools` from that session's user, never from a
request parameter, so no URL or form field can name another user's data (NFR-2).

    app = create_app(get_settings())
"""

import html
import logging
import re
import secrets
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from anyio.lowlevel import EventLoopToken, current_token
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from starlette.middleware.sessions import SessionMiddleware

from smart_financial_coach.access.feedback import FeedbackStore
from smart_financial_coach.access.ledger import INCOME, DataSources, Ledger
from smart_financial_coach.access.mcp_client import McpTools
from smart_financial_coach.access.mcp_server import PATH as MCP_PATH
from smart_financial_coach.access.mcp_server import build_mcp_server
from smart_financial_coach.access.review_items import alternatives, open_review_items
from smart_financial_coach.access.tokens import AccessTokens
from smart_financial_coach.access.tools import Feedback, Source, ToolError, Tools, span_label
from smart_financial_coach.config import PROJECT_ROOT, Settings
from smart_financial_coach.experience.accounts import Account, SharedPassword, load_accounts
from smart_financial_coach.experience.coach import Coach, CoachUnavailableError, Conversation
from smart_financial_coach.experience.demo import ACCOUNTS_FILE
from smart_financial_coach.experience.web import charts
from smart_financial_coach.experience.web.limits import RateLimit

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


def money(value: float, cents: bool = False, sign: bool = False) -> str:
    text = f"${abs(value):,.2f}" if cents else f"${abs(round(value)):,.0f}"
    if value < 0 and (sign or not cents):
        return MINUS + text
    if value > 0 and sign:
        return "+" + text
    return text


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
    mcp_server, mcp_asgi = build_mcp_server(
        sources, tokens, public_url=public_url, feedback=feedback
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
    templates.env.filters.update(money=money, day=day, month_name=month_name)
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
        return Tools(
            Ledger.load(sources, account.user_id), Feedback(feedback, feedback_id(request))
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
                "flags_live": flags_live,
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
                Ledger.load(sources, account.user_id)  # cached after the first check
        except Exception:
            log.exception("health check: the demo data doesn't load")
            return JSONResponse({"status": "error", "as_of": as_of.isoformat()}, 503)
        return JSONResponse({"status": "ok", "as_of": as_of.isoformat(), "users": len(accounts)})

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
            active_goals=[
                g for g in tools.list_goals().data["goals"] if g["target_date"] >= as_of.isoformat()
            ],
            trend=charts.trend(months, month_key(period.start)),
            flags=recent_flags(tools) if flags_live else [],
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

    def recent_flags(tools: Tools) -> list[dict[str, Any]]:
        """The last `FLAG_WINDOW_DAYS` days' unusual charges, newest first (FR-7 §8)."""
        start = as_of - timedelta(days=FLAG_WINDOW_DAYS - 1)
        found = tools.detect_anomalies(start.isoformat(), as_of.isoformat()).data
        flags: list[dict[str, Any]] = found.get("unusual_transactions", [])
        return sorted(flags, key=lambda f: (f["date"], f["transaction_id"]), reverse=True)

    @app.get("/worth-a-look")
    def worth_a_look(request: Request) -> Response:
        account = signed_in(request)
        if not flags_live:
            return page(request, "coming.html", account, active="flags", feature="flags", goals=[])
        return page(
            request,
            "worth_a_look.html",
            account,
            active="flags",
            flags=recent_flags(tools_for(account, request)),
            start=(as_of - timedelta(days=FLAG_WINDOW_DAYS - 1)).isoformat(),
            end=as_of.isoformat(),
            window_days=FLAG_WINDOW_DAYS,
        )

    @app.get("/goals")
    def goals(request: Request) -> Response:
        account = signed_in(request)
        listed = tools_for(account, request).list_goals().data["goals"]
        return page(request, "coming.html", account, active="goals", feature="goals", goals=listed)

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
