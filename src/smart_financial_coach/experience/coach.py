"""The coach: answers questions in plain English using only numbers the tools return.

Claude calls the same tools as the dashboard and outside assistants, through the MCP server with a
token for the signed-in user (`access.mcp_client`), so it can't read anyone else's data however
it's asked (NFR-2). Each tool result carries a source id
(S1, S2, …) and the coach tags every number with one, which the chat shows as a chip (FR-16). It
gives no investment advice (FR-15, NFR-4). With no API key, or when the API fails, chat says so
and the dashboard is unaffected (NFR-6).

    coach = make_coach(settings)  # SFC_COACH_BACKEND: the API, or the owner's subscription
    conversation = Conversation()
    reply = coach.answer(tools, conversation, "Why was September so high?")
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from time import perf_counter
from typing import Any

import anthropic

from smart_financial_coach.access.tools import (
    Source,
    ToolError,
    ToolGateway,
    ToolsUnavailableError,
)
from smart_financial_coach.config import Settings

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6
REFUSED = "I can't help with that one. I can answer questions about your own spending."
TOO_LONG = "That answer ran long. Could you ask about something narrower?"
TOO_MANY_STEPS = "That needed more steps than I can take in one answer. Could you narrow it down?"
EMPTY = "I couldn't put an answer together. Could you ask that another way?"
MAX_TOKENS = 4096
# Server-side fallback when the model's safeguards decline a request
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You are {coach_name}, the money coach in Smart Financial Coach. You help one person \
understand their own spending and savings, in plain English.

How you answer:
- Use only numbers that appear in tool results in this conversation. Never estimate, guess or \
invent a figure. You may add or subtract two numbers that tools returned; anything more needs a \
tool. If no tool gives what the question needs, say what you can't tell yet.
- Put the source id of every number right after it, like "$1,240.50 [S2]". Each tool result has \
a source_id.
- If a tool returns status "not_available", say that feature isn't available yet. Don't fill \
the gap yourself.
- Unusual charges from detect_anomalies look different from the person's usual pattern; that \
doesn't mean anything is wrong. Give the tool's reason, never call a charge fraud, and don't \
flag charges yourself. If they say a charge isn't theirs, suggest contacting their bank, and \
offer to mark it with act_on_flag (not_me); then give the tool's guidance as it is.
- Alerts follow the person's setting: detect_anomalies' sensitivity is how often they asked to \
be told (less, balanced or more), and hidden counts alerts they marked as recognized or expected. \
When nothing is listed but hidden isn't zero, or sensitivity is less, say so; never say nothing \
was unusual. Change the setting (set_alert_sensitivity) or act on an alert (act_on_flag) only \
when they ask. If a tool returns status "needs_confirmation", say what would change and ask; \
call again with confirm true only after they agree. Undo (undo_flag_action) only when asked.
- Spending spikes from detect_anomalies are whole months when a category ran well above the \
person's average month over the past year. Give the tool's reason and numbers, and call its \
largest_charges the largest charges, not the cause. If spending_spikes.status is too_short or \
month_in_progress, say you can't judge that month yet; never call a month a spike yourself. If \
spending_spikes.method is simple_rule, say the check used a simple rule because the \
spending-spike model isn't released, whether or not it found a spike. Asked why a month was high \
with no spike in it, say no category ran well above usual and use get_spending_summary for its \
biggest categories or charges.
- Today is {as_of:%B %-d, %Y}, the latest day with data. "This month" is {as_of:%B %Y} and "last \
month" is the month before.
- Spending amounts in tool results are positive numbers of dollars spent; transaction amounts \
are negative for money out.
- Be warm, brief and non-judgmental: two to five sentences, no headings. Use a short bullet \
list only to list transactions. Write money like $1,234.56 or $1,234.
- Categories (FR-5, FR-6): change one only when the person asks ("that Costco charge is \
groceries"), with correct_category or resolve_review_item. If a tool returns status \
"needs_confirmation", tell them what would change (how many transactions, how much) and ask; \
call again with confirm true only after they agree. After a change, say what moved. If they ask \
what to check, use list_review_items. When unreviewed_spend in a summary is large enough to \
matter for the answer, mention it ("$120 of this is still unconfirmed").
- You can see only the signed-in person's data. If they ask about anyone else's money, say you \
can't access it.
- Savings goals: call check_goal before suggesting or creating a goal and quote its numbers; \
never work out a monthly amount yourself. Ask before any change, and call create_goal, \
update_goal or archive_goal with confirm: true only after the person says yes in this \
conversation. Goals are due at the end of a month: repeat the date check_goal returns, not your \
own reading of theirs. After a change, say what changed and that it can be undone; undo only \
when asked.
- Whether they're on track for a goal: call forecast_goal and quote its status, its range and, \
when it gives one, extra_per_month ("setting aside $75 more a month would put you on track"). \
Never work out a chance, a range or a top-up yourself. Say how sure it is: "could go either way" \
is a real answer. Say when short_history is true (only a few months of history, so it's a rough \
guide), and when share_source is "typical" (a new goal, so it assumes a typical share of their \
savings). The forecast assumes a month where they spend more than they earn draws on what \
they've set aside; say so if they ask why it could fall. For a reached goal, say it's reached; \
mention that months of spending more than they earn could draw it back down only when \
may_draw_down is true. For a goal they're setting up, quote check_goal's fit and forecast.
- When a forecast's method is "simple_projection", say it's a simple projection of their pace \
so far, not a forecast with a chance: quote the projected amount and gap, and never say how \
likely they are to make it.
- Don't give investment, tax or legal advice, or recommend financial products, funds or \
securities. Suggest a licensed professional for those.
"""


class CoachUnavailableError(RuntimeError):
    """The LLM can't be reached or isn't configured; chat shows a clear message (NFR-6)."""


@dataclass
class Conversation:
    """One chat's messages and the sources its answers cite; kept server-side per session."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)  # as shown, for reloading the page
    # Held while a question is answered: two at once would interleave their messages
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # A backend's own state between questions (the subscription backend's session to resume)
    backend_state: dict[str, Any] = field(default_factory=dict)

    def source_id(self, source: Source) -> str:
        self.sources.append(source)
        return f"S{len(self.sources)}"


@dataclass(frozen=True)
class Reply:
    text: str
    cited: dict[str, Source]  # the sources this reply's text cites, by id
    seconds: float


class Coach:
    """The coach on the Anthropic API: the production backend (FR-13 to FR-15 design, §2)."""

    backend = "api"
    # Errors that mean the model can't be reached: chat says so (NFR-6)
    unavailable: tuple[type[Exception], ...] = (anthropic.APIError,)

    def __init__(
        self,
        client: Any,
        *,
        coach_name: str,
        model: str,
        effort: str = "low",
    ) -> None:
        self.client = client
        self.coach_name = coach_name
        self.model = model
        self.effort = effort

    @classmethod
    def from_settings(cls, settings: Settings) -> "Coach | None":
        """The configured coach, or None when there's no API key."""
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        key = key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=1)
        return cls(
            client,
            coach_name=settings.coach_name,
            model=settings.llm_model,
            effort=settings.llm_effort,
        )

    @property
    def credential_source(self) -> str:
        """Which credential the model calls run on, for logs and evaluation results."""
        return "api_key"

    def system_prompt(self, as_of: date) -> str:
        return SYSTEM.format(coach_name=self.coach_name, as_of=as_of)

    def answer(self, tools: ToolGateway, conversation: Conversation, question: str) -> Reply:
        started = perf_counter()
        turn_start, first_source = len(conversation.messages), len(conversation.sources)
        conversation.messages.append({"role": "user", "content": question})

        def roll_back() -> None:
            # Only a turn that ended cleanly stays in the history: anything else (a refusal, a
            # truncated answer, an error) would make the API reject every later question
            del conversation.messages[turn_start:]
            del conversation.sources[first_source:]

        unavailable: tuple[type[Exception], ...] = (*self.unavailable, ToolsUnavailableError)
        try:
            text, complete = self._loop(tools, conversation)
        except unavailable as error:
            log.warning("coach unavailable: %s", type(error).__name__)
            roll_back()
            raise CoachUnavailableError(str(error)) from error
        except Exception:
            roll_back()
            raise
        if not complete:
            roll_back()
        cited = {
            f"S{i + 1}": s for i, s in enumerate(conversation.sources) if f"[S{i + 1}]" in text
        }
        return Reply(text, cited, perf_counter() - started)

    def _loop(self, tools: ToolGateway, conversation: Conversation) -> tuple[str, bool]:
        """The answer, and whether the turn ended cleanly (kept in the history)."""
        system = [
            {
                "type": "text",
                "text": self.system_prompt(tools.as_of),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools.specs,
                messages=conversation.messages,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            # Append the whole content (thinking and fallback blocks included), never edit it
            conversation.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "refusal":
                return REFUSED, False
            if response.stop_reason == "max_tokens":
                return TOO_LONG, False
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                return (text, True) if text else (EMPTY, False)
            conversation.messages.append(
                {"role": "user", "content": [self._run(tools, conversation, c) for c in calls]}
            )
        return TOO_MANY_STEPS, False

    @classmethod
    def _run(cls, tools: ToolGateway, conversation: Conversation, call: Any) -> dict[str, Any]:
        content, is_error = cls.run_tool(tools, conversation, call.name, dict(call.input))
        result = {"type": "tool_result", "tool_use_id": call.id, "content": content}
        return {**result, "is_error": True} if is_error else result

    @staticmethod
    def run_tool(
        tools: ToolGateway, conversation: Conversation, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """One tool call's result as the model sees it, and whether it's an error. Every backend
        runs tools through here, so each result gets its source id the same way."""
        try:
            result = tools.call(name, arguments)
        except ToolsUnavailableError:
            raise
        except Exception as error:  # reported to the model, which can retry in the same turn
            if not isinstance(error, ToolError):
                log.exception("tool %s failed", name)
            return (str(error) if isinstance(error, ToolError) else f"{name} failed"), True
        payload = {"source_id": conversation.source_id(result.source), **result.data}
        return json.dumps(payload, separators=(",", ":")), False


def make_coach(settings: Settings) -> Coach | None:
    """The coach `SFC_COACH_BACKEND` asks for (FR-13 to FR-15 design, §2). `auto` and `api`: the
    API when there's a key, else None and chat says it's unavailable; putting a key in the
    deployment is what makes chat live for every user. `subscription`: the owner's Claude login,
    which refuses to start unless it can only ever serve the owner (decision 2)."""
    if settings.coach_backend == "subscription":
        from smart_financial_coach.experience.coach_subscription import SubscriptionCoach

        return SubscriptionCoach.from_settings(settings)
    return Coach.from_settings(settings)
