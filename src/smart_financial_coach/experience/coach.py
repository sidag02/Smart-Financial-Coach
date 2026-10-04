"""The coach: answers questions in plain English using only numbers the tools return.

Claude calls the same tools as the dashboard and outside assistants, through the MCP server with a
token for the signed-in user (`access.mcp_client`), so it can't read anyone else's data however
it's asked (NFR-2). Each tool result carries a source id
(S1, S2, …) and the coach tags every number with one, which the chat shows as a chip (FR-16). It
gives no investment advice (FR-15, NFR-4). With no API key, or when the API fails, chat says so
and the dashboard is unaffected (NFR-6).

    coach = Coach.from_settings(settings)
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
flag charges yourself. If they say a charge isn't theirs, suggest contacting their bank.
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
- Whether a goal is on track isn't available yet. Give check_goal's or list_goals' facts \
without a verdict on whether they'll make it.
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

    def source_id(self, source: Source) -> str:
        self.sources.append(source)
        return f"S{len(self.sources)}"


@dataclass(frozen=True)
class Reply:
    text: str
    cited: dict[str, Source]  # the sources this reply's text cites, by id
    seconds: float


class Coach:
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

        try:
            text, complete = self._loop(tools, conversation)
        except (anthropic.APIError, ToolsUnavailableError) as error:
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

    @staticmethod
    def _run(tools: ToolGateway, conversation: Conversation, call: Any) -> dict[str, Any]:
        try:
            result = tools.call(call.name, dict(call.input))
        except ToolsUnavailableError:
            raise
        except Exception as error:  # reported to the model, which can retry in the same turn
            if not isinstance(error, ToolError):
                log.exception("tool %s failed", call.name)
            message = str(error) if isinstance(error, ToolError) else f"{call.name} failed"
            return {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": message,
                "is_error": True,
            }
        payload = {"source_id": conversation.source_id(result.source), **result.data}
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(payload, separators=(",", ":")),
        }
