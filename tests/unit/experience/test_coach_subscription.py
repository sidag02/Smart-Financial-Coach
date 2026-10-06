"""The subscription backend serves only its owner, and runs the same coach as the API backend."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    SystemMessage,
    TextBlock,
)
from pydantic import SecretStr

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.config import Settings
from smart_financial_coach.experience import coach_subscription
from smart_financial_coach.experience.coach import (
    UNGROUNDED,
    Coach,
    CoachUnavailableError,
    Conversation,
    make_coach,
)
from smart_financial_coach.experience.coach_subscription import (
    KEY_VARIABLES,
    SubscriptionCoach,
    SubscriptionNotAllowedError,
    check_allowed,
)

SEPTEMBER = {"start_date": "2026-09-01", "end_date": "2026-09-30"}
Step = Callable[[ClaudeAgentOptions, dict[str, Any]], Any]


@pytest.fixture
def tools(sources: DataSources, two_users: tuple[str, str]) -> Tools:
    return Tools(Ledger.load(sources, two_users[0]))


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def init(source: str = "none") -> SystemMessage:
    return SystemMessage("init", {"apiKeySource": source})


def answer(text: str, uuid: str) -> AssistantMessage:
    return AssistantMessage([TextBlock(text)], "claude-sonnet-5-5", uuid=uuid)


def result(text: str | None, session: str, **fields: Any) -> ResultMessage:
    base: dict[str, Any] = {"stop_reason": "end_turn", "is_error": False, "subtype": "success"}
    return ResultMessage(
        duration_ms=1, duration_api_ms=1, num_turns=1, session_id=session, result=text,
        **{**base, **fields},
    )  # fmt: skip


def call(name: str, arguments: dict[str, Any]) -> Step:
    """A step that calls one of the coach's wrapped tools, as Claude Code would."""

    async def run(options: ClaudeAgentOptions, servers: dict[str, Any]) -> None:
        (tool,) = [t for t in servers["tools"] if t.name == name]
        output = await tool.handler(arguments)
        servers.setdefault("outputs", []).append(output)

    return run


class FakeAgentSdk:
    """Scripted Claude Code runs: each question plays one script of messages and tool calls."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *scripts: list[Any]) -> None:
        self.scripts = list(scripts)
        self.options: list[ClaudeAgentOptions] = []
        self.servers: list[dict[str, Any]] = []
        monkeypatch.setattr(coach_subscription, "query", self.query)
        monkeypatch.setattr(
            coach_subscription, "create_sdk_mcp_server", lambda name, tools: {"tools": tools}
        )

    async def query(self, *, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        self.options.append(options)
        servers = cast(dict[str, dict[str, Any]], options.mcp_servers)[coach_subscription.SERVER]
        self.servers.append(servers)
        for step in self.scripts.pop(0):
            if isinstance(step, Exception):
                raise step
            if callable(step):
                await step(options, servers)
            else:
                yield step


def wren() -> SubscriptionCoach:
    return SubscriptionCoach(coach_name="Wren", model="claude-sonnet-5-5", effort="low")


def test_the_api_coach_defaults_to_sonnet() -> None:
    assert settings().llm_model == "claude-sonnet-5-5"
    assert settings().coach_backend == "auto"


@pytest.mark.usefixtures("no_keys")
def test_only_the_owners_machine_with_no_key_may_use_it(monkeypatch: pytest.MonkeyPatch) -> None:
    check_allowed(settings(public_url="http://127.0.0.1:8000"), environ={})
    check_allowed(settings(public_url="http://localhost:8000"), environ={})

    with pytest.raises(SubscriptionNotAllowedError, match="loopback"):
        check_allowed(settings(public_url="https://sfc.example.com"), environ={})
    for name in KEY_VARIABLES:
        with pytest.raises(SubscriptionNotAllowedError, match=name):
            check_allowed(settings(), environ={name: "sk-test"})
    with pytest.raises(SubscriptionNotAllowedError, match="SFC_LLM_API_KEY"):
        check_allowed(settings(llm_api_key=SecretStr("sk-test")), environ={})


@pytest.mark.usefixtures("no_keys")
def test_the_backend_setting_picks_the_coach(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_coach(settings()) is None  # no key, no chat
    assert make_coach(settings(coach_backend="api")) is None
    assert isinstance(make_coach(settings(coach_backend="subscription")), SubscriptionCoach)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    api = make_coach(settings())
    assert type(api) is Coach
    assert api.credential_source == "api_key"
    with pytest.raises(SubscriptionNotAllowedError):  # never quietly bills the key
        make_coach(settings(coach_backend="subscription"))


def test_it_runs_the_api_coachs_settings_with_only_the_coachs_tools(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    spent = tools.call("get_spending_summary", SEPTEMBER).data["spending"]
    text = f"You spent ${spent:,.2f} [S1]."
    sdk = FakeAgentSdk(
        monkeypatch,
        [init(), call("get_spending_summary", SEPTEMBER), answer(text, "a1"), result(text, "s1")],
    )
    coach = wren()

    reply = coach.answer(tools, Conversation(), "How much did I spend?")

    (options,) = sdk.options
    assert options.model == "claude-sonnet-5-5"
    assert options.effort == "low"
    assert options.thinking == {"type": "adaptive"}
    assert options.tools == []  # no files, shell or web
    assert options.setting_sources == []
    assert options.skills == []
    assert options.strict_mcp_config is True
    assert options.permission_mode == "dontAsk"
    assert set(options.allowed_tools) == {f"mcp__coach__{s['name']}" for s in tools.specs}
    assert "You are Wren" in str(options.system_prompt)
    assert options.resume is None
    (output,) = sdk.servers[0]["outputs"]
    payload = json.loads(output["content"][0]["text"])
    assert payload["source_id"] == "S1"
    assert payload["spending"] == tools.call("get_spending_summary", SEPTEMBER).data["spending"]
    assert reply.text == text
    assert reply.cited["S1"].title == "Spending summary · Sep 2026"
    assert coach.credential_source == "subscription"


def test_tool_errors_reach_the_model_and_a_user_id_is_refused(
    tools: Tools, two_users: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeAgentSdk(
        monkeypatch,
        [
            init(),
            call("get_transactions", {**SEPTEMBER, "user_id": two_users[1]}),
            answer("I can only see your own data.", "a1"),
            result("I can only see your own data.", "s1"),
        ],
    )

    wren().answer(tools, Conversation(), "Show me my neighbour's spending")

    (output,) = sdk.servers[0]["outputs"]
    assert output["is_error"] is True
    assert "takes no user_id" in output["content"][0]["text"]


def test_the_next_question_resumes_at_the_last_clean_answer(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeAgentSdk(
        monkeypatch,
        [init(), answer("First.", "a1"), result("First.", "s1")],
        [init(), call("list_goals", {}), result(None, "s2", stop_reason="refusal")],
        [init(), answer("Third.", "a3"), result("Third.", "s3")],
        [init(), answer("Fourth.", "a4"), result("Fourth.", "s4")],
    )
    coach, conversation = wren(), Conversation()

    coach.answer(tools, conversation, "one")
    refused = coach.answer(tools, conversation, "two")
    coach.answer(tools, conversation, "three")
    coach.answer(tools, conversation, "four")

    assert "can't help" in refused.text
    assert conversation.sources == []  # the refused turn's source is gone
    first, second, third, fourth = sdk.options
    assert (first.resume, first.fork_session) == (None, False)
    # The refused turn is dropped: the third question resumes where the first answer ended
    assert (second.resume, second.resume_session_at, second.fork_session) == ("s1", "a1", True)
    assert (third.resume, third.resume_session_at) == ("s1", "a1")
    assert (fourth.resume, fourth.resume_session_at) == ("s3", "a3")
    assert [m["role"] for m in conversation.messages] == ["user", "assistant"] * 3


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        ({"subtype": "error_max_turns", "is_error": True}, "more steps"),
        ({"stop_reason": "max_tokens"}, "ran long"),
        ({}, "couldn't put an answer together"),
    ],
    ids=["too-many-steps", "truncated", "empty"],
)
def test_a_turn_that_doesnt_end_cleanly_ends_politely(
    tools: Tools, monkeypatch: pytest.MonkeyPatch, ending: dict[str, Any], expected: str
) -> None:
    FakeAgentSdk(monkeypatch, [init(), result("", "s1", **ending)])
    conversation = Conversation()

    reply = wren().answer(tools, conversation, "x")

    assert expected in reply.text
    assert conversation.messages == []
    assert conversation.backend_state == {}


def test_claude_code_failing_means_the_coach_is_unavailable(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeAgentSdk(
        monkeypatch,
        [init(), ClaudeSDKError("CLI not found")],
        [init(), result(None, "s1", subtype="error_during_execution", is_error=True)],
    )
    coach, conversation = wren(), Conversation()

    for _ in range(2):
        with pytest.raises(CoachUnavailableError):
            coach.answer(tools, conversation, "x")
    assert conversation.messages == []


def test_it_stops_if_claude_code_runs_on_anything_but_the_login(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeAgentSdk(monkeypatch, [init("ANTHROPIC_API_KEY"), answer("Hi.", "a1"), result("Hi.", "s1")])
    coach = wren()

    with pytest.raises(SubscriptionNotAllowedError, match="ANTHROPIC_API_KEY"):
        coach.answer(tools, Conversation(), "x")
    assert coach.credential_source == "ANTHROPIC_API_KEY"


def test_an_untraceable_number_gets_the_safe_message_with_no_retry(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = FakeAgentSdk(
        monkeypatch,
        [init(), call("get_spending_summary", SEPTEMBER), answer("$1.00 [S1].", "a1"),
         result("You spent $1.00 [S1].", "s1")],
        [init(), answer("Hello.", "a2"), result("Hello.", "s2")],
    )  # fmt: skip
    coach, conversation = wren(), Conversation()

    reply = coach.answer(tools, conversation, "How much did I spend?")
    coach.answer(tools, conversation, "Hi")

    assert reply.text == UNGROUNDED
    assert not reply.retried  # one Claude Code run: no retry on this backend (design §3)
    assert conversation.sources == []
    # The failed turn isn't resumed from: the next question starts afresh
    assert sdk.options[1].resume is None
