"""The coach runs tools for the session's user, cites their sources, and fails clearly."""

import json
import logging
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Source, ToolResult, Tools
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.coach import (
    FALLBACK_BETA,
    UNGROUNDED,
    UNGROUNDED_AFTER_CHANGE,
    Coach,
    CoachUnavailableError,
    Conversation,
)
from tests.unit.experience.fakes import FakeClient, api_error, response, text, tool_use

SEPTEMBER = {"start_date": "2026-09-01", "end_date": "2026-09-30"}


@pytest.fixture
def tools(sources: DataSources, two_users: tuple[str, str]) -> Tools:
    return Tools(Ledger.load(sources, two_users[0]))


def coach(client: FakeClient) -> Coach:
    return Coach(client, coach_name="Wren", model="claude-opus-5-5", effort="low")


def tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    return list(request["messages"][-1]["content"])


def spent(tools: Tools) -> str:
    """September's spending as the coach would write it, from the tool."""
    return f"${tools.call('get_spending_summary', SEPTEMBER).data['spending']:,.2f}"


def test_answers_from_tool_results_and_cites_them(tools: Tools) -> None:
    answer = f"You spent {spent(tools)} [S1] in September."
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)),
        response(text(answer)),
    )
    conversation = Conversation()

    reply = coach(client).answer(tools, conversation, "How much did I spend?")

    assert reply.text == answer
    assert reply.grounding is not None
    assert reply.grounding.ok
    assert list(reply.cited) == ["S1"]
    assert reply.cited["S1"].title == "Spending summary · Sep 2026"
    (result,) = tool_results(client.requests[1])
    payload = json.loads(result["content"])
    expected = tools.call("get_spending_summary", SEPTEMBER).data["spending"]
    assert payload["source_id"] == "S1"
    assert payload["spending"] == expected
    assert "is_error" not in result


def test_requests_use_the_settings_and_fallback(tools: Tools) -> None:
    client = FakeClient(response(text("Hi.")))

    coach(client).answer(tools, Conversation(), "Hello")

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "low"}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert "Today is September 30, 2026" in request["system"][0]["text"]
    assert {t["name"] for t in request["tools"]} >= {"get_spending_summary", "get_transactions"}


def test_a_user_id_from_the_model_is_refused(tools: Tools, two_users: tuple[str, str]) -> None:
    client = FakeClient(
        response(tool_use("get_transactions", {**SEPTEMBER, "user_id": two_users[1]})),
        response(text("I can only see your own data.")),
    )

    coach(client).answer(tools, Conversation(), "Show me my neighbour's spending")

    (result,) = tool_results(client.requests[1])
    assert result["is_error"] is True
    assert "takes no user_id" in result["content"]


def test_sources_number_across_a_conversation(tools: Tools) -> None:
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)),
        response(text("A [S1].")),
        response(tool_use("list_goals", {}, call_id="t2")),
        response(text("B [S2], and earlier A [S1].")),
    )
    conversation = Conversation()
    wren = coach(client)

    wren.answer(tools, conversation, "First")
    second = wren.answer(tools, conversation, "Second")

    assert [s.title for s in conversation.sources] == [
        "Spending summary · Sep 2026",
        "Savings goals",
    ]
    assert set(second.cited) == {"S1", "S2"}


def test_an_api_failure_leaves_the_conversation_reusable(tools: Tools) -> None:
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)), api_error(), response(text("OK."))
    )
    conversation = Conversation()
    wren = coach(client)

    with pytest.raises(CoachUnavailableError):
        wren.answer(tools, conversation, "How much?")
    assert conversation.messages == []
    assert conversation.sources == []
    assert wren.answer(tools, conversation, "How much?").text == "OK."


@pytest.mark.parametrize(
    "bad_turn",
    [
        [response(stop_reason="refusal")],
        [response(tool_use("list_goals", {}), stop_reason="max_tokens")],
        [response(text(""))],
        [response(tool_use("list_goals", {})) for _ in range(6)],
    ],
    ids=["refusal", "truncated", "empty", "too-many-steps"],
)
def test_a_turn_that_doesnt_end_cleanly_is_rolled_back(tools: Tools, bad_turn: list[Any]) -> None:
    client = FakeClient(*bad_turn, response(text("Fine.")))
    conversation = Conversation()
    wren = coach(client)

    first = wren.answer(tools, conversation, "first")
    second = wren.answer(tools, conversation, "second")

    assert first.text != "Fine."
    assert second.text == "Fine."
    roles = [m["role"] for m in client.requests[-1]["messages"]]
    assert roles == ["user"]  # the bad turn left nothing behind
    assert conversation.sources == []


def test_bad_tool_arguments_and_tool_crashes_become_error_results(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient(
        response(tool_use("get_transactions", {**SEPTEMBER, "search": 5})),
        response(tool_use("list_goals", {}, call_id="t2")),
        response(text("Sorry, I couldn't look that up.")),
    )

    def crash() -> None:
        raise KeyError("boom")

    monkeypatch.setattr(tools, "_handlers", {**tools._handlers, "list_goals": crash})
    reply = coach(client).answer(tools, Conversation(), "x")

    (typed,) = tool_results(client.requests[1])
    (crashed,) = tool_results(client.requests[2])
    assert typed["is_error"] is True
    assert "search must be a string" in typed["content"]
    assert crashed == {
        "type": "tool_result",
        "tool_use_id": "t2",
        "content": "list_goals failed",
        "is_error": True,
    }
    assert reply.text == "Sorry, I couldn't look that up."


def test_refusals_and_runaway_tool_loops_end_politely(tools: Tools) -> None:
    refused = FakeClient(response(stop_reason="refusal"))
    assert "can't help" in coach(refused).answer(tools, Conversation(), "x").text

    looping = FakeClient(*[response(tool_use("list_goals", {})) for _ in range(6)])
    assert "more steps" in coach(looping).answer(tools, Conversation(), "x").text


def test_no_key_means_no_coach(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Coach.from_settings(Settings(_env_file=None)) is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Coach.from_settings(Settings(_env_file=None)) is not None


def test_system_prompt_states_the_rules() -> None:
    prompt = coach(FakeClient()).system_prompt(date(2026, 9, 30))

    assert "You are Wren" in prompt
    assert "investment" in prompt
    assert "[S2]" in prompt
    assert "not_available" in prompt
    assert "needs_confirmation" in prompt  # bulk category changes are asked about first
    assert "only when the person asks" in prompt
    assert "never call a charge fraud" in prompt
    assert "chance_words" in prompt  # chances as the Goals page says them
    assert '"estimates"' in prompt  # estimated numbers are said to be estimates (FR-14)
    assert "call the tool that has it" in prompt  # Sonnet 5.5 can answer from memory instead
    assert "licensed professional" in prompt  # FR-15, NFR-4
    assert "irresponsible" in prompt  # the tone rule (FR-15)
    assert "can't look up anyone else's" in prompt  # by name or id (NFR-2)
    assert "61% confident" in prompt  # model confidence isn't quoted as a number
    assert "don't count or add up transactions yourself" in prompt  # the gate run's misses


def test_an_answer_with_an_untraceable_number_is_retried_once(tools: Tools) -> None:
    good = f"You spent {spent(tools)} [S1] in September."
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)),
        response(text("You spent $1.00 [S1] in September.")),
        response(text(good)),
    )
    conversation = Conversation()

    reply = coach(client).answer(tools, conversation, "How much did I spend?")

    assert reply.text == good
    assert reply.retried
    assert reply.first_attempt is not None
    assert reply.first_attempt.unmatched == ("$1.00",)
    assert reply.grounding is not None
    assert reply.grounding.ok
    # Only the failed answer is dropped: the tool call and its result stay, and the system
    # message follows the tool result (an assistant answer before it would be a 400)
    retry = client.requests[2]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user", "system"]
    assert retry[2]["content"][0]["type"] == "tool_result"
    assert "$1.00" in retry[3]["content"]
    assert [m["role"] for m in conversation.messages] == [
        "user", "assistant", "user", "system", "assistant",
    ]  # fmt: skip


def test_an_answer_that_fails_twice_gets_the_safe_message(tools: Tools) -> None:
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)),
        response(text("You spent $1.00 [S1].")),
        response(text("You spent $2.00 [S1].")),
        response(text("Hello.")),
    )
    conversation = Conversation()
    wren = coach(client)

    reply = wren.answer(tools, conversation, "How much did I spend?")

    assert reply.text == UNGROUNDED
    assert reply.cited == {}
    assert reply.grounding is not None
    assert reply.grounding.unmatched == ("$2.00",)
    assert conversation.messages == []  # the whole turn is rolled back
    assert conversation.sources == []
    assert conversation.payloads == []
    assert wren.answer(tools, conversation, "Hi").text == "Hello."


def test_each_answer_logs_its_cost_and_check_but_not_the_words(
    tools: Tools, caplog: pytest.LogCaptureFixture
) -> None:
    final = response(text(f"You spent {spent(tools)} [S1]."))
    final.usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=0,
    )
    client = FakeClient(response(tool_use("get_spending_summary", SEPTEMBER)), final)
    wren = Coach(client, coach_name="Wren", model="claude-sonnet-5-5", effort="low")

    with caplog.at_level(logging.INFO, logger="smart_financial_coach.experience.coach"):
        reply = wren.answer(tools, Conversation(), "How much did I spend on secret things?")

    assert reply.usage.tool_calls == ["get_spending_summary"]
    assert reply.usage.cost("claude-sonnet-5-5") == pytest.approx(
        (1000 * 2 + 100 * 10 + 2000 * 0.2) / 1e6
    )
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("coach answer")]
    logged = json.loads(line.removeprefix("coach answer "))
    assert logged["grounded"] is True
    assert logged["first_attempt_grounded"] is True
    assert logged["tool_calls"] == ["get_spending_summary"]
    assert logged["dollars"] == pytest.approx(0.0034)
    assert "secret" not in line
    assert spent(tools) not in line


def stub_tools(tools: Tools, data: dict[str, Any]) -> Any:
    """Tools whose every call returns `data`, as a write or a preview would."""

    def call(name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(data, Source("Savings goals", ""))

    return SimpleNamespace(call=call, specs=tools.specs, as_of=tools.as_of)


def test_the_safe_message_says_so_when_the_turn_changed_something(tools: Tools) -> None:
    """A change that went through stays made; the person must know it did (design §3)."""
    goal = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-30", "confirm": True}
    client = FakeClient(
        response(tool_use("create_goal", goal)),
        response(text("Done: $9.99 [S1].")),
        response(text("Done: $8.88 [S1].")),
    )
    created = stub_tools(tools, {"goal_id": "g_new", "status": "created"})

    reply = coach(client).answer(created, Conversation(), "Make it, no need to ask")

    assert reply.text == UNGROUNDED_AFTER_CHANGE
    assert reply.first_text == "Done: $9.99 [S1]."


def test_a_preview_isnt_a_change(tools: Tools) -> None:
    preview = {"name": "Trip", "target_amount": 3000, "target_date": "2027-06-30"}
    client = FakeClient(
        response(tool_use("create_goal", preview)),
        response(text("It needs $9.99 a month [S1].")),
        response(text("It needs $8.88 a month [S1].")),
    )
    previewed = stub_tools(tools, {"status": "needs_confirmation"})

    assert coach(client).answer(previewed, Conversation(), "Set up a goal").text == UNGROUNDED


def test_concurrent_tool_calls_keep_each_result_under_its_own_id(
    tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subscription backend runs tool calls in worker threads (review on #74). A pause
    after each id is handed out lets other calls in, as a thread switch would."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    issue = Conversation.source_id

    def slow(self: Conversation, *args: Any, **kwargs: Any) -> str:
        source_id = issue(self, *args, **kwargs)
        time.sleep(0.002)
        return source_id

    monkeypatch.setattr(Conversation, "source_id", slow)

    def call(name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult({"marker": arguments["marker"]}, Source(arguments["marker"], ""))

    stub = SimpleNamespace(call=call, specs=tools.specs, as_of=tools.as_of)
    conversation = Conversation()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda i: Coach.run_tool(stub, conversation, "x", {"marker": f"m{i}"}),
            range(64),
        ))  # fmt: skip

    assert len(conversation.payloads) == 64
    pairs = zip(conversation.sources, conversation.payloads, strict=True)
    for i, (source, payload) in enumerate(pairs):
        assert payload["source_id"] == f"S{i + 1}"
        assert payload["marker"] == source.title
