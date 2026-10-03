"""The coach runs tools for the session's user, cites their sources, and fails clearly."""

import json
from datetime import date
from typing import Any

import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.coach import (
    FALLBACK_BETA,
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


def test_answers_from_tool_results_and_cites_them(tools: Tools) -> None:
    client = FakeClient(
        response(tool_use("get_spending_summary", SEPTEMBER)),
        response(text("You spent $1,000.00 [S1] in September.")),
    )
    conversation = Conversation()

    reply = coach(client).answer(tools, conversation, "How much did I spend?")

    assert reply.text == "You spent $1,000.00 [S1] in September."
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
