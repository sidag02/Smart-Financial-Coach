"""The coach on the owner's Claude subscription: evaluation runs and prompt work, never users.

FR-13 to FR-15 design, §2 (owner decisions 2 and 5). The model runs through the Claude Agent SDK
(Claude Code) on the owner's claude.ai login, with the API coach's system prompt, model, effort
and adaptive thinking. Each tool is wrapped in process and runs through `Coach.run_tool`, so it
still calls the app's `/mcp` with the coach's token for one user and gets a source id. No
built-in tool (files, shell, web), skill, filesystem setting or other MCP server is loaded.

A claude.ai login may serve only its owner, so this refuses to start unless the app's address is
a loopback address and nothing in the environment would move Claude Code off the login (an API
key, a cloud provider or another endpoint) and bill it. It also stops if Claude Code reports
running on any credential but the login.

Known differences from the API backend (design §2): Claude Code's loop, no server-side refusal
fallback, and no grounding retry. A failed turn is dropped by resuming the session at the last
clean answer, as the API backend drops it from its history.

    coach = SubscriptionCoach.from_settings(settings)  # SubscriptionNotAllowedError if it can't
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import anyio.to_thread
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    SdkMcpTool,
    SystemMessage,
    create_sdk_mcp_server,
    query,
)

from smart_financial_coach.access.tools import ToolGateway, ToolSpec, ToolsUnavailableError
from smart_financial_coach.config import Settings
from smart_financial_coach.experience.coach import (
    EMPTY,
    MAX_TOOL_ROUNDS,
    REFUSED,
    TOO_LONG,
    TOO_MANY_STEPS,
    Coach,
    Conversation,
    Usage,
)

log = logging.getLogger(__name__)

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
# What would take Claude Code off the subscription login and bill something else: a key or token
# it uses first, the app's own key, a cloud provider, or another endpoint (review on #73)
KEY_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "SFC_LLM_API_KEY",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "ANTHROPIC_BASE_URL",
)
SERVER = "coach"
SUBSCRIPTION = "none"  # Claude Code's apiKeySource when it runs on the claude.ai login


class SubscriptionNotAllowedError(RuntimeError):
    """The subscription backend could serve someone other than its owner, or bill a key."""


def check_allowed(settings: Settings, environ: Mapping[str, str] = os.environ) -> None:
    host = urlsplit(settings.public_url).hostname or ""
    if host not in LOOPBACK:
        raise SubscriptionNotAllowedError(
            "the subscription backend serves only its owner: SFC_PUBLIC_URL must be a loopback "
            f"address, not {host!r}"
        )
    keys = [name for name in KEY_VARIABLES if environ.get(name)]
    if settings.llm_api_key is not None and "SFC_LLM_API_KEY" not in keys:
        keys.append("SFC_LLM_API_KEY")
    if keys:
        raise SubscriptionNotAllowedError(
            f"unset {', '.join(keys)} to use the subscription backend: Claude Code would run on "
            "that instead of the subscription, and bill it"
        )


class SubscriptionCoach(Coach):
    backend = "subscription"
    unavailable = (ClaudeSDKError,)
    # No grounding retry: the Agent SDK has no mid-conversation system message, and a follow-up
    # user turn would leave the failed answer in the session (design §3). A failing answer goes
    # straight to the safe message
    retries_grounding = False

    def __init__(
        self, *, coach_name: str, model: str, effort: str = "low", cwd: Path | None = None
    ) -> None:
        super().__init__(None, coach_name=coach_name, model=model, effort=effort)
        self.cwd = cwd  # where Claude Code runs; it loads no settings or CLAUDE.md from there
        self._credential: str | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "SubscriptionCoach":
        check_allowed(settings)
        return cls(
            coach_name=settings.coach_name, model=settings.llm_model, effort=settings.llm_effort
        )

    @property
    def credential_source(self) -> str:
        if self._credential is None:
            return "not run yet"
        return "subscription" if self._credential == SUBSCRIPTION else self._credential

    def options(
        self, system: str, specs: list[ToolSpec], state: dict[str, Any]
    ) -> ClaudeAgentOptions:
        """The Agent SDK's settings: the API coach's model, effort and thinking, only the coach's
        tools, nothing from the filesystem; resumed at the last clean answer, as a fork."""
        return ClaudeAgentOptions(
            system_prompt=system,
            model=self.model,
            effort=self.effort,  # type: ignore[arg-type]
            thinking={"type": "adaptive"},
            tools=[],
            allowed_tools=[f"mcp__{SERVER}__{spec['name']}" for spec in specs],
            strict_mcp_config=True,
            setting_sources=[],
            skills=[],
            permission_mode="dontAsk",
            max_turns=MAX_TOOL_ROUNDS,
            cwd=self.cwd,
            resume=state.get("session_id"),
            resume_session_at=state.get("resume_at"),
            fork_session="session_id" in state,
        )

    def _loop(
        self, tools: ToolGateway, conversation: Conversation, usage: Usage
    ) -> tuple[str, bool]:
        question = conversation.messages[-1]["content"]
        specs = tools.specs  # before the private event loop: McpTools blocks on its own
        return anyio.run(self._ask, tools, conversation, specs, question, usage)

    def _keep(self, conversation: Conversation, text: str) -> None:
        """The next question resumes after this answer, now that it passed the check."""
        conversation.backend_state.update(conversation.backend_state.pop("pending"))
        conversation.messages.append({"role": "assistant", "content": text})

    async def _ask(
        self,
        tools: ToolGateway,
        conversation: Conversation,
        specs: list[ToolSpec],
        question: str,
        usage: Usage,
    ) -> tuple[str, bool]:
        lost: list[ToolsUnavailableError] = []  # the MCP server failed: no tools at all

        def wrap(spec: ToolSpec) -> SdkMcpTool[Any]:
            name = spec["name"]

            async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
                usage.tool_calls.append(name)
                try:
                    content, is_error = await anyio.to_thread.run_sync(
                        self.run_tool, tools, conversation, name, dict(arguments)
                    )
                except ToolsUnavailableError as error:
                    lost.append(error)
                    content, is_error = f"{name} is unavailable", True
                return {"content": [{"type": "text", "text": content}], "is_error": is_error}

            return SdkMcpTool(name, spec["description"], spec["input_schema"], handler)

        state = conversation.backend_state
        options = self.options(self.system_prompt(tools.as_of), specs, state)
        options.mcp_servers = {
            SERVER: create_sdk_mcp_server(SERVER, tools=[wrap(s) for s in specs])
        }
        last_answer: str | None = None
        result: ResultMessage | None = None
        async for message in query(prompt=question, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                self._credential = message.data.get("apiKeySource")
                log.info("coach backend subscription, credential %s", self.credential_source)
                if self._credential != SUBSCRIPTION:
                    raise SubscriptionNotAllowedError(
                        f"Claude Code ran on {self._credential!r}, not the subscription login"
                    )
            elif isinstance(message, AssistantMessage):
                last_answer = message.uuid
            elif isinstance(message, ResultMessage):
                result = message
        if lost:
            raise lost[0]
        if result is None:
            raise ClaudeSDKError("Claude Code ended without a result")
        usage.add(result.usage)
        usage.billed = False  # tokens only: the subscription isn't charged per answer (§7)
        if result.stop_reason == "refusal":
            return REFUSED, False
        if result.subtype == "error_max_turns":
            return TOO_MANY_STEPS, False
        if result.stop_reason == "max_tokens":
            return TOO_LONG, False
        if result.is_error:
            raise ClaudeSDKError(f"Claude Code: {result.subtype}")
        text = (result.result or "").strip()
        if not text:
            return EMPTY, False
        # Kept only if the answer passes the grounding check (`_keep`)
        state["pending"] = {"session_id": result.session_id, "resume_at": last_answer}
        return text, True
