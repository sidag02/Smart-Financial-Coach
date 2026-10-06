"""`sfc-coach`: ask the coach questions from the command line, as one demo user.

sfc-coach ask --user u_te_yp_0030 "Why was August so high?"
sfc-coach ask --backend subscription --user u_te_yp_0030 "Am I on track?" "And by how much?"
sfc-coach eval --backend subscription --gate          # grounding, safety, rubric (decision 5)
sfc-coach eval --backend api --latency-cost            # latency and cost: one run, no judge

Questions after the first continue the same conversation. The app's MCP server runs in process
with the demo bundle (`sfc-web build-demo`), and the coach calls it with a token for the user, as
in the web app. Category corrections, goals and alert settings the coach changes go to a
throwaway directory, never the local app's stores. `--backend subscription` runs on the owner's
Claude login (FR-13 to FR-15 design, §2): unset any API key first.

`eval` runs the coach evaluation suite (`configs/coach_eval/cases.yaml`, design §5) and writes a
JSON result, a Markdown summary and a judge hand-check sheet next to it. `--gate` exits non-zero
when a target is missed.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from smart_financial_coach.config import get_settings
from smart_financial_coach.experience.coach import CoachUnavailableError, Conversation, make_coach
from smart_financial_coach.experience.coach_harness import LocalApp, local_settings


def _ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings = local_settings(settings, coach_backend=args.backend or settings.coach_backend)
    coach = make_coach(settings)
    if coach is None:
        print("No API key (SFC_LLM_API_KEY or ANTHROPIC_API_KEY): try --backend subscription")
        return 1
    with LocalApp(settings) as local:
        if args.user not in local.users:
            print(f"{args.user} isn't a demo account: {', '.join(local.users)}")
            return 1
        tools = local.tools(args.user)
        conversation = Conversation()
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
            check = reply.first_attempt
            grounded = "n/a" if check is None else ("yes" if check.ok else f"no {check.unmatched}")
            print(
                f"  ({reply.seconds:.1f} s, {coach.backend} backend, {coach.model}, "
                f"effort {coach.effort}, credential {coach.credential_source}; "
                f"grounded first time: {grounded}{', retried' if reply.retried else ''})\n"
            )
    return 0


def _eval(args: argparse.Namespace) -> int:
    from smart_financial_coach.evaluation.coach_judge import Judge
    from smart_financial_coach.evaluation.coach_suite import (
        CASES,
        RunResult,
        Suite,
        gate,
        git_commit,
        hand_check,
        load_cases,
        meta,
    )

    settings = get_settings()
    settings = local_settings(settings, coach_backend=args.backend or settings.coach_backend)
    coach = make_coach(settings)
    if coach is None:
        print("No API key (SFC_LLM_API_KEY or ANTHROPIC_API_KEY): try --backend subscription")
        return 1
    cases = load_cases(args.cases_file or CASES)
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.id in wanted or c.group in wanted]
    runs, judged = (1, False) if args.latency_cost else (args.runs, not args.no_judge)
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    judge = Judge.for_backend(coach.backend, api_key=key) if judged else None
    out: Path = args.out or Path("build") / "coach-eval" / (
        f"{coach.backend}-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    def progress(r: RunResult) -> None:
        mark = "pass" if r.passed else f"FAIL {r.error or ''}".strip()
        print(f"  {r.case} #{r.run}: {mark}", flush=True)

    commit, started = git_commit(), datetime.now(UTC)  # before the run, which takes a while
    with LocalApp(settings) as local:
        print(f"{len(cases)} cases x {runs} runs on the {coach.backend} backend ({coach.model})")
        suite = Suite(coach, judge, local, cases, progress=progress)
        result = suite.run(runs=runs, workers=args.workers)
        info = meta(coach, judge, local, args.cases_file or CASES, commit=commit, started=started)
        result = {"meta": info, **result}
    out.write_text(json.dumps(result, indent=2, default=str))
    if judged:
        out.with_suffix(".handcheck.md").write_text(hand_check(result["runs"]))
    summary = result["summary"]
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    missed = gate(summary, latency=coach.backend == "api")
    if args.gate and missed:
        print("Gate missed: " + "; ".join(missed))
        return 2
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
    ev = sub.add_parser("eval", help="run the coach evaluation suite")
    ev.add_argument("--backend", choices=("auto", "api", "subscription"))
    ev.add_argument("--runs", type=int, default=3, help="runs per case (default 3)")
    ev.add_argument("--workers", type=int, default=4, help="cases run at once (default 4)")
    ev.add_argument("--cases", help="comma-separated case ids or groups (default: all)")
    ev.add_argument("--cases-file", type=Path, default=None)
    ev.add_argument("--no-judge", action="store_true", help="skip the rubric and judge checks")
    ev.add_argument(
        "--latency-cost", action="store_true", help="one run per case, no judge (API backend)"
    )
    ev.add_argument("--gate", action="store_true", help="exit 2 when a target is missed")
    ev.add_argument("--out", type=Path, help="result JSON (default build/coach-eval/...)")
    ev.set_defaults(handler=_eval)
    args = parser.parse_args(argv)
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s %(message)s")
    return int(args.handler(args))
