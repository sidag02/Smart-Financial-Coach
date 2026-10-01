"""`sfc-experiment` and `sfc-model`: run experiments, pick finalists, promote a model.

sfc-experiment run configs/experiments/categorization/ --data data/synthetic/default.sqlite
sfc-experiment leaderboard --task categorization --data data/synthetic/default.sqlite
sfc-experiment finalize --task categorization --data data/synthetic/default.sqlite --runs ID ID
sfc-model promote --task categorization --run ID --note "linear weights explain each category"
sfc-model show --task categorization
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from smart_financial_coach.config import get_settings
from smart_financial_coach.evaluation.experiment import experiment_files, load_experiment
from smart_financial_coach.evaluation.promote import (
    SelectionError,
    finalize,
    leaderboard,
    promote,
)
from smart_financial_coach.evaluation.runner import LeakError, run_experiment
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.models.artifact import (
    ArtifactError,
    promoted_version,
    promotions,
)


def _metrics(metrics: dict[str, float], prefix: str) -> str:
    shown = sorted((k, v) for k, v in metrics.items() if k.startswith(prefix))
    return ", ".join(f"{k}={v:.3f}" for k, v in shown)


def _run(args: argparse.Namespace) -> int:
    tracker = Tracker(args.tracking_uri)
    for path in experiment_files(args.configs):
        config = load_experiment(path)
        result = run_experiment(
            config, args.data, tracker, force=args.force, reproduce_poc=args.reproduce_poc
        )
        status = "skipped (already run)" if result.skipped else f"chose {result.chosen}"
        print(f"{config.name}: run {result.run_id} {status}")
        print(f"  {_metrics(result.metrics, 'val_')}")
        if args.reproduce_poc:
            print(f"  {_metrics(result.metrics, 'test_')}")
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    standings = leaderboard(args.task, args.data, Tracker(args.tracking_uri), args.split_hash)
    if not standings:
        print("no candidate runs")
        return 0
    for i, s in enumerate(standings, start=1):
        place = f"{i:>2}." if s.eligible else " - "
        tie = " (tied with leader)" if s.tied_with_leader and s.eligible else ""
        print(f"{place} {s.estimate:.4f}  {s.name}  {s.run_id}{tie}")
    return 0


def _finalize(args: argparse.Namespace) -> int:
    scored = finalize(args.task, args.runs, args.data, Tracker(args.tracking_uri))
    for run_id, metrics in scored.items():
        print(f"{run_id}: {_metrics(metrics, 'test_')}")
    return 0


def _promote(args: argparse.Namespace) -> int:
    entry = promote(args.task, args.run, args.note, Tracker(args.tracking_uri), args.artifacts_dir)
    print(f"promoted {entry['version']} (run {entry['mlflow_run_id']})")
    for gate in entry["gates"]:
        print(f"  {'pass' if gate['passed'] else 'FAIL'} {gate['name']}: {gate['detail']}")
    return 0


def _show(args: argparse.Namespace) -> int:
    service_dir = (args.artifacts_dir or get_settings().artifacts_dir) / args.task
    print(f"promoted: {promoted_version(service_dir)}")
    for entry in promotions(service_dir):
        print(f"  {entry['promoted_at']}  {entry['version']}  run {entry['mlflow_run_id']}")
    return 0


def _dispatch(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    args = parser.parse_args(argv)
    try:
        result: int = args.handler(args)
    except (SelectionError, LeakError, ArtifactError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: SFC_MLFLOW_TRACKING_URI or local)"
    )


def experiment_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sfc-experiment", description="Run experiments and choose finalists."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run experiment configs (files or folders)")
    run.add_argument("configs", type=Path, nargs="+")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--force", action="store_true", help="rerun configs that already finished")
    run.add_argument(
        "--reproduce-poc",
        action="store_true",
        help="also score test sets; only for the task's published POC configurations",
    )
    run.set_defaults(handler=_run)

    board = commands.add_parser("leaderboard", help="order runs by the decision rule")
    board.add_argument("--task", required=True)
    board.add_argument("--data", type=Path, required=True)
    board.add_argument("--split-hash", help="compare runs on these splits (default: latest run's)")
    board.set_defaults(handler=_leaderboard)

    fin = commands.add_parser("finalize", help="score up to three finalists on the test sets")
    fin.add_argument("--task", required=True)
    fin.add_argument("--data", type=Path, required=True)
    fin.add_argument("--runs", nargs="+", required=True)
    fin.set_defaults(handler=_finalize)

    for sub in (run, board, fin):
        _common(sub)
    return _dispatch(parser, argv)


def model_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sfc-model", description="Promote and inspect models.")
    commands = parser.add_subparsers(dest="command", required=True)

    pro = commands.add_parser("promote", help="promote a finalist that passes every gate")
    pro.add_argument("--task", required=True)
    pro.add_argument("--run", required=True)
    pro.add_argument("--note", required=True, help="explainability and operations, in a sentence")
    pro.set_defaults(handler=_promote)

    show = commands.add_parser("show", help="the promoted version and promotion history")
    show.add_argument("--task", required=True)
    show.set_defaults(handler=_show)

    _common(pro)
    for sub in (pro, show):
        sub.add_argument(
            "--artifacts-dir", type=Path, help="default: SFC_ARTIFACTS_DIR or artifacts/"
        )
    return _dispatch(parser, argv)


if __name__ == "__main__":
    sys.exit(experiment_main())
