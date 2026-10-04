"""`sfc-experiment` and `sfc-model`: run experiments, pick finalists, promote a model.

sfc-experiment run configs/experiments/categorization/ --data data/synthetic/default.sqlite
sfc-experiment leaderboard --task categorization --data data/synthetic/default.sqlite
sfc-experiment report --task categorization --data data/synthetic/default.sqlite --out report.md
sfc-experiment finalize --task categorization --data data/synthetic/default.sqlite
sfc-model promote --task categorization --run ID --note "linear weights explain each category"

`promote` uploads the model file to a GitHub Release (needs `gh` with write access) and records its
URL, so other clones download it on first use. --no-publish keeps it on this machine only.

Departing from the decision rule (naming finalists, a second round on the same splits, promoting
a finalist other than #1) needs --override "<reason>", recorded on the run and in the log.
sfc-model show --task categorization
sfc-model attach-serving-files --task categorization --run ID --version VERSION \
    --data data/synthetic/default.sqlite --note "..."   # e.g. a review policy, after promotion
sfc-model predict --task categorization --data data/synthetic/default.sqlite \
    --out data/predictions/default.sqlite
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from smart_financial_coach.config import get_settings
from smart_financial_coach.evaluation.experiment import experiment_files, load_experiment
from smart_financial_coach.evaluation.promote import (
    SelectionError,
    attach_serving_files,
    finalize,
    leaderboard,
    promote,
)
from smart_financial_coach.evaluation.publish import GitHubReleases, PublishError
from smart_financial_coach.evaluation.report import comparison_report
from smart_financial_coach.evaluation.runner import LeakError, run_experiment, run_with_twin
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.categorization.batch import (
    DEFAULT_BATCH_ROWS,
    categorize_dataset,
)
from smart_financial_coach.intelligence.models.artifact import (
    URL_KEY,
    ArtifactError,
    attachments,
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
        if args.reproduce_poc:
            results = [
                (
                    config.name,
                    run_experiment(
                        config, args.data, tracker, force=args.force, reproduce_poc=True
                    ),
                )
            ]
        else:
            result, twin = run_with_twin(config, args.data, tracker, force=args.force)
            results = [(config.name, result)]
            if twin is not None:
                results.append((config.twin().name, twin))
        for name, run in results:
            status = "skipped (already run)" if run.skipped else f"chose {run.chosen}"
            print(f"{name}: run {run.run_id} {status}")
            print(f"  {_metrics(run.metrics, 'val_')}")
            if args.reproduce_poc:
                print(f"  {_metrics(run.metrics, 'test_')}")
    return 0


def _leaderboard(args: argparse.Namespace) -> int:
    standings = leaderboard(args.task, args.data, Tracker(args.tracking_uri), args.split_hash)
    if not standings:
        print("no candidate runs")
        return 0
    if get_task(args.task).shipping_params and not any(s.twin_id for s in standings):
        print(
            "warning: no candidate has a shipping twin, so tie-breaks (read from twins) can't "
            "apply; the tie set below is ordered by point estimate, not the decision rule"
        )
    if len(versions := {s.code for s in standings}) > 1:
        print(f"warning: runs come from {len(versions)} code versions; finalize will refuse a mix")
    for i, s in enumerate(standings, start=1):
        place = f"{i:>2}." if s.eligible else " - "
        tie = " (tied with leader)" if s.tied_with_leader and s.eligible else ""
        twin = f"  twin {s.twin_estimate:.4f}" if s.twin_id else ""
        stop = "  REVERSAL: its twin beats rank 1's twin" if s.reverses else ""
        print(f"{place} {s.estimate:.4f}  {s.name}  {s.run_id}{tie}{twin}{stop}")
    return 0


def _report(args: argparse.Namespace) -> int:
    text = comparison_report(args.task, args.data, Tracker(args.tracking_uri), args.split_hash)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _finalize(args: argparse.Namespace) -> int:
    scored = finalize(
        args.task,
        args.data,
        Tracker(args.tracking_uri),
        run_ids=args.runs,
        override=args.override,
        split_hash=args.split_hash,
    )
    for run_id, metrics in scored.items():
        print(f"{run_id}: {_metrics(metrics, 'test_')}")
    return 0


def _promote(args: argparse.Namespace) -> int:
    entry = promote(
        args.task,
        args.run,
        args.note,
        Tracker(args.tracking_uri),
        args.artifacts_dir,
        override=args.override,
        publisher=None if args.no_publish else GitHubReleases(get_settings().model_release_repo),
    )
    print(f"promoted {entry['version']} (run {entry['mlflow_run_id']})")
    if url := entry.get(URL_KEY):
        print(f"  model file: {url}")
    else:
        print("  model file not published: other clones can't load this promotion")
    for gate in entry["gates"]:
        print(f"  {'pass' if gate['passed'] else 'FAIL'} {gate['name']}: {gate['detail']}")
    return 0


def _attach(args: argparse.Namespace) -> int:
    written = attach_serving_files(
        args.task,
        args.run,
        args.version,
        args.data,
        Tracker(args.tracking_uri),
        args.artifacts_dir,
        note=args.note,
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def _show(args: argparse.Namespace) -> int:
    service_dir = (args.artifacts_dir or get_settings().artifacts_dir) / args.task
    print(f"promoted: {promoted_version(service_dir)}")
    for entry in promotions(service_dir):
        print(f"  {entry['promoted_at']}  {entry['version']}  run {entry['mlflow_run_id']}")
    for entry in attachments(service_dir):
        print(
            f"  {entry['attached_at']}  {entry['version']}  + {entry['file']} "
            f"(run {entry['mlflow_run_id']})"
        )
    return 0


def _predict(args: argparse.Namespace) -> int:
    if args.task == "unusual_transactions":
        from smart_financial_coach.intelligence.anomaly.batch import flag_dataset

        flags = flag_dataset(
            args.data, args.out, artifacts_dir=args.artifacts_dir, overwrite=args.overwrite
        )
        print(
            f"flagged {flags.flagged:,} of {flags.scored:,} charges with model "
            f"{flags.model_version} into {args.out} in {flags.seconds:.1f} s"
        )
        return 0
    if args.task != "categorization":
        raise ValueError(
            f"predict supports categorization and unusual_transactions, not {args.task!r}"
        )
    run = categorize_dataset(
        args.data,
        args.out,
        batch_rows=args.batch_rows,
        artifacts_dir=args.artifacts_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {run.rows:,} categories from model {run.model_version} to {args.out}")
    print(
        f"  model load {run.load_seconds:.1f} s; categorizing {run.categorize_seconds:.1f} s, "
        f"{run.rows_per_second:,.0f} rows per second in one process"
    )
    return 0


def _dispatch(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> int:
    args = parser.parse_args(argv)
    try:
        result: int = args.handler(args)
    except (
        SelectionError,
        LeakError,
        ArtifactError,
        PublishError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
    ) as error:
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

    rep = commands.add_parser("report", help="the comparison report (Markdown) for comparable runs")
    rep.add_argument("--task", required=True)
    rep.add_argument("--data", type=Path, required=True)
    rep.add_argument("--split-hash", help="report runs on these splits (default: latest run's)")
    rep.add_argument("--out", type=Path, help="write to this file instead of printing")
    rep.set_defaults(handler=_report)

    fin = commands.add_parser("finalize", help="score the leaderboard's top three on the test sets")
    fin.add_argument("--task", required=True)
    fin.add_argument("--data", type=Path, required=True)
    fin.add_argument("--split-hash", help="finalize runs on these splits (default: latest run's)")
    fin.add_argument("--runs", nargs="+", help="name the finalists instead (needs --override)")
    fin.add_argument("--override", help="why this departs from the decision rule (recorded)")
    fin.set_defaults(handler=_finalize)

    for sub in (run, board, rep, fin):
        _common(sub)
    return _dispatch(parser, argv)


def model_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sfc-model", description="Promote, inspect and run models."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pro = commands.add_parser("promote", help="promote a finalist that passes every gate")
    pro.add_argument("--task", required=True)
    pro.add_argument("--run", required=True)
    pro.add_argument("--note", required=True, help="explainability and operations, in a sentence")
    pro.add_argument("--override", help="why a finalist other than #1 is promoted (recorded)")
    pro.add_argument(
        "--no-publish",
        action="store_true",
        help="don't upload the model file to a GitHub Release (other clones can't load it)",
    )
    pro.set_defaults(handler=_promote)

    att = commands.add_parser(
        "attach-serving-files",
        help="give a promoted model the serving files its task now requires, from a run that "
        "reproduces it (same config and data, same validation metrics)",
    )
    att.add_argument("--task", required=True)
    att.add_argument("--run", required=True, help="a run reproducing the promoted model")
    att.add_argument("--version", required=True, help="the promoted model version")
    att.add_argument("--data", type=Path, required=True, help="the run's dataset")
    att.add_argument(
        "--note", required=True, help="why, and what a reader of the files should know"
    )
    att.set_defaults(handler=_attach)

    show = commands.add_parser("show", help="the promoted version and promotion history")
    show.add_argument("--task", required=True)
    show.set_defaults(handler=_show)

    pred = commands.add_parser(
        "predict",
        help="categorize (or flag unusual charges in) a dataset with the promoted model",
    )
    pred.add_argument("--task", required=True)
    pred.add_argument("--data", type=Path, required=True, help="a generated dataset")
    pred.add_argument("--out", type=Path, required=True, help="the predictions file to write")
    pred.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    pred.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    pred.set_defaults(handler=_predict)

    _common(pro)
    _common(att)
    for sub in (pro, att, show, pred):
        sub.add_argument(
            "--artifacts-dir", type=Path, help="default: SFC_ARTIFACTS_DIR or artifacts/"
        )
    return _dispatch(parser, argv)


if __name__ == "__main__":
    sys.exit(experiment_main())
