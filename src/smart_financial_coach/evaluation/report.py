"""The comparison report: every comparable run, in decision-rule order, as Markdown.

    sfc-experiment report --task categorization --data data/synthetic/default.sqlite --out FILE

Validation metrics only, with the selection metric's merchant-bootstrap interval and each run's
paired difference from the leader. Test metrics appear once `finalize` has scored the finalists.
"""

import tempfile
from pathlib import Path

import pandas as pd

from smart_financial_coach.evaluation.promote import (
    FINALIST_TAG,
    RANK_TAG,
    _baselines,
    _comparable,
    _rebuild,
    code_versions,
    leaderboard,
)
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import RunRecord, Tracker

DASH = "\u2013"  # en dash, for ranges and missing values in Markdown tables


def _cell(value: float | None, digits: int = 3) -> str:
    return DASH if value is None or value != value else f"{value:.{digits}f}"


def _metric_row(run: RunRecord, keys: tuple[str, ...]) -> list[str]:
    return [_cell(run.metrics.get(k), 2 if k.startswith("latency") else 3) for k in keys]


def comparison_report(
    task_name: str, data: Path, tracker: Tracker, split_hash: str | None = None
) -> str:
    task = get_task(task_name)
    standings = leaderboard(task_name, data, tracker, split_hash)
    runs = {r.run_id: r for r in _comparable(tracker, task, split_hash)}
    baselines = _baselines(list(runs.values()))
    selection = f"val_{task.selection_metric}"
    keys = task.report_metrics

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        first = runs[standings[0].run_id] if standings else next(iter(runs.values()))
        examples, _ = _rebuild(task, data, tracker, first, tmp)

        def pooled(run_id: str) -> pd.DataFrame:
            path = tracker.download(run_id, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", tmp / run_id)
            return pd.read_parquet(path)

        leader = pooled(standings[0].run_id) if standings else None
        rows = []
        for place, s in enumerate(standings, start=1):
            run = runs[s.run_id]
            mine = pooled(s.run_id)
            lo, hi = task.selection_interval(examples, mine)
            if leader is None or place == 1:
                diff = "—"
            else:
                d_lo, d_hi = task.difference_interval(examples, mine, leader)
                diff = f"{d_lo:+.3f} to {d_hi:+.3f}"
            rank = str(place) if s.eligible else DASH
            tie = " (tied)" if s.tied_with_leader and s.eligible else ""
            rows.append(
                [
                    rank,
                    f"`{s.name}`{tie}",
                    f"{_cell(s.estimate)} ({lo:.2f}{DASH}{hi:.2f})",
                    diff,
                    *_metric_row(run, keys),
                ]
            )

    first_run = next(iter(runs.values()))
    lines = [
        f"- Data hash `{first_run.tags['sfc.data_hash'][:12]}`, split hash "
        f"`{first_run.tags['sfc.split_hash'][:12]}`, code versions: "
        + ", ".join(f"`{v[:12]}`" for v in sorted(code_versions(list(runs.values())))),
        f"- Ranked by `{selection}` (95% merchant-bootstrap interval); the difference column is "
        "the paired interval against the leader. *(tied)* marks runs in the leader's tie set, "
        "which are ordered by "
        + ", then ".join(f"`{k}`" for k in (*task.tiebreak_metrics, "complexity"))
        + ".",
        "",
        "| Rank | Run | "
        + f"`{selection}` (95% CI) | vs leader | "
        + " | ".join(f"`{k}`" for k in keys)
        + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in keys) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
        "**Baselines** (the floor, not candidates):",
        "",
        "| Run | `" + selection + "` | " + " | ".join(f"`{k}`" for k in keys) + " |",
        "| --- | --- | " + " | ".join("---" for _ in keys) + " |",
        *(
            f"| `{b.name}` | {_cell(b.metrics.get(selection))} | "
            + " | ".join(_metric_row(b, keys))
            + " |"
            for b in baselines
        ),
    ]
    finalists = [r for r in runs.values() if r.tags.get(FINALIST_TAG) == "true"]
    if finalists:
        tests = sorted({k for r in finalists for k in r.metrics if k.startswith("test_")})
        lines += [
            "",
            "**Finalists on the test sets:**",
            "",
            "| Rank | Run | " + " | ".join(f"`{k}`" for k in tests) + " |",
            "| --- | --- | " + " | ".join("---" for _ in tests) + " |",
            *(
                f"| {r.tags.get(RANK_TAG)} | `{r.name}` | "
                + " | ".join(_cell(r.metrics.get(k)) for k in tests)
                + " |"
                for r in sorted(finalists, key=lambda r: r.tags.get(RANK_TAG, ""))
            ),
        ]
    return "\n".join(lines) + "\n"
