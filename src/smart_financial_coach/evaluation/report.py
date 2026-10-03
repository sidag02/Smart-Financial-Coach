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
    SelectionError,
    baseline_runs,
    code_versions,
    comparable_runs,
    leaderboard,
    rebuild_splits,
)
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import RunRecord, Tracker

DASH = "\u2013"  # en dash, for ranges and missing values in Markdown tables


def _cell(value: float | None, digits: int = 3) -> str:
    return DASH if value is None or value != value else f"{value:.{digits}f}"


def _metric_row(run: RunRecord, keys: tuple[str, ...]) -> list[str]:
    return [_cell(run.metrics.get(k), 2 if k.startswith("latency") else 3) for k in keys]


def _test_cell(run: RunRecord, key: str) -> str:
    """The metric, with its interval when the task logged one (`<key>_lo`, `<key>_hi`)."""
    value, lo, hi = (run.metrics.get(k) for k in (key, f"{key}_lo", f"{key}_hi"))
    if lo is None or hi is None:
        return _cell(value)
    return f"{_cell(value)} ({lo:.2f}{DASH}{hi:.2f})"


def comparison_report(
    task_name: str, data: Path, tracker: Tracker, split_hash: str | None = None
) -> str:
    task = get_task(task_name)
    runs = {r.run_id: r for r in comparable_runs(tracker, task, split_hash)}
    if not runs:
        raise SelectionError("no finished runs on these splits; run some experiments first")
    standings = leaderboard(task_name, data, tracker, split_hash)
    baselines = baseline_runs(list(runs.values()))
    selection = f"val_{task.selection_metric}"
    keys = task.report_metrics
    shipping = bool(task.shipping_params)  # comparison runs ranked; twins break ties and ship
    no_twins = shipping and not any(s.twin_id for s in standings)
    twin_breaker = task.tiebreak_metrics[0] if task.tiebreak_metrics else selection

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        first = runs[standings[0].run_id] if standings else next(iter(runs.values()))
        examples, _ = rebuild_splits(task, data, tracker, first, tmp)

        def pooled(run_id: str) -> pd.DataFrame:
            path = tracker.download(run_id, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", tmp / run_id)
            return pd.read_parquet(path)

        # The leader is the first *eligible* run; with none, there is nothing to compare against
        eligible = [s for s in standings if s.eligible]
        leader = pooled(eligible[0].run_id) if eligible else None
        rows = []
        for place, s in enumerate(standings, start=1):
            run = runs[s.run_id]
            mine = pooled(s.run_id)
            lo, hi = task.selection_interval(examples, mine)
            if leader is None or not s.eligible or s.run_id == eligible[0].run_id:
                diff = DASH
            else:
                d_lo, d_hi = task.difference_interval(examples, mine, leader)
                diff = f"{d_lo:+.3f} to {d_hi:+.3f}"
            rank = str(place) if s.eligible else DASH
            tie = " (tied)" if s.tied_with_leader and s.eligible else ""
            twin_cells = []
            if shipping:
                twin = runs.get(s.twin_id) if s.twin_id else None
                stop = " **reversal**" if s.reverses else ""
                twin_cells = [
                    f"{_cell(s.twin_estimate)}{stop}",
                    _cell(twin.metrics.get(twin_breaker)) if twin else DASH,
                ]
            rows.append(
                [
                    rank,
                    f"`{s.name}`{tie}",
                    f"{_cell(s.estimate)} ({lo:.2f}{DASH}{hi:.2f})",
                    diff,
                    *twin_cells,
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
        *(
            [
                "- Shipping twins (`label_noise` and other shipping params set explicitly) are "
                "what gets finalized. Ties among comparison runs are broken on the twins: "
                f"`{twin_breaker}` with its own tie test, then the remaining tie-breakers. "
                "**reversal** marks a twin that beats rank 1's twin on validation, which stops "
                "`finalize`.",
            ]
            if shipping
            else []
        ),
        *(
            [
                "- **No candidate has a shipping twin,** so tie-breaks can't be read (they come "
                "from twins) and the tie set is ordered by point estimate. This is not the "
                "decision rule's order; runs from before twins (e.g. FR-3's launch round) were "
                "ranked under the rule of their time.",
            ]
            if no_twins
            else []
        ),
        "",
        "| Rank | Run | "
        + f"`{selection}` (95% CI) | vs leader | "
        + (f"twin `{selection}` | twin `{twin_breaker}` | " if shipping else "")
        + " | ".join(f"`{k}`" for k in keys)
        + " |",
        "| --- | --- | --- | --- | "
        + ("--- | --- | " if shipping else "")
        + " | ".join("---" for _ in keys)
        + " |",
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
        # Baselines are scored with the finalists, so "beats the baseline" is visible here
        tests = task.test_report_metrics
        scored = [
            (r.tags.get(RANK_TAG, DASH), r)
            for r in sorted(finalists, key=lambda r: r.tags.get(RANK_TAG, ""))
        ] + [("baseline", b) for b in baselines if any(k in b.metrics for k in tests)]
        lines += [
            "",
            "**Finalists and baselines on the test sets** (scored once, by `finalize`; "
            "intervals are 95% merchant-bootstrap):",
            "",
            "| Rank | Run | " + " | ".join(f"`{k}`" for k in tests) + " |",
            "| --- | --- | " + " | ".join("---" for _ in tests) + " |",
            *(
                f"| {rank} | `{r.name}` | " + " | ".join(_test_cell(r, k) for k in tests) + " |"
                for rank, r in scored
            ),
        ]
    return "\n".join(lines) + "\n"
