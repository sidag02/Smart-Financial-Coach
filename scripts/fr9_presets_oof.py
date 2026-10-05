"""FR-9 milestone 1: each sensitivity preset measured out of fold, the way it ships (§6).

Decision 12 (owner, Oct 4, 2026, on #65): in each validation fold of the promoted config's round,
- fit the config on the other folds' users, as the round did (Balanced is its fitted cutoff);
- place Less and More on those same users with `sfc-model presets`' rule (`presets.place`:
  post-warm-up rows, `top_k`'s tie-break);
- apply all three to the held-out users, in the warm-up at the stricter of the level and
  Balanced (decision 17).

So it measures precision and recall per level, and whether 0.5x and 2x hold on users the cutoffs
weren't placed on. Precision and recall come from the label contract, which scores nothing in the
warm-up, so warm-up flags are counted apart. Intervals resample users (1,000 draws). Only train
users are read: test users are never scored for presets (decision 4).

    uv run python scripts/fr9_presets_oof.py data/synthetic/default.sqlite
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.store import load_transactions
from smart_financial_coach.evaluation.experiment import ExperimentConfig, load_experiment
from smart_financial_coach.evaluation.runner import prepare
from smart_financial_coach.evaluation.tasks.base import Task, get_task
from smart_financial_coach.intelligence.anomaly.threshold import Thresholded
from smart_financial_coach.intelligence.anomaly.threshold import user_months as charge_months
from smart_financial_coach.intelligence.models.artifact import promoted_version
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.presets import (
    LEVELS,
    Presets,
    charges_after_warmup,
    periods_after_warmup,
    place,
    warmup_end,
)
from smart_financial_coach.intelligence.spikes.threshold import SpikeThresholded
from smart_financial_coach.intelligence.spikes.threshold import user_months as period_months

CONFIGS = {
    "unusual_transactions": "31_isolation_forest_one_sided.yaml",
    "spending_spikes": "20_count_negbin.yaml",
}
REPS = 1000
SEED = 0

Mask = npt.NDArray[np.bool_]
Scores = npt.NDArray[np.float64]


def promoted_config(task: str) -> ExperimentConfig:
    """The promoted model's config, checked against its version."""
    config = load_experiment(PROJECT_ROOT / "configs" / "experiments" / task / CONFIGS[task])
    version = promoted_version(PROJECT_ROOT / "artifacts" / task)
    if not version.startswith(config.config_hash()[:8]):
        raise SystemExit(f"{CONFIGS[task]} isn't the promoted {task} model ({version})")
    return config


def _scores(task: str, model: Any, x: pd.DataFrame) -> Scores:
    """The model's scores, -inf where it can't flag (no reason; the product rules)."""
    if isinstance(model, Thresholded):
        scored = model._scores(x)
        score = scored["score"].to_numpy(dtype=np.float64)
        return np.where(scored["reason_code"].notna().to_numpy(), score, -np.inf)
    assert isinstance(model, SpikeThresholded), task
    return np.asarray(model._scores(x), dtype=np.float64)


def _post(task: str, x: pd.DataFrame, txns: pd.DataFrame, ends: pd.Series) -> Mask:
    if task == "unusual_transactions":
        return charges_after_warmup(x, ends)
    return periods_after_warmup(x, txns)


def run(task_name: str, data: Path) -> None:
    task: Task = get_task(task_name)
    config = promoted_config(task_name)
    examples, splits = prepare(task, data, config)
    txns = load_transactions(data)
    ends = warmup_end(txns)
    held: list[pd.DataFrame] = []
    placed: list[tuple[int, Presets, dict[str, int]]] = []
    for i, fold in enumerate(splits.folds):
        x, y = task.training_rows(examples, fold.train, config.task_params, config.seed + i)
        model = build(config.model_spec({})).fit(x, y)
        score, post = _scores(task_name, model, x), _post(task_name, x, txns, ends)
        presets = place(score, post, x[examples.id_column].to_numpy(), float(model.cutoff))
        placed.append(
            (i, presets, {lv: int((post & (score >= presets.cutoff(lv))).sum()) for lv in LEVELS})
        )
        for _, which in sorted(fold.held_out.items()):
            rows = examples.rows(which)
            labels = examples.labels_for(which)
            assert labels is not None
            s, p = _scores(task_name, model, rows), _post(task_name, rows, txns, ends)
            out = rows[[examples.id_column, "user_id"]].assign(
                fold=i, post=p, label=labels.to_numpy()
            )
            if "period_start" in rows:
                out["period_start"] = rows["period_start"].to_numpy()
                out["ts"] = None
            else:
                out["ts"] = rows["ts"].to_numpy()
            for level in LEVELS:
                cut = np.where(p, presets.cutoff(level), presets.cutoff(level, in_warmup=True))
                out[level] = s >= cut
            held.append(out)
    rows = pd.concat(held, ignore_index=True)
    report(task_name, task, examples, rows, placed)


def _outcomes(
    task_name: str, task: Any, examples: Any, rows: pd.DataFrame, level: str
) -> pd.DataFrame:
    """The label contract's outcome per flag and per missed label, with each row's user."""
    truth = task.truth
    users = set(rows["user_id"])
    flagged = rows[rows[level]]
    if task_name == "unusual_transactions":
        out = truth.score_transactions(flagged[["transaction_id"]])
        owner = examples.frame.set_index("transaction_id")["user_id"]
        out["user_id"] = out["transaction_id"].map(owner)
    else:
        real = flagged[flagged["label"] != "basket"]
        keys = examples.frame.set_index("period_id").loc[real["period_id"], ["category"]]
        flags = real[["user_id", "period_start"]].assign(category=keys["category"].to_numpy())
        out = truth.score_periods(flags.astype({"period_start": str}))
    return out[out["user_id"].isin(users)]


def _ratio(
    per_user: pd.DataFrame, num: str, den: list[str]
) -> Callable[[npt.NDArray[np.int64]], float]:
    def f(idx: npt.NDArray[np.int64]) -> float:
        part = per_user.iloc[idx]
        total = part[den].to_numpy().sum()
        return float(part[num].sum() / total) if total else float("nan")

    return f


def _interval(per_user: pd.DataFrame, num: str, den: list[str]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    f = _ratio(per_user, num, den)
    draws = [f(rng.integers(0, len(per_user), len(per_user))) for _ in range(REPS)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def report(
    task_name: str,
    task: Any,
    examples: Any,
    rows: pd.DataFrame,
    placed: list[tuple[int, Presets, dict[str, int]]],
) -> None:
    counted = rows[rows["label"] != "basket"]
    post = counted[counted["post"].astype(bool)]
    if task_name == "unusual_transactions":
        months = charge_months(post.assign(ts=pd.to_datetime(post["ts"])))
    else:
        months = float(period_months(post))
    everyone = pd.Index(sorted(set(counted["user_id"])))
    print(
        f"\n## {task_name}: out of fold, {len(everyone)} train users in {len(placed)} folds "
        f"({months:,.0f} post-warm-up user-months)\n"
    )
    print(
        "| Preset | Flags after warm-up | Per post-warm-up user-month | vs Balanced "
        "| Precision (95%) | Recall (95%) | Flags in warm-up |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    after = {lv: int(post[lv].sum()) for lv in LEVELS}
    for level in LEVELS:
        warm = int((counted[level] & ~counted["post"].astype(bool)).sum())
        outcomes = _outcomes(task_name, task, examples, counted, level)
        per_user = (
            outcomes.pivot_table(index="user_id", columns="outcome", aggfunc="size", fill_value=0)
            .reindex(columns=["tp", "fp", "fn"], fill_value=0)
            .reindex(everyone, fill_value=0)
        )
        tp, fp, fn = (int(per_user[c].sum()) for c in ("tp", "fp", "fn"))
        p_lo, p_hi = _interval(per_user, "tp", ["tp", "fp"])
        r_lo, r_hi = _interval(per_user, "tp", ["tp", "fn"])
        print(
            f"| {level} | {after[level]:,} | {after[level] / months:.4f} "
            f"| {after[level] / after['balanced']:.2f}x | {tp / (tp + fp):.3f} "
            f"({p_lo:.3f}-{p_hi:.3f}) | {tp / (tp + fn):.3f} ({r_lo:.3f}-{r_hi:.3f}) | {warm:,} |"
        )
    print("\nEach fold's cutoffs, and their flags on the users they were placed on:")
    for i, presets, counts in placed:
        print(
            f"- fold {i}: "
            + ", ".join(f"{lv} {presets.cutoff(lv):.4g}" for lv in LEVELS)
            + f"; flags {counts['less']:,} / {counts['balanced']:,} / {counts['more']:,}"
        )


def main() -> None:
    data = Path(sys.argv[1])
    for task_name in CONFIGS:
        run(task_name, data)


if __name__ == "__main__":
    main()
