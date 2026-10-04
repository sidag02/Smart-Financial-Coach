"""FR-11/FR-12 §6: the stage-9 sampler reproduces the dataset, and the goal-forecasting task's
examples, splits, leak check, metrics and gates."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.generator import load_spec
from smart_financial_coach.data.generator.goals import sample_goals
from smart_financial_coach.data.generator.population import user_seeds
from smart_financial_coach.data.generator.sqlite_io import read_sqlite
from smart_financial_coach.data.generator.timeline import Timeline
from smart_financial_coach.evaluation.splits import TRAIN, ids
from smart_financial_coach.evaluation.tasks import Examples, get_task
from smart_financial_coach.evaluation.tasks.goals import (
    TEST,
    GoalForecastingTask,
    history_leaks,
)
from smart_financial_coach.intelligence.forecasting.baseline import Flat, NaivePace
from smart_financial_coach.intelligence.forecasting.contract import CONTRACT, parse_history
from smart_financial_coach.intelligence.models.contract import Checked


@pytest.fixture(scope="module")
def task() -> GoalForecastingTask:
    return GoalForecastingTask(reps=200, draws=2)


@pytest.fixture(scope="module")
def examples(task: GoalForecastingTask, small_sqlite: Path) -> Examples:
    return task.load(small_sqlite)


def test_the_sampler_reproduces_stage_9_from_the_dataset(small_sqlite: Path) -> None:
    """Stage 9 refactored into `sample_goals` (§6): with each user's own goal stream and the net
    savings rebuilt from model-visible transactions, it gives back the dataset's goals."""
    ds = read_sqlite(small_sqlite)
    spec = load_spec(PROJECT_ROOT / "configs" / "data" / "small.yaml")
    tl = Timeline(spec.calendar.start, spec.calendar.end)
    start = pd.Period(spec.calendar.start, freq="M")
    seeds = {p.split: p.seed for p in spec.populations}
    txns = ds["transactions"]
    month = (pd.to_datetime(txns["ts"]).dt.to_period("M") - start).map(lambda d: d.n)
    for u in ds["users"].to_dict("records"):
        mine = txns["user_id"] == u["user_id"]
        net = np.bincount(month[mine], weights=txns.loc[mine, "amount"], minlength=tl.n_months)
        persona = spec.personas[u["persona"]]
        goals, _ = sample_goals(
            np.asarray(net, dtype=float),
            tl,
            [g.name for g in persona.goals],
            [g.weight for g in persona.goals],
            spec.goals,
            np.random.default_rng(user_seeds(seeds[u["split"]], u["user_id"])["goals"]),
            user_id=u["user_id"],
            goal_prefix=f"g_{u['user_id'][2:]}",
        )
        stored = ds["goals"][ds["goals"]["user_id"] == u["user_id"]].reset_index(drop=True)
        same = ["goal_id", "name", "created_date", "target_date", "as_of_date"]
        assert goals[same].equals(stored[same]), u["user_id"]
        for col in ("target_amount", "current_balance"):
            assert np.allclose(goals[col], stored[col], atol=0.05), (u["user_id"], col)


def test_examples_have_both_paths_and_only_known_outcomes(
    examples: Examples, small_sqlite: Path
) -> None:
    f = examples.frame
    assert set(f["path"]) == {"track", "new"}
    assert f.groupby("goal_id")["path"].nunique().eq(2).all()
    assert set(f["met"]) <= {0, 1}
    new = f[f["path"] == "new"]
    assert (new["created_date"] == new["as_of_date"]).all()
    assert (new["origin"] == "yours").all()
    assert set(f["draw"]) == {0, 1, 2}
    known = read_sqlite(small_sqlite)["truth_goals"]["met"].notna().sum()
    assert (f["draw"] == 0).sum() == 2 * known  # the dataset's own goals, on both paths
    assert (f["active_goals"] >= 1).all()


def test_histories_end_at_as_of_and_the_leak_check_proves_it(
    task: GoalForecastingTask, examples: Examples
) -> None:
    splits = task.split(examples, {}, seed=0)
    assert task.leak_errors(examples, splits) == []
    for r in examples.frame.head(50).to_dict("records"):
        history = parse_history(r["history_json"])
        assert history.index[-1] == pd.Period(r["as_of_date"], freq="M")
        future = parse_history(r["future_json"])
        assert future.index[0] == pd.Period(r["as_of_date"], freq="M") + 1
    leaky = examples.frame.head(3).copy()
    leaky["history_json"] = leaky["future_json"]
    assert history_leaks(leaky)


def test_splits_keep_users_whole_and_separate(
    task: GoalForecastingTask, examples: Examples
) -> None:
    splits = task.split(examples, {}, seed=0)
    f = examples.frame.set_index("example_id")
    assert set(f.loc[splits.sets[TRAIN].tolist(), "split"]) == {"train"}
    assert set(f.loc[splits.sets[TEST].tolist(), "split"]) == {"test"}
    for fold in splits.folds:
        held = set(f.loc[next(iter(fold.held_out.values())).tolist(), "user_id"])
        assert held.isdisjoint(f.loc[fold.train.tolist(), "user_id"])


def test_metrics_score_both_paths_and_the_baselines_order(
    task: GoalForecastingTask, examples: Examples
) -> None:
    rows = examples.rows(ids(examples.frame["example_id"]))
    flat = Checked(Flat(0.5), CONTRACT).predict(rows)
    naive = Checked(NaivePace(), CONTRACT).predict(rows)
    m_flat = task.validation_metrics(examples, flat)
    m_naive = task.validation_metrics(examples, naive)
    assert m_flat["brier.track"] == pytest.approx(0.25)
    assert m_flat["brier.new"] == pytest.approx(0.25)
    assert m_flat["neg_brier"] == pytest.approx(-0.25)
    assert m_naive["brier.track"] > 0.25  # the naive call is worse than a coin on this data
    assert np.isnan(m_flat["coverage.track"])  # no share, no coverage
    lo, hi = task.selection_interval(examples, flat)
    assert lo == pytest.approx(-0.25)
    assert hi == pytest.approx(-0.25)
    assert not task.tied(examples, flat, naive)


def test_gates_follow_the_design_and_decision_6(task: GoalForecastingTask) -> None:
    base = {"test_brier.track": 0.3, "test_brier.new": 0.3}
    metrics = {"test_brier.track": 0.17, "test_brier.new": 0.21}
    for path in ("track", "new"):
        metrics |= {f"test_coverage.{path}": 0.82}
        for persona in ("family_budgeter", "freelancer", "young_professional"):
            metrics |= {f"test_brier.{path}.{persona}": 0.2}
        for band, rate in (("off", 0.2), ("either", 0.5), ("on", 0.8)):
            metrics |= {
                f"test_met_rate.{path}.{band}": rate,
                f"test_met_rate.{path}.{band}_lo": rate - 0.05,
                f"test_met_rate.{path}.{band}_hi": rate + 0.05,
            }
    metrics["test_brier.new.freelancer"] = 0.26  # over a flat 50%
    gates = {g.name: g for g in task.gates(metrics, {"naive_pace": base, "flat_50": base})}
    freelancer = gates["brier_new_freelancer_below_flat"]
    assert (freelancer.passed, freelancer.blocking) == (False, False)  # reported, not blocking
    assert all(g.passed for g in gates.values() if g.blocking)
    assert gates["brier_track_freelancer_below_flat"].blocking


def test_the_task_is_registered() -> None:
    assert isinstance(get_task("goal_forecasting"), GoalForecastingTask)
