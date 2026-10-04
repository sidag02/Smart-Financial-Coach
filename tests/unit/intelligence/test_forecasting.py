"""FR-11/FR-12 §1-§3: monthly net savings, the balance recursion, share inference, the contract
and the baselines."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.generator.goals import _saved
from smart_financial_coach.intelligence.forecasting.baseline import Flat, NaivePace
from smart_financial_coach.intelligence.forecasting.contract import (
    CONTRACT,
    INPUT_COLUMNS,
    history_json,
    months_left,
    parse_history,
    status_for,
)
from smart_financial_coach.intelligence.forecasting.savings import (
    final_balance,
    infer_share,
    monthly_net,
    run_balance,
)
from smart_financial_coach.intelligence.models.contract import Checked, ContractError


def test_monthly_net_sums_each_month_and_fills_gaps() -> None:
    txns = pd.DataFrame(
        {
            "ts": ["2026-01-03 10:00", "2026-01-20 09:00", "2026-03-02 08:00"],
            "amount": [3000.0, -1200.5, -40.0],
        }
    )
    net = monthly_net(txns)
    assert list(net.index.astype(str)) == ["2026-01", "2026-02", "2026-03"]
    assert net.tolist() == [1799.5, 0.0, -40.0]
    assert monthly_net(txns.iloc[:0]).empty


def test_the_balance_recursion_is_fr1s_rule() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        net = rng.normal(200, 900, size=int(rng.integers(1, 30)))
        share = float(rng.uniform(0, 1))
        expected = _saved(net, share, 0, len(net) - 1)
        assert final_balance(0.0, share, net) == pytest.approx(expected)


def test_run_balance_runs_many_paths_at_once() -> None:
    paths = np.array([[100.0, -500.0, 50.0], [100.0, 100.0, 100.0]])
    out = run_balance(10.0, 0.5, paths)
    assert out.tolist() == [[60.0, 0.0, 25.0], [60.0, 110.0, 160.0]]


@pytest.mark.parametrize("start", [0.0, 750.0])
def test_share_inference_round_trips(start: float) -> None:
    """From $0 (a generated goal's track record) and from a first saved entry (`your_entries`)."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        net = rng.normal(400, 600, size=int(rng.integers(3, 24)))
        share = float(rng.uniform(0.05, 0.95))
        end = final_balance(start, share, net)
        if end <= start + 1:
            continue  # no growth: nothing to infer from
        inferred = infer_share(net, start, end)
        assert final_balance(start, inferred, net) == pytest.approx(end, rel=1e-6)


def test_share_inference_edges() -> None:
    net = np.array([100.0, 100.0])
    assert infer_share(net, 0.0, 0.0) == 0.0  # never grew
    assert infer_share(np.array([]), 0.0, 50.0) == 0.0  # no months
    assert infer_share(net, 0.0, 5000.0) == 1.0  # more than all of it: capped
    assert infer_share(net, 300.0, 250.0) == 0.0  # fell from the first entry


def test_months_left_and_status_bands() -> None:
    assert months_left(date(2026, 9, 30), date(2027, 6, 30)) == 9
    assert months_left(date(2026, 9, 15), date(2027, 6, 30)) == 10
    assert months_left(date(2026, 9, 30), date(2026, 9, 30)) == 0
    assert [status_for(p, False) for p in (0.0, 0.299, 0.3, 0.699, 0.7, 1.0)] == [
        "off_track",
        "off_track",
        "either_way",
        "either_way",
        "on_track",
        "on_track",
    ]
    assert status_for(0.1, True) == "reached"


def test_history_json_round_trips() -> None:
    net = pd.Series([1.5, -2.25], index=pd.period_range("2025-11", periods=2, freq="M"))
    back = parse_history(history_json(net))
    assert back.equals(net)
    assert parse_history(history_json(net.iloc[:0])).empty


def goal_rows(**overrides: object) -> pd.DataFrame:
    row = {
        "example_id": "g1:track",
        "goal_id": "g1",
        "user_id": "u1",
        "goal_set": "u1:0",
        "persona": "young_professional",
        "as_of_date": "2026-03-31",
        "created_date": "2025-10-15",
        "target_amount": 3000.0,
        "target_date": "2026-09-30",
        "saved": 1200.0,
        "saved_as_of": "2026-03-31",
        "first_saved": float("nan"),
        "first_saved_as_of": None,
        "origin": "existing",
        "active_goals": 1,
        "history_json": history_json(
            pd.Series([500.0] * 6, index=pd.period_range("2025-10", periods=6, freq="M"))
        ),
    } | overrides
    return pd.DataFrame([row], columns=list(INPUT_COLUMNS))


def test_naive_pace_extends_the_pace_so_far() -> None:
    out = Checked(NaivePace(), CONTRACT).predict(goal_rows())
    # $1,200 over 6 months is $200 a month; 6 more months give $2,400, short of $3,000
    row = out.iloc[0]
    assert row["projected_balance"] == pytest.approx(2400.0)
    assert (row["p_goal_met"], row["status"]) == (0.0, "off_track")
    assert row["gap"] == pytest.approx(600.0)
    assert row["extra_per_month"] == 100.0
    reached = Checked(NaivePace(), CONTRACT).predict(goal_rows(saved=3100.0)).iloc[0]
    assert reached["status"] == "reached"
    assert pd.isna(reached["extra_per_month"])


def test_flat_gives_every_goal_the_same_chance() -> None:
    out = Checked(Flat(0.5), CONTRACT).predict(goal_rows())
    assert (out.iloc[0]["p_goal_met"], out.iloc[0]["status"]) == (0.5, "either_way")
    with pytest.raises(ValueError, match="p must be 0-1"):
        Flat(1.5)


def test_the_contract_catches_inconsistent_forecasts() -> None:
    good = Checked(NaivePace(), CONTRACT).predict(goal_rows())
    x = goal_rows()
    for column, value, message in (
        ("status", "on_track", "band"),
        ("range_lo", 9999.0, "range_lo"),
        ("share", 0.4, "together"),
        ("p_goal_met", 1.2, "outside 0-1"),
    ):
        bad = good.copy()
        bad[column] = value
        assert any(message in e for e in CONTRACT.violations(x, bad)), column

    class Broken(NaivePace):
        def predict(self, x: pd.DataFrame) -> pd.DataFrame:
            out = super().predict(x)
            out["gap"] = -1.0
            return out

    with pytest.raises(ContractError, match="negative gap"):
        Checked(Broken(), CONTRACT).predict(x)
