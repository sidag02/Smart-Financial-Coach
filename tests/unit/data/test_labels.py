"""The FR-2 label contract on hand-built frames: every rule, one at a time."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.labels import Contract, Truth, metrics, precision_at_recall

CONTRACT = Contract(
    weak_lift=1.3,
    max_weak_share=0.15,
    max_drivers=2,
    duplicate_window_minutes=90,
    oracle_recall=0.5,
    oracle_min_precision=0.7,
    min_labels_for_oracle_check=1,
    baseline_days=90,
    baseline_months=3,
)
# Calendar starts 2025-01-01: warm-up ends 2025-04-01 (90 days) and in April (3 months)


def _tx(
    tid: str,
    ts: str,
    amount: float,
    *,
    user: str = "u1",
    category: str = "Dining",
    process: str = "discretionary",
    kind: str | None = None,
    related: str | None = None,
    tier: str = "clear",
) -> dict[str, object]:
    return {
        "transaction_id": tid,
        "user_id": user,
        "ts": ts,
        "amount": amount,
        "merchant_raw": "SQ *CAFE",
        "channel": "card_present",
        "category": category,
        "merchant_id": "m_cafe",
        "process": "unusual_charge" if kind else process,
        "is_recurring": 0,
        "anomaly_kind": kind,
        "tier": tier if kind else None,
        "related_transaction_id": related,
        "month": ts[:7] + "-01",
    }


def _spike(
    sid: str, granularity: str, start: str, *, category: str = "Dining", tier: str = "clear"
) -> dict[str, object]:
    return {
        "spike_id": sid,
        "user_id": "u1",
        "granularity": granularity,
        "period_start": start,
        "period_end": start,
        "category": category,
        "multiplier": 2.0,
        "expected_count": 10.0,
        "expected_spend": 100.0,
        "base_spend": 100.0,
        "extra_spend": 100.0,
        "tier": tier,
    }


def _truth(tx: list[dict[str, object]], spikes: list[dict[str, object]]) -> Truth:
    months = [f"2025-{m:02d}-01" for m in range(1, 13)]
    expected = pd.DataFrame(
        {
            "user_id": "u1",
            "category": "Dining",
            "granularity": "month",
            "period_start": months,
            "expected_count": 10.0,
            "expected_spend": 100.0,
        }
    )
    return Truth(
        transactions=pd.DataFrame(tx),
        periods=pd.DataFrame(spikes, columns=list(_spike("x", "month", "2025-01-01"))),
        expected=expected,
        merchants=pd.DataFrame(
            {"merchant_id": ["m_cafe"], "price_median": [5.0], "price_sigma": [0.3]}
        ).assign(is_ambiguous=0),
        users=pd.DataFrame({"user_id": ["u1"], "split": ["test"]}),
        preferences=pd.DataFrame(columns=["user_id", "merchant_id", "category"]),
        contract=CONTRACT,
        calendar_start=date(2025, 1, 1),
    )


def _outcomes(scored: pd.DataFrame, key: str) -> dict[str, str]:
    return dict(zip(scored[key], scored["outcome"], strict=True))


def test_transaction_rules() -> None:
    truth = _truth(
        [
            _tx("early", "2025-02-01 09:00", -5.0),
            _tx("orig", "2025-06-01 09:00", -5.0),
            _tx("dupe", "2025-06-01 09:30", -5.0, kind="duplicate", related="orig"),
            _tx("big", "2025-07-01 09:00", -90.0, kind="amount_outlier", tier="weak"),
            _tx("new", "2025-08-01 09:00", -400.0, kind="new_merchant_large"),
            _tx("plain", "2025-08-02 09:00", -6.0),
        ],
        [],
    )
    flags = pd.DataFrame(
        {
            "transaction_id": ["early", "orig", "dupe", "big", "plain"],
            "reason_code": ["duplicate", "duplicate", "duplicate", "new_merchant", "duplicate"],
        }
    )

    scored = truth.score_transactions(flags)

    assert _outcomes(scored, "transaction_id") == {
        "early": "ignored",  # warm-up
        "orig": "ignored",  # original of a labeled duplicate
        "dupe": "tp",
        "big": "tp",
        "plain": "fp",
        "new": "fn",
    }
    m = metrics(scored)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["reason_accuracy"] == pytest.approx(0.5)  # "big" carries the wrong reason
    assert m["recall_clear"] == pytest.approx(1 / 2)  # "dupe" found, "new" missed; "big" is weak


def test_unknown_flagged_transaction_is_rejected() -> None:
    truth = _truth([_tx("a", "2025-06-01 09:00", -5.0)], [])

    with pytest.raises(ValueError, match="not in the dataset"):
        truth.score_transactions(pd.DataFrame({"transaction_id": ["zzz"]}))


def test_period_rules() -> None:
    truth = _truth(
        [_tx("u", "2025-09-10 09:00", -300.0, kind="amount_outlier")],
        [
            _spike("hit", "month", "2025-05-01"),
            _spike("missed", "month", "2025-06-01", tier="weak"),
            _spike("wk", "week", "2025-07-07"),
        ],
    )
    flags = pd.DataFrame(
        {
            "user_id": "u1",
            "category": "Dining",
            "period_start": ["2025-02-01", "2025-05-01", "2025-07-01", "2025-09-01", "2025-10-01"],
        }
    )

    scored = truth.score_periods(flags)

    assert _outcomes(scored, "period_start") == {
        "2025-02-01": "ignored",  # warm-up
        "2025-05-01": "tp",
        "2025-07-01": "ignored",  # contains a planted weekly spike
        "2025-09-01": "ignored",  # contains a planted unusual charge
        "2025-10-01": "fp",
        "2025-06-01": "fn",
    }
    m = metrics(scored)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["recall_clear"] == pytest.approx(1.0)  # the missed spike was weak


def test_periods_must_start_on_the_first() -> None:
    truth = _truth([_tx("a", "2025-05-03 09:00", -5.0)], [_spike("s", "month", "2025-05-01")])
    bad = pd.DataFrame({"user_id": ["u1"], "category": ["Dining"], "period_start": ["2025-05-03"]})

    with pytest.raises(ValueError, match="don't start on the 1st"):
        truth.score_periods(bad)
    with pytest.raises(ValueError, match="don't start on the 1st"):
        truth.score_drivers(bad.assign(transaction_id="a"))


def test_weekly_periods_are_not_scored() -> None:
    truth = _truth([], [])

    with pytest.raises(ValueError, match="monthly"):
        truth.score_periods(pd.DataFrame(columns=["user_id", "category", "period_start"]), "week")


def test_driver_cap_period_and_coverage() -> None:
    truth = _truth(
        [
            _tx("a", "2025-05-03 09:00", -150.0),
            _tx("b", "2025-05-04 09:00", -50.0),
            _tx("c", "2025-05-05 09:00", -40.0),
            _tx("other", "2025-06-05 09:00", -40.0),
        ],
        [_spike("s", "month", "2025-05-01")],
    )
    key = {"user_id": "u1", "category": "Dining", "period_start": "2025-05-01"}
    drivers = pd.DataFrame([{**key, "transaction_id": t} for t in ("b", "c")])

    scored = truth.score_drivers(drivers).iloc[0]

    # Realized 240 vs expected 100: excess 140; b + c cover 90
    assert scored["valid"]
    assert scored["coverage"] == pytest.approx(90 / 140)

    # An invalid set covers nothing: returning everything can't score well
    too_many = truth.score_drivers(
        pd.DataFrame([{**key, "transaction_id": t} for t in ("a", "b", "c")])
    )
    outside = truth.score_drivers(
        pd.DataFrame([{**key, "transaction_id": t} for t in ("a", "other")])
    )
    assert not too_many.iloc[0]["valid"]
    assert too_many.iloc[0]["coverage"] == 0.0
    assert not outside.iloc[0]["valid"]
    assert outside.iloc[0]["coverage"] == 0.0

    # Repeating a driver counts it once
    repeated = truth.score_drivers(pd.DataFrame([{**key, "transaction_id": "b"}] * 5)).iloc[0]
    assert repeated["valid"]
    assert repeated["n"] == 1
    assert repeated["coverage"] == pytest.approx(50 / 140)

    baseline = truth.baseline_drivers()
    assert list(baseline["transaction_id"]) == ["a", "b"]  # the 2 largest


def test_precision_at_recall() -> None:
    score = pd.Series([0.9, 0.8, 0.7, 0.6, 0.1])
    label = pd.Series([True, False, True, False, True])

    assert precision_at_recall(score, label, 0.5) == pytest.approx(2 / 3)
    assert precision_at_recall(score, label, 1.0) == pytest.approx(3 / 5)
    assert np.isnan(precision_at_recall(score, pd.Series([False] * 5), 0.5))
