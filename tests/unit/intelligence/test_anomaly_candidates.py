import json

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.intelligence.anomaly.contract import CONTRACT
from smart_financial_coach.intelligence.anomaly.forest import Forest
from smart_financial_coach.intelligence.anomaly.probabilistic import Probabilistic
from smart_financial_coach.intelligence.anomaly.reasons import reason
from smart_financial_coach.intelligence.anomaly.rules import Rules
from smart_financial_coach.intelligence.anomaly.threshold import Thresholded
from smart_financial_coach.intelligence.models.contract import Checked


def ledger() -> pd.DataFrame:
    """One user: 40 coffees around $5, then a $60 coffee (amount), an exact repeat of a coffee 5
    minutes later (duplicate), and a $900 first visit to a store whose profile says $100 (new
    merchant). Other merchants have profiles of $5 (the cafe) and $100 (the store)."""
    rows = []
    for i in range(40):
        ts = pd.Timestamp("2025-01-01 08:00") + pd.Timedelta(days=i)
        rows.append((f"c{i:02d}", ts, -(5.0 + (i % 3) * 0.25), "CAFE"))
    rows += [
        ("big", pd.Timestamp("2025-02-15 08:00"), -60.0, "CAFE"),
        ("orig", pd.Timestamp("2025-02-16 08:00"), -5.25, "CAFE"),
        ("dup", pd.Timestamp("2025-02-16 08:05"), -5.25, "CAFE"),
        ("store", pd.Timestamp("2025-02-17 12:00"), -900.0, "BIG STORE"),
    ]
    x = pd.DataFrame(rows, columns=["transaction_id", "ts", "amount", "merchant_raw"])
    x["ts"] = x["ts"].dt.strftime("%Y-%m-%d %H:%M")
    store = (x["merchant_raw"] == "BIG STORE").to_numpy()
    return x.assign(
        user_id="u1",
        currency="USD",
        channel="card",
        profile_users=5,
        profile_typical=np.log(np.where(store, 100.0, 5.0)),
        profile_spread=0.3,
    )


def labels(x: pd.DataFrame) -> pd.Series:
    planted = x["transaction_id"].isin(["big", "dup", "store"])
    return pd.Series(np.where(planted, "anomaly", "normal"))


def by_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("transaction_id")


@pytest.mark.parametrize("scorer", [Rules(), Probabilistic(), Forest(n_estimators=50)])
def test_each_candidate_ranks_the_planted_charges_first(scorer: Rules) -> None:
    x = ledger()
    s = by_id(scorer.fit(x).scores(x))

    top = s.sort_values("score", ascending=False).index[:3]
    assert set(top) == {"big", "dup", "store"}
    assert s.loc["dup", "reason_code"] == "duplicate"
    assert s.loc["store", "reason_code"] == "new_merchant"
    assert s.loc["big", "reason_code"] == "amount_unusual"
    assert np.isinf(s["score"].to_dict()["dup"])


@pytest.mark.parametrize("scorer", [Rules(), Probabilistic()])
def test_flags_carry_evidence_that_renders(scorer: Rules) -> None:
    x = ledger()
    model = Thresholded(scorer, precision=1.0).fit(x, labels(x))
    out = by_id(Checked(model, CONTRACT).predict(x))

    assert set(out.index[out["is_flagged"]]) == {"big", "dup", "store"}
    big = json.loads(str(out.loc["big", "evidence"]))
    assert big["prior_charges"] == 40
    assert big["usual_amount"] == pytest.approx(5.25)
    assert reason("amount_unusual", big).startswith(
        "About 11\N{MULTIPLICATION SIGN} your usual charge here"
    )
    dup = json.loads(str(out.loc["dup", "evidence"]))
    assert dup == {"original_transaction_id": "orig", "minutes_apart": 5.0, "amount": 5.25}
    store = json.loads(str(out.loc["store", "evidence"]))
    assert store["prior_charges"] == 43  # every earlier charge, not those at the store
    assert "First charge here" in reason("new_merchant", store)


def test_rules_new_merchant_floor_and_missing_profile() -> None:
    x = ledger()
    s = by_id(Rules(min_amount=1000.0).scores(x))
    assert pd.isna(s.loc["store", "reason_code"])

    unknown = x.assign(profile_typical=np.nan, profile_spread=np.nan)
    s = by_id(Rules().scores(unknown))
    assert pd.isna(s.loc["store", "reason_code"])  # a first visit nothing can judge
    assert s.loc["big", "reason_code"] == "amount_unusual"  # the user's own history still can


def test_probabilistic_trusts_the_profile_less_as_history_grows() -> None:
    x = ledger()
    # A profile that says coffee costs $50: early charges look cheap, not unusual, and the $60
    # charge after 40 coffees at $5 is judged on the user's own history
    s = by_id(Probabilistic(prior_weight=3.0).scores(x.assign(profile_typical=np.log(50.0))))
    score = s["score"].to_dict()
    assert score["big"] > score["c39"]
    assert score["c01"] < 0.5


def test_search_picks_the_point_with_more_recall() -> None:
    x = ledger()
    # min_amount 1000 hides the $900 first visit, so 0 must win
    model = Thresholded(
        Rules(), precision=1.0, search={"min_amount": [1000.0, 0.0]}, search_rate=3.0
    )
    model.fit(x, labels(x))

    assert model.chosen == {"min_amount": 0.0}
    assert isinstance(model.base, Rules)
    assert model.base.min_amount == 0.0


def test_search_refuses_a_rate_cutoff() -> None:
    model = Thresholded(Rules(), precision=None, rate=0.1, search={"min_amount": [0.0]})
    with pytest.raises(ValueError, match="search needs labels"):
        model.fit(ledger())


def test_rules_validates_its_scales() -> None:
    with pytest.raises(ValueError, match="ratio_scale"):
        Rules(ratio_scale=1.0)


def test_a_one_sided_rank_does_not_flag_cheap_first_visits() -> None:
    x = ledger()
    cheap = x.iloc[[0]].assign(
        transaction_id="cheap", ts="2025-02-18 12:00", amount=-0.5, merchant_raw="KIOSK"
    )
    x = pd.concat([x, cheap], ignore_index=True)
    two = by_id(Forest(n_estimators=100).fit(x).scores(x))
    one = by_id(Forest(n_estimators=100, one_sided_rank=True).fit(x).scores(x))

    # The cheapest charge yet is no more unusual than a typical coffee once the rank is one-sided
    assert one["score"].to_dict()["cheap"] < two["score"].to_dict()["cheap"]
    assert Forest().params["one_sided_rank"] is False


def test_search_records_the_params_in_use() -> None:
    x = ledger()
    model = Thresholded(
        Rules(), precision=1.0, search={"min_amount": [1000.0, 0.0]}, search_rate=3.0
    )
    model.fit(x, labels(x))

    assert model.params["base"].params["min_amount"] == 0.0
    assert model.report["chosen.min_amount"] == 0.0
    assert "search_recall.min_amount_1000.0" in model.report


def test_new_merchant_evidence_says_whether_it_is_above_the_merchants_price() -> None:
    x = ledger()
    model = Thresholded(Rules(), precision=1.0).fit(x, labels(x))
    out = by_id(Checked(model, CONTRACT).predict(x))

    store = json.loads(str(out.loc["store", "evidence"]))
    assert store["above_merchant_usual"] is True  # $900 against a $100 profile
