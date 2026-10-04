import json
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.intelligence.models.contract import ContractError
from smart_financial_coach.intelligence.service import get_service
from smart_financial_coach.intelligence.spikes.contract import (
    CONTRACT,
    INPUT_COLUMNS,
    SpikeModel,
    SpikeScorer,
    evidence_errors,
    scoring_periods,
)
from smart_financial_coach.intelligence.spikes.reasons import reason

EVIDENCE = {
    "category": "Dining",
    "period_start": "2026-08-01",
    "actual": 1853.0,
    "usual": 748.0,
    "usual_months": 12,
    "excess": 1105.0,
    "ratio": 2.477,
    "count": 48,
    "usual_count": 18.6,
}


def ledger() -> pd.DataFrame:
    """u1 buys Dining 4 times a month for 6 months, then 12 times; u2 buys Dining once a month,
    then 4 times; both have one Income payment a month."""
    rows = []
    for user, usual, last in (("u1", 4, 12), ("u2", 1, 4)):
        for m in range(1, 8):
            n = last if m == 7 else usual
            rows += [(user, f"2025-{m:02d}-1{k % 10} 12:00", -25.0, "Dining") for k in range(n)]
            rows.append((user, f"2025-{m:02d}-01 09:00", 3000.0, "Income"))
    return pd.DataFrame(rows, columns=["user_id", "ts", "amount", "category"])


class Fixed(SpikeModel):
    """Flags every period it's given; `broken` edits the output to break the contract."""

    name = "test/fixed_spikes"

    def __init__(self, broken: Any = None) -> None:
        super().__init__()
        self.broken = broken
        self.version = "v1"

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "Fixed":
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        out = self._output(x, np.arange(len(x), dtype=float), np.ones(len(x), dtype=bool))
        return self.broken(out) if self.broken else out


def scorer(broken: Any = None) -> SpikeScorer:
    return SpikeScorer(Fixed(broken), CONTRACT)


def rows() -> pd.DataFrame:
    t = ledger()
    return scoring_periods(t, t, as_of="2025-07-31", min_users=1)


def test_the_service_is_registered() -> None:
    service = get_service("spending_spikes")
    assert service.contract is CONTRACT
    assert service.wrapper is SpikeScorer


def test_scoring_periods_need_three_earlier_months() -> None:
    r = rows()
    assert tuple(r.columns) == INPUT_COLUMNS
    assert sorted(set(r["period_start"])) == [f"2025-{m:02d}-01" for m in range(4, 8)]
    assert set(r["category"]) == {"Dining"}


def test_only_the_product_rules_let_a_period_through() -> None:
    r = rows()
    out = scorer().score_periods(r).set_index("period_id")
    # u1's July (12 against 4) passes both rules; u2's July (4 against 1) has too little volume;
    # every month at the usual level fails the spend floor
    assert out.index[out["is_flagged"]].tolist() == ["u1|Dining|2025-07-01"]
    e = json.loads(str(out.loc["u1|Dining|2025-07-01", "evidence"]))
    assert e == {
        "category": "Dining",
        "period_start": "2025-07-01",
        "actual": 300.0,
        "usual": 100.0,
        "usual_months": 6,
        "excess": 200.0,
        "ratio": 3.0,
        "count": 12,
        "usual_count": 4.0,
    }
    assert out.loc[~out["is_flagged"], "evidence"].isna().all()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda r: r.assign(persona="freelancer"), "never see"),
        (lambda r: r.assign(expected_spend=1.0), "never see"),
        (lambda r: r.assign(tier="clear"), "never see"),
        (lambda r: r.drop(columns="profile_season"), "lack"),
        (lambda r: r.assign(category="Income"), "Income"),
        (lambda r: r.assign(usual_months=2), "3 earlier months"),
    ],
)
def test_the_scorer_rejects_bad_input(edit: Any, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        scorer().score_periods(edit(rows()))


def test_an_incomplete_month_is_never_scored() -> None:
    t = ledger()
    assert scoring_periods(t, t, as_of="2025-07-30", min_users=1)["period_start"].max() == (
        "2025-06-01"
    )


def test_the_scorer_passes_only_the_contracts_columns() -> None:
    seen: list[list[str]] = []

    def spy(out: pd.DataFrame) -> pd.DataFrame:
        return out

    model = Fixed(spy)
    original = model.predict

    def predict(x: pd.DataFrame) -> pd.DataFrame:
        seen.append(list(x.columns))
        return original(x)

    model.predict = predict  # type: ignore[method-assign]
    SpikeScorer(model, CONTRACT).score_periods(rows().assign(extra_column=1.0))
    assert seen == [list(INPUT_COLUMNS)]


def flag_all(out: pd.DataFrame) -> pd.DataFrame:
    return out.assign(is_flagged=True, evidence=json.dumps(EVIDENCE))


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (
            lambda o: flag_all(o).assign(evidence=json.dumps(EVIDENCE | {"actual": 900.0})),
            "spend floor",
        ),
        (
            lambda o: flag_all(o).assign(evidence=json.dumps(EVIDENCE | {"usual_count": 1.5})),
            "fewer than 2",
        ),
        (lambda o: flag_all(o).assign(evidence=json.dumps(EVIDENCE | {"excess": 1.0})), "excess"),
        (lambda o: flag_all(o).assign(evidence=json.dumps(EVIDENCE | {"ratio": 3.1})), "ratio"),
        (lambda o: flag_all(o), "another period's evidence"),
        (lambda o: flag_all(o).assign(evidence=None), "not JSON text"),
        (lambda o: o.assign(evidence=json.dumps(EVIDENCE)), "unflagged rows carry evidence"),
        (lambda o: o.assign(model_version=""), "empty model_version"),
    ],
)
def test_the_contract_catches_broken_flags(broken: Any, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        scorer(broken).score_periods(rows())


def test_evidence_must_be_complete() -> None:
    incomplete = {k: v for k, v in EVIDENCE.items() if k != "usual"}
    assert evidence_errors(json.dumps(incomplete)) == ["evidence 'usual' is not a number"]
    assert evidence_errors(json.dumps(EVIDENCE)) == []


def test_the_reason_quotes_only_the_evidence() -> None:
    assert reason(EVIDENCE) == (
        "You spent $1,853 on Dining in August 2026, $1,105 more than your average month over "
        "the past year ($748). That came from 48 purchases, against about 19 in an average month."
    )
    short = EVIDENCE | {"usual_months": 5, "usual_count": 19.0, "count": 1}
    assert "over the past 5 months" in reason(short)
    assert "1 purchase, against 19 in an average month" in reason(short)


def test_the_reasons_arithmetic_adds_up_in_whole_dollars() -> None:
    e = EVIDENCE | {"actual": 1000.6, "usual": 500.4, "excess": 500.2, "ratio": 1000.6 / 500.4}
    assert reason(e).startswith(
        "You spent $1,001 on Dining in August 2026, $501 more than your average month over the "
        "past year ($500)."
    )
    assert "against about 3 in an average month" in reason(EVIDENCE | {"usual_count": 2.5})


def test_a_reason_needs_valid_evidence() -> None:
    with pytest.raises(ValueError, match="spend floor"):
        reason(EVIDENCE | {"actual": 800.0})
