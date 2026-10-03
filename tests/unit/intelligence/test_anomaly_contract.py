import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.intelligence.anomaly.contract import (
    CONTRACT,
    INPUT_COLUMNS,
    AnomalyModel,
    AnomalyScorer,
    scoring_rows,
)
from smart_financial_coach.intelligence.anomaly.reasons import KIND_LABELS, reason
from smart_financial_coach.intelligence.models.contract import ContractError
from smart_financial_coach.intelligence.service import get_service

DUPLICATE = {"original_transaction_id": "t1", "minutes_apart": 6.0, "amount": 38.2}
AMOUNT = {"usual_amount": 20.99, "ratio": 9.0, "prior_charges": 14}
NEW = {"amount": 1249.0, "prior_charges": 400, "rank_in_history": 1.0, "largest_since": None}


def txns() -> pd.DataFrame:
    rows = [
        ("t1", "u1", "2025-01-10 12:00", -38.2, "LYFT RIDE"),
        ("t2", "u1", "2025-01-10 12:06", -38.2, "LYFT RIDE"),
        ("t3", "u1", "2025-01-11 12:00", 900.0, "ACME PAYROLL"),
        ("t4", "u2", "2025-01-12 12:00", -12.0, "LYFT RIDE"),
    ]
    df = pd.DataFrame(rows, columns=["transaction_id", "user_id", "ts", "amount", "merchant_raw"])
    return df.assign(currency="USD", channel="card")


class Fixed(AnomalyModel):
    """Flags the second row as a duplicate; `broken` edits the output to break the contract."""

    name = "test/fixed_anomaly"

    def __init__(self, broken: Any = None) -> None:
        super().__init__()
        self.broken = broken
        self.version = "v1"

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "Fixed":
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        flagged = np.arange(len(x)) == 1
        out = self._output(
            x,
            score=np.where(flagged, np.inf, 0.0),
            is_flagged=flagged,
            reason_code=[("duplicate" if f else None) for f in flagged],
            evidence=[(DUPLICATE if f else None) for f in flagged],
        )
        return self.broken(out) if self.broken else out


def rows() -> pd.DataFrame:
    return scoring_rows(txns(), pool=txns())


def test_scoring_rows_are_outflows_with_profiles() -> None:
    r = rows()

    assert tuple(r.columns) == INPUT_COLUMNS
    assert r["transaction_id"].tolist() == ["t1", "t2", "t4"]


def test_the_service_is_registered_with_its_scorer() -> None:
    service = get_service("unusual_transactions")

    assert service.contract is CONTRACT
    assert service.wrapper is AnomalyScorer


def test_a_good_model_passes() -> None:
    out = AnomalyScorer(Fixed(), CONTRACT).score_transactions(rows())

    assert out["is_flagged"].tolist() == [False, True, False]
    assert json.loads(str(out.loc[1, "evidence"])) == DUPLICATE
    assert pd.isna(out.loc[0, "reason_code"])
    assert pd.isna(out.loc[0, "evidence"])


def _set(row: int, column: str, value: Any) -> Any:
    def edit(out: pd.DataFrame) -> pd.DataFrame:
        out = out.copy()
        out[column] = out[column].astype(object)
        out.at[row, column] = value
        return out

    return edit


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (_set(1, "reason_code", None), "without a reason code"),
        (_set(1, "reason_code", "fraud"), "unknown reason codes"),
        (_set(1, "evidence", None), "not JSON text"),
        (_set(1, "evidence", "{"), "not valid JSON"),
        (_set(1, "evidence", json.dumps({"amount": 1.0})), "lacks 'original_transaction_id'"),
        (
            _set(1, "evidence", json.dumps({**DUPLICATE, "minutes_apart": "6"})),
            "not a finite number",
        ),
        (_set(1, "reason_code", "amount_unusual"), "lacks 'usual_amount'"),
        (_set(0, "reason_code", "duplicate"), "unflagged rows carry"),
        (_set(0, "is_flagged", "yes"), "is_flagged is not boolean"),
        (_set(0, "score", np.nan), "missing values in ['score']"),
        (lambda out: out.assign(model_version=""), "empty model_version"),
    ],
)
def test_contract_catches_broken_flags(broken: Any, message: str) -> None:
    with pytest.raises(ContractError, match=message.replace("[", r"\[").replace("]", r"\]")):
        AnomalyScorer(Fixed(broken), CONTRACT).score_transactions(rows())


def test_score_transactions_wants_scoring_rows() -> None:
    scorer = AnomalyScorer(Fixed(), CONTRACT)
    with pytest.raises(ValueError, match="scoring_rows"):
        scorer.score_transactions(txns())
    with pytest.raises(ValueError, match="only outflows"):
        scorer.score_transactions(rows().assign(amount=1.0))


@pytest.mark.parametrize(
    ("code", "evidence", "text"),
    [
        (
            "duplicate",
            DUPLICATE,
            "Same amount at the same merchant, 6 minutes after an earlier charge.",
        ),
        ("duplicate", {**DUPLICATE, "minutes_apart": 1.2}, "1 minute after"),
        ("duplicate", {**DUPLICATE, "minutes_apart": 0.2}, "less than a minute after"),
        (
            "amount_unusual",
            AMOUNT,
            "About 9× your usual charge here ($20.99, from 14 earlier charges).",
        ),
        (
            "amount_unusual",
            {**AMOUNT, "ratio": 2.34, "prior_charges": 1},
            "About 2.3× your usual charge here ($20.99, from 1 earlier charge).",
        ),
        (
            "amount_unusual",
            {**AMOUNT, "ratio": 12.6, "usual_amount": 1234.5},
            "About 13× your usual charge here ($1,234.50,",
        ),
        ("new_merchant", NEW, "First charge here, and your largest charge yet."),
        (
            "new_merchant",
            {**NEW, "largest_since": "2024-10-03 12:00"},
            "your largest charge since Oct 2024.",
        ),
        (
            "new_merchant",
            {**NEW, "rank_in_history": 0.957, "largest_since": "2026-09-01 12:00"},
            "larger than 95% of your earlier charges.",
        ),
        ("new_merchant", {**NEW, "prior_charges": 12}, "one of your first charges"),
        (
            "new_merchant",
            {**NEW, "prior_charges": 0, "rank_in_history": None},
            "one of your first charges",
        ),
    ],
)
def test_reasons_render_from_evidence(code: str, evidence: dict[str, Any], text: str) -> None:
    assert text in reason(code, evidence)
    assert reason(code, json.dumps(evidence)) == reason(code, evidence)


def test_reasons_refuse_bad_evidence() -> None:
    with pytest.raises(ValueError, match="unknown reason code"):
        reason("fraud", DUPLICATE)
    with pytest.raises(ValueError, match="lacks 'ratio'"):
        reason("amount_unusual", {"usual_amount": 20.99, "prior_charges": 3})


def test_every_reason_code_has_a_kind_label() -> None:
    from smart_financial_coach.intelligence.anomaly.contract import REASON_CODES

    assert set(KIND_LABELS) == set(REASON_CODES)
