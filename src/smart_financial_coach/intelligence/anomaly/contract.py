"""The unusual-transactions service contract (FR-7 §1), and what callers get from `load_service`.

in:  the outflows (amount < 0) of one or more users, each with its full earlier history, as
     model-visible rows plus the merchant-profile columns: `scoring_rows` builds them
out: transaction_id, score, is_flagged, reason_code, evidence, model_version

A flagged row must carry a reason code and the evidence its reason template needs (NFR-7); an
unflagged row carries neither. `evidence` is a JSON object, so it can be stored as text and quoted
by the coach (FR-14).
"""

import json
import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from smart_financial_coach.data.features.history import outflows
from smart_financial_coach.data.features.merchant_profiles import (
    COLUMNS as PROFILE_COLUMNS,
)
from smart_financial_coach.data.features.merchant_profiles import (
    DEFAULT_MIN_USERS,
    profile_features,
)
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.contract import Checked, Contract
from smart_financial_coach.intelligence.service import register_service

TRANSACTION_COLUMNS = (
    "transaction_id",
    "user_id",
    "ts",
    "amount",
    "currency",
    "merchant_raw",
    "channel",
)
INPUT_COLUMNS = (*TRANSACTION_COLUMNS, *PROFILE_COLUMNS[1:])
OUTPUT_COLUMNS = (
    "transaction_id",
    "score",
    "is_flagged",
    "reason_code",
    "evidence",
    "model_version",
)
# Evidence each reason's template needs (§7). Values are numbers, except IDs and dates; a
# `None` is allowed only where listed in OPTIONAL.
EVIDENCE: dict[str, tuple[str, ...]] = {
    "duplicate": ("original_transaction_id", "minutes_apart", "amount"),
    "amount_unusual": ("usual_amount", "ratio", "prior_charges"),
    "new_merchant": ("amount", "prior_charges", "rank_in_history", "largest_since"),
}
REASON_CODES = tuple(EVIDENCE)
TEXT = frozenset({"original_transaction_id", "largest_since"})
OPTIONAL = frozenset({"rank_in_history", "largest_since"})  # absent without earlier charges
# Evidence a reason may carry beyond what its template needs: a yes/no, never a number, so
# flag files written before it was added still render. `above_merchant_usual`: a new-merchant
# charge above its merchant profile's typical price (owner wording decision on #40). It says
# nothing about what other users pay beyond that one comparison (NFR-2)
EXTRA_EVIDENCE: dict[str, tuple[str, ...]] = {"new_merchant": ("above_merchant_usual",)}


def scoring_rows(
    transactions: pd.DataFrame, pool: pd.DataFrame, min_users: int = DEFAULT_MIN_USERS
) -> pd.DataFrame:
    """The service's input: `transactions`' outflows with merchant profiles built from `pool`.

    Serving and test scoring pass every user as the pool; validation passes train users only, so
    test users never shape model selection (FR-7 §2).
    """
    rows = outflows(transactions)[list(TRANSACTION_COLUMNS)].reset_index(drop=True)
    profiles = profile_features(rows, pool, min_users=min_users)
    return pd.concat([rows, profiles.drop(columns="transaction_id")], axis=1)


def evidence_errors(reason_code: str, evidence: Any) -> list[str]:
    """What's wrong with one flag's evidence, if anything."""
    if not isinstance(evidence, str):
        return ["evidence is not JSON text"]
    try:
        values = json.loads(evidence)
    except json.JSONDecodeError:
        return ["evidence is not valid JSON"]
    if not isinstance(values, dict):
        return ["evidence is not a JSON object"]
    errors = []
    for key in EVIDENCE[reason_code]:
        if key not in values:
            errors.append(f"{reason_code} evidence lacks {key!r}")
            continue
        value = values[key]
        if value is None:
            if key not in OPTIONAL:
                errors.append(f"{reason_code} evidence {key!r} is null")
        elif key in TEXT:
            if not isinstance(value, str) or not value:
                errors.append(f"{reason_code} evidence {key!r} is not text")
        elif (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            errors.append(f"{reason_code} evidence {key!r} is not a finite number")
    for key in EXTRA_EVIDENCE.get(reason_code, ()):
        if values.get(key) is not None and not isinstance(values[key], bool):
            errors.append(f"{reason_code} evidence {key!r} is not true, false or null")
    return errors


def _check(out: pd.DataFrame) -> list[str]:
    errors = []
    flagged = out["is_flagged"]
    if not flagged.map(lambda v: isinstance(v, bool | np.bool_)).all():
        errors.append("is_flagged is not boolean")
        return errors
    flagged = flagged.astype(bool)
    if out["score"].map(lambda v: isinstance(v, bool) or not isinstance(v, int | float)).any():
        errors.append("score is not a number")
    codes = out.loc[flagged, "reason_code"]
    if codes.isna().any():
        errors.append(f"{int(codes.isna().sum())} flags without a reason code (NFR-7)")
    elif unknown := sorted(set(codes) - set(REASON_CODES)):
        errors.append(f"unknown reason codes {unknown[:5]}")
    else:
        for code, evidence in zip(codes, out.loc[flagged, "evidence"], strict=True):
            if problems := evidence_errors(code, evidence):
                errors.append(problems[0])
                break
    unflagged = out.loc[~flagged, ["reason_code", "evidence"]]
    if unflagged.notna().any().any():
        errors.append("unflagged rows carry a reason code or evidence")
    if (out["model_version"].astype(str) == "").any():
        errors.append("empty model_version")
    return errors


class AnomalyScorer(Checked):
    """The promoted unusual-transaction scorer, contract-checked. `score_transactions` is the
    service's verb; its input comes from `scoring_rows`."""

    def score_transactions(self, rows: pd.DataFrame) -> pd.DataFrame:
        if missing := [c for c in INPUT_COLUMNS if c not in rows.columns]:
            raise ValueError(f"scoring rows lack {missing}; build them with scoring_rows")
        if (rows["amount"] >= 0).any():
            raise ValueError("only outflows (amount < 0) are scored")
        return self.predict(rows)


CONTRACT = Contract(
    "unusual_transactions",
    "transaction_id",
    OUTPUT_COLUMNS,
    _check,
    nullable=("reason_code", "evidence"),
)
register_service(CONTRACT, AnomalyScorer)


class AnomalyModel(BaseModel):
    """Base for unusual-transaction models. They are unsupervised: `fit` must not need labels."""

    contract: ClassVar[Contract] = CONTRACT

    def _output(
        self,
        x: pd.DataFrame,
        score: Any,
        is_flagged: Any,
        reason_code: Any,
        evidence: list[dict[str, Any] | None],
    ) -> pd.DataFrame:
        """The contract's frame; `evidence` dicts are serialized, None for unflagged rows."""
        return pd.DataFrame(
            {
                "transaction_id": x["transaction_id"].to_numpy(),
                "score": np.asarray(score, dtype=np.float64),
                "is_flagged": np.asarray(is_flagged, dtype=bool),
                "reason_code": pd.Series(reason_code, dtype=object).to_numpy(),
                "evidence": [None if e is None else json.dumps(e) for e in evidence],
                "model_version": self.version,
            }
        )
