"""The spending-spikes service contract (FR-8 §1), and what callers get from `load_service`.

in:  one row per (user, spending category, complete month) with at least `MIN_HISTORY` earlier
     months: the monthly aggregates, their point-in-time history and the category season
     profile columns. `scoring_periods` builds them
out: period_id, score, is_flagged, evidence, model_version

A flag must pass both product rules, fixed before the round and never tuned (FR-8 §3, decision
1): the month's spend is at least `SPEND_FLOOR` times the user's usual, and a usual month has at
least `MIN_USUAL_COUNT` purchases. It must carry the evidence its reason needs (NFR-7); an
unflagged row carries none. Evidence is built here from the input row, never by a model, so its
numbers are the user's own (FR-8 §7, NFR-2).
"""

import json
import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from smart_financial_coach.data.features.monthly import (
    AGGREGATE_COLUMNS,
    HISTORY_COLUMNS,
    INCOME,
    MIN_HISTORY,
    monthly_aggregates,
    period_history,
)
from smart_financial_coach.data.features.season_profiles import COLUMNS as SEASON_COLUMNS
from smart_financial_coach.data.features.season_profiles import (
    DEFAULT_MIN_USERS,
    season_profiles,
)
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.contract import Checked, Contract
from smart_financial_coach.intelligence.service import register_service

SERVICE = "spending_spikes"
SPEND_FLOOR = 1.3  # FR-2's weak_lift: below it, the reason would describe ordinary spending
MIN_USUAL_COUNT = 2.0  # purchases in a usual month: 1 becoming 4 isn't a meaningful spike
INPUT_COLUMNS = (*AGGREGATE_COLUMNS, *HISTORY_COLUMNS, *SEASON_COLUMNS[1:])
OUTPUT_COLUMNS = ("period_id", "score", "is_flagged", "evidence", "model_version")
# Columns that must never reach a scorer: real users have no persona (FR-11, decision 11)
FORBIDDEN_INPUT = frozenset({"persona"})
EVIDENCE = (
    "category",
    "period_start",
    "actual",
    "usual",
    "usual_months",
    "excess",
    "ratio",
    "count",
    "usual_count",
)
TEXT = frozenset({"category", "period_start"})
TOLERANCE = 0.005  # dollars and ratios are rounded to cents in evidence


def scoring_periods(
    transactions: pd.DataFrame,
    pool: pd.DataFrame,
    through: str | None = None,
    min_users: int = DEFAULT_MIN_USERS,
) -> pd.DataFrame:
    """The service's input for `transactions`' users, with season profiles from `pool`.

    Both frames have a `category` column on the basis the caller wants (FR-8 §2). Serving and
    test scoring pass every user as the pool; validation passes train users only, so test users
    never shape model selection. Periods with fewer than `MIN_HISTORY` earlier months are left
    out: there's too little history to judge them (FR-8 §1).
    """
    scored = period_history(monthly_aggregates(transactions, through=through))
    pool_periods = period_history(monthly_aggregates(pool, through=through))
    season = season_profiles(scored, pool_periods, min_users=min_users)
    rows = pd.concat([scored, season.drop(columns="period_id")], axis=1)
    rows = rows[rows["usual_months"] >= MIN_HISTORY].reset_index(drop=True)
    return rows[list(INPUT_COLUMNS)]


def evidence_for(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Each row's evidence: the user's own numbers only, rounded as the reason shows them."""
    actual = rows["spend"].to_numpy(dtype=float)
    usual = rows["usual"].to_numpy(dtype=float)
    return [
        {
            "category": str(c),
            "period_start": str(s),
            "actual": round(float(a), 2),
            "usual": round(float(u), 2),
            "usual_months": int(n),
            "excess": round(float(a - u), 2),
            "ratio": round(float(a / u), 3) if u > 0 else None,
            "count": int(k),
            "usual_count": round(float(uc), 2),
        }
        for c, s, a, u, n, k, uc in zip(
            rows["category"],
            rows["period_start"],
            actual,
            usual,
            rows["usual_months"],
            rows["count"],
            rows["usual_count"],
            strict=True,
        )
    ]


def evidence_errors(evidence: Any) -> list[str]:
    """What's wrong with one flag's evidence, including the two product rules."""
    if not isinstance(evidence, str):
        return ["evidence is not JSON text"]
    try:
        e = json.loads(evidence)
    except json.JSONDecodeError:
        return ["evidence is not valid JSON"]
    if not isinstance(e, dict):
        return ["evidence is not a JSON object"]
    errors = []
    for key in EVIDENCE:
        value = e.get(key)
        if key in TEXT:
            if not isinstance(value, str) or not value:
                errors.append(f"evidence {key!r} is not text")
        elif isinstance(value, bool) or not isinstance(value, int | float):
            errors.append(f"evidence {key!r} is not a number")
        elif not math.isfinite(value):
            errors.append(f"evidence {key!r} is not finite")
    if errors:
        return errors
    if e["usual"] <= 0 or e["actual"] < SPEND_FLOOR * e["usual"] - TOLERANCE:
        errors.append(f"spend isn't {SPEND_FLOOR}x usual (the spend floor)")
    if e["usual_count"] < MIN_USUAL_COUNT:
        errors.append(f"a usual month has fewer than {MIN_USUAL_COUNT:g} purchases")
    if e["usual_months"] < MIN_HISTORY:
        errors.append(f"fewer than {MIN_HISTORY} earlier months")
    if abs(e["excess"] - (e["actual"] - e["usual"])) > 2 * TOLERANCE:
        errors.append("excess isn't actual minus usual")
    return errors


def _check(out: pd.DataFrame) -> list[str]:
    errors = []
    flagged = out["is_flagged"]
    if not flagged.map(lambda v: isinstance(v, bool | np.bool_)).all():
        return ["is_flagged is not boolean"]
    flagged = flagged.astype(bool)
    if out["score"].map(lambda v: isinstance(v, bool) or not isinstance(v, int | float)).any():
        errors.append("score is not a number")
    for evidence in out.loc[flagged, "evidence"]:
        if problems := evidence_errors(evidence):
            errors.append(f"a flag breaks the contract: {problems[0]}")
            break
    if out.loc[~flagged, "evidence"].notna().any():
        errors.append("unflagged rows carry evidence")
    if (out["model_version"].astype(str) == "").any():
        errors.append("empty model_version")
    return errors


class SpikeScorer(Checked):
    """The promoted spending-spike scorer, contract-checked. `score_periods` is the service's
    verb; its input comes from `scoring_periods`."""

    def score_periods(self, rows: pd.DataFrame) -> pd.DataFrame:
        if forbidden := sorted(FORBIDDEN_INPUT & set(rows.columns)):
            raise ValueError(f"spike scorers never see {forbidden} (FR-11, decision 11)")
        if missing := [c for c in INPUT_COLUMNS if c not in rows.columns]:
            raise ValueError(f"scoring periods lack {missing}; build them with scoring_periods")
        if (rows["category"] == INCOME).any():
            raise ValueError("Income is never a spending period")
        if (rows["usual_months"] < MIN_HISTORY).any():
            raise ValueError(f"periods need {MIN_HISTORY} earlier months to be scored")
        return self.predict(rows[list(INPUT_COLUMNS)].reset_index(drop=True))


CONTRACT = Contract(SERVICE, "period_id", OUTPUT_COLUMNS, _check, nullable=("evidence",))
register_service(CONTRACT, SpikeScorer)


class SpikeModel(BaseModel):
    """Base for spike models. They are unsupervised: `fit` must not need labels.

    A model supplies a score and its own flag decision; `_output` applies both product rules on
    top, so no model can flag a month they exclude, and builds the evidence from the row.
    """

    contract: ClassVar[Contract] = CONTRACT

    @staticmethod
    def eligible(x: pd.DataFrame) -> np.ndarray:
        """Periods the product rules allow to be flagged."""
        usual = x["usual"].to_numpy(dtype=float)
        return (
            (usual > 0)
            & (x["spend"].to_numpy(dtype=float) >= SPEND_FLOOR * usual)
            & (x["usual_count"].to_numpy(dtype=float) >= MIN_USUAL_COUNT)
        )

    def _output(self, x: pd.DataFrame, score: Any, is_flagged: Any) -> pd.DataFrame:
        flagged = np.asarray(is_flagged, dtype=bool) & self.eligible(x)
        evidence: list[str | None] = [None] * len(x)
        for i, e in zip(np.flatnonzero(flagged), evidence_for(x[flagged]), strict=True):
            evidence[i] = json.dumps(e)
        return pd.DataFrame(
            {
                "period_id": x["period_id"].to_numpy(),
                "score": np.asarray(score, dtype=np.float64),
                "is_flagged": flagged,
                "evidence": evidence,
                "model_version": self.version,
            }
        )
