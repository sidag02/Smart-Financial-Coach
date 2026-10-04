"""Spending spikes in serving (FR-8 §8): the state file, and the scorer the tools call per request.

Spikes are computed on request from the session's ledger, on the categories that session sees
(decision 8), so they always agree with its Spending totals. What serving needs is built ahead,
into one JSON file (`spikes.json`), as goal forecasts are:

- the scorer, with what it fitted (its cutoff, β, dispersion): the promoted model's, or, with no
  promoted model, the simple rule (decision 12): the simple count rule with both product rules,
  its cutoff placed at 0.035 flags per user-month on the pool, with no labels;
- the category season profile table, from every user of the pool on the promoted categorizer's
  predictions, which carry nobody's corrections (§2). The demo bundle builds it from the full
  `--data` pool, not just its accounts, since 3 users can't make a profile.

    state = build_state(pool_transactions, as_of="2026-09-30")  # categories: the predictions
    write_state(state, "build/demo/spikes.json")
    state = load_state("build/demo/spikes.json")
    state.score(user_id, effective_ledger, model_ledger)  # one user's periods, contract-checked

The pool's own rows never leave this function: the file holds the table's sums and counts per
(category, month of year, as-of month) and the scorer's numbers, nothing per user.
"""

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from smart_financial_coach.data.features.monthly import (
    MIN_HISTORY,
    monthly_aggregates,
    period_history,
)
from smart_financial_coach.data.features.season_profiles import (
    DEFAULT_MIN_USERS,
    SeasonTable,
    season_features,
    season_table,
    user_terms,
)
from smart_financial_coach.intelligence.models.artifact import POINTER_FILE
from smart_financial_coach.intelligence.models.base import Model
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.service import load_service
from smart_financial_coach.intelligence.spikes.contract import (
    CONTRACT,
    INPUT_COLUMNS,
    SERVICE,
    SpikeScorer,
)
from smart_financial_coach.intelligence.spikes.count import CountScorer
from smart_financial_coach.intelligence.spikes.threshold import SpikeThresholded

SPIKES_FILE = "spikes.json"
SIMPLE_RULE = "simple-rule"  # the fallback's version: not a promoted model
SIMPLE_RULE_RATE = 0.035  # flags per user-month on the pool (decision 12)
METHOD_MODEL, METHOD_SIMPLE = "model", "simple_rule"
# What each scorer fitted, so the file rebuilds it exactly (attribute names)
FITTED = ("cutoff", "beta", "pooled_dispersion")


def simple_rule() -> SpikeThresholded:
    """The fallback (decision 12): no season, no income, a rate cutoff fitted without labels."""
    return SpikeThresholded(
        CountScorer(season=False, income=False), precision=None, rate=SIMPLE_RULE_RATE
    )


def _fitted(model: Any) -> dict[str, float]:
    return {k: float(getattr(model, k)) for k in FITTED if hasattr(model, k)}


def _spec(model: Model) -> dict[str, Any]:
    """The model's type and constructor params, nested models as `{"$model": ...}`."""
    params = {
        k: ({"$model": _spec(v)} if isinstance(v, Model) else v)
        for k, v in model.params.items()
        if k != "search"  # the search already ran: its choice is in the base's params
    }
    return {"type": model.name, "params": params}


@dataclass(frozen=True)
class SpikeState:
    model: SpikeThresholded
    version: str
    method: str  # METHOD_MODEL or METHOD_SIMPLE ("simple rule" in the UI)
    table: SeasonTable
    as_of: str
    min_users: int = DEFAULT_MIN_USERS

    def periods(
        self, user_id: str, effective: pd.DataFrame, model_basis: pd.DataFrame
    ) -> pd.DataFrame:
        """One user's scoring periods as of `as_of`, from their ledger (`ts`, `amount`,
        `category`; a ledger is one user's, so it has no `user_id` column).

        `effective` has the categories the session sees (the user's corrections applied);
        `model_basis` the same transactions on the promoted categorizer's categories, the basis
        the season table was built on, so the user's own term is left out exactly (§2).
        """
        mine = effective.assign(user_id=user_id)
        theirs = model_basis.assign(user_id=user_id)
        scored = period_history(monthly_aggregates(mine, as_of=self.as_of))
        own = user_terms(period_history(monthly_aggregates(theirs, as_of=self.as_of)))
        season = season_features(scored, self.table, own=own, min_users=self.min_users)
        rows = pd.concat([scored, season.drop(columns="period_id")], axis=1)
        return rows[rows["usual_months"] >= MIN_HISTORY].reset_index(drop=True)

    def score(
        self, user_id: str, effective: pd.DataFrame, model_basis: pd.DataFrame
    ) -> pd.DataFrame:
        """The scoring periods with the contract-checked scorer output joined on."""
        rows = self.periods(user_id, effective, model_basis)
        if rows.empty:
            return rows.assign(score=[], is_flagged=[], evidence=[], model_version=[])
        out = SpikeScorer(self.model, CONTRACT).score_periods(rows[list(INPUT_COLUMNS)])
        return rows.merge(out, on="period_id", how="left", validate="one_to_one")


def build_state(
    pool: pd.DataFrame,
    *,
    as_of: str,
    artifacts_dir: Path | None = None,
    min_users: int = DEFAULT_MIN_USERS,
) -> SpikeState:
    """The serving state from `pool` (every user; `category` = the promoted categorizer's).

    With a promoted spike model it's served as promoted; without one, the simple rule's cutoff is
    fitted on the pool at its rate, with no labels.
    """
    periods = period_history(monthly_aggregates(pool, as_of=as_of))
    table = season_table(periods)
    root = (artifacts_dir or _artifacts()) / SERVICE
    if (root / POINTER_FILE).exists():
        checked = load_service(SERVICE, artifacts_dir)
        model = checked.model
        if not isinstance(model, SpikeThresholded):
            raise TypeError(f"the promoted spike model is a {type(model).__name__}")
        return SpikeState(model, checked.version, METHOD_MODEL, table, as_of, min_users)
    rows = pd.concat(
        [
            periods,
            season_features(periods, table, own=user_terms(periods), min_users=min_users).drop(
                columns="period_id"
            ),
        ],
        axis=1,
    )
    rows = rows[rows["usual_months"] >= MIN_HISTORY].reset_index(drop=True)
    model = simple_rule().fit(rows[list(INPUT_COLUMNS)])
    model.version = SIMPLE_RULE
    return SpikeState(model, SIMPLE_RULE, METHOD_SIMPLE, table, as_of, min_users)


def _artifacts() -> Path:
    from smart_financial_coach.config import get_settings

    return get_settings().artifacts_dir


def write_state(state: SpikeState, out: str | Path, *, users: int) -> None:
    """The state as JSON, written to a temporary name and renamed into place when complete."""
    base = state.model.base
    payload = {
        "meta": {
            "service": SERVICE,
            "model_version": state.version,
            "method": state.method,
            "as_of": state.as_of,
            "min_users": state.min_users,
            "pool_users": users,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "model": {
            "spec": _spec(state.model),
            "fitted": _fitted(state.model),
            "base_fitted": _fitted(base),
        },
        "season_table": state.table.cells.to_dict(orient="list"),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=out.parent, delete=False, suffix=".part") as f:
        json.dump(payload, f)
        temp = Path(f.name)
    temp.chmod(0o644)
    temp.replace(out)


def load_state(path: str | Path) -> SpikeState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta, spec = payload["meta"], payload["model"]
    model = build(spec["spec"])
    if not isinstance(model, SpikeThresholded):
        raise TypeError(f"{path} doesn't hold a thresholded spike scorer")
    for key, value in spec["fitted"].items():
        setattr(model, key, value)
    for key, value in spec["base_fitted"].items():
        setattr(model.base, key, value)
    model.version = meta["model_version"]
    cells = pd.DataFrame(payload["season_table"])
    if not cells.empty:
        cells = cells.astype({"moy": int, "as_of": int, "total": float, "users": int})
    return SpikeState(
        model,
        meta["model_version"],
        meta["method"],
        SeasonTable(cells),
        meta["as_of"],
        int(meta["min_users"]),
    )
