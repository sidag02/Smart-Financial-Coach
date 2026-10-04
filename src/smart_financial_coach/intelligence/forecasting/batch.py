"""Goal forecasts in serving (FR-11 and FR-12 design, §7): the nightly batch, and the forecaster
the tools call per request.

The nightly batch fits each user's net-savings forecast state at the dataset's last day with the
promoted model, and writes it to a forecasts file with what serving needs of the model (its
version, params, typical total allocation and spread). Per request, `GoalForecaster` rebuilds the
model from that file and runs the same `PathsModel` code the round scored, on the stored states:
paths from a seed of (user, `as_of` month, model version), so every answer for the same goal and
model version is identical (NFR-8), and a draft gets exactly the numbers the goal gets once saved.

    run = forecast_dataset("data/synthetic/default.sqlite", "build/demo/forecasts.json")
    forecaster = GoalForecaster.load("build/demo/forecasts.json")
    forecaster.forecast(goal_rows)  # contract-checked, one row per goal
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.store import load_meta, load_transactions, load_users
from smart_financial_coach.intelligence.forecasting.contract import CONTRACT, INTERVAL, SERVICE
from smart_financial_coach.intelligence.forecasting.paths import ForecastState, PathsModel
from smart_financial_coach.intelligence.forecasting.savings import monthly_net, run_balance
from smart_financial_coach.intelligence.forecasting.states import (
    load_forecasts,
    write_forecasts,
)
from smart_financial_coach.intelligence.models.base import describe
from smart_financial_coach.intelligence.models.contract import ContractError
from smart_financial_coach.intelligence.models.registry import build
from smart_financial_coach.intelligence.service import load_service

Floats = npt.NDArray[np.float64]

FORECASTS_FILE = "forecasts.json"
_PATHS_CACHED = 64  # users whose paths are kept in memory (about 1 MB each)


@dataclass(frozen=True)
class ForecastRun:
    model_version: str
    users: int
    seconds: float


def forecast_dataset(
    data: str | Path,
    out: str | Path,
    *,
    artifacts_dir: Path | None = None,
    overwrite: bool = False,
) -> ForecastRun:
    """Fit every user's forecast state at `data`'s last day with the promoted model, into the
    forecasts file `out`. A user without a full month of history gets no state, and so no
    forecast. Outcomes and truth tables are never read."""
    started = perf_counter()
    service = load_service(SERVICE, artifacts_dir)
    model = service.model
    if not isinstance(model, PathsModel):
        raise TypeError(f"the {SERVICE} service is a {type(model).__name__}, not a paths model")
    meta = load_meta(data)
    as_of = date.fromisoformat(meta["calendar_end"])
    users = load_users(data)
    persona = dict(zip(users["user_id"].astype(str), users["persona"].astype(str), strict=True))
    states: dict[str, ForecastState] = {}
    for user_id, frame in load_transactions(data)[["user_id", "ts", "amount"]].groupby("user_id"):
        history = monthly_net(frame, as_of)
        if len(history):
            states[str(user_id)] = model.state_for(history, persona[str(user_id)])
    write_forecasts(
        out,
        {
            "model_version": model.version,
            "data_spec_name": meta.get("spec_name", ""),
            "data_spec_hash": meta.get("spec_hash", ""),
            "as_of": as_of.isoformat(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        {
            "name": model.name,
            "version": model.version,
            "params": describe(dict(model.params)),
            "typical_total": model.typical_total,
            "fitted_spread": model.fitted_spread,
        },
        states,
        overwrite=overwrite,
    )
    return ForecastRun(model.version, len(states), perf_counter() - started)


class GoalForecaster:
    """The promoted goal forecast as serving runs it: the model's learned values and each user's
    state, from a forecasts file. Safe to share across request threads (reads only; the paths
    cache holds immutable arrays)."""

    def __init__(self, model: PathsModel, states: dict[str, ForecastState], as_of: date) -> None:
        self.model = model
        self.states = states
        self.as_of = as_of
        self._paths: OrderedDict[str, Floats] = OrderedDict()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        stored = load_forecasts(path)
        spec = stored.model
        model = build({"type": spec["name"], "params": spec["params"]})
        if not isinstance(model, PathsModel):
            raise TypeError(f"{path} holds a {spec['name']} model, not a paths model")
        model.version = str(spec["version"])
        model.typical_total = float(spec["typical_total"])
        model.fitted_spread = float(spec["fitted_spread"])
        return cls(model, stored.states, date.fromisoformat(stored.meta["as_of"]))

    @property
    def version(self) -> str:
        return self.model.version

    def state(self, user_id: str) -> ForecastState | None:
        return self.states.get(user_id)

    def forecast(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The goal forecast for `rows` (the contract's input columns), checked against the
        contract like every service's output. Forecast a user's goals together, a draft
        included: their shares are capped as one set (§3)."""
        out = self.model.predict_with_states(rows, self.states)
        if errors := CONTRACT.violations(rows, out):
            raise ContractError(f"goal forecast {self.version} broke its contract: {errors}")
        return out

    def paths(self, user_id: str) -> Floats:
        """The user's simulated monthly net savings: the same paths `forecast` used."""
        if user_id not in self._paths:
            state = self.states[user_id]
            self._paths[user_id] = self.model.paths_for(user_id, self.as_of, state)
            while len(self._paths) > _PATHS_CACHED:
                self._paths.popitem(last=False)
        self._paths.move_to_end(user_id)
        return self._paths[user_id]

    def monthly(
        self, user_id: str, saved: float, share: float, months: int
    ) -> list[dict[str, Any]]:
        """The balance's median and 80% range at each of the next `months` month ends, over the
        same paths and share as the forecast, so the chart ends where the forecast does."""
        balances = run_balance(saved, share, self.paths(user_id)[:, :months])
        lo, mid, hi = np.quantile(balances, [INTERVAL[0], 0.5, INTERVAL[1]], axis=0)
        start = pd.Period(self.as_of, freq="M")
        return [
            {
                "month": str(start + i + 1),
                "median": float(mid[i]),
                "low": float(lo[i]),
                "high": float(hi[i]),
            }
            for i in range(months)
        ]
