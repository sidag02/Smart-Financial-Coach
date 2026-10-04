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

Without a promoted model, `baseline=True` writes a file with naive pace behind the same pipeline
(owner decision 10 on #54): a simple projection of the pace so far, with no chance and no range,
until a model is promoted and swaps in without changing anything else.
"""

import threading
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.store import load_meta, load_transactions
from smart_financial_coach.intelligence.forecasting.baseline import NaivePace
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
BASELINE_VERSION = "naive-pace"  # the baseline's version: not a promoted model
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
    baseline: bool = False,
) -> ForecastRun:
    """Fit every user's forecast state at `data`'s last day with the promoted model, into the
    forecasts file `out`. A user without a full month of history gets no state, and so no
    forecast. Outcomes and truth tables are never read. `baseline` writes naive pace instead,
    which has no state, for when no model is promoted."""
    started = perf_counter()
    meta = load_meta(data)
    as_of = date.fromisoformat(meta["calendar_end"])
    if baseline:
        spec = {"name": NaivePace.name, "version": BASELINE_VERSION, "params": {}}
        write_forecasts(out, _meta(meta, as_of, BASELINE_VERSION), spec, {}, overwrite=overwrite)
        return ForecastRun(BASELINE_VERSION, 0, perf_counter() - started)
    if as_of != pd.Period(as_of, freq="M").end_time.date():
        # States must end at the forecasts' month: a partial last month isn't in the history,
        # so every forecast at this as_of would be refused (review on #55)
        raise ValueError(f"the dataset ends {as_of}, not on a month end")
    service = load_service(SERVICE, artifacts_dir)
    model = service.model
    if not isinstance(model, PathsModel):
        raise TypeError(f"the {SERVICE} service is a {type(model).__name__}, not a paths model")
    if model.personas != "mixture":
        # A real user has no persona label (owner decision 11 on #54)
        raise TypeError(f"the {SERVICE} model reads a persona label; serve a persona mixture")
    states: dict[str, ForecastState] = {}
    for user_id, frame in load_transactions(data)[["user_id", "ts", "amount"]].groupby("user_id"):
        history = monthly_net(frame, as_of)
        if len(history):
            states[str(user_id)] = model.state_for(history, "")  # never a persona label
    write_forecasts(
        out,
        _meta(meta, as_of, model.version),
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


def _meta(dataset: dict[str, str], as_of: date, version: str) -> dict[str, str]:
    return {
        "model_version": version,
        "data_spec_name": dataset.get("spec_name", ""),
        "data_spec_hash": dataset.get("spec_hash", ""),
        "as_of": as_of.isoformat(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


class GoalForecaster:
    """The promoted goal forecast as serving runs it: the model's learned values and each user's
    state, from a forecasts file; or naive pace, the baseline, when no model is promoted
    (`baseline`). Safe to share across request threads: the paths cache is locked, and its
    arrays are never changed."""

    def __init__(
        self, model: PathsModel | NaivePace, states: dict[str, ForecastState], as_of: date
    ) -> None:
        self.model = model
        self.states = states
        self.as_of = as_of
        self._paths: OrderedDict[str, Floats] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str | Path) -> Self:
        stored = load_forecasts(path)
        spec = stored.model
        model = build({"type": spec["name"], "params": spec["params"]})
        model.version = str(spec["version"])
        if isinstance(model, PathsModel):
            model.typical_total = float(spec["typical_total"])
            model.fitted_spread = float(spec["fitted_spread"])
        elif not isinstance(model, NaivePace):
            raise TypeError(f"{path} holds a {spec['name']} model, not a paths model or naive pace")
        return cls(model, stored.states, date.fromisoformat(stored.meta["as_of"]))

    @property
    def baseline(self) -> bool:
        """Naive pace, not a promoted model: a simple projection with no chance or range."""
        return isinstance(self.model, NaivePace)

    def covers(self, user_id: str) -> bool:
        return self.baseline or user_id in self.states

    @property
    def version(self) -> str:
        return self.model.version

    def state(self, user_id: str) -> ForecastState | None:
        return self.states.get(user_id)

    def forecast(self, rows: pd.DataFrame, only: Collection[str] | None = None) -> pd.DataFrame:
        """The goal forecast for `rows` (the contract's input columns), checked against the
        contract like every service's output. Forecast a user's goals together, a draft
        included: their shares are capped as one set (§3). `only` limits the output to those
        example ids, so a caller needing one goal doesn't pay for the rest."""
        wanted = rows if only is None else rows[rows["example_id"].astype(str).isin(set(only))]
        if isinstance(self.model, NaivePace):
            out = self.model.predict(wanted)
        else:
            out = self.model.predict_with_states(rows, self.states, paths_of=self.paths, only=only)
        if errors := CONTRACT.violations(wanted.reset_index(drop=True), out):
            raise ContractError(f"goal forecast {self.version} broke its contract: {errors}")
        return out

    def paths(self, user_id: str) -> Floats:
        """The user's simulated monthly net savings: the same paths `forecast` used."""
        if not isinstance(self.model, PathsModel):
            raise ValueError("the baseline has no paths")
        with self._lock:  # request threads share it (review on #55)
            paths = self._paths.get(user_id)
            if paths is not None:
                self._paths.move_to_end(user_id)
                return paths
        paths = self.model.paths_for(user_id, self.as_of, self.states[user_id])
        with self._lock:
            self._paths[user_id] = paths
            while len(self._paths) > _PATHS_CACHED:
                self._paths.popitem(last=False)
        return paths

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
