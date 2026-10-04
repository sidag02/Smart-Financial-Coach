"""Goal-forecast states, kept in their own file like categorization predictions and FR-7 flags
(FR-11 and FR-12 design, §7). One file holds one model version's states for one dataset.

    meta    model_version, data_spec_name, data_spec_hash, as_of (the dataset's last day),
            created_at, users
    model   what serving needs of the promoted model to forecast from a state: its registry name,
            version and params, the typical total allocation and the spread it fitted
    states  per user: the net-savings forecast state at `as_of` (level, month-of-year profile,
            residuals, spread, months of history, the last month, exponential smoothing's own
            state or null, the persona mixture's weights and profiles or null, and the trend)

It's JSON, a few KB per user, and never a pickle: the web image loads it without trusting a model
file or reaching the network (Web App UI, "Demo build"). The file appears only when a run
completes, readable by everyone, as the flag file does.

    write_forecasts("build/demo/forecasts.json", meta, model, states)
    load_forecasts("build/demo/forecasts.json")  # -> Forecasts
"""

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from smart_financial_coach.intelligence.forecasting.paths import ForecastState

FORMAT = 2  # 2: the persona mixture and the trend (round 5)


@dataclass(frozen=True)
class Forecasts:
    meta: dict[str, str]
    model: dict[str, Any]
    states: dict[str, ForecastState]


def state_to_json(state: ForecastState) -> dict[str, Any]:
    return {
        "level": state.level,
        "seasonal": list(state.seasonal),
        "residuals": list(state.residuals),
        "spread": state.spread,
        "months": state.months,
        "end": str(state.end),
        "ets": None if state.ets is None else [*state.ets[:3], list(state.ets[3])],
        "mixture": (
            None if state.mixture is None else [[w, list(dev)] for w, dev in state.mixture]
        ),
        "slope": state.slope,
        "lead": state.lead,
        "damping": state.damping,
    }


def state_from_json(data: Mapping[str, Any]) -> ForecastState:
    ets, mixture = data["ets"], data["mixture"]
    return ForecastState(
        level=float(data["level"]),
        seasonal=tuple(float(v) for v in data["seasonal"]),
        residuals=tuple(float(v) for v in data["residuals"]),
        spread=float(data["spread"]),
        months=int(data["months"]),
        end=pd.Period(str(data["end"]), freq="M"),
        ets=(
            None
            if ets is None
            else (float(ets[0]), float(ets[1]), float(ets[2]), tuple(float(v) for v in ets[3]))
        ),
        mixture=(
            None
            if mixture is None
            else tuple((float(w), tuple(float(v) for v in dev)) for w, dev in mixture)
        ),
        slope=float(data["slope"]),
        lead=float(data["lead"]),
        damping=float(data["damping"]),
    )


def write_forecasts(
    path: str | Path,
    meta: Mapping[str, str],
    model: Mapping[str, Any],
    states: Mapping[str, ForecastState],
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite to replace it")
    body = {
        "format": FORMAT,
        "meta": {**meta, "users": str(len(states))},
        "model": dict(model),
        "states": {u: state_to_json(s) for u, s in sorted(states.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f"{path.name}.", suffix=".part", delete=False
    ) as f:
        partial = Path(f.name)
        json.dump(body, f, sort_keys=True)
    try:
        # Readable by everyone, like any data file: serving may run as another user (#26)
        partial.chmod(0o644)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_forecasts(path: str | Path) -> Forecasts:
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("format") != FORMAT:
        raise ValueError(f"{path} is forecasts format {body.get('format')}, not {FORMAT}")
    return Forecasts(
        meta={str(k): str(v) for k, v in body["meta"].items()},
        model=dict(body["model"]),
        states={str(u): state_from_json(s) for u, s in body["states"].items()},
    )
