"""The goal-forecasting service contract (FR-11 and FR-12 design, §1).

in:  one row per goal to forecast, with what a forecast may use: the goal (target, dates, the
     saved amount and when it was recorded, the first saved entry for a goal the user keeps up),
     its goal set (the user's goals that share their savings) and the user's monthly net savings
     up to the forecast's `as_of_date`, as JSON (`history_json`)
out: example_id, status, p_goal_met, projected_balance, range_lo, range_hi, gap,
     extra_per_month, share, share_source, net_next_6, model_version

`range_lo`/`range_hi` are the 10th and 90th percentiles of the balance by the target date: an 80%
interval, fixed here (Web App UI, gap 4). `p_goal_met` is always set, so a reached goal can still
be scored; the tools hide it for a reached goal (owner decision on #50). `extra_per_month` is the
monthly deposit that would bring a goal to the on-track band, or null when it's already there.
A model without a share (a baseline) leaves `share` and `share_source` null. `net_next_6` is the
model's point forecast of the user's total net savings over the next 6 months, scored for the
PRD's RMSE metric (owner decision 1 on #50); null for a model without a net-savings forecast.
"""

import json
from datetime import date

import numpy as np
import pandas as pd

from smart_financial_coach.intelligence.models.contract import Contract
from smart_financial_coach.intelligence.service import register_service

SERVICE = "goal_forecasting"
INPUT_COLUMNS = (
    "example_id",
    "goal_id",
    "user_id",
    "goal_set",  # the goals that share the user's savings with this one
    "persona",
    "as_of_date",  # the forecast's "today"; `history_json` ends at its month
    "created_date",
    "target_amount",
    "target_date",  # a month end
    "saved",
    "saved_as_of",
    "first_saved",  # the first saved entry of a goal the user keeps up; NaN without one
    "first_saved_as_of",  # its date; None without one
    "origin",  # "existing" (generated: a track record from $0) or "yours" (entered by hand)
    "active_goals",  # goals in the set still running at `as_of_date`, this one included
    "history_json",
)
OUTPUT_COLUMNS = (
    "example_id",
    "status",
    "p_goal_met",
    "projected_balance",
    "range_lo",
    "range_hi",
    "gap",
    "extra_per_month",
    "share",
    "share_source",
    "net_next_6",
    "model_version",
)
STATUSES = ("on_track", "either_way", "off_track", "reached")
SHARE_SOURCES = ("track_record", "your_entries", "typical")
ORIGINS = ("existing", "yours")
# The status bands, fixed (owner decision 5 on #50): nothing tunes them
ON_TRACK = 0.7
OFF_TRACK = 0.3
INTERVAL = (0.1, 0.9)  # the 80% range's percentiles


def history_json(net: pd.Series) -> str:
    """A user's monthly net savings, `pd.Period`-indexed, as the input's `history_json`."""
    if len(net) == 0:
        return json.dumps({"start": None, "net": []})
    return json.dumps({"start": str(net.index[0]), "net": [round(float(v), 2) for v in net]})


def parse_history(text: str) -> pd.Series:
    data = json.loads(text)
    if data["start"] is None:
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
    start = pd.Period(data["start"], freq="M")
    return pd.Series(
        data["net"], index=pd.period_range(start, periods=len(data["net"]), freq="M"), dtype=float
    )


def months_left(as_of: date, target: date) -> int:
    """Month ends after `as_of` up to the target's (FR-10's counting)."""
    a, t = pd.Period(as_of, freq="M"), pd.Period(target, freq="M")
    ends = (t - a).n
    if as_of != a.end_time.date():
        ends += 1
    return max(0, ends)


def status_for(p: float, reached: bool) -> str:
    if reached:
        return "reached"
    if p >= ON_TRACK:
        return "on_track"
    return "off_track" if p < OFF_TRACK else "either_way"


def _check(out: pd.DataFrame) -> list[str]:
    errors = []
    if not out["status"].isin(STATUSES).all():
        errors.append(f"status outside {STATUSES}")
    p = out["p_goal_met"].astype(float)
    if ((p < 0) | (p > 1)).any():
        errors.append("p_goal_met outside 0-1")
    lo, mid, hi = (out[c].astype(float) for c in ("range_lo", "projected_balance", "range_hi"))
    if ((lo > mid + 1e-6) | (mid > hi + 1e-6)).any():
        errors.append("range_lo <= projected_balance <= range_hi doesn't hold")
    if (lo < 0).any():
        errors.append("a balance below zero")
    if (out["gap"].astype(float) < 0).any():
        errors.append("a negative gap")
    extra = out["extra_per_month"].astype(float)
    if (extra.dropna() < 0).any():
        errors.append("a negative extra_per_month")
    share = out["share"].astype(float)
    if ((share < 0) | (share > 1)).any():
        errors.append("share outside 0-1")
    source = out["share_source"]
    if not source.dropna().isin(SHARE_SOURCES).all():
        errors.append(f"share_source outside {SHARE_SOURCES}")
    if (share.isna() != source.isna()).any():
        errors.append("share and share_source must be set together")
    reached = out["status"] == "reached"
    if extra[reached].notna().any():
        errors.append("a reached goal with an extra_per_month")
    banded = out.loc[~reached, "status"]
    expected = [status_for(float(v), False) for v in p[~reached]]
    if list(banded) != expected:
        errors.append("status doesn't match p_goal_met's band")
    return errors


CONTRACT = Contract(
    service=SERVICE,
    id_column="example_id",
    columns=OUTPUT_COLUMNS,
    check=_check,
    nullable=("extra_per_month", "share", "share_source", "net_next_6"),
)
register_service(CONTRACT)


def output_frame(rows: list[dict[str, object]], version: str) -> pd.DataFrame:
    """Model outputs in the contract's column order, with the model's version."""
    frame = pd.DataFrame(rows, columns=[c for c in OUTPUT_COLUMNS if c != "model_version"])
    frame["model_version"] = version
    for c in (
        "p_goal_met",
        "projected_balance",
        "range_lo",
        "range_hi",
        "gap",
        "share",
        "net_next_6",
    ):
        frame[c] = frame[c].astype(float)
    frame["extra_per_month"] = frame["extra_per_month"].astype(float)
    frame["share_source"] = (
        frame["share_source"].astype(object).where(frame["share_source"].notna(), None)
    )
    return frame


def to_date(value: object) -> date:
    return date.fromisoformat(str(value)[:10])


def round_up_dollars(value: float) -> float:
    return float(np.ceil(value - 1e-9))
