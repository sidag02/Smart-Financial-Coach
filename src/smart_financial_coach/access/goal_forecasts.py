"""Goal forecasts for the tools (FR-11 and FR-12 design, §3, §7 and "Tools").

A user's running goals (active or reached), and a draft when there is one, are forecast together
as one goal set: they share one future, and a `typical` share divides the typical total by every
running goal, the draft included. So `check_goal`'s numbers for a draft are the numbers
`forecast_goal` gives the same goal once it's saved.

A goal entered by hand (`origin` "yours") gets its own share once two of its saved entries are 3+
months apart (`your_entries`): the first entry comes from FR-10's event log, the latest is the
goal's saved amount.

    rows = goal_rows(user_id, goals, today, history, first_entries(revisions))
    forecaster.forecast(rows)
"""

import math
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

import pandas as pd

from smart_financial_coach.access.goals import Goal, Revision
from smart_financial_coach.intelligence.forecasting.contract import INPUT_COLUMNS

SHORT_HISTORY = 6  # fewer full months: "a rough guide" (PRD risk: short histories)
MONTHLY_POINTS = 24  # at most this many months in `monthly`; longer goals are sampled evenly
FITS = {"on_track": "within_reach", "either_way": "either_way", "off_track": "stretch"}


def first_entries(revisions: Iterable[Revision]) -> dict[str, tuple[int, date]]:
    """Each goal's earliest live saved entry in the subject's event log: (cents, day)."""
    out: dict[str, tuple[int, date]] = {}
    for r in sorted(revisions, key=lambda r: r.seq):
        if r.undone or r.op == "archive" or r.goal_id in out:
            continue
        out[r.goal_id] = (r.saved_cents, date.fromisoformat(r.saved_as_of))
    return out


def goal_rows(
    user_id: str,
    goals: list[Goal],
    today: date,
    history: str,
    entries: Mapping[str, tuple[int, date]],
) -> pd.DataFrame:
    """The forecast's input rows for one goal set at `today`. `history` is the user's monthly
    net savings up to `today` (the contract's `history_json`)."""
    rows = []
    for g in goals:
        first = entries.get(g.goal_id)
        rows.append(
            {
                "example_id": g.goal_id,
                "goal_id": g.goal_id,
                "user_id": user_id,
                "goal_set": user_id,
                "persona": "",  # only fitting a state reads it; serving's states are stored
                "as_of_date": today.isoformat(),
                "created_date": g.created_date.isoformat(),
                "target_amount": g.target_cents / 100,
                "target_date": g.target_date.isoformat(),
                "saved": g.saved_cents / 100,
                "saved_as_of": g.saved_as_of.isoformat(),
                "first_saved": first[0] / 100 if first else math.nan,
                "first_saved_as_of": first[1].isoformat() if first else None,
                "origin": g.origin,
                "active_goals": len(goals),
                # Fit-only (review on #52): a live user's goal set is the goals running now
                "set_goals": len(goals),
                "history_json": history,
            }
        )
    return pd.DataFrame(rows, columns=list(INPUT_COLUMNS))


def forecast_fields(
    out: Mapping[str, Any], months_of_history: int, money: Any, *, baseline: bool = False
) -> dict[str, Any]:
    """One goal's forecast as the tools return it. A reached goal shows no probability, range,
    gap or top-up (owner decision 9 on #50), only whether it might be drawn down. The baseline
    (`method` "simple_projection": naive pace, while no model is promoted) has no probability
    or range either: it projects the pace so far (owner decision 10 on #54)."""
    reached = out["status"] == "reached"
    extra = out["extra_per_month"]
    return {
        "method": "simple_projection" if baseline else "simulation",
        "status": out["status"],
        "p_goal_met": None if reached or baseline else round(float(out["p_goal_met"]), 3),
        "projected_balance": money(out["projected_balance"]),
        "range": (
            None
            if reached or baseline
            else {"low": money(out["range_lo"]), "high": money(out["range_hi"]), "chance": 0.8}
        ),
        "gap": None if reached else money(out["gap"]),
        "extra_per_month": None if reached or extra is None or pd.isna(extra) else money(extra),
        "share_source": out["share_source"],
        "months_of_history": months_of_history,
        "short_history": months_of_history < SHORT_HISTORY,
        "may_draw_down": bool(out["may_draw_down"]) if reached else None,
        "model_version": out["model_version"],
    }


def thin(monthly: list[dict[str, Any]], points: int = MONTHLY_POINTS) -> list[dict[str, Any]]:
    """At most `points` months, evenly spaced and always ending at the target month."""
    if len(monthly) <= points:
        return monthly
    step = math.ceil(len(monthly) / points)
    last = len(monthly) - 1
    return [monthly[i] for i in range(last % step, len(monthly), step)]
