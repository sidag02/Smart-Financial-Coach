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
from smart_financial_coach.intelligence.forecasting.savings import run_balance

SHORT_HISTORY = 6  # fewer full months: "a rough guide" (PRD risk: short histories)
MONTHLY_POINTS = 24  # at most this many months in `monthly`; longer goals are sampled evenly
HISTORY_POINTS = 12  # at most this many past months in `history`, sampled the same way
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
    out: Mapping[str, Any],
    months_of_history: int,
    money: Any,
    *,
    target: float,
    baseline: bool = False,
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
        # How far the likely amount is past the target, so no one computes it (FR-14)
        "ahead": None if reached else money(max(0.0, float(out["projected_balance"]) - target)),
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


def saved_history(
    goal: Goal,
    share: float | None,
    share_source: str | None,
    net: pd.Series,
    first: tuple[int, date] | None,
    today: date,
) -> list[dict[str, Any]]:
    """A goal's estimated saved amount at the end of each past month, oldest first, under the
    share its forecast uses (overview feedback, Oct 5, 2026). Goals are notional, so this is the
    forecast's own account of how the goal got here, not a record of deposits:
    - `track_record`: from $0 at creation, `share` of each month's net savings since (§3);
    - `your_entries`: from the person's first entry, the same way up to their latest;
    - the simple projection (no share): the pace so far, a straight line from creation;
    - `typical`: no share of its own, so only the person's first entry, if they made one.

    `net` is monthly net savings (`parse_history`). Months before `today`'s only: today's
    amount is the goal's saved amount."""
    now = pd.Period(today, freq="M")
    saved = goal.saved_cents / 100
    points: list[tuple[pd.Period, float]] = []
    if share is not None and share_source == "track_record":
        since = net[net.index >= pd.Period(goal.created_date, freq="M")]
        points = list(zip(since.index, run_balance(0.0, share, since.to_numpy()), strict=True))
    elif share is not None and share_source == "your_entries" and first is not None:
        start, at = first[0] / 100, pd.Period(first[1], freq="M")
        between = net[(net.index > at) & (net.index <= pd.Period(goal.saved_as_of, freq="M"))]
        path = run_balance(start, share, between.to_numpy())
        points = [(at, start), *zip(between.index, path, strict=True)]
    elif share_source is None:
        created = pd.Period(goal.created_date, freq="M")
        months = max(1, (pd.Period(goal.saved_as_of, freq="M") - created).n + 1)
        points = [(created + i, saved * (i + 1) / months) for i in range(months)]
    elif first is not None:
        points = [(pd.Period(first[1], freq="M"), first[0] / 100)]
    past = [{"month": str(m), "saved": round(float(v), 2)} for m, v in points if m < now]
    return thin(past, HISTORY_POINTS)
