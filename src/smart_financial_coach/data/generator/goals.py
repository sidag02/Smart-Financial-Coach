"""Stage 9: savings goals, generated after the ledger so their outcomes are known.

About half the goals (spec-set) have a target date inside the history, so `truth_goals.met` is
realized; the rest end after the history, as a live product would see them.

`current_balance` is reported as of `as_of_date`, the goal's backtest origin. For goals that end
inside the history that date is 3-12 months (spec-set) before the target, so the goals row itself
doesn't reveal `met`. The transactions table still covers the whole history: goal forecasting must
only use transactions with `ts <= as_of_date`, which the evaluation harness enforces.
"""

from datetime import date

import numpy as np
import pandas as pd

from smart_financial_coach.data.generator.population import User
from smart_financial_coach.data.generator.spec import GoalsSpec
from smart_financial_coach.data.generator.timeline import Timeline

# Target as a multiple of the reference amount (saved by target date, or projected), per class.
OUTCOME_RANGES = {"on_track": (0.6, 0.85), "borderline": (0.95, 1.05), "off_track": (1.3, 1.8)}


def _month_end_after(end: date, months_after: int) -> str:
    month = np.datetime64(end, "M") + months_after
    return str((month + 1).astype("datetime64[D]") - 1)


def _saved(net: np.ndarray, share: float, first: int, last: int) -> float:
    """Running share of monthly net savings; savings toward a goal can't go negative."""
    saved = 0.0
    for m in range(first, last + 1):
        saved = max(0.0, saved + share * float(net[m]))
    return saved


def generate_goals(
    user: User, tl: Timeline, txns: pd.DataFrame, spec: GoalsSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    net = np.bincount(
        tl.month_idx[txns["day"].to_numpy(dtype=np.int64)],
        weights=txns["amount"].to_numpy(),
        minlength=tl.n_months,
    )
    return sample_goals(
        net,
        tl,
        [t.name for t in user.persona.goals],
        [t.weight for t in user.persona.goals],
        spec,
        user.rng("goals"),
        user_id=user.user_id,
        goal_prefix=f"g_{user.user_id[2:]}",
    )


def sample_goals(
    net: np.ndarray,
    tl: Timeline,
    names: list[str],
    weights: list[float],
    spec: GoalsSpec,
    rng: np.random.Generator,
    *,
    user_id: str,
    goal_prefix: str,
    targets_from: str = "realized",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One user's goals and their truth, from the user's monthly net savings (`net`, one value
    per month of `tl`). A pure function of its arguments: stage 9 calls it once per user with the
    user's own stream; FR-11's evaluation calls it again with other streams to draw more labeled
    goals without changing the dataset (FR-11 and FR-12 design, §6).

    `targets_from` is what a goal's target is a multiple of, for goals that end inside the history:
    - "realized" (stage 9): the balance actually reached by the target date. The outcome is then
      planted (an `on_track` goal is always met), and a target carries the future with it;
    - "projection" (FR-11's evaluation goals since round 5): the balance projected at `as_of`
      from the 12 months before it, as for goals that end after the history. Whether it's met is
      then up to the months that follow, as for a real person's goal.
    Neither changes what the random generator draws, so the goals are otherwise the same."""
    if targets_from not in ("realized", "projection"):
        raise ValueError(f"targets_from must be realized or projection, not {targets_from!r}")
    k = min(int(rng.integers(spec.per_user[0], spec.per_user[1] + 1)), len(names))
    w = np.array(weights)
    picked = rng.choice(len(names), size=k, replace=False, p=w / w.sum())
    total_share = float(rng.uniform(*spec.allocation_share))
    shares = total_share * (rng.dirichlet(np.full(k, 2.0)) if k > 1 else np.ones(1))
    classes = list(spec.outcome_mix)
    class_p = np.array([spec.outcome_mix[c] for c in classes])
    last = tl.n_months - 1
    h_low, h_high = spec.forecast_horizon_months

    goals: list[dict[str, object]] = []
    truth: list[dict[str, object]] = []
    for j, (type_idx, share) in enumerate(zip(picked, shares, strict=True)):
        # Room for: 2 warm-up months, created < as_of, and the shortest horizon
        inside = last - h_low > 3 and rng.random() < spec.target_inside_history_share
        if inside:
            # created < as_of < target, all inside the history; target is h_low..h_high months out
            created_m = int(rng.integers(2, last - h_low))
            as_of_m = min(created_m + int(rng.integers(1, 7)), last - h_low)
            target_m = as_of_m + int(rng.integers(h_low, min(h_high, last - as_of_m) + 1))
            target_date = str(tl.date_of(int(tl.month_last[target_m])))
            current = _saved(net, share, created_m, as_of_m)
            realized = _saved(net, share, created_m, target_m)
            final: float | None = realized
            reference = realized
            if targets_from == "projection":
                recent = net[max(0, as_of_m - 11) : as_of_m + 1]
                reference = current + share * max(0.0, float(recent.mean())) * (target_m - as_of_m)
        else:
            created_m = int(rng.integers(max(0, tl.n_months - 12), last))
            as_of_m = last
            ahead = int(rng.integers(h_low, h_high + 1))
            target_date = _month_end_after(tl.end, ahead)
            current = _saved(net, share, created_m, last)
            final = None
            reference = current + share * max(0.0, float(net[-12:].mean())) * ahead
        outcome = str(rng.choice(classes, p=class_p / class_p.sum()))
        if reference <= 100:
            outcome = "off_track"
            target = float(rng.uniform(1000, 5000))
        else:
            target = reference * float(rng.uniform(*OUTCOME_RANGES[outcome]))
        target = max(100.0, round(target / 50) * 50)
        created_day = min(
            int(tl.month_first[created_m]) + int(rng.integers(0, 28)), int(tl.month_last[created_m])
        )
        goal_id = f"{goal_prefix}_{j + 1}"
        goals.append(
            {
                "goal_id": goal_id,
                "user_id": user_id,
                "name": names[int(type_idx)],
                "target_amount": target,
                "created_date": str(tl.date_of(created_day)),
                "target_date": target_date,
                "as_of_date": str(tl.date_of(int(tl.month_last[as_of_m]))),
                "current_balance": round(current, 2),
            }
        )
        truth.append(
            {
                "goal_id": goal_id,
                "outcome_class": outcome,
                "met": None if final is None else int(final >= target),
            }
        )
    return pd.DataFrame(goals), pd.DataFrame(truth)
