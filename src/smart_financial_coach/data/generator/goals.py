"""Stage 9: savings goals, generated after the ledger so their outcomes are known.

About half the goals (spec-set) have a target date inside the history, so `truth_goals.met` is
realized; the rest end after the history, as a live product would see them.
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


def generate_goals(
    user: User, tl: Timeline, txns: pd.DataFrame, spec: GoalsSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = user.rng("goals")
    net = np.bincount(
        tl.month_idx[txns["day"].to_numpy(dtype=np.int64)],
        weights=txns["amount"].to_numpy(),
        minlength=tl.n_months,
    )
    types = user.persona.goals
    k = min(int(rng.integers(spec.per_user[0], spec.per_user[1] + 1)), len(types))
    weights = np.array([t.weight for t in types])
    picked = rng.choice(len(types), size=k, replace=False, p=weights / weights.sum())
    total_share = float(rng.uniform(*spec.allocation_share))
    shares = total_share * (rng.dirichlet(np.full(k, 2.0)) if k > 1 else np.ones(1))
    classes = list(spec.outcome_mix)
    class_p = np.array([spec.outcome_mix[c] for c in classes])
    last = tl.n_months - 1

    goals: list[dict[str, object]] = []
    truth: list[dict[str, object]] = []
    for j, (type_idx, share) in enumerate(zip(picked, shares, strict=True)):
        inside = tl.n_months >= 12 and rng.random() < spec.target_inside_history_share
        if inside:
            created_m = int(rng.integers(2, tl.n_months - 7))
            target_m = created_m + int(rng.integers(6, min(18, last - created_m) + 1))
            target_date = str(tl.date_of(int(tl.month_last[target_m])))
        else:
            created_m = int(rng.integers(max(0, tl.n_months - 12), last))
            ahead = int(rng.integers(3, 13))
            target_m = last + ahead
            target_date = _month_end_after(tl.end, ahead)
        saved = 0.0
        for m in range(created_m, min(target_m, last) + 1):
            saved = max(0.0, saved + share * net[m])  # savings toward a goal can't go negative
        if inside:
            reference = saved
        else:
            recent = max(0.0, float(net[-12:].mean()))
            reference = saved + share * recent * (target_m - last)
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
        goal_id = f"g_{user.user_id[2:]}_{j + 1}"
        goals.append(
            {
                "goal_id": goal_id,
                "user_id": user.user_id,
                "name": types[int(type_idx)].name,
                "target_amount": target,
                "created_date": str(tl.date_of(created_day)),
                "target_date": target_date,
                "current_balance": round(saved, 2),
            }
        )
        truth.append(
            {
                "goal_id": goal_id,
                "outcome_class": outcome,
                "met": int(saved >= target) if inside else None,
            }
        )
    return pd.DataFrame(goals), pd.DataFrame(truth)
