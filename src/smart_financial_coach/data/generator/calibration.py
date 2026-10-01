"""Stage 7: bring each user's long-run savings rate near their target, and set a safe balance.

Calibration scales discretionary purchase *rates* and regenerates from the same seed. Amounts are
never rescaled, so prices stay within each merchant's range.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User
from smart_financial_coach.data.generator.spending import generate_discretionary
from smart_financial_coach.data.generator.timeline import Timeline

SCALE_BOUNDS = (0.25, 3.0)


def calibrate_discretionary(
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    *,
    daily_mult: npt.NDArray[np.float64],
) -> float:
    """Generate normal discretionary spending at the rate scale that meets the savings target.

    Spikes are excluded, so turning them off leaves every normal purchase unchanged.
    """
    generate_discretionary(user, tl, catalog, ledger, rate_scale=1.0, daily_mult=daily_mult)
    income = float(ledger.frame(("income",))["amount"].sum())
    fixed = -float(ledger.frame(("recurring", "one_off"))["amount"].sum())
    discretionary = -float(ledger.frame(("discretionary",))["amount"].sum())
    if income <= 0 or discretionary <= 0:
        return 1.0
    scale = (income * (1 - user.savings_target) - fixed) / discretionary
    scale = float(np.clip(scale, *SCALE_BOUNDS))
    generate_discretionary(user, tl, catalog, ledger, rate_scale=scale, daily_mult=daily_mult)
    return scale


def starting_balance(user: User, txns: pd.DataFrame) -> float:
    """A sampled cushion, raised if needed so the running balance never goes below zero."""
    running = txns["amount"].cumsum()
    needed = max(0.0, -float(running.min())) + 100.0 if len(running) else 0.0
    return round(max(user.cushion_months * user.monthly_net, needed), 2)
