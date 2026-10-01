"""Stage 6: labeled events. Spending spikes (period-level), unusual charges, and refunds.

Spikes are planned before discretionary spending and applied as rate multipliers, so calibration
and balances include them. Unusual charges and refunds are added after discretionary is final.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User, merchant_weights
from smart_financial_coach.data.generator.spec import EventsSpec, season_vector
from smart_financial_coach.data.generator.timeline import Timeline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Spike:
    category: str
    granularity: str
    first_day: int
    last_day: int  # inclusive
    multiplier: float


def plan_spikes(
    user: User, tl: Timeline, events: EventsSpec
) -> tuple[dict[str, FloatArray], list[Spike]]:
    """Choose spike periods that never coincide with a seasonal peak of the same category."""
    spec = events.spending_spike
    rng = user.rng("spikes")
    n = int(rng.poisson(spec.rate_per_user_year * tl.years))
    rates: dict[str, float] = {}
    peaks: dict[str, FloatArray] = {}
    for stream in user.streams:
        cat = stream.spec.category
        rates[cat] = rates.get(cat, 0.0) + stream.weekly_rate
        peaks[cat] = np.maximum(
            peaks.get(cat, np.zeros(13)), season_vector(stream.spec.seasonality)
        )
    eligible = sorted(c for c, r in rates.items() if r >= spec.min_weekly_rate)
    multipliers: dict[str, FloatArray] = {}
    spikes: list[Spike] = []
    if not eligible or not spec.granularity:
        return multipliers, spikes
    full_months = [
        m
        for m in range(spec.baseline_months, tl.n_months)
        if tl.dom[tl.month_first[m]] == 1
        and (tl.month_last[m] + 1 == tl.n_days or tl.dom[tl.month_last[m] + 1] == 1)
    ]
    for _ in range(n):
        category = str(rng.choice(eligible))
        granularity = str(rng.choice(spec.granularity))
        quiet = [
            m
            for m in full_months
            if peaks[category][tl.moy[tl.month_first[m]]] <= spec.peak_threshold
        ]
        taken = multipliers.get(category, np.ones(tl.n_days))
        if granularity == "month":
            periods = [(int(tl.month_first[m]), int(tl.month_last[m])) for m in quiet]
        else:
            mondays = [
                d
                for m in quiet
                for d in range(int(tl.month_first[m]), int(tl.month_last[m]) - 5)
                if tl.dow[d] == 0
            ]
            periods = [(d, d + 6) for d in mondays]
        periods = [(a, b) for a, b in periods if (taken[max(0, a - 7) : b + 8] == 1.0).all()]
        if not periods:
            continue
        first, last = periods[int(rng.integers(len(periods)))]
        mult = float(rng.uniform(*spec.multiplier))
        taken = taken.copy()
        taken[first : last + 1] = mult
        multipliers[category] = taken
        spikes.append(Spike(category, granularity, first, last, round(mult, 3)))
    return multipliers, spikes


def generate_unusual_charges(
    user: User, tl: Timeline, catalog: Catalog, ledger: Ledger, events: EventsSpec, bias: float
) -> None:
    spec = events.unusual_charge
    rng = user.rng("unusual")
    n = int(rng.poisson(spec.rate_per_user_year * tl.years))
    if n == 0 or not spec.kinds or tl.n_days <= spec.baseline_days:
        return
    disc = ledger.frame(("discretionary",))
    disc = disc[disc["day"] >= spec.baseline_days]
    seen = set(ledger.frame()["merchant_id"])
    favorites = [(s, m) for s in user.streams for m in s.merchants]
    novel = [
        m
        for c in spec.new_merchant_categories
        for m in catalog.select(c, [], include_holdout=user.include_holdout).index
        if m not in seen
    ]
    for _ in range(n):
        kind = str(rng.choice(spec.kinds))
        day = int(rng.integers(spec.baseline_days, tl.n_days))
        if kind == "duplicate" and len(disc):
            orig = disc.iloc[int(rng.integers(len(disc)))]
            ledger.add(
                "unusual_charge",
                day=[orig["day"]],
                minute=[min(int(orig["minute"]) + int(rng.integers(1, 90)), 1439)],
                amount=[orig["amount"]],
                merchant_id=[orig["merchant_id"]],
                category=[orig["category"]],
                channel=[orig["channel"]],
                anomaly_kind=kind,
                copy_of=[orig["transaction_id"]],
            )
            continue
        if kind == "amount_outlier" and favorites:
            stream, merchant = favorites[int(rng.integers(len(favorites)))]
            base = catalog.price(merchant) * stream.amount_scale
            amount = base * float(rng.uniform(*spec.outlier_multiplier))
        elif kind == "new_merchant_large" and novel:
            p = merchant_weights(catalog, novel, user.include_holdout, bias)
            merchant = str(rng.choice(np.array(novel), p=p))
            base = catalog.price(merchant)
            amount = max(
                spec.new_merchant_min_amount, base * float(rng.uniform(*spec.outlier_multiplier))
            )
        else:
            continue
        ids = np.array([merchant])
        ledger.add(
            "unusual_charge",
            day=[day],
            minute=catalog.sample_minutes(ids, rng),
            amount=[-amount],
            merchant_id=ids,
            category=catalog.sample_categories(ids, rng),
            channel=catalog.sample_channels(ids, rng),
            anomaly_kind=kind,
        )


def generate_refunds(user: User, tl: Timeline, ledger: Ledger, events: EventsSpec) -> None:
    spec = events.refunds
    rng = user.rng("refunds")
    purchases = ledger.frame(("discretionary", "one_off"))
    purchases = purchases[purchases["category"].isin(spec.categories)]
    if purchases.empty or spec.share <= 0:
        return
    chosen = purchases[rng.random(len(purchases)) < spec.share]
    lag = rng.integers(spec.lag_days[0], spec.lag_days[1] + 1, size=len(chosen))
    days = chosen["day"].to_numpy() + lag
    keep = days < tl.n_days
    chosen, days = chosen[keep], days[keep]
    ledger.add(
        "refund",
        day=days,
        minute=rng.integers(0, 24 * 60, size=len(days)),
        amount=-chosen["amount"].to_numpy(),
        merchant_id=chosen["merchant_id"].to_numpy(),
        category=chosen["category"].to_numpy(),
        channel=chosen["channel"].to_numpy(),
        copy_of=chosen["transaction_id"].to_numpy(),
    )


def spike_rows(user: User, tl: Timeline, spikes: list[Spike]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [user.user_id] * len(spikes),
            "granularity": [s.granularity for s in spikes],
            "period_start": [str(tl.date_of(s.first_day)) for s in spikes],
            "category": [s.category for s in spikes],
            "multiplier": [s.multiplier for s in spikes],
        }
    )
