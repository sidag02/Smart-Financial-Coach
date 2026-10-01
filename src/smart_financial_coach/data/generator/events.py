"""Stage 6: labeled events. Spending spikes (period-level), unusual charges, and refunds.

Spikes are planned before discretionary spending. Their extra purchases are drawn after calibration
as a separate process (`spending.generate_spike_extras`), so normal purchases don't depend on them
and each spike's realized effect is known exactly. Unusual charges and refunds are added after
discretionary spending is final, and draw only from normal purchases.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.expected import Expected
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User, merchant_weights
from smart_financial_coach.data.generator.spec import EventsSpec, season_vector
from smart_financial_coach.data.generator.spending import SPIKE_EXTRA
from smart_financial_coach.data.generator.timeline import Timeline

FloatArray = npt.NDArray[np.float64]


SPIKE_COLUMNS = [
    "spike_id",
    "user_id",
    "granularity",
    "period_start",
    "period_end",
    "category",
    "multiplier",
    "expected_count",
    "expected_spend",
    "base_spend",
    "extra_spend",
    "tier",
]


@dataclass(frozen=True)
class Spike:
    category: str
    granularity: str
    first_day: int
    last_day: int  # inclusive
    multiplier: float


def spike_categories(user: User, events: EventsSpec) -> list[str]:
    """Categories the user buys in often enough to be spiked (rate before calibration)."""
    rates: dict[str, float] = {}
    for stream in user.streams:
        rates[stream.spec.category] = rates.get(stream.spec.category, 0.0) + stream.weekly_rate
    return sorted(c for c, r in rates.items() if r >= events.spending_spike.min_weekly_rate)


def plan_spikes(
    user: User, tl: Timeline, events: EventsSpec
) -> tuple[dict[str, FloatArray], list[Spike]]:
    """Choose spike periods that never coincide with a seasonal peak of the same category."""
    spec = events.spending_spike
    rng = user.rng("spikes")
    n = int(rng.poisson(spec.rate_per_user_year * tl.years))
    peaks: dict[str, FloatArray] = {}
    for stream in user.streams:
        cat = stream.spec.category
        peaks[cat] = np.maximum(
            peaks.get(cat, np.zeros(13)), season_vector(stream.spec.seasonality)
        )
    eligible = spike_categories(user, events)
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
) -> int:
    """Add the user's unusual charges; returns how many could not be produced by any kind."""
    spec = events.unusual_charge
    rng = user.rng("unusual")
    n = int(rng.poisson(spec.rate_per_user_year * tl.years))
    if n == 0 or not spec.kinds or tl.n_days <= spec.baseline_days:
        return 0
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

    def duplicate() -> bool:
        if disc.empty:
            return False
        orig = disc.iloc[int(rng.integers(len(disc)))]
        ledger.add(
            "unusual_charge",
            day=[orig["day"]],
            minute=[min(int(orig["minute"]) + int(rng.integers(1, 90)), 1439)],
            amount=[orig["amount"]],
            merchant_id=[orig["merchant_id"]],
            category=[orig["category"]],
            channel=[orig["channel"]],
            anomaly_kind="duplicate",
            copy_of=[orig["transaction_id"]],
        )
        return True

    def charge(kind: str, merchant: str, amount: float) -> None:
        ids = np.array([merchant])
        ledger.add(
            "unusual_charge",
            day=[int(rng.integers(spec.baseline_days, tl.n_days))],
            minute=catalog.sample_minutes(ids, rng),
            amount=[-amount],
            merchant_id=ids,
            category=catalog.sample_categories(ids, rng),
            channel=catalog.sample_channels(ids, rng),
            anomaly_kind=kind,
        )

    def amount_outlier() -> bool:
        if not favorites:
            return False
        stream, merchant = favorites[int(rng.integers(len(favorites)))]
        base = catalog.price(merchant) * stream.amount_scale
        charge("amount_outlier", merchant, base * float(rng.uniform(*spec.outlier_multiplier)))
        return True

    def new_merchant_large() -> bool:
        if not novel:
            return False
        p = merchant_weights(catalog, novel, user.include_holdout, bias)
        merchant = str(rng.choice(np.array(novel), p=p))
        novel.remove(merchant)  # a second charge there would no longer be at a new merchant
        base = catalog.price(merchant) * float(rng.uniform(*spec.outlier_multiplier))
        charge("new_merchant_large", merchant, max(spec.new_merchant_min_amount, base))
        return True

    makers = {
        "duplicate": duplicate,
        "amount_outlier": amount_outlier,
        "new_merchant_large": new_merchant_large,
    }
    dropped = 0
    for _ in range(n):
        first = str(rng.choice(spec.kinds))
        # Fall back to the other kinds when the drawn one is impossible for this user
        others = [str(k) for k in rng.permutation(spec.kinds) if k != first]
        if not any(makers[kind]() for kind in [first, *others]):
            dropped += 1
    return dropped


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


def spike_rows(
    user: User,
    tl: Timeline,
    spikes: list[Spike],
    expected: Expected,
    txns: pd.DataFrame,
    weak_lift: float,
) -> pd.DataFrame:
    """`truth_periods` rows: each planted spike with its expected and realized spend.

    `txns` is the user's ledger with internal process names, so spike extras can be told apart.
    A period's spend is the net outflow of every transaction in the category.
    """
    rows = []
    for k, s in enumerate(spikes, start=1):
        in_period = (
            (txns["category"] == s.category)
            & (txns["day"] >= s.first_day)
            & (txns["day"] <= s.last_day)
        )
        outflow = -txns.loc[in_period, "amount"]
        extra = float(outflow[txns.loc[in_period, "process"] == SPIKE_EXTRA].sum())
        base = float(outflow.sum()) - extra
        expected_count, expected_spend = expected.total(s.category, s.first_day, s.last_day)
        rows.append(
            {
                "spike_id": f"s_{user.user_id[2:]}_{k}",
                "user_id": user.user_id,
                "granularity": s.granularity,
                "period_start": str(tl.date_of(s.first_day)),
                "period_end": str(tl.date_of(s.last_day)),
                "category": s.category,
                "multiplier": s.multiplier,
                "expected_count": round(expected_count, 4),
                "expected_spend": round(expected_spend, 2),
                "base_spend": round(base, 2),
                "extra_spend": round(extra, 2),
                "tier": "weak" if base + extra < expected_spend * weak_lift else "clear",
            }
        )
    return pd.DataFrame(rows, columns=SPIKE_COLUMNS)


def unusual_tiers(txns: pd.DataFrame) -> pd.Series:
    """`clear` / `weak` per unusual charge; None for other transactions (one user's ledger).

    An amount outlier is weak when it doesn't exceed the user's largest normal charge at the same
    merchant: high-variance marketplaces sometimes reach the planted multiple on their own.
    """
    normal = txns[txns["anomaly_kind"].isna() & (txns["amount"] < 0)]
    largest = (-normal["amount"]).groupby(normal["merchant_id"]).max()
    inside = -txns["amount"] <= txns["merchant_id"].map(largest).fillna(0.0)
    weak = (txns["anomaly_kind"] == "amount_outlier") & inside
    tier = pd.Series(np.where(weak, "weak", "clear"), index=txns.index, dtype=object)
    return tier.where(txns["anomaly_kind"].notna(), None)
