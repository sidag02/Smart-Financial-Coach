"""Expected spending per user, category and day for the user's normal behavior (FR-2).

Poisson processes (discretionary purchases and one-offs) contribute their exact expected count and
spend. Recurring bills contribute their realized amounts, since their schedule is fixed once drawn.
Refunds contribute their expected share of refundable spend, spread over the refund lag. Spikes and
unusual charges are excluded.

Expectations are per true category: an ambiguous merchant splits its count and spend across its
category mix, as `Catalog.sample_categories` does per transaction.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User, merchant_weights
from smart_financial_coach.data.generator.spec import EventsSpec, OneOffSpec, season_vector
from smart_financial_coach.data.generator.spending import stream_rate
from smart_financial_coach.data.generator.timeline import Timeline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Expected:
    """Per category, arrays over the calendar's days."""

    count: dict[str, FloatArray]  # expected Poisson-process purchases
    spend: dict[str, FloatArray]  # expected net outflow, all processes

    def total(self, category: str, first: int, last: int) -> tuple[float, float]:
        """Expected (count, spend) for days first..last inclusive."""
        count = self.count.get(category)
        spend = self.spend.get(category)
        return (
            float(count[first : last + 1].sum()) if count is not None else 0.0,
            float(spend[first : last + 1].sum()) if spend is not None else 0.0,
        )


def _category_mix(catalog: Catalog, merchant_id: str) -> tuple[list[str], FloatArray]:
    if (mix := catalog.categories.get(merchant_id)) is not None:
        return mix
    return [str(catalog.merchants.at[merchant_id, "category"])], np.ones(1)


def capped_pareto_mean(amount_min: float, alpha: float, amount_max: float | None) -> float:
    """Mean of min(amount_min * (1 + Lomax(alpha)), amount_max), as `_one_off_item` draws."""
    if amount_max is None:
        if alpha <= 1:
            raise ValueError("an uncapped one-off needs pareto_alpha > 1 for a finite mean")
        return amount_min * alpha / (alpha - 1)
    cap = amount_max / amount_min
    if cap <= 1:
        return amount_max
    if math.isclose(alpha, 1.0):
        return amount_min * (1 + math.log(cap))
    return amount_min * (1 + (1 - float(cap ** (1 - alpha))) / (alpha - 1))


def _one_off_daily(spec: OneOffSpec, tl: Timeline, scale: float) -> FloatArray:
    month_moy = tl.moy[tl.month_first]
    lam = spec.rate_per_month * scale * season_vector(spec.seasonality)[month_moy]
    if spec.months is not None:
        lam = np.where(np.isin(month_moy, spec.months), lam, 0.0)
    length = tl.month_last - tl.month_first + 1
    daily: FloatArray = (lam / length)[tl.month_idx]
    return daily


def expected_spending(
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    events: EventsSpec,
    *,
    rate_scale: float,
    daily_mult: FloatArray,
    bias: float,
) -> Expected:
    """Expected normal spending; call after calibration, with the calibrated `rate_scale`."""
    count: dict[str, FloatArray] = defaultdict(lambda: np.zeros(tl.n_days))
    spend: dict[str, FloatArray] = defaultdict(lambda: np.zeros(tl.n_days))
    prices = catalog.merchants["price_median"] * np.exp(catalog.merchants["price_sigma"] ** 2 / 2)

    for stream in user.streams:
        if stream.weekly_rate <= 0 or len(stream.merchants) == 0:
            continue
        lam = stream_rate(stream, tl, rate_scale, daily_mult)
        for merchant, weight in zip(stream.merchants, stream.weights, strict=True):
            price = float(prices[merchant]) * stream.amount_scale
            for category, p in zip(*_category_mix(catalog, str(merchant)), strict=True):
                count[category] += lam * weight * p
                spend[category] += lam * weight * p * price

    for spec in user.persona.one_off:
        candidates = list(
            catalog.select(spec.category, spec.subtypes, include_holdout=user.include_holdout).index
        )
        if not candidates:
            continue
        lam = _one_off_daily(spec, tl, events.one_off.rate_scale)
        amount = capped_pareto_mean(spec.amount_min, spec.pareto_alpha, spec.amount_max)
        weights = merchant_weights(catalog, candidates, user.include_holdout, bias)
        for merchant, weight in zip(candidates, weights, strict=True):
            for category, p in zip(*_category_mix(catalog, str(merchant)), strict=True):
                count[category] += lam * weight * p
                spend[category] += lam * weight * p * amount

    # Refunds reverse a share of normal purchases in their categories, lag_days later
    low, high = events.refunds.lag_days
    lags = range(low, high + 1)
    for category in events.refunds.categories:
        if category not in spend or events.refunds.share <= 0:
            continue
        refundable = spend[category].copy()
        refunds = np.zeros(tl.n_days)
        for lag in lags:
            if lag < tl.n_days:
                refunds[lag:] += refundable[: tl.n_days - lag]
        spend[category] -= events.refunds.share * refunds / len(lags)

    recurring = ledger.frame(("recurring",))
    for category in recurring["category"].unique():
        rows = recurring[recurring["category"] == category]
        spend[str(category)] += np.bincount(
            rows["day"].to_numpy(dtype=np.int64),
            weights=-rows["amount"].to_numpy(dtype=np.float64),
            minlength=tl.n_days,
        )
    return Expected(count=dict(count), spend=dict(spend))


def expected_rows(user: User, tl: Timeline, expected: Expected) -> pd.DataFrame:
    """Monthly `truth_expected` rows for every category with Poisson-process purchases."""
    rows = []
    for category in sorted(c for c, n in expected.count.items() if n.sum() > 0):
        for m in range(tl.n_months):
            n, s = expected.total(category, int(tl.month_first[m]), int(tl.month_last[m]))
            rows.append(
                {
                    "user_id": user.user_id,
                    "category": category,
                    "granularity": "month",
                    "period_start": str(tl.date_of(int(tl.month_first[m]))),
                    "expected_count": round(n, 4),
                    "expected_spend": round(s, 2),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "user_id",
            "category",
            "granularity",
            "period_start",
            "expected_count",
            "expected_spend",
        ],
    )
