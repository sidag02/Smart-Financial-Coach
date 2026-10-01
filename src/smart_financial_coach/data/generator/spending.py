"""Stage 5: spending processes. Recurring bills, discretionary purchases and normal one-offs."""

import numpy as np
import numpy.typing as npt

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import Stream, User, merchant_weights
from smart_financial_coach.data.generator.spec import OneOffSpec, RecurringSpec, season_vector
from smart_financial_coach.data.generator.timeline import IntArray, Timeline

FloatArray = npt.NDArray[np.float64]

# Internal ledger process for spike extras; written to truth tables as "discretionary"
SPIKE_EXTRA = "spike_extra"


def _pick_merchants(
    catalog: Catalog,
    category: str,
    subtypes: list[str],
    user: User,
    n: int,
    bias: float,
    rng: np.random.Generator,
) -> list[str]:
    candidates = list(
        catalog.select(category, subtypes, include_holdout=user.include_holdout).index
    )
    n = min(n, len(candidates))
    if n == 0:
        return []
    p = merchant_weights(catalog, candidates, user.include_holdout, bias)
    return [str(m) for m in rng.choice(np.array(candidates), size=n, replace=False, p=p)]


# --- Recurring ------------------------------------------------------------------------------


def _active_months(spec: RecurringSpec, tl: Timeline, rng: np.random.Generator) -> IntArray:
    start = int(rng.integers(1, tl.n_months)) if rng.random() < spec.late_start_prob else 0
    end = tl.n_months
    if spec.churn_per_year > 0:
        end = min(end, start + int(rng.geometric(min(spec.churn_per_year / 12, 1.0))))
    months = np.arange(start, end)
    if spec.months is not None:
        months = months[np.isin(tl.moy[tl.month_first[months]], spec.months)]
    return months


def _step_changes(
    n_months: int, per_year: float, low: float, high: float, rng: np.random.Generator
) -> FloatArray:
    """Multiplier per month with at most one random step change per 12-month block."""
    factor = np.ones(n_months)
    for block in range(0, n_months, 12):
        if rng.random() < per_year:
            at = block + int(rng.integers(0, 12))
            factor[at:] *= rng.uniform(low, high)
    return factor


def _recurring_item(
    spec: RecurringSpec,
    merchant: str,
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    rng: np.random.Generator,
) -> None:
    if spec.amount is not None:
        base = float(rng.uniform(*spec.amount))
    elif spec.income_share is not None:
        base = user.monthly_net * float(rng.uniform(*spec.income_share))
    else:
        base = catalog.price(merchant)
    dom = int(rng.integers(spec.day_of_month[0], spec.day_of_month[1] + 1))
    drift = float(rng.uniform(*spec.drift_per_year))
    months = _active_months(spec, tl, rng)
    if len(months) == 0:
        return
    price = _step_changes(tl.n_months, spec.price_change_per_year, 1.05, 1.2, rng)
    season = season_vector(spec.seasonality)

    days = np.array([tl.month_day(int(m), dom) for m in months], dtype=np.int64)
    keep = days >= 0
    months, days = months[keep], days[keep]
    if spec.jitter_days:
        days = np.clip(
            days + rng.integers(-spec.jitter_days, spec.jitter_days + 1, size=len(days)),
            0,
            tl.n_days - 1,
        )
    years_active = (months - months[0]) // 12 if len(months) else months
    amounts = (
        base
        * (1 + drift) ** years_active
        * price[months]
        * season[tl.moy[tl.month_first[months]]]
        * (rng.lognormal(0.0, spec.amount_noise, size=len(months)) if spec.amount_noise else 1.0)
    )
    ids = np.array([merchant] * len(days))
    minutes = (
        np.full(len(days), spec.hour * 60) + rng.integers(0, 60, size=len(days))
        if spec.hour is not None
        else rng.integers(0, 6 * 60, size=len(days))
    )
    ledger.add(
        "recurring",
        day=days,
        minute=minutes,
        amount=-amounts,
        merchant_id=ids,
        category=catalog.sample_categories(ids, rng),
        channel=catalog.sample_channels(ids, rng),
        is_recurring=True,
    )


def generate_recurring(
    user: User, tl: Timeline, catalog: Catalog, ledger: Ledger, bias: float
) -> None:
    rng = user.rng("recurring")
    for spec in user.persona.recurring:
        if rng.random() >= spec.probability:
            continue
        count = int(rng.integers(spec.count[0], spec.count[1] + 1))
        for merchant in _pick_merchants(
            catalog, spec.category, spec.subtypes, user, count, bias, rng
        ):
            _recurring_item(spec, merchant, user, tl, catalog, ledger, rng)


# --- One-offs -------------------------------------------------------------------------------


def _one_off_item(
    spec: OneOffSpec,
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    scale: float,
    bias: float,
    rng: np.random.Generator,
) -> None:
    month_moy = tl.moy[tl.month_first]
    lam = spec.rate_per_month * scale * season_vector(spec.seasonality)[month_moy]
    if spec.months is not None:
        lam = np.where(np.isin(month_moy, spec.months), lam, 0.0)
    counts = rng.poisson(lam)
    if counts.sum() == 0:
        return
    month = np.repeat(np.arange(tl.n_months), counts)
    length = tl.month_last[month] - tl.month_first[month] + 1
    days = tl.month_first[month] + (rng.random(len(month)) * length).astype(np.int64)
    candidates = list(
        catalog.select(spec.category, spec.subtypes, include_holdout=user.include_holdout).index
    )
    if not candidates:
        return
    p = merchant_weights(catalog, candidates, user.include_holdout, bias)
    ids = rng.choice(np.array(candidates), size=len(days), p=p).astype(str)
    amounts = spec.amount_min * (1.0 + rng.pareto(spec.pareto_alpha, size=len(days)))
    if spec.amount_max is not None:
        amounts = np.minimum(amounts, spec.amount_max)
    ledger.add(
        "one_off",
        day=days,
        minute=catalog.sample_minutes(ids, rng),
        amount=-amounts,
        merchant_id=ids,
        category=catalog.sample_categories(ids, rng),
        channel=catalog.sample_channels(ids, rng),
    )


def generate_one_offs(
    user: User, tl: Timeline, catalog: Catalog, ledger: Ledger, scale: float, bias: float
) -> None:
    rng = user.rng("one_off")
    for spec in user.persona.one_off:
        _one_off_item(spec, user, tl, catalog, ledger, scale, bias, rng)


# --- Discretionary --------------------------------------------------------------------------


def daily_multiplier(
    user: User, tl: Timeline, income_days: IntArray, income_amounts: FloatArray
) -> FloatArray:
    """Income coupling: payday bump, and spending that follows trailing two-month income."""
    mult = np.ones(tl.n_days)
    if user.persona.payday_bump and len(income_days):
        bumped = np.unique(
            np.clip(np.concatenate([income_days, income_days + 1]), 0, tl.n_days - 1)
        )
        mult[bumped] *= 1.0 + user.persona.payday_bump
    if user.persona.income_elasticity and len(income_days):
        monthly = np.bincount(
            tl.month_idx[income_days], weights=income_amounts, minlength=tl.n_months
        )
        mean = monthly.mean()
        if mean > 0:
            trailing = np.array(
                [monthly[max(0, m - 2) : m].mean() if m else mean for m in range(tl.n_months)]
            )
            factor = np.clip((trailing / mean) ** user.persona.income_elasticity, 0.5, 1.6)
            mult *= factor[tl.month_idx]
    return mult


def stream_rate(
    stream: Stream, tl: Timeline, rate_scale: float, daily_mult: FloatArray
) -> FloatArray:
    """Expected purchases per day: rate x day-of-week x season x income coupling."""
    season = season_vector(stream.spec.seasonality)
    rate: FloatArray = (
        stream.weekly_rate
        / 7
        * rate_scale
        * stream.dow_weights[tl.dow]
        * season[tl.moy]
        * daily_mult
    )
    return rate


def _purchases(
    process: str,
    stream: Stream,
    lam: FloatArray,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    rng: np.random.Generator,
) -> None:
    counts = rng.poisson(lam)
    days = np.repeat(np.arange(tl.n_days), counts)
    if len(days) == 0:
        return
    ids = rng.choice(stream.merchants, size=len(days), p=stream.weights)
    m = catalog.merchants.reindex(ids)
    median = m["price_median"].to_numpy(dtype=np.float64)
    sigma = m["price_sigma"].to_numpy(dtype=np.float64)
    amounts = median * stream.amount_scale * rng.lognormal(0.0, sigma)
    ledger.add(
        process,
        day=days,
        minute=catalog.sample_minutes(ids, rng),
        amount=-np.maximum(amounts, 0.5),
        merchant_id=ids,
        category=catalog.sample_categories(ids, rng),
        channel=catalog.sample_channels(ids, rng),
    )


def generate_discretionary(
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    *,
    rate_scale: float,
    daily_mult: FloatArray,
) -> None:
    """Normal Poisson purchases per day. Spikes are drawn separately (`generate_spike_extras`)."""
    rng = user.rng("discretionary")  # same seed on every calibration pass
    ledger.clear("discretionary")
    for stream in user.streams:
        if stream.weekly_rate <= 0 or len(stream.merchants) == 0:
            continue
        lam = stream_rate(stream, tl, rate_scale, daily_mult)
        _purchases("discretionary", stream, lam, tl, catalog, ledger, rng)


def generate_spike_extras(
    user: User,
    tl: Timeline,
    catalog: Catalog,
    ledger: Ledger,
    *,
    rate_scale: float,
    daily_mult: FloatArray,
    spikes: dict[str, FloatArray],
) -> None:
    """Extra purchases on spike days at rate λ·(m - 1), on top of the normal draw.

    Normal + extra has the same distribution as one draw at λ·m (Poisson superposition). A separate
    seed and ledger process keep normal purchases, and their IDs, independent of spikes.
    """
    rng = user.rng("spike_extra")
    for stream in user.streams:
        mult = spikes.get(stream.spec.category)
        if mult is None or stream.weekly_rate <= 0 or len(stream.merchants) == 0:
            continue
        lam = stream_rate(stream, tl, rate_scale, daily_mult) * (mult - 1.0)
        _purchases(SPIKE_EXTRA, stream, lam, tl, catalog, ledger, rng)
