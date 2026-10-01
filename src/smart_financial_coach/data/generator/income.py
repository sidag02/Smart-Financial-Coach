"""Stage 4: income. Salaried biweekly pay, dual earners, or irregular freelance client payments."""

import numpy as np
import numpy.typing as npt

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User
from smart_financial_coach.data.generator.spec import IncomeSpec, season_vector
from smart_financial_coach.data.generator.taxonomy import INCOME
from smart_financial_coach.data.generator.timeline import IntArray, Timeline

FloatArray = npt.NDArray[np.float64]


def _month_moy(tl: Timeline) -> IntArray:
    return tl.moy[tl.month_first]


def _raise_factors(
    tl: Timeline, months: IntArray, spec: IncomeSpec, rng: np.random.Generator
) -> FloatArray:
    """Cumulative pay multiplier per payment: one raise a year, in a month sampled per user."""
    raise_moy = int(rng.integers(1, 13))
    month_moy = _month_moy(tl)
    events = [m for m in range(1, tl.n_months) if month_moy[m] == raise_moy]
    pcts = rng.uniform(*spec.raise_pct, size=len(events))
    factor = np.ones(len(months))
    for month, pct in zip(events, pcts, strict=True):
        factor[months >= month] *= 1.0 + pct
    return factor


def _biweekly(
    tl: Timeline, monthly: float, spec: IncomeSpec, rng: np.random.Generator
) -> tuple[IntArray, FloatArray]:
    fridays = np.flatnonzero(tl.dow == 4)
    checks = fridays[int(rng.integers(0, 2)) :: 2]
    factor = _raise_factors(tl, tl.month_idx[checks], spec, rng)
    amounts = monthly * 12 / 26 * factor * rng.normal(1.0, 0.004, size=len(checks))
    return tl.previous_business_day(checks), amounts


def _semimonthly(
    tl: Timeline, monthly: float, spec: IncomeSpec, rng: np.random.Generator
) -> tuple[IntArray, FloatArray]:
    days: list[int] = []
    for month in range(tl.n_months):
        if (mid := tl.month_day(month, 15)) >= 0:
            days.append(int(tl.previous_business_day(np.array([mid]))[0]))
        days.append(tl.last_business_day(month))
    checks = np.array(sorted({d for d in days if d >= 0}), dtype=np.int64)
    factor = _raise_factors(tl, tl.month_idx[checks], spec, rng)
    return checks, monthly / 2 * factor * rng.normal(1.0, 0.004, size=len(checks))


def _bonus(
    tl: Timeline, monthly: float, spec: IncomeSpec, rng: np.random.Generator
) -> tuple[IntArray, FloatArray]:
    if rng.random() >= spec.bonus_prob:
        return np.array([], dtype=np.int64), np.array([])
    bonus_moy = int(rng.integers(1, 13))
    months = [m for m in range(tl.n_months) if _month_moy(tl)[m] == bonus_moy]
    days = np.array([tl.month_day(m, 15) for m in months], dtype=np.int64)
    days = tl.previous_business_day(days[days >= 0])
    return days, monthly * 12 * rng.uniform(*spec.bonus_share, size=len(days))


def _payers(catalog: Catalog, subtypes: list[str], n: int, rng: np.random.Generator) -> list[str]:
    candidates = catalog.select(INCOME, subtypes, include_holdout=True)
    weights = candidates["popularity"].to_numpy(dtype=np.float64)
    n = min(n, len(candidates))
    return [
        str(m)
        for m in rng.choice(
            candidates.index.to_numpy(), size=n, replace=False, p=weights / weights.sum()
        )
    ]


def _add(
    ledger: Ledger,
    catalog: Catalog,
    days: IntArray,
    amounts: FloatArray,
    merchant: str,
    rng: np.random.Generator,
) -> None:
    ledger.add(
        "income",
        day=days,
        minute=rng.integers(120, 540, size=len(days)),  # ACH credits post overnight / early morning
        amount=amounts,
        merchant_id=[merchant] * len(days),
        category=[INCOME] * len(days),
        channel=catalog.sample_channels(np.array([merchant] * len(days)), rng),
        is_recurring=True,
    )


def _freelance(
    user: User, tl: Timeline, catalog: Catalog, ledger: Ledger, rng: np.random.Generator
) -> None:
    spec = user.persona.income
    season = season_vector(spec.seasonality)
    season[1:] /= season[1:].mean()
    month_moy = _month_moy(tl)
    slow = np.ones(tl.n_months)
    years = tl.days[tl.month_first].astype("datetime64[Y]").astype(np.int64)
    for year in np.unique(years):
        months = np.flatnonzero(years == year)
        k = min(
            int(rng.integers(spec.slow_months_per_year[0], spec.slow_months_per_year[1] + 1)),
            len(months),
        )
        slow[rng.choice(months, size=k, replace=False)] = spec.slow_month_factor

    clients = _payers(
        catalog, spec.client_subtypes, int(rng.integers(spec.clients[0], spec.clients[1] + 1)), rng
    )
    client_p = rng.dirichlet(np.full(len(clients), 1.5))
    sigma = spec.payment_sigma
    carry = 0.0  # invoiced but not yet paid rolls into the next month
    for month in range(tl.n_months):
        level = season[month_moy[month]] * slow[month]
        billed = user.monthly_net * level * float(rng.lognormal(-(sigma**2) / 2, sigma)) + carry
        count = min(int(rng.poisson(spec.payments_per_month * level)), spec.max_payments_per_month)
        business = (
            np.flatnonzero(tl.is_business[tl.month_first[month] : tl.month_last[month] + 1])
            + tl.month_first[month]
        )
        if count == 0 or len(business) == 0:
            carry = billed
            continue
        carry = 0.0
        days = np.sort(rng.choice(business, size=count, replace=count > len(business)))
        amounts = billed * rng.dirichlet(np.full(count, 2.0))
        payers = rng.choice(np.array(clients), size=count, p=client_p)
        for payer in np.unique(payers):
            idx = payers == payer
            _add(ledger, catalog, days[idx], amounts[idx], str(payer), rng)


def generate_income(user: User, tl: Timeline, catalog: Catalog, ledger: Ledger) -> None:
    rng = user.rng("income")
    spec = user.persona.income
    if spec.pattern == "freelance":
        _freelance(user, tl, catalog, ledger, rng)
        return
    if spec.pattern == "salaried":
        (employer,) = _payers(catalog, spec.employer_subtypes, 1, rng)
        streams = [(employer, *_biweekly(tl, user.monthly_net, spec, rng))]
    else:
        first, second = _payers(catalog, spec.employer_subtypes, 2, rng)
        share = float(rng.uniform(*spec.second_earner_share))
        streams = [
            (first, *_biweekly(tl, user.monthly_net * (1 - share), spec, rng)),
            (second, *_semimonthly(tl, user.monthly_net * share, spec, rng)),
        ]
    for employer, days, amounts in streams:
        _add(ledger, catalog, days, amounts, employer, rng)
        bonus_days, bonus = _bonus(tl, float(amounts.sum()) / tl.n_months, spec, rng)
        _add(ledger, catalog, bonus_days, bonus, employer, rng)


def paydays(ledger: Ledger) -> IntArray:
    frame = ledger.frame(("income",))
    return np.unique(frame["day"].to_numpy(dtype=np.int64))
