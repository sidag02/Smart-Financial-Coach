"""Stages 1 and 3: sample users from personas, give each one seeds and merchant preferences."""

import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.spec import PersonaSpec, PopulationSpec, Spec, StreamSpec
from smart_financial_coach.data.generator.taxonomy import CITIES, City

# One child seed per stage. Order is part of the output contract: append, never reorder.
STAGES = (
    "params",
    "preferences",
    "income",
    "recurring",
    "one_off",
    "spikes",
    "discretionary",
    "unusual",
    "refunds",
    "balance",
    "rendering",
    "goals",
    "spike_extra",
)


@dataclass(frozen=True)
class Stream:
    """A user's instance of a persona discretionary stream."""

    spec: StreamSpec
    weekly_rate: float
    amount_scale: float
    dow_weights: npt.NDArray[np.float64]  # 7 weights, Monday first, mean 1
    merchants: npt.NDArray[np.str_]  # favorites, most preferred first
    weights: npt.NDArray[np.float64]  # Zipf preference over favorites


@dataclass(frozen=True)
class User:
    user_id: str
    split: str
    persona: PersonaSpec
    city: City
    monthly_net: float
    savings_target: float
    cushion_months: float
    streams: list[Stream]
    seeds: dict[str, np.random.SeedSequence] = field(repr=False)

    def rng(self, stage: str) -> np.random.Generator:
        """A fresh generator for a stage; calling twice gives the same stream."""
        return np.random.default_rng(self.seeds[stage])

    @property
    def include_holdout(self) -> bool:
        return self.split == "test"


def user_seeds(population_seed: int, user_id: str) -> dict[str, np.random.SeedSequence]:
    """Seeds depend only on the population seed and the user's ID, never on generation order."""
    root = np.random.SeedSequence(population_seed, spawn_key=(zlib.crc32(user_id.encode()),))
    return dict(zip(STAGES, root.spawn(len(STAGES)), strict=True))


def merchant_weights(
    merchants: "Catalog", candidates: list[str], include_holdout: bool, bias: float
) -> npt.NDArray[np.float64]:
    """Selection weights: popularity, boosted for holdout merchants when the user may see them."""
    m = merchants.merchants.loc[candidates]
    weights = m["popularity"].to_numpy(dtype=np.float64)
    if include_holdout:
        weights = np.where(m["holdout"].to_numpy(), weights * bias, weights)
    result: npt.NDArray[np.float64] = weights / weights.sum()
    return result


def _dow_weights(boost: float) -> npt.NDArray[np.float64]:
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0 + boost / 2, 1.0 + boost, 1.0 + boost])
    weights = np.clip(weights, 0.05, None)
    result: npt.NDArray[np.float64] = weights / weights.mean()
    return result


def _stream(
    spec: StreamSpec, user_split: str, catalog: Catalog, bias: float, rng: np.random.Generator
) -> Stream:
    include_holdout = user_split == "test"
    candidates = list(
        catalog.select(spec.category, spec.subtypes, include_holdout=include_holdout).index
    )
    n = min(int(rng.integers(spec.favorites[0], spec.favorites[1] + 1)), len(candidates))
    weekly_rate = float(rng.uniform(*spec.weekly_rate))
    amount_scale = float(rng.uniform(*spec.amount_scale))
    dow = _dow_weights(float(rng.uniform(*spec.weekend_boost)))
    if n == 0:
        return Stream(spec, 0.0, amount_scale, dow, np.array([], dtype=str), np.array([]))
    p = merchant_weights(catalog, candidates, include_holdout, bias)
    favorites = rng.choice(np.array(candidates), size=n, replace=False, p=p)
    zipf = 1.0 / np.arange(1, n + 1) ** spec.zipf
    return Stream(spec, weekly_rate, amount_scale, dow, favorites.astype(str), zipf / zipf.sum())


def make_user(
    population: PopulationSpec, persona: PersonaSpec, index: int, catalog: Catalog, bias: float
) -> User:
    user_id = f"u_{population.id_prefix}_{persona.code}_{index:04d}"
    seeds = user_seeds(population.seed, user_id)
    rng = np.random.default_rng(seeds["params"])
    city = CITIES[int(rng.integers(len(CITIES)))]
    monthly_net = float(rng.uniform(*persona.income.monthly_net))
    savings_target = float(rng.uniform(*persona.savings_rate))
    cushion = float(rng.uniform(*persona.starting_cushion_months))
    pref_rng = np.random.default_rng(seeds["preferences"])
    streams = [_stream(s, population.split, catalog, bias, pref_rng) for s in persona.discretionary]
    return User(
        user_id,
        population.split,
        persona,
        city,
        monthly_net,
        savings_target,
        cushion,
        streams,
        seeds,
    )


def iter_users(spec: Spec, catalog: Catalog) -> Iterator[User]:
    bias = spec.catalog.holdout.test_user_bias
    for population in spec.populations:
        for persona_name, count in population.users.items():
            persona = spec.personas[persona_name]
            for index in range(count):
                yield make_user(population, persona, index, catalog, bias)
