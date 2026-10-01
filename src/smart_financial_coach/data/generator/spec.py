"""The generator spec: every generator parameter, loaded from YAML and validated by pydantic.

A spec may `extends:` another spec file; mappings are merged recursively and everything else
(lists, scalars) is replaced. `personas:` may be a directory of persona YAML files or inline.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from smart_financial_coach.data.generator.taxonomy import ANOMALY_KINDS, DEFAULT_CATEGORIES

_N = TypeVar("_N", int, float)


def _ordered(value: tuple[_N, _N]) -> tuple[_N, _N]:
    if value[0] > value[1]:
        raise ValueError(f"range low {value[0]} is above high {value[1]}")
    return value


def _months(value: dict[int, float]) -> dict[int, float]:
    if bad := [m for m in value if not 1 <= m <= 12]:
        raise ValueError(f"months must be 1..12, got {bad}")
    if any(v < 0 for v in value.values()):
        raise ValueError("seasonal multipliers must be non-negative")
    return value


def _positive(value: tuple[int, int]) -> tuple[int, int]:
    if value[0] < 1:
        raise ValueError("must be at least 1")
    return value


FloatRange = Annotated[tuple[float, float], AfterValidator(_ordered)]
IntRange = Annotated[tuple[int, int], AfterValidator(_ordered)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
# Sparse month-of-year multipliers; months not listed are 1.0.
Seasonality = Annotated[dict[int, float], AfterValidator(_months)]


def season_vector(seasonality: dict[int, float]) -> npt.NDArray[np.float64]:
    """Index 0 is unused so the vector can be indexed directly by month of year (1..12)."""
    vec = np.ones(13)
    for month, mult in seasonality.items():
        vec[month] = mult
    return vec


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Personas -------------------------------------------------------------------------------


class IncomeSpec(_Model):
    pattern: Literal["salaried", "dual", "freelance"]
    monthly_net: FloatRange  # mean net monthly inflow (combined for dual)
    employer_subtypes: list[str] = ["payroll"]
    client_subtypes: list[str] = ["client"]
    # Salaried / dual
    raise_pct: FloatRange = (0.0, 0.05)
    bonus_prob: Probability = 0.0
    bonus_share: FloatRange = (0.05, 0.12)  # share of annual net pay
    second_earner_share: FloatRange = (0.35, 0.5)
    # Freelance
    payments_per_month: NonNegative = 3.0
    max_payments_per_month: int = Field(default=6, ge=1)
    clients: IntRange = (2, 5)
    payment_sigma: NonNegative = 0.25  # month-to-month log-normal spread of total billings
    seasonality: Seasonality = {}
    slow_months_per_year: IntRange = (0, 0)
    slow_month_factor: NonNegative = 0.3


class StreamSpec(_Model):
    """A discretionary purchase stream, e.g. coffee. Poisson purchases at a user's favorites."""

    category: str
    subtypes: list[str] = []
    weekly_rate: FloatRange
    favorites: IntRange = (2, 5)
    zipf: NonNegative = 1.1
    amount_scale: FloatRange = (0.85, 1.2)
    weekend_boost: FloatRange = (0.0, 0.5)
    seasonality: Seasonality = {}


class RecurringSpec(_Model):
    """A recurring bill. Amount is `amount`, a share of income, or the merchant's list price."""

    category: str
    subtypes: list[str] = []
    probability: Probability = 1.0
    count: IntRange = (1, 1)
    amount: FloatRange | None = None
    income_share: FloatRange | None = None
    day_of_month: IntRange = (1, 28)
    jitter_days: int = Field(default=0, ge=0)
    months: list[int] | None = None  # active months of year; None = every month
    seasonality: Seasonality = {}  # absolute multipliers, e.g. a heating/cooling curve
    amount_noise: NonNegative = 0.0  # per-charge log-normal sigma
    drift_per_year: FloatRange = (0.0, 0.0)
    price_change_per_year: Probability = 0.0
    churn_per_year: NonNegative = 0.0
    late_start_prob: Probability = 0.0
    hour: int | None = Field(default=None, ge=0, le=23)

    @model_validator(mode="after")
    def _one_amount_source(self) -> "RecurringSpec":
        if self.amount is not None and self.income_share is not None:
            raise ValueError("set at most one of amount and income_share")
        if self.months is not None and any(not 1 <= m <= 12 for m in self.months):
            raise ValueError("months must be 1..12")
        return self


class OneOffSpec(_Model):
    """Rare, heavy-tailed purchases (Pareto above `amount_min`). Normal behavior, not anomalies."""

    category: str
    subtypes: list[str] = []
    rate_per_month: NonNegative
    amount_min: Annotated[float, Field(gt=0)]
    pareto_alpha: Annotated[float, Field(gt=0)] = 2.0
    amount_max: float | None = None
    months: list[int] | None = None  # months of year the purchase can happen; None = any
    seasonality: Seasonality = {}


class GoalTypeSpec(_Model):
    name: str
    weight: Annotated[float, Field(gt=0)] = 1.0


class PersonaSpec(_Model):
    name: str
    code: Annotated[str, Field(pattern=r"^[a-z]{2,4}$")]
    income: IncomeSpec
    savings_rate: FloatRange
    income_elasticity: NonNegative = 0.0  # discretionary rate vs trailing 2-month income
    payday_bump: NonNegative = 0.0
    starting_cushion_months: FloatRange = (0.5, 3.0)
    discretionary: list[StreamSpec]
    recurring: list[RecurringSpec] = []
    one_off: list[OneOffSpec] = []
    goals: list[GoalTypeSpec] = Field(min_length=1)

    def categories(self) -> set[str]:
        streams: list[StreamSpec | RecurringSpec | OneOffSpec] = [
            *self.discretionary,
            *self.recurring,
            *self.one_off,
        ]
        return {s.category for s in streams}


# --- Dataset-level spec ---------------------------------------------------------------------


class CalendarSpec(_Model):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "CalendarSpec":
        if self.end <= self.start:
            raise ValueError("calendar end must be after start")
        return self


class PopulationSpec(_Model):
    name: str
    split: Literal["train", "test"]
    seed: int
    id_prefix: Annotated[str, Field(pattern=r"^[a-z]{2}$")]
    users: dict[str, Annotated[int, Field(ge=0)]]  # persona name -> user count


class HoldoutSpec(_Model):
    share: Probability = 0.2
    exclude_top_n_per_category: int = Field(default=3, ge=0)
    exclude_categories: list[str] = ["Income"]
    test_user_bias: Annotated[float, Field(gt=0)] = 1.0  # preference boost for holdout merchants
    # Share of test users' spending transactions at holdout merchants that validation accepts
    test_share_range: FloatRange = (0.15, 0.25)
    seed: int = 7


class CatalogSpec(_Model):
    merchants: Path
    holdout: HoldoutSpec = HoldoutSpec()


class RenderingSpec(_Model):
    distortion: Literal["none", "light", "realistic", "heavy"] = "realistic"
    seed: int = 11


class OneOffEventsSpec(_Model):
    rate_scale: NonNegative = 1.0


class UnusualChargeSpec(_Model):
    rate_per_user_year: NonNegative = 0.0
    kinds: list[Literal["duplicate", "amount_outlier", "new_merchant_large"]] = list(ANOMALY_KINDS)  # type: ignore[arg-type]
    outlier_multiplier: FloatRange = (6.0, 15.0)
    new_merchant_categories: list[str] = ["Shopping", "Travel", "Entertainment"]
    new_merchant_min_amount: NonNegative = 250.0
    baseline_days: int = Field(default=90, ge=0)  # no anomalies before the user has history


class SpikeSpec(_Model):
    rate_per_user_year: NonNegative = 0.0
    multiplier: FloatRange = (1.8, 3.0)
    granularity: list[Literal["week", "month"]] = ["week", "month"]
    peak_threshold: NonNegative = 1.1  # never spike a category in a month seasonally above this
    min_weekly_rate: NonNegative = 1.0  # only spike categories the user buys in often enough
    baseline_months: int = Field(default=3, ge=0)


class RefundSpec(_Model):
    share: Probability = 0.03
    categories: list[str] = ["Shopping"]
    lag_days: IntRange = (3, 20)


class EventsSpec(_Model):
    one_off: OneOffEventsSpec = OneOffEventsSpec()
    unusual_charge: UnusualChargeSpec = UnusualChargeSpec()
    spending_spike: SpikeSpec = SpikeSpec()
    refunds: RefundSpec = RefundSpec()


class LabelsSpec(_Model):
    """FR-2 label contract: how flags are scored against ground truth, and the label checks."""

    weak_lift: Annotated[float, Field(gt=0)] = 1.3  # spike is weak below expected spend x this
    max_weak_share: Probability = 0.15  # of monthly spike labels; checked by validate
    max_drivers: int = Field(default=5, ge=1)  # driving transactions returned per spike
    duplicate_window_minutes: int = Field(default=90, ge=1)
    oracle_recall: Probability = 0.5  # oracle precision is measured at this recall
    oracle_min_precision: Probability = 0.7  # validate fails below this (PRD target)
    min_labels_for_oracle_check: int = Field(default=20, ge=1)


OutcomeClass = Literal["on_track", "borderline", "off_track"]


class GoalsSpec(_Model):
    per_user: IntRange = (1, 2)
    outcome_mix: dict[OutcomeClass, NonNegative] = {
        "on_track": 0.34,
        "borderline": 0.33,
        "off_track": 0.33,
    }
    target_inside_history_share: Probability = 0.5
    allocation_share: FloatRange = (0.5, 0.9)  # share of monthly net savings put toward goals
    # Months from as_of_date to target_date, for goals that end inside the history
    forecast_horizon_months: Annotated[IntRange, AfterValidator(_positive)] = (3, 12)


class Spec(_Model):
    name: str
    calendar: CalendarSpec
    categories: list[str] = list(DEFAULT_CATEGORIES)
    populations: list[PopulationSpec] = Field(min_length=1)
    personas: dict[str, PersonaSpec]
    catalog: CatalogSpec
    rendering: RenderingSpec = RenderingSpec()
    events: EventsSpec = EventsSpec()
    goals: GoalsSpec = GoalsSpec()
    labels: LabelsSpec = LabelsSpec()

    @model_validator(mode="after")
    def _consistent(self) -> "Spec":
        errors: list[str] = []
        if len({p.name for p in self.populations}) != len(self.populations):
            errors.append("population names must be unique")
        if len({p.id_prefix for p in self.populations}) != len(self.populations):
            errors.append("population id_prefix values must be unique")
        for pop in self.populations:
            errors += [
                f"population {pop.name}: unknown persona {name!r}"
                for name in pop.users
                if name not in self.personas
            ]
        if len({p.code for p in self.personas.values()}) != len(self.personas):
            errors.append("persona codes must be unique")
        known = set(self.categories)
        for key, persona in self.personas.items():
            if key != persona.name:
                errors.append(f"persona key {key!r} does not match its name {persona.name!r}")
            errors += [
                f"persona {key}: unknown category {c!r}"
                for c in sorted(persona.categories() - known)
            ]
        events = self.events
        referenced = {*events.unusual_charge.new_merchant_categories, *events.refunds.categories}
        errors += [f"events: unknown category {c!r}" for c in sorted(referenced - known)]
        if sum(self.goals.outcome_mix.values()) <= 0:
            errors.append("goals.outcome_mix must have a positive weight")
        if errors:
            raise ValueError("; ".join(errors))
        return self


# --- Loading --------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def _resolve_raw(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Read a spec file, applying `extends` and making relative paths absolute."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular extends at {path}")
    raw = _read_yaml(path)
    if isinstance(raw.get("personas"), str):
        raw["personas"] = str((path.parent / raw["personas"]).resolve())
    catalog = raw.get("catalog")
    if isinstance(catalog, dict) and isinstance(catalog.get("merchants"), str):
        catalog["merchants"] = str((path.parent / catalog["merchants"]).resolve())
    if parent := raw.pop("extends", None):
        raw = _deep_merge(_resolve_raw(path.parent / parent, seen | {path}), raw)
    return raw


def load_spec(path: str | Path) -> Spec:
    raw = _resolve_raw(Path(path))
    if isinstance(personas := raw.get("personas"), str):
        files = sorted(Path(personas).glob("*.yaml"))
        if not files:
            raise ValueError(f"no persona files in {personas}")
        raw["personas"] = {}
        for file in files:
            persona = _read_yaml(file)
            if persona.get("name") in raw["personas"]:
                raise ValueError(f"{file}: duplicate persona name {persona['name']!r}")
            raw["personas"][persona.get("name")] = persona
    return Spec.model_validate(raw)


def spec_hash(spec: Spec) -> str:
    """Hash of every parameter, with the catalog file's content in place of its path."""
    payload = spec.model_dump(mode="json")
    payload["catalog"]["merchants"] = hashlib.sha256(
        spec.catalog.merchants.read_bytes()
    ).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
