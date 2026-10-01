"""`generate(spec) -> Dataset`: run every stage for every user and assemble the tables."""

import json
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd

from smart_financial_coach import __version__
from smart_financial_coach.data.generator.calibration import (
    calibrate_discretionary,
    starting_balance,
)
from smart_financial_coach.data.generator.catalog import Catalog, load_catalog
from smart_financial_coach.data.generator.dataset import TABLES, Dataset
from smart_financial_coach.data.generator.events import (
    generate_refunds,
    generate_unusual_charges,
    plan_spikes,
    spike_rows,
    unusual_tiers,
)
from smart_financial_coach.data.generator.expected import expected_rows, expected_spending
from smart_financial_coach.data.generator.goals import generate_goals
from smart_financial_coach.data.generator.income import generate_income
from smart_financial_coach.data.generator.ledger import Ledger
from smart_financial_coach.data.generator.population import User, iter_users
from smart_financial_coach.data.generator.rendering import Renderer
from smart_financial_coach.data.generator.spec import Spec, spec_hash
from smart_financial_coach.data.generator.spending import (
    SPIKE_EXTRA,
    daily_multiplier,
    generate_one_offs,
    generate_recurring,
    generate_spike_extras,
)
from smart_financial_coach.data.generator.timeline import Timeline

SCHEMA_VERSION = "2"
LABEL_CONTRACT_VERSION = "1"


def _timestamps(tl: Timeline, day: pd.Series, minute: pd.Series) -> list[str]:
    stamps = tl.days[day.to_numpy()].astype("datetime64[m]") + minute.to_numpy().astype(
        "timedelta64[m]"
    )
    return [s.replace("T", " ") for s in np.datetime_as_string(stamps, unit="m").tolist()]


def generate_user(
    user: User, spec: Spec, tl: Timeline, catalog: Catalog, renderer: Renderer
) -> tuple[dict[str, pd.DataFrame], int]:
    """All stages for one user. Depends only on the spec, the catalog and the user's own seeds.

    Returns the user's rows per table, and how many planned unusual charges were dropped.
    """
    bias = spec.catalog.holdout.test_user_bias
    ledger = Ledger(user.user_id)
    generate_income(user, tl, catalog, ledger)
    generate_recurring(user, tl, catalog, ledger, bias)
    generate_one_offs(user, tl, catalog, ledger, spec.events.one_off.rate_scale, bias)
    spike_mult, spikes = plan_spikes(user, tl, spec.events)
    income = ledger.frame(("income",))
    daily = daily_multiplier(
        user,
        tl,
        income["day"].to_numpy(dtype=np.int64),
        income["amount"].to_numpy(dtype=np.float64),
    )
    scale = calibrate_discretionary(user, tl, catalog, ledger, daily_mult=daily)
    expected = expected_spending(
        user, tl, catalog, ledger, spec.events, rate_scale=scale, daily_mult=daily, bias=bias
    )
    generate_spike_extras(
        user, tl, catalog, ledger, rate_scale=scale, daily_mult=daily, spikes=spike_mult
    )
    dropped = generate_unusual_charges(user, tl, catalog, ledger, spec.events, bias)
    generate_refunds(user, tl, ledger, spec.events)

    txns = ledger.frame().sort_values(
        ["day", "minute", "transaction_id"], kind="mergesort", ignore_index=True
    )
    raw = renderer.render_user(txns, user.city, user.rng("rendering"))
    # Refunds and duplicate charges reuse the original transaction's text
    position = {tid: i for i, tid in enumerate(txns["transaction_id"])}
    for i, original in enumerate(txns["copy_of"].tolist()):
        if isinstance(original, str):
            raw[i] = raw[position[original]]

    goals, truth_goals = generate_goals(user, tl, txns, spec.goals)
    income_total = float(txns.loc[txns["process"] == "income", "amount"].sum())
    tables = {
        "users": pd.DataFrame(
            [
                {
                    "user_id": user.user_id,
                    "split": user.split,
                    "persona": user.persona.name,
                    "timezone": user.city.timezone,
                    "monthly_income_estimate": float(round(income_total / tl.n_months)),
                    "starting_balance": starting_balance(user, txns),
                }
            ]
        ),
        "transactions": pd.DataFrame(
            {
                "transaction_id": txns["transaction_id"],
                "user_id": user.user_id,
                "ts": _timestamps(tl, txns["day"], txns["minute"]),
                "amount": txns["amount"],
                "currency": "USD",
                "merchant_raw": raw,
                "channel": txns["channel"],
            }
        ),
        "goals": goals,
        "truth_transactions": pd.DataFrame(
            {
                "transaction_id": txns["transaction_id"],
                "category": txns["category"],
                "merchant_id": txns["merchant_id"],
                "process": txns["process"].replace(SPIKE_EXTRA, "discretionary"),
                "is_recurring": txns["is_recurring"].astype(int),
                "anomaly_kind": txns["anomaly_kind"],
                "tier": unusual_tiers(txns),
                "related_transaction_id": txns["copy_of"],
            }
        ),
        "truth_periods": spike_rows(user, tl, spikes, expected, txns, spec.labels.weak_lift),
        "truth_expected": expected_rows(user, tl, expected),
        "truth_goals": truth_goals,
    }
    return tables, dropped


def _merchants_table(catalog: Catalog) -> pd.DataFrame:
    m = catalog.merchants
    return pd.DataFrame(
        {
            "merchant_id": m["merchant_id"].to_numpy(),
            "canonical_name": m["canonical_name"].to_numpy(),
            "category": m["category"].to_numpy(),
            "subtype": m["subtype"].to_numpy(),
            "price_median": m["price_median"].to_numpy(),
            "price_sigma": m["price_sigma"].to_numpy(),
            "is_ambiguous": m["is_ambiguous"].astype(int).to_numpy(),
            "holdout": m["holdout"].astype(int).to_numpy(),
        }
    )


def label_contract(spec: Spec) -> dict[str, float | int]:
    """Contract parameters stored in `meta`, so labels can be scored from the SQLite file alone."""
    return {
        **spec.labels.model_dump(),
        "baseline_days": spec.events.unusual_charge.baseline_days,
        "baseline_months": spec.events.spending_spike.baseline_months,
    }


def generate(spec: Spec, *, progress: Callable[[int, int], None] | None = None) -> Dataset:
    tl = Timeline(spec.calendar.start, spec.calendar.end)
    catalog = load_catalog(spec.catalog.merchants, spec.catalog.holdout, spec.categories)
    renderer = Renderer(catalog, spec.rendering.distortion, spec.rendering.seed)
    total = sum(sum(p.users.values()) for p in spec.populations)
    parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
    dropped = 0
    for done, user in enumerate(iter_users(spec, catalog), start=1):
        user_tables, user_dropped = generate_user(user, spec, tl, catalog, renderer)
        dropped += user_dropped
        for name, frame in user_tables.items():
            parts[name].append(frame)
        if progress is not None:
            progress(done, total)

    tables: dict[str, pd.DataFrame] = {"truth_merchants": _merchants_table(catalog)}
    for name, table in TABLES.items():
        if name == "truth_merchants":
            continue
        frames = [f for f in parts[name] if not f.empty]
        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=table.column_names)
        )
        tables[name] = df[table.column_names].sort_values(
            list(table.primary_key), kind="mergesort", ignore_index=True
        )
    seeds = {
        "populations": {p.name: p.seed for p in spec.populations},
        "holdout": spec.catalog.holdout.seed,
        "rendering": spec.rendering.seed,
    }
    meta = {
        "generator_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "spec_name": spec.name,
        "spec_hash": spec_hash(spec),
        "calendar_start": str(spec.calendar.start),
        "calendar_end": str(spec.calendar.end),
        "distortion": spec.rendering.distortion,
        "seeds": json.dumps(seeds, sort_keys=True),
        "categories": json.dumps(spec.categories),
        "unusual_charges_dropped": str(dropped),
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "label_contract": json.dumps(label_contract(spec), sort_keys=True),
    }
    return Dataset(tables=tables, meta=meta)
