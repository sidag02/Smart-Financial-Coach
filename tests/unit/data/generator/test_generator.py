import json
import time
from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.data.generator import (
    MODEL_TABLES,
    TRUTH_TABLES,
    Dataset,
    Spec,
    generate,
    load_spec,
    read_sqlite,
)
from smart_financial_coach.data.generator import validate as validate_module
from smart_financial_coach.data.generator.validate import validate


def _ledger(ds: Dataset) -> pd.DataFrame:
    return ds["transactions"].merge(ds["truth_transactions"], on="transaction_id")


def test_small_dataset_passes_quality_checks(small_dataset: Dataset, small_spec: Spec) -> None:
    report = validate(small_dataset, small_spec)

    assert report.ok, report.errors
    assert report.stats["users"] == 30


def test_regeneration_is_identical(small_dataset: Dataset, small_spec: Spec) -> None:
    assert generate(small_spec).content_hash() == small_dataset.content_hash()


def test_sqlite_round_trip_keeps_content(small_dataset: Dataset, tmp_path: Path) -> None:
    path = small_dataset.to_sqlite(tmp_path / "small.sqlite")

    assert read_sqlite(path).content_hash() == small_dataset.content_hash()
    with pytest.raises(FileExistsError):
        small_dataset.to_sqlite(path)


def test_failed_write_leaves_no_files(small_dataset: Dataset, tmp_path: Path) -> None:
    broken = Dataset(
        tables={k: v for k, v in small_dataset.tables.items() if k != "goals"},
        meta=small_dataset.meta,
    )

    with pytest.raises(KeyError):
        broken.to_sqlite(tmp_path / "broken.sqlite")
    assert list(tmp_path.iterdir()) == []


def test_validation_rejects_a_different_spec(small_dataset: Dataset, configs: Path) -> None:
    report = validate(small_dataset, load_spec(configs / "clean.yaml"))

    assert not report.ok
    assert report.errors[0].startswith("spec:")


def test_validation_checks_holdout_share(small_spec: Spec, monkeypatch: pytest.MonkeyPatch) -> None:
    holdout = small_spec.catalog.holdout.model_copy(update={"test_share_range": (0.5, 0.6)})
    catalog = small_spec.catalog.model_copy(update={"holdout": holdout})
    spec = small_spec.model_copy(update={"catalog": catalog})
    monkeypatch.setattr(validate_module, "MIN_TEST_USERS_FOR_HOLDOUT_CHECK", 1)

    report = validate(generate(spec), spec)

    assert any("holdout merchants" in e for e in report.errors)


def test_user_data_does_not_depend_on_other_users(small_spec: Spec, small_dataset: Dataset) -> None:
    """Changing another persona's user count must not change this user's data."""
    train, test = small_spec.populations
    fewer = train.model_copy(update={"users": {**train.users, "young_professional": 1}})
    other = generate(small_spec.model_copy(update={"populations": [fewer, test]}))

    user = "u_tr_fl_0003"
    a = small_dataset["transactions"].loc[lambda t: t["user_id"] == user].reset_index(drop=True)
    b = other["transactions"].loc[lambda t: t["user_id"] == user].reset_index(drop=True)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


def test_model_tables_carry_no_labels(small_dataset: Dataset) -> None:
    labels = {"category", "merchant_id", "process", "is_recurring", "anomaly_kind"}
    for name in MODEL_TABLES:
        assert not labels & set(small_dataset[name].columns), name
    assert set(TRUTH_TABLES) <= set(small_dataset.tables)
    assert "recurring" not in set(small_dataset["transactions"]["channel"])


def test_salaries_land_on_business_days(small_dataset: Dataset) -> None:
    tx = _ledger(small_dataset)
    pay = tx[(tx["process"] == "income") & tx["user_id"].str.contains("_yp_")]
    weekday = pd.to_datetime(pay["ts"]).dt.dayofweek

    assert len(pay) > 0
    assert (weekday < 5).all()


def test_refunds_and_duplicates_reuse_original_text(small_dataset: Dataset) -> None:
    tx = _ledger(small_dataset)
    refunds = tx[tx["process"] == "refund"]
    dupes = tx[tx["anomaly_kind"] == "duplicate"]

    assert len(refunds) > 0
    assert (refunds["amount"] > 0).all()
    assert (refunds["category"] == "Shopping").all()
    assert len(dupes) > 0
    for row in dupes.itertuples():
        same = tx[(tx["user_id"] == row.user_id) & (tx["amount"] == row.amount)]
        assert (same["merchant_raw"] == row.merchant_raw).sum() >= 2


def test_goals_split_between_known_and_future_outcomes(
    small_dataset: Dataset, small_spec: Spec
) -> None:
    goals = small_dataset["goals"].merge(small_dataset["truth_goals"], on="goal_id")
    end = str(small_spec.calendar.end)
    inside = goals[goals["target_date"] <= end]
    future = goals[goals["target_date"] > end]

    assert len(inside) > 0
    assert len(future) > 0
    assert inside["met"].notna().all()
    assert future["met"].isna().all()
    assert (goals["created_date"] <= goals["as_of_date"]).all()
    assert (future["as_of_date"] == end).all()


def test_goal_balance_is_reported_before_target(small_dataset: Dataset, small_spec: Spec) -> None:
    """The goals row alone can't decide `met`; the transactions after as_of_date still can."""
    goals = small_dataset["goals"].merge(small_dataset["truth_goals"], on="goal_id")
    known = goals[goals["met"].notna()]
    horizon = pd.to_datetime(known["target_date"]) - pd.to_datetime(known["as_of_date"])
    min_months = small_spec.goals.forecast_horizon_months[0]

    assert (horizon.dt.days >= min_months * 28).all()
    assert set(known["met"]) == {0, 1}
    trivial = known["current_balance"] >= known["target_amount"]
    assert (trivial == known["met"].astype(bool)).mean() < 0.9


def test_new_merchant_charges_are_at_merchants_the_user_never_used(small_dataset: Dataset) -> None:
    tx = _ledger(small_dataset)
    novel = tx[tx["anomaly_kind"] == "new_merchant_large"]

    assert len(novel) > 0
    assert not novel.duplicated(["user_id", "merchant_id"]).any()
    for row in novel.itertuples():
        others = tx[(tx["user_id"] == row.user_id) & (tx["transaction_id"] != row.transaction_id)]
        assert row.merchant_id not in set(others["merchant_id"])


@pytest.mark.slow
def test_default_spec_meets_runtime_budget(configs: Path) -> None:
    spec = load_spec(configs / "default.yaml")
    started = time.perf_counter()
    dataset = generate(spec)
    elapsed = time.perf_counter() - started

    assert elapsed < 60
    assert validate(dataset, spec).ok


def _without_spikes(spec: Spec) -> Spec:
    spike = spec.events.spending_spike.model_copy(update={"rate_per_user_year": 0.0})
    events = spec.events.model_copy(update={"spending_spike": spike})
    return spec.model_copy(update={"events": events})


def test_spikes_leave_normal_purchases_unchanged(small_spec: Spec, small_dataset: Dataset) -> None:
    """Spike extras are a separate process: turning spikes off changes no normal purchase."""
    columns = ["transaction_id", "user_id", "ts", "amount", "merchant_id", "category"]
    with_spikes = _ledger(small_dataset)
    without = _ledger(generate(_without_spikes(small_spec)))
    normal = without[without["process"] == "discretionary"][columns]
    spiked = with_spikes[with_spikes["process"] == "discretionary"][columns]

    assert len(small_dataset["truth_periods"]) > 0
    assert len(spiked) > len(normal)
    kept = normal.merge(spiked, on=columns, how="left", indicator=True)
    assert (kept["_merge"] == "both").all()


def test_spike_rows_record_realized_effect(small_dataset: Dataset) -> None:
    spikes = small_dataset["truth_periods"]
    base = spikes["base_spend"] / spikes["expected_spend"]
    total = (spikes["base_spend"] + spikes["extra_spend"]) / spikes["expected_spend"]

    assert spikes["spike_id"].is_unique
    assert (spikes["extra_spend"] >= 0).all()
    # Normal spending in a spike period is ordinary; the extras lift it (superposition)
    assert 0.8 <= base.median() <= 1.2
    assert total.median() > 1.3
    weak_lift = json.loads(small_dataset.meta["label_contract"])["weak_lift"]
    clear_cut = (total - weak_lift).abs() > 1e-3  # stored spends are rounded to cents
    assert ((spikes["tier"] == "weak") == (total < weak_lift))[clear_cut].all()


def test_related_transactions_are_recorded(small_dataset: Dataset) -> None:
    truth = small_dataset["truth_transactions"]
    linked = truth["process"].eq("refund") | truth["anomaly_kind"].eq("duplicate")

    assert truth.loc[linked, "related_transaction_id"].notna().all()
    assert truth.loc[~linked, "related_transaction_id"].isna().all()
    assert set(truth.loc[linked, "related_transaction_id"]) <= set(truth["transaction_id"])


def test_expected_spend_covers_spike_categories(small_dataset: Dataset) -> None:
    expected = small_dataset["truth_expected"]
    spikes = small_dataset["truth_periods"]

    assert (expected["granularity"] == "month").all()
    assert (expected["expected_count"] >= 0).all()
    spiked = set(zip(spikes["user_id"], spikes["category"], strict=True))
    assert spiked <= set(zip(expected["user_id"], expected["category"], strict=True))


def test_validation_catches_label_corruption(small_dataset: Dataset, small_spec: Spec) -> None:
    periods = small_dataset["truth_periods"].copy()
    periods["extra_spend"] = periods["extra_spend"].where(periods.index != 0, 999_999.0)
    corrupted = Dataset(
        tables={**small_dataset.tables, "truth_periods": periods}, meta=small_dataset.meta
    )

    report = validate(corrupted, small_spec)

    assert any("doesn't match ledger" in e for e in report.errors)


def test_validation_gates_on_the_oracle(small_dataset: Dataset, small_spec: Spec) -> None:
    contract = json.loads(small_dataset.meta["label_contract"])
    meta = {
        **small_dataset.meta,
        "label_contract": json.dumps({**contract, "oracle_min_precision": 1.01}),
    }

    report = validate(Dataset(tables=small_dataset.tables, meta=meta), small_spec)

    assert report.stats["oracle_month_precision"] >= 0.7
    assert any(e.startswith("ceiling: monthly spike oracle") for e in report.errors)
    assert any(e.startswith("ceiling: unusual-charge oracle") for e in report.errors)


def test_validation_catches_purchases_off_their_expected_counts(
    small_dataset: Dataset, small_spec: Spec
) -> None:
    expected = small_dataset["truth_expected"].copy()
    expected["expected_count"] *= 1.5
    skewed = Dataset(
        tables={**small_dataset.tables, "truth_expected": expected}, meta=small_dataset.meta
    )

    report = validate(skewed, small_spec)

    assert any("don't fit their expected counts" in e for e in report.errors)


def test_unusual_charges_carry_a_tier(small_dataset: Dataset) -> None:
    truth = small_dataset["truth_transactions"]
    unusual = truth["anomaly_kind"].notna()

    assert truth.loc[unusual, "tier"].isin(["clear", "weak"]).all()
    assert truth.loc[~unusual, "tier"].isna().all()
    # Only amount outliers can be weak; duplicates and new-merchant charges are always visible
    weak = truth[truth["tier"] == "weak"]
    assert (weak["anomaly_kind"] == "amount_outlier").all()
