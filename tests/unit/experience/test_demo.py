"""The demo bundle: the accounts' rows only, categorized, and readable by the serving user."""

import json
import stat
from pathlib import Path

import pytest
import yaml

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.data import store
from smart_financial_coach.data.flags import load_flags
from smart_financial_coach.data.predictions import CategoryWriter
from smart_financial_coach.experience import demo
from smart_financial_coach.intelligence.categorization.batch import BatchRun
from smart_financial_coach.intelligence.forecasting.states import load_forecasts
from tests.unit.conftest import STUB_META, stub_categories


def fake_categorize(data: Path, out: Path, **_: object) -> BatchRun:
    """Write predictions the way the real batch does (through an owner-only temporary file)."""
    txns = store.load_transactions(data)
    with CategoryWriter(out, STUB_META, overwrite=True) as writer:
        writer.append(txns["user_id"], stub_categories(txns))
    return BatchRun("stub", len(txns), 0.0, 0.0)


def test_the_bundle_holds_only_the_accounts_and_everyone_can_read_it(
    small_sqlite: Path,
    two_users: tuple[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounts = tmp_path / "accounts.yaml"
    accounts.write_text(
        yaml.safe_dump(
            {"accounts": [{"user_id": two_users[0], "name": "Maya Chen", "email": "m@x.com"}]}
        )
    )
    monkeypatch.setattr(demo, "categorize_dataset", fake_categorize)

    # No FR-7 model in this artifacts folder: no flag file
    bundle = demo.build_demo(small_sqlite, accounts, tmp_path / "demo", artifacts_dir=tmp_path)

    assert set(store.load_users(bundle.root / "dataset.sqlite")["user_id"]) == {two_users[0]}
    assert bundle.flag_model_version is None
    assert not (bundle.root / "flags.sqlite").exists()
    # Nor a goal-forecasting one: naive pace stands in (owner decision 10 on #54)
    assert bundle.forecast_model_version == "naive-pace"
    assert (
        load_forecasts(bundle.root / "forecasts.json").model["name"]
        == "goal_forecasting/naive_pace"
    )
    # Nor a spike model: the simple rule serves spikes (owner decision 12 on #58), and its season
    # profiles come from every user in the data, not just the bundle's accounts
    assert (bundle.spike_model_version, bundle.spike_method) == ("simple-rule", "simple_rule")
    spikes = json.loads((bundle.root / "spikes.json").read_text())
    assert spikes["meta"]["pool_users"] == len(store.load_users(small_sqlite))
    assert Ledger.load(DataSources.from_dir(bundle.root), two_users[0]).spikes is not None
    assert bundle.transactions == len(store.load_transactions(small_sqlite, user_id=two_users[0]))
    for path in bundle.root.iterdir():
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o044 == 0o044, f"{path.name} is {oct(mode)}"


def test_the_bundle_carries_flags_once_an_fr7_model_is_promoted(
    small_sqlite: Path,
    two_users: tuple[str, str],
    flag_artifacts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounts = tmp_path / "accounts.yaml"
    accounts.write_text(
        yaml.safe_dump(
            {"accounts": [{"user_id": two_users[0], "name": "Maya Chen", "email": "m@x.com"}]}
        )
    )
    monkeypatch.setattr(demo, "categorize_dataset", fake_categorize)

    bundle = demo.build_demo(
        small_sqlite, accounts, tmp_path / "demo", artifacts_dir=flag_artifacts
    )

    assert bundle.flag_model_version == "fr7-test"
    flags = load_flags(bundle.root / "flags.sqlite")
    assert set(flags["user_id"]) <= {two_users[0]}
    assert bundle.flags == len(flags)


def test_the_bundle_carries_forecast_states_once_a_goal_model_is_promoted(
    small_sqlite: Path,
    two_users: tuple[str, str],
    forecast_artifacts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounts = tmp_path / "accounts.yaml"
    accounts.write_text(
        yaml.safe_dump(
            {"accounts": [{"user_id": two_users[0], "name": "Maya Chen", "email": "m@x.com"}]}
        )
    )
    monkeypatch.setattr(demo, "categorize_dataset", fake_categorize)

    bundle = demo.build_demo(
        small_sqlite, accounts, tmp_path / "demo", artifacts_dir=forecast_artifacts
    )

    assert bundle.forecast_model_version == "fr11-test"
    stored = load_forecasts(bundle.root / "forecasts.json")
    assert set(stored.states) == {two_users[0]}  # the accounts only
    assert DataSources.from_dir(bundle.root).forecasts == bundle.root / "forecasts.json"
    assert Ledger.load(DataSources.from_dir(bundle.root), two_users[0]).forecaster is not None


def test_the_bundle_serves_a_promoted_spike_model(
    small_sqlite: Path,
    two_users: tuple[str, str],
    spike_artifacts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounts = tmp_path / "accounts.yaml"
    accounts.write_text(
        yaml.safe_dump(
            {"accounts": [{"user_id": two_users[0], "name": "Maya Chen", "email": "m@x.com"}]}
        )
    )
    monkeypatch.setattr(demo, "categorize_dataset", fake_categorize)

    bundle = demo.build_demo(
        small_sqlite, accounts, tmp_path / "demo", artifacts_dir=spike_artifacts
    )

    assert (bundle.spike_model_version, bundle.spike_method) == ("fr8-test", "model")
