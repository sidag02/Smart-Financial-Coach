"""Schema 4 preferences: the FR-5/FR-6 data contract (accepted on PR #15)."""

from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.data.generator import Dataset, Spec, generate
from smart_financial_coach.data.generator.preferences import adopted
from smart_financial_coach.data.generator.spec import PreferencesSpec, RemapSpec
from smart_financial_coach.data.generator.sqlite_io import read_sqlite
from smart_financial_coach.data.generator.validate import Report, _preferences
from smart_financial_coach.data.labels import Truth


def _subtypes(ds: Dataset) -> pd.Series:
    return ds["truth_merchants"].set_index("merchant_id")["subtype"]


def test_only_test_users_hold_preferences(small_dataset: Dataset) -> None:
    prefs = small_dataset["truth_preferences"]
    split = small_dataset["users"].set_index("user_id")["split"]

    assert not prefs.empty
    assert set(prefs["user_id"].map(split)) == {"test"}


def test_a_preference_covers_its_whole_subtype_with_the_remaps_category(
    small_spec: Spec, small_dataset: Dataset
) -> None:
    prefs = small_dataset["truth_preferences"]
    subtypes = _subtypes(small_dataset)
    remaps = {r.subtype: r.category for r in small_spec.preferences.remaps}
    prefs = prefs.assign(subtype=prefs["merchant_id"].map(subtypes))

    assert (prefs["category"] == prefs["subtype"].map(remaps)).all()
    sizes = subtypes.value_counts()
    held = prefs.groupby(["user_id", "subtype"]).size()
    expected_sizes = [sizes[t] for t in held.index.get_level_values("subtype")]
    assert held.tolist() == expected_sizes


def test_preferences_change_no_other_row(small_spec: Spec, small_dataset: Dataset) -> None:
    """Drawn last, from their own seed: removing them leaves every other table identical."""
    without = generate(small_spec.model_copy(update={"preferences": PreferencesSpec()}))

    assert without["truth_preferences"].empty
    for name, table in small_dataset.tables.items():
        if name != "truth_preferences":
            pd.testing.assert_frame_equal(table, without[name], check_dtype=False)


def test_a_users_draw_depends_only_on_the_seed_and_their_id() -> None:
    spec = PreferencesSpec(
        seed=13,
        remaps=[RemapSpec(subtype=s, category="Shopping", share=0.5) for s in ("gym", "books")],
    )

    assert adopted("u_te_yp_0001", spec) == adopted("u_te_yp_0001", spec)
    draws = [adopted(f"u_te_yp_{i:04d}", spec) for i in range(400)]
    share = sum(d[0] for d in draws) / len(draws)
    assert 0.4 < share < 0.6
    assert adopted("u_te_yp_0001", spec.model_copy(update={"seed": 14})) != adopted(
        "u_te_yp_0001", spec
    ) or adopted("u_te_yp_0002", spec.model_copy(update={"seed": 14})) != adopted(
        "u_te_yp_0002", spec
    )


def test_user_categories_use_the_preference_else_the_true_category(
    small_dataset: Dataset,
) -> None:
    truth = Truth.from_dataset(small_dataset)
    mine = truth.user_categories()
    tx = truth.transactions
    keys = set(zip(truth.preferences["user_id"], truth.preferences["merchant_id"], strict=True))
    preferred = [(u, m) in keys for u, m in zip(tx["user_id"], tx["merchant_id"], strict=True)]
    preferred_mask = pd.Series(preferred, index=tx.index)

    assert preferred_mask.any()
    assert (mine[~preferred_mask] == tx.loc[~preferred_mask, "category"]).all()
    lookup = truth.preferences.set_index(["user_id", "merchant_id"])["category"]
    expected = [
        lookup[(u, m)] for u, m in tx.loc[preferred_mask, ["user_id", "merchant_id"]].values
    ]
    assert mine[preferred_mask].tolist() == expected


def test_validation_catches_a_train_user_preference(
    small_spec: Spec, small_dataset: Dataset
) -> None:
    users = small_dataset["users"]
    train_user = users.loc[users["split"] == "train", "user_id"].iloc[0]
    leaked = pd.concat(
        [
            small_dataset["truth_preferences"],
            small_dataset["truth_preferences"].head(1).assign(user_id=train_user),
        ],
        ignore_index=True,
    )
    report = Report()
    _preferences(
        Dataset(tables={**small_dataset.tables, "truth_preferences": leaked}, meta={}),
        small_spec,
        report,
    )

    assert any("not for test users" in e for e in report.errors)


def test_spec_rejects_unknown_categories_and_repeated_subtypes() -> None:
    from smart_financial_coach.data.generator.spec import load_spec

    spec = load_spec(Path(__file__).parents[4] / "configs" / "data" / "small.yaml")
    raw = spec.model_dump()
    for remaps, message in (
        ([{"subtype": "gym", "category": "Nope", "share": 0.3}], "unknown category"),
        (
            [{"subtype": "gym", "category": "Shopping", "share": 0.3}] * 2,
            "remapped once",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            Spec.model_validate({**raw, "preferences": {"seed": 1, "remaps": remaps}})


def test_reading_an_older_schema_fails_with_a_way_forward(
    small_sqlite: Path, tmp_path: Path
) -> None:
    import shutil
    import sqlite3

    old = tmp_path / "old.sqlite"
    shutil.copy(small_sqlite, old)
    with sqlite3.connect(old) as conn:
        conn.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")

    with pytest.raises(ValueError, match="data/fr3-default"):
        read_sqlite(old)
