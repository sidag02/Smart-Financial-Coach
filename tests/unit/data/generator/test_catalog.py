from pathlib import Path

import numpy as np
import pytest

from smart_financial_coach.data.generator.catalog import load_catalog
from smart_financial_coach.data.generator.spec import HoldoutSpec, Spec


def test_holdout_split_protects_popular_merchants(small_spec: Spec) -> None:
    holdout = small_spec.catalog.holdout
    catalog = load_catalog(small_spec.catalog.merchants, holdout, small_spec.categories)
    m = catalog.merchants

    assert not m.loc[m["category"] == "Income", "holdout"].any()
    for category, rows in m[m["category"] != "Income"].groupby("category"):
        top = rows.sort_values(["popularity", "merchant_id"], ascending=[False, True]).head(
            holdout.exclude_top_n_per_category
        )
        assert not top["holdout"].any(), category
        assert 0 < rows["holdout"].sum() <= round(holdout.share * len(rows)), category
    # Every subtype keeps at least one merchant training users can see
    assert m.groupby("subtype")["holdout"].all().sum() == 0


def test_holdout_is_seeded(small_spec: Spec) -> None:
    a = load_catalog(small_spec.catalog.merchants, HoldoutSpec(seed=1), small_spec.categories)
    b = load_catalog(small_spec.catalog.merchants, HoldoutSpec(seed=1), small_spec.categories)
    c = load_catalog(small_spec.catalog.merchants, HoldoutSpec(seed=2), small_spec.categories)

    assert a.merchants["holdout"].equals(b.merchants["holdout"])
    assert not a.merchants["holdout"].equals(c.merchants["holdout"])


def test_ambiguous_merchants_draw_their_category(small_spec: Spec) -> None:
    catalog = load_catalog(
        small_spec.catalog.merchants, small_spec.catalog.holdout, small_spec.categories
    )
    ids = np.array(["m_costco"] * 2000)

    categories = catalog.sample_categories(ids, np.random.default_rng(0))

    assert set(categories) == {"Groceries", "Shopping"}
    assert 0.55 < float(np.mean(categories == "Groceries")) < 0.65


def test_unknown_category_in_catalog_is_rejected(tmp_path: Path, small_spec: Spec) -> None:
    text = small_spec.catalog.merchants.read_text(encoding="utf-8")
    bad = tmp_path / "merchants.csv"
    bad.write_text(text.replace(",Dining,", ",Dinning,", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown categories"):
        load_catalog(bad, small_spec.catalog.holdout, small_spec.categories)


def test_holdout_eligible_is_everything_not_protected(small_spec: Spec) -> None:
    holdout = small_spec.catalog.holdout
    m = load_catalog(small_spec.catalog.merchants, holdout, small_spec.categories).merchants

    assert (~m["holdout"] | m["holdout_eligible"]).all()  # the holdout picks only eligible ones
    assert not m.loc[m["category"] == "Income", "holdout_eligible"].any()
    for _, rows in m[m["category"] != "Income"].groupby("category"):
        ordered = rows.sort_values(["popularity", "merchant_id"], ascending=[False, True])
        protected = set(ordered.index[: holdout.exclude_top_n_per_category])
        protected |= set(ordered.groupby("subtype", sort=False).head(1).index)
        assert set(rows.index[~rows["holdout_eligible"]]) == protected
