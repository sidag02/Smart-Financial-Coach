"""Round 0 baselines: behavior, contract, and the keyword file's no-merchant-names rule."""

import csv
import re

import pandas as pd
import pytest
import yaml

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.intelligence.categorization.baseline import (
    keywords_path,
    keywords_sha256,
)
from smart_financial_coach.intelligence.categorization.contract import CONTRACT
from smart_financial_coach.intelligence.models import Checked, build

# Generic words that also occur in catalog names. Each was reviewed: it describes what a business
# sells or is, not which business it is, so it would be a fair keyword on real bank data too.
ALLOWED_GENERIC = {
    "airlines", "air", "apartments", "auto", "bakery", "bar", "bistro", "books", "boutique",
    "cafe", "camp", "cantina", "cinemas", "coffee", "comedy", "dental", "dumpling", "elementary",
    "energy", "espresso", "fee", "fitness", "gas", "golf", "grill", "grocery", "hardware",
    "homes", "insurance", "internet", "kitchen", "learning", "lending", "lines", "living",
    "lofts", "lounge", "market", "membership", "mercado", "mgmt", "mobile", "mortgage",
    "museum", "mutual", "noodle", "oil", "optometry", "overdraft", "parking", "pediatrics",
    "pharmacy", "power", "premium", "preschool", "produce", "properties", "pub", "rentals",
    "renters", "residential", "school", "shop", "store", "tavern", "tax", "taproom", "theatres",
    "thrift", "tire", "toy", "water", "wholesale", "wireless", "yoga", "farmers", "daycare",
    "montessori", "fiber", "arcade", "bowling", "lube", "foods", "hoa", "rent", "burger", "taco",
    "eats", "inn", "car", "a", "ride", "medical", "fees", "utility", "property",
}  # fmt: skip

KEYWORD = {"type": "categorization/keyword", "params": {"keywords_sha256": keywords_sha256()}}

TX = pd.DataFrame(
    {
        "transaction_id": ["1", "2", "3", "4", "5"],
        "user_id": "u",
        "ts": "2026-01-01T10:00:00",
        "amount": [-4.5, 12.0, 2500.0, -30.0, -80.0],
        "currency": "USD",
        "merchant_raw": [
            "SQ *DRIFTWOOD COFFEE CO",
            "MAPLE STREET BOOKS",
            "ACME CORP PAYROLL PPD ID: 1",
            "ZXQV",
            "SHELL OIL 1234",
        ],
        "channel": ["card_present", "online", "ach", "card_present", "card_present"],
    }
)
Y = pd.Series(["Dining", "Shopping", "Income", "Dining", "Transportation"])


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_keywords_are_not_merchant_names() -> None:
    with (PROJECT_ROOT / "configs" / "data" / "merchants.csv").open() as f:
        names = set().union(*(_tokens(r["canonical_name"]) for r in csv.DictReader(f)))
    keywords = yaml.safe_load(keywords_path().read_text())
    words = set().union(*(_tokens(w) for ws in keywords.values() for w in ws))

    assert sorted(words & names - ALLOWED_GENERIC) == []


def test_keyword_rules() -> None:
    model = build(KEYWORD).fit(TX, Y)
    out = Checked(model, CONTRACT).predict(TX)

    assert out["category"].tolist() == [
        "Dining",  # "coffee"
        "Shopping",  # "books": a refund keeps its category, the sign doesn't make it Income
        "Income",  # "payroll"
        "Dining",  # no keyword, money out: the most frequent training spending category
        "Transportation",  # "oil"
    ]


def test_keyword_confidence_is_the_rules_training_precision() -> None:
    y = Y.copy()
    y.iloc[3] = "Travel"  # the fallback (Dining, first of a four-way tie) is now wrong on its row
    out = build(KEYWORD).fit(TX, y).predict(TX)

    assert out.loc[3, "confidence"] == 0.0
    assert out.loc[0, "confidence"] == 1.0


def test_lookup_memorizes_normalized_strings() -> None:
    model = build({"type": "categorization/lookup"}).fit(TX, Y)
    seen = TX.assign(merchant_raw=["DRIFTWOOD COFFEE CO", "X", "Y", "Z", "SHELL OIL 9"])
    out = model.predict(seen)

    assert out["category"].tolist()[0] == "Dining"  # same string once prefixes are stripped
    assert out["category"].tolist()[4] == "Transportation"  # store number differs
    assert out.loc[1, "category"] == "Dining"  # never seen: the majority category
    assert out.loc[0, "confidence"] == 1.0
    assert out["familiar"].tolist() == [True, False, False, False, True]


@pytest.mark.parametrize("kind", ["majority", "keyword", "lookup"])
def test_familiar_means_the_normalized_string_was_in_training(kind: str) -> None:
    spec = KEYWORD if kind == "keyword" else {"type": f"categorization/{kind}"}
    model = build(spec).fit(TX, Y)
    out = model.predict(TX.assign(merchant_raw=["POS DEBIT DRIFTWOOD COFFEE CO", *["NEW"] * 4]))

    assert out["familiar"].tolist() == [True, False, False, False, False]


@pytest.mark.parametrize("kind", ["majority", "keyword", "lookup"])
def test_baselines_meet_the_contract_and_are_deterministic(kind: str) -> None:
    spec = KEYWORD if kind == "keyword" else {"type": f"categorization/{kind}"}
    a = Checked(build(spec).fit(TX, Y), CONTRACT).predict(TX)
    b = Checked(build(spec).fit(TX, Y), CONTRACT).predict(TX)

    assert a.equals(b)


def test_supervised_models_need_labels() -> None:
    with pytest.raises(ValueError, match="needs labels"):
        build({"type": "categorization/lookup"}).fit(TX)


def test_keyword_file_content_is_part_of_the_config() -> None:
    with pytest.raises(TypeError, match="keywords_sha256"):  # required: no silent opt-out
        build({"type": "categorization/keyword"})
    with pytest.raises(ValueError, match="Update the config's keywords_sha256"):
        build({"type": "categorization/keyword", "params": {"keywords_sha256": "0" * 64}})


def test_committed_keyword_config_matches_the_file() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiments/categorization/00_keyword.yaml").read_text()
    )
    build(config["model"])  # raises if the file changed without the config
