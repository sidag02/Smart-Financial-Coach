"""The merchant-text normalizer: every rendering distortion, stacking, idempotence."""

import sqlite3
from pathlib import Path

import pytest

from smart_financial_coach.data.features.merchant_text import (
    normalize_merchant,
    normalize_merchant_poc,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SQ *BLUE BOTTLE COFFEE", "blue bottle coffee"),  # processor prefix
        ("TST* HOPS & BARLEY TAP", "hops & barley tap"),
        ("CLV*LITTLE OWL ESPRESSO", "little owl espresso"),
        ("PAYPAL *ETSY", "etsy"),
        ("POS DEBIT SQ *PHO SAIGON NOODLE", "pho saigon noodle"),  # stacked prefixes
        ("POS DEBIT TST* VELVET LOU", "velvet lou"),  # stacked and truncated
        ("REI #1234 DENVER CO", "rei"),  # store number, and the location after it
        ("STARBUCKS STORE 01234", "starbucks"),
        ("TARGET 00012345", "target"),  # bare store number
        ("AMAZON MKTPL*1A2B3C", "amazon mktpl"),  # reference code
        ("ACME CORP PAYROLL PPD ID: 4455", "acme corp payroll"),  # ACH originator ID
        ("ACH DEBIT GEICO AUTOPAY 9911", "geico autopay"),
        ("ACH  DEBIT COMED", "comed"),  # double space inside a prefix
        ("Morning Ritual Cafe  Austin TX", "morning ritual cafe austin tx"),  # spacing, location
        ("#123", "#123"),  # nothing left: keep the raw text rather than return ""
    ],
)
def test_normalizes_each_distortion(raw: str, expected: str) -> None:
    assert normalize_merchant(raw) == expected


def test_idempotent_and_never_empty_on_generated_strings(small_sqlite: Path) -> None:
    with sqlite3.connect(small_sqlite) as conn:
        strings = [r[0] for r in conn.execute("SELECT DISTINCT merchant_raw FROM transactions")]
    normalized = [normalize_merchant(s) for s in strings]

    assert all(normalized)
    assert all(normalize_merchant(n) == n for n in normalized)


def test_poc_variant_keeps_the_pocs_stacked_prefix_behavior() -> None:
    assert normalize_merchant_poc("POS DEBIT SQ *PHO SAIGON") == "sq"
    assert normalize_merchant_poc("SQ *PHO SAIGON") == normalize_merchant("SQ *PHO SAIGON")
