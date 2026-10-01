"""Merchant text normalization, shared by categorization and the anomaly service.

    normalize_merchant("POS DEBIT SQ *PHO SAIGON NOODLE")  # -> "pho saigon noodle"
    normalize_merchant("TARGET 00012345 AUSTIN TX")         # -> "target austin tx"

The prefix list is generic bank-feed vocabulary (card processors, channel words), written by hand
from the rendering rules' types and never read from the merchant catalog, so no holdout names
leak in. Trailing locations are kept unless they follow a store number.
"""

import re

# Processor and channel prefixes. They can stack ("POS DEBIT SQ *"), so they're stripped repeatedly.
PREFIX = re.compile(
    r"^(POS DEBIT|ACH DEBIT|ACH CREDIT|SQ \*|TST\* ?|CLV\*|PAYPAL \*|SP \* ?)\s*", re.IGNORECASE
)
REFERENCE = re.compile(r"\*.*$")  # "AMAZON MKTPL*1A2B3C" -> "AMAZON MKTPL"
PPD = re.compile(r"\bPPD ID:?", re.IGNORECASE)  # ACH originator ID label
STORE_NUMBER = re.compile(r"(#|STORE)\s*\d+.*$", re.IGNORECASE)  # and anything after it
DIGITS = re.compile(r"\d+")
SPACES = re.compile(r"\s+")


def _finish(text: str, *, ppd: bool = True) -> str:
    text = REFERENCE.sub("", text)
    if ppd:
        text = PPD.sub("", text)
    text = STORE_NUMBER.sub("", text)
    text = DIGITS.sub(" ", text)
    return SPACES.sub(" ", text).strip().lower()


def normalize_merchant(raw: str) -> str:
    """Lower-case merchant identity without prefixes, reference codes and numbers.

    Idempotent, and never empty for non-empty input: if nothing is left, the raw text is kept
    (lower-cased, whitespace collapsed).
    """
    text = SPACES.sub(" ", raw).strip()  # bank feeds double spaces, even inside "ACH  DEBIT"
    while (stripped := PREFIX.sub("", text)) != text:
        text = stripped
    return _finish(text) or SPACES.sub(" ", raw).strip().lower()


def normalize_merchant_poc(raw: str) -> str:
    """The FR-3 POC's normalizer, kept only so the POC's numbers can be reproduced exactly.

    It strips a single prefix, so stacked prefixes collapse to the processor's name
    ("POS DEBIT SQ *PHO SAIGON" -> "sq"): 3.1% of the default dataset's transactions.
    """
    single = re.compile(
        r"^(POS DEBIT|ACH DEBIT|ACH CREDIT|SQ \*|TST\* ?|PAYPAL \*|SP \* ?)\s*", re.I
    )
    return _finish(single.sub("", raw), ppd=False)
