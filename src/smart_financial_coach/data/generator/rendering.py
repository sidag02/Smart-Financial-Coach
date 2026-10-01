"""Stage 8: turn canonical merchant names into messy bank-feed strings.

Each merchant gets a few stable "house styles" (shared by all users, seeded by merchant ID), so
the same shop looks similar most of the time but not always. Per-transaction noise (reference
codes, typos, spacing) is drawn from the user's rendering seed.
"""

import string
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.taxonomy import City

ALNUM = np.array(list(string.ascii_uppercase + string.digits))
PROCESSOR_PREFIX = {"square": "SQ *", "toast": "TST* ", "clover": "CLV*", "paypal": "PAYPAL *"}


@dataclass(frozen=True)
class Profile:
    """Probabilities that a house style (or a transaction) carries each distortion."""

    p_upper: float
    p_descriptor: float
    p_processor_prefix: float
    p_store: float
    p_location: float
    p_short_location: float
    p_channel_prefix: float
    p_ref: float
    p_drop_apostrophe: float
    p_typo: float
    p_double_space: float
    limit: tuple[int, int]
    n_styles: tuple[int, int]


PROFILES = {
    "light": Profile(0.9, 0.5, 0.6, 0.3, 0.2, 0.2, 0.0, 0.2, 0.3, 0.0, 0.0, (32, 40), (1, 2)),
    "realistic": Profile(
        0.9, 0.85, 0.95, 0.7, 0.6, 0.4, 0.25, 0.7, 0.6, 0.01, 0.02, (22, 32), (2, 4)
    ),
    "heavy": Profile(0.95, 0.9, 1.0, 0.85, 0.8, 0.6, 0.4, 0.9, 0.8, 0.04, 0.06, (18, 26), (3, 5)),
}


@dataclass(frozen=True)
class Style:
    upper: bool
    descriptor: bool
    processor_prefix: bool
    store_format: str  # "", "#{n}", " {n:04d}", " STORE {n:05d}"
    location: str  # "", "full", "short"
    channel_prefix: bool
    ref: bool
    drop_apostrophe: str  # how apostrophes render: "", " ", or "'"
    ach_form: int
    limit: int


class Renderer:
    def __init__(self, catalog: Catalog, distortion: str, seed: int) -> None:
        self.catalog = catalog
        self.profile = PROFILES.get(distortion)
        self.seed = seed
        self._rows: dict[str, dict[str, Any]] = {
            str(mid): {str(k): v for k, v in row.items()}
            for mid, row in catalog.merchants.to_dict("index").items()
        }
        self._styles: dict[str, tuple[list[Style], np.ndarray, int]] = {}

    def _house_styles(self, merchant_id: str) -> tuple[list[Style], np.ndarray, int]:
        if (cached := self._styles.get(merchant_id)) is not None:
            return cached
        assert self.profile is not None
        p = self.profile
        rng = np.random.default_rng(
            np.random.SeedSequence(self.seed, spawn_key=(zlib.crc32(merchant_id.encode()),))
        )
        n = int(rng.integers(p.n_styles[0], p.n_styles[1] + 1))
        styles = [
            Style(
                upper=bool(rng.random() < p.p_upper),
                descriptor=bool(rng.random() < p.p_descriptor),
                processor_prefix=bool(rng.random() < p.p_processor_prefix),
                store_format=str(rng.choice(["#{n}", " {n:04d}", " #{n}", " STORE {n:05d}"]))
                if rng.random() < p.p_store
                else "",
                location=("short" if rng.random() < p.p_short_location else "full")
                if rng.random() < p.p_location
                else "",
                channel_prefix=bool(rng.random() < p.p_channel_prefix),
                ref=bool(rng.random() < p.p_ref),
                drop_apostrophe=str(rng.choice([" ", ""]))
                if rng.random() < p.p_drop_apostrophe
                else "'",
                ach_form=int(rng.integers(0, 3)),
                limit=int(rng.integers(p.limit[0], p.limit[1] + 1)),
            )
            for _ in range(n)
        ]
        weights = np.sort(rng.dirichlet(np.full(n, 0.8)))[::-1]  # one dominant style
        ppd = int(rng.integers(1000, 10000))
        self._styles[merchant_id] = (styles, weights, ppd)
        return self._styles[merchant_id]

    def render_user(self, txns: pd.DataFrame, city: City, rng: np.random.Generator) -> list[str]:
        """Render merchant_raw for one user's transactions, in the frame's row order."""
        if self.profile is None:
            return [str(self._rows[mid]["canonical_name"]) for mid in txns["merchant_id"]]
        stores: dict[str, list[int]] = {}
        out: list[str] = []
        for mid, channel, process in txns[["merchant_id", "channel", "process"]].itertuples(
            index=False, name=None
        ):
            row = self._rows[mid]
            name = str(row["canonical_name"])
            styles, weights, ppd = self._house_styles(mid)
            style = (
                styles[int(rng.choice(len(styles), p=weights))] if len(styles) > 1 else styles[0]
            )
            if row["category"] == "Income":
                text = self._income(name, row, style, ppd, rng)
            elif channel == "ach":
                text = self._ach(name, row, style, ppd)
            else:
                text = self._card(
                    mid, name, row, style, channel, process == "recurring", city, stores, rng
                )
            out.append(self._noise(text, style.limit, rng))
        return out

    # --- forms ------------------------------------------------------------------------------

    @staticmethod
    def _name(name: str, style: Style) -> str:
        text = name.replace("'", style.drop_apostrophe).replace("\u2019", style.drop_apostrophe)
        return text.upper() if style.upper else text

    @staticmethod
    def _ref(rng: np.random.Generator, n: int) -> str:
        return "".join(rng.choice(ALNUM, size=n))

    def _income(
        self, name: str, row: dict[str, Any], style: Style, ppd: int, rng: np.random.Generator
    ) -> str:
        upper = name.upper().replace("'", "")
        if row["subtype"] == "client":
            processor = row["processor"]
            if processor == "stripe":
                return f"STRIPE TRANSFER ST-{self._ref(rng, 8)}"
            if processor == "paypal":
                return f"PAYPAL TRANSFER {self._ref(rng, 10)}"
            if processor == "zelle":
                return f"ZELLE FROM {upper}"
            return (f"ACH CREDIT {upper}", f"{upper} ACH PMT", f"{upper} INVOICE")[style.ach_form]
        return (f"{upper} PAYROLL PPD ID: {ppd}", f"{upper} DIR DEP", f"DIRECT DEP {upper}")[
            style.ach_form
        ]

    def _ach(self, name: str, row: dict[str, Any], style: Style, ppd: int) -> str:
        base = (
            row["descriptor"]
            if (row["descriptor"] and style.descriptor)
            else name.upper().replace("'", "")
        )
        if row["processor"] == "zelle":
            return f"ZELLE TO {base}"
        return (f"{base} WEB PMT", f"ACH DEBIT {base}", f"{base} AUTOPAY {ppd}")[style.ach_form]

    def _card(
        self,
        mid: str,
        name: str,
        row: dict[str, Any],
        style: Style,
        channel: str,
        recurring: bool,
        city: City,
        stores: dict[str, list[int]],
        rng: np.random.Generator,
    ) -> str:
        descriptor: str = row["descriptor"]
        use_descriptor = bool(descriptor) and style.descriptor
        base = descriptor if use_descriptor else self._name(name, style)
        prefix = ""
        if style.processor_prefix and row["processor"] in PROCESSOR_PREFIX and not use_descriptor:
            prefix = PROCESSOR_PREFIX[row["processor"]]
        if style.channel_prefix and channel == "card_present":
            prefix = "POS DEBIT " + prefix
        text = prefix + base
        if recurring:  # subscriptions bill with the same text every month
            return text.rstrip("*")
        if base.endswith("*"):
            return text + self._ref(rng, int(rng.integers(5, 10)))
        national = channel == "card_present" and row["scope"] == "national"
        if national and (base.endswith(("#", "-")) or style.store_format):
            # Descriptors like "REI #" or "76 -" expect a store number right after them
            fmt = (
                "{n}"
                if base.endswith("#")
                else " {n}"
                if base.endswith("-")
                else style.store_format
            )
            if mid not in stores:
                stores[mid] = [int(n) for n in rng.integers(10, 9999, size=int(rng.integers(1, 3)))]
            text += fmt.format(n=stores[mid][int(rng.integers(len(stores[mid])))])
        if channel == "card_present" and style.location:
            city_name = city.name[:9].rstrip() if style.location == "short" else city.name
            text = f"{text} {city_name} {city.state}"
        elif channel == "online" and style.ref and row["scope"] == "online":
            text = f"{text}*{self._ref(rng, int(rng.integers(6, 10)))}"
        return text

    def _noise(self, text: str, limit: int, rng: np.random.Generator) -> str:
        assert self.profile is not None
        if rng.random() < self.profile.p_typo and len(text) > 6:
            at = int(rng.integers(3, len(text)))
            text = text[:at] + text[at + 1 :]
        if rng.random() < self.profile.p_double_space and " " in text:
            at = text.index(" ")
            text = text[:at] + " " + text[at:]
        return text[:limit].rstrip()
