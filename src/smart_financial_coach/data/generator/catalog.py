"""Merchant catalog: loading, validation, the seeded holdout split, and per-merchant sampling."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.data.generator.spec import HoldoutSpec
from smart_financial_coach.data.generator.taxonomy import CHANNELS, PROCESSORS

REQUIRED_COLUMNS = (
    "merchant_id",
    "canonical_name",
    "category",
    "subtype",
    "scope",
    "processor",
    "channel_mix",
    "price_median",
    "price_sigma",
    "peak_hour",
    "popularity",
    "descriptor",
    "ambiguous_categories",
)
SCOPES = ("national", "local", "online")


def _parse_mix(
    text: str, *, what: str, allowed: set[str] | None
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Parse `a=0.6;b=0.4` into names and probabilities that sum to one."""
    names: list[str] = []
    weights: list[float] = []
    for part in text.split(";"):
        name, _, weight = part.partition("=")
        name = name.strip()
        if allowed is not None and name not in allowed:
            raise ValueError(f"unknown {what} {name!r} in {text!r}")
        names.append(name)
        weights.append(float(weight))
    probs = np.asarray(weights, dtype=np.float64)
    if (probs <= 0).any():
        raise ValueError(f"{what} weights must be positive in {text!r}")
    return names, probs / probs.sum()


@dataclass(frozen=True)
class Catalog:
    """Merchants indexed by merchant_id, with parsed channel and category mixes."""

    merchants: pd.DataFrame
    channels: dict[str, tuple[list[str], npt.NDArray[np.float64]]]
    categories: dict[str, tuple[list[str], npt.NDArray[np.float64]]]

    def price(self, merchant_id: str) -> float:
        return float(cast(float, self.merchants.at[merchant_id, "price_median"]))

    def select(self, category: str, subtypes: list[str], *, include_holdout: bool) -> pd.DataFrame:
        m = self.merchants
        mask = m["category"] == category
        if subtypes:
            mask &= m["subtype"].isin(subtypes)
        if not include_holdout:
            mask &= ~m["holdout"]
        return m[mask]

    def sample_channels(
        self, merchant_ids: npt.NDArray[np.str_], rng: np.random.Generator
    ) -> npt.NDArray[np.str_]:
        out = np.empty(len(merchant_ids), dtype=object)
        for mid in np.unique(merchant_ids):
            idx = np.flatnonzero(merchant_ids == mid)
            names, probs = self.channels[str(mid)]
            out[idx] = names[0] if len(names) == 1 else rng.choice(names, size=len(idx), p=probs)
        return out.astype(str)

    def sample_categories(
        self, merchant_ids: npt.NDArray[np.str_], rng: np.random.Generator
    ) -> npt.NDArray[np.str_]:
        """True category per transaction; ambiguous merchants draw it from their mix."""
        out = self.merchants["category"].reindex(merchant_ids).to_numpy(dtype=object)
        for mid in np.unique(merchant_ids):
            if (mix := self.categories.get(str(mid))) is None:
                continue
            idx = np.flatnonzero(merchant_ids == mid)
            out[idx] = rng.choice(mix[0], size=len(idx), p=mix[1])
        return out.astype(str)

    def sample_minutes(
        self, merchant_ids: npt.NDArray[np.str_], rng: np.random.Generator
    ) -> npt.NDArray[np.int64]:
        """Minute of day around each merchant's peak hour; online merchants spread wider."""
        m = self.merchants.reindex(merchant_ids)
        peak = m["peak_hour"].to_numpy(dtype=np.float64)
        spread = np.where(m["scope"].to_numpy() == "online", 4.0, 1.75)
        hours = np.clip(rng.normal(peak, spread), 0.0, 23.99)
        return (hours * 60).astype(np.int64)


def load_catalog(path: Path, holdout: HoldoutSpec, categories: list[str]) -> Catalog:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if missing := [c for c in REQUIRED_COLUMNS if c not in raw.columns]:
        raise ValueError(f"{path}: missing columns {missing}")
    errors: list[str] = []
    if raw["merchant_id"].duplicated().any():
        dupes = sorted(raw.loc[raw["merchant_id"].duplicated(), "merchant_id"])
        errors.append(f"duplicate merchant_id: {dupes}")
    if bad := sorted(set(raw["category"]) - set(categories)):
        errors.append(f"unknown categories {bad}")
    if bad := sorted(set(raw["scope"]) - set(SCOPES)):
        errors.append(f"unknown scopes {bad}")
    if bad := sorted(set(raw["processor"]) - set(PROCESSORS)):
        errors.append(f"unknown processors {bad}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))

    m = raw.set_index("merchant_id", drop=False).sort_index()
    m.index.name = None
    m["price_median"] = m["price_median"].astype(float)
    m["price_sigma"] = m["price_sigma"].astype(float)
    m["peak_hour"] = m["peak_hour"].astype(float)
    m["popularity"] = m["popularity"].astype(float)
    m["is_ambiguous"] = m["ambiguous_categories"] != ""

    channels = {
        str(mid): _parse_mix(mix, what="channel", allowed=set(CHANNELS))
        for mid, mix in m["channel_mix"].items()
    }
    ambiguous = {
        str(mid): _parse_mix(mix, what="category", allowed=set(categories))
        for mid, mix in m.loc[m["is_ambiguous"], "ambiguous_categories"].items()
    }
    m["holdout"] = _holdout(m, holdout)
    return Catalog(merchants=m, channels=channels, categories=ambiguous)


def _holdout(m: pd.DataFrame, spec: HoldoutSpec) -> pd.Series:
    """Hold out a seeded share of each category's merchants.

    Never held out: the category's most popular merchants, and the most popular merchant of each
    subtype, so training users can always find a merchant for every subtype.
    """
    rng = np.random.default_rng(np.random.SeedSequence(spec.seed))
    held = pd.Series(False, index=m.index)
    for category in sorted(m["category"].unique()):
        if category in spec.exclude_categories:
            continue
        rows = m[m["category"] == category].sort_values(
            ["popularity", "merchant_id"], ascending=[False, True]
        )
        protected = set(rows.index[: spec.exclude_top_n_per_category])
        protected |= set(rows.groupby("subtype", sort=False).head(1).index)
        eligible = [mid for mid in rows.index if mid not in protected]
        n = min(round(spec.share * len(rows)), len(eligible))
        if n:
            held[rng.choice(np.asarray(eligible), size=n, replace=False)] = True
    return held
