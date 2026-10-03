"""Category preferences (schema 4): how some test users see merchants differently.

The FR-5/FR-6 data contract. Each test user adopts each remap independently with its share; a user
who adopts one sees every merchant of that subtype in the remap's category. The result is
`truth_preferences(user_id, merchant_id, category)`, one row per user and remapped merchant.

Drawn after everything else, from the preference seed and the user's ID alone, so adding or
changing preferences never changes another generated row, and one user's draw doesn't depend on
who else is in the dataset. Train users never get preferences: the cold model's labels stay the
true categories.
"""

import hashlib

import numpy as np
import pandas as pd

from smart_financial_coach.data.generator.catalog import Catalog
from smart_financial_coach.data.generator.spec import PreferencesSpec

COLUMNS = ["user_id", "merchant_id", "category"]


def _user_key(user_id: str) -> int:
    return int.from_bytes(hashlib.sha256(user_id.encode()).digest()[:8], "little")


def remapped_merchants(catalog: Catalog, spec: PreferencesSpec) -> list[list[str]]:
    """For each remap, the merchants of its subtype; fails on a subtype that can't be remapped."""
    m = catalog.merchants
    out, errors = [], []
    for remap in spec.remaps:
        rows = m[m["subtype"] == remap.subtype]
        if rows.empty:
            errors.append(f"subtype {remap.subtype!r} has no merchants")
        elif (rows["category"] == remap.category).any():
            errors.append(f"subtype {remap.subtype!r} is already {remap.category}")
        out.append(sorted(rows["merchant_id"].astype(str)))
    if errors:
        raise ValueError("preferences: " + "; ".join(errors))
    return out


def adopted(user_id: str, spec: PreferencesSpec) -> list[bool]:
    """Which remaps this user adopts: one independent draw per remap, in spec order."""
    draws = np.random.default_rng([spec.seed, _user_key(user_id)]).random(len(spec.remaps))
    return [bool(d < r.share) for d, r in zip(draws, spec.remaps, strict=True)]


def preferences_table(users: pd.DataFrame, catalog: Catalog, spec: PreferencesSpec) -> pd.DataFrame:
    merchants = remapped_merchants(catalog, spec)
    rows = []
    test_users = sorted(users.loc[users["split"] == "test", "user_id"].astype(str))
    for user_id in test_users:
        for remap, ids, take in zip(spec.remaps, merchants, adopted(user_id, spec), strict=True):
            if take:
                rows += [(user_id, mid, remap.category) for mid in ids]
    return pd.DataFrame(rows, columns=COLUMNS)
