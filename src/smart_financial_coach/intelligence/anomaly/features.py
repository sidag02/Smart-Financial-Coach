"""The relative features every unusual-transaction scorer reads (FR-7 §2, §3).

Built from the scoring rows alone (model-visible columns plus merchant profiles), point in time:

    amount, log_amount
    repeat            an exact repeat (same raw text and amount) within `duplicate_minutes`
    first_visit       no earlier charge at this merchant key
    z_merchant        log amount against the user's earlier charges at the key, in units of the
                      larger of the user's and the profile's spread (floor `min_spread`); NaN on a
                      first visit
    profile_log_ratio log amount minus the profile's typical log amount; NaN without a profile

and the evidence each reason needs (`EVIDENCE`): `usual_amount`, `ratio` and `prior_charges` for
an unusual amount; `amount`, `prior_charges`, `rank_in_history` and `largest_since` for a new
merchant; `original_transaction_id`, `minutes_apart` and `amount` for a duplicate. `prior_charges`
counts earlier charges at the merchant for an unusual amount, and all earlier charges otherwise,
so the template reads true either way.
"""

import numpy as np
import pandas as pd

from smart_financial_coach.data.features.history import history_features

DUPLICATE_MINUTES = 90.0
MIN_SPREAD = 0.25  # log-amount spread floor: a merchant's spread is never trusted below this


def relative_features(
    x: pd.DataFrame,
    duplicate_minutes: float = DUPLICATE_MINUTES,
    min_spread: float = MIN_SPREAD,
) -> pd.DataFrame:
    """One row per input row (all outflows), in input order."""
    h = history_features(x)
    amount = -x["amount"].to_numpy(dtype=float)
    log_amount = np.log(amount)
    user_spread = h["key_spread"].fillna(0.0).to_numpy()
    profile_spread = x["profile_spread"].fillna(0.0).to_numpy()
    spread = np.maximum(np.maximum(user_spread, profile_spread), min_spread)
    first = (h["key_prior"] == 0).to_numpy()
    z = np.where(first, np.nan, (log_amount - h["key_median"].to_numpy()) / spread)
    minutes = h["minutes_since_repeat"].to_numpy(dtype=float)
    repeat = np.nan_to_num(minutes, nan=np.inf) <= duplicate_minutes
    usual = np.exp(h["key_median"].to_numpy())
    return pd.DataFrame(
        {
            "transaction_id": x["transaction_id"].to_numpy(),
            "amount": amount,
            "log_amount": log_amount,
            "repeat": repeat,
            "first_visit": first,
            "z_merchant": z,
            "profile_log_ratio": log_amount - x["profile_typical"].to_numpy(dtype=float),
            "rank_in_history": h["rank_in_history"].to_numpy(),
            # Evidence
            "original_transaction_id": h["repeat_of"].where(pd.Series(repeat), None).to_numpy(),
            "minutes_apart": np.where(repeat, minutes, np.nan),
            "usual_amount": np.round(usual, 2),
            "ratio": np.round(amount / usual, 1),
            "key_prior": h["key_prior"].to_numpy(),
            "key_median": h["key_median"].to_numpy(dtype=float),
            "key_spread": h["key_spread"].to_numpy(dtype=float),
            "profile_typical": x["profile_typical"].to_numpy(dtype=float),
            "profile_spread": x["profile_spread"].to_numpy(dtype=float),
            "overall_prior": h["prior_charges"].to_numpy(),
            "largest_since": h["largest_since"].to_numpy(),
        }
    )


def scores_frame(f: pd.DataFrame, score: np.ndarray, reason: np.ndarray) -> pd.DataFrame:
    """A scorer's output: score, reason code, and the evidence columns its reasons use."""
    amount_reason = reason == "amount_unusual"
    return pd.DataFrame(
        {
            "transaction_id": f["transaction_id"].to_numpy(),
            "score": score,
            "reason_code": pd.Series(reason, dtype=object).where(pd.notna(reason), None).to_numpy(),
            "original_transaction_id": f["original_transaction_id"].to_numpy(),
            "minutes_apart": np.round(f["minutes_apart"].to_numpy(dtype=float), 1),
            "amount": np.round(f["amount"].to_numpy(), 2),
            "usual_amount": f["usual_amount"].to_numpy(),
            "ratio": f["ratio"].to_numpy(),
            "prior_charges": np.where(amount_reason, f["key_prior"], f["overall_prior"]),
            "rank_in_history": np.round(f["rank_in_history"].to_numpy(dtype=float), 3),
            "largest_since": f["largest_since"].to_numpy(),
        }
    )
