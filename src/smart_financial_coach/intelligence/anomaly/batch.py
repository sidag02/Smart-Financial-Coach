"""Score a dataset's charges with the promoted unusual-transaction model: the nightly flag job
(FR-7 §8; Technical Design, compute timing (a)), and the placing of its sensitivity presets
(FR-9 §1).

Merchant profiles come from `pool`, every user's model-visible rows (the whole platform), even
when only some users are scored: the demo bundle scores its few accounts against the full
dataset's profiles, as serving would.

When the promoted model has presets (`presets.json` beside its manifest), the flag file holds
every charge down to the More often cutoff, and its meta records the three cutoffs, so serving
can show any level (FR-9 §2). Without presets it holds Balanced's flags only, as before FR-9.

    run = flag_dataset("data/synthetic/default.sqlite", "data/flags/default.sqlite")
    run = flag_dataset(bundle / "dataset.sqlite", bundle / "flags.sqlite", pool=full_dataset)
    placed = place_presets("data/synthetic/default.sqlite")  # what `sfc-model presets` writes
"""

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from smart_financial_coach.data.flags import FlagWriter, presets_meta
from smart_financial_coach.data.store import load_meta, load_transactions
from smart_financial_coach.intelligence.anomaly.contract import AnomalyScorer, scoring_rows
from smart_financial_coach.intelligence.anomaly.threshold import Thresholded, user_months
from smart_financial_coach.intelligence.models.artifact import promoted_version
from smart_financial_coach.intelligence.presets import (
    LEVELS,
    PlacedPresets,
    Presets,
    charges_after_warmup,
    load_presets,
    place,
    warmup_end,
)
from smart_financial_coach.intelligence.service import get_service, load_service, service_dir

SERVICE = "unusual_transactions"


@dataclass(frozen=True)
class FlagRun:
    model_version: str
    scored: int  # outflows scored
    flagged: int  # rows stored: down to More often's cutoff when the model has presets
    seconds: float
    balanced: int  # of them, flagged at Balanced (the promoted cutoff)
    presets: Presets | None = None


def promoted_presets(artifacts_dir: Path | None = None) -> Presets | None:
    """The promoted model's presets, or None if none were placed for it."""
    root = service_dir(SERVICE, artifacts_dir)
    version = promoted_version(root)
    return load_presets(root / version, version)


def _scorer(artifacts_dir: Path | None) -> tuple[AnomalyScorer, Thresholded]:
    scorer = load_service(SERVICE, artifacts_dir)
    if not isinstance(scorer, AnomalyScorer):
        raise TypeError(f"the {SERVICE} service returned {type(scorer).__name__}")
    if not isinstance(scorer.model, Thresholded):
        raise TypeError(f"the {SERVICE} model isn't thresholded: {type(scorer.model).__name__}")
    return scorer, scorer.model


def flag_dataset(
    data: str | Path,
    out: str | Path,
    *,
    pool: str | Path | None = None,
    artifacts_dir: Path | None = None,
    overwrite: bool = False,
) -> FlagRun:
    """Flag every outflow of `data`'s users into the flag file `out`.

    Each user's whole history is scored in one batch (point-in-time features need it); `pool`
    defaults to `data` itself.
    """
    started = perf_counter()
    scorer, model = _scorer(artifacts_dir)
    presets = promoted_presets(artifacts_dir)
    transactions = load_transactions(data)
    everyone = transactions if pool is None else load_transactions(pool)
    rows = scoring_rows(transactions, pool=everyone)
    flagging = scorer
    if presets is not None:
        # The same model cut at More often's score, still contract-checked, so every stored row
        # has its reason and evidence
        loosest = copy.copy(model)
        loosest.cutoff = presets.loosest
        wrapped = get_service(SERVICE).wrapper(loosest, scorer.contract)
        assert isinstance(wrapped, AnomalyScorer)
        flagging = wrapped
    scored = flagging.score_transactions(rows)
    dataset = load_meta(data)
    meta = {
        "model_version": scorer.version,
        "data_spec_name": dataset.get("spec_name", ""),
        "data_spec_hash": dataset.get("spec_hash", ""),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if presets is not None:
        meta |= presets_meta(presets.to_dict())
    with FlagWriter(out, meta, overwrite=overwrite) as writer:
        writer.append(rows["user_id"], scored)
    flagged = scored["is_flagged"].astype(bool).to_numpy()
    balanced = flagged & (scored["score"].to_numpy(dtype=np.float64) >= model.cutoff)
    return FlagRun(
        scorer.version,
        len(rows),
        writer.rows,
        perf_counter() - started,
        int(balanced.sum()),
        presets,
    )


def place_presets(data: str | Path, *, artifacts_dir: Path | None = None) -> PlacedPresets:
    """Less and More for the promoted model, placed on every user of `data` (FR-9 §1).

    The pool is scored as serving scores it, with merchant profiles from every user. Counts are
    reported per level after the warm-up, and in it, where serving uses the stricter of the level
    and Balanced (decision 17).
    """
    scorer, model = _scorer(artifacts_dir)
    transactions = load_transactions(data)
    rows = scoring_rows(transactions, pool=transactions)
    scored = model._scores(rows)
    has_reason = scored["reason_code"].notna().to_numpy()
    score = np.where(has_reason, scored["score"].to_numpy(dtype=np.float64), -np.inf)
    post = charges_after_warmup(rows, warmup_end(transactions))
    presets = place(score, post, rows["transaction_id"].to_numpy(), float(model.cutoff))

    months = user_months(rows[post])
    counts: dict[str, dict[str, float]] = {}
    for level in LEVELS:
        n = int((post & (score >= presets.cutoff(level))).sum())
        counts[level] = {
            "after_warmup": n,
            "in_warmup": int((~post & (score >= presets.cutoff(level, in_warmup=True))).sum()),
            "per_user_month_after_warmup": n / months if months else 0.0,
        }
    meta = load_meta(data)
    record = {
        "service": SERVICE,
        "model_version": scorer.version,
        "pool": {
            "data_spec_name": meta.get("spec_name", ""),
            "data_spec_hash": meta.get("spec_hash", ""),
            "users": int(transactions["user_id"].nunique()),
            "user_months_after_warmup": months,
        },
        "counts": counts,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return PlacedPresets(scorer.version, presets, record)
