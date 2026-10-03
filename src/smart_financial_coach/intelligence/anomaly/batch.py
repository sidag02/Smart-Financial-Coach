"""Score a dataset's charges with the promoted unusual-transaction model: the nightly flag job
(FR-7 §8; Technical Design, compute timing (a)).

Merchant profiles come from `pool`, every user's model-visible rows (the whole platform), even
when only some users are scored: the demo bundle scores its few accounts against the full
dataset's profiles, as serving would.

    run = flag_dataset("data/synthetic/default.sqlite", "data/flags/default.sqlite")
    run = flag_dataset(bundle / "dataset.sqlite", bundle / "flags.sqlite", pool=full_dataset)
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from smart_financial_coach.data.flags import FlagWriter
from smart_financial_coach.data.store import load_meta, load_transactions
from smart_financial_coach.intelligence.anomaly.contract import AnomalyScorer, scoring_rows
from smart_financial_coach.intelligence.service import load_service

SERVICE = "unusual_transactions"


@dataclass(frozen=True)
class FlagRun:
    model_version: str
    scored: int  # outflows scored
    flagged: int
    seconds: float


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
    scorer = load_service(SERVICE, artifacts_dir)
    if not isinstance(scorer, AnomalyScorer):
        raise TypeError(f"the {SERVICE} service returned {type(scorer).__name__}")
    transactions = load_transactions(data)
    everyone = transactions if pool is None else load_transactions(pool)
    rows = scoring_rows(transactions, pool=everyone)
    scored = scorer.score_transactions(rows)
    dataset = load_meta(data)
    meta = {
        "model_version": scorer.version,
        "data_spec_name": dataset.get("spec_name", ""),
        "data_spec_hash": dataset.get("spec_hash", ""),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with FlagWriter(out, meta, overwrite=overwrite) as writer:
        writer.append(rows["user_id"], scored)
    return FlagRun(scorer.version, len(rows), writer.rows, perf_counter() - started)
