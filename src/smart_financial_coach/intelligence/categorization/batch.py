"""Categorize a whole dataset with the promoted model, in batches across users.

Categorization runs on ingestion, batched across users (Technical Design, compute timing), so this
is the serving path, and its throughput is the number a cluster is sized by. Batches mix users:
one merchant string seen by many users is embedded once.

Each row is flagged for review here, by the promoted model's own review policy (FR-5 §1), so
the dashboard and the tools read flags without calling the model.

    run = categorize_dataset("data/synthetic/default.sqlite", "data/predictions/default.sqlite")
    run.rows_per_second
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from smart_financial_coach.data.predictions import CategoryWriter
from smart_financial_coach.data.store import iter_transactions, load_meta
from smart_financial_coach.intelligence.categorization.contract import Categorizer
from smart_financial_coach.intelligence.categorization.review import load_review_policy
from smart_financial_coach.intelligence.service import load_service

# Throughput is flat from 20k to 100k rows per batch (about 12k rows/s with bge-base on a laptop),
# but peak memory isn't: 2.1 GB at 20k against 7.2 GB at 100k, on the default dataset
DEFAULT_BATCH_ROWS = 20_000


@dataclass(frozen=True)
class BatchRun:
    model_version: str
    rows: int
    load_seconds: float  # loading the promoted model, including a first-use download
    categorize_seconds: float

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.categorize_seconds if self.categorize_seconds else float("nan")


def categorize_dataset(
    data: str | Path,
    out: str | Path,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    artifacts_dir: Path | None = None,
    overwrite: bool = False,
) -> BatchRun:
    """Categorize every transaction with the promoted model into the predictions file `out`."""
    started = perf_counter()
    categorizer = load_service("categorization", artifacts_dir)
    if not isinstance(categorizer, Categorizer):
        raise TypeError(f"the categorization service returned {type(categorizer).__name__}")
    policy = load_review_policy(artifacts_dir)
    if policy.model_version != categorizer.version:  # the pointer moved while loading
        raise RuntimeError(f"review policy {policy.model_version} for model {categorizer.version}")
    loaded = perf_counter()
    dataset = load_meta(data)
    meta = {
        "model_version": categorizer.version,
        "data_spec_name": dataset.get("spec_name", ""),
        "data_spec_hash": dataset.get("spec_hash", ""),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "review_familiar_below": str(policy.familiar_threshold),
        "review_unfamiliar_below": str(policy.unfamiliar_threshold),
    }
    categorize_seconds = 0.0
    with CategoryWriter(out, meta, overwrite=overwrite) as writer:
        for batch in iter_transactions(data, batch_rows):  # memory follows the batch, not the data
            start = perf_counter()
            categories = categorizer.categorize(batch)
            categorize_seconds += perf_counter() - start
            needs_review, reason = policy.flag(categories["confidence"], categories["familiar"])
            flagged = categories.assign(needs_review=needs_review, review_reason=reason)
            writer.append(batch["user_id"], flagged)
    return BatchRun(categorizer.version, writer.rows, loaded - started, categorize_seconds)
