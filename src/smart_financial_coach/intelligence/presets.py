"""Alert sensitivity presets (FR-9 §1): "Less often", "Balanced" and "More often".

Each preset is a cutoff on an alert model's own score. Balanced is the promoted cutoff itself.
Less and More flag 0.5x and 2x as many rows as Balanced does after the warm-up (decision 11):
count Balanced's post-warm-up flags on a pool of every user, take the k highest post-warm-up
scores (k = 0.5x or 2x that count, ties broken by id as `top_k` breaks them), and the lowest of
them is the cutoff. No labels are read, so the rule carries to real data.

The warm-up is each user's first 90 days from their first transaction (charges), or their first 3
months (spike periods): FR-2's lengths, counted per user. During it, serving uses the stricter of
the session's preset and Balanced (decision 17), so More often starts once a user's history can
judge a charge, and Less often stays Less often.

Cutoffs are placed once, by `sfc-model presets`, and committed beside the model's manifest as
`presets.json` (decision 8), so serving reads fixed numbers:

    presets = place(score, post_warmup, ids, balanced=model.cutoff)
    write_presets(artifacts / service / version, presets, {...})
    presets = load_presets(artifacts / service / version, version)  # None if not placed
    presets.cutoff("more", in_warmup=True)  # Balanced's: the stricter of the two
"""

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.anomaly.threshold import top_k
from smart_financial_coach.intelligence.models.artifact import ArtifactError

LEVELS = ("less", "balanced", "more")
DEFAULT_LEVEL = "balanced"
MULTIPLIERS = {"less": 0.5, "balanced": 1.0, "more": 2.0}
WARMUP_DAYS = 90  # charges: FR-2's warm-up, per user
WARMUP_MONTHS = 3  # spike periods
PRESETS_FILE = "presets.json"


def check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"sensitivity must be one of {', '.join(LEVELS)}, not {level!r}")
    return level


@dataclass(frozen=True)
class Presets:
    """One model's cutoff per level; higher scores are more unusual, so Less's is the highest."""

    cutoffs: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.cutoffs) != set(LEVELS):
            raise ValueError(f"presets need a cutoff for each of {LEVELS}, got {set(self.cutoffs)}")
        less, balanced, more = (self.cutoffs[k] for k in LEVELS)
        if not less >= balanced >= more:
            raise ValueError(f"preset cutoffs out of order: {dict(self.cutoffs)}")

    def cutoff(self, level: str, *, in_warmup: bool = False) -> float:
        """The level's cutoff; in the warm-up, the stricter of it and Balanced's (decision 17)."""
        own = float(self.cutoffs[check_level(level)])
        return max(own, float(self.cutoffs[DEFAULT_LEVEL])) if in_warmup else own

    @property
    def loosest(self) -> float:
        """The lowest cutoff: what a flag file must store down to."""
        return float(self.cutoffs["more"])

    def to_dict(self) -> dict[str, float]:
        return {k: float(self.cutoffs[k]) for k in LEVELS}


@dataclass(frozen=True)
class PlacedPresets:
    """What `sfc-model presets` writes: a model version's presets and how they were placed."""

    model_version: str
    presets: Presets
    record: dict[str, Any]  # what presets.json records beside the cutoffs


def place(
    score: npt.NDArray[np.float64],
    post_warmup: npt.NDArray[np.bool_],
    ids: npt.NDArray[Any],
    balanced: float,
) -> Presets:
    """Less and More at 0.5x and 2x Balanced's post-warm-up flag count on these rows.

    `score` is -inf where a row can't be flagged (no reason, or the product rules forbid it).
    """
    eligible = np.where(post_warmup, score, -np.inf)
    n = int((eligible >= balanced).sum())
    cutoffs = {DEFAULT_LEVEL: float(balanced)}
    for level in ("less", "more"):
        chosen = top_k(eligible, ids, round(n * MULTIPLIERS[level])) & (eligible > -np.inf)
        cutoffs[level] = float(eligible[chosen].min()) if chosen.any() else np.inf
    # Ties at Balanced's own score can put a cutoff on the wrong side of it; never cross it
    cutoffs["less"] = max(cutoffs["less"], cutoffs[DEFAULT_LEVEL])
    cutoffs["more"] = min(cutoffs["more"], cutoffs[DEFAULT_LEVEL])
    return Presets(cutoffs)


def warmup_end(transactions: pd.DataFrame) -> pd.Series:
    """Each user's first transaction (`user_id`, `ts`; any kind) plus the warm-up, by user."""
    first = pd.to_datetime(transactions["ts"]).groupby(transactions["user_id"].to_numpy()).min()
    return first + pd.Timedelta(days=WARMUP_DAYS)


def charges_after_warmup(rows: pd.DataFrame, ends: pd.Series) -> npt.NDArray[np.bool_]:
    """Which charges (`user_id`, `ts`) come after their user's warm-up (`warmup_end`)."""
    end = rows["user_id"].map(ends)
    return (pd.to_datetime(rows["ts"]) >= end).to_numpy()


def periods_after_warmup(
    periods: pd.DataFrame, transactions: pd.DataFrame
) -> npt.NDArray[np.bool_]:
    """Which periods (`user_id`, `period_start`) start 3 or more months after the month of their
    user's first transaction."""
    first = pd.to_datetime(transactions["ts"]).groupby(transactions["user_id"].to_numpy()).min()
    first_month = first.dt.year * 12 + first.dt.month
    start = pd.to_datetime(periods["period_start"])
    month = start.dt.year * 12 + start.dt.month
    return (month - periods["user_id"].map(first_month) >= WARMUP_MONTHS).to_numpy()


def write_presets(directory: Path, presets: Presets, record: Mapping[str, Any]) -> Path:
    """`presets.json` in a model's artifact folder: the cutoffs and what placed them."""
    path = directory / PRESETS_FILE
    payload = {**record, "multipliers": MULTIPLIERS, "cutoffs": presets.to_dict()}
    payload["warmup"] = {"days": WARMUP_DAYS, "months": WARMUP_MONTHS}
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".part") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        temp = Path(f.name)
    temp.chmod(0o644)
    temp.replace(path)
    return path


def load_presets(directory: Path, version: str) -> Presets | None:
    """A model version's presets, or None if none were placed (serving then offers Balanced
    only). A file placed for another version is refused."""
    path = directory / PRESETS_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model_version") != version:
        raise ArtifactError(
            f"{path} was placed for model {payload.get('model_version')!r}, not {version!r}"
        )
    return Presets(payload["cutoffs"])
