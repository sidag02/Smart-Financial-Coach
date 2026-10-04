"""How many of FR-11's blocking calibration gates a perfectly calibrated model fails by chance
(owner, on #54). Train users only; test users are never read.

    uv run python scripts/fr11_gate_chance.py data/synthetic/default.sqlite \
        --tracking-uri sqlite:///data/mlflow.db --run RUN_ID

Takes a run's pooled out-of-fold validation predictions and makes the model calibrated by
construction, two ways:
- **independent draws:** every goal's outcome is a draw from its own predicted chance. Each
  simulation applies the task's calibration gates (each band's met rate inside the band, widened
  by the run's own validation tolerance), on all validation users and on a random half of them
  per persona (the test set's size). This understates the noise: a user's goals share one future;
- **correlated:** each gate's met rate is normal around the band's mean predicted chance, with
  the spread of the run's own user-bootstrap interval (which carries the within-user
  correlation), widened by sqrt(2) at the test set's size.
Prints the expected number of failing gates, the chance of at least one, and the likeliest.
"""

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from smart_financial_coach.evaluation.promote import rebuild_splits
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tasks.goals import BANDS, PATHS, PERSONAS, _in_band
from smart_financial_coach.evaluation.tracking import Tracker

Z95 = 1.959964

SIMULATIONS = 2000


def gates(
    rows: pd.DataFrame, metrics: dict[str, float]
) -> list[tuple[str, np.ndarray, float, float]]:
    """Each calibration gate: its name, the rows it reads, and the band widened by tolerance."""
    out = []
    for path in PATHS:
        for scope in ("", *(f".{p}" for p in PERSONAS)):
            part = rows["path"] == path
            if scope:
                part &= rows["persona"] == scope[1:]
            for band, lo, hi in BANDS:
                name = f"met_rate.{path}{scope}.{band}"
                a, b = metrics.get(f"val_{name}_lo"), metrics.get(f"val_{name}_hi")
                tol = 0.0 if a is None or b is None or np.isnan(a) else (b - a) / 2
                idx = np.flatnonzero(part & _in_band(rows["p_goal_met"], lo, hi))
                out.append((name, idx, lo - tol, hi + tol))
    return out


def simulate(rows: pd.DataFrame, checks: list, half: bool, rng: np.random.Generator) -> list[str]:
    p = rows["p_goal_met"].to_numpy(dtype=float)
    met = rng.random(len(p)) < p
    keep = np.ones(len(p), dtype=bool)
    if half:  # a random half of each persona's users: the test set's size
        users = rows.drop_duplicates("user_id")[["user_id", "persona"]]
        chosen = users.groupby("persona")["user_id"].apply(
            lambda u: u.sample(frac=0.5, random_state=int(rng.integers(1 << 31)))
        )
        keep = rows["user_id"].isin(set(chosen)).to_numpy()
    failed = []
    for name, idx, lo, hi in checks:
        idx = idx[keep[idx]]
        if len(idx) and not lo <= met[idx].mean() <= hi:
            failed.append(name)
    return failed


def correlated(
    rows: pd.DataFrame, checks: list, metrics: dict[str, float], scale: float
) -> list[tuple[str, float]]:
    """Each gate's chance of failing for a calibrated model, from its bootstrap spread."""
    from scipy.stats import norm

    p = rows["p_goal_met"].to_numpy(dtype=float)
    out = []
    for name, idx, lo, hi in checks:
        a, b = metrics.get(f"val_{name}_lo"), metrics.get(f"val_{name}_hi")
        if not len(idx) or a is None or b is None or np.isnan(a):
            continue
        sd = (b - a) / (2 * Z95) * scale
        mean = p[idx].mean()
        out.append((name, float(norm.cdf(lo, mean, sd) + norm.sf(hi, mean, sd))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    tracker = Tracker(args.tracking_uri)
    task = get_task("goal_forecasting")
    run = tracker.get(args.run)
    with tempfile.TemporaryDirectory() as tmp:
        examples, _ = rebuild_splits(task, args.data, tracker, run, Path(tmp))
        path = tracker.download(args.run, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", Path(tmp) / "p")
        pooled = pd.read_parquet(path)
    rows = task._rows(examples, pooled).reset_index(drop=True)
    checks = gates(rows, run.metrics)
    rng = np.random.default_rng(0)
    print(f"{run.name} ({args.run}): {len(checks)} calibration gates, {SIMULATIONS} simulations")
    for label, half in (("all validation users", False), ("half of them (test size)", True)):
        fails = [simulate(rows, checks, half, rng) for _ in range(SIMULATIONS)]
        counts = np.array([len(f) for f in fails])
        often = Counter(g for f in fails for g in f).most_common(4)
        print(
            f"  {label}: expected failures {counts.mean():.2f}, "
            f"P(at least one) {(counts > 0).mean():.2f}, P(4 or more) {(counts >= 4).mean():.3f}"
        )
        print("    most often: " + ", ".join(f"{g} {n / SIMULATIONS:.2f}" for g, n in often))
    for label, scale in (("correlated, validation size", 1.0), ("correlated, test size", 2**0.5)):
        chances = correlated(rows, checks, run.metrics, scale)
        q = np.array([c for _, c in chances])
        top = sorted(chances, key=lambda c: -c[1])[:4]
        print(
            f"  {label}: expected failures {q.sum():.2f}, "
            f"P(at least one) {1 - np.prod(1 - q):.2f} (gates taken as independent)"
        )
        print("    likeliest: " + ", ".join(f"{g} {c:.2f}" for g, c in top))
    print("  margins at test size (the band's mean chance to the gate's nearer edge, in SDs):")
    p = rows["p_goal_met"].to_numpy(dtype=float)
    margins = []
    for name, idx, lo, hi in checks:
        a, b = run.metrics.get(f"val_{name}_lo"), run.metrics.get(f"val_{name}_hi")
        if len(idx) and a is not None and b is not None and not np.isnan(a):
            sd = (b - a) / (2 * Z95) * 2**0.5
            mean = p[idx].mean()
            margins.append((min(mean - lo, hi - mean) / sd, name, mean, len(idx)))
    for z, name, mean, n in sorted(margins)[:5]:
        print(f"    {name}: mean chance {mean:.2f} on {n} goals, {z:.1f} SD from the edge")


if __name__ == "__main__":
    main()
