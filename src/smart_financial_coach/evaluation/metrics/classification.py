"""Classification metrics: macro F1 over a fixed label set, calibration, merchant bootstrap.

Macro F1 is over exactly the labels given (FR-3: the 12 spending categories), on rows whose true
label is one of them, so a prediction outside the set (Income) counts against recall without adding
a label to the average.

The bootstrap resamples groups (merchants), not rows: the unseen-merchant set has many rows but
few merchants, so merchants are the independent units.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

ECE_BINS = 15
CONFIDENT = 0.9

Floats = npt.NDArray[np.float64]


def _f1_from_confusion(confusion: Floats, labels: Sequence[int]) -> Floats:
    """Per-label F1 from confusion matrices shaped (..., k, k): rows truth, columns prediction."""
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)[..., labels]
    predicted = confusion.sum(axis=-2)[..., labels]
    actual = confusion.sum(axis=-1)[..., labels]
    denominator = predicted + actual
    f1: Floats = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return f1


def confusion(truth: pd.Series, pred: pd.Series, labels: Sequence[str]) -> Floats:
    """Confusion counts over `labels` plus one extra column for any other prediction."""
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels) + 1
    t = truth.map(index).to_numpy()
    p = pred.map(index).fillna(k - 1).to_numpy(dtype=int)
    keep = ~pd.isna(t)
    out = np.zeros((k, k))
    np.add.at(out, (t[keep].astype(int), p[keep]), 1)
    return out


def macro_f1(truth: pd.Series, pred: pd.Series, labels: Sequence[str]) -> float:
    """Macro F1 over `labels` that occur in `truth`; NaN if none do."""
    c = confusion(truth, pred, labels)
    present = [i for i in range(len(labels)) if c[i].sum() > 0]
    return float(_f1_from_confusion(c, present).mean()) if present else float("nan")


def per_class_f1(truth: pd.Series, pred: pd.Series, labels: Sequence[str]) -> dict[str, float]:
    c = confusion(truth, pred, labels)
    f1 = _f1_from_confusion(c, list(range(len(labels))))
    return {label: float(f1[i]) for i, label in enumerate(labels) if c[i].sum() > 0}


def calibration(truth: pd.Series, pred: pd.Series, confidence: pd.Series) -> dict[str, float]:
    """Quality of the top-class confidence as a probability of being right.

    - `brier`: mean squared error of confidence against correctness. A proper score: it is
      minimized only by confidences that are both calibrated and discriminating, so it is the one
      used to choose (calibration methods, tie-breaks).
    - `ece`: expected calibration error over 15 equal-width bins. Easy to read ("0.9 means 90%")
      but not proper (a constant confidence equal to the accuracy scores 0), so it is reported only.
    - `acc_at_90`, `coverage_at_90`: the FR-5 operating point: how often confidence >= 0.9 is
      right, and how many transactions clear it.
    """
    conf = confidence.to_numpy(dtype=float)
    correct = (truth.to_numpy() == pred.to_numpy()).astype(float)
    if not len(conf):
        nan = float("nan")
        return {"brier": nan, "ece": nan, "acc_at_90": nan, "coverage_at_90": nan}
    bins = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
    weight = np.bincount(bins, minlength=ECE_BINS) / len(conf)
    acc = np.bincount(bins, weights=correct, minlength=ECE_BINS)
    mean_conf = np.bincount(bins, weights=conf, minlength=ECE_BINS)
    counts = np.maximum(np.bincount(bins, minlength=ECE_BINS), 1)
    ece = float((weight * np.abs(acc / counts - mean_conf / counts)).sum())
    confident = conf >= CONFIDENT
    return {
        "brier": float(np.mean((conf - correct) ** 2)),
        "ece": ece,
        "acc_at_90": float(correct[confident].mean()) if confident.any() else float("nan"),
        "coverage_at_90": float(confident.mean()),
    }


def misallocated_spend(truth: pd.Series, pred: pd.Series, amount: pd.Series) -> float:
    """Share of dollars put in the wrong category: what the spend-by-category views feel."""
    dollars = amount.abs().to_numpy(dtype=float)
    wrong = truth.to_numpy() != pred.to_numpy()
    total = dollars.sum()
    return float(dollars[wrong].sum() / total) if total else float("nan")


def group_confusions(
    truth: pd.Series, pred: pd.Series, groups: pd.Series, labels: Sequence[str]
) -> tuple[list[str], Floats, npt.NDArray[np.int64]]:
    """Per-group confusion matrices (g, k, k) and each group's majority true label."""
    names = sorted(set(groups.astype(str)))
    stack = np.stack(
        [
            confusion(truth[mask], pred[mask], labels)
            for mask in (groups.astype(str) == g for g in names)
        ]
    )
    majority = stack.sum(axis=2).argmax(axis=1)
    return names, stack, majority


def bootstrap_weights(
    majority: npt.NDArray[np.int64], reps: int, rng: np.random.Generator
) -> Floats:
    """(reps, groups) resample counts: groups drawn with replacement within their majority label."""
    weights = np.zeros((reps, len(majority)))
    for label in np.unique(majority):
        members = np.flatnonzero(majority == label)
        weights[:, members] = rng.multinomial(
            len(members), np.full(len(members), 1 / len(members)), size=reps
        )
    return weights


def bootstrap_macro_f1(stack: Floats, weights: Floats, n_labels: int) -> Floats:
    """Macro F1 for each resample; labels with no true rows in a resample are left out."""
    totals = np.einsum("rg,gij->rij", weights, stack)
    f1 = _f1_from_confusion(totals, list(range(n_labels)))
    present = totals.sum(axis=-1)[:, :n_labels] > 0
    return np.asarray((f1 * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1))


def interval(samples: Floats, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return float(lo), float(hi)
