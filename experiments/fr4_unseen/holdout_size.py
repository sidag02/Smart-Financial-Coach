"""How precise is unseen-merchant macro F1 with N held-out merchants? (FR-4 measurement question)

Uses bge-base's pooled validation predictions (169 held-out merchants, launch round) and draws
category-stratified subsets of N merchants. For each N:

- the mean merchant-bootstrap SD and 95% interval width: resampling merchants with replacement,
  this treats them as independent draws, as a fresh holdout's merchants are;
- the spread of the point estimate across subsets. Subsets come without replacement from the same
  169 merchants and share most of them as N grows, so this understates a fresh holdout's spread
  beyond N = 58 (finite-population factor 0.81 at 58, 0.57 at 115). Reported for 58 only.

Then the chance that a model whose true score is 0.745 (or 0.73, allowing for 0.745 being the
best of nine runs) scores at least 0.70 on a fresh holdout of N merchants, under a normal
approximation with the bootstrap SD.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from smart_financial_coach.evaluation.metrics.classification import (
    bootstrap_macro_f1,
    bootstrap_weights,
    interval,
)
from smart_financial_coach.evaluation.promote import rebuild_splits
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import Tracker

RUN = "000ef7d304514b3da0fc87046f760299"
tracker, task = Tracker(None), get_task("categorization")
with tempfile.TemporaryDirectory() as tmp:
    examples, _ = rebuild_splits(
        task, Path("data/synthetic/default.sqlite"), tracker, tracker.get(RUN), Path(tmp)
    )
    pooled = pd.read_parquet(
        tracker.download(RUN, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", Path(tmp))
    )
names, stack, majority = task._unseen_stack(examples, pooled)
n_labels = len(task.spending)
full = float(bootstrap_macro_f1(stack, np.ones((1, len(names))), n_labels)[0])
weights = bootstrap_weights(majority, 1000, np.random.default_rng(0))
lo, hi = interval(bootstrap_macro_f1(stack, weights, n_labels))
print(f"all {len(names)} merchants: {full:.3f}, interval {lo:.3f}-{hi:.3f}")
rng = np.random.default_rng(1)
for n in (58, 87, 115):
    points, widths, sds = [], [], []
    for _ in range(200):
        keep_list: list[int] = []
        for label in np.unique(majority):
            members = np.flatnonzero(majority == label)
            take = max(1, round(n * len(members) / len(names)))
            keep_list += list(rng.choice(members, size=min(take, len(members)), replace=False))
        keep = np.asarray(keep_list)
        points.append(float(bootstrap_macro_f1(stack[keep], np.ones((1, len(keep))), n_labels)[0]))
        if len(widths) < 40:
            samples = bootstrap_macro_f1(
                stack[keep], bootstrap_weights(majority[keep], 500, rng), n_labels
            )
            lo, hi = interval(samples)
            widths.append(hi - lo)
            sds.append(float(samples.std()))
    sd = float(np.mean(sds))
    passes = {true: float(norm.sf((0.70 - true) / sd)) for true in (0.745, 0.73)}
    line = f"N~{n}: bootstrap SD {sd:.3f}, interval width {np.mean(widths):.3f}"
    if n == 58:
        low, high = np.percentile(np.asarray(points), [2.5, 97.5])
        line += f", subsets (understated beyond 58) {low:.3f}-{high:.3f}"
    print(line + ", P(>= 0.70): " + ", ".join(f"true {t}: {p:.0%}" for t, p in passes.items()))
