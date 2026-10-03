"""How precise is unseen-merchant macro F1 with N held-out merchants? (FR-4 measurement question)

Uses bge-base's pooled validation predictions (169 held-out merchants, launch round) and draws
category-stratified subsets of N merchants. For each N: the spread of the point estimate across
draws (the luck of one holdout) and the mean 95% merchant-bootstrap interval width.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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
for n in (58, 87, 115, 140):
    points, widths = [], []
    for _ in range(200):
        keep = []
        for label in np.unique(majority):
            members = np.flatnonzero(majority == label)
            take = max(1, round(n * len(members) / len(names)))
            keep += list(rng.choice(members, size=min(take, len(members)), replace=False))
        keep = np.asarray(keep)
        points.append(float(bootstrap_macro_f1(stack[keep], np.ones((1, len(keep))), n_labels)[0]))
        if len(widths) < 40:
            lo, hi = interval(
                bootstrap_macro_f1(
                    stack[keep], bootstrap_weights(majority[keep], 500, rng), n_labels
                )
            )
            widths.append(hi - lo)
    p = np.asarray(points)
    low, high = np.percentile(p, [2.5, 97.5])
    print(
        f"N~{n}: draw-to-draw SD {p.std():.3f}, 95% of draws in {low:.3f}-{high:.3f}, "
        f"mean bootstrap width {np.mean(widths):.3f}"
    )
