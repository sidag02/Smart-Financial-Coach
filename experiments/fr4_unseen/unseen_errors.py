"""Where do a run's validation unseen-merchant errors go? Per-class F1 and predicted-class mix."""

import sys
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from smart_financial_coach.evaluation.promote import rebuild_splits
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import Tracker

uri, run_id = (sys.argv[1] if sys.argv[1] != "-" else None), sys.argv[2]
tracker, task = Tracker(uri), get_task("categorization")
with tempfile.TemporaryDirectory() as tmp:
    examples, _ = rebuild_splits(
        task, Path("data/synthetic/default.sqlite"), tracker, tracker.get(run_id), Path(tmp)
    )
    pooled = pd.read_parquet(
        tracker.download(run_id, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", Path(tmp))
    )
rows = task._join(examples, pooled[pooled["held_out"] == "unseen"].reset_index(drop=True))
rows = rows[rows["category"] != "Income"]
labels = list(task.spending)
f1 = f1_score(rows["category"], rows["predicted"], labels=labels, average=None, zero_division=0)
out = pd.DataFrame(
    {
        "f1": f1.round(3),
        "true_rows": rows["category"].value_counts().reindex(labels).fillna(0).astype(int),
        "predicted_rows": rows["predicted"].value_counts().reindex(labels).fillna(0).astype(int),
        # Each merchant once, under its majority category (a few merchants have rows in two),
        # as the merchant bootstrap groups them
        "merchants": rows.groupby("merchant_id")["category"]
        .agg(lambda c: c.mode().iloc[0])
        .value_counts()
        .reindex(labels)
        .fillna(0)
        .astype(int),
    },
    index=labels,
)
print(out.to_string())
print("accuracy", round((rows["category"] == rows["predicted"]).mean(), 3), "rows", len(rows))
