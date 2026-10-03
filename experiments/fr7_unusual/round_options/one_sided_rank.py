"""Exploratory, validation only: the forest with a one-sided history rank (max(rank, 0.5)),
as the design's "one-sided relative features" says. Scratch MLflow store, off the leaderboard."""

import sys
from pathlib import Path

import numpy as np
from smart_financial_coach.intelligence.anomaly import forest

from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.tracking import Tracker

original = forest._matrix


def one_sided(f):
    m = original(f)
    m[:, 4] = np.maximum(m[:, 4], 0.5)  # a charge below the user's median is not unusual
    return m


forest._matrix = one_sided
data, store = Path(sys.argv[1]), sys.argv[2]
config = load_experiment("configs/experiments/unusual_transactions/30_isolation_forest.yaml")
config = config.model_copy(update={"name": "isolation_forest_one_sided_rank"})
tracker = Tracker(f"sqlite:///{store}", artifact_root=Path(store).parent / "art-onesided")
result = run_experiment(config, data, tracker, force=True)
keys = [
    "val_recall_at_rate",
    "val_precision_at_rate",
    "val_precision",
    "val_recall",
    "val_flag_rate",
    "val_recall.duplicate",
    "val_recall.amount_outlier",
    "val_recall.new_merchant_large",
    "val_reason_accuracy",
    "val_fp",
    "val_tp",
]
print({k: round(result.metrics[k], 3) for k in keys})
