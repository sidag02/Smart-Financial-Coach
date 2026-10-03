"""Exploratory, validation only: rank 1 (isolation_forest) under a profile-rule option.
A: a merchant's typical price counts only users with 2+ charges there.
B: profiles need 6+ other users (min_users=6) for every judgment.
Runs go to a scratch MLflow store, never the round's."""

import sys
from pathlib import Path

import numpy as np
from smart_financial_coach.intelligence.anomaly import contract

from smart_financial_coach.data.features import merchant_profiles as mp
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.tracking import Tracker

option, data, store = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
if option == "A":
    original = mp._per_user

    def per_user(pool):
        stats = original(pool)
        stats.loc[stats["size"] < 2, "median"] = np.nan  # typical price from 2+ charges only
        return stats

    mp._per_user = per_user
elif option == "B":
    original_rows = contract.scoring_rows
    contract.scoring_rows = lambda t, pool, min_users=6: original_rows(t, pool, min_users=6)
    import smart_financial_coach.evaluation.tasks.unusual as task_module

    task_module.scoring_rows = contract.scoring_rows
config = load_experiment("configs/experiments/unusual_transactions/30_isolation_forest.yaml")
config = config.model_copy(update={"name": f"isolation_forest_option_{option}"})
tracker = Tracker(f"sqlite:///{store}", artifact_root=Path(store).parent / f"art-{option}")
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
]
print(option, {k: round(result.metrics[k], 3) for k in keys})
