"""FR-5 feasibility: what a review threshold flags, catches and costs (validation data only).

    uv run python experiments/fr5_review/review_thresholds.py --run <twin run ID> \
        [--tracking-uri sqlite:///path/to/mlflow.db] [--data data/synthetic/default.sqlite]

1. A run's out-of-fold calibrated confidences on validation: per familiarity group (rows at
   held-out merchants stand in for unfamiliar strings, the seen-merchant sample for familiar ones),
   the share of rows flagged below a threshold, the share of errors caught, and the share of flags
   that are errors; mixed to production at the test users' unfamiliar-row share.
2. The per-group thresholds the design's rule (§1) would choose: unfamiliar, the lowest catching
   at least 80% of unfamiliar errors; familiar, the highest at which at least 25% of flags are
   errors.
3. Review burden: distinct merchant strings a test user meets for the first time per month,
   familiar or not to the promoted model. Model-visible data only (no truth).
"""

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.store import load_transactions, load_users
from smart_financial_coach.evaluation.promote import rebuild_splits
from smart_financial_coach.evaluation.runner import PREDICTIONS_FILE, PREDICTIONS_PATH
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.service import load_service

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--tracking-uri")
parser.add_argument("--data", type=Path, default=Path("data/synthetic/default.sqlite"))
args = parser.parse_args()

tracker, task = Tracker(args.tracking_uri), get_task("categorization")
with tempfile.TemporaryDirectory() as tmp:
    examples, _ = rebuild_splits(task, args.data, tracker, tracker.get(args.run), Path(tmp))
    path = tracker.download(args.run, f"{PREDICTIONS_PATH}/{PREDICTIONS_FILE}", Path(tmp))
    pooled = pd.read_parquet(path)
rows = task._join(examples, pooled.reset_index(drop=True)).assign(
    held_out=pooled["held_out"].to_numpy()
)
rows = rows[rows["category"] != "Income"]
rows["wrong"] = rows["category"] != rows["predicted"]

# Production mix: the share of test users' spending rows at strings the promoted model hasn't seen
model = load_service("categorization").model
vocab = model.base.vocabulary
users = load_users(args.data)
test = set(users.loc[users["split"] == "test", "user_id"])
tx = load_transactions(args.data)
tx = tx[tx["user_id"].isin(test) & (tx["amount"] < 0)].copy()
tx["string"] = tx["merchant_raw"].map(normalize_merchant)
mix = float((~tx["string"].isin(vocab)).mean())

seen, unseen = rows[rows.held_out == "seen"], rows[rows.held_out == "unseen"]
print(f"model {model.version}; unfamiliar share of test users' rows {mix:.3f}")
print(f"error rate: familiar {seen.wrong.mean():.3f}, unfamiliar {unseen.wrong.mean():.3f}")
table = []
for t in THRESHOLDS:
    r: dict[str, float] = {"threshold": t}
    for name, d in (("familiar", seen), ("unfamiliar", unseen)):
        flag = d["confidence"] < t
        r[f"{name}_flagged"] = float(flag.mean())
        r[f"{name}_errors_caught"] = float((flag & d.wrong).sum() / max(d.wrong.sum(), 1))
        r[f"{name}_flags_wrong"] = float((flag & d.wrong).sum() / max(flag.sum(), 1))
    r["mix_flagged"] = (1 - mix) * r["familiar_flagged"] + mix * r["unfamiliar_flagged"]
    errors = ((1 - mix) * seen.wrong.mean(), mix * unseen.wrong.mean())
    caught = errors[0] * r["familiar_errors_caught"] + errors[1] * r["unfamiliar_errors_caught"]
    r["mix_errors_caught"] = caught / sum(errors)
    table.append(r)
frame = pd.DataFrame(table)
print(frame.round(3).to_string(index=False))

catch = frame[frame["unfamiliar_errors_caught"] >= 0.8]["threshold"]
precise = frame[frame["familiar_flags_wrong"] >= 0.25]["threshold"]
print(
    "rule (§1): unfamiliar threshold",
    catch.min() if len(catch) else "none reaches 80%",
    "| familiar threshold",
    precise.max() if len(precise) else "none at 25% wrong",
)

first = tx.sort_values("ts").drop_duplicates(["user_id", "string"])
first["familiar"] = first["string"].isin(vocab)
first["month"] = first["ts"].str.slice(0, 7)
per = first.groupby(["user_id", "month", "familiar"]).size().unstack(fill_value=0)
per.columns = ["familiar" if c else "unfamiliar" for c in per.columns]
months = sorted(first["month"].unique())
month = per.index.get_level_values("month")
print("first month, strings per user (mean):", per[month == months[0]].mean().round(1).to_dict())
print(
    "months 4+, new strings per user-month (mean):",
    per[month >= months[3]].mean().round(2).to_dict(),
)
