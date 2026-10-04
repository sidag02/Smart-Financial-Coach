"""How well `PersonaWeights` recovers a user's persona from their history alone (FR-11 round 5,
decision 11), held out by user over the round's own validation folds.

    uv run python scripts/fr11_persona_recovery.py data/synthetic/default.sqlite

Prints, by months of history and by persona: the share of held-out (user, as_of) histories whose
most likely persona is the true one, the mean weight on the true persona, and how many there are.
Train users only; test users are never read.
"""

import sys
from pathlib import Path

import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.evaluation.experiment import load_experiment
from smart_financial_coach.evaluation.runner import prepare
from smart_financial_coach.evaluation.tasks.base import get_task
from smart_financial_coach.intelligence.forecasting.contract import parse_history
from smart_financial_coach.intelligence.forecasting.persona import PersonaWeights

CONFIG = PROJECT_ROOT / "configs/experiments/goal_forecasting/21_seasonal_mixture.yaml"


def main(data: Path) -> None:
    task = get_task("goal_forecasting")
    examples, splits = prepare(task, data, load_experiment(CONFIG))
    frame = examples.frame.set_index("example_id")
    rows = []
    for fold in splits.folds:
        train = frame.loc[fold.train.tolist()].drop_duplicates(["user_id", "as_of_date"])
        held = frame.loc[next(iter(fold.held_out.values())).tolist()]
        held = held.drop_duplicates(["user_id", "as_of_date"])
        weights = PersonaWeights().fit(
            [parse_history(h) for h in train["history_json"]], list(train["persona"])
        )
        for history, persona in zip(held["history_json"], held["persona"], strict=True):
            h = parse_history(history)
            w = weights.predict(h)
            rows.append((persona, len(h), max(w, key=w.__getitem__) == persona, w[persona]))
    d = pd.DataFrame(rows, columns=["persona", "months", "right", "weight"])
    d["history"] = pd.cut(d["months"], [0, 6, 12, 23, 1000], labels=["<=6", "7-12", "13-23", "24+"])
    agg = {"right": "mean", "weight": "mean", "months": "size"}
    print(d.groupby("history", observed=True).agg(agg).round(3).rename(columns={"months": "n"}))
    by = d.groupby(["persona", "history"], observed=True).agg(agg).round(3)
    print(by.rename(columns={"months": "n"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
