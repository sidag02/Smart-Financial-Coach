"""The simulated feedback replay (FR-5 and FR-6 design, §7): the learning loop, month by month.

Synthetic test users with their own category preferences (`truth_preferences`) review, correct,
slip and occasionally misbehave; their feedback becomes global labels by agreement (§4); the
categorizer is retrained on them (§5), gated and, if it passes, promoted within the replay.

- **Who learns and who's measured:** test users are split in two by a hash of their id.
  *Feedback* users review and correct; *evaluation* users never do, so the model's gain on them
  is learning, not their own overrides (Technical Design, "Learning from user feedback").
- **Each month:** the month's transactions are categorized and flagged by the current model and its
  review policy; each feedback user may open their queue and resolve its top items, and may notice
  an unflagged wrong category; the measures are recorded.
- **Each quarter** (after a first stretch): the agreement rule runs over the votes so far. With
  new labels, a candidate is retrained up to the month's end and gated against the incumbent on
  the evaluation users' last quarter, scored against their own view, with data available at that
  time only. A promoted candidate recategorizes the history; the following quarters show the gain.

Run once with N = 3 and the default settings; nothing is chosen from its results (owner, Oct 3,
2026). The gates' tolerances are provisional, like N, the majority and the cadence.

    result = run_replay(Path("data/synthetic/default.sqlite"), ReplayConfig())
    result.to_json()
"""

import hashlib
import json
import logging
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.generator.taxonomy import INCOME
from smart_financial_coach.data.labels import load_truth
from smart_financial_coach.evaluation.experiment import ExperimentConfig, load_experiment
from smart_financial_coach.evaluation.metrics.classification import macro_f1
from smart_financial_coach.evaluation.retrain import RetrainError, retrain
from smart_financial_coach.evaluation.runner import code_version, git_commit, git_dirty
from smart_financial_coach.evaluation.tasks.categorization import CategorizationTask
from smart_financial_coach.evaluation.tracking import KIND_TAG, MODEL_PATH, Tracker
from smart_financial_coach.intelligence.categorization.agreement import (
    AgreementRule,
    Vote,
    current_votes,
    global_labels,
    label_history,
)
from smart_financial_coach.intelligence.categorization.contract import INPUT_COLUMNS
from smart_financial_coach.intelligence.categorization.review import (
    ReviewPolicy,
    load_review_policy,
    save_review_policy,
)
from smart_financial_coach.intelligence.models.artifact import save_artifact
from smart_financial_coach.intelligence.models.contract import Checked
from smart_financial_coach.intelligence.service import load_service

log = logging.getLogger(__name__)

SHIPPED_CONFIG = PROJECT_ROOT / "configs/experiments/categorization/fr4/21_small_unweighted.yaml"


@dataclass(frozen=True)
class Behavior:
    """How simulated users respond (§7 Behavior). Defaults are plausible, not fitted."""

    engagement: float = 0.6  # chance a feedback user opens the review queue in a month
    items_per_visit: int = 5  # items they resolve when they do, most spend first
    accept_rate: float = 0.15  # confirm the suggestion without checking (automation bias)
    slip: float = 0.03  # a correction lands on a wrong category
    notice: float = 0.05  # chance per month of fixing an unflagged string they see differently
    adversarial: float = 0.05  # share of feedback users who correct at random


@dataclass(frozen=True)
class Gates:
    """§5's gates for a retrained model, against the incumbent, on the evaluation users' last
    quarter before the cutoff, scored against their own view. Tolerances are provisional."""

    familiar_drop: float = 0.005  # familiar-string accuracy may not fall by more than this
    brier_rise: float = 0.005  # unfamiliar Brier may not rise by more than this


@dataclass(frozen=True)
class ReplayConfig:
    behavior: Behavior = field(default_factory=Behavior)
    rule: AgreementRule = field(default_factory=AgreementRule)
    gates: Gates = field(default_factory=Gates)
    first_retrain: int = 6  # months of feedback before the first retraining
    every: int = 3  # months between retrainings (quarterly)
    min_new_labels: int = 1  # skip a retraining with fewer new global labels
    seed: int = 0


@dataclass
class ReplayResult:
    config: dict[str, Any]
    users: dict[str, int]
    months: list[dict[str, Any]] = field(default_factory=list)
    retrainings: list[dict[str, Any]] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    remaps: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(_plain(asdict(self)), indent=2, default=str) + "\n"


def _plain(value: Any) -> Any:
    """JSON-ready: mapping keys as strings (a tuple key becomes "a/b"), numpy scalars as Python."""
    if isinstance(value, dict):
        return {
            ("/".join(map(str, k)) if isinstance(k, tuple) else str(k)): _plain(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class _Vote:
    """A merchant-scope feedback event, in the shape `agreement.current_votes` reads."""

    seq: int
    subject: str
    action: str
    merchant_key: str
    to_category: str
    scope: str = "merchant"
    undone: bool = False


def _half(user_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


def _spend(amount: pd.Series) -> pd.Series:
    return (-amount).clip(lower=0)


def _mode(values: pd.Series) -> str:
    counts = values.value_counts()
    return str(counts[counts == counts.max()].index.sort_values()[0])


class _Model:
    """The replay's serving model: the categorizer and its review policy."""

    def __init__(self, model: Checked, policy: ReviewPolicy, name: str) -> None:
        self.model, self.policy, self.name = model, policy, name

    def categorize(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = self.model.predict(rows[list(INPUT_COLUMNS)].reset_index(drop=True))
        needs, _ = self.policy.flag(out["confidence"], out["familiar"])
        spending = (out["category"] != INCOME).to_numpy()
        return pd.DataFrame(
            {
                "predicted": out["category"].to_numpy(),
                "confidence": out["confidence"].to_numpy(),
                "familiar": out["familiar"].to_numpy(),
                "needs_review": needs & spending,
            },
            index=rows.index,
        )


def _scores(rows: pd.DataFrame, predicted: pd.Series, spending: tuple[str, ...]) -> dict[str, Any]:
    """Accuracy and macro F1 against the users' view and against the truth, on spending rows."""
    r = rows[rows["user_category"] != INCOME]
    p = predicted.loc[r.index]
    if r.empty:
        return {"rows": 0}
    return {
        "rows": len(r),
        "accuracy_view": float((p == r["user_category"]).mean()),
        "macro_f1_view": macro_f1(r["user_category"], p, spending),
        "macro_f1_truth": macro_f1(r["category"], p, spending),
    }


def run_replay(
    data: Path,
    config: ReplayConfig | None = None,
    *,
    experiment: Path = SHIPPED_CONFIG,
    artifacts_dir: Path | None = None,
    progress: Callable[[str], None] = log.info,
) -> ReplayResult:
    config = config or ReplayConfig()
    b = config.behavior
    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()

    task = CategorizationTask()
    examples = task.load(data)
    spending = task.spending
    shipped: ExperimentConfig = load_experiment(experiment).twin()
    splits = task.split(examples, shipped.task_params, shipped.seed)

    truth = load_truth(data)
    tx = truth.transactions.assign(user_category=truth.user_categories())
    tx = tx[tx["user_id"].isin(set(truth.users.loc[truth.users["split"] == "test", "user_id"]))]
    tx = tx.assign(
        merchant_key=tx["merchant_raw"].map(normalize_merchant),
        month=pd.to_datetime(tx["ts"]).dt.strftime("%Y-%m"),
    ).set_index("transaction_id", drop=False)
    subtype = truth.merchants.set_index("merchant_id")["subtype"]
    holdout = set(truth.merchants.loc[truth.merchants["holdout"] == 1, "merchant_id"])
    tx["subtype"] = tx["merchant_id"].map(subtype)
    tx["unseen"] = tx["merchant_id"].isin(holdout)

    users = sorted(tx["user_id"].unique())
    feedback = [u for u in users if _half(u, "replay-group") < 0.5]
    evaluation = [u for u in users if u not in set(feedback)]
    # Exactly round(share x feedback users), lowest hash first: a threshold on the hash gave 6 of
    # 69 users at a 20% share; at 5% both pick the same 3
    by_hash = sorted(feedback, key=lambda u: _half(u, "replay-adversarial"))
    adversarial = set(by_hash[: round(b.adversarial * len(feedback))])
    result = ReplayResult(
        config=asdict(config) | {"experiment": shipped.name},
        users={
            "feedback": len(feedback),
            "evaluation": len(evaluation),
            "adversarial": len(adversarial),
        },
    )
    months = sorted(tx["month"].unique())
    progress(f"replay: {len(users)} test users, {len(months)} months, {len(tx):,} transactions")

    current = _Model(
        load_service("categorization", artifacts_dir),
        load_review_policy(artifacts_dir),
        "promoted",
    )
    preds = pd.DataFrame(columns=["predicted", "confidence", "familiar", "needs_review"])
    overrides: dict[str, dict[str, str]] = {u: {} for u in feedback}
    votes: list[_Vote] = []
    labels_in_model: dict[str, str] = {}
    seq = 0
    last_eval = pd.Series(dtype=object)  # evaluation users' predictions after the last month
    promoted_since = False

    def vote(user: str, key: str, category: str, corrected: bool) -> None:
        nonlocal seq
        seq += 1
        votes.append(_Vote(seq, user, "correct" if corrected else "confirm", key, category))
        overrides[user][key] = category

    for i, month in enumerate(months):
        rows = tx[tx["month"] == month]
        preds = pd.concat([preds, current.categorize(rows)])
        seen = tx.loc[preds.index]
        shown = queued = 0  # items resolved, and items open in the queues, this month
        for user in feedback:
            mine = seen[seen["user_id"] == user]
            p = preds.loc[mine.index]
            spend = mine[mine["user_category"] != INCOME]
            mine_over = overrides[user]
            open_items = p[p["needs_review"] & ~mine["merchant_key"].isin(mine_over)]
            queued += mine.loc[open_items.index, "merchant_key"].nunique()
            if len(open_items) and rng.random() < b.engagement:
                ranked = (
                    _spend(mine.loc[open_items.index, "amount"])
                    .groupby(mine.loc[open_items.index, "merchant_key"])
                    .sum()
                    .sort_values(ascending=False)
                )
                for key in ranked.index[: b.items_per_visit]:
                    shown += 1
                    at = mine["merchant_key"] == key
                    suggestion = _mode(
                        p.loc[
                            open_items.index[mine.loc[open_items.index, "merchant_key"] == key],
                            "predicted",
                        ]
                    )
                    view = _mode(mine.loc[at, "user_category"])
                    if user in adversarial:
                        choices = [c for c in spending if c != suggestion]
                        vote(user, key, str(rng.choice(choices)), True)
                    elif rng.random() < b.accept_rate or suggestion == view:
                        vote(user, key, suggestion, False)
                    else:
                        target = view
                        if rng.random() < b.slip:
                            target = str(rng.choice([c for c in spending if c != view]))
                        if target == suggestion:
                            vote(user, key, suggestion, False)
                        else:
                            vote(user, key, target, True)
            # Noticing an unflagged string they see differently
            this_month = spend[spend["month"] == month]
            for key, group in this_month.groupby("merchant_key"):
                if key in mine_over or p.loc[group.index, "needs_review"].any():
                    continue
                view = _mode(group["user_category"])
                if _mode(p.loc[group.index, "predicted"]) != view and rng.random() < b.notice:
                    vote(user, str(key), view, True)

        # Measures for the month
        month_rows = tx[tx["month"] == month]
        predicted = preds.loc[month_rows.index, "predicted"]
        fb_rows = month_rows[month_rows["user_id"].isin(set(feedback))]
        effective = predicted.loc[fb_rows.index].copy()
        for idx, (u, k) in zip(
            fb_rows.index,
            zip(fb_rows["user_id"], fb_rows["merchant_key"], strict=True),
            strict=True,
        ):
            # Overrides leave rows the model predicted as Income alone (§3), as the app does
            if k in overrides[u] and predicted.at[idx] != INCOME:
                effective.at[idx] = overrides[u][k]
        ev_rows = month_rows[month_rows["user_id"].isin(set(evaluation))]
        # Isolation: evaluation users' categories change only when a model is promoted, and
        # they never vote; nobody's overrides reach another user (each user's view applies only
        # their own, above)
        ev_all = preds.index[tx.loc[preds.index, "user_id"].isin(set(evaluation))]
        before = last_eval.reindex(ev_all).dropna()
        changed = int((preds.loc[before.index, "predicted"] != before).sum())
        isolation = {
            "evaluation_rows_changed": changed,
            "after_promotion": promoted_since,
            "evaluation_votes": sum(v.subject in set(evaluation) for v in votes),
        }
        last_eval = preds.loc[ev_all, "predicted"].copy()
        promoted_since = False
        result.months.append(
            {
                "month": month,
                "model": current.name,
                "personal": _scores(fb_rows, effective, spending),
                "personal_model_only": _scores(fb_rows, predicted.loc[fb_rows.index], spending),
                "evaluation": _scores(ev_rows, predicted.loc[ev_rows.index], spending),
                "evaluation_unseen": _scores(
                    ev_rows[ev_rows["unseen"]], predicted.loc[ev_rows.index], spending
                ),
                "open_items_per_user": queued / len(feedback),
                "items_resolved_per_user": shown / len(feedback),
                "share_resolved": shown / queued if queued else None,
                "isolation": isolation,
                "votes": len(votes),
                "global_labels": len(global_labels(current_votes(votes), config.rule)),
            }
        )

        # Quarterly retraining
        n = i + 1
        if (
            n < config.first_retrain
            or (n - config.first_retrain) % config.every
            or n == len(months)
        ):
            continue
        standing = current_votes(votes)
        labels = global_labels(standing, config.rule)
        agreed = {k: lab.category for k, lab in labels.items()}
        new = {k: c for k, c in agreed.items() if labels_in_model.get(k) != c}
        revoked = sorted(set(labels_in_model) - set(agreed))
        step: dict[str, Any] = {
            "month": month,
            "labels": len(agreed),
            "new_labels": sorted(new.items()),
            "revoked": revoked,
        }
        if len(new) + len(revoked) < config.min_new_labels:
            step["decision"] = "skipped: no new labels"
            result.retrainings.append(step)
            continue
        cutoff = pd.Timestamp(f"{months[i]}-01") + pd.offsets.MonthBegin(1)
        contributors = {
            k: [v.subject for v in standing if v.merchant_key == k and v.category == c]
            for k, c in agreed.items()
        }
        # Everything the retraining reads, so `retrain_step` can rebuild it exactly
        step["inputs"] = {
            "cutoff": str(cutoff.date()),
            "agreed": agreed,
            "contributors": contributors,
        }
        progress(f"{month}: retraining on {len(agreed)} global labels ({len(new)} new)")
        try:
            candidate = retrain(
                task,
                examples,
                splits,
                shipped,
                agreed,
                contributors,
                cutoff,
                evaluation,
                version=f"replay-{month}",
            )
        except RetrainError as error:
            step["decision"] = f"not trained: {error}"
            result.retrainings.append(step)
            continue
        step |= {
            "relabelled_rows": candidate.relabelled_rows,
            "added_rows": candidate.added_rows,
            "training_rows": candidate.training_rows,
            "policy": {
                "familiar_below": candidate.policy.familiar_threshold,
                "unfamiliar_below": candidate.policy.unfamiliar_threshold,
            },
        }
        # Gate on the evaluation users' last quarter before the cutoff (never trained on)
        window = months[max(0, i - 2) : i + 1]
        gate_rows = tx[tx["month"].isin(window) & tx["user_id"].isin(set(evaluation))]
        gate_rows = gate_rows[gate_rows["user_category"] != INCOME]
        old = preds.loc[gate_rows.index]
        cand = _Model(candidate.model, candidate.policy, f"retrained {month}").categorize(gate_rows)
        step["gates"] = _gates(gate_rows, old, cand, set(agreed), spending, config.gates)
        passed = all(g["passed"] for g in step["gates"])
        step["decision"] = "promoted" if passed else "rejected"
        result.retrainings.append(step)
        if passed:
            current = _Model(candidate.model, candidate.policy, f"retrained {month}")
            labels_in_model = agreed
            promoted_since = True
            preds = current.categorize(tx.loc[preds.index])
        progress(f"{month}: {step['decision']}")

    _report(result, tx, votes, feedback, evaluation, spending, config, preds, overrides)
    result.summary["seconds"] = round(time.perf_counter() - started)
    return result


FEEDBACK_KIND = "feedback-retrain"  # a run's sfc.kind: never on the leaderboard or finalized


def groups(data: Path) -> tuple[list[str], list[str]]:
    """The replay's feedback and evaluation users (test users split by a hash of their id)."""
    truth = load_truth(data)
    users = sorted(truth.users.loc[truth.users["split"] == "test", "user_id"])
    feedback = [u for u in users if _half(u, "replay-group") < 0.5]
    return feedback, [u for u in users if u not in set(feedback)]


def retrain_step(
    data: Path,
    replay: dict[str, Any],
    month: str,
    tracker: Tracker,
    *,
    experiment: Path = SHIPPED_CONFIG,
) -> str:
    """Rebuild one of the replay's retrainings from its recorded inputs and log the model as a
    tracked run (`sfc.kind` = feedback-retrain). Not a finalist, so it can't be promoted without
    `finalize` and `promote`; nothing is deployed. Returns the run ID."""
    steps = [s for s in replay["retrainings"] if s["month"] == month and "inputs" in s]
    if not steps:
        raise ValueError(f"the replay has no retraining with recorded inputs in {month}")
    step = steps[0]
    inputs = step["inputs"]
    task = CategorizationTask()
    examples = task.load(data)
    shipped = load_experiment(experiment).twin()
    splits = task.split(examples, shipped.task_params, shipped.seed)
    _, evaluation = groups(data)
    version = f"feedback-{month}-{shipped.config_hash()[:8]}-{examples.data_hash[:8]}"
    retrained = retrain(
        task,
        examples,
        splits,
        shipped,
        inputs["agreed"],
        inputs["contributors"],
        pd.Timestamp(inputs["cutoff"]),
        evaluation,
        version=version,
    )
    tags = {
        KIND_TAG: FEEDBACK_KIND,
        "sfc.task": task.name,
        "sfc.config_hash": shipped.config_hash(),
        "sfc.data_hash": examples.data_hash,
        "sfc.code_version": code_version(),
        "sfc.git_commit": git_commit(),
        "sfc.git_dirty": str(git_dirty()).lower(),
        "sfc.version": version,
        "sfc.replay_month": month,
        "sfc.replay_decision": str(step.get("decision")),
        "sfc.note": (
            "Retrained from the FR-5/FR-6 replay's agreed labels (simulated test users' "
            "feedback). Its FR-3/FR-4 test-set scores aren't clean; it isn't promoted."
        ),
    }
    with tracker.run(task.name, f"{shipped.name} feedback {month}", tags) as run_id:
        tracker.log(
            run_id,
            params={
                "cutoff": inputs["cutoff"],
                "global_labels": str(len(inputs["agreed"])),
                "relabelled_rows": str(retrained.relabelled_rows),
                "added_rows": str(retrained.added_rows),
                "training_rows": str(retrained.training_rows),
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / MODEL_PATH
            manifest = {
                "task": task.name,
                "config": json.loads(shipped.model_dump_json()),
                "mlflow_run_id": run_id,
                "replay_step": {k: v for k, v in step.items() if k != "inputs"},
                **{k.removeprefix("sfc."): v for k, v in tags.items()},
            }
            save_artifact(retrained.model.model, out, manifest)
            save_review_policy(retrained.policy, out)
            labels = Path(tmp) / "agreed_labels.json"
            labels.write_text(json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8")
            tracker.log_artifacts(run_id, out, MODEL_PATH)
            tracker.log_artifacts(run_id, Path(tmp), "")
    return run_id


def _gates(
    rows: pd.DataFrame,
    old: pd.DataFrame,
    new: pd.DataFrame,
    labelled: set[str],
    spending: tuple[str, ...],
    gates: Gates,
) -> list[dict[str, Any]]:
    view = rows["user_category"]

    def brier(p: pd.DataFrame, mask: pd.Series) -> float:
        correct = (p.loc[mask[mask].index, "predicted"] == view[mask]).astype(float)
        return float(((p.loc[mask[mask].index, "confidence"] - correct) ** 2).mean())

    out = []
    f_old, f_new = (
        macro_f1(view, old["predicted"], spending),
        macro_f1(view, new["predicted"], spending),
    )
    out.append(
        {"name": "macro_f1_view", "passed": f_new >= f_old, "detail": f"{f_new:.3f} vs {f_old:.3f}"}
    )
    fam = old["familiar"].astype(bool)
    a_old = float((old.loc[fam, "predicted"] == view[fam]).mean())
    a_new = float((new.loc[fam, "predicted"] == view[fam]).mean())
    out.append(
        {
            "name": "familiar_no_regression",
            "passed": a_new >= a_old - gates.familiar_drop,
            "detail": f"{a_new:.3f} vs {a_old:.3f}",
        }
    )
    at = rows["merchant_key"].isin(labelled)
    if at.any():
        l_old = float((old.loc[at, "predicted"] == view[at]).mean())
        l_new = float((new.loc[at, "predicted"] == view[at]).mean())
        out.append(
            {
                "name": "labelled_strings_improve",
                "passed": l_new > l_old or l_old == 1.0,
                "detail": f"{l_new:.3f} vs {l_old:.3f} on {int(at.sum())} rows",
            }
        )
    # Both on the rows unfamiliar to the incumbent, so the comparison is on the same rows
    b_old, b_new = brier(old, ~fam), brier(new, ~fam)
    out.append(
        {
            "name": "unfamiliar_brier",
            "passed": b_new <= b_old + gates.brier_rise,
            "detail": f"{b_new:.3f} vs {b_old:.3f}",
        }
    )
    out.append(
        {
            "name": "truth_macro_f1 (reported, not gated)",
            "passed": True,
            "detail": (
                f"{macro_f1(rows['category'], new['predicted'], spending):.3f} vs "
                f"{macro_f1(rows['category'], old['predicted'], spending):.3f}"
            ),
        }
    )
    return out


def _report(
    result: ReplayResult,
    tx: pd.DataFrame,
    votes: list[_Vote],
    feedback: list[str],
    evaluation: list[str],
    spending: tuple[str, ...],
    config: ReplayConfig,
    preds: pd.DataFrame,
    overrides: dict[str, dict[str, str]],
) -> None:
    standing = current_votes(votes)
    final = global_labels(standing, config.rule)
    stream = [
        Vote(v.subject, v.merchant_key, v.to_category, v.action == "correct", v.seq) for v in votes
    ]
    history = label_history(stream, config.rule)
    subtype_of = tx.groupby("merchant_key")["subtype"].agg(lambda s: s.mode().iloc[0])
    truth_of = tx.groupby("merchant_key")["category"].agg(lambda s: s.mode().iloc[0])
    for key, lab in sorted(final.items()):
        result.labels.append(
            {
                "merchant_key": key,
                "category": lab.category,
                "true_category": truth_of.get(key),
                "subtype": subtype_of.get(key),
                "voters": lab.voters,
                "agreeing": lab.agreeing,
                "corrections": lab.corrections,
            }
        )
    # Per remap: who holds it, what the model now says to evaluation users, label churn
    prefs = tx[tx["user_category"] != tx["category"]]
    remaps = prefs.groupby(["subtype", "category", "user_category"]).size().reset_index()
    last = sorted(tx["month"].unique())[-3:]
    ev = tx[tx["user_id"].isin(set(evaluation)) & tx["month"].isin(last)]
    rng = np.random.default_rng(config.seed + 1)
    for r in remaps.itertuples():
        at = ev[ev["subtype"] == r.subtype]
        holders = set(prefs.loc[prefs["subtype"] == r.subtype, "user_id"])
        keys = set(tx.loc[tx["subtype"] == r.subtype, "merchant_key"])
        changes = [c for c in history if c.merchant_key in keys]
        match = (
            (preds.loc[at.index, "predicted"] == at["user_category"]).groupby(at["user_id"]).mean()
        )
        reps = (
            [
                float(
                    match.sample(
                        len(match), replace=True, random_state=int(rng.integers(1 << 31))
                    ).mean()
                )
                for _ in range(500)
            ]
            if len(match)
            else []
        )
        result.remaps.append(
            {
                "subtype": r.subtype,
                "default": r.category,
                "remapped_to": r.user_category,
                "holders": len(holders),
                "holders_feedback": len(holders & set(feedback)),
                "evaluation_match_view": float(match.mean()) if len(match) else None,
                "evaluation_match_interval": [
                    float(np.percentile(reps, 2.5)),
                    float(np.percentile(reps, 97.5)),
                ]
                if reps
                else None,
                "labels_now": dict(Counter(final[k].category for k in keys if k in final)),
                "label_changes": [asdict(c) for c in changes],
            }
        )
    first = result.months[0]["evaluation"] if result.months else {}
    final_q = [m["evaluation"] for m in result.months[-3:]]
    result.summary |= {
        "votes": len(votes),
        "corrections": sum(v.action == "correct" for v in votes),
        "global_labels": len(final),
        "labels_matching_truth": sum(
            1 for x in result.labels if x["category"] == x["true_category"]
        ),
        "promotions": sum(s["decision"] == "promoted" for s in result.retrainings),
        "evaluation_first_month": first,
        "evaluation_last_quarter_macro_f1_view": float(
            np.mean([m.get("macro_f1_view", np.nan) for m in final_q])
        )
        if final_q
        else None,
        "personal_accuracy_last_quarter": float(
            np.mean([m["personal"].get("accuracy_view", np.nan) for m in result.months[-3:]])
        )
        if result.months
        else None,
        # Burden (§7): items open in a feedback user's queue in a month, and how many they resolve
        "open_items_per_user_month": float(
            np.mean([m["open_items_per_user"] for m in result.months])
        )
        if result.months
        else None,
        "items_resolved_per_user_month": float(
            np.mean([m["items_resolved_per_user"] for m in result.months])
        )
        if result.months
        else None,
        "isolation_holds": all(
            m["isolation"]["evaluation_votes"] == 0
            and (
                m["isolation"]["evaluation_rows_changed"] == 0 or m["isolation"]["after_promotion"]
            )
            for m in result.months
        ),
    }
