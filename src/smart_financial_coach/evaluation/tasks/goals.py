"""The goal-forecasting task (FR-11 and FR-12 design, §5-§6): examples, splits, metrics and gates.

Examples are goals whose outcome is known, at their `as_of_date`, scored on two paths:
- `track`: the goal as generated, with its track record from $0 at `created_date`;
- `new`: the same goal as if it were created at `as_of_date`, its balance entered by hand. That's
  the path every goal created through FR-10, and every draft behind the fit badge, takes.

The goals are the dataset's own (draw 0) plus `draws` more per user from FR-1's stage-9 sampler,
over the user's own monthly net savings and under the task's seed, so labeled goals are plentiful
without changing the dataset (§6, owner decision 4). Each draw is its own goal set: the 1-2 goals
stage 9 makes together; draws never coexist.

A model sees only months up to `as_of_date` (`history_json`); the leak check proves it. Labels
(`met`) are truth, and nothing is tuned on them: outcomes are planted relative to the realized
future ("Why nothing is tuned on outcomes"). The months after `as_of_date` are evaluation-only,
for the range's coverage.

Splits: validation is 5 folds of whole train users, by persona; test users are scored once.
Every interval resamples users, since one user's goals share one future (§6).
"""

import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.generator.goals import sample_goals
from smart_financial_coach.data.generator.spec import Spec, load_spec, spec_hash
from smart_financial_coach.data.generator.sqlite_io import read_sqlite
from smart_financial_coach.data.generator.timeline import Timeline
from smart_financial_coach.data.store import load_meta, load_transactions, load_users
from smart_financial_coach.evaluation.metrics.classification import interval
from smart_financial_coach.evaluation.splits import (
    TRAIN,
    Ids,
    Splits,
    group_kfold,
    ids,
    leak_errors,
)
from smart_financial_coach.evaluation.tasks.base import Examples, Gate, register_task
from smart_financial_coach.evaluation.tasks.categorization import content_hash
from smart_financial_coach.intelligence.forecasting.contract import (
    INPUT_COLUMNS,
    OFF_TRACK,
    ON_TRACK,
    history_json,
    parse_history,
)
from smart_financial_coach.intelligence.forecasting.savings import final_balance, monthly_net
from smart_financial_coach.intelligence.models.contract import Checked

TEST = "test"
PATHS = ("track", "new")
PERSONAS = ("family_budgeter", "freelancer", "young_professional")
BANDS = (("off", 0.0, OFF_TRACK), ("either", OFF_TRACK, ON_TRACK), ("on", ON_TRACK, 1.0))
FLAT = 0.25  # a flat 50% scores 0.25 on any outcomes
COVERAGE = (0.70, 0.90)  # the 80% range must hold 70-90% of realized balances (§5)
BASELINES = ("naive_pace", "flat_50")
# Decision 6 (owner, on #50): a failing freelancer gate on the new-goal path is reported, not
# loosened, and doesn't block promotion on its own
NON_BLOCKING = {("new", "freelancer")}
BOOTSTRAP_REPS = 1000
DEFAULTS: dict[str, Any] = {"k": 5}
DRAWS = 10  # sampler draws per user beyond the dataset's own goals (§6)
SAMPLER_SEED = 11

Boot = tuple[Ids, npt.NDArray[np.float64]]  # the users (sorted) and how often each is drawn


def _spec_for(meta: Mapping[str, str]) -> Spec:
    """The spec the dataset was generated from, checked against its hash: the sampler must draw
    goals by the same rules (FR-1's validation compares the same hash)."""
    path = PROJECT_ROOT / "configs" / "data" / f"{meta['spec_name']}.yaml"
    spec = load_spec(path)
    if spec_hash(spec) != meta["spec_hash"]:
        raise ValueError(f"{path} isn't the spec this dataset was generated from (hash differs)")
    return spec


def _user_seed(user_id: str) -> int:
    return zlib.crc32(user_id.encode())


def goal_examples(
    *,
    users: pd.DataFrame,
    transactions: pd.DataFrame,
    goals: pd.DataFrame,
    truth_goals: pd.DataFrame,
    spec: Spec,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Two example rows (`track`, `new`) per known-outcome goal, dataset goals and sampled."""
    tl = Timeline(spec.calendar.start, spec.calendar.end)
    start = pd.Period(spec.calendar.start, freq="M")
    month = (pd.to_datetime(transactions["ts"]).dt.to_period("M") - start).map(lambda d: d.n)
    by_user = transactions.assign(m=month.to_numpy()).groupby("user_id")
    labeled = goals.merge(truth_goals, on="goal_id")
    rows: list[dict[str, Any]] = []
    for u in users.to_dict("records"):
        user_id, persona_name = str(u["user_id"]), str(u["persona"])
        frame = by_user.get_group(user_id)
        net = np.asarray(
            np.bincount(
                frame["m"].to_numpy(dtype=np.int64),
                weights=frame["amount"].to_numpy(dtype=float),
                minlength=tl.n_months,
            ),
            dtype=float,
        )
        # What a model may see: full months up to the dataset's end, by `monthly_net`'s rules,
        # the same definition serving uses; each example takes the months up to its as_of
        visible = monthly_net(frame[["ts", "amount"]], spec.calendar.end)
        persona = spec.personas[persona_name]
        sets = [labeled[labeled["user_id"] == user_id]]
        for d in range(1, draws + 1):
            g, t = sample_goals(
                net,
                tl,
                [x.name for x in persona.goals],
                [x.weight for x in persona.goals],
                spec.goals,
                np.random.default_rng([seed, _user_seed(user_id), d]),
                user_id=user_id,
                goal_prefix=f"e{d}_{user_id[2:]}",
            )
            sets.append(g.merge(t, on="goal_id"))
        for draw, goal_set in enumerate(sets):
            rows += _set_rows(u, goal_set, draw, net, start, visible)
    return pd.DataFrame(rows)


def _set_rows(
    user: dict[Any, Any],
    goal_set: pd.DataFrame,
    draw: int,
    net: npt.NDArray[np.float64],
    start: pd.Period,
    visible: pd.Series,
) -> list[dict[str, Any]]:
    """`net` is the generator's monthly net (every month of the calendar), which labels and the
    realized future come from; `visible` is what a model may see."""
    out = []
    set_id = f"{user['user_id']}:{draw}"
    for g in goal_set[goal_set["met"].notna()].to_dict("records"):
        as_of_date, target_date = str(g["as_of_date"]), str(g["target_date"])
        as_of = pd.Period(as_of_date, freq="M")
        a, t = (as_of - start).n, (pd.Period(target_date, freq="M") - start).n
        history = visible[visible.index <= as_of]
        future = pd.Series(net[a + 1 : t + 1], index=pd.period_range(as_of + 1, periods=t - a))
        # Goals of the set already created and not yet ended at this as_of (a sibling created
        # later is the future; review on #51)
        active = int(
            (
                (goal_set["target_date"] > as_of_date) & (goal_set["created_date"] <= as_of_date)
            ).sum()
        )
        common = {
            "goal_id": g["goal_id"],
            "user_id": user["user_id"],
            "goal_set": set_id,
            "persona": user["persona"],
            "as_of_date": as_of_date,
            "target_amount": float(g["target_amount"]),
            "target_date": target_date,
            "saved": float(g["current_balance"]),
            "saved_as_of": as_of_date,
            "first_saved": float("nan"),
            "first_saved_as_of": None,
            "active_goals": active,
            "history_json": history_json(history),
            # Evaluation only
            "split": user["split"],
            "draw": draw,
            "outcome_class": g["outcome_class"],
            "future_json": history_json(future),
            "met": int(g["met"]),
        }
        out.append(
            common
            | {
                "example_id": f"{g['goal_id']}:track",
                "path": "track",
                "created_date": g["created_date"],
                "origin": "existing",
            }
        )
        out.append(
            common
            | {
                "example_id": f"{g['goal_id']}:new",
                "path": "new",
                "created_date": as_of_date,
                "origin": "yours",
            }
        )
    return out


class GoalForecastingTask:
    """Metric choices and the alternatives considered: FR-11 and FR-12 design, "Metrics and why"."""

    name = "goal_forecasting"
    selection_metric = "neg_brier"  # minus the mean of the two paths' Brier
    tuning_metric = "neg_brier"
    # Among tied runs (lower is better): the worst band's calibration error, then batch cost.
    # Scaled RMSE of net savings joins in milestone 2, with the net-savings forecasters
    tiebreak_metrics: tuple[str, ...] = ("val_band_error", "latency_batch_ms")
    shipping_params: tuple[str, ...] = ()  # nothing trains on labels: no shipping twins
    serving_files_required: tuple[str, ...] = ()
    report_metrics: tuple[str, ...] = (
        "val_brier.track",
        "val_brier.new",
        "val_band_error",
        "val_coverage.track",
        "val_coverage.new",
        *(f"val_brier.{p}.{persona}" for p in PATHS for persona in PERSONAS),
        "latency_batch_ms",
    )
    test_report_metrics: tuple[str, ...] = tuple(
        f"test_{m}"
        for p in PATHS
        for m in (
            f"brier.{p}",
            f"brier.{p}_lo",
            f"brier.{p}_hi",
            f"coverage.{p}",
            *(f"brier.{p}.{persona}" for persona in PERSONAS),
        )
    )
    required_baselines: tuple[str, ...] = BASELINES
    bootstrap_unit = "user"
    diagnostics_scope = "on both paths"

    def __init__(
        self, reps: int = BOOTSTRAP_REPS, draws: int = DRAWS, seed: int = SAMPLER_SEED
    ) -> None:
        self.reps = reps
        self.draws = draws
        self.seed = seed

    # --- Data and splits -------------------------------------------------------------------

    def load(self, data: Path) -> Examples:
        meta = load_meta(data)
        dataset = read_sqlite(data)
        frame = goal_examples(
            users=load_users(data)[["user_id", "split", "persona"]],
            transactions=load_transactions(data)[["user_id", "ts", "amount"]],
            goals=dataset["goals"],
            truth_goals=dataset["truth_goals"],
            spec=_spec_for(meta),
            draws=self.draws,
            seed=self.seed,
        )
        return Examples(
            frame=frame,
            labels=frame["met"],
            id_column="example_id",
            model_columns=INPUT_COLUMNS,
            data_hash=content_hash(frame[list(INPUT_COLUMNS)], meta),
        )

    def split(self, examples: Examples, params: Mapping[str, Any], seed: int) -> Splits:
        p = DEFAULTS | dict(params)
        f = examples.frame
        train = f[f["split"] == "train"]
        folds = group_kfold(
            train,
            group="user_id",
            k=int(p["k"]),
            seed=seed,
            id_column="example_id",
            stratify="persona",
        )
        return Splits(
            sets={
                TRAIN: ids(train["example_id"]),
                TEST: ids(f.loc[f["split"] == "test", "example_id"]),
            },
            folds=folds,
        )

    def leak_errors(self, examples: Examples, splits: Splits) -> list[str]:
        f = examples.frame.set_index("example_id")
        errors = leak_errors(splits, examples.frame, id_column="example_id", group="user_id")
        for name, split in ((TRAIN, "train"), (TEST, "test")):
            if (f.loc[splits.sets[name].tolist(), "split"] != split).any():
                errors.append(
                    f"{name} has goals of {'test' if split == 'train' else 'train'} users"
                )
        if shared := set(f.loc[splits.sets[TRAIN].tolist(), "user_id"]) & set(
            f.loc[splits.sets[TEST].tolist(), "user_id"]
        ):
            errors.append(f"users in both train and test: {sorted(shared)[:5]}")
        errors += history_leaks(examples.frame)
        return errors

    def training_rows(
        self, examples: Examples, which: Ids, params: Mapping[str, Any], seed: int
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """No labels: outcomes are planted relative to the realized future, so nothing may be
        fitted on them ("Why nothing is tuned on outcomes"); withholding them here enforces it
        for every candidate (review on #51)."""
        return examples.rows(which), None

    # --- Metrics ---------------------------------------------------------------------------

    def _rows(self, examples: Examples, predictions: pd.DataFrame) -> pd.DataFrame:
        cols = ["user_id", "persona", "path", "met", "saved", "future_json", "outcome_class"]
        f = examples.frame.set_index("example_id")
        joined = f.loc[predictions["example_id"].tolist(), cols].reset_index(drop=True)
        out = pd.concat([predictions.reset_index(drop=True), joined], axis=1)
        out["sq"] = (out["p_goal_met"].astype(float) - out["met"]) ** 2
        out["covered"] = _covered(out)
        return out

    def _metrics(self, rows: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for path in PATHS:
            r = rows[rows["path"] == path]
            if r.empty:
                continue
            out[f"brier.{path}"] = float(r["sq"].mean())
            out[f"accuracy.{path}"] = float(((r["p_goal_met"] >= 0.5) == (r["met"] == 1)).mean())
            out[f"coverage.{path}"] = float(r["covered"].mean())  # NaN without a share
            out[f"goals.{path}"] = float(len(r))
            for band, met_rate, n in _bands(r):
                out[f"met_rate.{path}.{band}"] = met_rate
                out[f"goals.{path}.{band}"] = float(n)
            for persona, part in r.groupby("persona"):
                out[f"brier.{path}.{persona}"] = float(part["sq"].mean())
                for band, met_rate, n in _bands(part):
                    out[f"met_rate.{path}.{persona}.{band}"] = met_rate
                    out[f"goals.{path}.{persona}.{band}"] = float(n)
        briers = [out[f"brier.{p}"] for p in PATHS if f"brier.{p}" in out]
        out["brier"] = float(np.mean(briers)) if briers else float("nan")
        out["neg_brier"] = -out["brier"]
        out["band_error"] = max(
            (
                _outside(out[f"met_rate.{p}.{b}"], lo, hi)
                for p in PATHS
                for b, lo, hi in BANDS
                if f"met_rate.{p}.{b}" in out
            ),
            default=float("nan"),
        )
        return out

    def _intervals(self, rows: pd.DataFrame) -> dict[str, float]:
        """95% user-bootstrap intervals of what the gates read: Brier per path and persona, and
        each band's met rate per path, overall and per persona."""
        out: dict[str, float] = {}
        boot = self._boot(rows)

        def add(name: str, part: pd.DataFrame, column: str) -> None:
            if len(part):
                out[f"{name}_lo"], out[f"{name}_hi"] = interval(self._mean(boot, part, column))

        for path in PATHS:
            r = rows[rows["path"] == path]
            add(f"brier.{path}", r, "sq")
            for scope, part in [("", r), *((f".{p}", r[r["persona"] == p]) for p in PERSONAS)]:
                if scope:
                    add(f"brier.{path}{scope}", part, "sq")
                for band, lo_p, hi_p in BANDS:
                    add(
                        f"met_rate.{path}{scope}.{band}",
                        part[_in_band(part["p_goal_met"], lo_p, hi_p)],
                        "met",
                    )
        return out

    def validation_metrics(self, examples: Examples, pooled: pd.DataFrame) -> dict[str, float]:
        """With the intervals the gates' tolerances come from, logged before any test scoring."""
        rows = self._rows(examples, pooled)
        return self._metrics(rows) | self._intervals(rows)

    def test_metrics(self, examples: Examples, splits: Splits, model: Checked) -> dict[str, float]:
        rows = self._rows(examples, model.predict(examples.rows(splits.sets[TEST])))
        return self._metrics(rows) | self._intervals(rows)

    # --- User bootstrap ----------------------------------------------------------------------

    def _boot(self, rows: pd.DataFrame) -> Boot:
        """The users in `rows` (sorted) and how often each is drawn in each bootstrap sample."""
        users = ids(np.sort(rows["user_id"].unique()))
        rng = np.random.default_rng(0)
        draws = rng.integers(len(users), size=(self.reps, len(users)))
        weights = np.stack([np.bincount(d, minlength=len(users)) for d in draws]).astype(float)
        return users, weights

    @staticmethod
    def _mean(boot: Boot, rows: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:
        """Per bootstrap sample, the mean of `column` over the drawn users' rows. `rows` may be
        any subset (a path, a persona, a band): its users are lined up with the whole draw."""
        users, weights = boot
        per_user = rows.groupby("user_id")[column].agg(["sum", "count"]).reindex(users)
        per_user = per_user.fillna(0.0)
        total = weights @ per_user["sum"].to_numpy(dtype=float)
        count = weights @ per_user["count"].to_numpy(dtype=float)
        mean: npt.NDArray[np.float64] = np.divide(
            total, count, out=np.full(len(total), np.nan), where=count > 0
        )
        return mean

    def _brier_samples(self, boot: Boot, rows: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Per bootstrap sample, the mean of the two paths' Brier."""
        per_path = [self._mean(boot, rows[rows["path"] == p], "sq") for p in PATHS]
        mean: npt.NDArray[np.float64] = np.nanmean(np.stack(per_path), axis=0)
        return mean

    def selection_interval(self, examples: Examples, pooled: pd.DataFrame) -> tuple[float, float]:
        """95% user-bootstrap interval of `neg_brier` (minus the paths' mean Brier)."""
        rows = self._rows(examples, pooled)
        lo, hi = interval(self._brier_samples(self._boot(rows), rows))
        return -hi, -lo

    def difference_interval(
        self, examples: Examples, a: pd.DataFrame, b: pd.DataFrame
    ) -> tuple[float, float]:
        """95% paired user-bootstrap interval of `neg_brier`, run a minus run b."""
        rows_a, rows_b = self._rows(examples, a), self._rows(examples, b)
        if set(rows_a["user_id"]) != set(rows_b["user_id"]):
            raise ValueError("runs scored different users; they aren't comparable")
        boot = self._boot(rows_a)
        return interval(self._brier_samples(boot, rows_b) - self._brier_samples(boot, rows_a))

    def tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        lo, hi = self.difference_interval(examples, leader, other)
        return lo <= 0 <= hi

    def tiebreak_tied(self, examples: Examples, leader: pd.DataFrame, other: pd.DataFrame) -> bool:
        return False  # the band error breaks ties strictly

    def diagnostics(self, examples: Examples, pooled: pd.DataFrame) -> list[str]:
        """Calibration by status band and Brier by persona, per path (§5)."""
        rows = self._rows(examples, pooled)
        lines = ["| Path | Band | Goals | Met |", "| --- | --- | --- | --- |"]
        for path in PATHS:
            for band, met_rate, n in _bands(rows[rows["path"] == path]):
                lines.append(f"| {path} | {band} | {n} | {met_rate:.2f} |")
        lines += ["", "| Path | " + " | ".join(PERSONAS) + " |", "| --- |" + " --- |" * 3]
        for path in PATHS:
            r = rows[rows["path"] == path]
            cells = [f"{r.loc[r['persona'] == p, 'sq'].mean():.3f}" for p in PERSONAS]
            lines.append(f"| {path} | " + " | ".join(cells) + " |")
        return lines

    # --- Selection and gates -----------------------------------------------------------------

    def eligible(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> bool:
        """Below both baselines' validation Brier with a track record, and below a flat 50% on
        new goals. Naive pace isn't a floor for new goals: a goal created this month has a
        "pace" of its whole balance per month (review on #51)."""
        for path, floors in (("track", BASELINES), ("new", ("flat_50",))):
            mine = metrics.get(f"val_brier.{path}", float("nan"))
            for b in floors:
                if not mine < baselines.get(b, {}).get(f"val_brier.{path}", float("inf")):
                    return False
        return True

    def gates(
        self, metrics: Mapping[str, float], baselines: Mapping[str, Mapping[str, float]]
    ) -> list[Gate]:
        """§5, in order, on test users: Brier below the baselines; per persona below a flat 50%;
        calibration inside each band, overall and per persona; the range's coverage; RMSE.

        One reading of "within the user-bootstrap tolerance" (review on #51): a gate's target
        widens by half the width of the same metric's 95% interval on validation, which the run
        logged before any test scoring. A gate with no validation interval gets no tolerance."""

        def tolerance(name: str) -> float:
            lo, hi = metrics.get(f"val_{name}_lo"), metrics.get(f"val_{name}_hi")
            return 0.0 if lo is None or hi is None or np.isnan(lo) else (hi - lo) / 2

        gates = []
        for path, floors in (("track", BASELINES), ("new", ("flat_50",))):
            mine = metrics.get(f"test_brier.{path}", float("nan"))
            for b in floors:
                theirs = baselines.get(b, {}).get(f"test_brier.{path}")
                gates.append(
                    Gate(
                        f"brier_{path}_below_{b}",
                        theirs is not None and mine < theirs,
                        f"{mine:.3f} vs {theirs:.3f}" if theirs is not None else "no baseline run",
                    )
                )
        for path in PATHS:
            for persona in PERSONAS:
                name = f"brier.{path}.{persona}"
                value, tol = metrics.get(f"test_{name}", float("nan")), tolerance(name)
                gates.append(
                    Gate(
                        f"brier_{path}_{persona}_below_flat",
                        value <= FLAT + tol,
                        f"{value:.3f} vs {FLAT} + {tol:.3f} (validation tolerance)",
                        blocking=(path, persona) not in NON_BLOCKING,
                    )
                )
        for path in PATHS:
            for scope in ("", *(f".{p}" for p in PERSONAS)):
                for band, lo_p, hi_p in BANDS:
                    name = f"met_rate.{path}{scope}.{band}"
                    rate = metrics.get(f"test_{name}", float("nan"))
                    tol = tolerance(name)
                    label = f"calibrated_{path}{scope.replace('.', '_')}_{band}"
                    if np.isnan(rate):
                        gates.append(Gate(label, True, "no goals in this band"))
                        continue
                    gates.append(
                        Gate(
                            label,
                            lo_p - tol <= rate <= hi_p + tol,
                            f"met {rate:.2f} vs band {lo_p}-{hi_p} ± {tol:.2f} (validation)",
                        )
                    )
        for path in PATHS:
            coverage = metrics.get(f"test_coverage.{path}", float("nan"))
            gates.append(
                Gate(
                    f"coverage_{path}",
                    COVERAGE[0] <= coverage <= COVERAGE[1],
                    f"{coverage:.2f} vs {COVERAGE[0]}-{COVERAGE[1]}",
                )
            )
        return gates

    def serving_files(
        self,
        examples: Examples,
        pooled: pd.DataFrame,
        model_version: str,
        source: Mapping[str, str],
    ) -> dict[str, str]:
        return {}  # the forecast state is built nightly from data, not from a run (§7)

    def reproduction_configs(self) -> set[str]:
        return set()


def history_leaks(frame: pd.DataFrame) -> list[str]:
    """Every example's history must end at its `as_of_date`'s month: a month after it would let a
    model read the outcome (Technical Design, evaluation controls; FR-1 stage 9)."""
    late = []
    for r in frame[["example_id", "as_of_date", "history_json"]].to_dict("records"):
        history = parse_history(str(r["history_json"]))
        if len(history) and history.index[-1] > pd.Period(str(r["as_of_date"]), freq="M"):
            late.append(r["example_id"])
    return [f"{len(late)} examples see months after their as_of_date: {late[:3]}"] if late else []


def _covered(rows: pd.DataFrame) -> pd.Series:
    """Whether the realized balance by the target date fell inside the 80% range: the goal's own
    share run over the months that actually followed. NaN for a model without a share."""
    out: list[float] = []
    for r in rows[["share", "saved", "future_json", "range_lo", "range_hi"]].to_dict("records"):
        if pd.isna(r["share"]):
            out.append(np.nan)
            continue
        future = parse_history(str(r["future_json"])).to_numpy()
        realized = final_balance(float(r["saved"]), float(r["share"]), future)
        lo, hi = float(r["range_lo"]), float(r["range_hi"])
        out.append(float(lo - 0.005 <= realized <= hi + 0.005))
    return pd.Series(out, index=rows.index, dtype=float)


def _in_band(p: pd.Series, lo: float, hi: float) -> pd.Series:
    p = p.astype(float)
    return (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))


def _bands(rows: pd.DataFrame) -> list[tuple[str, float, int]]:
    out = []
    for band, lo, hi in BANDS:
        part = rows[_in_band(rows["p_goal_met"], lo, hi)]
        out.append((band, float(part["met"].mean()) if len(part) else float("nan"), len(part)))
    return out


def _outside(rate: float, lo: float, hi: float) -> float:
    """How far a band's met rate falls outside the band (0 inside; NaN with no goals in it)."""
    if np.isnan(rate):
        return 0.0
    return max(0.0, lo - rate, rate - hi)


register_task("goal_forecasting", GoalForecastingTask)
