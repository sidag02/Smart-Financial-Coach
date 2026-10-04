"""Simulated net-savings paths and the goal forecast built on them (FR-11 and FR-12 design, §2-§3).

Each user's forecast state is small (§7): a level, a month-of-year profile and the user's own
monthly deviations, widened by one spread factor. `simulate` turns it into paths of future
monthly net savings with a fixed seed, so the same goal and model version always give the same
answer (NFR-8). Every goal, saved or a draft, is arithmetic over those paths (§3).

What `fit` learns, all from model-visible months, never from outcomes ("Why nothing is tuned on
outcomes"):
- each persona's seasonal profile and pooled deviations (for users with short histories);
- the typical total allocation of savings to goals, from goals with a track record;
- the spread, so that the 80% range covers 80% of realized goal balances (§2): training goals
  whose target month lies inside their user's visible history, run on their own share over the
  months that followed. It reads realized balances, never a target or whether a goal was met.

A user's paths are simulated once per `as_of`, over a fixed horizon, with a seed from (user,
`as_of` month, model version) (§7). Every goal of that user runs over the same paths, so a draft
and the same goal once saved get identical numbers, and two goals share one future.

    model = PathsModel(seasonal=True).fit(training_rows)
    model.predict(goal_rows)  # contract-checked by the caller
"""

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.forecasting.contract import (
    DRAW_DOWN,
    INTERVAL,
    ON_TRACK,
    months_left,
    output_frame,
    parse_history,
    round_up_dollars,
    status_for,
    to_date,
)
from smart_financial_coach.intelligence.forecasting.savings import (
    SHARE_CAP,
    final_balance,
    infer_share,
    run_balance,
)
from smart_financial_coach.intelligence.models.base import BaseModel
from smart_financial_coach.intelligence.models.registry import register

Floats = npt.NDArray[np.float64]

MIN_TRACK_MONTHS = 3  # a share is inferred only from 3+ months of history (§3)
MIN_OWN_RESIDUALS = 6  # fewer of the user's own deviations: pool with the persona's (§2)
SEASONAL_FROM = 12  # months of history before a seasonal profile is used
ETS_FROM = 24  # exponential smoothing needs two full cycles (owner decision 3 on #50)
ETS_WINDOW = 36
NEXT = 6  # the point forecast reported for the RMSE metric: the next 6 months' total
HORIZON = 120  # months simulated per user: FR-10's furthest target, so any goal fits (§7)
MIN_SPREAD_GOALS = 50  # fewer realized goals to tune on: keep a spread of 1
SPREAD_PATHS = 200  # paths per goal while tuning the spread
SPREAD_RANGE = (0.25, 4.0)


@dataclass(frozen=True)
class ForecastState:
    """One user's forecast at one `as_of`: what's stored nightly per user (§7)."""

    level: float
    seasonal: tuple[float, ...]  # 12 dollar deviations, January first
    residuals: tuple[float, ...]
    spread: float
    months: int  # months of history it was fitted on
    end: pd.Period  # the last month of that history
    # Exponential smoothing's own state, to simulate from the fitted model (§4): the smoothing
    # weights (alpha, gamma), the last level, and the seasonal states by month of year
    ets: tuple[float, float, float, tuple[float, ...]] | None = None

    def point(self, months: int) -> Floats:
        """The point forecast for the `months` after `end`."""
        moy = [(self.end + i + 1).month for i in range(months)]
        return np.array([self.level + self.seasonal[m - 1] for m in moy])

    def simulate(self, months: int, n_paths: int, seed: int) -> Floats:
        """`n_paths` paths of the next `months` months, deterministic for a seed: the point plus
        resampled, scaled deviations; for exponential smoothing, each month's error also feeds
        its level and season forward, so its paths widen with the horizon."""
        rng = np.random.default_rng(seed)
        noise = self.spread * rng.choice(
            np.asarray(self.residuals), size=(n_paths, months), replace=True
        )
        if self.ets is None:
            paths: Floats = self.point(months)[None, :] + noise
            return paths
        alpha, gamma, level0, season0 = self.ets
        level = np.full(n_paths, level0)
        season = np.tile(np.asarray(season0), (n_paths, 1))
        out = np.empty((n_paths, months))
        for h in range(months):
            m = (self.end + h + 1).month - 1
            e = noise[:, h]
            out[:, h] = level + season[:, m] + e
            level = level + alpha * e
            season[:, m] = season[:, m] + gamma * e
        return out


@dataclass(frozen=True)
class PersonaPrior:
    """A persona's typical seasonal profile and deviations, as multiples of a user's scale (their
    mean absolute monthly net), so they fit any income."""

    profile: tuple[float, ...]  # 12 values
    residuals: tuple[float, ...]


def scale_of(history: Floats) -> float:
    return float(np.mean(np.abs(history))) if len(history) else 0.0


def _own_seasonal(y: Floats, end: pd.Period) -> tuple[Floats, Floats]:
    """Per month of year: the mean deviation from the trailing-12 mean, and how many years."""
    total, count = np.zeros(12), np.zeros(12)
    start = end - (len(y) - 1)
    for t in range(SEASONAL_FROM, len(y)):
        m = (start + t).month - 1
        total[m] += y[t] - y[t - 12 : t].mean()
        count[m] += 1
    mean: Floats = np.divide(total, count, out=np.zeros(12), where=count > 0)
    return mean, count


def _ets_state(y: Floats, end: pd.Period, spread: float) -> ForecastState:
    """Exponential smoothing with additive seasonality and no trend (§4). Its forecast repeats
    every 12 months, so it's stored exactly as a level and a month-of-year profile."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # a candidate's dependency

    recent = y[-ETS_WINDOW:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter on short, noisy series
        fit = ExponentialSmoothing(
            recent,
            trend=None,
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        ).fit()
    ahead = np.asarray(fit.forecast(12), dtype=float)
    level = float(ahead.mean())
    dev = np.zeros(12)
    season = np.zeros(12)
    states = np.asarray(fit.season, dtype=float)[-12:]  # states for the next 12 months, in order
    for i, value in enumerate(ahead):
        dev[(end + i + 1).month - 1] = value - level
        season[(end + i + 1).month - 1] = states[i]
    residuals = np.asarray(fit.resid, dtype=float)
    smoothing = (
        float(fit.params["smoothing_level"]),
        float(fit.params["smoothing_seasonal"]),
        float(np.asarray(fit.level, dtype=float)[-1]),
        tuple(float(v) for v in season),
    )
    return ForecastState(
        level=level,
        seasonal=tuple(float(v) for v in dev),
        residuals=tuple(float(v) for v in residuals),
        spread=spread,
        months=len(y),
        end=end,
        ets=smoothing,
    )


def fit_state(
    history: pd.Series,
    prior: PersonaPrior | None,
    *,
    window: int,
    seasonal: bool,
    shrink: float,
    spread: float,
    level_model: str = "mean",
) -> ForecastState:
    """A user's state from their monthly net savings up to `as_of` (§2). `level_model="ets"`
    uses exponential smoothing once there are two full years, and the mean-and-profile state
    before that."""
    y = history.to_numpy(dtype=float)
    end = history.index[-1] if len(history) else pd.Period("1970-01", freq="M")
    if level_model == "ets" and len(y) >= ETS_FROM:
        return _ets_state(y, end, spread)
    recent = y[-window:] if len(y) else np.zeros(1)
    level = float(recent.mean())
    scale = scale_of(y)
    dev = np.zeros(12)
    if seasonal and len(y) >= SEASONAL_FROM:
        own, years = _own_seasonal(y, end)
        prior_dollars = np.asarray(prior.profile) * scale if prior else np.zeros(12)
        weight = years / (years + shrink) if shrink > 0 else np.ones(12)
        dev = weight * own + (1 - weight) * prior_dollars
    start = end - (len(recent) - 1)
    moy = np.array([(start + i).month - 1 for i in range(len(recent))])
    residuals = recent - (level + dev[moy])
    if len(residuals) < MIN_OWN_RESIDUALS and prior is not None and scale > 0:
        residuals = np.concatenate([residuals, np.asarray(prior.residuals) * scale])
    if not len(residuals):
        residuals = np.zeros(1)
    return ForecastState(
        level=level,
        seasonal=tuple(float(v) for v in dev),
        residuals=tuple(float(v) for v in residuals),
        spread=spread,
        months=len(y),
        end=end,
    )


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def run_with_deposit(start: float, share: float, net: Floats, deposit: float) -> Floats:
    """Final balances when `deposit` is added each month on top of the share (§3: a top-up is
    money set aside on purpose, so it isn't scaled by the share)."""
    balance = np.full(net.shape[0], float(start))
    for m in range(net.shape[1]):
        balance = np.maximum(0.0, balance + share * net[:, m] + deposit)
    return balance


@register("goal_forecasting/paths")
class PathsModel(BaseModel):
    """The simulated-paths goal forecast. `seasonal=False` is the flat-level candidate."""

    def __init__(
        self,
        seasonal: bool = True,
        shrink: float = 2.0,
        window: int = 24,
        n_paths: int = 1000,
        coverage: float = 0.8,
        spread: float | None = None,
        level_model: str = "mean",
    ) -> None:
        if level_model not in ("mean", "ets"):
            raise ValueError(f"level_model must be mean or ets, not {level_model!r}")
        super().__init__(
            seasonal=seasonal,
            shrink=shrink,
            window=window,
            n_paths=n_paths,
            coverage=coverage,
            spread=spread,
            level_model=level_model,
        )
        self.level_model = level_model
        self.seasonal = seasonal
        self.shrink = shrink
        self.window = window
        self.n_paths = n_paths
        self.coverage = coverage
        self.spread = spread
        self.priors: dict[str, PersonaPrior] = {}
        self.typical_total = 0.5
        self.fitted_spread = 1.0 if spread is None else spread

    # --- Fitting ----------------------------------------------------------------------------

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Learns from the rows' histories and saved amounts only; `y` (outcomes) is ignored."""
        histories = _longest_histories(x)
        self.priors = self._persona_priors(histories)
        self.typical_total = _typical_total(x)
        tuned, n = (self.spread, 0) if self.spread is not None else self._tune_spread(x, histories)
        self.fitted_spread = tuned
        self.report = {
            "spread": tuned,
            "spread_goals": float(n),
            "typical_total": self.typical_total,
        }
        return self

    def _persona_priors(self, histories: pd.DataFrame) -> dict[str, PersonaPrior]:
        priors = {}
        for persona, part in histories.groupby("persona"):
            profiles, residuals = [], []
            for h in part["history"]:
                y = h.to_numpy(dtype=float)
                scale = scale_of(y)
                if scale <= 0:
                    continue
                if len(y) >= SEASONAL_FROM + 12:
                    own, _ = _own_seasonal(y, h.index[-1])
                    profiles.append(own / scale)
                recent = y[-self.window :]
                residuals.append((recent - recent.mean()) / scale)
            profile = np.median(np.stack(profiles), axis=0) if profiles else np.zeros(12)
            pooled = np.concatenate(residuals) if residuals else np.zeros(1)
            priors[str(persona)] = PersonaPrior(
                tuple(float(v) for v in profile), tuple(float(v) for v in pooled)
            )
        return priors

    def _tune_spread(self, x: pd.DataFrame, histories: pd.DataFrame) -> tuple[float, int]:
        """The spread at which the 80% range covers 80% of realized goal balances (§2).

        The goals: training goals with a track-record share whose target month lies inside their
        user's visible history (another of the user's rows reaches it). Each is run on its own
        share over the months that followed its `as_of`. Only realized balances are read; a
        target or whether a goal was met never is. Coverage grows with the spread, so it's found
        by bisection. Fewer than `MIN_SPREAD_GOALS` such goals keep a spread of 1."""
        longest = dict(zip(histories["user_id"], histories["history"], strict=True))
        cases = []
        for r in x[x["origin"] == "existing"].drop_duplicates("goal_id").to_dict("records"):
            share = _track_record_share(r)
            visible = longest.get(str(r["user_id"]))
            if share is None or visible is None:
                continue
            as_of, target = (
                pd.Period(to_date(r[c]), freq="M") for c in ("as_of_date", "target_date")
            )
            future = visible[(visible.index > as_of) & (visible.index <= target)]
            if target > visible.index[-1] or not len(future):
                continue
            realized = final_balance(float(r["saved"]), share, future.to_numpy())
            state = self.state_for(parse_history(str(r["history_json"])), str(r["persona"]), 1.0)
            seed = _seed("spread", r["goal_id"], self.version)
            cases.append((state, float(r["saved"]), share, len(future), realized, seed))
        if len(cases) < MIN_SPREAD_GOALS:
            return 1.0, len(cases)

        def coverage(spread: float) -> float:
            inside = 0
            for state, saved, share, months, realized, seed in cases:
                paths = replace(state, spread=spread).simulate(months, SPREAD_PATHS, seed)
                finals = run_balance(saved, share, paths)[:, -1]
                lo, hi = np.quantile(finals, INTERVAL)
                inside += int(lo - 0.005 <= realized <= hi + 0.005)
            return inside / len(cases)

        lo, hi = SPREAD_RANGE
        if coverage(hi) < self.coverage:
            return hi, len(cases)
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if coverage(mid) < self.coverage else (lo, mid)
        return round(hi, 3), len(cases)

    # --- Forecasting ------------------------------------------------------------------------

    def state_for(
        self, history: pd.Series, persona: str, spread: float | None = None
    ) -> ForecastState:
        return fit_state(
            history,
            self.priors.get(persona),
            window=self.window,
            seasonal=self.seasonal,
            shrink=self.shrink,
            spread=self.fitted_spread if spread is None else spread,
            level_model=self.level_model,
        )

    def paths_for(self, user_id: str, as_of: object, state: ForecastState) -> Floats:
        """A user's paths at an `as_of`: one future every goal of theirs shares (§7)."""
        month = pd.Period(to_date(as_of), freq="M")
        return state.simulate(HORIZON, self.n_paths, _seed(user_id, month, self.version))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        cache: dict[tuple[str, str], tuple[ForecastState, Floats]] = {}
        rows: list[dict[str, Any]] = []
        shares = _shares(x, self.typical_total)
        for r, (share, source) in zip(x.to_dict("records"), shares, strict=True):
            key = (str(r["user_id"]), str(r["as_of_date"]))
            if key not in cache:
                state = self.state_for(parse_history(str(r["history_json"])), str(r["persona"]))
                cache[key] = (state, self.paths_for(key[0], r["as_of_date"], state))
            state, paths = cache[key]
            rows.append(self._forecast(r, state, paths, share, source))
        return output_frame(rows, self.version)

    def _forecast(
        self,
        r: Mapping[Any, Any],
        state: ForecastState,
        user_paths: Floats,
        share: float,
        source: str,
    ) -> dict[str, Any]:
        saved, target = float(r["saved"]), float(r["target_amount"])
        left = months_left(to_date(r["as_of_date"]), to_date(r["target_date"]))
        reached = saved >= target
        paths = user_paths[:, :left]
        if left == 0:
            finals = np.full(self.n_paths, saved)
        else:
            finals = run_balance(saved, share, paths)[:, -1]
        p = float((finals >= target).mean())
        lo, mid, hi = (float(v) for v in np.quantile(finals, [INTERVAL[0], 0.5, INTERVAL[1]]))
        extra = None
        if not reached and p < ON_TRACK and left > 0:
            extra = self._extra_per_month(saved, share, paths, target)
        return {
            "example_id": r["example_id"],
            "status": status_for(p, reached),
            "p_goal_met": p,
            "projected_balance": mid,
            "range_lo": lo,
            "range_hi": hi,
            "gap": max(0.0, target - mid),
            "extra_per_month": extra,
            "share": share,
            "share_source": source,
            "net_next_6": float(state.point(NEXT).sum()),
            # A reached goal's drawdown note: 10% or more of paths end below the target
            "may_draw_down": bool((finals < target).mean() >= DRAW_DOWN) if reached else None,
        }

    @staticmethod
    def _extra_per_month(saved: float, share: float, paths: Floats, target: float) -> float:
        """The smallest whole-dollar monthly deposit that brings the chance to the on-track band
        (§3), by bisection over the same paths."""

        def p(deposit: float) -> float:
            return float((run_with_deposit(saved, share, paths, deposit) >= target).mean())

        lo, hi = 0.0, max(1.0, target / paths.shape[1])
        while p(hi) < ON_TRACK:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if p(mid) < ON_TRACK else (lo, mid)
        return round_up_dollars(hi)


def _longest_histories(x: pd.DataFrame) -> pd.DataFrame:
    """Each user's longest history in `x` (histories are prefixes of one series)."""
    best = x.assign(n=x["history_json"].str.len()).sort_values("n").groupby("user_id").tail(1)
    return pd.DataFrame(
        {
            "user_id": best["user_id"].to_numpy(),
            "persona": best["persona"].to_numpy(),
            "history": [parse_history(str(h)) for h in best["history_json"]],
        }
    )


def _track_record_share(r: Mapping[Any, Any]) -> float | None:
    """A generated goal's share, from $0 at creation (§3), or None without enough history."""
    if r["origin"] != "existing" or float(r["saved"]) <= 0:
        return None
    history = parse_history(str(r["history_json"]))
    created = pd.Period(to_date(r["created_date"]), freq="M")
    since = history[history.index >= created]
    if len(since) < MIN_TRACK_MONTHS:
        return None
    share = infer_share(since.to_numpy(), 0.0, float(r["saved"]))
    return None if share is None else min(SHARE_CAP, share)


def _entries_share(r: Mapping[Any, Any]) -> float | None:
    """A share from two of the user's own entries 3+ months apart, from the first (§3)."""
    first, first_at = r.get("first_saved"), r.get("first_saved_as_of")
    if r["origin"] != "yours" or first_at is None or pd.isna(first):
        return None
    a = pd.Period(to_date(first_at), freq="M")
    b = pd.Period(to_date(r["saved_as_of"]), freq="M")
    if (b - a).n < MIN_TRACK_MONTHS or float(r["saved"]) == float(first):
        return None
    history = parse_history(str(r["history_json"]))
    between = history[(history.index > a) & (history.index <= b)]
    share = infer_share(between.to_numpy(), float(first), float(r["saved"]))
    return None if share is None else min(SHARE_CAP, share)


def _typical_total(x: pd.DataFrame) -> float:
    """The typical total allocation: per goal set, its known goals' mean inferred share times
    the goals active alongside them, then the median over sets (§3). Goals only with a track
    record count; outcomes are never read."""
    totals = []
    track = x[x["origin"] == "existing"]
    for _, part in track.groupby("goal_set"):
        shares, active = [], []
        for r in part.to_dict("records"):
            s = _track_record_share(r)
            if s is not None:
                shares.append(s)
                active.append(int(r["active_goals"]))
        if shares:
            totals.append(float(np.mean(shares)) * float(np.median(active)))
    return float(np.clip(np.median(totals), 0.0, SHARE_CAP)) if totals else 0.5


def _shares(x: pd.DataFrame, typical_total: float) -> list[tuple[float, str]]:
    """Each row's share and its source (§3): its own track record or entries when it has them,
    otherwise the typical total split equally over the active goals. A set's shares never total
    more than 1: only the typical ones are scaled down."""
    out: list[tuple[float, str]] = []
    for r in x.to_dict("records"):
        if (s := _track_record_share(r)) is not None:
            out.append((s, "track_record"))
        elif (s := _entries_share(r)) is not None:
            out.append((s, "your_entries"))
        else:
            out.append((typical_total / max(1, int(r["active_goals"])), "typical"))
    # The cap, per goal set and as_of: measured shares stay, typical ones shrink to fit
    keys = list(zip(x["goal_set"].astype(str), x["as_of_date"].astype(str), strict=True))
    for key in set(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        total = sum(out[i][0] for i in idx)
        if total <= SHARE_CAP:
            continue
        measured = sum(out[i][0] for i in idx if out[i][1] != "typical")
        room = max(0.0, SHARE_CAP - measured)
        typical = sum(out[i][0] for i in idx if out[i][1] == "typical")
        for i in idx:
            if out[i][1] == "typical" and typical > 0:
                out[i] = (out[i][0] * room / typical, "typical")
    return out
