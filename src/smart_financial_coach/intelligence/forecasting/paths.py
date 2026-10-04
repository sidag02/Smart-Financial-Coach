"""Simulated net-savings paths and the goal forecast built on them (FR-11 and FR-12 design, §2-§3).

Each user's forecast state is small (§7): a level, a month-of-year profile and the user's own
monthly deviations, widened by one spread factor. `simulate` turns it into paths of future
monthly net savings with a fixed seed, so the same goal and model version always give the same
answer (NFR-8). Every goal, saved or a draft, is arithmetic over those paths (§3).

What `fit` learns, all from model-visible months, never from outcomes ("Why nothing is tuned on
outcomes"):
- each persona's seasonal profile and pooled deviations (for users with short histories);
- the typical total allocation of savings to goals, from goals with a track record;
- the spread, so that 80% ranges of the coming months' net savings cover 80% of what followed,
  backtested inside the training users' own histories (target-free; §2).

    model = PathsModel(seasonal=True).fit(training_rows)
    model.predict(goal_rows)  # contract-checked by the caller
"""

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.forecasting.contract import (
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
_SPREADS = np.round(np.arange(0.5, 3.0001, 0.025), 3)


@dataclass(frozen=True)
class ForecastState:
    """One user's forecast at one `as_of`: what's stored nightly per user (§7)."""

    level: float
    seasonal: tuple[float, ...]  # 12 dollar deviations, January first
    residuals: tuple[float, ...]
    spread: float
    months: int  # months of history it was fitted on
    end: pd.Period  # the last month of that history

    def point(self, months: int) -> Floats:
        """The point forecast for the `months` after `end`."""
        moy = [(self.end + i + 1).month for i in range(months)]
        return np.array([self.level + self.seasonal[m - 1] for m in moy])

    def simulate(self, months: int, n_paths: int, seed: int) -> Floats:
        """`n_paths` paths of the next `months` months: the point plus resampled, scaled
        deviations. Deterministic for a seed."""
        rng = np.random.default_rng(seed)
        noise = rng.choice(np.asarray(self.residuals), size=(n_paths, months), replace=True)
        paths: Floats = self.point(months)[None, :] + self.spread * noise
        return paths


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
    for i, value in enumerate(ahead):
        dev[(end + i + 1).month - 1] = value - level
    residuals = np.asarray(fit.resid, dtype=float)
    return ForecastState(
        level=level,
        seasonal=tuple(float(v) for v in dev),
        residuals=tuple(float(v) for v in residuals),
        spread=spread,
        months=len(y),
        end=end,
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
        self.fitted_spread = (
            self.spread if self.spread is not None else self._tune_spread(histories)
        )
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

    def _tune_spread(self, histories: pd.DataFrame) -> float:
        """The spread at which 80% ranges of the next 3-12 months' total net savings cover 80% of
        what followed, backtested inside the training users' histories. Reads only months the
        model is allowed to see; never an outcome or a target."""
        lo_q, hi_q = INTERVAL
        cases = []  # (realized - point, noise quantile low, noise quantile high) per case
        for r in histories.to_dict("records"):
            h: pd.Series = r["history"]
            prior = self.priors.get(str(r["persona"]))
            for origin in range(SEASONAL_FROM, len(h) - 3, 3):
                state = fit_state(
                    h.iloc[:origin],
                    prior,
                    window=self.window,
                    seasonal=self.seasonal,
                    shrink=self.shrink,
                    spread=1.0,
                    level_model=self.level_model,
                )
                for months in (3, 6, 12):
                    if origin + months > len(h):
                        continue
                    realized = float(h.iloc[origin : origin + months].sum())
                    rng = np.random.default_rng(_seed("spread", r["user_id"], origin, months))
                    noise = rng.choice(np.asarray(state.residuals), size=(400, months)).sum(axis=1)
                    q_lo, q_hi = np.quantile(noise, [lo_q, hi_q])
                    cases.append((realized - float(state.point(months).sum()), q_lo, q_hi))
        if not cases:
            return 1.0
        err, q_lo, q_hi = (np.array(c) for c in zip(*cases, strict=True))
        coverage = np.array(
            [float(((err >= s * q_lo) & (err <= s * q_hi)).mean()) for s in _SPREADS]
        )
        return float(_SPREADS[int(np.argmin(np.abs(coverage - self.coverage)))])

    # --- Forecasting ------------------------------------------------------------------------

    def state_for(self, history: pd.Series, persona: str) -> ForecastState:
        return fit_state(
            history,
            self.priors.get(persona),
            window=self.window,
            seasonal=self.seasonal,
            shrink=self.shrink,
            spread=self.fitted_spread,
            level_model=self.level_model,
        )

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        states: dict[tuple[str, str], ForecastState] = {}
        rows: list[dict[str, Any]] = []
        shares = _shares(x, self.typical_total)
        for r, (share, source) in zip(x.to_dict("records"), shares, strict=True):
            key = (str(r["user_id"]), str(r["as_of_date"]))
            if key not in states:
                states[key] = self.state_for(
                    parse_history(str(r["history_json"])), str(r["persona"])
                )
            rows.append(self._forecast(r, states[key], share, source))
        return output_frame(rows, self.version)

    def _forecast(
        self, r: Mapping[Any, Any], state: ForecastState, share: float, source: str
    ) -> dict[str, Any]:
        saved, target = float(r["saved"]), float(r["target_amount"])
        left = months_left(to_date(r["as_of_date"]), to_date(r["target_date"]))
        reached = saved >= target
        if left == 0:
            finals = np.full(self.n_paths, saved)
            paths = np.zeros((self.n_paths, 0))
        else:
            paths = state.simulate(left, self.n_paths, _seed(r["goal_id"], self.version))
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
    return min(SHARE_CAP, infer_share(since.to_numpy(), 0.0, float(r["saved"])))


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
    return min(SHARE_CAP, infer_share(between.to_numpy(), float(first), float(r["saved"])))


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
