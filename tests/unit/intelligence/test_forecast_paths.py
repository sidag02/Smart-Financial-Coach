"""FR-11/FR-12 §2-§3: forecast states, simulated paths, shares by source, the cap, the top-up,
and fitting that never reads outcomes."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.intelligence.forecasting.contract import (
    CONTRACT,
    INPUT_COLUMNS,
    history_json,
    parse_history,
)
from smart_financial_coach.intelligence.forecasting.paths import (
    PathsModel,
    PersonaPrior,
    _shares,
    _typical_total,
    fit_state,
    run_with_deposit,
)
from smart_financial_coach.intelligence.models.contract import Checked

PRIOR = PersonaPrior(profile=tuple([0.0] * 12), residuals=(-0.5, 0.0, 0.5))


def series(values: list[float], start: str = "2024-01") -> pd.Series:
    return pd.Series(values, index=pd.period_range(start, periods=len(values), freq="M"))


def row(**overrides: object) -> dict[str, object]:
    history = series([800.0 + 300 * np.sin(i) for i in range(24)])
    base = {
        "example_id": "g1:track",
        "goal_id": "g1",
        "user_id": "u1",
        "goal_set": "u1:0:track",
        "persona": "young_professional",
        "as_of_date": "2025-12-31",
        "created_date": "2025-01-10",
        "target_amount": 6000.0,
        "target_date": "2026-09-30",
        "saved": 3000.0,
        "saved_as_of": "2025-12-31",
        "first_saved": float("nan"),
        "first_saved_as_of": None,
        "origin": "existing",
        "active_goals": 1,
        "set_goals": 1,
        "history_json": history_json(history),
    }
    return base | overrides


def frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(INPUT_COLUMNS))


@pytest.mark.parametrize("months", [1, 5, 11, 13, 30])
def test_states_fall_back_on_short_histories(months: int) -> None:
    h = series([500.0 + 10 * i for i in range(months)])
    for level_model in ("mean", "ets"):
        state = fit_state(
            h, PRIOR, window=24, seasonal=True, shrink=2.0, spread=1.0, level_model=level_model
        )
        assert state.months == months
        assert len(state.residuals) >= min(months, 6)
        paths = state.simulate(6, 50, seed=1)
        assert paths.shape == (50, 6)
        assert np.isfinite(paths).all()
    flat = fit_state(h, PRIOR, window=24, seasonal=False, shrink=2.0, spread=1.0)
    assert flat.seasonal == tuple([0.0] * 12)


def test_paths_are_identical_for_a_seed_and_scale_with_the_spread() -> None:
    h = series([1000.0, -200.0, 700.0, 1500.0, 300.0, 900.0] * 3)
    narrow = fit_state(h, PRIOR, window=24, seasonal=False, shrink=2.0, spread=1.0)
    wide = fit_state(h, PRIOR, window=24, seasonal=False, shrink=2.0, spread=2.0)
    assert np.array_equal(narrow.simulate(12, 100, 7), narrow.simulate(12, 100, 7))
    assert np.allclose(
        wide.simulate(12, 100, 7) - wide.point(12),
        2 * (narrow.simulate(12, 100, 7) - narrow.point(12)),
    )


def test_shares_by_source_and_the_cap() -> None:
    track = row()
    new = row(example_id="g1:new", origin="yours", created_date="2025-12-31")
    entries = row(
        example_id="g2:new",
        goal_id="g2",
        origin="yours",
        first_saved=1000.0,
        first_saved_as_of="2025-06-30",
        saved=2500.0,
    )
    same_entry = row(
        example_id="g3:new",
        goal_id="g3",
        origin="yours",
        first_saved=2500.0,
        first_saved_as_of="2025-06-30",
        saved=2500.0,  # never changed: no information
    )
    zero = row(example_id="g4:track", goal_id="g4", saved=0.0)
    sources = [s for _, s in _shares(frame(track, new, entries, same_entry, zero), 0.66)]
    assert sources == ["track_record", "typical", "your_entries", "typical", "typical"]

    # Two goals in one set, one of them typical: the typical share shrinks so the set totals 1
    greedy = row(goal_id="g5", example_id="g5:track", saved=20000.0)  # a share near the cap
    newcomer = row(goal_id="g6", example_id="g6:track", origin="yours", active_goals=2)
    shares = _shares(frame(greedy, newcomer), 0.66)
    assert shares[0][1] == "track_record"
    assert shares[1][1] == "typical"
    assert shares[0][0] + shares[1][0] <= 1.0 + 1e-9


def test_a_typical_share_is_the_total_over_the_active_goals() -> None:
    new = row(origin="yours", created_date="2025-12-31", active_goals=2)
    ((share, source),) = _shares(frame(new), 0.662)
    assert source == "typical"
    assert share == pytest.approx(0.331)


def test_the_top_up_brings_the_goal_on_track() -> None:
    model = PathsModel(seasonal=False, spread=1.0, n_paths=400).fit(frame(row()))
    out = Checked(model, CONTRACT).predict(frame(row(target_amount=30000.0)))
    r = out.iloc[0]
    assert r["status"] == "off_track"
    extra = r["extra_per_month"]
    assert extra > 0
    # Re-run the same paths with the top-up: on track, and a dollar less isn't
    state = model.state_for(parse_history(str(row()["history_json"])), "young_professional")
    paths = model.paths_for("u1", "2025-12-31", state)[:, :9]
    share = float(r["share"])
    assert (run_with_deposit(3000.0, share, paths, extra) >= 30000.0).mean() >= 0.7
    assert (run_with_deposit(3000.0, share, paths, extra - 1) >= 30000.0).mean() < 0.7


def test_reached_goals_and_the_contract() -> None:
    model = PathsModel(seasonal=True, spread=1.0, n_paths=200).fit(frame(row()))
    out = Checked(model, CONTRACT).predict(
        frame(row(saved=7000.0), row(example_id="g1:new", origin="yours"))
    )
    assert out.iloc[0]["status"] == "reached"
    assert pd.isna(out.iloc[0]["extra_per_month"])
    assert out["net_next_6"].notna().all()


def test_fitting_never_reads_outcomes() -> None:
    rows = frame(
        row(), row(goal_id="g2", example_id="g2:track", user_id="u2", goal_set="u2:0:track")
    )
    a = PathsModel(seasonal=True).fit(rows, pd.Series([1, 0]))
    b = PathsModel(seasonal=True).fit(rows, pd.Series([0, 1]))
    c = PathsModel(seasonal=True).fit(rows, None)
    assert a.fitted_spread == b.fitted_spread == c.fitted_spread
    assert a.typical_total == b.typical_total == c.typical_total
    assert Checked(a, CONTRACT).predict(rows).equals(Checked(b, CONTRACT).predict(rows))


def test_level_model_is_checked() -> None:
    with pytest.raises(ValueError, match="level_model"):
        PathsModel(level_model="arima")


def test_a_draft_and_the_saved_goal_share_one_future() -> None:
    """Paths come from (user, as_of month, model version), not the goal (§7): the same goal under
    another id, as a draft would have, gets identical numbers."""
    model = PathsModel(seasonal=True, spread=1.0, n_paths=300).fit(frame(row()))
    saved, draft = row(), row(goal_id="draft", example_id="draft:track")
    out = Checked(model, CONTRACT).predict(frame(saved, draft))
    columns = ["p_goal_met", "projected_balance", "range_lo", "range_hi", "extra_per_month"]
    assert out.iloc[0][columns].equals(out.iloc[1][columns])


def test_ets_paths_widen_with_the_horizon() -> None:
    rng = np.random.default_rng(3)
    h = series(list(900 + 300 * np.sin(np.arange(36) / 12 * 2 * np.pi) + rng.normal(0, 250, 36)))
    ets = fit_state(h, PRIOR, window=24, seasonal=True, shrink=2.0, spread=1.0, level_model="ets")
    assert ets.ets is not None
    paths = ets.simulate(24, 2000, seed=5)
    assert paths[:, :12].mean() == pytest.approx(ets.point(12).mean(), rel=0.1, abs=60)
    if ets.ets[0] > 0.01:  # errors feed the level forward: later months spread wider
        assert paths[:, 23].std() > paths[:, 0].std()


def test_the_spread_is_tuned_on_realized_goal_balances() -> None:
    """Goals whose target lies inside the user's visible history (another row of theirs reaches
    it) are run on their own share over the months that followed; their count is reported."""
    long = series([700.0 + 400 * np.sin(i / 2) for i in range(36)], start="2023-10")
    rows = []
    for k in range(60):
        as_of = long.index[14 + k % 10]
        rows.append(
            row(
                example_id=f"g{k}:track",
                goal_id=f"g{k}",
                user_id="u1",
                goal_set=f"u1:{k}:track",
                as_of_date=str(as_of.end_time.date()),
                created_date="2023-11-15",
                saved_as_of=str(as_of.end_time.date()),
                target_date=str((as_of + 6).end_time.date()),
                saved=2000.0 + 10 * k,
                history_json=history_json(long[long.index <= as_of]),
            )
        )
    rows.append(row(example_id="last:track", goal_id="last", history_json=history_json(long)))
    model = PathsModel(seasonal=False, n_paths=100).fit(frame(*rows))
    assert model.report["spread_goals"] >= 50
    assert 0.25 <= model.report["spread"] <= 4.0


def test_a_reached_goal_on_a_falling_future_may_draw_down() -> None:
    falling = history_json(series([-400.0 + 50 * np.sin(i) for i in range(24)]))
    model = PathsModel(seasonal=False, spread=1.0, n_paths=300).fit(frame(row()))
    out = Checked(model, CONTRACT).predict(
        frame(
            row(saved=6100.0, history_json=falling),  # just past the target, losing money
            row(example_id="g2:track", goal_id="g2", user_id="u2", saved=9000.0),  # well clear
        )
    )
    assert out.iloc[0]["status"] == "reached"
    assert out.iloc[0]["may_draw_down"] is True
    assert out.iloc[1]["may_draw_down"] is False


def test_the_typical_total_sums_a_sets_shares_over_all_its_goals() -> None:
    """Per set, the measured goals' mean share times every goal in the set, created before or
    after this as_of; the median over sets (§3, review on #52)."""
    a = row(goal_id="a", example_id="a:track", goal_set="u1:0:track", set_goals=2, active_goals=1)
    b = row(goal_id="b", example_id="b:track", goal_set="u1:0:track", set_goals=2, active_goals=2)
    share = _shares(frame(a), 0.0)[0][0]  # each one's own track-record share
    assert _typical_total(frame(a, b)) == pytest.approx(min(1.0, 2 * share))


def test_predictions_never_read_set_goals() -> None:
    model = PathsModel(seasonal=True, spread=1.0, n_paths=200).fit(frame(row()))
    new = row(example_id="g1:new", origin="yours", created_date="2025-12-31", active_goals=1)
    one = Checked(model, CONTRACT).predict(frame(new | {"set_goals": 1}))
    many = Checked(model, CONTRACT).predict(frame(new | {"set_goals": 5}))
    assert one.equals(many)


def mixture_rows() -> pd.DataFrame:
    """Two users per persona with different shapes, so the weights have something to learn."""
    rows = []
    rng = np.random.default_rng(3)
    shapes: dict[str, Callable[[int], float]] = {
        "family_budgeter": lambda i: 900 - 700 * (i % 12 == 7) - 500 * (i % 12 == 11),
        "freelancer": lambda i: 600 + 1500 * rng.standard_normal(),
        "young_professional": lambda i: 700 + 40 * rng.standard_normal(),
    }
    for persona, shape in shapes.items():
        for u in range(2):
            history = series([float(shape(i)) for i in range(36)], start="2023-01")
            user = f"{persona[:2]}{u}"
            rows.append(
                row(
                    example_id=f"{user}:track",
                    goal_id=f"g_{user}",
                    user_id=user,
                    goal_set=f"{user}:0:track",
                    persona=persona,
                    as_of_date="2025-12-31",
                    history_json=history_json(history),
                )
            )
    return frame(*rows)


def test_the_persona_mixture_never_reads_a_persona() -> None:
    x = mixture_rows()
    model = PathsModel(seasonal=True, personas="mixture", n_paths=200, spread=1.0).fit(x)
    out = Checked(model, CONTRACT).predict(x)
    blank = Checked(model, CONTRACT).predict(x.assign(persona=""))
    pd.testing.assert_frame_equal(out, blank)
    # The label model does read it: an unknown persona gets no prior
    labelled = PathsModel(seasonal=True, n_paths=200, spread=1.0).fit(x)
    assert not labelled.predict(x).equals(labelled.predict(x.assign(persona="")))


def test_a_mixture_state_weighs_every_personas_profile() -> None:
    x = mixture_rows()
    model = PathsModel(seasonal=True, personas="mixture", n_paths=200, spread=1.0).fit(x)
    state = model.state_for(parse_history(str(x["history_json"].iloc[0])), "")
    assert state.mixture is not None
    assert len(state.mixture) == 3
    weights = np.array([w for w, _ in state.mixture])
    assert weights.sum() == pytest.approx(1.0)
    mean = sum(w * np.asarray(dev) for w, dev in state.mixture)
    assert np.asarray(state.seasonal) == pytest.approx(mean)
    paths = state.simulate(24, 500, seed=1)
    assert np.array_equal(paths, state.simulate(24, 500, seed=1))  # NFR-8


def test_the_trend_follows_a_raise_and_is_damped() -> None:
    rising = series([500.0 + 20 * i for i in range(24)])  # $20 more each month
    flat = fit_state(rising, PRIOR, window=24, seasonal=False, shrink=2.0, spread=1.0)
    trended = fit_state(
        rising, PRIOR, window=24, seasonal=False, shrink=2.0, spread=1.0, damping=0.95
    )
    assert flat.slope == 0.0
    assert np.allclose(flat.point(12), rising.mean())
    # Shrunk halfway toward the prior's flat slope with two years seen: $10 a month
    assert trended.slope == pytest.approx(10.0)
    ahead = trended.point(60)
    assert ahead[0] > rising.mean() + 10 * 11.5  # past the window's center
    steps = np.diff(ahead)
    assert (steps > 0).all()
    assert steps[-1] < steps[0] / 5  # damped: it levels off
