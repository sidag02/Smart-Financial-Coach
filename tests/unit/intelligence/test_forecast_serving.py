"""FR-11/FR-12 §7: the nightly forecast states, the forecasts file, and serving forecasting with
the same code and numbers as the scored model."""

import stat
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.data.store import load_meta, load_transactions, load_users
from smart_financial_coach.evaluation.cli import model_main
from smart_financial_coach.intelligence.forecasting.batch import GoalForecaster, forecast_dataset
from smart_financial_coach.intelligence.forecasting.contract import INPUT_COLUMNS, history_json
from smart_financial_coach.intelligence.forecasting.paths import ForecastState, PathsModel
from smart_financial_coach.intelligence.forecasting.savings import monthly_net
from smart_financial_coach.intelligence.forecasting.states import (
    load_forecasts,
    state_from_json,
    state_to_json,
)
from smart_financial_coach.intelligence.models.contract import ContractError
from smart_financial_coach.intelligence.service import load_service

AS_OF = "2026-09-30"


def goal_row(user_id: str, history: str, **overrides: object) -> dict[str, object]:
    return {
        "example_id": "g1",
        "goal_id": "g1",
        "user_id": user_id,
        "goal_set": user_id,
        "persona": "",
        "as_of_date": AS_OF,
        "created_date": "2026-01-10",
        "target_amount": 6000.0,
        "target_date": "2027-06-30",
        "saved": 1500.0,
        "saved_as_of": AS_OF,
        "first_saved": float("nan"),
        "first_saved_as_of": None,
        "origin": "existing",
        "active_goals": 2,
        "set_goals": 2,
        "history_json": history,
    } | overrides


def user_history(sources: DataSources, user_id: str) -> str:
    txns = load_transactions(sources.dataset, user_id=user_id)
    return history_json(monthly_net(txns, pd.Timestamp(AS_OF).date()))


@pytest.fixture(scope="module")
def forecaster(forecast_sources: DataSources) -> GoalForecaster:
    assert forecast_sources.forecasts is not None
    return GoalForecaster.load(forecast_sources.forecasts)


def test_states_round_trip_through_json() -> None:
    plain = ForecastState(
        level=812.5,
        seasonal=tuple(float(i) for i in range(12)),
        residuals=(-40.0, 0.0, 55.5),
        spread=1.04,
        months=30,
        end=pd.Period("2026-09", freq="M"),
    )
    ets = replace(plain, ets=(0.2, 0.1, 790.0, tuple(float(-i) for i in range(12))))
    mixed = replace(
        plain,
        mixture=((0.7, tuple([1.0] * 12)), (0.3, tuple([-2.0] * 12))),
        slope=4.5,
        lead=11.5,
        damping=0.95,
    )
    for state in (plain, ets, mixed):
        assert state_from_json(state_to_json(state)) == state


def test_the_batch_stores_every_user_at_the_datasets_last_day(
    forecast_sources: DataSources, forecast_artifacts: Path
) -> None:
    assert forecast_sources.forecasts is not None
    stored = load_forecasts(forecast_sources.forecasts)
    users = set(load_users(forecast_sources.dataset)["user_id"])

    assert set(stored.states) == users
    assert stored.meta["as_of"] == load_meta(forecast_sources.dataset)["calendar_end"] == AS_OF
    assert {s.end for s in stored.states.values()} == {pd.Period(AS_OF, freq="M")}
    assert stored.meta["model_version"] == stored.model["version"] == "fr11-test"
    # Readable by everyone: the image serves the bundle as another user (#26)
    mode = stat.S_IMODE(forecast_sources.forecasts.stat().st_mode)
    assert mode & 0o044 == 0o044


def test_serving_gives_the_promoted_models_own_numbers(
    forecast_sources: DataSources, forecast_artifacts: Path, forecaster: GoalForecaster
) -> None:
    """The model rebuilt from the file, on the stored state, forecasts exactly as the promoted
    model does from the history (NFR-8; the model the gates passed is the one served)."""
    promoted = load_service("goal_forecasting", forecast_artifacts)
    users = load_users(forecast_sources.dataset)
    for user_id, persona in users[["user_id", "persona"]].head(4).itertuples(index=False):
        history = user_history(forecast_sources, user_id)
        rows = pd.DataFrame(
            [
                goal_row(user_id, history, persona=persona),
                goal_row(
                    user_id,
                    history,
                    persona=persona,
                    example_id="g2",
                    goal_id="g2",
                    origin="yours",
                    created_date=AS_OF,
                ),
            ],
            columns=list(INPUT_COLUMNS),
        )
        pd.testing.assert_frame_equal(forecaster.forecast(rows), promoted.predict(rows))
    assert forecaster.version == promoted.version == "fr11-test"


def test_a_state_from_another_month_is_refused(
    forecast_sources: DataSources, forecaster: GoalForecaster
) -> None:
    user_id = next(iter(forecaster.states))
    row = goal_row(user_id, user_history(forecast_sources, user_id), as_of_date="2026-10-31")
    with pytest.raises(ValueError, match="ends 2026-09, not at 2026-10-31"):
        forecaster.forecast(pd.DataFrame([row], columns=list(INPUT_COLUMNS)))


def test_a_broken_forecast_is_refused(
    forecast_sources: DataSources, forecaster: GoalForecaster, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = next(iter(forecaster.states))
    rows = pd.DataFrame(
        [goal_row(user_id, user_history(forecast_sources, user_id))], columns=list(INPUT_COLUMNS)
    )
    good = PathsModel.predict_with_states

    def broken(self: PathsModel, x: pd.DataFrame, states: object) -> pd.DataFrame:
        return good(self, x, states).assign(p_goal_met=1.5)  # type: ignore[arg-type]

    monkeypatch.setattr(PathsModel, "predict_with_states", broken)
    with pytest.raises(ContractError, match="p_goal_met outside 0-1"):
        forecaster.forecast(rows)


def test_the_monthly_series_ends_where_the_forecast_does(
    forecast_sources: DataSources, forecaster: GoalForecaster
) -> None:
    user_id = next(iter(forecaster.states))
    rows = pd.DataFrame(
        [goal_row(user_id, user_history(forecast_sources, user_id))], columns=list(INPUT_COLUMNS)
    )
    out = forecaster.forecast(rows).iloc[0]
    monthly = forecaster.monthly(user_id, 1500.0, float(out["share"]), 9)  # Oct 2026 - Jun 2027

    assert [m["month"] for m in (monthly[0], monthly[-1])] == ["2026-10", "2027-06"]
    last = monthly[-1]
    assert (last["low"], last["median"], last["high"]) == pytest.approx(
        (out["range_lo"], out["projected_balance"], out["range_hi"])
    )
    assert all(m["low"] <= m["median"] <= m["high"] for m in monthly)
    # The same paths every time (NFR-8)
    assert np.array_equal(
        forecaster.paths(user_id),
        forecaster.model.paths_for(user_id, forecaster.as_of, forecaster.states[user_id]),
    )


def test_cli_writes_forecast_states(
    small_sqlite: Path, forecast_artifacts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "forecasts.json"
    args = ["--data", str(small_sqlite), "--out", str(out), "--artifacts-dir"]
    args.append(str(forecast_artifacts))

    assert model_main(["predict", "--task", "goal_forecasting", *args]) == 0
    assert "forecast states from model fr11-test" in capsys.readouterr().out
    assert model_main(["predict", "--task", "goal_forecasting", *args]) == 1  # exists
    assert model_main(["predict", "--task", "goal_forecasting", *args, "--overwrite"]) == 0


def test_the_batch_needs_a_promoted_model(small_sqlite: Path, flag_artifacts: Path) -> None:
    with pytest.raises(Exception, match="no promoted model"):
        forecast_dataset(small_sqlite, small_sqlite.parent / "x.json", artifacts_dir=flag_artifacts)
