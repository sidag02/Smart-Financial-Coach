"""Spending spikes in serving (FR-8 §8): the spikes file, and `detect_anomalies`' spikes half."""

import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.intelligence.spikes.batch import (
    METHOD_MODEL,
    METHOD_SIMPLE,
    SIMPLE_RULE,
    load_state,
)
from smart_financial_coach.intelligence.spikes.count import CountScorer
from smart_financial_coach.intelligence.spikes.reasons import whole

EVERYTHING = {"start_date": "2023-01-01", "end_date": "2026-09-30"}


def spikes(tools: Tools, **args: str) -> dict[str, Any]:
    found = tools.call("detect_anomalies", {**EVERYTHING, **args}).data
    spiking: dict[str, Any] = found["spending_spikes"]
    return spiking


def low_cutoff(ledger: Ledger) -> Ledger:
    """The ledger with a copy of its scorer cut low, so the small data has spikes to check; the
    product rules still apply. The shared state isn't touched."""
    assert ledger.spikes is not None
    model = copy.deepcopy(ledger.spikes.model)
    model.cutoff = 0.0
    return replace(ledger, spikes=replace(ledger.spikes, model=model))


def test_without_a_promoted_model_the_simple_rule_serves(spike_sources: DataSources) -> None:
    assert spike_sources.spikes is not None
    state = load_state(spike_sources.spikes)
    assert (state.method, state.version) == (METHOD_SIMPLE, SIMPLE_RULE)
    assert state.model.rate == 0.035
    assert isinstance(state.model.base, CountScorer)
    assert not state.model.base.use_season
    assert not state.model.base.use_income


def test_a_promoted_model_serves_as_promoted(promoted_spike_sources: DataSources) -> None:
    assert promoted_spike_sources.spikes is not None
    state = load_state(promoted_spike_sources.spikes)
    assert (state.method, state.version) == (METHOD_MODEL, "fr8-test")
    assert isinstance(state.model.base, CountScorer)
    assert state.model.base.use_season


@pytest.mark.parametrize("fixture", ["spike_sources", "promoted_spike_sources"])
def test_the_file_rebuilds_the_scorer_exactly(
    fixture: str, request: pytest.FixtureRequest, two_users: tuple[str, str]
) -> None:
    sources: DataSources = request.getfixturevalue(fixture)
    assert sources.spikes is not None
    ledger = Ledger.load(sources, two_users[0])
    state = load_state(sources.spikes)
    once = state.score(ledger.user_id, ledger.transactions, ledger.transactions)
    again = load_state(sources.spikes).score(
        ledger.user_id, ledger.transactions, ledger.transactions
    )
    pd.testing.assert_frame_equal(once, again)
    assert len(once) > 0


def test_spikes_are_the_users_own_and_match_the_spending_summary(
    spike_sources: DataSources, two_users: tuple[str, str]
) -> None:
    mine, theirs = (Tools(low_cutoff(Ledger.load(spike_sources, u))) for u in two_users)
    found, other = spikes(mine), spikes(theirs)
    assert found["spikes"]
    assert other["spikes"]

    assert found["status"] == other["status"] == "ok"
    assert found["method"] == METHOD_SIMPLE
    my_ids = {c["transaction_id"] for s in found["spikes"] for c in s["largest_charges"]}
    their_ids = {c["transaction_id"] for s in other["spikes"] for c in s["largest_charges"]}
    assert my_ids.isdisjoint(their_ids)
    assert my_ids <= set(mine.ledger.transactions["transaction_id"])
    for s in found["spikes"]:
        summary = mine.call(
            "get_spending_summary", {"start_date": s["period_start"], "end_date": s["period_end"]}
        ).data
        spent = {c["category"]: c["amount"] for c in summary["by_category"]}
        assert s["actual"] == pytest.approx(spent[s["category"]], abs=0.01)
        assert s["ratio"] >= 1.3
        assert s["usual_count"] >= 2
        assert len(s["largest_charges"]) <= 5
        assert s["other_purchases"] == s["count"] - len(s["largest_charges"])
        assert s["simple_rule"] is True
        assert s["reason"].startswith(f"You spent ${whole(s['actual']):,} on {s['category']}")


def test_corrections_move_the_spikes_with_the_charges(
    spike_sources: DataSources, two_users: tuple[str, str]
) -> None:
    ledger = low_cutoff(Ledger.load(spike_sources, two_users[0]))
    found = spikes(Tools(ledger))["spikes"]
    assert found
    category = found[0]["category"]
    t = ledger.transactions
    moved = t.assign(category=np.where(t["category"] == category, "Entertainment", t["category"]))
    # As the session sees it: every charge in that category corrected to another one
    tools = Tools(replace(ledger, transactions=moved))
    tools.base = ledger  # the model's categories, which the season table was built on
    assert all(s["category"] != category for s in spikes(tools)["spikes"])


def test_short_histories_and_months_in_progress_are_said(
    spike_sources: DataSources, two_users: tuple[str, str]
) -> None:
    tools = Tools(Ledger.load(spike_sources, two_users[0]))
    first = tools.ledger.transactions["day"].min()
    early = spikes(tools, start_date=first.isoformat(), end_date=first.isoformat())
    assert early["status"] == "too_short"
    assert early["spikes"] == []
    # Scored as of mid-September, September isn't over yet
    assert tools.ledger.spikes is not None
    mid = replace(tools.ledger.spikes, as_of="2026-09-15")
    tools = Tools(replace(tools.ledger, spikes=mid))
    later = spikes(tools, start_date="2026-09-01", end_date="2026-09-30")
    assert later["status"] == "month_in_progress"


def test_spikes_are_not_available_without_a_spikes_file(
    flagged_sources: DataSources, two_users: tuple[str, str]
) -> None:
    found = spikes(Tools(Ledger.load(flagged_sources, two_users[0])))
    assert found["status"] == "not_available"


def test_one_users_spikes_are_fast_enough_for_a_request(
    spike_sources: DataSources, two_users: tuple[str, str]
) -> None:
    tools = Tools(Ledger.load(spike_sources, two_users[0]))
    spikes(tools)  # warm
    started = time.perf_counter()
    spikes(tools)
    assert time.perf_counter() - started < 1.0  # NFR-5's 2 s is for the whole page


def test_the_file_holds_no_rows_of_any_user(spike_sources: DataSources) -> None:
    """Its schema: the scorer and the table's sums and counts, nothing per user or charge."""
    assert spike_sources.spikes is not None
    payload = json.loads(Path(spike_sources.spikes).read_text())
    assert set(payload) == {"meta", "model", "season_table", "presets"}
    assert set(payload["season_table"]) == {"category", "moy", "as_of", "total", "users"}
    assert set(payload["presets"]) == {"less", "balanced", "more"}  # three cutoffs (FR-9)
    assert set(payload["model"]) == {"spec", "fitted", "base_fitted"}
    assert payload["meta"]["categorizer_version"] == "stub"


def test_a_bundle_whose_predictions_dont_match_the_table_fails_to_load(
    spike_sources: DataSources, tmp_path: Path
) -> None:
    assert spike_sources.spikes is not None
    payload = json.loads(Path(spike_sources.spikes).read_text())
    payload["meta"]["categorizer_version"] = "another-categorizer"
    other = tmp_path / "spikes.json"
    other.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rebuild the bundle"):
        replace(spike_sources, spikes=other).spike_state()


def test_the_source_says_when_unusual_charges_arent_checked(
    spike_sources: DataSources, two_users: tuple[str, str]
) -> None:
    result = Tools(Ledger.load(spike_sources, two_users[0])).call("detect_anomalies", EVERYTHING)
    assert result.source.detail.startswith("unusual charges not available")


def test_the_served_fallback_is_the_rule_finalize_scores() -> None:
    """Decision 15: the simple count rule scored on test users is decision 12's fallback."""
    from smart_financial_coach.config import PROJECT_ROOT
    from smart_financial_coach.evaluation.experiment import load_experiment
    from smart_financial_coach.intelligence.models.registry import build
    from smart_financial_coach.intelligence.spikes.batch import _spec, simple_rule

    config = load_experiment(
        PROJECT_ROOT / "configs" / "experiments" / "spending_spikes" / "01_simple_count.yaml"
    )
    assert _spec(build(config.model_spec({}))) == _spec(simple_rule())
