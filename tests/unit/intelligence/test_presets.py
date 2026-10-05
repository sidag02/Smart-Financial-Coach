"""Alert sensitivity presets (FR-9 §1, §2): placing them, the warm-up rule, flag files and spike
states that carry them, and the levels serving shows."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import Tools
from smart_financial_coach.data import store
from smart_financial_coach.data.flags import load_flag_presets, load_flags
from smart_financial_coach.evaluation.cli import model_main
from smart_financial_coach.intelligence.anomaly.batch import flag_dataset
from smart_financial_coach.intelligence.anomaly.contract import evidence_errors
from smart_financial_coach.intelligence.models.artifact import ArtifactError
from smart_financial_coach.intelligence.presets import (
    LEVELS,
    PRESETS_FILE,
    Presets,
    charges_after_warmup,
    check_level,
    load_presets,
    periods_after_warmup,
    place,
    warmup_end,
    write_presets,
)
from smart_financial_coach.intelligence.spikes.batch import (
    SIMPLE_RULE_RATE,
    load_state,
)
from smart_financial_coach.intelligence.spikes.threshold import user_months

EVERYTHING = {"start_date": "2023-01-01", "end_date": "2026-09-30"}


# --- Placing -------------------------------------------------------------------------------


def test_less_and_more_flag_half_and_twice_balanced_after_the_warm_up() -> None:
    rng = np.random.default_rng(0)
    score = rng.normal(size=1000)
    post = np.arange(1000) >= 100  # the first 100 rows are in the warm-up
    ids = np.array([f"t{i:04d}" for i in range(1000)])
    balanced = float(np.sort(score[post])[::-1][99])  # flags 100 post-warm-up rows

    presets = place(score, post, ids, balanced)

    assert presets.cutoffs["balanced"] == balanced
    assert ((score >= presets.cutoff("less")) & post).sum() == 50
    assert ((score >= presets.cutoff("more")) & post).sum() == 200


def test_ties_are_broken_by_id_and_never_cross_balanced() -> None:
    score = np.array([3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.5, -np.inf])
    post = np.ones(len(score), dtype=bool)
    ids = np.array(list("abcdefgh"))

    presets = place(score, post, ids, balanced=2.0)  # 4 flags at Balanced

    assert presets.cutoff("less") == 2.0  # the 2nd highest is tied with Balanced
    assert presets.cutoff("more") == 0.5  # 8 wanted; only 7 can be flagged
    assert presets.cutoff("less") >= presets.cutoff("balanced") >= presets.cutoff("more")


def test_cutoffs_out_of_order_are_refused() -> None:
    with pytest.raises(ValueError, match="out of order"):
        Presets({"less": 1.0, "balanced": 2.0, "more": 0.5})
    with pytest.raises(ValueError, match="sensitivity must be"):
        check_level("loud")


def test_the_warm_up_takes_the_stricter_of_the_level_and_balanced() -> None:
    presets = Presets({"less": 3.0, "balanced": 2.0, "more": 1.0})

    assert presets.cutoff("more", in_warmup=True) == 2.0  # More starts after the warm-up
    assert presets.cutoff("less", in_warmup=True) == 3.0  # Less stays Less (review on #65)
    assert presets.cutoff("balanced", in_warmup=True) == 2.0
    assert [presets.cutoff(level) for level in LEVELS] == [3.0, 2.0, 1.0]


def test_the_warm_up_is_each_users_first_90_days_and_3_months() -> None:
    txns = pd.DataFrame(
        {
            "user_id": ["a", "a", "b"],
            "ts": ["2024-01-15 10:00", "2024-06-01 09:00", "2024-03-01 08:00"],
        }
    )
    rows = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "b"],
            "ts": ["2024-04-14", "2024-04-15", "2024-05-29", "2024-05-31"],
        }
    )
    assert charges_after_warmup(rows, warmup_end(txns)).tolist() == [False, True, False, True]

    periods = pd.DataFrame(
        {"user_id": ["a", "a", "b"], "period_start": ["2024-03-01", "2024-04-01", "2024-06-01"]}
    )
    assert periods_after_warmup(periods, txns).tolist() == [False, True, True]


def test_the_per_user_warm_up_matches_the_label_contract(small_sqlite: Path) -> None:
    from smart_financial_coach.data.labels import load_truth

    txns = store.load_transactions(small_sqlite)
    truth = load_truth(small_sqlite)
    after = charges_after_warmup(txns, warmup_end(txns))
    assert (
        after == (pd.to_datetime(txns["ts"]) >= pd.Timestamp(truth.warmup_end_day)).to_numpy()
    ).all()


def test_presets_files_belong_to_one_model_version(tmp_path: Path) -> None:
    presets = Presets({"less": 3.0, "balanced": 2.0, "more": 1.0})
    write_presets(tmp_path, presets, {"model_version": "v1"})

    assert load_presets(tmp_path, "v1") == presets
    assert json.loads((tmp_path / PRESETS_FILE).read_text())["multipliers"]["more"] == 2.0
    with pytest.raises(ArtifactError, match="placed for model 'v1'"):
        load_presets(tmp_path, "v2")
    assert load_presets(tmp_path / "missing", "v1") is None


def test_the_presets_command_writes_beside_the_manifest(
    small_sqlite: Path, flag_artifacts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    root = tmp_path / "artifacts"
    shutil.copytree(flag_artifacts, root)
    argv = ["presets", "--task", "unusual_transactions", "--data", str(small_sqlite)]

    assert model_main([*argv, "--artifacts-dir", str(root)]) == 0
    assert "more" in capsys.readouterr().out
    written = json.loads((root / "unusual_transactions" / "fr7-test" / PRESETS_FILE).read_text())
    assert written["model_version"] == "fr7-test"
    counts = written["counts"]
    assert counts["less"]["after_warmup"] <= counts["balanced"]["after_warmup"]
    assert counts["more"]["after_warmup"] >= counts["balanced"]["after_warmup"]
    # In the warm-up, More flags no more than Balanced (decision 17)
    assert counts["more"]["in_warmup"] == counts["balanced"]["in_warmup"]


# --- Unusual charges ----------------------------------------------------------------------


def test_the_flag_file_stores_down_to_more_often(
    flagged_sources: DataSources, preset_flagged_sources: DataSources
) -> None:
    assert flagged_sources.flags is not None
    assert preset_flagged_sources.flags is not None
    before, after = load_flags(flagged_sources.flags), load_flags(preset_flagged_sources.flags)
    cutoffs = load_flag_presets(preset_flagged_sources.flags)

    assert load_flag_presets(flagged_sources.flags) is None  # a pre-FR-9 file
    assert cutoffs is not None
    assert set(cutoffs) == set(LEVELS)
    assert len(after) > len(before)
    assert (after["score"] >= cutoffs["more"]).all()
    for row in after.itertuples():  # every stored row is contract-checked
        assert evidence_errors(str(row.reason_code), row.evidence) == []
    # Balanced's flags are exactly the pre-FR-9 file's
    balanced = after[after["score"] >= cutoffs["balanced"]]
    assert set(balanced["transaction_id"]) == set(before["transaction_id"])


def test_flag_run_reports_balanced_and_stored(
    small_sqlite: Path, preset_flag_artifacts: Path, tmp_path: Path
) -> None:
    run = flag_dataset(small_sqlite, tmp_path / "f.sqlite", artifacts_dir=preset_flag_artifacts)
    assert run.presets is not None
    assert run.flagged == len(load_flags(tmp_path / "f.sqlite")) > run.balanced > 0


def _every_user(sources: DataSources) -> list[str]:
    return list(store.load_users(sources.dataset)["user_id"])


def test_the_ledger_shows_each_levels_flags(
    flagged_sources: DataSources, preset_flagged_sources: DataSources
) -> None:
    for user in _every_user(preset_flagged_sources):
        ledger = Ledger.load(preset_flagged_sources, user)
        shown = {level: ledger.flags_at(level) for level in LEVELS}
        assert all(v is not None for v in shown.values())
        ids = {level: set(v["transaction_id"]) for level, v in shown.items() if v is not None}
        assert ids["less"] <= ids["balanced"] <= ids["more"]
        old = Ledger.load(flagged_sources, user).flags
        assert old is not None
        assert ids["balanced"] == set(old["transaction_id"])


def test_more_often_waits_for_the_warm_up(preset_flagged_sources: DataSources) -> None:
    """In a user's first 90 days, More shows only what Balanced shows (decision 17)."""
    seen_warm = False
    for user in _every_user(preset_flagged_sources):
        ledger = Ledger.load(preset_flagged_sources, user)
        (end,) = warmup_end(ledger.transactions.assign(user_id=user))
        ts = ledger.transactions.set_index("transaction_id")["ts"]
        more, balanced = ledger.flags_at("more"), ledger.flags_at("balanced")
        assert more is not None
        assert balanced is not None
        warm_more = {t for t in more["transaction_id"] if ts[t] < end}
        warm_balanced = {t for t in balanced["transaction_id"] if ts[t] < end}
        assert warm_more == warm_balanced
        seen_warm |= bool(warm_balanced)
        assert ledger.flag_presets is not None
        stored = ledger.flags
        assert stored is not None
        loose = stored[(stored["score"] >= ledger.flag_presets.cutoff("more"))]
        assert {t for t in loose["transaction_id"] if ts[t] >= end} == {
            t for t in more["transaction_id"] if ts[t] >= end
        }


def test_without_presets_every_level_is_balanced(flagged_sources: DataSources) -> None:
    user = _every_user(flagged_sources)[0]
    ledger = Ledger.load(flagged_sources, user)
    flags = ledger.flags
    assert flags is not None
    for level in LEVELS:
        shown = ledger.flags_at(level)
        assert shown is not None
        assert shown.equals(flags)
    found = Tools(ledger, sensitivity="more").call("detect_anomalies", EVERYTHING).data
    assert found["sensitivity"] == "balanced"


def test_detect_anomalies_applies_the_sensitivity(preset_flagged_sources: DataSources) -> None:
    totals = dict.fromkeys(LEVELS, 0)
    for user in _every_user(preset_flagged_sources):
        ledger = Ledger.load(preset_flagged_sources, user)
        for level in LEVELS:
            found = Tools(ledger, sensitivity=level).call("detect_anomalies", EVERYTHING).data
            assert found["sensitivity"] == level
            totals[level] += found["count"]
    assert totals["less"] < totals["balanced"] < totals["more"]


# --- Spending spikes ----------------------------------------------------------------------


def test_the_simple_rule_fits_its_presets_at_build_time(
    spike_sources: DataSources, spike_pool: pd.DataFrame
) -> None:
    assert spike_sources.spikes is not None
    state = load_state(spike_sources.spikes)
    assert state.presets is not None
    assert state.presets.cutoff("balanced") == state.model.cutoff
    # 0.5x and 2x the rule's rate on the pool, without labels (decision 13)
    from smart_financial_coach.data.features.monthly import monthly_aggregates, period_history
    from smart_financial_coach.data.features.season_profiles import season_table
    from smart_financial_coach.intelligence.spikes.batch import _pool_rows
    from smart_financial_coach.intelligence.spikes.contract import INPUT_COLUMNS

    periods = period_history(monthly_aggregates(spike_pool, as_of=state.as_of))
    x = _pool_rows(periods, season_table(periods), state.min_users)[list(INPUT_COLUMNS)]
    score = state.model._scores(x)
    months = user_months(x)
    for level, m in (("less", 0.5), ("more", 2.0)):
        n = int((score >= state.presets.cutoff(level)).sum())
        assert n == pytest.approx(SIMPLE_RULE_RATE * m * months, abs=3)


def test_a_promoted_model_without_presets_serves_balanced(
    promoted_spike_sources: DataSources,
) -> None:
    assert promoted_spike_sources.spikes is not None
    state = load_state(promoted_spike_sources.spikes)
    assert state.presets is None
    assert state.model_at("more") is state.model


def test_a_promoted_models_presets_reach_the_spikes_file(
    preset_spike_sources: DataSources, preset_spike_artifacts: Path
) -> None:
    assert preset_spike_sources.spikes is not None
    state = load_state(preset_spike_sources.spikes)
    placed = load_presets(preset_spike_artifacts / "spending_spikes" / "fr8-test", "fr8-test")
    assert state.presets == placed
    assert placed is not None
    assert state.model_at("more").cutoff == placed.cutoff("more")
    assert state.model.cutoff == state.model_at("balanced").cutoff  # the shared model untouched


def test_spikes_at_each_level_keep_the_product_rules(preset_spike_sources: DataSources) -> None:
    counts = dict.fromkeys(LEVELS, 0)
    for user in _every_user(preset_spike_sources):
        ledger = Ledger.load(preset_spike_sources, user)
        assert ledger.spikes is not None
        for level in LEVELS:
            # The contract rejects any flag that breaks the spend floor or the minimum volume
            scored = ledger.spikes.score(user, ledger.transactions, ledger.transactions, level)
            flagged = scored[scored["is_flagged"].astype(bool)]
            assert (flagged["spend"] >= 1.3 * flagged["usual"]).all()
            counts[level] += len(flagged)
    assert counts["less"] <= counts["balanced"] <= counts["more"]
    assert counts["less"] < counts["more"]


def test_spike_presets_survive_the_file(preset_spike_sources: DataSources, tmp_path: Path) -> None:
    from smart_financial_coach.intelligence.spikes.batch import write_state

    assert preset_spike_sources.spikes is not None
    state = load_state(preset_spike_sources.spikes)
    write_state(state, tmp_path / "s.json", users=1, categorizer="stub")
    assert load_state(tmp_path / "s.json").presets == state.presets
    assert replace(state, presets=None).model_at("less") is state.model


def test_the_level_is_reported_per_half(
    flagged_sources: DataSources, preset_spike_sources: DataSources
) -> None:
    """Spikes with presets but a pre-FR-9 flag file: only spikes move (review on #66)."""
    sources = DataSources(
        flagged_sources.dataset,
        flagged_sources.predictions,
        flagged_sources.flags,
        spikes=preset_spike_sources.spikes,
    )
    ledger = Ledger.load(sources, _every_user(sources)[0])
    found = Tools(ledger, sensitivity="more").call("detect_anomalies", EVERYTHING).data
    assert found["sensitivity"] == "more"
    assert found["sensitivity_applied"] == {
        "unusual_charges": "balanced",
        "spending_spikes": "more",
    }
