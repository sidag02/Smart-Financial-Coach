"""Alert sensitivity and flag actions (FR-9 §3, §5): the store, what actions hide, and the tools."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from smart_financial_coach.access.alerts import AlertError, AlertStore, AlertView
from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.tools import (
    NOT_ME_GUIDANCE,
    AlertAccess,
    ToolError,
    Tools,
)
from smart_financial_coach.data import store as data_store

EVERYTHING = {"start_date": "2023-01-01", "end_date": "2026-09-30"}


@pytest.fixture
def alerts(tmp_path: Path) -> AlertStore:
    return AlertStore(tmp_path / "feedback.sqlite")


@pytest.fixture(scope="module")
def alert_sources(
    preset_flagged_sources: DataSources, preset_spike_sources: DataSources
) -> DataSources:
    """Flags stored down to More often, and a spike scorer with presets."""
    return DataSources(
        preset_flagged_sources.dataset,
        preset_flagged_sources.predictions,
        preset_flagged_sources.flags,
        spikes=preset_spike_sources.spikes,
    )


def charge(store: AlertStore, subject: str, user: str, **fields: Any) -> str:
    base = {
        "flag_id": f"v1:{fields.get('transaction_id', 't1')}",
        "kind": "charge",
        "action": "recognize",
        "source": "page",
        "transaction_id": "t1",
        "transaction_ts": "2026-05-01T10:00:00",
        "merchant_key": "target",
        "reason_code": "amount_unusual",
    }
    return store.record(subject, user, **(base | fields)).action_id


# --- The store -----------------------------------------------------------------------------


def test_the_setting_is_the_latest_and_balanced_by_default(alerts: AlertStore) -> None:
    assert alerts.level("s1", "u1") == "balanced"
    alerts.set_level("s1", "u1", "more", source="page")
    alerts.set_level("s1", "u1", "less", source="coach")
    assert alerts.level("s1", "u1") == "less"
    assert alerts.level("s2", "u1") == "balanced"  # another visitor on the same account
    assert alerts.level("s1", "u2") == "balanced"
    with pytest.raises(ValueError, match="sensitivity must be"):
        alerts.set_level("s1", "u1", "loud", source="page")


def test_recognize_hides_this_and_later_flags_at_the_merchant(alerts: AlertStore) -> None:
    charge(alerts, "s1", "u1", transaction_id="t1", transaction_ts="2026-05-01T10:00:00")
    view = alerts.view("s1", "u1")

    def hidden(tid: str, ts: str, merchant: str = "target", code: str = "amount_unusual") -> bool:
        return view.hiding_charge(tid, merchant, code, pd.Timestamp(ts)) is not None

    assert hidden("t1", "2026-05-01T10:00:00")
    assert hidden("t9", "2026-08-01")  # later, same reason and merchant
    assert not hidden("t0", "2026-04-01")  # earlier ones stay (FR-7 §8)
    assert not hidden("t9", "2026-08-01", merchant="costco")
    assert not hidden("t9", "2026-08-01", code="new_merchant")
    assert (
        alerts.view("s2", "u1").hiding_charge("t1", "target", "amount_unusual", pd.Timestamp(0))
        is None
    )


def test_recognize_on_a_duplicate_hides_that_flag_only(alerts: AlertStore) -> None:
    charge(alerts, "s1", "u1", transaction_id="t1", reason_code="duplicate")
    view = alerts.view("s1", "u1")
    later = pd.Timestamp("2026-08-01")
    assert view.hiding_charge("t1", "target", "duplicate", later) is not None
    assert view.hiding_charge("t2", "target", "duplicate", later) is None  # decision 9


def test_expected_hides_one_month_and_not_me_hides_nothing(alerts: AlertStore) -> None:
    alerts.record(
        "s1",
        "u1",
        flag_id="v1:u1|Dining|2026-08-01",
        kind="spike",
        action="expected",
        source="page",
        category="Dining",
        period_start="2026-08-01",
    )
    not_me = charge(alerts, "s1", "u1", transaction_id="t5", action="not_me")
    view = alerts.view("s1", "u1")
    assert view.hiding_spike("Dining", "2026-08-01") is not None
    assert view.hiding_spike("Dining", "2026-09-01") is None
    assert view.hiding_spike("Groceries", "2026-08-01") is None
    assert view.hiding_charge("t5", "target", "amount_unusual", pd.Timestamp(0)) is None
    assert view.not_me == {"t5": not_me}


def test_undo_restores_and_is_the_subjects_own(alerts: AlertStore) -> None:
    action = charge(alerts, "s1", "u1")
    with pytest.raises(AlertError, match="already done"):
        charge(alerts, "s1", "u1")
    with pytest.raises(AlertError, match="no alert action"):
        alerts.undo("s2", "u1", action)  # someone else's is indistinguishable from none
    with pytest.raises(AlertError, match="no alert action"):
        alerts.undo("s1", "u2", action)  # nor one on another account (review on #67)
    assert not alerts.actions("s1", "u1")[0].undone  # and nothing was written
    alerts.undo("s1", "u1", action)
    assert alerts.view("s1", "u1") == AlertView()
    with pytest.raises(AlertError, match="already undone"):
        alerts.undo("s1", "u1", action)
    charge(alerts, "s1", "u1")  # after undo, the same action can be taken again


def test_actions_match_on_stored_fields_not_the_flag_id(alerts: AlertStore) -> None:
    """A promotion gives flags new ids; the actions still apply (decision 14)."""
    charge(alerts, "s1", "u1", flag_id="old-model:t1", transaction_id="t1")
    charge(alerts, "s1", "u1", flag_id="old-model:t2", transaction_id="t2", action="not_me")
    view = alerts.view("s1", "u1")
    later = pd.Timestamp("2026-09-01")
    assert view.hiding_charge("t3", "target", "amount_unusual", later) is not None
    assert "t2" in view.not_me


def test_actions_are_checked(alerts: AlertStore) -> None:
    with pytest.raises(AlertError, match="isn't an action"):
        charge(alerts, "s1", "u1", action="expected")
    with pytest.raises(AlertError, match="needs its reason"):
        charge(alerts, "s1", "u1", reason_code=None)
    with pytest.raises(AlertError, match="subject"):
        charge(alerts, "", "u1")


# --- The tools ------------------------------------------------------------------------------


def _tools(
    sources: DataSources, user: str, store: AlertStore, subject: str = "s1", source: str = "page"
) -> Tools:
    return Tools(Ledger.load(sources, user), alerts=AlertAccess(store, subject, source))


def _users(sources: DataSources) -> list[str]:
    return list(data_store.load_users(sources.dataset)["user_id"])


def _with_charges(sources: DataSources, store: AlertStore, n: int = 1) -> str:
    for user in _users(sources):
        found = _tools(sources, user, store).call("detect_anomalies", EVERYTHING).data
        if len(found["unusual_transactions"]) >= n:
            return user
    raise AssertionError("no user with flags")


def _with_spikes(sources: DataSources, store: AlertStore) -> str:
    for user in _users(sources):
        store.set_level("probe", user, "more", source="page")
        found = _tools(sources, user, store, "probe").call("detect_anomalies", EVERYTHING).data
        if found["spending_spikes"].get("spikes"):
            return user
    raise AssertionError("no user with spikes")


def test_detect_anomalies_follows_the_subjects_setting(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    counts = {}
    for level in ("less", "balanced", "more"):
        alerts.set_level("s1", user, level, source="page")
        found = _tools(alert_sources, user, alerts).call("detect_anomalies", EVERYTHING).data
        assert found["sensitivity"] == level
        assert found["hidden"] == {"recognized": 0, "expected": 0}
        assert all(
            f["flag_id"].endswith(f["transaction_id"]) for f in found["unusual_transactions"]
        )
        counts[level] = found["count"]
    assert counts["less"] <= counts["balanced"] <= counts["more"]
    other = _tools(alert_sources, user, alerts, "s2").call("detect_anomalies", EVERYTHING).data
    assert other["sensitivity"] == "balanced"  # another visitor's setting is their own


def test_recognizing_a_charge_hides_it_and_counts_it(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    tools = _tools(alert_sources, user, alerts)
    before = tools.call("detect_anomalies", EVERYTHING).data
    flag = before["unusual_transactions"][0]

    done = tools.call("act_on_flag", {"flag_id": flag["flag_id"], "action": "recognize"}).data
    after = tools.call("detect_anomalies", EVERYTHING).data

    assert done["status"] == "applied"
    shown = {f["flag_id"] for f in after["unusual_transactions"]}
    assert flag["flag_id"] not in shown
    assert after["hidden"]["recognized"] >= 1
    assert after["count"] == before["count"] - after["hidden"]["recognized"]
    listed = tools.detect_anomalies(**EVERYTHING, include_hidden=True).data
    hidden = [f for f in listed["unusual_transactions"] if f.get("hidden_by")]
    assert {f["hidden_by"] for f in hidden} == {done["action"]["action_id"]}
    other = _tools(alert_sources, user, alerts, "s2").call("detect_anomalies", EVERYTHING).data
    assert other["count"] == before["count"]  # nobody else's view changes

    tools.call("undo_flag_action", {"action_id": done["action"]["action_id"]})
    assert tools.call("detect_anomalies", EVERYTHING).data["count"] == before["count"]


def test_not_me_marks_the_charge_with_fixed_guidance(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    tools = _tools(alert_sources, user, alerts)
    flag = tools.call("detect_anomalies", EVERYTHING).data["unusual_transactions"][0]

    done = tools.call("act_on_flag", {"flag_id": flag["flag_id"], "action": "not_me"}).data
    after = tools.call("detect_anomalies", EVERYTHING).data

    assert done["guidance"] == list(NOT_ME_GUIDANCE)
    marked = [f for f in after["unusual_transactions"] if f["flag_id"] == flag["flag_id"]]
    assert marked
    assert marked[0]["your_action"] == "not_me"
    assert marked[0]["action_id"] == done["action"]["action_id"]
    assert after["hidden"]["recognized"] == 0  # "not me" hides nothing


def test_expected_hides_that_months_spike(alert_sources: DataSources, alerts: AlertStore) -> None:
    user = _with_spikes(alert_sources, alerts)
    alerts.set_level("s1", user, "more", source="page")
    tools = _tools(alert_sources, user, alerts)
    spikes = tools.call("detect_anomalies", EVERYTHING).data["spending_spikes"]["spikes"]
    spike = spikes[0]

    with pytest.raises(ToolError, match="the action is expected"):
        tools.call("act_on_flag", {"flag_id": spike["flag_id"], "action": "recognize"})
    tools.call("act_on_flag", {"flag_id": spike["flag_id"], "action": "expected"})
    after = tools.call("detect_anomalies", EVERYTHING).data

    left = after["spending_spikes"]["spikes"]
    assert spike["flag_id"] not in {s["flag_id"] for s in left}
    assert len(left) == len(spikes) - 1
    assert after["hidden"]["expected"] == 1


def test_the_coach_previews_until_the_person_agrees(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    coach = _tools(alert_sources, user, alerts, source="coach")
    flag = coach.call("detect_anomalies", EVERYTHING).data["unusual_transactions"][0]

    preview = coach.call("set_alert_sensitivity", {"level": "more"}).data
    assert preview["status"] == "needs_confirmation"
    assert alerts.level("s1", user) == "balanced"
    assert (
        coach.call("set_alert_sensitivity", {"level": "more", "confirm": True}).data["status"]
        == "applied"
    )
    assert alerts.level("s1", user) == "more"

    args = {"flag_id": flag["flag_id"], "action": "recognize"}
    assert coach.call("act_on_flag", args).data["status"] == "needs_confirmation"
    assert alerts.actions("s1", user) == []
    assert coach.call("act_on_flag", args | {"confirm": True}).data["status"] == "applied"
    assert alerts.actions("s1", user)[0].source == "coach"


def test_alert_tools_need_a_store_and_a_known_flag(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    read_only = Tools(Ledger.load(alert_sources, user))
    settings = read_only.call("get_alert_settings", {}).data
    assert settings["can_change"] is False
    assert settings["available"] is True
    with pytest.raises(ToolError, match="can't be changed here"):
        read_only.call("set_alert_sensitivity", {"level": "more"})
    tools = _tools(alert_sources, user, alerts)
    with pytest.raises(ToolError, match="no alert"):
        tools.call("act_on_flag", {"flag_id": "nope", "action": "recognize"})
    with pytest.raises(ToolError, match="level must be"):
        tools.call("set_alert_sensitivity", {"level": "loud"})


def test_another_accounts_action_cant_be_undone_here(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    mine, other = _users(alert_sources)[:2]
    action = charge(alerts, "s1", other)
    with pytest.raises(ToolError, match="no alert action"):
        _tools(alert_sources, mine, alerts).call("undo_flag_action", {"action_id": action})
    assert not alerts.actions("s1", other)[0].undone  # the other account's alert stays hidden


def test_a_double_submit_records_one_action(alerts: AlertStore) -> None:
    """Two quick submits of the same form race; only one is recorded (review on #67)."""
    import threading

    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def submit() -> None:
        barrier.wait()
        try:
            charge(alerts, "s1", "u1")
            outcomes.append("recorded")
        except AlertError:
            outcomes.append("refused")

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("recorded") == 1
    assert len([a for a in alerts.actions("s1", "u1") if not a.undone]) == 1


def test_settings_list_levels_and_recent_actions(
    alert_sources: DataSources, alerts: AlertStore
) -> None:
    user = _with_charges(alert_sources, alerts)
    tools = _tools(alert_sources, user, alerts)
    flag = tools.call("detect_anomalies", EVERYTHING).data["unusual_transactions"][0]
    tools.call("act_on_flag", {"flag_id": flag["flag_id"], "action": "recognize"})

    data = tools.call("get_alert_settings", {}).data
    assert [lv["level"] for lv in data["levels"]] == ["less", "balanced", "more"]
    assert data["in_warmup"] is False
    assert data["recent_actions"][0]["action"] == "recognize"
    assert data["recent_actions"][0]["what"].startswith(flag["merchant"])


def test_without_presets_the_setting_cant_change_anything(
    flagged_sources: DataSources, alerts: AlertStore
) -> None:
    user = _users(flagged_sources)[0]
    with pytest.raises(ToolError, match="can't change anything"):
        _tools(flagged_sources, user, alerts).call("set_alert_sensitivity", {"level": "more"})
