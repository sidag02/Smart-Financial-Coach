"""The "How it learns" page: the replay's story, and visitors' agreement live (#15 §7)."""

import json
from pathlib import Path

from smart_financial_coach.access.ledger import DataSources, Ledger
from smart_financial_coach.access.review_items import open_review_items
from smart_financial_coach.experience.accounts import Account
from tests.unit.experience.test_web import make_client, sign_in

MONTH = {
    "personal": {"rows": 10, "accuracy_view": 0.9},
    "personal_model_only": {"rows": 10, "accuracy_view": 0.8},
    "evaluation_unseen": {"rows": 5, "macro_f1_view": 0.7, "macro_f1_truth": 0.75},
    "items_shown_per_user": 1.5,
    "votes": 3,
    "global_labels": 1,
    "model": "promoted",
}
REPLAY = {
    "config": {"rule": {"n": 3, "majority": 0.667, "corrections": 1}, "every": 3},
    "users": {"feedback": 60, "evaluation": 60, "adversarial": 3},
    "months": [
        {**MONTH, "month": f"2026-0{i}", "evaluation": {"rows": 9, "accuracy_view": 0.80 + i / 100}}
        for i in range(1, 7)
    ],
    "retrainings": [
        {"month": "2026-03", "labels": 0, "new_labels": [], "revoked": [],
         "decision": "skipped: no new labels"},
        {
            "month": "2026-06", "labels": 4, "new_labels": [["netflix", "Entertainment"]],
            "revoked": [], "relabelled_rows": 1200, "added_rows": 40, "training_rows": 500000,
            "gates": [{"name": "macro_f1_view", "passed": True, "detail": "0.81 vs 0.79"}],
            "decision": "promoted",
        },
    ],
    "labels": [
        {"merchant_key": "netflix", "category": "Entertainment", "true_category": "Subscriptions",
         "subtype": "streaming", "voters": 9, "agreeing": 7, "corrections": 5},
    ],
    "remaps": [
        {"subtype": "streaming", "default": "Subscriptions", "remapped_to": "Entertainment",
         "holders": 84, "labels_now": {"Entertainment": 1},
         "label_changes": [{"seq": 3, "merchant_key": "netflix", "category": None, "voters": 4}]},
    ],
    "summary": {"global_labels": 4, "corrections": 120, "labels_matching_truth": 1,
                "promotions": 1, "open_items_per_user_month": 3.0,
                "items_resolved_per_user_month": 1.5},
}  # fmt: skip


def bundle(sources: DataSources, root: Path, replay: dict[str, object] | None) -> DataSources:
    root.mkdir()
    for f in (sources.dataset, sources.predictions):
        (root / f.name).symlink_to(f)
    if replay is not None:
        (root / "replay.json").write_text(json.dumps(replay))
    return DataSources(root / sources.dataset.name, root / sources.predictions.name)


def accounts(two_users: tuple[str, str]) -> list[Account]:
    return [
        Account(two_users[0], "Maya Chen", "maya@example.com"),
        Account(two_users[1], "Ada Okafor", "ada@example.com"),
    ]


def test_without_a_replay_the_page_says_so(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    with make_client(bundle(sources, tmp_path / "b", None), accounts(two_users)) as c:
        sign_in(c)
        page = c.get("/learning").text
    assert "hasn't been run for this build yet" in page
    assert "No corrections yet" in page


def test_the_replay_story_and_visitors_agreement(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    data = bundle(sources, tmp_path / "b", REPLAY)
    with make_client(data, accounts(two_users)) as c:
        sign_in(c)
        ledger = Ledger.load(data, two_users[0])
        item = open_review_items(ledger.transactions, two_users[0]).iloc[0]
        target = "Travel" if item["suggested_category"] != "Travel" else "Entertainment"
        c.post(f"/review/{item['item_id']}", data={"action": "correct", "category": target})
        page = c.get("/learning").text

    assert "How it learns" in page
    assert "85.0%" in page  # the last quarter's mean for people who never corrected (0.84-0.86)
    assert "Netflix" in page
    assert "model error fixed" not in page  # Netflix was a preference, not an error
    assert "promoted" in page
    assert "<polyline" in page
    assert "75.0% → 75.0% (macro F1, true categories)" in page
    # One session's correction isn't shown to others (§4): only that a merchant has a vote
    assert f"{item['merchant_key'].title()} → {target}" not in page
    assert "1 merchant corrected by one session so far" in page


def test_a_merchant_shows_once_two_sessions_have_voted(
    sources: DataSources, two_users: tuple[str, str], tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    data = bundle(sources, tmp_path / "b", REPLAY)
    with make_client(data, accounts(two_users)) as c:
        other = TestClient(c.app)  # a second visitor, same app
        sign_in(c)
        sign_in(other)
        ledger = Ledger.load(data, two_users[0])
        item = open_review_items(ledger.transactions, two_users[0]).iloc[0]
        target = "Travel" if item["suggested_category"] != "Travel" else "Entertainment"
        for client in (c, other):
            client.post(
                f"/review/{item['item_id']}", data={"action": "correct", "category": target}
            )
        page = c.get("/learning").text

    assert f"{item['merchant_key'].title()} → {target}" in page
    assert "2 of 2 agree · 3 people needed" in page
