"""End to end on the toy task: run -> leaderboard -> finalize -> promote -> load_service."""

import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from smart_financial_coach.evaluation import runner
from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.promote import (
    FINALIST_TAG,
    OVERRIDE_TAG,
    RANK_TAG,
    SelectionError,
    finalize,
    leaderboard,
    promote,
)
from smart_financial_coach.evaluation.runner import run_experiment
from smart_financial_coach.evaluation.tracking import Tracker
from smart_financial_coach.intelligence.models import ContractError
from smart_financial_coach.intelligence.models.artifact import (
    MODEL_FILE,
    POINTER_FILE,
    URL_KEY,
    ArtifactError,
    promotion_errors,
    promotions,
)
from smart_financial_coach.intelligence.models.registry import model_class
from smart_financial_coach.intelligence.service import load_service

MakeConfig = Callable[..., ExperimentConfig]
NOTE = "group lookup with a hint fallback; no external dependencies"


@pytest.fixture
def runs(toy_data: Path, tracker: Tracker, make_config: MakeConfig) -> dict[str, str]:
    """A baseline and three candidates on the same data and splits."""
    configs = [
        make_config("majority", "toy/majority", baseline=True),
        make_config("memory_hint", grid={"confidence": [0.7, 0.9]}),
        make_config(
            "memory_no_hint", model={"type": "toy/memory", "params": {"trust_hint": False}}
        ),
        make_config("memory_hint_complex", complexity=3),
    ]
    return {c.name: run_experiment(c, toy_data, tracker).run_id for c in configs}


def test_run_logs_validation_only(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    run = tracker.get(runs["memory_hint"])

    assert run.metrics["val_unseen_accuracy"] > 0.7
    assert run.metrics["val_seen_accuracy"] == 1.0
    assert not [k for k in run.metrics if k.startswith("test_")]
    for tag in ("sfc.config_hash", "sfc.data_hash", "sfc.split_hash", "sfc.code_version"):
        assert run.tags[tag]
    assert run.params["chosen.confidence"] in {"0.7", "0.9"}
    assert len({tracker.get(r).tags["sfc.split_hash"] for r in runs.values()}) == 1


def test_latency_is_timed_from_a_cold_start(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    memory = model_class("toy/memory")  # the toy model counts cache resets
    before = memory.resets  # type: ignore[attr-defined]
    result = run_experiment(make_config("cold"), toy_data, tracker)

    assert memory.resets == before + 1  # type: ignore[attr-defined]
    assert {"latency_p95_ms", "latency_warm_p95_ms"} <= set(result.metrics)


def test_identical_run_is_skipped_unless_forced(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    again = run_experiment(make_config("memory_hint_complex", complexity=3), toy_data, tracker)
    forced = run_experiment(
        make_config("memory_hint_complex", complexity=3), toy_data, tracker, force=True
    )

    assert again.skipped
    assert again.run_id == runs["memory_hint_complex"]
    assert not forced.skipped
    assert forced.run_id != again.run_id


def test_rerun_gives_identical_metrics(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    rerun = run_experiment(
        make_config("memory_hint_complex", complexity=3), toy_data, tracker, force=True
    )
    first = tracker.get(runs["memory_hint_complex"]).metrics

    assert {k: v for k, v in rerun.metrics.items() if k.startswith("val_")} == {
        k: v for k, v in first.items() if k.startswith("val_")
    }


def test_leaderboard_orders_by_rule(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    standings = leaderboard("toy", toy_data, tracker)
    names = [s.name for s in standings]

    # Hint models tie on unseen accuracy; the simpler one wins the tie. No baselines listed.
    assert names[:2] == ["memory_hint", "memory_hint_complex"]
    assert standings[1].tied_with_leader
    assert names[-1] == "memory_no_hint"
    assert "majority" not in names


def test_finalize_takes_the_rules_top_three(
    toy_data: Path, tracker: Tracker, runs: dict[str, str]
) -> None:
    scored = finalize("toy", toy_data, tracker)
    ranks = {name: tracker.get(runs[name]).tags.get(RANK_TAG) for name in runs}

    assert set(scored) == set(runs.values())  # three candidates + the baseline
    assert ranks == {
        "majority": None,
        "memory_hint": "1",
        "memory_hint_complex": "2",
        "memory_no_hint": "3",
    }
    assert scored[runs["memory_hint"]]["test_known_accuracy"] == 1.0
    assert FINALIST_TAG not in tracker.get(runs["majority"]).tags


def test_test_sets_are_used_once_per_split(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str]
) -> None:
    """A second call can't test-score more runs, even new ones, without a recorded reason."""
    finalize("toy", toy_data, tracker)
    late = run_experiment(make_config("late", "toy/memory_alt"), toy_data, tracker).run_id

    with pytest.raises(SelectionError, match="already used"):
        finalize("toy", toy_data, tracker)
    with pytest.raises(SelectionError, match="override reason"):
        finalize("toy", toy_data, tracker, run_ids=[late])
    with pytest.raises(SelectionError, match="already scored"):
        finalize("toy", toy_data, tracker, run_ids=[runs["memory_hint"]], override="recheck")

    finalize("toy", toy_data, tracker, run_ids=[late], override="new model type added late")

    tags = tracker.get(late).tags
    assert tags[OVERRIDE_TAG] == "new model type added late"
    assert tags[RANK_TAG] in {"1", "2"}  # it ties the leader on every tie-breaker


def test_finalize_refusals(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    with pytest.raises(SelectionError, match="1 to 3"):
        finalize("toy", toy_data, tracker, run_ids=list(runs.values()), override="all")
    with pytest.raises(SelectionError, match="baseline"):
        finalize("toy", toy_data, tracker, run_ids=[runs["majority"]], override="x")


def test_finalize_needs_baselines(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    run_experiment(make_config("alone"), toy_data, tracker)

    with pytest.raises(SelectionError, match="no majority baseline run"):
        finalize("toy", toy_data, tracker)


def test_finalize_refuses_other_data(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    other = tmp_path / "other.csv"
    pd.read_csv(toy_data).iloc[::-1].to_csv(other, index=False)

    with pytest.raises(SelectionError, match="not the dataset"):
        finalize("toy", other, tracker)


def test_promote_then_load_service(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    finalize("toy", toy_data, tracker)
    entry = promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)

    served = load_service("toy", artifacts)
    x = pd.DataFrame({"id": ["n1"], "group": ["g05"], "hint": ["c"]})

    assert served.predict(x)["label"].tolist() == ["c"]  # g05 is a "c" group
    assert served.version == entry["version"]
    assert "override" not in entry
    assert (artifacts / "toy" / POINTER_FILE).read_text().strip() == entry["version"]
    assert promotions(artifacts / "toy")[-1]["mlflow_run_id"] == runs["memory_hint"]
    assert promotion_errors(artifacts / "toy") == []
    assert tracker.champion_run("toy") == runs["memory_hint"]


def test_switching_models_needs_no_code_change(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, runs: dict[str, str], tmp_path: Path
) -> None:
    """Promote one model type, then another: the caller's code is the same two lines.

    Promoting a runner-up departs from the decision rule, so it carries a recorded override.
    """
    artifacts = tmp_path / "artifacts"
    alt = run_experiment(make_config("alt", "toy/memory_alt", complexity=1), toy_data, tracker)
    finalize("toy", toy_data, tracker)
    x = pd.DataFrame({"id": ["n1"], "group": ["g05"], "hint": ["c"]})

    promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)
    first = load_service("toy", artifacts)
    first_out = first.predict(x)
    promote("toy", alt.run_id, NOTE, tracker, artifacts, override="demonstrate a model switch")
    second = load_service("toy", artifacts)
    second_out = second.predict(x)

    assert (first.model.name, second.model.name) == ("toy/memory", "toy/memory_alt")
    assert first_out["confidence"].tolist() != second_out["confidence"].tolist()
    log = promotions(artifacts / "toy")
    assert [e.get("override") for e in log] == [None, "demonstrate a model switch"]
    assert promotion_errors(artifacts / "toy") == []
    assert tracker.champion_run("toy") == alt.run_id


class FolderPublisher:
    """Publishes into a local folder and returns `file://` URLs: a release with no network."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.tags: list[str] = []
        self.commits: list[str | None] = []

    def publish(self, tag: str, path: Path, notes: str, commit: str | None = None) -> str:
        self.tags.append(tag)
        self.commits.append(commit)
        (self.folder / tag).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self.folder / tag / path.name)
        return (self.folder / tag / path.name).as_uri()


def test_published_model_is_downloaded_on_first_use(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    """A fresh clone has the committed manifest and log, but not the model file."""
    artifacts, publisher = tmp_path / "artifacts", FolderPublisher(tmp_path / "releases")
    finalize("toy", toy_data, tracker)
    entry = promote("toy", runs["memory_hint"], NOTE, tracker, artifacts, publisher=publisher)
    model_file = artifacts / "toy" / entry["version"] / MODEL_FILE
    model_file.unlink()

    served = load_service("toy", artifacts)

    assert publisher.tags == [f"toy-{entry['version']}"]
    assert publisher.commits == [tracker.get(runs["memory_hint"]).tags["sfc.git_commit"]]
    assert entry[URL_KEY].startswith("file://")
    assert model_file.exists()
    assert served.version == entry["version"]


def test_downloaded_model_must_match_the_committed_manifest(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts, releases = tmp_path / "artifacts", tmp_path / "releases"
    finalize("toy", toy_data, tracker)
    entry = promote(
        "toy", runs["memory_hint"], NOTE, tracker, artifacts, publisher=FolderPublisher(releases)
    )
    folder = artifacts / "toy" / entry["version"]
    (folder / MODEL_FILE).unlink()
    (releases / f"toy-{entry['version']}" / MODEL_FILE).write_bytes(b"not the promoted model")

    with pytest.raises(ArtifactError, match="doesn't match"):
        load_service("toy", artifacts)
    assert sorted(p.name for p in folder.iterdir()) == ["manifest.json"]  # nothing left to load


def test_unpublished_model_cant_be_loaded_elsewhere(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    finalize("toy", toy_data, tracker)
    entry = promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)
    (artifacts / "toy" / entry["version"] / MODEL_FILE).unlink()

    assert URL_KEY not in entry
    with pytest.raises(ArtifactError, match="no URL"):
        load_service("toy", artifacts)


def test_retraining_after_a_code_change_gets_a_new_version(
    toy_data: Path,
    tracker: Tracker,
    make_config: MakeConfig,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same config and data, different code: a different model, so a different version and tag."""
    artifacts, publisher = tmp_path / "artifacts", FolderPublisher(tmp_path / "releases")
    run_experiment(make_config("majority", "toy/majority", baseline=True), toy_data, tracker)
    first = run_experiment(make_config("memory"), toy_data, tracker)
    finalize("toy", toy_data, tracker)
    one = promote("toy", first.run_id, NOTE, tracker, artifacts, publisher=publisher)

    monkeypatch.setattr(runner, "code_version", lambda: "c0ffee00" * 8)  # after a code fix
    run_experiment(make_config("majority", "toy/majority", baseline=True), toy_data, tracker)
    second = run_experiment(make_config("memory"), toy_data, tracker)
    reason = "retrain after a code fix"
    finalize("toy", toy_data, tracker, run_ids=[second.run_id], override=reason)
    two = promote(
        "toy", second.run_id, NOTE, tracker, artifacts, publisher=publisher, override=reason
    )

    assert one["version"].rsplit("-", 1)[0] == two["version"].rsplit("-", 1)[0]  # config, data
    assert one["version"] != two["version"]
    assert publisher.tags == [f"toy-{one['version']}", f"toy-{two['version']}"]
    assert load_service("toy", artifacts).version == two["version"]


def test_promote_refusals(
    toy_data: Path, tracker: Tracker, runs: dict[str, str], tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    with pytest.raises(SelectionError, match="not a finalist"):
        promote("toy", runs["memory_hint"], NOTE, tracker, artifacts)

    finalize("toy", toy_data, tracker)
    with pytest.raises(SelectionError, match="finalist #2, not the decision rule's winner"):
        promote("toy", runs["memory_hint_complex"], NOTE, tracker, artifacts)
    with pytest.raises(SelectionError, match="note"):
        promote("toy", runs["memory_hint"], "  ", tracker, artifacts)
    with pytest.raises(SelectionError, match="unseen_accuracy"):  # unseen falls to majority
        promote("toy", runs["memory_no_hint"], NOTE, tracker, artifacts, override="try #3")
    assert not (artifacts / "toy").exists()


def test_broken_model_fails_in_validation(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    with pytest.raises(ContractError, match="rows for"):
        run_experiment(make_config("broken", "toy/broken"), toy_data, tracker)
    assert tracker.find("toy", {}) == []  # the failed run isn't a finished experiment


def test_reproduce_poc_only_for_poc_configs(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    with pytest.raises(ValueError, match="not a POC configuration"):
        run_experiment(make_config("other", complexity=1), toy_data, tracker, reproduce_poc=True)

    result = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)

    assert result.metrics["test_known_accuracy"] == 1.0
    assert tracker.find("toy", {}) == []  # kept off the leaderboard and out of finalize


def test_reproduction_runs_resume(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    first = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)
    again = run_experiment(make_config("poc"), toy_data, tracker, reproduce_poc=True)

    assert again.skipped
    assert again.run_id == first.run_id


def test_resume_needs_the_same_code(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = run_experiment(make_config("same"), toy_data, tracker)
    monkeypatch.setattr(runner, "code_version", lambda: "another-version")
    second = run_experiment(make_config("same"), toy_data, tracker)

    assert not second.skipped
    assert second.run_id != first.run_id


def test_finalize_refuses_mixed_code_versions(
    toy_data: Path,
    tracker: Tracker,
    make_config: MakeConfig,
    runs: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "code_version", lambda: "newer-version")
    newer = run_experiment(make_config("newer", "toy/memory_alt"), toy_data, tracker)

    assert len({s.code for s in leaderboard("toy", toy_data, tracker)}) == 2
    with pytest.raises(SelectionError, match="code versions"):
        finalize("toy", toy_data, tracker, override="")
    finalize("toy", toy_data, tracker, run_ids=[newer.run_id], override="baseline code unchanged")


def test_leaderboard_names_a_missing_baseline(
    toy_data: Path, tracker: Tracker, make_config: MakeConfig
) -> None:
    run_experiment(make_config("alone"), toy_data, tracker)

    with pytest.raises(SelectionError, match="no majority baseline run"):
        leaderboard("toy", toy_data, tracker)


def test_code_version_covers_code_not_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Untracked code changes the version; docs and experiment configs don't (PR #8 review)."""
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    base = runner.code_version()

    (tmp_path / "docs" / "notes.md").write_text("docs only\n")
    (tmp_path / "configs" / "experiments" / "round2.yaml").write_text("name: next\n")
    assert runner.code_version() == base  # docs and new experiment configs keep runs valid
    (tmp_path / "src" / "café.py").write_text("z = 1\n")  # non-ASCII untracked name
    assert runner.code_version() != base
    (tmp_path / "src" / "café.py").unlink()
    (tmp_path / "src" / "new_model.py").write_text("y = 2\n")  # untracked
    untracked = runner.code_version()
    (tmp_path / "src" / "new_model.py").write_text("y = 3\n")

    assert untracked != base
    assert runner.code_version() != untracked


def test_comparison_report(toy_data: Path, tracker: Tracker, runs: dict[str, str]) -> None:
    from smart_financial_coach.evaluation.report import comparison_report

    before = comparison_report("toy", toy_data, tracker)
    finalize("toy", toy_data, tracker)
    after = comparison_report("toy", toy_data, tracker)

    assert "| 1 | `memory_hint` (tied) |" in before
    assert "`majority`" in before.split("**Baselines**")[1]
    assert "on the test sets" not in before
    tests = after.split("**Finalists and baselines on the test sets**")[1]
    assert "| 1 | `memory_hint` | " in tests
    assert "| baseline | `majority` | " in tests
    assert "`test_unseen_accuracy`" in tests


def test_report_edge_cases(toy_data: Path, tracker: Tracker, make_config: MakeConfig) -> None:
    from smart_financial_coach.evaluation.report import DASH, comparison_report

    with pytest.raises(SelectionError, match="no finished runs"):
        comparison_report("toy", toy_data, tracker)

    run_experiment(make_config("majority", "toy/majority", baseline=True), toy_data, tracker)
    run_experiment(make_config("also_majority", "toy/majority"), toy_data, tracker)  # ineligible
    report = comparison_report("toy", toy_data, tracker)
    row = next(line for line in report.splitlines() if "`also_majority`" in line)

    assert row.startswith(f"| {DASH} |")  # unranked
    assert row.split(" | ")[3] == DASH  # no leader to compare against
