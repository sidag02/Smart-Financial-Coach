"""The nightly flag job and its flag file (FR-7 §8)."""

import json
from pathlib import Path

import pytest

from smart_financial_coach.access.ledger import DataSources
from smart_financial_coach.data import store
from smart_financial_coach.data.flags import FlagWriter, load_flag_meta, load_flags
from smart_financial_coach.evaluation.cli import model_main
from smart_financial_coach.experience.demo import _subset
from smart_financial_coach.intelligence.anomaly.batch import flag_dataset
from smart_financial_coach.intelligence.anomaly.contract import evidence_errors
from smart_financial_coach.intelligence.anomaly.reasons import reason


def test_flags_are_written_with_reasons(flagged_sources: DataSources) -> None:
    assert flagged_sources.flags is not None
    flags = load_flags(flagged_sources.flags)

    assert len(flags) > 0
    assert set(flags["reason_code"]) <= {"duplicate", "amount_unusual", "new_merchant"}
    for row in flags.itertuples():
        assert evidence_errors(str(row.reason_code), row.evidence) == []
        assert reason(str(row.reason_code), str(row.evidence))
        assert row.flag_id == f"fr7-test:{row.transaction_id}"
    meta = load_flag_meta(flagged_sources.flags)
    assert meta["model_version"] == "fr7-test"
    assert int(meta["rows"]) == len(flags)


def test_a_subset_scored_against_the_full_pool_gets_the_same_flags(
    small_sqlite: Path, flag_artifacts: Path, two_users: tuple[str, str], tmp_path: Path
) -> None:
    everyone = tmp_path / "all.sqlite"
    flag_dataset(small_sqlite, everyone, artifacts_dir=flag_artifacts)
    subset = tmp_path / "subset.sqlite"
    _subset(small_sqlite, list(two_users), subset)

    alone = tmp_path / "alone.sqlite"
    flag_dataset(subset, alone, pool=small_sqlite, artifacts_dir=flag_artifacts)

    full = load_flags(everyone)
    mine = full[full["user_id"].isin(two_users)].reset_index(drop=True)
    assert load_flags(alone)[["transaction_id", "reason_code", "evidence"]].equals(
        mine[["transaction_id", "reason_code", "evidence"]]
    )


def test_a_failed_run_leaves_no_file(tmp_path: Path) -> None:
    out = tmp_path / "flags.sqlite"
    with pytest.raises(RuntimeError), FlagWriter(out, {"model_version": "x"}):
        raise RuntimeError("scoring failed")

    assert not out.exists()
    assert not list(tmp_path.glob("*.part"))


def test_predict_cli_flags_a_dataset(
    small_sqlite: Path, flag_artifacts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "flags.sqlite"
    argv = ["predict", "--task", "unusual_transactions", "--data", str(small_sqlite)]
    argv += ["--out", str(out), "--artifacts-dir", str(flag_artifacts)]

    assert model_main(argv) == 0
    assert "flagged" in capsys.readouterr().out
    flagged = load_flags(out)
    assert set(flagged["user_id"]) <= set(store.load_users(small_sqlite)["user_id"])
    assert all(json.loads(e) for e in flagged["evidence"])


def test_the_flag_file_is_readable_by_everyone(flagged_sources: DataSources) -> None:
    import stat

    assert flagged_sources.flags is not None
    assert stat.S_IMODE(flagged_sources.flags.stat().st_mode) & 0o044 == 0o044
