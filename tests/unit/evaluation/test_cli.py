"""The CLIs end to end on the toy task, with a temporary tracking store and artifacts folder."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from smart_financial_coach.evaluation.cli import experiment_main, model_main


def test_cli_run_to_promote(
    toy_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    runs: list[tuple[str, str, dict[str, Any]]] = [
        ("majority", "toy/majority", {"baseline": True}),
        ("b_memory", "toy/memory", {"grid": {"confidence": [0.7, 0.9]}}),
    ]
    for name, model, extra in runs:
        spec = {"name": name, "task": "toy", "model": {"type": model}, **extra}
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(spec))
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}"]
    data = ["--data", str(toy_data)]
    artifacts = ["--artifacts-dir", str(tmp_path / "artifacts")]

    assert experiment_main(["run", str(configs), *data, *uri]) == 0
    run_id = re.search(r"b_memory: run (\w+)", capsys.readouterr().out)
    assert run_id is not None

    assert experiment_main(["leaderboard", "--task", "toy", *data, *uri]) == 0
    assert "b_memory" in capsys.readouterr().out
    assert experiment_main(["finalize", "--task", "toy", *data, *uri]) == 0
    assert "test_known_accuracy=1.000" in capsys.readouterr().out

    assert (
        model_main(
            ["promote", "--task", "toy", "--run", run_id[1], "--note", "n", *uri, *artifacts]
        )
        == 0
    )
    assert "pass beats_baseline" in capsys.readouterr().out
    assert model_main(["show", "--task", "toy", *artifacts]) == 0
    assert run_id[1] in capsys.readouterr().out


def test_cli_reports_errors(
    toy_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    uri = ["--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}"]

    assert (
        experiment_main(
            [
                "finalize",
                "--task",
                "toy",
                "--runs",
                "a",
                "b",
                "c",
                "d",
                "--data",
                str(toy_data),
                *uri,
            ]
        )
        == 1
    )
    assert "error: naming finalists departs from the decision rule" in capsys.readouterr().err
