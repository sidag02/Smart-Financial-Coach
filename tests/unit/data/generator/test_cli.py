from pathlib import Path

import pytest

from smart_financial_coach.data.generator.cli import main


def test_generate_validate_and_hash(
    tmp_path: Path, configs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "small.sqlite"
    spec = str(configs / "small.yaml")

    assert main(["generate", "--spec", spec, "--out", str(out)]) == 0
    generated = capsys.readouterr().out
    assert "validation passed" in generated

    assert main(["generate", "--spec", spec, "--out", str(out)]) == 1  # refuses to overwrite
    assert main(["validate", "--spec", spec, str(out)]) == 0
    capsys.readouterr()

    assert main(["hash", str(out)]) == 0
    digest = capsys.readouterr().out.strip()
    assert f"content hash {digest}" in generated

    # A database checked against a spec it wasn't built from is rejected
    assert main(["validate", "--spec", str(configs / "clean.yaml"), str(out)]) == 1
