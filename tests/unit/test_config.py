from pathlib import Path

import pytest

from smart_financial_coach import __version__
from smart_financial_coach.config import Settings


def test_version() -> None:
    assert __version__


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SFC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SFC_TRAIN_SEED", "7")
    monkeypatch.setenv("SFC_LLM_API_KEY", "secret")

    settings = Settings(_env_file=None)

    assert settings.sqlite_path == tmp_path / "sfc.sqlite3"
    assert settings.train_seed == 7
    assert settings.train_seed != settings.test_seed
    assert "secret" not in repr(settings)
