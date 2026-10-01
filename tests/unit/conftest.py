from pathlib import Path

import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.generator import generate, load_spec


@pytest.fixture(scope="session")
def small_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The small spec (30 users), generated once per session and written to SQLite."""
    path = tmp_path_factory.mktemp("data") / "small.sqlite"
    generate(load_spec(PROJECT_ROOT / "configs" / "data" / "small.yaml")).to_sqlite(path)
    return path
