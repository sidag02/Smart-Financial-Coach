from pathlib import Path

import pytest

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.generator import Dataset, Spec, generate, load_spec

CONFIGS = PROJECT_ROOT / "configs" / "data"


@pytest.fixture(scope="session")
def configs() -> Path:
    return CONFIGS


@pytest.fixture(scope="session")
def small_spec() -> Spec:
    return load_spec(CONFIGS / "small.yaml")


@pytest.fixture(scope="session")
def small_dataset(small_spec: Spec) -> Dataset:
    return generate(small_spec)
