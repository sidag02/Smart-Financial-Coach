"""Model code must never read ground truth (FR-2 §5): no `truth_*` tables, no label contract.

Serving code must not depend on the experiment tracker either: no `mlflow` under `intelligence/`.
"""

import re
from pathlib import Path

import pytest

from smart_financial_coach.config import PROJECT_ROOT

PACKAGE = PROJECT_ROOT / "src" / "smart_financial_coach"
MODEL_CODE = PACKAGE / "intelligence"
# Model-visible code outside intelligence/: the data-access layer, the predictions store and the
# feature pipeline
MODEL_VISIBLE = (
    MODEL_CODE,
    PACKAGE / "data" / "store.py",
    PACKAGE / "data" / "predictions.py",
    PACKAGE / "data" / "features",
)
FORBIDDEN = re.compile(r"truth_|data\.labels|data import labels|\bmlflow\b", re.IGNORECASE)


def truth_reads(root: Path) -> list[str]:
    files = [root] if root.is_file() else sorted(root.rglob("*.py"))
    return [
        f"{path.relative_to(PACKAGE if root.is_file() else root)}:{number}: {line.strip()}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]


@pytest.mark.parametrize("root", MODEL_VISIBLE, ids=lambda p: p.name)
def test_model_code_never_reads_truth(root: Path) -> None:
    assert root.exists()
    assert truth_reads(root) == []


def test_check_catches_truth_reads(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text(
        'rows = conn.execute("SELECT * FROM truth_periods")\n'
        "from smart_financial_coach.data import labels\n"
        "from smart_financial_coach.data.labels import load_truth\n"
        "from smart_financial_coach.data.generator import TRUTH_TABLES\n"
        "import mlflow\n",
        encoding="utf-8",
    )

    assert len(truth_reads(tmp_path)) == 5
