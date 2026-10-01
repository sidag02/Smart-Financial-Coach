"""Model code must never read ground truth (FR-2 §5): no `truth_*` tables, no label contract."""

import re
from pathlib import Path

from smart_financial_coach.config import PROJECT_ROOT

MODEL_CODE = PROJECT_ROOT / "src" / "smart_financial_coach" / "intelligence"
FORBIDDEN = re.compile(r"truth_|data\.labels|data import labels")


def truth_reads(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}: {line.strip()}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FORBIDDEN.search(line)
    ]


def test_model_code_never_reads_truth() -> None:
    assert MODEL_CODE.is_dir()
    assert truth_reads(MODEL_CODE) == []


def test_check_catches_truth_reads(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text(
        'rows = conn.execute("SELECT * FROM truth_periods")\n'
        "from smart_financial_coach.data import labels\n"
        "from smart_financial_coach.data.labels import load_truth\n",
        encoding="utf-8",
    )

    assert len(truth_reads(tmp_path)) == 3
