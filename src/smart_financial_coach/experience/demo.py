"""Build the demo bundle the web app serves: a few users' data, categorized, read-only.

    sfc-web build-demo --data data/synthetic/default.sqlite --out build/demo

writes `dataset.sqlite` (the demo accounts' model-visible rows and the dataset's meta),
`predictions.sqlite` (the promoted categorizer's output for them, FR-3) and `accounts.yaml`.
The image ships this bundle, so serving needs no model, no network and no writable disk
(Web App UI, "Demo build").
"""

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from smart_financial_coach.data import store
from smart_financial_coach.data.generator.dataset import MODEL_TABLES, TABLES
from smart_financial_coach.experience.accounts import load_accounts
from smart_financial_coach.intelligence.categorization.batch import categorize_dataset

ACCOUNTS_FILE = "accounts.yaml"


@dataclass(frozen=True)
class DemoBundle:
    root: Path
    users: int
    transactions: int
    model_version: str


def _ddl(name: str) -> str:
    table = TABLES[name]
    cols = [f"{c.name} {c.type}{'' if c.nullable else ' NOT NULL'}" for c in table.columns]
    cols.append(f"PRIMARY KEY ({', '.join(table.primary_key)})")
    return f"CREATE TABLE {name} ({', '.join(cols)}) WITHOUT ROWID"


def _subset(source: Path, user_ids: list[str], target: Path) -> int:
    """Copy the users' model-visible rows and the meta table; never a truth table."""
    target.unlink(missing_ok=True)
    conn = sqlite3.connect(target, uri=True)  # so ATTACH can open the source read-only
    try:
        conn.execute("ATTACH DATABASE ? AS src", (f"file:{source}?mode=ro",))
        marks = ", ".join("?" for _ in user_ids)
        for name in MODEL_TABLES:
            conn.execute(_ddl(name))
            columns = ", ".join(TABLES[name].column_names)
            conn.execute(
                f"INSERT INTO {name} SELECT {columns} FROM src.{name} WHERE user_id IN ({marks})",
                user_ids,
            )
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta SELECT key, value FROM src.meta")
        conn.execute("CREATE INDEX idx_transactions_user_ts ON transactions (user_id, ts)")
        conn.commit()
        conn.execute("DETACH DATABASE src")
        (rows,) = conn.execute("SELECT count(*) FROM transactions").fetchone()
    finally:
        conn.close()
    return int(rows)


def build_demo(
    data: Path, accounts_file: Path, out: Path, *, artifacts_dir: Path | None = None
) -> DemoBundle:
    accounts = load_accounts(accounts_file)
    user_ids = [a.user_id for a in accounts]
    known = set(store.load_users(data)["user_id"])
    missing = [u for u in user_ids if u not in known]
    if missing:
        raise ValueError(f"demo accounts name users that aren't in {data}: {missing}")
    out.mkdir(parents=True, exist_ok=True)
    dataset = out / "dataset.sqlite"
    rows = _subset(data, user_ids, dataset)
    run = categorize_dataset(
        dataset, out / "predictions.sqlite", artifacts_dir=artifacts_dir, overwrite=True
    )
    if run.rows != rows:
        raise RuntimeError(f"categorized {run.rows} of {rows} transactions")
    shutil.copyfile(accounts_file, out / ACCOUNTS_FILE)
    # Readable by everyone: the predictions file is written through a temporary file, which is
    # owner-only, and the image serves the bundle as a different user from the one that built it
    for path in out.iterdir():
        path.chmod(0o644)
    return DemoBundle(out, len(user_ids), rows, run.model_version)
