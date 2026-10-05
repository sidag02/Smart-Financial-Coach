"""Build the demo bundle the web app serves: a few users' data, categorized, read-only.

    sfc-web build-demo --data data/synthetic/default.sqlite --out build/demo

writes `dataset.sqlite` (the demo accounts' model-visible rows and the dataset's meta),
`predictions.sqlite` (the promoted categorizer's output for them, FR-3), `accounts.yaml`,
`replay.json`, the feedback replay's result (FR-5/FR-6 §7) that the "How it learns" page shows,
when one has been committed, and, once an FR-7 model is promoted, `flags.sqlite`: its
unusual-charge flags for them, scored against merchant profiles of every user in `--data` (FR-7
§8). Without a promoted FR-7 model the bundle has no flag file and unusual charges stay "not
available yet". Likewise, once a goal-forecasting model is promoted, `forecasts.json`: each demo
account's forecast state and what serving needs of the model (FR-11 and FR-12 design, §7); without
one, the file has naive pace behind it (owner decision 10 on #54): a simple projection of each
goal's pace so far, with no chance or range, so the pipeline runs end to end until a model is
promoted. And `spikes.json` (FR-8 §8): the spike scorer and the category season profiles, built
from every user in `--data` on the promoted categorizer's predictions, since the demo accounts
alone are too few for a profile. Without a promoted spike model it holds the simple rule (owner
decision 12 on #58), labelled as such.
The image ships this bundle, so serving needs no model, no network and no writable disk
(Web App UI, "Demo build").
"""

import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

from smart_financial_coach.config import PROJECT_ROOT, get_settings
from smart_financial_coach.data import store
from smart_financial_coach.data.generator.dataset import MODEL_TABLES, TABLES
from smart_financial_coach.data.predictions import load_categories
from smart_financial_coach.experience.accounts import load_accounts
from smart_financial_coach.intelligence.anomaly.batch import SERVICE as FLAG_SERVICE
from smart_financial_coach.intelligence.anomaly.batch import flag_dataset
from smart_financial_coach.intelligence.categorization.batch import categorize_dataset
from smart_financial_coach.intelligence.forecasting.batch import FORECASTS_FILE, forecast_dataset
from smart_financial_coach.intelligence.forecasting.contract import SERVICE as FORECAST_SERVICE
from smart_financial_coach.intelligence.models.artifact import POINTER_FILE
from smart_financial_coach.intelligence.spikes.batch import (
    SPIKES_FILE,
    SpikeState,
    build_state,
    write_state,
)

ACCOUNTS_FILE = "accounts.yaml"
REPLAY_FILE = "replay.json"
REPLAY = PROJECT_ROOT / "docs" / "reports" / "fr5-replay.json"  # sfc-experiment replay's output


@dataclass(frozen=True)
class DemoBundle:
    root: Path
    users: int
    transactions: int
    model_version: str
    flag_model_version: str | None = None  # None: no FR-7 model promoted, no flags
    flags: int = 0  # rows in the flag file: down to More often's cutoff with presets (FR-9)
    flags_balanced: int = 0  # of them, flagged at Balanced
    flag_presets: bool = False  # the FR-7 model has sensitivity presets
    forecast_model_version: str | None = None  # BASELINE_VERSION: no forecasting model promoted
    spike_model_version: str | None = None  # SIMPLE_RULE: no spike model promoted
    spike_method: str | None = None


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
    data: Path,
    accounts_file: Path,
    out: Path,
    *,
    artifacts_dir: Path | None = None,
    replay: Path = REPLAY,
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
    flags_file = out / "flags.sqlite"
    flags_file.unlink(missing_ok=True)  # a stale file would show another model's flags
    flag_run = None
    artifacts = artifacts_dir or get_settings().artifacts_dir
    if (artifacts / FLAG_SERVICE / POINTER_FILE).exists():
        flag_run = flag_dataset(dataset, flags_file, pool=data, artifacts_dir=artifacts_dir)
    forecasts_file = out / FORECASTS_FILE
    forecasts_file.unlink(missing_ok=True)  # a stale file would forecast with another model
    promoted = (artifacts / FORECAST_SERVICE / POINTER_FILE).exists()
    forecast_run = forecast_dataset(
        dataset, forecasts_file, artifacts_dir=artifacts_dir, baseline=not promoted
    )
    spike_state = _spikes(data, out / SPIKES_FILE, artifacts_dir)
    shutil.copyfile(accounts_file, out / ACCOUNTS_FILE)
    if replay.exists():
        shutil.copyfile(replay, out / REPLAY_FILE)
    # Readable by everyone: the predictions file is written through a temporary file, which is
    # owner-only, and the image serves the bundle as a different user from the one that built it
    for path in out.iterdir():
        path.chmod(0o644)
    return DemoBundle(
        out,
        len(user_ids),
        rows,
        run.model_version,
        flag_run.model_version if flag_run else None,
        flag_run.flagged if flag_run else 0,
        flag_run.balanced if flag_run else 0,
        flag_run is not None and flag_run.presets is not None,
        forecast_run.model_version,
        spike_state.version,
        spike_state.method,
    )


def _spikes(data: Path, out: Path, artifacts_dir: Path | None) -> SpikeState:
    """The spikes file: season profiles from every user in `data`, on the promoted categorizer's
    predictions (FR-8 §2: shared, nobody's corrections), and the scorer."""
    out.unlink(missing_ok=True)  # a stale file would score with another model
    with tempfile.TemporaryDirectory() as tmp:
        predictions = Path(tmp) / "pool_predictions.sqlite"
        run = categorize_dataset(data, predictions, artifacts_dir=artifacts_dir, overwrite=True)
        pool = store.load_transactions(data).merge(
            load_categories(predictions)[["transaction_id", "category"]], on="transaction_id"
        )
    state = build_state(
        pool, as_of=store.load_meta(data)["calendar_end"], artifacts_dir=artifacts_dir
    )
    write_state(state, out, users=int(pool["user_id"].nunique()), categorizer=run.model_version)
    return state
