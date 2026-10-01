# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "fastembed==0.8.1",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
# ]
# ///
"""FR-3 feasibility: how far do simple categorizers get on the default dataset?

Proof of concept for the FR-3 Transaction Categorization feature design. Not product code:
it reads truth tables directly and lives outside `src/`.

    uv run experiments/fr3_categorization/feasibility.py data/synthetic/default.sqlite

Writes `results/feasibility.json` and `results/feasibility.md` next to this file.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

SEED = 0
MAX_ROWS_PER_CLASS = 20_000
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BOOTSTRAP_REPS = 1_000
CHANNELS = ["card_present", "online", "ach", "other"]
INCOME = "Income"

# Generic bank-feed vocabulary (FR-3 design §1), not read from the merchant catalog.
PREFIX = re.compile(r"^(POS DEBIT|ACH DEBIT|ACH CREDIT|SQ \*|TST\* ?|PAYPAL \*|SP \* ?)\s*", re.I)
STORE_NUMBER = re.compile(r"(#|STORE)\s*\d+.*$", re.I)

FEATURE_SETS = {
    "ngrams": "Character 2-4-grams of merchant_raw + amount bin, sign, channel, hour",
    "embeddings": f"Sentence embedding of normalized text ({EMBEDDING_MODEL}) + same side features",
    "both": "Both",
}


def normalize_merchant(raw: str) -> str:
    text = PREFIX.sub("", raw)
    text = re.sub(r"\*.*$", "", text)
    text = STORE_NUMBER.sub("", text)
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with sqlite3.connect(path) as con:
        frame = pd.read_sql(
            """
            SELECT t.transaction_id, t.user_id, t.ts, t.amount, t.merchant_raw, t.channel,
                   u.split, tt.category, tt.merchant_id, m.canonical_name, m.holdout,
                   m.is_ambiguous
            FROM transactions t
            JOIN users u USING (user_id)
            JOIN truth_transactions tt USING (transaction_id)
            JOIN truth_merchants m ON m.merchant_id = tt.merchant_id
            ORDER BY t.transaction_id
            """,
            con,
        )
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    frame["norm"] = frame["merchant_raw"].map(normalize_merchant)
    return frame, meta


def dataset_facts(frame: pd.DataFrame) -> dict[str, Any]:
    spending = frame[frame["category"] != INCOME]
    holdout = frame[(frame["split"] == "test") & (frame["holdout"] == 1)]
    return {
        "transactions": len(frame),
        "users": int(frame["user_id"].nunique()),
        "transactions_by_split": frame["split"].value_counts().to_dict(),
        "transactions_by_category": frame["category"].value_counts().to_dict(),
        "merchants": int(frame["merchant_id"].nunique()),
        "distinct_merchant_raw": int(frame["merchant_raw"].nunique()),
        "distinct_normalized": int(frame["norm"].nunique()),
        "positive_spending_by_category": spending[spending["amount"] > 0]["category"]
        .value_counts()
        .to_dict(),
        "ambiguous_spending_share": float(spending["is_ambiguous"].mean()),
        "holdout_merchants_used_by_test_users": int(holdout["merchant_id"].nunique()),
        "holdout_transactions": len(holdout),
        "holdout_merchants_by_category": holdout.groupby("category")["merchant_id"]
        .nunique()
        .to_dict(),
        "holdout_transactions_by_category": holdout["category"].value_counts().to_dict(),
        "holdout_merchants_seen_by_train_users": int(
            frame[(frame["split"] == "train") & (frame["holdout"] == 1)]["merchant_id"].nunique()
        ),
    }


def ambiguity_ceiling(frame: pd.DataFrame) -> dict[str, float]:
    """Share of spending a perfect per-merchant majority-category rule still gets wrong."""
    spending = frame[frame["category"] != INCOME]
    majority = spending.groupby("merchant_id")["category"].agg(lambda s: s.value_counts().index[0])
    wrong = spending["category"] != spending["merchant_id"].map(majority)
    ambiguous = spending[spending["is_ambiguous"] == 1]
    return {
        "error_share_all_spending": float(wrong.mean()),
        "error_share_at_ambiguous_merchants": float(wrong[ambiguous.index].mean()),
    }


def side_features(frame: pd.DataFrame) -> sp.csr_matrix:
    n = len(frame)
    rows = np.arange(n)
    log_amount = np.log1p(frame["amount"].abs().to_numpy())
    amount_bins = np.digitize(log_amount, np.linspace(0, 9, 19))
    amount = sp.csr_matrix((np.ones(n), (rows, amount_bins)), shape=(n, 20))
    channel = (
        pd.get_dummies(frame["channel"]).reindex(columns=CHANNELS, fill_value=0).to_numpy(float)
    )
    sign = (frame["amount"].to_numpy() > 0).astype(float)[:, None]
    hour = pd.to_datetime(frame["ts"]).dt.hour.to_numpy() // 3
    hours = sp.csr_matrix((np.ones(n), (rows, hour)), shape=(n, 8))
    return sp.hstack([amount, sp.csr_matrix(channel), sp.csr_matrix(sign), hours]).tocsr()


class Featurizer:
    def __init__(self, kind: str, embeddings: dict[str, np.ndarray]) -> None:
        self.kind = kind
        self.embeddings = embeddings
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            sublinear_tf=True,
            max_features=200_000,
        )

    def fit(self, frame: pd.DataFrame) -> Featurizer:
        if self.kind in ("ngrams", "both"):
            self.vectorizer.fit(frame["merchant_raw"])
        return self

    def transform(self, frame: pd.DataFrame) -> sp.csr_matrix:
        blocks = [side_features(frame)]
        if self.kind in ("embeddings", "both"):
            blocks.append(sp.csr_matrix(np.vstack([self.embeddings[s] for s in frame["norm"]])))
        if self.kind in ("ngrams", "both"):
            blocks.append(self.vectorizer.transform(frame["merchant_raw"]))
        return sp.hstack(blocks).tocsr()


def macro_f1_spending(truth: pd.Series, pred: np.ndarray) -> float:
    mask = (truth != INCOME).to_numpy()
    return float(f1_score(truth[mask], pred[mask], average="macro"))


def per_class(truth: pd.Series, pred: np.ndarray, labels: list[str]) -> dict[str, dict[str, Any]]:
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    return {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": int(n[i])}
        for i, label in enumerate(labels)
        if n[i] > 0
    }


def merchant_bootstrap(
    frame: pd.DataFrame, pred: np.ndarray, labels: list[str], rng: np.random.Generator
) -> dict[str, float]:
    """95% interval for spending macro F1, resampling merchants within each true category.

    The unseen set has few merchants per class, so merchants, not transactions, are the
    independent units.
    """
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    spending = (frame["category"] != INCOME).to_numpy()
    truth_idx = frame["category"].map(index).to_numpy()[spending]
    pred_idx = pd.Series(pred).map(index).to_numpy()[spending]
    merchants = frame["merchant_id"].to_numpy()[spending]
    codes, uniques = pd.factorize(merchants)
    confusion = np.zeros((len(uniques), k, k))
    np.add.at(confusion, (codes, truth_idx, pred_idx), 1)
    merchant_class = confusion.sum(axis=2).argmax(axis=1)
    groups = [np.flatnonzero(merchant_class == c) for c in range(k)]
    groups = [g for g in groups if len(g)]
    present = sorted({int(c) for c in merchant_class})

    def macro(total: np.ndarray) -> float:
        tp = np.diag(total)
        precision = np.divide(tp, total.sum(0), out=np.zeros(k), where=total.sum(0) > 0)
        recall = np.divide(tp, total.sum(1), out=np.zeros(k), where=total.sum(1) > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
        return float(f1[present].mean())

    scores = []
    for _ in range(BOOTSTRAP_REPS):
        sample = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        scores.append(macro(confusion[sample].sum(axis=0)))
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"point": macro(confusion.sum(axis=0)), "low": float(low), "high": float(high)}


def top_errors(frame: pd.DataFrame, pred: np.ndarray, n: int = 12) -> list[dict[str, Any]]:
    wrong = frame.assign(pred=pred)
    wrong = wrong[wrong["pred"] != wrong["category"]]
    counts = (
        wrong.groupby(["canonical_name", "category", "pred"]).size().sort_values(ascending=False)
    )
    return [
        {"merchant": m, "true": t, "predicted": p, "transactions": int(c)}
        for (m, t, p), c in counts.head(n).items()
    ]


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def run(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    frame, meta = load(path)
    labels = sorted(frame["category"].unique())
    spending_labels = [c for c in labels if c != INCOME]

    train_users = frame[frame["split"] == "train"]
    fit_rows, known = train_test_split(
        train_users, test_size=0.2, stratify=train_users["category"], random_state=SEED
    )
    fit_rows = fit_rows.sample(frac=1, random_state=SEED)
    fit_rows = fit_rows[fit_rows.groupby("category").cumcount() < MAX_ROWS_PER_CLASS]
    test_users = frame[frame["split"] == "test"]
    unseen = test_users[test_users["holdout"] == 1]
    test_sets = {"known": known, "test_users": test_users, "unseen": unseen}

    from fastembed import TextEmbedding

    strings = sorted(frame["norm"].unique())
    t0 = time.perf_counter()
    vectors = list(TextEmbedding(EMBEDDING_MODEL).embed(strings, batch_size=256))
    embed_seconds = time.perf_counter() - t0
    embeddings = dict(zip(strings, vectors, strict=True))

    rng = np.random.default_rng(SEED)
    experiments: dict[str, Any] = {}
    for kind, description in FEATURE_SETS.items():
        featurizer = Featurizer(kind, embeddings).fit(fit_rows)
        t0 = time.perf_counter()
        model = LogisticRegression(
            C=10 if kind == "ngrams" else 3, max_iter=500, class_weight="balanced"
        )
        model.fit(featurizer.transform(fit_rows), fit_rows["category"])
        fit_seconds = time.perf_counter() - t0
        result: dict[str, Any] = {"description": description, "fit_seconds": fit_seconds}
        for name, rows in test_sets.items():
            pred = model.predict(featurizer.transform(rows))
            result[name] = {
                "n": len(rows),
                "macro_f1_spending": macro_f1_spending(rows["category"], pred),
                "per_class": per_class(rows["category"], pred, labels),
            }
            if name == "unseen":
                result[name]["merchant_bootstrap"] = merchant_bootstrap(rows, pred, labels, rng)
                result[name]["top_errors"] = top_errors(rows, pred)
        experiments[kind] = result
        print(
            f"{kind:10s} known={result['known']['macro_f1_spending']:.3f} "
            f"test_users={result['test_users']['macro_f1_spending']:.3f} "
            f"unseen={result['unseen']['macro_f1_spending']:.3f}",
            flush=True,
        )

    return {
        "dataset": {
            "path": str(path),
            "spec_name": meta.get("spec_name"),
            "spec_hash": meta.get("spec_hash"),
            "generator_version": meta.get("generator_version"),
        },
        "environment": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": sklearn.__version__,
            "embedding_model": EMBEDDING_MODEL,
        },
        "setup": {
            "seed": SEED,
            "max_rows_per_class": MAX_ROWS_PER_CLASS,
            "fit_rows": len(fit_rows),
            "spending_labels": spending_labels,
            "bootstrap_reps": BOOTSTRAP_REPS,
        },
        "facts": dataset_facts(frame),
        "ambiguity_ceiling": ambiguity_ceiling(frame),
        "embedding": {"strings": len(strings), "seconds": embed_seconds},
        "experiments": experiments,
        "total_seconds": time.perf_counter() - started,
    }


def report(results: dict[str, Any]) -> str:
    exps = results["experiments"]
    facts = results["facts"]
    lines = [
        "# FR-3 feasibility results",
        "",
        "Generated by `feasibility.py`; do not edit by hand. "
        "Macro F1 is over the 12 spending categories (Income excluded).",
        "",
        f"- Dataset: `{results['dataset']['spec_name']}`, spec hash "
        f"`{results['dataset']['spec_hash'][:12]}`, generator "
        f"{results['dataset']['generator_version']}",
        f"- Code: commit `{results['environment']['git_commit'][:7]}`, Python "
        f"{results['environment']['python']}, scikit-learn {results['environment']['sklearn']}",
        f"- Training rows: {results['setup']['fit_rows']:,} (train users, at most "
        f"{results['setup']['max_rows_per_class']:,} per class), seed {results['setup']['seed']}",
        f"- Runtime: {results['total_seconds']:.0f} s total; {results['embedding']['strings']:,} "
        f"normalized strings embedded in {results['embedding']['seconds']:.1f} s",
        "",
        "## Headline",
        "",
        "| Features | Known merchants | All test users | Unseen merchants | "
        "Unseen, 95% merchant-bootstrap interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in exps.values():
        boot = result["unseen"]["merchant_bootstrap"]
        lines.append(
            f"| {result['description']} | {result['known']['macro_f1_spending']:.3f} | "
            f"{result['test_users']['macro_f1_spending']:.3f} | "
            f"{result['unseen']['macro_f1_spending']:.3f} | "
            f"{boot['low']:.3f} to {boot['high']:.3f} |"
        )
    sizes = " / ".join(f"{exps['ngrams'][k]['n']:,}" for k in ("known", "test_users", "unseen"))
    lines += ["", f"Test-set sizes (known / all test users / unseen): {sizes} transactions.", ""]

    lines += ["## Unseen merchants: per-class F1", "", "| Category | Merchants | Transactions |"]
    lines[-1] += " " + " | ".join(exps) + " |"
    lines.append("| --- | --- | --- |" + " --- |" * len(exps))
    for category, n in sorted(facts["holdout_transactions_by_category"].items()):
        cells = " | ".join(
            f"{e['unseen']['per_class'].get(category, {}).get('f1', 0):.3f}" for e in exps.values()
        )
        lines.append(
            f"| {category} | {facts['holdout_merchants_by_category'][category]} | {n:,} | {cells} |"
        )

    lines += ["", "## Largest unseen-merchant errors (both)", ""]
    lines += ["| Merchant | True | Predicted | Transactions |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {e['merchant']} | {e['true']} | {e['predicted']} | {e['transactions']:,} |"
        for e in exps["both"]["unseen"]["top_errors"]
    ]
    lines += ["", "## Largest unseen-merchant errors (ngrams)", ""]
    lines += ["| Merchant | True | Predicted | Transactions |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {e['merchant']} | {e['true']} | {e['predicted']} | {e['transactions']:,} |"
        for e in exps["ngrams"]["unseen"]["top_errors"]
    ]

    ceiling = results["ambiguity_ceiling"]
    lines += [
        "",
        "## Dataset facts",
        "",
        f"- {facts['transactions']:,} transactions, {facts['users']} users, "
        f"{facts['merchants']} merchants",
        f"- {facts['distinct_merchant_raw']:,} distinct `merchant_raw` strings; "
        f"{facts['distinct_normalized']:,} after normalization",
        f"- Positive spending amounts (refunds): {facts['positive_spending_by_category']}",
        f"- Spending at ambiguous merchants: {facts['ambiguous_spending_share']:.1%}; "
        f"majority-category rule error: {ceiling['error_share_all_spending']:.1%} of all spending",
        f"- Holdout merchants used by test users: "
        f"{facts['holdout_merchants_used_by_test_users']}, "
        f"{facts['holdout_transactions']:,} transactions; holdout merchants seen by train "
        f"users: {facts['holdout_merchants_seen_by_train_users']}",
        "- Transactions by category: "
        + ", ".join(f"{k} {v:,}" for k, v in facts["transactions_by_category"].items()),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    results = run(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "feasibility.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "feasibility.md").write_text(report(results))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
