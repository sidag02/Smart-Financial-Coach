"""FR-7 feasibility: unusual-transaction detectors on train users only. No test user is read.

    uv run sfc-data generate --spec configs/data/default.yaml --out data/synthetic/default.sqlite
    uv run python experiments/fr7_unusual/feasibility.py data/synthetic/default.sqlite

Every detector is causal: a charge is scored only against the same user's earlier charges. The
population priors (a merchant's typical price and spread) pool train users' charges at the same
normalized merchant text, need at least 3 other users, and are not causal in time; see the design's
Feasibility section. Thresholds are chosen and measured on the same train users, so the operating
points here are optimistic; the FR-7 round tunes on folds and measures out of fold.
"""

import sys

import numpy as np
import pandas as pd

from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.labels import load_truth

MIN_POP_USERS = 3
MIN_SD = 0.25  # log-amount spread floor
REASONS = {
    "duplicate": "duplicate",
    "amount_outlier": "amount_unusual",
    "new_merchant_large": "new_merchant",
}


def features(path: str) -> pd.DataFrame:
    truth = load_truth(path)
    split = truth.users.set_index("user_id")["split"]
    tx = truth.transactions
    tx = tx[tx["user_id"].map(split) == "train"].copy()
    tx["t"] = pd.to_datetime(tx["ts"])
    tx = tx.sort_values(["user_id", "t", "transaction_id"], kind="mergesort", ignore_index=True)
    out = tx[tx["amount"] < 0].copy()  # outflows; refunds and income are never scored
    out["key"] = out["merchant_raw"].map(normalize_merchant)
    out["x"] = np.log(-out["amount"])
    out["y"] = out["anomaly_kind"].notna()

    # Baseline (Technical Design): z-score of the amount against the user's earlier outflows
    amt, by_user = -out["amount"], out["user_id"]
    n = by_user.groupby(by_user).cumcount()
    s1 = amt.groupby(by_user).cumsum() - amt
    s2 = (amt**2).groupby(by_user).cumsum() - amt**2
    mean = s1 / n.clip(lower=1)
    sd = np.sqrt((s2 / n.clip(lower=1) - mean**2).clip(lower=0))
    out["baseline"] = ((amt - mean) / sd.replace(0, np.nan)).where(n >= 10).fillna(0.0)

    # Duplicate: same user, raw text and amount as an earlier charge, within 90 minutes
    cols = ["user_id", "merchant_raw", "amount"]
    o = out.sort_values([*cols, "t"], kind="mergesort")
    same = (o[cols] == o[cols].shift()).all(axis=1)
    gap = o["t"].diff().dt.total_seconds() / 60
    out["dup"] = (same & (gap <= 90)).reindex(out.index)

    # The user's own history at this merchant: count, running median and MAD of log amount
    gk = out.groupby(["user_id", "key"], sort=False)["x"]
    out["k_n"] = gk.cumcount()
    out["k_med"] = gk.transform(lambda s: s.expanding().median().shift())
    out["k_mad"] = gk.transform(
        lambda s: (s - s.expanding().median()).abs().expanding().median().shift()
    )
    out["first"] = out["k_n"] == 0
    # Rank of the amount among the user's earlier outflows (needs 30 of them)
    out["u_rank"] = out.groupby("user_id")["x"].transform(
        lambda s: pd.Series(
            [np.mean(s.values[:i] < v) if i >= 30 else np.nan for i, v in enumerate(s.values)],
            index=s.index,
        )
    )

    # Population priors per merchant key, from other train users
    per_user = out.groupby(["key", "user_id"])["x"].median().reset_index()
    users_at_key = per_user.groupby("key")["user_id"].nunique()
    out["pop_users"] = out["key"].map(users_at_key) - 1
    has_pop = out["pop_users"] >= MIN_POP_USERS
    typical = per_user.groupby("key")["x"].median()  # median of users' medians
    out["pop_ratio"] = np.exp(out["x"] - out["key"].map(typical)).where(has_pop)
    dev = out["x"] - out.groupby(["user_id", "key"])["x"].transform("median")
    spread = dev.abs().groupby(out["key"]).median() * 1.4826
    pop_sd = out["key"].map(spread).where(has_pop).fillna(0.0)
    sd = np.maximum(np.maximum(1.4826 * out["k_mad"].fillna(0.0), pop_sd), MIN_SD)
    out["z"] = ((out["x"] - out["k_med"]) / sd).where(out["k_n"] >= 1)

    keep = out["y"] | ~out["transaction_id"].isin(truth._ignored_transaction_ids())
    return out[keep]


def rules(f: pd.DataFrame, z: float, ratio: float, min_amount: float) -> pd.Series:
    """Reason code per charge, or None: duplicate first, then new merchant, then amount."""
    dup = f["dup"]
    new = f["first"] & (f["pop_ratio"] >= ratio) & (-f["amount"] >= min_amount) & ~dup
    big = (f["z"] >= z) & ~dup & ~new
    code = pd.Series(None, index=f.index, dtype=object)
    code[dup], code[new], code[big] = "duplicate", "new_merchant", "amount_unusual"
    return code


def pr(flag: pd.Series, y: pd.Series) -> tuple[int, int, float, float]:
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    return tp, fp, tp / max(tp + fp, 1), tp / int(y.sum())


def main(path: str) -> None:
    f = features(path)
    y, kind = f["y"], f["anomaly_kind"]
    months = f.groupby("user_id")["t"].agg(lambda t: (t.max() - t.min()).days / 30.4).sum()
    print(
        f"## Data\n\n{len(f):,} scored train outflows, {int(y.sum())} planted "
        f"({y.mean():.3%}), {months:,.0f} user-months; "
        + ", ".join(f"{k} {v}" for k, v in kind[y].value_counts().sort_index().items())
    )

    print("\n## Components alone (precision, recall of all planted)\n")
    print("| Component | Threshold | Flags | Precision | TP by kind |\n|---|---|---|---|---|")
    comps = {
        "Duplicate (exact repeat within 90 min)": [(None, f["dup"])],
        "Amount vs own history, own spread only": [
            (t, (f["x"] - f["k_med"]) / np.maximum(1.4826 * f["k_mad"].fillna(0), MIN_SD) >= t)
            for t in (5, 8)
        ],
        "Amount vs own history, population spread": [(t, f["z"] >= t) for t in (4, 5, 6)],
        "New merchant: rank in the user's own history": [
            (t, f["first"] & (f["u_rank"] >= t)) for t in (0.95, 0.99)
        ],
        "New merchant: population price ratio": [
            (t, f["first"] & (f["pop_ratio"] >= t)) for t in (3, 5, 8)
        ],
    }
    for name, points in comps.items():
        for th, flag in points:
            tp, fp, p, _ = pr(flag.fillna(False), y)
            kinds = kind[flag.fillna(False) & y].value_counts().to_dict()
            print(f"| {name} | {th if th is not None else '-'} | {tp + fp} | {p:.3f} | {kinds} |")

    big_first = f["first"] & (-f["amount"] >= 250)
    planted_new = f["anomaly_kind"] == "new_merchant_large"
    print(
        f"\n- profile price ratio, median: planted new-merchant charges "
        f"{f.loc[planted_new, 'pop_ratio'].median():.1f} "
        f"({f.loc[planted_new, 'pop_ratio'].notna().mean():.0%} have a profile); "
        f"normal first visits of $250 or more {f.loc[big_first & ~y, 'pop_ratio'].median():.1f} "
        f"({int((big_first & ~y).sum())} of them)"
    )
    counts = f.groupby(["user_id", "key"]).size()
    print(
        f"- charges per (user, merchant key): median {counts.median():.0f}, "
        f"share with 5 or fewer {(counts <= 5).mean():.0%}"
    )

    print("\n## Combined rules: best recall at a precision target (grid on the same users)\n")
    print(
        "| Target | z | Ratio | Min $ | Flags | Per user-month | Precision | Recall | "
        "Recall (clear) | Duplicate | Amount outlier | New merchant | Reason accuracy |"
    )
    print("|---" * 13 + "|")
    for target in (0.70, 0.75, 0.80):
        best = None
        for z in (3, 3.5, 4, 4.5, 5, 6, 7):
            for ratio in (3, 4, 5, 6, 8):
                for min_amount in (0, 100, 250):
                    code = rules(f, z, ratio, min_amount)
                    tp, fp, p, r = pr(code.notna(), y)
                    if p >= target and (best is None or r > best[0]):
                        best = (r, z, ratio, min_amount, code)
        r, z, ratio, min_amount, code = best
        flag = code.notna()
        tp, fp, p, r = pr(flag, y)
        clear = y & (f["tier"] == "clear")
        by_kind = (kind[flag & y].value_counts() / kind[y].value_counts()).fillna(0)
        want = kind.map(REASONS)
        reason_acc = float((code[flag & y] == want[flag & y]).mean())
        print(
            f"| {target:.2f} | {z} | {ratio} | {min_amount} | {tp + fp} | "
            f"{(tp + fp) / months:.3f} | {p:.3f} | {r:.3f} | "
            f"{(flag & clear).sum() / clear.sum():.3f} | {by_kind['duplicate']:.3f} | "
            f"{by_kind['amount_outlier']:.3f} | {by_kind['new_merchant_large']:.3f} | "
            f"{reason_acc:.3f} |"
        )
        if target == 0.70:
            at70 = (code, tp + fp)

    code, volume = at70
    flag = code.notna()
    print("\n## At the 0.70 operating point\n")
    for reason in ("duplicate", "new_merchant", "amount_unusual"):
        print(
            f"- precision of `{reason}` flags: {y[code == reason].mean():.3f} "
            f"({int((code == reason).sum())} flags)"
        )
    fp = f[flag & ~y]
    print(f"- false positives by process: {fp['process'].value_counts().to_dict()}")
    print(f"- most frequent false-positive merchants: {fp['key'].value_counts().head(6).to_dict()}")
    missed = f[(kind == "new_merchant_large") & ~flag]
    print(
        f"- missed `new_merchant_large`: {len(missed)}; {int(missed['pop_ratio'].isna().sum())} "
        f"with no population price; median price ratio of the rest "
        f"{missed['pop_ratio'].median():.2f}"
    )
    missed = f[(kind == "amount_outlier") & ~flag]
    print(
        f"- missed `amount_outlier`: {len(missed)}; median z {missed['z'].median():.2f}; "
        f"tiers {missed['tier'].value_counts().to_dict()}"
    )
    rng = np.random.default_rng(0)
    per = f.assign(tp=flag & y, fp=flag & ~y, pos=y).groupby("user_id")[["tp", "fp", "pos"]].sum()
    draws = [per.loc[rng.choice(per.index, len(per))].sum() for _ in range(1000)]
    p_ci = np.percentile([d.tp / (d.tp + d.fp) for d in draws], [2.5, 97.5])
    r_ci = np.percentile([d.tp / d.pos for d in draws], [2.5, 97.5])
    print(
        f"- user bootstrap, 95%: precision {p_ci[0]:.3f}-{p_ci[1]:.3f}, "
        f"recall {r_ci[0]:.3f}-{r_ci[1]:.3f}"
    )

    print("\n## Baseline at the same flag volume\n")
    for name, score in (
        ("Per-user z on amount (Technical Design baseline)", f["baseline"]),
        ("Baseline plus the duplicate rule", f["baseline"].where(~f["dup"], np.inf)),
    ):
        top = y.to_numpy()[np.argsort(-score.to_numpy(), kind="mergesort")[:volume]]
        print(
            f"- {name}: {volume} flags, precision {top.mean():.3f}, "
            f"recall {top.sum() / y.sum():.3f}"
        )
    ranked = y.to_numpy()[np.argsort(-f["baseline"].to_numpy(), kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    deepest = int(np.where(precision >= 0.70)[0].max()) + 1
    print(
        f"- the baseline's deepest cutoff with precision >= 0.70: {deepest} flags, "
        f"recall {ranked[:deepest].sum() / y.sum():.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
