"""FR-8 feasibility: monthly spending spikes on TRAIN USERS ONLY.

The dataset file is loaded whole, then every truth table is cut to train users (`train_only`)
before anything is computed, so no test user's rows or labels reach a number below.

Every (user, spending category, month) after the warm-up is a period. Spend and counts use true
categories (FR-2: spike metrics are scored on true categories). Every detector is point in time:
a month is scored only from the user's earlier months. Outcomes come from FR-2's label contract
(`Truth.score_periods`), so ignored months (weekly spikes, unusual charges, warm-up) are ignored.

    uv run python experiments/fr8_spikes/feasibility.py data/synthetic/default.sqlite

Simplifications (each makes the numbers optimistic, as in FR-7's POC):
- Thresholds are chosen and measured on the same users.
- The population seasonal index pools all train users and all months: the scored user, the
  scored month and later months included. The round builds it as of each month, leaving the
  scored user out.
- The income elasticity is fitted once on all train users (it reads no labels).
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from smart_financial_coach.data.labels import INCOME, PERIOD_KEY, Truth, load_truth

TRAILING = 12  # months of history a detector looks back over
RATES = (
    0.02,
    0.03,
    0.035,
    0.05,
)  # flags per post-warm-up user-month, for the equal-volume comparison
TARGETS = (0.70, 0.80)
REPS = 500
SPEND_FLOOR = 1.3  # spend at least this times the usual level (FR-2's weak_lift, reused)
MIN_USUAL_COUNT = 2.0  # purchases in a usual month
SPIKE_CATEGORIES = ("Dining", "Groceries", "Shopping", "Transportation")  # where FR-1 plants
LOOKALIKE_LIFT = 1.5  # non-label months with spend at least this times usual
BASKET = (1.8, 3.0)  # simulated basket-size spikes: spend x U(BASKET), counts unchanged
BASKET_N = 400
OUT = Path(__file__).parent / "results"


def md(t: pd.DataFrame) -> str:
    """A markdown table without the optional `tabulate` dependency."""
    cols = [t.index.name or "", *t.columns]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    rows = [
        [str(i), *(fmt(v) for v in r)]
        for i, r in zip(t.index, t.itertuples(index=False), strict=True)
    ]
    out = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def train_only(truth: Truth) -> Truth:
    """The same truth, with every table cut to train users."""
    users = truth.users[truth.users["split"] == "train"]
    keep = set(users["user_id"])

    def cut(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["user_id"].isin(keep)].reset_index(drop=True)

    return replace(
        truth,
        transactions=cut(truth.transactions),
        periods=cut(truth.periods),
        expected=cut(truth.expected),
        users=users.reset_index(drop=True),
        preferences=cut(truth.preferences),
    )


def periods(truth: Truth) -> pd.DataFrame:
    """One row per (train user, spending category, month), every month of the calendar."""
    users = truth.users.loc[truth.users["split"] == "train", "user_id"]
    tx = truth.transactions[truth.transactions["user_id"].isin(users)]
    spend_tx = tx[tx["category"] != INCOME]
    months = sorted(tx["month"].unique())
    grid = (
        spend_tx[["user_id", "category"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"period_start": months}), how="cross")
    )
    g = spend_tx.groupby(["user_id", "category", "month"])
    agg = (
        pd.DataFrame(
            {
                "spend": -g["amount"].sum(),
                # model-visible count: outflows in the category (no process is visible to a model)
                "count": g["amount"].apply(lambda a: int((a < 0).sum())),
            }
        )
        .reset_index()
        .rename(columns={"month": "period_start"})
    )
    p = grid.merge(agg, on=PERIOD_KEY, how="left").fillna({"spend": 0.0, "count": 0})
    income = (
        tx[tx["category"] == INCOME]
        .groupby(["user_id", "month"])["amount"]
        .sum()
        .rename("income")
        .reset_index()
        .rename(columns={"month": "period_start"})
    )
    p = p.merge(income, on=["user_id", "period_start"], how="left").fillna({"income": 0.0})
    p["m"] = p["period_start"].map({m: i for i, m in enumerate(months)})
    p["moy"] = p["period_start"].str[5:7].astype(int)
    exp = truth.expected[truth.expected["granularity"] == "month"]
    p = p.merge(exp[[*PERIOD_KEY, "expected_count", "expected_spend"]], on=PERIOD_KEY, how="left")
    return p.sort_values(["user_id", "category", "m"]).reset_index(drop=True)


def trailing(s: pd.Series, groups: list[pd.Series], fn: str, window: int = TRAILING) -> pd.Series:
    """Rolling statistic over the previous `window` months only (shifted: point in time)."""
    return s.groupby(groups).transform(
        lambda x: getattr(x.shift(1).rolling(window, min_periods=3), fn)()
    )


def features(p: pd.DataFrame) -> pd.DataFrame:
    keys = [p["user_id"], p["category"]]
    p["mean_spend"] = trailing(p["spend"], keys, "mean")
    p["std_spend"] = trailing(p["spend"], keys, "std")
    p["all_mean_spend"] = p.groupby(keys)["spend"].transform(
        lambda x: x.shift(1).expanding(3).mean()
    )
    p["all_std_spend"] = p.groupby(keys)["spend"].transform(lambda x: x.shift(1).expanding(3).std())
    p["med_spend"] = trailing(p["spend"], keys, "median")
    ls = np.log1p(p["spend"].clip(lower=0))
    p["med_log"] = trailing(ls, keys, "median")
    dev = (ls - p["med_log"]).abs()
    p["mad_log"] = dev.groupby(keys).transform(
        lambda x: x.shift(1).rolling(TRAILING, min_periods=3).median()
    )
    p["mean_count"] = trailing(p["count"].astype(float), keys, "mean")
    p["var_count"] = trailing(p["count"].astype(float), keys, "var")
    # Seasonal index, population: per (category, month of year), the median over users of
    # count / their own trailing mean. Pools all train users (simplification, see docstring).
    ratio = p["count"] / p["mean_count"]
    pop = (
        p.assign(r=ratio)
        .dropna(subset=["r"])
        .groupby(["category", "moy"])["r"]
        .median()
        .rename("pop_season")
    )
    p = p.join(pop, on=["category", "moy"])
    # The user's own index: same month a year earlier over that month's trailing mean
    own = ratio.groupby(keys).shift(12)
    n_years = own.notna().astype(float)
    shrink = n_years / (n_years + 1.0)  # one prior year counts as much as the population
    p["own_season"] = own
    p["season"] = np.where(
        own.notna(), shrink * own.fillna(1) + (1 - shrink) * p["pop_season"], p["pop_season"]
    )
    # Income coupling: trailing two-month income over its trailing-12 mean (model-visible)
    inc = p.drop_duplicates(["user_id", "m"]).set_index(["user_id", "m"])["income"].sort_index()
    by_user = inc.groupby(level=0)
    two = by_user.transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    year = by_user.transform(lambda x: x.shift(1).rolling(TRAILING, min_periods=3).mean())
    p["income_ratio"] = pd.MultiIndex.from_frame(p[["user_id", "m"]]).map((two / year).to_dict())
    return p


def scores(p: pd.DataFrame, beta: float) -> dict[str, pd.Series]:
    s: dict[str, pd.Series] = {}
    s["baseline_mean_k_std"] = (p["spend"] - p["all_mean_spend"]) / p["all_std_spend"].replace(
        0, np.nan
    )
    s["robust_log_mad"] = (np.log1p(p["spend"].clip(lower=0)) - p["med_log"]) / (
        1.4826 * p["mad_log"]
    ).clip(lower=0.05)
    s["spend_ratio_seasonal"] = p["spend"] / (p["mean_spend"] * p["season"]).clip(lower=1)
    lam = p["mean_count"].clip(lower=0.3)
    s["count_poisson"] = -poisson.logsf(p["count"] - 1, lam)
    lam_s = (p["mean_count"] * p["season"]).clip(lower=0.3)
    s["count_poisson_seasonal"] = -poisson.logsf(p["count"] - 1, lam_s)
    inc = p["income_ratio"].clip(0.5, 1.6).fillna(1.0) ** beta
    lam_si = (lam_s * inc).clip(lower=0.3)
    s["count_poisson_seasonal_income"] = -poisson.logsf(p["count"] - 1, lam_si)
    # The same, flagged only where spend is materially up (a fixed product rule: a "spike" whose
    # spend isn't 1.3x the usual level would contradict its own reason) and the usual level is
    # at least 2 purchases a month
    usual = (p["mean_spend"] * p["season"]).clip(lower=1)
    floor = (p["spend"] >= SPEND_FLOOR * usual) & (p["mean_count"] >= MIN_USUAL_COUNT)
    s["count_poisson_seasonal_income_floor"] = np.where(
        floor, s["count_poisson_seasonal_income"], np.nan
    )
    # Ablations of the two rules, and (diagnostic only) the categories FR-1 plants in
    spend_up = p["spend"] >= SPEND_FLOOR * usual
    volume = p["mean_count"] >= MIN_USUAL_COUNT
    planted = p["category"].isin(SPIKE_CATEGORIES)
    csi = s["count_poisson_seasonal_income"]
    s["ablation: spend floor only"] = np.where(spend_up, csi, np.nan)
    s["ablation: min usual count only"] = np.where(volume, csi, np.nan)
    s["diagnostic: planted categories only"] = np.where(planted, csi, np.nan)
    # Negative binomial: over-dispersion from the trailing count variance, floored at Poisson
    var = np.maximum(p["var_count"].fillna(lam_si), lam_si * 1.05)
    n = lam_si**2 / (var - lam_si)
    s["count_negbin_seasonal_income"] = -nbinom.logsf(p["count"] - 1, n, n / (n + lam_si))
    s["oracle_true_expected_count"] = -poisson.logsf(
        p["count"] - 1, p["expected_count"].clip(lower=0.01)
    )
    return {
        k: pd.Series(np.asarray(v, dtype=float), index=p.index).replace([np.inf, -np.inf], np.nan)
        for k, v in s.items()
    }


def fit_beta(p: pd.DataFrame) -> float:
    """Pooled income elasticity of category counts, persona-free, from model-visible data."""
    d = p[(p["mean_count"] >= 3) & p["income_ratio"].between(0.5, 1.6)]
    y = np.log((d["count"] + 0.5) / (d["mean_count"] * d["season"]))
    x = np.log(d["income_ratio"])
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def outcomes(truth: Truth, p: pd.DataFrame) -> pd.DataFrame:
    """tp / fp / ignored for every scored period if flagged; `is_spike` for labels."""
    o = truth.score_periods(p[PERIOD_KEY])
    o = o[o["outcome"] != "fn"][[*PERIOD_KEY, "outcome", "tier"]]
    return p.merge(o, on=PERIOD_KEY, how="left")


def curve(score: pd.Series, outcome: pd.Series) -> pd.DataFrame:
    keep = (outcome != "ignored") & score.notna()
    sc, oc = score[keep], outcome[keep]
    order = np.argsort(-sc.to_numpy(), kind="mergesort")
    tp = np.cumsum((oc.to_numpy() == "tp")[order])
    n = np.arange(1, len(tp) + 1)
    return pd.DataFrame({"n": n, "tp": tp, "precision": tp / n, "score": sc.to_numpy()[order]})


def evaluate(name: str, score: pd.Series, o: pd.DataFrame, n_labels: int, user_months: int) -> dict:
    c = curve(score, o["outcome"])
    row: dict[str, object] = {"detector": name}
    for t in TARGETS:
        ok = c[c["precision"] >= t]
        best = ok.iloc[ok["tp"].to_numpy().argmax()] if len(ok) else None
        row[f"recall@p{t:.2f}"] = (best["tp"] / n_labels) if best is not None else 0.0
        row[f"flags@p{t:.2f}"] = int(best["n"]) if best is not None else 0
    for r in RATES:
        k = round(r * user_months)
        hit = c.iloc[min(k, len(c)) - 1]
        row[f"recall@{r}"] = hit["tp"] / n_labels
        row[f"precision@{r}"] = hit["precision"]
    k = round(0.03 * user_months)
    keep = (o["outcome"] != "ignored") & score.notna()
    ranked = o[keep].assign(score=score[keep]).nlargest(k, "score")
    clear_total = int(((o["outcome"] == "tp") & (o["tier"] == "clear")).sum())
    row["recall_clear@0.03"] = (
        (ranked["outcome"] == "tp") & (ranked["tier"] == "clear")
    ).sum() / clear_total
    return row


def bootstrap(
    score: pd.Series, o: pd.DataFrame, rate: float, user_months_by_user: pd.Series, seed: int = 0
):
    """User bootstrap of precision and recall at a fixed flag rate."""
    rng = np.random.default_rng(seed)
    users = user_months_by_user.index.to_numpy()
    keep = (o["outcome"] != "ignored") & score.notna()
    d = o.loc[keep, ["user_id", "outcome"]].assign(score=score[keep])
    labels_by_user = o[o["outcome"] == "tp"].groupby("user_id").size()
    out = []
    for _ in range(REPS):
        pick = pd.Series(rng.choice(users, len(users))).value_counts()
        w = d["user_id"].map(pick).fillna(0).to_numpy()
        order = np.argsort(-d["score"].to_numpy(), kind="mergesort")
        cw = np.cumsum(w[order])
        k = rate * float((user_months_by_user * pick.reindex(users).fillna(0).to_numpy()).sum())
        at = int(np.searchsorted(cw, k))
        tp = float((w[order][: at + 1] * (d["outcome"].to_numpy()[order][: at + 1] == "tp")).sum())
        nl = float(
            (
                labels_by_user.reindex(users).fillna(0) * pick.reindex(users).fillna(0).to_numpy()
            ).sum()
        )
        out.append((tp / max(cw[min(at, len(cw) - 1)], 1), tp / max(nl, 1)))
    a = np.array(out)
    return np.percentile(a, [5, 95], axis=0)


def ranked(score: pd.Series, o: pd.DataFrame, rate: float, user_months: int) -> pd.DataFrame:
    """The top `rate` x user-months scored, non-ignored periods."""
    keep = (o["outcome"] != "ignored") & score.notna()
    return o[keep].assign(score=score[keep]).nlargest(round(rate * user_months), "score")


def counts(s: pd.Series) -> str:
    return ", ".join(f"{k} {v}" for k, v in s.value_counts().items())


def spend_detectors(s: dict[str, pd.Series], scored: pd.DataFrame) -> list[str]:
    """What tops the spend-based scores, overall and on frequent categories only."""
    lines = ["## What tops the spend-based scores (top 240, non-ignored)", ""]
    keep = scored["outcome"] != "ignored"
    for k in ("baseline_mean_k_std", "robust_log_mad", "spend_ratio_seasonal"):
        v = s[k]
        top = scored[keep & v.notna()].assign(score=v).nlargest(240, "score")
        freq = keep & v.notna() & (scored["mean_count"] >= 4)
        top_f = scored[freq].assign(score=v).nlargest(240, "score")
        top3 = counts(top["category"]).split(", ")[:3]
        lines.append(
            f"- {k}: {', '.join(top3)}; true spikes {int((top['outcome'] == 'tp').sum())}. "
            f"With at least 4 purchases in a usual month: true spikes "
            f"{int((top_f['outcome'] == 'tp').sum())} of 240"
        )
    return [*lines, ""]


def ablations(table: pd.DataFrame) -> list[str]:
    """Each piece's change in recall (points) at every operating point."""
    steps = [
        ("seasonal index", "count_poisson", "count_poisson_seasonal"),
        ("income coupling", "count_poisson_seasonal", "count_poisson_seasonal_income"),
        ("spend floor only", "count_poisson_seasonal_income", "ablation: spend floor only"),
        ("min usual only", "count_poisson_seasonal_income", "ablation: min usual count only"),
        ("both rules", "count_poisson_seasonal_income", "count_poisson_seasonal_income_floor"),
    ]
    cols = ["recall@p0.70", "recall@p0.80", *(f"recall@{r}" for r in RATES)]
    rows = {
        name: [100 * (float(table.loc[b, c]) - float(table.loc[a, c])) for c in cols]
        for name, a, b in steps
    }
    out = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    out.index.name = "change in recall (points)"
    return ["## Ablations", "", md(out), ""]


def lookalikes(
    truth: Truth, scored: pd.DataFrame, leader: pd.Series, user_months: int
) -> list[str]:
    """Non-label months in the planted categories whose spend is well up, and simulated
    basket-size spikes: spend up, counts unchanged. The leader can't see either."""
    cutoff = ranked(leader, scored, 0.035, user_months)["score"].min()
    usual = scored["mean_spend"]
    cand = scored[
        scored["category"].isin(SPIKE_CATEGORIES)
        & (scored["outcome"] == "fp")
        & (scored["spend"] >= LOOKALIKE_LIFT * usual)
        & usual.notna()
    ]
    tx = truth.transactions[(truth.transactions["amount"] < 0)].rename(
        columns={"month": "period_start"}
    )
    tx = tx.merge(cand[PERIOD_KEY], on=PERIOD_KEY)
    tx["out"] = -tx["amount"]
    top3 = tx.sort_values("out", ascending=False).groupby(PERIOD_KEY).head(3)
    top3 = top3.groupby(PERIOD_KEY)["out"].sum().rename("top3").reset_index()
    c = cand.merge(top3, on=PERIOD_KEY, how="left").fillna({"top3": 0.0})
    covered = c["top3"] >= c["spend"] - c["mean_spend"]
    flat = c["count"] < 1.3 * c["mean_count"]
    flat_adj = c["count"] < 1.3 * c["mean_count"] * c["season"]
    n_labels = int((scored["outcome"] == "tp").sum())
    flagged = (leader.loc[c.index] >= cutoff).sum()
    # Simulated basket-size spikes on a seeded sample of non-label months
    rng = np.random.default_rng(0)
    pool = scored[
        scored["category"].isin(SPIKE_CATEGORIES)
        & (scored["outcome"] == "fp")
        & (scored["mean_count"] >= MIN_USUAL_COUNT)
    ]
    sample = pool.loc[rng.choice(pool.index, BASKET_N, replace=False)].copy()
    sample["spend"] = sample["spend"] * rng.uniform(*BASKET, BASKET_N)
    # The leader's score on the changed months: the count tail is unchanged, the floor re-checked
    inc = sample["income_ratio"].clip(0.5, 1.6).fillna(1.0) ** FIT_BETA[0]
    lam = ((sample["mean_count"] * sample["season"]).clip(lower=0.3) * inc).clip(lower=0.3)
    tail = -poisson.logsf(sample["count"] - 1, lam)
    usual_s = (sample["mean_spend"] * sample["season"]).clip(lower=1)
    hit = (tail >= cutoff) & (sample["spend"] >= SPEND_FLOOR * usual_s)
    return [
        "## Look-alikes and basket-size spikes (planted categories)",
        "",
        f"- non-label months with spend >= {LOOKALIKE_LIFT}x the user's 12-month average: "
        f"{len(c)} ({len(c) / n_labels:.1f} per label)",
        f"- the 3 largest charges cover the whole excess: {covered.mean():.3f}",
        f"- count up less than 1.3x: {flat.mean():.3f} against the plain average, "
        f"{flat_adj.mean():.3f} against the seasonal one",
        f"- flagged by the leader at its 0.035 cutoff: {int(flagged)}",
        f"- simulated basket-size spikes (spend x U{BASKET}, counts unchanged, "
        f"{BASKET_N} non-label months): recall at the 0.035 cutoff {hit.mean():.3f}",
        "",
    ]


FIT_BETA = [0.0]  # set in main, read by `lookalikes`


def main(path: str) -> None:
    truth = train_only(load_truth(path))
    p = features(periods(truth))
    o = outcomes(truth, p)
    scored = o[o["period_start"] >= truth.warmup_end_month].reset_index(drop=True)
    labels = truth.spikes("month")
    train_users = set(truth.users.loc[truth.users["split"] == "train", "user_id"])
    labels = labels[labels["user_id"].isin(train_users)]
    n_labels = len(labels)
    um = scored.drop_duplicates(["user_id", "period_start"]).groupby("user_id").size()
    user_months = int(um.sum())
    beta = fit_beta(scored)
    FIT_BETA[0] = beta
    s = scores(scored, beta)
    lines = [
        "# FR-8 feasibility (train users only)",
        "",
        f"- Dataset `{Path(path).name}`; train users: {len(train_users)}; "
        f"post-warm-up user-months: {user_months}",
        f"- Scored periods: {len(scored)} (ignored: {(scored['outcome'] == 'ignored').sum()}); "
        f"monthly spike labels: {n_labels} (clear {int((labels['tier'] == 'clear').sum())})",
        f"- Labels per user-month: {n_labels / user_months:.4f}",
        f"- Pooled income elasticity of counts (fitted, persona-free): {beta:.3f}",
        "",
    ]
    rows = [evaluate(k, v, scored, n_labels, user_months) for k, v in s.items()]
    table = pd.DataFrame(rows).set_index("detector")
    lines += ["## Detectors", "", md(table), ""]
    lines += ablations(table)
    lines += spend_detectors(s, scored)
    lines += ["## User bootstrap at 0.03 flags per user-month (5-95%: precision, recall)", ""]
    for k in (
        "baseline_mean_k_std",
        "count_poisson",
        "count_poisson_seasonal_income_floor",
        "count_negbin_seasonal_income",
    ):
        lo, hi = bootstrap(s[k], scored, 0.03, um)
        lines.append(f"- {k}: precision {lo[0]:.3f}-{hi[0]:.3f}, recall {lo[1]:.3f}-{hi[1]:.3f}")
    lines.append("")
    # Where the best non-oracle detector's false positives come from, at 0.03
    best = "count_poisson_seasonal_income_floor"
    keep = (scored["outcome"] != "ignored") & s[best].notna()
    top = scored[keep].assign(score=s[best][keep]).nlargest(int(0.03 * user_months), "score")
    fp = top[top["outcome"] == "fp"]
    persona = truth.users.set_index("user_id")["persona"]
    lines += [
        f"## {best} at 0.03: false positives",
        "",
        f"- flags {len(top)}, false positives {len(fp)}; in categories FR-1 never plants in: "
        f"{int((~fp['category'].isin(SPIKE_CATEGORIES)).sum())}",
        "- by category: " + ", ".join(f"{k} {v}" for k, v in fp["category"].value_counts().items()),
        "- by month of year: "
        + ", ".join(f"{k} {v}" for k, v in fp["moy"].value_counts().sort_index().items()),
        "- by persona (analysis only): "
        + ", ".join(f"{k} {v}" for k, v in fp["user_id"].map(persona).value_counts().items()),
        "",
    ]
    tp = top[top["outcome"] == "tp"]
    missed = scored[
        (scored["outcome"] == "tp")
        & ~scored.set_index(PERIOD_KEY).index.isin(top.set_index(PERIOD_KEY).index)
    ]
    oracle_top = ranked(s["oracle_true_expected_count"], scored, 0.03, user_months)
    oracle_missed = ~missed.set_index(PERIOD_KEY).index.isin(oracle_top.set_index(PERIOD_KEY).index)
    lines += [
        f"- of the leader's misses, the oracle misses at the same rate: {int(oracle_missed.sum())}",
        "- true positives by tier: "
        + ", ".join(f"{k} {v}" for k, v in tp["tier"].value_counts().items()),
        "- missed by tier: "
        + ", ".join(f"{k} {v}" for k, v in missed["tier"].value_counts().items()),
        "- missed by category: "
        + ", ".join(f"{k} {v}" for k, v in missed["category"].value_counts().items()),
        "",
    ]
    # Size of the deviation: estimated expected spend vs the generator's, on flagged true spikes
    est = (tp["mean_spend"] * tp["season"]).to_numpy()
    true_exp = tp["expected_spend"].to_numpy()
    own = tp["mean_spend"].to_numpy()  # the user's own average of the previous 12 months
    lines += ["## Size of the deviation (flagged true spikes, against the generator's)", ""]
    for name, e in (("trailing mean x season", est), ("own 12-month average", own)):
        err = np.abs(e - true_exp) / true_exp
        lines.append(
            f"- {name}: median absolute error of the usual level {np.median(err):.3f}, "
            f"90th percentile {np.percentile(err, 90):.3f}; median excess "
            f"${np.median(tp['spend'] - e):,.0f} (true ${np.median(tp['spend'] - true_exp):,.0f})"
        )
    plain = s["count_poisson_seasonal_income"].where(
        (scored["spend"] >= SPEND_FLOOR * scored["mean_spend"].clip(lower=1))
        & (scored["mean_count"] >= MIN_USUAL_COUNT)
    )
    plain_top = ranked(plain, scored, 0.035, user_months)
    lines.append(
        "- the leader with the floor on the plain 12-month average: precision at 0.035 "
        f"{(plain_top['outcome'] == 'tp').mean():.3f}"
    )
    lines.append(
        f"- purchases in a flagged spike month: median {tp['count'].median():.0f} against "
        f"{tp['mean_count'].median():.1f} in the user's average month"
    )
    lines.append("")
    # Driving transactions: top 5 by amount, on all labeled spikes
    cov = truth.score_drivers(truth.baseline_drivers())
    lines += [
        "## Driving transactions (labeled spikes, train users)",
        "",
        f"- top 5 by amount: mean excess coverage {cov['coverage'].mean():.3f}, "
        f"median {cov['coverage'].median():.3f}, "
        f"share at 1.0 {(cov['coverage'] >= 0.999).mean():.3f}",
        "",
    ]
    lines += lookalikes(truth, scored, s[best], user_months)
    OUT.mkdir(exist_ok=True)
    (OUT / "feasibility.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/default.sqlite")
