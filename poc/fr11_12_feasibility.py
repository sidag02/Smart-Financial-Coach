"""FR-11/12 feasibility POC (train users only). Part A: monthly net-savings forecasts in a
rolling-origin backtest. Part B: P(goal met) for goals whose outcome is known."""

import sqlite3
import sys

import numpy as np
import pandas as pd

DB = sys.argv[1]
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "train"
rng = np.random.default_rng(0)

con = sqlite3.connect(DB)
users = pd.read_sql("select user_id, persona, split from users", con)
users = users[users["split"] == SPLIT].set_index("user_id")
tx = pd.read_sql("select user_id, ts, amount from transactions", con)
tx = tx[tx["user_id"].isin(users.index)]
goals = pd.read_sql("select * from goals", con).merge(
    pd.read_sql("select * from truth_goals", con), on="goal_id"
)
goals = goals[goals["user_id"].isin(users.index)]
meta = dict(con.execute("select key, value from meta").fetchall())

start = pd.Period(meta["calendar_start"][:7], "M")
tx["m"] = (pd.PeriodIndex(tx["ts"].str[:7], freq="M") - start).map(lambda x: x.n)
N = int(tx["m"].max()) + 1
net = (
    tx.groupby(["user_id", "m"])["amount"].sum().unstack(fill_value=0.0).reindex(columns=range(N), fill_value=0.0)
)
net = net.loc[users.index]
moy = np.array([(start + i).month for i in range(N)])
persona = users["persona"]
print(f"{SPLIT}: {len(users)} users, {N} months ({start}..{start + N - 1}); goals {len(goals)}, known {goals['met'].notna().sum()}")


# ---------------------------------------------------------------- Part A: forecasts


def fc_naive(y, k, h):
    return np.full(h, y[k - 1])


def fc_mean(y, k, h):
    return np.full(h, y[k - 12 : k].mean())


def fc_snaive(y, k, h):
    return np.array([y[k + i - 12 * (1 + i // 12)] for i in range(h)])


def seasonal_index(y, k, months):
    """Per month-of-year deviation from the trailing-12 level, averaged over the history."""
    dev = {m: [] for m in range(1, 13)}
    for t in range(12, k):
        dev[months[t]].append(y[t] - y[t - 12 : t].mean())
    return {m: (np.mean(v) if v else 0.0, len(v)) for m, v in dev.items()}


def make_profile(shrink):
    def fc(y, k, h, pool=None):
        level = y[max(0, k - LEVEL_W) : k].mean()
        idx = seasonal_index(y, k, moy)
        out = []
        for i in range(h):
            m = moy[k + i]
            own, n = idx[m]
            prior = pool[m] if pool is not None else 0.0
            w = n / (n + shrink) if shrink else 1.0
            out.append(level + w * own + (1 - w) * prior)
        return np.array(out)

    return fc


def pooled_index(k, scale_by_level=True):
    """Persona-level seasonal profile as a share of level, from all train users' history < k."""
    prof = {}
    for p in persona.unique():
        ids = persona.index[persona == p]
        rows = []
        for u in ids:
            y = net.loc[u].to_numpy()
            lvl = np.abs(y[max(0, k - 12) : k].mean()) + 1e-9
            idx = seasonal_index(y, k, moy)
            rows.append({m: idx[m][0] / lvl for m in idx})
        prof[p] = pd.DataFrame(rows).median().to_dict()
    return prof


def fc_window(w):
    return lambda y, k, h: np.full(h, y[max(0, k - w) : k].mean())


def fc_median(w):
    return lambda y, k, h: np.full(h, np.median(y[max(0, k - w) : k]))


def fc_trim(w, q):
    def fc(y, k, h):
        x = np.sort(y[max(0, k - w) : k])
        n = int(len(x) * q)
        return np.full(h, x[n : len(x) - n].mean() if len(x) - 2 * n > 0 else x.mean())

    return fc


def fc_hindsight(y, k, h):  # level from the whole series, future included: an upper bound
    return np.full(h, y.mean())


LEVEL_W = 12
MODELS = {
    "mean24": fc_window(24),
    "mean_all": fc_window(99),
    "median12": fc_median(12),
    "median24": fc_median(24),
    "trim24_10": fc_trim(24, 0.1),
    "hindsight_level": fc_hindsight,
    "naive": fc_naive,
    "mean12": fc_mean,
    "snaive": fc_snaive,
    "profile": make_profile(0),
    "profile_shrink2": make_profile(2),
}
H = (3, 6, 12)
origins = [k for k in range(13, N) if k + 3 <= N]
pools = {k: pooled_index(k) for k in origins}

errs = []
for u in users.index:
    y = net.loc[u].to_numpy()
    p = persona[u]
    for k in origins:
        hmax = min(12, N - k)
        actual = y[k : k + hmax]
        lvl = np.abs(y[k - 12 : k].mean())
        for name, fc in MODELS.items():
            f = fc(y, k, hmax)
            for h in H:
                if h <= hmax:
                    errs.append((p, name, h, k, f[:h].sum() - actual[:h].sum(), lvl))
        # persona-pooled seasonal prior, shrunk
        pool = {m: pools[k][p][m] * lvl for m in range(1, 13)}
        for shrink in (1, 2, 4):
            f = make_profile(shrink)(y, k, hmax, pool)
            for h in H:
                if h <= hmax:
                    errs.append((p, f"profile_pool{shrink}", h, k, f[:h].sum() - actual[:h].sum(), lvl))

E = pd.DataFrame(errs, columns=["persona", "model", "h", "k", "err", "level"])
rmse = E.groupby(["h", "model"])["err"].apply(lambda e: float(np.sqrt((e**2).mean())))
print("\nRMSE of cumulative net over h months, all train users ($):")
tab = rmse.unstack("h").round(0)
tab["vs_mean12_h6"] = (tab[6] / tab.loc["mean12", 6]).round(3)
tab["vs_snaive_h6"] = (tab[6] / tab.loc["snaive", 6]).round(3)
print(tab.sort_values(6).to_string())
print("\nRMSE ratio to mean12, by persona (h=3, 6, 12):")
r = E.groupby(["persona", "model", "h"])["err"].apply(lambda e: float(np.sqrt((e**2).mean()))).unstack("h")
for p in r.index.get_level_values(0).unique():
    base = r.loc[(p, "mean12")]
    print(f"  {p}:")
    print((r.loc[p] / base).round(3).sort_values(6).to_string().replace("\n", "\n    "))

# ---------------------------------------------------------------- Part B: goals


def saved_path(y, share, first, last):
    s = 0.0
    for m in range(first, last + 1):
        s = max(0.0, s + share * y[m])
    return s


def month_of(d):
    return (pd.Period(d[:7], "M") - start).n


def infer_share(y, created, as_of, balance):
    if balance <= 0:
        return 0.0
    lo, hi = 0.0, 3.0
    if saved_path(y, hi, created, as_of) < balance:
        return hi
    for _ in range(50):
        mid = (lo + hi) / 2
        if saved_path(y, mid, created, as_of) < balance:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


known = goals[goals["met"].notna()].copy()
POINT = next((a.split('=')[1] for a in sys.argv if a.startswith('--point=')), 'mean')
WIN = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--win=')), '24'))
SHARE_PRIOR = None
pri = []
for g in known.itertuples():
    if g.current_balance > 0:
        yy = net.loc[g.user_id].to_numpy()
        pri.append(infer_share(yy, month_of(g.created_date), month_of(g.as_of_date), g.current_balance))
SHARE_PRIOR = float(np.median(pri)) if "--prior" in sys.argv else None
rows = []
for g in known.itertuples():
    y = net.loc[g.user_id].to_numpy()
    c, a, t = month_of(g.created_date), month_of(g.as_of_date), month_of(g.target_date)
    left = t - a
    elapsed = a - c + 1
    bal = g.current_balance
    # naive linear pace
    pace = bal / elapsed
    naive_proj = bal + pace * left
    # share model
    share = infer_share(y, c, a, bal)
    share_raw = share
    if bal <= 0 and SHARE_PRIOR is not None:
        share = SHARE_PRIOR
    k = a + 1
    hist = y[:k]
    # point path: seasonal profile with persona pool (shrink 2); residual bootstrap by month
    lvl = np.abs(hist[k - 12 : k].mean())
    pool = {m: pools.get(k, pools[max(pools)])[persona[g.user_id]][m] * lvl for m in range(1, 13)} if k in pools else None
    lvl_w = hist[max(0, k - WIN) : k]
    if POINT == "profile" and k >= 13:
        point = make_profile(2)(np.concatenate([hist, np.zeros(left)]), k, left, pool)
    else:
        point = np.full(left, lvl_w.mean())
    resid = lvl_w - lvl_w.mean()
    sims = []
    for _ in range(400):
        path = point + rng.choice(resid, size=left, replace=True)
        s = bal
        for v in path:
            s = max(0.0, s + share * v)
        sims.append(s)
    sims = np.array(sims)
    p_met = float((sims >= g.target_amount).mean())
    point_final = saved_path(np.concatenate([hist, point]), share, k, k + left - 1) if left else bal
    point_final = bal
    for v in point:
        point_final = max(0.0, point_final + share * v)
    # mean12 share model (no seasonality) for comparison
    s2 = bal
    for v in np.full(left, hist[k - 12 : k].mean()):
        s2 = max(0.0, s2 + share * v)
    actual_final = saved_path(y, share, a + 1, t) if False else None
    fut = y[a + 1 : t + 1]
    s3 = bal
    for v in fut:
        s3 = max(0.0, s3 + share * v)
    rows.append(
        dict(
            oracle_call=int(s3 >= g.target_amount), actual_final=s3, share_raw=share_raw,
            persona=persona[g.user_id], cls=g.outcome_class, met=int(g.met), left=left,
            naive_call=int(naive_proj >= g.target_amount), share=share,
            p_sim=p_met, flat_call=int(s2 >= g.target_amount),
            point_call=int(point_final >= g.target_amount),
            lo=np.quantile(sims, 0.1), hi=np.quantile(sims, 0.9), target=g.target_amount,
        )
    )
G = pd.DataFrame(rows)


def brier(p, y):
    return float(((p - y) ** 2).mean())


print(f"\nGoals with known outcome ({SPLIT}): {len(G)}  met rate {G['met'].mean():.2f}")
print(G.groupby("cls")["met"].agg(["count", "mean"]).round(2).to_string())
for col in ("oracle_call", "naive_call", "flat_call", "point_call", "p_sim"):
    acc = float(((G[col] >= 0.5).astype(int) == G["met"]).mean())
    print(f"  {col:11s} Brier {brier(G[col], G['met']):.3f}  accuracy {acc:.3f}")
print(f"  climatology (0.5) Brier {brier(np.full(len(G), 0.5), G['met']):.3f}")
inside = ((G["actual_final"] >= G["lo"]) & (G["actual_final"] <= G["hi"])).mean()
print(f"  80% range covers the actual final balance (inferred share): {inside:.2f}")
print("\nBrier by class:")
print(G.groupby("cls").apply(lambda d: pd.Series({c: brier(d[c], d["met"]) for c in ("naive_call", "p_sim")}), include_groups=False).round(3).to_string())
print("\nBrier by persona:")
print(G.groupby("persona").apply(lambda d: pd.Series({c: brier(d[c], d["met"]) for c in ("naive_call", "p_sim")}), include_groups=False).round(3).to_string())
G["bin"] = pd.cut(G["p_sim"], [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
print("\nReliability of p_sim:")
print(G.groupby("bin", observed=True)["met"].agg(["count", "mean"]).round(2).to_string())
print(f"\nShare inferred: median {G['share'].median():.2f}, 10-90% {G['share'].quantile(.1):.2f}-{G['share'].quantile(.9):.2f}")

print("\nWhere p_sim <= 0.1 but met:")
bad = G[(G["p_sim"] <= 0.1) & (G["met"] == 1)]
print(bad[["persona", "cls", "left", "share", "target", "lo", "hi", "oracle_call"]].round(2).head(15).to_string())
print("months left:", G["left"].describe().round(1).to_dict())

if "--debug" in sys.argv:
    g = known.iloc[7]
    y = net.loc[g.user_id].to_numpy()
    c, a, t = month_of(g.created_date), month_of(g.as_of_date), month_of(g.target_date)
    print("\nDEBUG", g.goal_id, g.outcome_class, "created", c, "as_of", a, "target", t, "bal", g.current_balance, "target", g.target_amount)
    print("net history (k-12..a):", np.round(y[a - 11 : a + 1]).astype(int).tolist())
    print("net future (a+1..t):", np.round(y[a + 1 : t + 1]).astype(int).tolist())
    k = a + 1
    print("point:", np.round(make_profile(2)(np.concatenate([y[:k], np.zeros(t - a)]), k, t - a, None)).astype(int).tolist())
    print("mean12:", round(y[k - 12 : k].mean()), "mean24:", round(y[max(0, k - 24) : k].mean()))

if "--se" in sys.argv:
    sq = (G["p_sim"] - G["met"]) ** 2
    print(f"\nBrier p_sim {sq.mean():.3f}, standard error {sq.std() / np.sqrt(len(sq)):.3f} (n={len(sq)})")
    boot = [float(sq.sample(len(sq), replace=True, random_state=i).mean()) for i in range(1000)]
    print(f"bootstrap 90% interval {np.quantile(boot, .05):.3f}-{np.quantile(boot, .95):.3f}")
    print(G.groupby("persona").size().to_dict())
