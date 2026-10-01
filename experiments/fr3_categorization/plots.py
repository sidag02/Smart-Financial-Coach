# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pillow",
# ]
# ///
"""Charts, an HTML report and a screenshot for the FR-3 feasibility results.

Reads `results/feasibility.json` (written by `feasibility.py`); never reruns the models.

    uv run experiments/fr3_categorization/plots.py

Writes `results/figures/*.png`, `results/report.html` and, when Google Chrome is installed,
`results/screenshots/report.png`.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = Path(__file__).parent
RESULTS = HERE / "results"

# Reference palette (dataviz skill): light surface, ink tokens, first three categorical slots,
# which validate all-pairs for colour-vision deficiency. Aqua sits below 3:1 contrast, so
# every chart also carries a legend or direct labels.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MODELS = {
    "ngrams": ("Character n-grams", "#2a78d6"),
    "embeddings": ("Sentence embeddings", "#eb6834"),
    "both": ("Both", "#1baf7a"),
}
BLUES = LinearSegmentedColormap.from_list(
    "blues",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
TARGETS = {"known": 0.90, "unseen": 0.80}
TEST_SETS = {
    "known": "Known merchants",
    "test_users": "All test users",
    "unseen": "Unseen merchants",
}
DPI = 200

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def style(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)


def title(fig: plt.Figure, text: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, text, fontsize=13, weight="bold", va="top", color=INK)
    fig.text(0.01, 0.985 - 0.33 / fig.get_figheight(), subtitle, va="top", color=INK_2)


def legend(fig: plt.Figure, y: float) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS.values()]
    fig.legend(
        handles,
        [name for name, _ in MODELS.values()],
        loc="upper left",
        bbox_to_anchor=(0.01, y),
        ncol=3,
        frameon=False,
        handlelength=1,
        handleheight=1,
        labelcolor=INK_2,
    )


def save(fig: plt.Figure, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def headline(r: dict[str, Any], out: Path) -> Path:
    exps = r["experiments"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.7, bottom=0.12, wspace=0.12)
    title(
        fig,
        "Macro F1 by test set",
        "12 spending categories. Unseen whiskers: 95% merchant-bootstrap interval. "
        "Vertical line: PRD target.",
    )
    kinds = list(MODELS)
    y = np.arange(len(kinds))[::-1]
    for ax, (key, label) in zip(axes, TEST_SETS.items(), strict=True):
        style(ax)
        for yi, kind in zip(y, kinds, strict=True):
            res = exps[kind][key]
            value = res["macro_f1_spending"]
            ax.barh(yi, value, height=0.42, color=MODELS[kind][1])
            if key == "unseen":
                boot = res["merchant_bootstrap"]
                ax.plot([boot["low"], boot["high"]], [yi, yi], color=INK, linewidth=1.2)
                for x in (boot["low"], boot["high"]):
                    ax.plot([x, x], [yi - 0.12, yi + 0.12], color=INK, linewidth=1.2)
                text_x = boot["high"] + 0.09
            else:
                text_x = value + 0.02
            ax.text(text_x, yi, f"{value:.3f}", va="center", color=INK, fontsize=9)
        if key in TARGETS:
            ax.axvline(TARGETS[key], color=INK_2, linewidth=1)
            ax.text(
                TARGETS[key] - 0.015,
                -0.62,
                f"target {TARGETS[key]:.2f}",
                ha="right",
                va="center",
                color=INK_2,
                fontsize=8,
            )
        ax.set_ylim(-0.85, len(kinds) - 0.5)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{label}  ({exps['both'][key]['n']:,} txns)", pad=14)
    axes[0].set_yticks(y, [MODELS[k][0] for k in kinds])
    return save(fig, out, "headline")


def bootstrap(r: dict[str, Any], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.7, bottom=0.15)
    title(
        fig,
        "Unseen-merchant macro F1: merchant-bootstrap distribution",
        "1,000 resamples of holdout merchants within each category. "
        "No resample of any model reaches the 0.80 target.",
    )
    legend(fig, 0.84)
    style(ax, "y")
    bins = np.linspace(0.35, 0.85, 51)
    for kind, (_, color) in MODELS.items():
        samples = np.array(r["experiments"][kind]["unseen"]["merchant_bootstrap"]["samples"])
        counts, edges = np.histogram(samples, bins=bins)
        ax.stairs(counts, edges, color=color, linewidth=2)
        ax.stairs(counts, edges, color=color, alpha=0.1, fill=True)
    ax.axvline(TARGETS["unseen"], color=INK_2, linewidth=1)
    ax.text(
        TARGETS["unseen"] - 0.005,
        ax.get_ylim()[1] * 0.92,
        "target 0.80 ",
        ha="right",
        color=INK_2,
        fontsize=8,
    )
    ax.set_xlabel("Macro F1 (12 spending categories)")
    ax.set_ylabel("Resamples")
    return save(fig, out, "unseen_bootstrap")


def heat(ax: plt.Axes, values: np.ndarray, rows: list[str], cols: list[str], fmt: str) -> None:
    ax.imshow(values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v) or (fmt == "share" and v < 0.005):
                continue
            text = f"{v:.2f}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK
            )


def per_class(r: dict[str, Any], out: Path) -> Path:
    labels = r["setup"]["spending_labels"]
    exps = r["experiments"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.78, bottom=0.04, wspace=0.08)
    title(fig, "Per-class F1", "Known merchants are near-perfect; unseen merchants vary by class.")
    for ax, key in zip(axes, ("known", "unseen"), strict=True):
        values = np.array(
            [
                [exps[k][key]["per_class"].get(c, {}).get("f1", np.nan) for k in MODELS]
                for c in labels
            ]
        )
        heat(ax, values, labels, ["N-grams", "Embeddings", "Both"], "f1")
        ax.set_title(TEST_SETS[key], pad=24)
        ax.xaxis.tick_top()
    return save(fig, out, "per_class_f1")


def by_scope(r: dict[str, Any], out: Path) -> Path:
    exps = r["experiments"]
    frames = {k: pd.DataFrame(exps[k]["unseen"]["by_merchant"]) for k in MODELS}
    scopes = [
        ("national", "National (real chains)"),
        ("local", "Local (fictional)"),
        ("online", "Online services"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.7, bottom=0.14)
    title(
        fig,
        "Unseen-merchant accuracy by merchant type",
        "Transaction-weighted. Real chains are hardest; embeddings carry outside knowledge.",
    )
    legend(fig, 0.84)
    style(ax)
    height = 0.24
    for s_idx, (scope, _label) in enumerate(scopes):
        base = len(scopes) - 1 - s_idx
        for m_idx, (kind, (_, color)) in enumerate(MODELS.items()):
            d = frames[kind][frames[kind]["scope"] == scope]
            acc = float((d["accuracy"] * d["transactions"]).sum() / d["transactions"].sum())
            yi = base + (1 - m_idx) * (height + 0.03)
            ax.barh(yi, acc, height=height, color=color)
            ax.text(acc + 0.01, yi, f"{acc:.2f}", va="center", fontsize=8, color=INK)
    n = frames["both"].groupby("scope").agg(m=("merchant", "size"), t=("transactions", "sum"))
    ax.set_yticks(
        [len(scopes) - 1 - i for i in range(len(scopes))],
        [
            f"{lab}\n{n.loc[s, 'm']} merchants, {n.loc[s, 't'] / 1000:.1f}k txns"
            for s, lab in scopes
        ],
    )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    return save(fig, out, "unseen_by_scope")


def per_merchant(r: dict[str, Any], out: Path) -> Path:
    exps = r["experiments"]
    base = pd.DataFrame(exps["both"]["unseen"]["by_merchant"])[
        ["merchant", "category", "scope", "transactions"]
    ]
    for kind in MODELS:
        base[kind] = pd.DataFrame(exps[kind]["unseen"]["by_merchant"])["accuracy"].to_numpy()
    base = base.sort_values(["category", "both"], ascending=[True, False]).reset_index(drop=True)
    # One header row per category, then its merchants; y counts down from the top.
    positions, headers, y_pos = [], [], 0
    for category, group in base.groupby("category", sort=True):
        headers.append((y_pos, category))
        y_pos += 1
        for idx in group.index:
            positions.append((idx, y_pos))
            y_pos += 1
    total = y_pos
    y = np.empty(len(base))
    for idx, pos in positions:
        y[idx] = total - 1 - pos
    fig, ax = plt.subplots(figsize=(9, 0.21 * total + 1.5))
    height = fig.get_figheight()
    fig.subplots_adjust(left=0.36, right=0.97, top=1 - 1.25 / height, bottom=0.3 / height)
    title(
        fig,
        "Unseen-merchant accuracy, per merchant",
        "Each row is one holdout merchant (type · transactions). Grey bar spans the three "
        "models; Both is drawn on top where dots coincide.",
    )
    legend(fig, 1 - 0.6 / height)
    style(ax)
    lo = base[list(MODELS)].min(axis=1)
    hi = base[list(MODELS)].max(axis=1)
    ax.hlines(y, lo, hi, color=GRID, linewidth=3)
    for kind, (_, color) in MODELS.items():
        ax.scatter(base[kind], y, s=34, color=color, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ticks = list(y) + [total - 1 - pos for pos, _ in headers]
    labels = [
        f"{m}  ·  {sc}  ·  {t:,}"
        for m, sc, t in zip(base["merchant"], base["scope"], base["transactions"], strict=True)
    ] + [category for _, category in headers]
    ax.set_yticks(ticks, labels, fontsize=8)
    for tick in ax.get_yticklabels()[len(base) :]:
        tick.set_fontweight("bold")
        tick.set_color(INK)
        tick.set_fontsize(9)
    for pos, _ in headers[1:]:
        ax.axhline(total - 1 - pos + 0.5, color=BASELINE, linewidth=1)
    ax.set_ylim(-0.7, total - 0.3)
    ax.set_xlim(-0.03, 1.03)
    ax.xaxis.tick_top()
    return save(fig, out, "unseen_per_merchant")


def confusion(r: dict[str, Any], out: Path, kind: str = "both") -> Path:
    labels = r["setup"]["labels"]
    spending = r["setup"]["spending_labels"]
    matrix = np.array(r["experiments"][kind]["unseen"]["confusion"], dtype=float)
    keep = [labels.index(c) for c in spending]
    rows = matrix[keep]
    shares = rows / rows.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.78, bottom=0.04)
    title(
        fig,
        f"Unseen-merchant confusion ({MODELS[kind][0].lower()})",
        "Row-normalized: each row is a true category; columns are the predicted category.",
    )
    heat(ax, shares, spending, [c.replace(" & ", "\n& ") for c in labels], "share")
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor", fontsize=8)
    ax.set_ylabel("True category")
    return save(fig, out, f"unseen_confusion_{kind}")


def classes(r: dict[str, Any], out: Path) -> Path:
    facts = r["facts"]
    counts = pd.Series(facts["transactions_by_category"]).sort_values()
    holdout = facts["holdout_merchants_by_category"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.2, right=0.9, top=0.84, bottom=0.11)
    title(
        fig,
        "Transactions per category (default dataset)",
        "Macro F1 weighs Travel (2.9k) as much as Dining (418k). Right: holdout merchants.",
    )
    style(ax)
    ax.barh(counts.index, counts.to_numpy(), height=0.6, color=MODELS["ngrams"][1])
    for i, (cat, v) in enumerate(counts.items()):
        ax.text(v + 4000, i, f"{v:,}", va="center", fontsize=8, color=INK)
        ax.text(
            1.08,
            i,
            str(holdout.get(cat, "—")),
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    ax.text(
        1.08,
        len(counts) - 0.3,
        "holdout",
        transform=ax.get_yaxis_transform(),
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlim(0, counts.max() * 1.15)
    return save(fig, out, "dataset_categories")


CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --good:#006300; --bad:#d03b3b; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; }
.sub { color: var(--ink2); margin: 0 0 20px; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:14px 16px; }
.tile .label { color:var(--ink2); font-size:13px; }
.tile .value { font-size:32px; font-weight:600; margin:2px 0; }
.tile .note { font-size:13px; }
.note { color:var(--ink2); } .pass { color:var(--good); } .fail { color:var(--bad); }
figure { margin: 12px 0 0; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:8px; }
figure img { width:100%; height:auto; display:block; }
table { border-collapse: collapse; width:100%; background:var(--surface); font-size:13px;
  font-variant-numeric: tabular-nums; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--ink2); font-weight:600; }
td.num, th.num { text-align:right; }
.meta { color:var(--muted); font-size:12px; margin-top:24px; }
"""


def report_html(r: dict[str, Any], figures: dict[str, Path], embed: bool) -> str:
    exps = r["experiments"]

    def img(name: str, alt: str) -> str:
        path = figures[name]
        if embed:
            src = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        else:
            src = f"figures/{path.name}"
        return f'<figure><img src="{src}" alt="{html.escape(alt)}"></figure>'

    known = exps["both"]["known"]["macro_f1_spending"]
    unseen = exps["both"]["unseen"]
    best_unseen = max(exps.values(), key=lambda e: e["unseen"]["macro_f1_spending"])
    boot = best_unseen["unseen"]["merchant_bootstrap"]

    def tile(label: str, value: str, note: str, ok: bool | None) -> str:
        cls, mark = {True: ("pass", "✓ "), False: ("fail", "✗ "), None: ("", "")}[ok]
        return (
            f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note {cls}">{mark}{note}</div></div>'
        )

    tiles = "".join(
        [
            tile("Known merchants (FR-3)", f"{known:.3f}", "target ≥ 0.90", known >= 0.90),
            tile(
                "Unseen merchants (FR-4)",
                f"{best_unseen['unseen']['macro_f1_spending']:.3f}",
                f"target ≥ 0.80 · 95% interval {boot['low']:.2f}&ndash;{boot['high']:.2f}",
                best_unseen["unseen"]["macro_f1_spending"] >= 0.80,
            ),
            tile(
                "All test users",
                f"{exps['both']['test_users']['macro_f1_spending']:.3f}",
                "realistic mix of known and new merchants; no target",
                None,
            ),
        ]
    )
    head = "".join(f'<th class="num">{TEST_SETS[k]}</th>' for k in TEST_SETS)
    body = "".join(
        "<tr><td>"
        + MODELS[k][0]
        + "</td>"
        + "".join(f'<td class="num">{exps[k][t]["macro_f1_spending"]:.3f}</td>' for t in TEST_SETS)
        + f'<td class="num">{exps[k]["unseen"]["merchant_bootstrap"]["low"]:.3f}&ndash;'
        f"{exps[k]['unseen']['merchant_bootstrap']['high']:.3f}</td></tr>"
        for k in MODELS
    )
    errors = "".join(
        f"<tr><td>{html.escape(e['merchant'])}</td><td>{e['true']}</td><td>{e['predicted']}</td>"
        f'<td class="num">{e["transactions"]:,}</td></tr>'
        for e in unseen["top_errors"]
    )
    d, env = r["dataset"], r["environment"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FR-3 Feasibility Results</title><style>{CSS}</style></head>
<body><main>
<h1>FR-3 categorization feasibility</h1>
<p class="sub">Logistic regression on the default synthetic dataset. Macro F1 over the 12
spending categories; Income excluded.</p>
<div class="tiles">{tiles}</div>
<h2>Headline</h2>
<table><tr><th>Features</th>{head}<th class="num">Unseen 95% interval</th></tr>{body}</table>
{img("headline", "Macro F1 by test set and feature set")}
<h2>Is the unseen-merchant gap real?</h2>
{img("unseen_bootstrap", "Bootstrap distribution of unseen-merchant macro F1")}
<h2>Where unseen merchants fail</h2>
{img("unseen_by_scope", "Accuracy by merchant type")}
{img("per_class_f1", "Per-class F1 heatmaps")}
{img("unseen_confusion_both", "Unseen-merchant confusion matrix")}
<h2>Largest unseen-merchant errors (both)</h2>
<table><tr><th>Merchant</th><th>True</th><th>Predicted</th><th class="num">Transactions</th></tr>
{errors}</table>
{img("unseen_per_merchant", "Per-merchant accuracy")}
<h2>Dataset</h2>
{img("dataset_categories", "Transactions per category")}
<p class="meta">Dataset {d["spec_name"]} · spec hash {d["spec_hash"][:12]} · generator
{d["generator_version"]} · code {env["git_commit"][:7]} · Python {env["python"]} ·
scikit-learn {env["sklearn"]} · embeddings {env["embedding_model"]} ·
seed {r["setup"]["seed"]}</p>
</main></body></html>
"""


def chrome() -> str | None:
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac.exists():
        return str(mac)
    for name in ("google-chrome", "chromium", "chromium-browser"):
        if found := shutil.which(name):
            return found
    return None


def screenshot(page: str, out: Path, width: int = 1200) -> Path | None:
    browser = chrome()
    if browser is None:
        print("Chrome not found; skipping screenshot")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.html"
        source.write_text(page, encoding="utf-8")
        raw = Path(tmp) / "raw.png"
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=1",
                f"--window-size={width},12000",
                f"--screenshot={raw}",
                source.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        image = Image.open(raw).convert("RGB")
    # Trim the empty page below the content.
    pixels = np.asarray(image)
    background = pixels[-1, 0]
    content = np.flatnonzero((np.abs(pixels.astype(int) - background).sum(axis=2) > 6).any(axis=1))
    bottom = min(int(content[-1]) + 32, image.height) if len(content) else image.height
    image.crop((0, 0, image.width, bottom)).save(out, optimize=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    results = json.loads((args.results / "feasibility.json").read_text())
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        p.stem: p
        for p in (
            headline(results, out),
            bootstrap(results, out),
            per_class(results, out),
            by_scope(results, out),
            per_merchant(results, out),
            confusion(results, out),
            classes(results, out),
        )
    }
    (args.results / "report.html").write_text(report_html(results, figures, embed=False))
    shot = screenshot(
        report_html(results, figures, embed=True), args.results / "screenshots" / "report.png"
    )
    print(f"wrote {len(figures)} figures, report.html" + (f", {shot.name}" if shot else ""))


if __name__ == "__main__":
    main()
