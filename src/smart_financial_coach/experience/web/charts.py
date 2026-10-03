"""Server-rendered chart geometry: the money-flow diagram and the monthly trend (mockup 1a).

Pure functions of tool results, so the charts can't disagree with the numbers beside them.
Positions are percentages of the chart box; ribbons are SVG paths in a 920 x 260 viewBox.
"""

from dataclasses import dataclass, field
from typing import Any

WIDTH, HEIGHT = 920.0, 260.0
GAP_COLUMN, GAP_ITEM, MIN_ITEM, SLIM_BLOCK = 8.0, 4.0, 18.0, 44.0
X_A1, X_B0, X_B1, X_C0 = 150.0, 350.0, 520.0, 700.0

# Category colours from the mockups: (mark, soft fill light, soft fill dark)
CATEGORY_COLORS: dict[str, tuple[str, str, str]] = {
    "Housing": ("#8FA9C9", "#E3EAF3", "#22303F"),
    "Dining": ("#F2A36B", "#FDE6D4", "#3D2A1E"),
    "Transportation": ("#8EC3AE", "#E0F0E9", "#26372F"),
    "Groceries": ("#AFCB8C", "#EAF2DF", "#2F3726"),
    "Shopping": ("#E3A6BF", "#F8E5EC", "#3F2A33"),
    "Travel": ("#A3B3E6", "#E6EAF8", "#2A3045"),
    "Utilities": ("#D8BE7E", "#F5EDD7", "#3B3322"),
    "Entertainment": ("#C2A3DE", "#EFE6F7", "#352B40"),
    "Health & Fitness": ("#86C4BB", "#DFF0ED", "#243934"),
    "Subscriptions": ("#E59A8F", "#F9E2DE", "#3F2723"),
    "Childcare & Education": ("#EDC27A", "#FBEFD8", "#3E3220"),
    "Insurance & Fees": ("#B6ACA1", "#EDE9E4", "#34302C"),
    "Income": ("#7FB38A", "#E4F0E5", "#1F3326"),
}


def colors(category: str) -> tuple[str, str, str]:
    return CATEGORY_COLORS.get(category, ("#B6ACA1", "#EDE9E4", "#34302C"))


def pct(value: float, whole: float) -> str:
    return f"{value / whole * 100:.2f}%" if whole else "0%"


@dataclass
class _Node:
    label: str
    value: float
    kind: str  # css modifier: income, savings, essentials, other, left
    kids: list[tuple[str, float]] = field(default_factory=list)
    u0: float = 0.0
    u1: float = 0.0
    index: int = 0


def _ribbon(x0: float, a0: float, a1: float, x1: float, b0: float, b1: float) -> str:
    xm = (x0 + x1) / 2
    return (
        f"M{x0:.1f},{a0:.1f} C{xm:.1f},{a0:.1f} {xm:.1f},{b0:.1f} {x1:.1f},{b0:.1f} "
        f"L{x1:.1f},{b1:.1f} C{xm:.1f},{b1:.1f} {xm:.1f},{a1:.1f} {x0:.1f},{a1:.1f} Z"
    )


def money_flow(
    income: float, by_category: list[tuple[str, float]], essentials: frozenset[str]
) -> dict[str, Any]:
    """Came in -> essentials / everything else / left over -> categories.

    `by_category` is spending per category (positive); categories with no net spending are left
    out. When spending exceeds income, the difference shows as "From savings".
    """
    cats = [(c, a) for c, a in by_category if a > 0]
    needed = [(c, a) for c, a in cats if c in essentials]
    others = [(c, a) for c, a in cats if c not in essentials]
    spent = sum(a for _, a in cats)
    income = max(income, 0.0)
    left = income - spent
    column_a = [_Node("Came in", income, "income")] if income > 0 else []
    if left < 0:
        column_a.append(_Node("From savings", -left, "savings"))
    column_b = [
        node
        for node in (
            _Node("Essentials", sum(a for _, a in needed), "essentials", needed),
            _Node("Everything else", sum(a for _, a in others), "other", others),
            _Node("Left over", max(left, 0.0), "left"),
        )
        if node.value > 0
    ]
    total = sum(n.value for n in column_a)
    if total <= 0 or not column_b:
        return {"empty": True}
    k = (HEIGHT - GAP_COLUMN * (max(len(column_a), len(column_b)) - 1)) / total
    for column in (column_a, column_b):
        u = 0.0
        for i, node in enumerate(column):
            node.u0, node.u1, node.index = u, u + node.value, i
            u += node.value

    def y(node: _Node, u: float) -> float:
        return u * k + GAP_COLUMN * node.index

    ribbons = []
    for source in column_a:
        for target in column_b:
            lo, hi = max(source.u0, target.u0), min(source.u1, target.u1)
            if hi > lo:
                path = _ribbon(
                    X_A1, y(source, lo), y(source, hi), X_B0, y(target, lo), y(target, hi)
                )
                ribbons.append({"d": path, "kind": target.kind})

    # Category items on the right, at least MIN_ITEM tall so every label fits
    items = [(c, a, b) for b in column_b for c, a in b.kids]
    available = HEIGHT - GAP_ITEM * (len(items) - 1)
    fixed: set[str] = set()
    scale = 1.0
    for _ in range(5):
        free = sum(a for c, a, _ in items if c not in fixed)
        scale = (available - MIN_ITEM * len(fixed)) / free if free else 0.0
        fixed |= {c for c, a, _ in items if a * scale < MIN_ITEM}
    tops: dict[str, tuple[float, float]] = {}
    top = 0.0
    for c, a, _ in items:
        height = MIN_ITEM if c in fixed else a * scale
        tops[c] = (top, height)
        top += height + GAP_ITEM
    for b in column_b:
        u = b.u0
        for c, a in b.kids:
            t, h = tops[c]
            ribbons.append(
                {"d": _ribbon(X_B1, y(b, u), y(b, u + a), X_C0, t, t + h), "category": c}
            )
            u += a

    blocks = []
    for column, x0, x1 in ((column_a, 0.0, X_A1), (column_b, X_B0, X_B1)):
        for n in column:
            blocks.append(
                {
                    "label": n.label,
                    "amount": n.value,
                    "kind": n.kind,
                    "left": pct(x0, WIDTH),
                    "width": pct(x1 - x0, WIDTH),
                    "top": pct(y(n, n.u0), HEIGHT),
                    "height": pct((n.u1 - n.u0) * k, HEIGHT),
                    "slim": (n.u1 - n.u0) * k < SLIM_BLOCK,  # label and amount on one line
                }
            )
    categories = [
        {
            "name": c,
            "amount": a,
            "color": colors(c)[0],
            "top": pct(tops[c][0], HEIGHT),
            "height": pct(tops[c][1], HEIGHT),
        }
        for c, a, _ in items
    ]
    return {
        "empty": False,
        "viewbox": f"0 0 {WIDTH:.0f} {HEIGHT:.0f}",
        "ribbons": [
            {**r, "color": colors(r["category"])[0] if "category" in r else None} for r in ribbons
        ],
        "blocks": blocks,
        "categories": categories,
    }


def trend(months: list[tuple[str, float]], highlight: str) -> list[dict[str, object]]:
    """Bars for monthly spending; the selected month and the year's peak are labelled."""
    peak = max((v for _, v in months), default=0.0)
    top = peak * 1.08 or 1.0
    bars = []
    for key, value in months:
        selected, is_peak = key == highlight, value == peak and value > 0
        bars.append(
            {
                "month": key,
                "value": value,
                "height": pct(value, top),
                "kind": "selected" if selected else "peak" if is_peak else "",
                "label": selected or is_peak,
            }
        )
    return bars
