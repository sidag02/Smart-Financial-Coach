"""The grounding check (FR-14): every number in an answer comes from a tool result it cites.

FR-13 to FR-15 design, §3. A deterministic check that runs on every answer before it's shown,
and is the evaluation suite's grounding grader, so serving and evaluation agree on "grounded".

- **Numbers in the answer:** money, plain numbers, percentages, "x" multiples and "N in 10"
  chances. Dates, years, list markers, store numbers ("#8492") and source tags are left out, and
  so are numbers the person wrote in any of their messages.
- **Values in tool results:** every number, every number inside a text field (a spike's reason
  says "$1,383"), and every list's length (counts), kept with its source id and path.
- **Citations:** each number takes the source tags that follow it in its sentence, or else the
  ones before it in the sentence, or else (a list cited once, at its end) every tag in its
  paragraph. A number with none fails, and a paragraph never borrows another's tags.
- **A direct number** matches a value in a result it cites, rounded as written: cents, whole
  dollars or a decimal place. A percentage may also be a ratio as a change (1.42 → 42%) or a
  share or probability (0.62 → 62%). "N in 10" must be exactly `chance_words(p_goal_met)`.
- **A derived number** (a sum or difference of amounts, which the prompt allows) passes only when
  both values come from the cited results and are summary fields: two fields of one record, or
  one field in two items of one list. Never transaction rows, text or counts.
- **Amounts and counts don't mix:** the tools return money as floats and counts as integers, so
  a dollar amount never matches a count, or a count an amount.

The check matches values, not meaning: a real value under the wrong label ("$412 on groceries"
when $412 is dining) passes. The suite's required facts, tied to named fields, catch those.

    result = check(answer, payloads, user_texts, constants=instruction_numbers(prompt, specs))
    result.ok, result.unmatched  # e.g. False, ("$1,250",)
"""

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from smart_financial_coach.access.goal_forecasts import chance_words

# Lists whose items are single transactions: quoted one by one, never added up (§3.5)
ROW_LISTS = frozenset({"transactions", "unusual_transactions", "largest_charges"})
# Text fields that are identifiers or dates, never numbers someone would quote
SKIP_TEXT = re.compile(r"(^|_)(id|date|month|as_of|start|end|version)$|^(source_id|currency)$")
PROBABILITIES = frozenset({"p_goal_met"})
# Floats in tool results that aren't amounts: proportions (0.62 → "62%"), ratios (2.05 → "105%
# more", "2.05x") and other numbers (about 27 purchases); every other float is money
PROPORTIONS = frozenset({"p_goal_met", "chance", "confidence", "share"})
RATIOS = frozenset({"ratio"})
OTHER_NUMBERS = frozenset({"usual_count", "minutes_apart"})

MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
# A number with thousands separators only where they belong: "2026," is a year and a comma
DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
TAGS = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
PARAGRAPH = re.compile(r"\n[ \t]*\n")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\[(\"'])|\n+")
NUMBER = re.compile(
    rf"""
    (?P<chance>(?:(?P<qual>about|better\ than|less\ than)\s+)?(?:an?\s+)?
        (?P<tenths>\d)\s+in\s+10\b(?:\s+chance)?)
    | (?P<iso>\b\d{{4}}-\d{{2}}(?:-\d{{2}})?\b)
    | (?P<date>\b(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?
        | \b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\b)
    | (?P<money>(?P<sign>[-\u2212])?\$\s?(?P<m>{DIGITS})(?P<k>[kK]\b)?)
    | (?P<plain>(?<![\w#.,/$-])(?P<p>{DIGITS})(?![\w/]|\.\d|-\d)
        (?P<pct>\s?%)?(?P<times>\s?[\u00d7x](?!\w))?)
    """,
    re.IGNORECASE | re.VERBOSE,
)
LIST_MARKER = re.compile(r"^[ \t]*\d+[.)][ \t]", re.MULTILINE)


@dataclass(frozen=True)
class Value:
    number: float
    source: str  # "S3"
    path: tuple[str | int, ...]
    kind: str  # "field", "text" (inside a string) or "count" (a list's length)
    row: bool  # inside a transaction row
    # "money", "count" (an integer or a list's length), "proportion", "ratio", "percent" (in text)
    # or "number": what an answer's number may match it as
    unit: str

    @property
    def money(self) -> bool:
        return self.unit == "money"


@dataclass(frozen=True)
class Number:
    text: str  # as written
    value: float
    kind: str  # "money", "percent", "times", "plain" or "chance"
    decimals: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Grounding:
    numbers: int  # numbers checked (user and instruction numbers aren't)
    unmatched: tuple[str, ...]  # as written, in order

    @property
    def ok(self) -> bool:
        return not self.unmatched


def values(source: str, payload: Any) -> list[Value]:
    """Every number a tool result holds, as the model saw it."""
    found: list[Value] = []

    def walk(node: Any, path: tuple[str | int, ...], row: bool) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, int | float):
            if math.isfinite(node):
                found.append(Value(float(node), source, path, "field", row, _unit(path, node)))
        elif isinstance(node, str):
            key = path[-1] if path else ""
            if not (isinstance(key, str) and SKIP_TEXT.search(key)):
                for number in _numbers(node):
                    unit = {"money": "money", "percent": "percent"}.get(number.kind, "number")
                    found.append(Value(number.value, source, path, "text", row, unit))
        elif isinstance(node, Mapping):
            for key, child in node.items():
                walk(child, (*path, key), row)
        elif isinstance(node, Sequence):
            found.append(Value(float(len(node)), source, (*path, "len"), "count", row, "count"))
            in_rows = row or (bool(path) and path[-1] in ROW_LISTS)
            for i, child in enumerate(node):
                walk(child, (*path, i), in_rows)

    walk(payload, (), False)
    return found


def instruction_numbers(*texts: str) -> frozenset[float]:
    """Plain numbers the coach was told (the prompt and tool descriptions: "12 months",
    "1.3x"): product rules it may repeat with no source."""
    return frozenset(n.value for text in texts for n in _numbers(text) if n.kind == "plain")


def check(
    answer: str,
    payloads: Mapping[str, Any],
    user_texts: Iterable[str] = (),
    *,
    constants: frozenset[float] = frozenset(),
) -> Grounding:
    """Check `answer` against the tool results it cites. `payloads` are this conversation's
    tool results by source id ("S1"), `user_texts` the person's messages."""
    said = {(_said_kind(n), n.value) for text in user_texts for n in _numbers(text)}
    by_source = {source: values(source, payload) for source, payload in payloads.items()}
    numbers = [n for n in cited_numbers(answer) if not _exempt(n, said, constants)]
    unmatched = tuple(n.text for n in numbers if not _grounded(n, by_source))
    return Grounding(len(numbers), unmatched)


def cited_numbers(answer: str) -> list[Number]:
    """The answer's numbers, each with the source tags that cite it: the tags that follow it in
    its sentence, else the ones before it there. A number in a sentence with no tags (a bullet
    of a list cited once at its end) takes every tag in its paragraph."""
    found: list[Number] = []
    answer = LIST_MARKER.sub(lambda m: " " * len(m.group(0)), answer)  # "1. Housing …"
    for paragraph in PARAGRAPH.split(answer):
        # A list cited once, at its end: its items take the paragraph's tags. Never another
        # paragraph's (review on #75): a paragraph with no tags has no sources
        shared = tuple(dict.fromkeys(t for _, group in _tags(paragraph) for t in group))
        for sentence in SENTENCE.split(paragraph):
            tags = _tags(sentence)
            clean = TAGS.sub(lambda m: " " * len(m.group(0)), sentence)
            for number, start in _located(clean):
                after = [t for at, t in tags if at >= start]
                before = [t for at, t in tags if at < start]
                sources = after[0] if after else (before[-1] if before else shared)
                found.append(
                    Number(number.text, number.value, number.kind, number.decimals, sources)
                )
    return found


def _tags(text: str) -> list[tuple[int, tuple[str, ...]]]:
    return [
        (m.start(), tuple(t.strip() for t in m.group(1).split(","))) for m in TAGS.finditer(text)
    ]


def numbers_in(text: str) -> list[Number]:
    """The numbers in `text` as the check reads them (the evaluation suite's fact grader)."""
    return _numbers(text)


def matches(number: Number, value: float) -> bool:
    """Whether `number`, rounded as written, is `value` (or its absolute value)."""
    return abs(number.value - abs(value)) <= _tolerance(number)


def _numbers(text: str) -> list[Number]:
    return [n for n, _ in _located(text)]


def _located(text: str) -> list[tuple[Number, int]]:
    found: list[tuple[Number, int]] = []
    for m in NUMBER.finditer(text):
        if m.group("iso") or m.group("date"):
            continue
        if m.group("chance"):
            qual = (m.group("qual") or "about").lower()
            found.append(
                (
                    Number(
                        f"{qual} {m.group('tenths')} in 10",
                        float(m.group("tenths")),
                        "chance",
                        0,
                        (),
                    ),
                    m.start(),
                )
            )
            continue
        digits = m.group("m") if m.group("money") else m.group("p")
        if digits is None:
            continue
        value = float(digits.replace(",", ""))
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if m.group("money"):
            if m.group("k"):
                value, decimals = value * 1000, -2
            kind = "money"
        elif m.group("pct"):
            kind = "percent"
        elif m.group("times"):
            kind = "times"
        else:
            if decimals == 0 and "," not in digits and 1900 <= value <= 2100:
                continue  # a year
            kind = "plain"
        found.append((Number(m.group(0).strip(), value, kind, decimals, ()), m.start()))
    return found


def _unit(path: tuple[str | int, ...], node: float) -> str:
    key = path[-1] if path else ""
    if isinstance(node, int):
        return "count"
    if key in PROPORTIONS:
        return "proportion"
    if key in RATIOS:
        return "ratio"
    return "number" if key in OTHER_NUMBERS else "money"


def _said_kind(number: Number) -> str:
    """Money matches money and a percentage a percentage: "3 months" doesn't exempt "$3"."""
    return number.kind if number.kind in ("money", "percent") else "plain"


def _exempt(number: Number, said: set[tuple[str, float]], constants: frozenset[float]) -> bool:
    if number.kind == "chance":
        return False
    if (_said_kind(number), number.value) in said:
        return True
    return number.kind in ("plain", "times") and number.value in constants


def _tolerance(number: Number) -> float:
    if number.decimals < 0:  # "$1.2k"
        return 50.0
    return 0.5 * 10.0 ** (-number.decimals) + 1e-9


def _grounded(number: Number, by_source: Mapping[str, list[Value]]) -> bool:
    if not number.sources or any(s not in by_source for s in number.sources):
        return False
    cited = [v for s in number.sources for v in by_source[s]]
    tol = _tolerance(number)
    if number.kind == "chance":
        phrase = {
            "about": f"about a {int(number.value)} in 10 chance",
            "better than": f"better than a {int(number.value)} in 10 chance",
            "less than": f"less than a {int(number.value)} in 10 chance",
        }
        qual = number.text.rsplit(" ", 3)[0]
        return any(
            v.kind == "field"
            and v.path[-1] in PROBABILITIES
            and chance_words(v.number) == phrase.get(qual)
            for v in cited
        )
    if number.kind == "percent":  # never from an amount or a transaction row
        forms: list[float] = []
        for v in cited:
            if v.row:
                continue
            if v.unit == "proportion":
                forms.append(v.number * 100)
            elif v.unit == "ratio":
                forms += [abs(v.number - 1) * 100, v.number * 100]
            elif v.unit == "percent":
                forms.append(v.number)
        return any(abs(f - number.value) <= tol for f in forms)
    # An amount is never a count or a ratio, and a count or a ratio never an amount
    if number.kind == "money":
        cited = [v for v in cited if v.money]
    else:
        cited = [v for v in cited if v.unit in ("count", "number", "ratio", "proportion")]
    if any(abs(abs(v.number) - number.value) <= tol for v in cited):
        return True
    if number.kind == "money":  # counts aren't added up: small ones would match almost anything
        return any(abs(d - number.value) <= tol for d in _derived(cited))
    return False


def _derived(cited: list[Value]) -> set[float]:
    """Sums and differences of two summary fields: two fields of one record, or one field in
    two items of one list (design §3.5)."""
    fields = [v for v in cited if v.kind == "field" and v.money and not v.row and v.path]
    records: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    siblings: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for v in fields:
        records[(v.source, v.path[:-1])].append(v.number)
        if len(v.path) >= 2 and isinstance(v.path[-2], int):
            siblings[(v.source, v.path[:-2], v.path[-1])].append(v.number)
    out: set[float] = set()
    for group in (*records.values(), *siblings.values()):
        for a, b in combinations(group, 2):
            out.update((abs(a + b), abs(a - b)))
    return out


def payload_of(content: str) -> Any:
    """A tool result's payload from the JSON the model saw, or None for an error message."""
    try:
        return json.loads(content)
    except ValueError:
        return None
