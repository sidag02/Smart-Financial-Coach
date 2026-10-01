"""Round 0 baselines: the floor every categorizer must beat (FR-3 experiment plan).

- `categorization/majority`: the most frequent training category for everything.
- `categorization/keyword`: generic words to categories (`configs/models/category_keywords.yaml`).
- `categorization/lookup`: the most common training category of each normalized merchant string.

Each confidence is the training precision of the rule that fired, so it means what it says on
data like the training data.
"""

import hashlib
import re
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import yaml

from smart_financial_coach.config import PROJECT_ROOT
from smart_financial_coach.data.features.merchant_text import normalize_merchant
from smart_financial_coach.data.generator.taxonomy import INCOME
from smart_financial_coach.intelligence.categorization.contract import CategorizerModel
from smart_financial_coach.intelligence.models.registry import register

KEYWORDS_FILE = "configs/models/category_keywords.yaml"


def _precision(rule: pd.Series, predicted: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Training precision per rule key, plus the overall rate under ""."""
    correct = predicted.to_numpy() == truth.to_numpy()
    per_rule = pd.Series(correct).groupby(rule.to_numpy()).mean()
    return {str(k): float(v) for k, v in per_rule.items()} | {"": float(correct.mean())}


def _most_common(values: pd.Series) -> str:
    counts = values.value_counts()
    return str(counts[counts == counts.max()].index.sort_values()[0])


@register("categorization/majority")
class Majority(CategorizerModel):
    def __init__(self) -> None:
        super().__init__()
        self.category = ""
        self.share = 0.0

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        labels = self._remember(y)
        self.category = _most_common(labels)
        self.share = float((labels == self.category).mean())
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._output(x, self.category, self.share)


@register("categorization/keyword")
class Keyword(CategorizerModel):
    """The most-matched keyword category; unmatched: Income if money came in, else the fallback.

    The fallback is the most frequent spending category in training. A spending keyword wins over
    the sign, so refunds (positive amounts at shops) keep their category.
    """

    def __init__(
        self, keywords_file: str = KEYWORDS_FILE, keywords_sha256: str | None = None
    ) -> None:
        """`keywords_sha256` puts the file's content in the config hash: if given, it must match."""
        content = (PROJECT_ROOT / keywords_file).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if keywords_sha256 is not None and keywords_sha256 != actual:
            raise ValueError(
                f"{keywords_file} has SHA-256 {actual}; the config expects {keywords_sha256}. "
                "Update the config's keywords_sha256 so the change gets a new run"
            )
        super().__init__(keywords_file=keywords_file, keywords_sha256=actual)
        raw = yaml.safe_load(content.decode("utf-8"))
        self.keywords: dict[str, list[str]] = {
            str(c): [str(w) for w in ws] for c, ws in raw.items()
        }
        self.patterns = {
            category: re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")
            for category, words in self.keywords.items()
        }
        self.fallback = ""
        self.confidence: dict[str, float] = {}

    def _rules(self, x: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        text = x["merchant_raw"].map(normalize_merchant)
        hits = np.column_stack(
            [text.map(lambda s, p=p: len(p.findall(s))).to_numpy() for p in self.patterns.values()]
        )
        matched = hits.max(axis=1) > 0
        best = np.asarray(list(self.patterns))[hits.argmax(axis=1)]  # first listed wins ties
        income = (x["amount"] > 0).to_numpy()
        category = np.where(matched, best, np.where(income, INCOME, self.fallback))
        rule = np.where(
            matched, np.char.add("keyword:", best), np.where(income, "sign", "fallback")
        )
        return pd.Series(category, dtype=str), pd.Series(rule, dtype=str)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        labels = self._remember(y)
        self.fallback = _most_common(labels[labels != INCOME])
        category, rule = self._rules(x.reset_index(drop=True))
        self.confidence = _precision(rule, category, labels)
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        category, rule = self._rules(x.reset_index(drop=True))
        default = self.confidence[""]
        return self._output(x, category.to_numpy(), rule.map(self.confidence).fillna(default))


@register("categorization/lookup")
class Lookup(CategorizerModel):
    """Exact normalized-string memory; strings never seen in training get the majority category."""

    def __init__(self) -> None:
        super().__init__()
        self.table: dict[str, tuple[str, float]] = {}
        self.fallback = ("", 0.0)

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        labels = self._remember(y)
        text = x["merchant_raw"].map(normalize_merchant).reset_index(drop=True)
        counts = pd.crosstab(text, labels)
        top = counts.idxmax(axis=1)  # columns are sorted, so ties go to the first category
        share = counts.max(axis=1) / counts.sum(axis=1)
        self.table = {str(s): (str(top[s]), float(share[s])) for s in counts.index}
        majority = _most_common(labels)
        self.fallback = (majority, float((labels == majority).mean()))
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        found = [
            self.table.get(s, self.fallback) for s in x["merchant_raw"].map(normalize_merchant)
        ]
        return self._output(x, [c for c, _ in found], [p for _, p in found])


def keywords_path(keywords_file: str = KEYWORDS_FILE) -> Path:
    return PROJECT_ROOT / keywords_file
