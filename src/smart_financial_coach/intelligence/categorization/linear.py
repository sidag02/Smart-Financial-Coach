"""The starting candidate: multinomial logistic regression over merchant text and side features.

Feature blocks (FR-3 §3), each switchable from config so experiments can ablate them:

- n-grams: character 2-4-grams of `merchant_raw`, TF-IDF, fitted on training rows only.
- embeddings: a frozen sentence embedding of the normalized text, cached per unique string.
- amount: log1p(|amount|) in 20 bins, so price bands can be non-monotone.
- sign: amount > 0, which with the text separates Income from refunds.
- channel: one-hot.
- hour: 3-hour bins of the local timestamp; kept only if the ablation shows it helps.

Familiarity (a normalized string seen in training) is recorded from all training rows before
the per-class cap, so a known merchant isn't "unseen" just because the cap sampled it out.
"""

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from smart_financial_coach.data.features.merchant_text import (
    normalize_merchant,
    normalize_merchant_poc,
)
from smart_financial_coach.intelligence.categorization.contract import CategorizerModel
from smart_financial_coach.intelligence.categorization.embeddings import (
    clear_cache,
    embed,
    get_embedder,
    verify,
)
from smart_financial_coach.intelligence.models.registry import register

CHANNELS = ("card_present", "online", "ach", "other")
AMOUNT_EDGES = np.linspace(0, 9, 19)  # log1p(|amount|) up to ~$8,100; 20 bins
NORMALIZERS = {"v1": normalize_merchant, "poc": normalize_merchant_poc}

Floats = npt.NDArray[np.float64]


def _one_hot(index: npt.NDArray[np.integer[Any]], width: int) -> sp.csr_matrix:
    n = len(index)
    return sp.csr_matrix((np.ones(n), (np.arange(n), index)), shape=(n, width))


@register("categorization/linear_text")
class LinearText(CategorizerModel):
    def __init__(
        self,
        C: float = 3.0,  # noqa: N803 (scikit-learn's name)
        ngrams: bool = True,
        embeddings: str | None = "BAAI/bge-small-en-v1.5",
        amount: bool = True,
        sign: bool = True,
        channel: bool = True,
        hour: bool = True,
        max_rows_per_class: int | None = 20_000,
        normalizer: str = "v1",
        seed: int = 0,
        max_iter: int = 500,
    ) -> None:
        super().__init__(
            C=C,
            ngrams=ngrams,
            embeddings=embeddings,
            amount=amount,
            sign=sign,
            channel=channel,
            hour=hour,
            max_rows_per_class=max_rows_per_class,
            normalizer=normalizer,
            seed=seed,
            max_iter=max_iter,
        )
        if not ngrams and not embeddings:
            raise ValueError("linear_text needs n-grams, embeddings or both")
        self.C = C
        self.use_ngrams = ngrams
        self.embedding_model = embeddings
        self.side = {"amount": amount, "sign": sign, "channel": channel, "hour": hour}
        self.max_rows_per_class = max_rows_per_class
        self.normalize = NORMALIZERS[normalizer]
        self.seed = seed
        self.max_iter = max_iter
        self.vocabulary: frozenset[str] = frozenset()
        self.vectorizer: TfidfVectorizer | None = None
        self.embedder = get_embedder(embeddings) if embeddings else None
        self.embedding_file: dict[str, str] = {}
        self.classifier: LogisticRegression | None = None

    # --- Features --------------------------------------------------------------------------

    def _side(self, x: pd.DataFrame) -> list[sp.csr_matrix]:
        blocks = []
        amount = x["amount"].to_numpy(dtype=float)
        if self.side["amount"]:
            bins = np.digitize(np.log1p(np.abs(amount)), AMOUNT_EDGES).astype(np.int64)
            blocks.append(_one_hot(bins, 20))
        if self.side["sign"]:
            blocks.append(sp.csr_matrix((amount > 0).astype(float)[:, None]))
        if self.side["channel"]:
            index = x["channel"].map({c: i for i, c in enumerate(CHANNELS)}).fillna(3)
            blocks.append(_one_hot(index.to_numpy(dtype=np.int64), len(CHANNELS)))
        if self.side["hour"]:
            hours = x["ts"].str.slice(11, 13).astype(int).to_numpy() // 3  # "YYYY-MM-DD HH:MM"
            blocks.append(_one_hot(hours.astype(np.int64), 8))
        return blocks

    def _features(self, x: pd.DataFrame, text: pd.Series) -> sp.csr_matrix:
        blocks = self._side(x)
        if self.embedder is not None:
            fingerprint = self.embedding_file["sha256"]
            blocks.append(sp.csr_matrix(embed(self.embedder, text.tolist(), fingerprint)))
        if self.vectorizer is not None:
            blocks.append(self.vectorizer.transform(x["merchant_raw"]))
        return sp.hstack(blocks).tocsr()

    # --- Fit and predict -------------------------------------------------------------------

    def _cap(self, labels: pd.Series) -> npt.NDArray[np.int64]:
        """Row positions kept: at most `max_rows_per_class` per class, seeded."""
        order = np.random.default_rng(self.seed).permutation(len(labels))
        if self.max_rows_per_class is None:
            return np.sort(order)
        shuffled = labels.iloc[order].reset_index(drop=True)
        keep = shuffled.groupby(shuffled).cumcount() < self.max_rows_per_class
        return np.sort(order[keep.to_numpy()])

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        labels = self._remember(y)
        x = x.reset_index(drop=True)
        text = x["merchant_raw"].map(self.normalize)
        self.vocabulary = frozenset(text)  # before the cap
        keep = self._cap(labels)
        rows, kept_text, kept_labels = x.iloc[keep], text.iloc[keep], labels.iloc[keep]
        if self.use_ngrams:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=2,
                sublinear_tf=True,
                max_features=200_000,
            ).fit(rows["merchant_raw"])
        if self.embedder is not None:
            self.embedding_file = self.embedder.fingerprint()
        self.classifier = LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight="balanced"
        ).fit(self._features(rows, kept_text), kept_labels)
        self.categories = tuple(str(c) for c in self.classifier.classes_)
        return self

    def scores(self, x: pd.DataFrame) -> tuple[Floats, npt.NDArray[np.bool_]]:
        """Class probabilities (columns in `categories` order) and whether strings are familiar."""
        if self.classifier is None:
            raise RuntimeError("fit the model first")
        x = x.reset_index(drop=True)
        text = x["merchant_raw"].map(self.normalize)
        proba: Floats = self.classifier.predict_proba(self._features(x, text))
        # A Python set lookup: Series.isin rebuilds the 10k-string set on every call
        familiar = np.fromiter((s in self.vocabulary for s in text), dtype=bool, count=len(text))
        return proba, familiar

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        proba, _ = self.scores(x)
        top = proba.argmax(axis=1)
        return self._output(x, np.asarray(self.categories)[top], proba[np.arange(len(top)), top])

    def reset_caches(self) -> None:
        """Drop process-wide caches, so the next calls cost what a fresh server's would."""
        clear_cache()

    def verify_environment(self) -> None:
        """Called when an artifact is loaded: the embedding file must be the one trained with."""
        if self.embedder is not None:
            verify(self.embedder, self.embedding_file)

    def describe_fit(self) -> dict[str, Any]:
        n_ngrams = len(self.vectorizer.vocabulary_) if self.vectorizer is not None else 0
        return {"ngram_features": n_ngrams, "familiar_strings": len(self.vocabulary)}
