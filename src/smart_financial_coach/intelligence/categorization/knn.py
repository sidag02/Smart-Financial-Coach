"""FR-4 feasibility candidate: nearest neighbours over embeddings of known merchant strings.

Each distinct normalized merchant string in the training rows is one example, labelled with its
most common category. Frequent merchants then count once, not once per transaction, and a few
noisy labels on a string are outvoted. A new string takes the similarity-weighted vote of its `k`
closest known strings (cosine similarity of unit-length embeddings, raised to `power` so the
closest neighbours dominate). A known string finds itself at similarity 1.

`scores` returns class probabilities and familiarity, so `categorization/calibrated` can wrap it.
"""

from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from smart_financial_coach.intelligence.categorization.contract import CategorizerModel
from smart_financial_coach.intelligence.categorization.embeddings import (
    clear_cache,
    embed,
    get_embedder,
    verify,
)
from smart_financial_coach.intelligence.categorization.linear import NORMALIZERS
from smart_financial_coach.intelligence.models.registry import register

Floats = npt.NDArray[np.float64]
CHUNK = 2_048  # query strings per similarity block: CHUNK x known strings x 4 bytes
SMOOTHING = 1e-3  # keeps every class's probability above 0, so calibration stays finite


@register("categorization/knn_text")
class KnnText(CategorizerModel):
    def __init__(
        self,
        embeddings: str = "BAAI/bge-small-en-v1.5",
        k: int = 10,
        power: float = 4.0,
        normalizer: str = "v1",
    ) -> None:
        super().__init__(embeddings=embeddings, k=k, power=power, normalizer=normalizer)
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        self.embedding_model = embeddings
        self.k = k
        self.power = power
        self.normalize = NORMALIZERS[normalizer]
        self.embedder = get_embedder(embeddings)
        self.embedding_file: dict[str, str] = {}
        self.strings: list[str] = []
        self.labels: npt.NDArray[np.integer[Any]] = np.empty(0, dtype=np.int64)
        self.vectors: npt.NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)
        self.vocabulary: frozenset[str] = frozenset()

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> Self:
        labels = self._remember(y)
        text = x["merchant_raw"].reset_index(drop=True).map(self.normalize)
        # One example per string: its most common label (ties alphabetical, for determinism)
        counts = pd.DataFrame({"text": text, "label": labels}).value_counts().reset_index()
        majority = counts.sort_values(["text", "count", "label"], ascending=[True, False, True])
        majority = majority.drop_duplicates("text")
        index = {c: i for i, c in enumerate(self.categories)}
        self.strings = majority["text"].tolist()
        self.labels = majority["label"].map(index).to_numpy(dtype=np.int64)
        self.vocabulary = frozenset(self.strings)
        self.embedding_file = self.embedder.fingerprint()
        self.vectors = embed(self.embedder, self.strings, self.embedding_file["sha256"])
        return self

    def _vote(self, queries: npt.NDArray[np.float32]) -> Floats:
        k = min(self.k, len(self.strings))
        proba = np.zeros((len(queries), len(self.categories)))
        for start in range(0, len(queries), CHUNK):
            similarity = queries[start : start + CHUNK] @ self.vectors.T
            nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
            weight = np.clip(np.take_along_axis(similarity, nearest, axis=1), 0, None) ** self.power
            block = proba[start : start + CHUNK]
            for column in range(k):
                np.add.at(
                    block,
                    (np.arange(len(block)), self.labels[nearest[:, column]]),
                    weight[:, column],
                )
        proba += SMOOTHING
        normalized: Floats = proba / proba.sum(axis=1, keepdims=True)
        return normalized

    def scores(self, x: pd.DataFrame) -> tuple[Floats, npt.NDArray[np.bool_]]:
        if not self.strings:
            raise RuntimeError("fit the model first")
        text = x["merchant_raw"].reset_index(drop=True).map(self.normalize)
        unique, inverse = np.unique(text.to_numpy(dtype=str), return_inverse=True)
        vectors = embed(self.embedder, unique.tolist(), self.embedding_file["sha256"])
        proba = self._vote(vectors)[inverse]
        familiar = np.fromiter((s in self.vocabulary for s in text), dtype=bool, count=len(text))
        return proba, familiar

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        proba, _ = self.scores(x)
        top = proba.argmax(axis=1)
        return self._output(x, np.asarray(self.categories)[top], proba[np.arange(len(top)), top])

    def reset_caches(self) -> None:
        clear_cache()

    def verify_environment(self) -> None:
        verify(self.embedder, self.embedding_file)

    def describe_fit(self) -> dict[str, Any]:
        return {"known_strings": len(self.strings)}
