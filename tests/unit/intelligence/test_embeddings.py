"""Embedders: the stub's determinism, the per-string cache, pickling and the checksum check."""

import pickle
from collections import OrderedDict

import numpy as np
import pytest

from smart_financial_coach.intelligence.categorization import embeddings
from smart_financial_coach.intelligence.categorization.embeddings import (
    EmbeddingError,
    FastEmbedder,
    embed,
    get_embedder,
    verify,
)


def test_stub_is_deterministic_and_unit_length() -> None:
    stub = get_embedder("stub:16")
    a, b = stub.embed(["taco bell", "blue bottle"]), stub.embed(["taco bell", "blue bottle"])

    assert a.shape == (2, 16)
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1)
    assert not np.allclose(a[0], a[1])


def test_cache_embeds_each_string_once(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = get_embedder("stub:8")
    calls: list[list[str]] = []
    original = stub.embed

    def counting(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(stub, "embed", counting)
    monkeypatch.setattr(embeddings, "_CACHE", OrderedDict())
    first = embed(stub, ["a", "b", "a"], "fp")
    second = embed(stub, ["b", "c"], "fp")

    assert calls == [["a", "b"], ["c"]]
    assert np.array_equal(first[0], first[2])
    assert np.array_equal(first[1], second[0])


def test_cache_is_bounded_and_evicts_least_recently_used(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = get_embedder("stub:4")
    monkeypatch.setattr(embeddings, "_CACHE", OrderedDict())
    monkeypatch.setattr(embeddings, "MAX_CACHED", 2)
    embed(stub, ["a", "b"], "fp")
    embed(stub, ["a"], "fp")  # "a" is now the most recently used
    embed(stub, ["c"], "fp")

    assert [t for _, t in embeddings._CACHE] == ["a", "c"]  # "b" evicted
    embeddings.clear_cache()
    assert not embeddings._CACHE


def test_fast_embedder_pickles_without_its_session() -> None:
    embedder = FastEmbedder("BAAI/bge-small-en-v1.5")
    embedder._model = object()  # stands in for a loaded ONNX session

    restored = pickle.loads(pickle.dumps(embedder))

    assert restored._model is None
    assert restored.name == embedder.name


def test_verify_refuses_a_different_file() -> None:
    stub = get_embedder("stub:8")
    verify(stub, stub.fingerprint())

    with pytest.raises(EmbeddingError, match="trained with"):
        verify(stub, {"sha256": "0" * 64, "revision": "old"})
