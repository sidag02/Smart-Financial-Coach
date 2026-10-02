"""Frozen sentence embeddings of merchant text, cached per unique string.

    embedder = get_embedder("BAAI/bge-small-en-v1.5")  # fastembed (ONNX, CPU)
    embedder = get_embedder("stub:16")                  # deterministic hashing, for tests
    vectors = embed(embedder, ["taco bell", "blue bottle coffee"])  # (n, dim), unit length

`fastembed` downloads the model file at run time, so results depend on that exact file. Its source,
revision and SHA-256 go into the model's manifest (`fingerprint`), and `verify` refuses to run if
the cached file differs. The cache lives under `SFC_DATA_DIR/models/fastembed` (git-ignored).
"""

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from smart_financial_coach.config import get_settings

Vectors = npt.NDArray[np.float32]
STUB = "stub:"


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    dim: int

    def fingerprint(self) -> dict[str, str]: ...

    def embed(self, texts: Sequence[str]) -> Vectors: ...


class HashEmbedder:
    """Deterministic pseudo-embeddings from a hash of the text: no download, for tests only."""

    def __init__(self, dim: int) -> None:
        self.name = f"{STUB}{dim}"
        self.dim = dim

    def fingerprint(self) -> dict[str, str]:
        return {"source": self.name, "sha256": hashlib.sha256(self.name.encode()).hexdigest()}

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out[i] = v / np.linalg.norm(v)
        return out


class FastEmbedder:
    """A fastembed model, loaded on first use. The ONNX session isn't pickled with the model."""

    def __init__(self, name: str, cache_dir: Path | None = None) -> None:
        self.name = name
        self.cache_dir = cache_dir or get_settings().data_dir / "models" / "fastembed"
        self._model: Any = None
        self.dim = 0

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_model": None}

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir))
            self.dim = int(self._model.model.model_description.dim)
        return self._model

    def fingerprint(self) -> dict[str, str]:
        folder = Path(self._load().model._model_dir)
        files = sorted(folder.glob("*.onnx"))
        if len(files) != 1:
            raise EmbeddingError(f"expected one ONNX file in {folder}, found {len(files)}")
        return {
            "source": folder.parents[1].name.removeprefix("models--").replace("--", "/"),
            "revision": folder.name,
            "file": files[0].name,
            "sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        }

    def embed(self, texts: Sequence[str]) -> Vectors:
        # One string per batch: in a padded batch, a string's vector shifts slightly with its
        # neighbours (up to 4e-4 per component on the quantized model), so training (batched) and
        # serving (one string) would see different features. Alone, every call gives the same
        # vector. Costs ~21 s instead of ~10 s for the default dataset's 13k strings, once per run.
        vectors = list(self._load().embed(list(texts), batch_size=1))
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), -1)


def get_embedder(name: str) -> Embedder:
    if name.startswith(STUB):
        return HashEmbedder(int(name.removeprefix(STUB)))
    return FastEmbedder(name)


# Process-wide, least-recently-used cache: fold models in one run embed each string once, and a
# long-running server keeps its memory bounded (100k vectors of 384 floats is about 150 MB)
MAX_CACHED = 100_000
_CACHE: OrderedDict[tuple[str, str], npt.NDArray[np.float32]] = OrderedDict()


def clear_cache() -> None:
    """Forget every cached vector, as a freshly started process would."""
    _CACHE.clear()


def embed(embedder: Embedder, texts: Sequence[str], fingerprint: str) -> Vectors:
    """Vectors for `texts`, embedding only strings not in the cache."""
    if not texts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    unique = sorted(set(texts))
    missing = [t for t in unique if (fingerprint, t) not in _CACHE]
    found = {t: _CACHE[(fingerprint, t)] for t in unique if (fingerprint, t) in _CACHE}
    if missing:
        found |= dict(zip(missing, embedder.embed(missing), strict=True))
    for text in unique:  # most recently used last; the oldest are evicted first
        _CACHE[(fingerprint, text)] = found[text]
        _CACHE.move_to_end((fingerprint, text))
    while len(_CACHE) > MAX_CACHED:
        _CACHE.popitem(last=False)
    return np.stack([found[t] for t in texts])


def verify(embedder: Embedder, expected: dict[str, str]) -> None:
    """Refuse to run on an embedding file other than the one the model was trained with."""
    actual = embedder.fingerprint()
    if actual["sha256"] != expected["sha256"]:
        raise EmbeddingError(
            f"embedding file for {embedder.name} has SHA-256 {actual['sha256'][:12]}, "
            f"the model was trained with {expected['sha256'][:12]} ({expected.get('revision')})"
        )
