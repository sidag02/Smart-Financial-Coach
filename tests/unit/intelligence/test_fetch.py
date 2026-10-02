"""First-use model download: verified against the manifest, and safe with concurrent workers."""

import hashlib
import json
import threading
import urllib.request
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest

from smart_financial_coach.intelligence.models.artifact import (
    MANIFEST_FILE,
    MODEL_FILE,
    ArtifactError,
    fetch_model,
)

CONTENT = b"first half|second half"


class Response:
    """A download that can pause after its first chunk until `resume` is set."""

    def __init__(self, resume: threading.Event | None = None) -> None:
        self.chunks = [CONTENT[:11], CONTENT[11:], b""]
        self.resume = resume
        self.paused = threading.Event()

    def read(self, size: int) -> bytes:
        if len(self.chunks) == 2 and self.resume is not None:
            self.paused.set()
            assert self.resume.wait(timeout=10)
        return self.chunks.pop(0)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    manifest = {"model_sha256": hashlib.sha256(CONTENT).hexdigest()}
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest))
    return tmp_path


def test_concurrent_downloads_never_expose_a_partial_file(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = threading.Event()
    slow, fast = Response(resume), Response()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout: slow if url == "slow" else fast
    )
    errors: list[BaseException] = []

    def first() -> None:
        try:
            fetch_model(folder, "slow")
        except BaseException as error:  # surfaced by the assertion below
            errors.append(error)

    worker = threading.Thread(target=first)
    worker.start()
    assert slow.paused.wait(timeout=10)

    assert not (folder / MODEL_FILE).exists()  # half-downloaded: not under the final name
    fetch_model(folder, "fast")  # a second worker isn't blocked or corrupted by the first
    assert (folder / MODEL_FILE).read_bytes() == CONTENT
    resume.set()
    worker.join(timeout=10)

    assert errors == []
    assert (folder / MODEL_FILE).read_bytes() == CONTENT
    assert sorted(p.name for p in folder.iterdir()) == [MANIFEST_FILE, MODEL_FILE]


def test_a_mismatched_download_leaves_nothing(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Wrong(Response):
        def __init__(self) -> None:
            super().__init__()
            self.chunks = [b"something else", b""]

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Wrong())

    with pytest.raises(ArtifactError, match="doesn't match"):
        fetch_model(folder, "wrong")
    assert sorted(p.name for p in folder.iterdir()) == [MANIFEST_FILE]
