"""Publishing a promoted model file, so every clone can load what `PROMOTED` names.

Model files are too large to commit (10-13 MB against the repo's 5 MB limit), so `promote` attaches
each one to a GitHub Release of this repository, tagged `<service>-<version>`, and records the
download URL in the committed promotion log. The manifest, with the file's checksum, is committed,
so a file downloaded from the release is trusted only if it matches (see `artifact.fetch_model`).

Continuous deployment will move serving to a container registry; only the URL changes.
"""

import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from smart_financial_coach.intelligence.models.artifact import sha256

Run = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class PublishError(RuntimeError):
    pass


class Publisher(Protocol):
    def publish(self, tag: str, path: Path, notes: str, commit: str | None = None) -> str:
        """Make `path` downloadable under `tag`, at `commit` (the code that trained it) when the
        store can record one; return its URL."""
        ...


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


class GitHubReleases:
    """One release per promoted version, through the `gh` CLI (needs write access to the repo).

    Publishing a version that already has a release (promoting it again, e.g. a rollback) uploads
    nothing: the release's file is downloaded and must match the local one.

    The release's tag points at the commit that trained the model, so its source archives are the
    training code. `promote` passes a commit only for runs trained from a clean tree
    (`sfc.git_dirty`); with uncommitted changes, the commit isn't the training code. A commit that
    GitHub doesn't have (rebased away or never pushed) falls back to the default branch, and the
    notes say so. The manifest records the commit and code version either way.
    """

    def __init__(self, repo: str | None = None, run: Run = _run) -> None:
        self.repo = repo  # "owner/name"; None: the repository of the current checkout
        self.run = run

    def _gh(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return self.run(["gh", *args])
        except FileNotFoundError as error:
            raise PublishError("publishing needs the GitHub CLI (`gh`) on PATH") from error

    def _repo(self) -> str:
        if self.repo is None:
            done = self._gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
            if done.returncode != 0 or not done.stdout.strip():
                raise PublishError(
                    f"can't tell which GitHub repository to publish to: {done.stderr}"
                )
            self.repo = done.stdout.strip()
        return self.repo

    def publish(self, tag: str, path: Path, notes: str, commit: str | None = None) -> str:
        repo = self._repo()
        url = f"https://github.com/{repo}/releases/download/{tag}/{path.name}"
        if self._gh("release", "view", tag, "--repo", repo).returncode == 0:
            self._check_existing(repo, tag, path)
            return url
        target: list[str] = []
        if commit and self._gh("api", f"repos/{repo}/commits/{commit}").returncode == 0:
            target = ["--target", commit]
        elif commit:
            notes += (
                f" Trained at commit {commit}, which isn't on GitHub, so this release is tagged "
                "at the default branch instead."
            )
        done = self._gh(
            "release",
            "create",
            tag,
            str(path),
            "--repo",
            repo,
            "--title",
            tag,
            "--notes",
            notes,
            *target,
        )
        if done.returncode != 0:
            raise PublishError(f"couldn't create release {tag} in {repo}: {done.stderr}")
        return url

    def _check_existing(self, repo: str, tag: str, path: Path) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            done = self._gh(
                "release", "download", tag, "--repo", repo, "--pattern", path.name, "--dir", tmp
            )
            published = Path(tmp) / path.name
            if done.returncode != 0 or not published.exists():
                raise PublishError(f"release {tag} exists in {repo} without {path.name}")
            if sha256(published) != sha256(path):
                raise PublishError(f"release {tag} in {repo} holds a different {path.name}")
