"""GitHub Release publishing, with a fake `gh` that keeps releases in a dict."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from smart_financial_coach.evaluation.publish import GitHubReleases, PublishError

REPO = "owner/repo"


class FakeGh:
    def __init__(
        self, releases: dict[str, dict[str, bytes]] | None = None, commits: set[str] | None = None
    ) -> None:
        self.releases = releases if releases is not None else {}
        self.commits = commits or set()  # commits GitHub has
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(args)
        self.calls.append(args)
        command, tag = args[1:3], args[3] if len(args) > 3 else ""
        ok, missing = (0, "", ""), (1, "", "release not found")
        code, out, err = ok
        if command == ["repo", "view"]:
            out = REPO + "\n"
        elif args[1] == "api":
            sha = args[2].rsplit("/", 1)[-1]
            code, out, err = ok if sha in self.commits else (1, "", "No commit found")
        elif command == ["release", "view"]:
            code, out, err = ok if tag in self.releases else missing
        elif command == ["release", "create"]:
            path = Path(args[4])
            self.releases[tag] = {path.name: path.read_bytes()}
        elif command == ["release", "download"]:
            if tag not in self.releases:
                code, out, err = missing
            else:
                pattern, folder = args[args.index("--pattern") + 1], args[args.index("--dir") + 1]
                if pattern in self.releases[tag]:
                    (Path(folder) / pattern).write_bytes(self.releases[tag][pattern])
        return subprocess.CompletedProcess(args, code, out, err)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"model bytes")
    return path


def test_creates_a_release_and_returns_its_download_url(model_file: Path) -> None:
    gh = FakeGh()

    url = GitHubReleases(run=gh).publish("categorization-v1", model_file, "notes")

    assert url == f"https://github.com/{REPO}/releases/download/categorization-v1/model.joblib"
    assert gh.releases["categorization-v1"] == {"model.joblib": b"model bytes"}
    create = next(c for c in gh.calls if c[1:3] == ["release", "create"])
    assert create[create.index("--repo") + 1] == REPO


def create_call(gh: FakeGh) -> list[str]:
    return next(c for c in gh.calls if c[1:3] == ["release", "create"])


def test_release_is_tagged_at_the_training_commit(model_file: Path) -> None:
    gh = FakeGh(commits={"abc123"})

    GitHubReleases(REPO, run=gh).publish("categorization-v1", model_file, "notes", commit="abc123")

    create = create_call(gh)
    assert create[create.index("--target") + 1] == "abc123"


def test_training_commit_missing_from_github_falls_back_and_says_so(model_file: Path) -> None:
    gh = FakeGh()

    GitHubReleases(REPO, run=gh).publish("categorization-v1", model_file, "notes", commit="abc123")

    create = create_call(gh)
    assert "--target" not in create
    assert "abc123, which isn't on GitHub" in create[create.index("--notes") + 1]


def test_existing_release_with_the_same_file_is_reused(model_file: Path) -> None:
    """Promoting a version again (a rollback) uploads nothing."""
    gh = FakeGh({"categorization-v1": {"model.joblib": b"model bytes"}})

    GitHubReleases(REPO, run=gh).publish("categorization-v1", model_file, "notes")

    assert not [c for c in gh.calls if c[1:3] == ["release", "create"]]


def test_existing_release_with_another_file_is_refused(model_file: Path) -> None:
    gh = FakeGh({"categorization-v1": {"model.joblib": b"something else"}})

    with pytest.raises(PublishError, match=r"different model\.joblib"):
        GitHubReleases(REPO, run=gh).publish("categorization-v1", model_file, "notes")


def test_existing_release_without_the_file_is_refused(model_file: Path) -> None:
    gh = FakeGh({"categorization-v1": {}})

    with pytest.raises(PublishError, match=r"without model\.joblib"):
        GitHubReleases(REPO, run=gh).publish("categorization-v1", model_file, "notes")


def test_missing_gh_is_named(model_file: Path) -> None:
    def no_gh(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    with pytest.raises(PublishError, match="GitHub CLI"):
        GitHubReleases(REPO, run=no_gh).publish("categorization-v1", model_file, "notes")
