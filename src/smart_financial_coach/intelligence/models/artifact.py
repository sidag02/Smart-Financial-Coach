"""Model artifacts on disk: a fitted model, its manifest, and the promotion pointer and log.

    artifacts/<service>/<version>/model.joblib   the fitted model
    artifacts/<service>/<version>/manifest.json  what produced it, and the model file's checksum
    artifacts/<service>/PROMOTED                 the version callers get
    artifacts/<service>/promotions.jsonl         every promotion, newest last

joblib files are pickles, so models are loaded only from a folder the caller names as trusted,
and only if the file matches the checksum in its manifest.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import joblib

from smart_financial_coach.intelligence.models.base import Model, describe

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"
POINTER_FILE = "PROMOTED"
LOG_FILE = "promotions.jsonl"


class ArtifactError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_artifact(model: Model, directory: Path, manifest: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE)
    full = {
        **manifest,
        "name": model.name,
        "version": model.version,
        "params": describe(dict(model.params)),
        "model_sha256": sha256(directory / MODEL_FILE),
    }
    (directory / MANIFEST_FILE).write_text(
        json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.exists():
        raise ArtifactError(f"{directory} has no {MANIFEST_FILE}")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_artifact(directory: Path, *, trusted_root: Path) -> Model:
    directory = directory.resolve()
    if not directory.is_relative_to(trusted_root.resolve()):
        raise ArtifactError(f"refusing to load {directory}: not under {trusted_root}")
    manifest = read_manifest(directory)
    if sha256(directory / MODEL_FILE) != manifest["model_sha256"]:
        raise ArtifactError(f"{directory / MODEL_FILE} doesn't match its manifest checksum")
    model = joblib.load(directory / MODEL_FILE)
    if not isinstance(model, Model):
        raise ArtifactError(f"{directory / MODEL_FILE} is not a model")
    if (model.name, model.version) != (manifest["name"], manifest["version"]):
        raise ArtifactError(
            f"{directory}: model is {model.name} {model.version}, "
            f"manifest says {manifest['name']} {manifest['version']}"
        )
    return model


def promoted_version(service_dir: Path) -> str:
    pointer = service_dir / POINTER_FILE
    if not pointer.exists():
        raise ArtifactError(f"no promoted model: {pointer} doesn't exist")
    return pointer.read_text(encoding="utf-8").strip()


def record_promotion(service_dir: Path, entry: Mapping[str, Any]) -> None:
    """Point `PROMOTED` at `entry["version"]` and append the entry to the promotion log."""
    version = str(entry["version"])
    if not (service_dir / version / MANIFEST_FILE).exists():
        raise ArtifactError(f"{service_dir / version} is not an exported artifact")
    with (service_dir / LOG_FILE).open("a", encoding="utf-8") as log:
        log.write(json.dumps(dict(entry), sort_keys=True) + "\n")
    (service_dir / POINTER_FILE).write_text(version + "\n", encoding="utf-8")


def promotions(service_dir: Path) -> list[dict[str, Any]]:
    path = service_dir / LOG_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def promotion_errors(service_dir: Path) -> list[str]:
    """`PROMOTED` must name the log's last entry, and that version must be exported."""
    log = promotions(service_dir)
    pointer = service_dir / POINTER_FILE
    if not pointer.exists():
        return [f"{service_dir}: promotions logged but no {POINTER_FILE}"] if log else []
    version = promoted_version(service_dir)
    errors = []
    if not log or log[-1]["version"] != version:
        errors.append(f"{service_dir}: {POINTER_FILE} is {version}, not the last logged promotion")
    if not (service_dir / version / MANIFEST_FILE).exists():
        errors.append(f"{service_dir}: promoted version {version} is not exported")
    return errors
