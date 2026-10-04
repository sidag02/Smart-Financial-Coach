"""Callers get a service's promoted model here, never a model class.

    categorizer = load_service("categorization")
    categorizer.predict(transactions)  # contract-checked

Promoting a different run changes what this returns; no calling code changes. Serving reads only
files under `artifacts/`, never the experiment tracker. A promoted model file that isn't there yet
(a fresh clone) is downloaded once from the URL in the promotion log and checked against the
committed manifest before it is loaded.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path

from smart_financial_coach.config import get_settings
from smart_financial_coach.intelligence.models.artifact import (
    MODEL_FILE,
    fetch_model,
    load_artifact,
    model_url,
    promoted_version,
)
from smart_financial_coach.intelligence.models.contract import Checked, Contract

# Modules whose import registers the product's services; tests register their own
SERVICE_MODULES: tuple[str, ...] = (
    "smart_financial_coach.intelligence.categorization.contract",
    "smart_financial_coach.intelligence.anomaly.contract",
    "smart_financial_coach.intelligence.forecasting.contract",
    "smart_financial_coach.intelligence.spikes.contract",
)


@dataclass(frozen=True)
class Service:
    name: str
    contract: Contract
    wrapper: type[Checked] = Checked  # what callers get: adds the service's own verbs


_SERVICES: dict[str, Service] = {}


def register_service(contract: Contract, wrapper: type[Checked] = Checked) -> Service:
    service = Service(contract.service, contract, wrapper)
    if _SERVICES.get(service.name, service) != service:
        raise ValueError(f"service {service.name!r} is already registered")
    _SERVICES[service.name] = service
    return service


def get_service(name: str) -> Service:
    for module in SERVICE_MODULES:
        importlib.import_module(module)
    if name not in _SERVICES:
        raise ValueError(f"unknown service {name!r}; registered: {', '.join(sorted(_SERVICES))}")
    return _SERVICES[name]


def load_service(name: str, artifacts_dir: Path | None = None) -> Checked:
    service = get_service(name)
    root = (artifacts_dir or get_settings().artifacts_dir) / name
    version = promoted_version(root)
    if not (root / version / MODEL_FILE).exists():
        fetch_model(root / version, model_url(root, version))
    model = load_artifact(root / version, trusted_root=root)
    return service.wrapper(model, service.contract)
