"""Callers get a service's promoted model here, never a model class.

    categorizer = load_service("categorization")
    categorizer.predict(transactions)  # contract-checked

Promoting a different run changes what this returns; no calling code changes. Serving reads only
files under `artifacts/`, never the experiment tracker.
"""

import importlib
from dataclasses import dataclass
from pathlib import Path

from smart_financial_coach.config import get_settings
from smart_financial_coach.intelligence.models.artifact import load_artifact, promoted_version
from smart_financial_coach.intelligence.models.contract import Checked, Contract

# Modules whose import registers the product's services; tests register their own
SERVICE_MODULES: tuple[str, ...] = ()


@dataclass(frozen=True)
class Service:
    name: str
    contract: Contract


_SERVICES: dict[str, Service] = {}


def register_service(contract: Contract) -> Service:
    service = Service(contract.service, contract)
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
    model = load_artifact(root / promoted_version(root), trusted_root=root)
    return Checked(model, service.contract)
