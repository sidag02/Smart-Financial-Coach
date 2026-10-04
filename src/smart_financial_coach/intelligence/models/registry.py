"""Models by name, built from `{type, params}` specs, so a new candidate is a class and a config.

A param that is a model is marked explicitly, `{"$model": {type, params}}`, and built first. That
is how wrappers take inner models, and why a plain value like `method: {type: isotonic}` is never
mistaken for one:

    {"type": "routed", "params": {"seen": {"$model": {"type": ...}}, "unseen": {"$model": ...}}}
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from smart_financial_coach.intelligence.models.base import Model

M = TypeVar("M", bound=type[Model])

# Modules whose import registers the product's models; tests register their own
MODEL_MODULES: tuple[str, ...] = (
    "smart_financial_coach.intelligence.categorization.baseline",
    "smart_financial_coach.intelligence.categorization.linear",
    "smart_financial_coach.intelligence.categorization.calibration",
    "smart_financial_coach.intelligence.anomaly.baseline",
    "smart_financial_coach.intelligence.anomaly.threshold",
    "smart_financial_coach.intelligence.anomaly.rules",
    "smart_financial_coach.intelligence.anomaly.probabilistic",
    "smart_financial_coach.intelligence.anomaly.forest",
    "smart_financial_coach.intelligence.spikes.threshold",
    "smart_financial_coach.intelligence.spikes.baseline",
    "smart_financial_coach.intelligence.forecasting.baseline",
    "smart_financial_coach.intelligence.forecasting.paths",
)

_REGISTRY: dict[str, type[Model]] = {}


def register(name: str) -> Callable[[M], M]:
    def decorate(cls: M) -> M:
        if _REGISTRY.get(name, cls) is not cls:
            raise ValueError(f"model type {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def _import_models() -> None:
    for module in MODEL_MODULES:
        importlib.import_module(module)


def registered() -> list[str]:
    _import_models()
    return sorted(_REGISTRY)


def model_class(name: str) -> type[Model]:
    _import_models()
    if name not in _REGISTRY:
        raise ValueError(f"unknown model type {name!r}; registered: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


MODEL_MARKER = "$model"


def is_nested_model(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == {MODEL_MARKER}


def build(spec: Mapping[str, Any]) -> Model:
    params = {
        key: build(value[MODEL_MARKER]) if is_nested_model(value) else value
        for key, value in dict(spec.get("params") or {}).items()
    }
    return model_class(str(spec["type"]))(**params)
