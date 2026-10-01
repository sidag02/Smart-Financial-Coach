"""Models by name, built from `{type, params}` specs, so a new candidate is a class and a config.

Params that are themselves specs are built first, which is how wrappers take inner models:
`{"type": "routed", "params": {"seen": {"type": ...}, "unseen": {"type": ...}}}`.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from smart_financial_coach.intelligence.models.base import Model

M = TypeVar("M", bound=type[Model])

# Modules whose import registers the product's models; tests register their own
MODEL_MODULES: tuple[str, ...] = ()

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


def is_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and "type" in value and set(value) <= {"type", "params"}


def build(spec: Mapping[str, Any]) -> Model:
    params = {
        key: build(value) if is_spec(value) else value
        for key, value in dict(spec.get("params") or {}).items()
    }
    return model_class(str(spec["type"]))(**params)
