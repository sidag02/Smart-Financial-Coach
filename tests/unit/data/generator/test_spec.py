from pathlib import Path

import pytest
from pydantic import ValidationError

from smart_financial_coach.data.generator.spec import Spec, StreamSpec, load_spec, spec_hash


def test_default_spec_loads_personas_and_catalog(configs: Path) -> None:
    spec = load_spec(configs / "default.yaml")

    assert set(spec.personas) == {"young_professional", "family_budgeter", "freelancer"}
    assert spec.catalog.merchants.is_absolute()
    assert spec.catalog.merchants.exists()
    assert sum(sum(p.users.values()) for p in spec.populations) == 300


def test_extends_overrides_only_what_it_sets(configs: Path) -> None:
    default = load_spec(configs / "default.yaml")
    small = load_spec(configs / "small.yaml")
    clean = load_spec(configs / "clean.yaml")

    assert small.name == "small"
    assert small.calendar.start != default.calendar.start
    assert small.rendering == default.rendering
    assert clean.rendering.distortion == "none"
    assert clean.events.unusual_charge.rate_per_user_year == 0
    # Nested mappings merge: untouched keys keep the parent's values
    assert clean.events.refunds == default.events.refunds


def _write_override(tmp_path: Path, configs: Path, body: str) -> Path:
    path = tmp_path / "override.yaml"
    path.write_text(f"extends: {configs / 'default.yaml'}\n{body}", encoding="utf-8")
    return path


def test_unknown_persona_is_rejected(tmp_path: Path, configs: Path) -> None:
    path = _write_override(
        tmp_path,
        configs,
        "populations:\n"
        "  - {name: train, split: train, seed: 1, id_prefix: tr, users: {astronaut: 3}}\n",
    )
    with pytest.raises(ValidationError, match="unknown persona 'astronaut'"):
        load_spec(path)


def test_unknown_category_is_rejected(tmp_path: Path, configs: Path) -> None:
    path = _write_override(tmp_path, configs, "events:\n  refunds: {categories: [Yachts]}\n")
    with pytest.raises(ValidationError, match="unknown category 'Yachts'"):
        load_spec(path)


def test_duplicate_persona_names_are_rejected(tmp_path: Path, configs: Path) -> None:
    personas = tmp_path / "personas"
    personas.mkdir()
    for copy in ("a.yaml", "b.yaml"):
        (personas / copy).write_text(
            (configs / "personas" / "freelancer.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )
    path = _write_override(tmp_path, configs, f"personas: {personas}\n")

    with pytest.raises(ValueError, match="duplicate persona name 'freelancer'"):
        load_spec(path)


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="range low"):
        StreamSpec(category="Dining", weekly_rate=(3.0, 1.0))


def test_spec_hash_tracks_parameters(small_spec: Spec) -> None:
    changed = small_spec.model_copy(
        update={"rendering": small_spec.rendering.model_copy(update={"seed": 99})}
    )

    assert spec_hash(small_spec) == spec_hash(small_spec)
    assert spec_hash(changed) != spec_hash(small_spec)
