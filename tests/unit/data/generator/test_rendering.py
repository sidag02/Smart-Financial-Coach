import numpy as np
import pandas as pd
import pytest

from smart_financial_coach.data.generator.catalog import Catalog, load_catalog
from smart_financial_coach.data.generator.rendering import PROFILES, Renderer
from smart_financial_coach.data.generator.spec import Spec
from smart_financial_coach.data.generator.taxonomy import CITIES


@pytest.fixture(scope="module")
def catalog(small_spec: Spec) -> Catalog:
    return load_catalog(
        small_spec.catalog.merchants, small_spec.catalog.holdout, small_spec.categories
    )


def _txns(merchant_id: str, channel: str, n: int, process: str = "discretionary") -> pd.DataFrame:
    return pd.DataFrame({"merchant_id": [merchant_id] * n, "channel": channel, "process": process})


def test_none_returns_canonical_names(catalog: Catalog) -> None:
    renderer = Renderer(catalog, "none", seed=1)
    txns = pd.concat([_txns("m_trader_joe_s", "card_present", 3), _txns("m_netflix", "online", 2)])

    out = renderer.render_user(txns, CITIES[0], np.random.default_rng(0))

    assert out == ["Trader Joe's"] * 3 + ["Netflix"] * 2


def test_realistic_text_is_messy_but_bounded(catalog: Catalog) -> None:
    renderer = Renderer(catalog, "realistic", seed=1)
    txns = _txns("m_morning_ritual_cafe", "card_present", 400)

    out = renderer.render_user(txns, CITIES[0], np.random.default_rng(0))

    assert max(len(s) for s in out) <= PROFILES["realistic"].limit[1]
    assert len(set(out)) >= 2  # several house styles and per-transaction noise
    assert any("SQ *" in s for s in out)  # Square merchant
    assert all(s != "Morning Ritual Cafe" for s in out)


def test_house_styles_are_shared_across_users(catalog: Catalog) -> None:
    renderer = Renderer(catalog, "realistic", seed=1)
    txns = _txns("m_safeway", "card_present", 300)

    first = renderer.render_user(txns, CITIES[0], np.random.default_rng(1))
    second = Renderer(catalog, "realistic", seed=1).render_user(
        txns, CITIES[0], np.random.default_rng(1)
    )

    assert first == second


def test_payroll_and_subscriptions_render_as_bank_feeds_do(catalog: Catalog) -> None:
    renderer = Renderer(catalog, "realistic", seed=1)

    payroll = renderer.render_user(
        _txns("m_acme_corp", "ach", 20), CITIES[0], np.random.default_rng(0)
    )
    netflix = renderer.render_user(
        _txns("m_netflix", "online", 20, process="recurring"), CITIES[0], np.random.default_rng(0)
    )

    assert all("ACME CORP" in " ".join(s.split()) for s in payroll)  # allow doubled spaces
    assert all(s.startswith(("NETFLIX", "Netflix")) and "*" not in s for s in netflix)
