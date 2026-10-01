"""The decision rule: ties against the leader only, tie-breakers within the tie set."""

from smart_financial_coach.evaluation.experiment import ExperimentConfig
from smart_financial_coach.evaluation.selection import Candidate, rank


def c(run_id: str, estimate: float, *tiebreak: float, eligible: bool = True) -> Candidate:
    return Candidate(run_id, estimate, eligible, tuple(tiebreak))


def ties(*pairs: tuple[str, str]) -> set[frozenset[str]]:
    return {frozenset(p) for p in pairs}


def test_non_transitive_ties_give_one_order() -> None:
    """A ties B and B ties C, but A beats C: C is ranked by estimate, not pulled into the tie."""
    tied = ties(("A", "B"), ("B", "C"))
    candidates = [c("A", 0.80, 0.3), c("B", 0.79, 0.1), c("C", 0.75, 0.0)]

    def rule(leader: str, other: str) -> bool:
        return frozenset((leader, other)) in tied

    orders = {tuple(rank(list(p), rule)) for p in (candidates, candidates[::-1])}

    assert orders == {("B", "A", "C")}  # B wins the A-B tie on its lower tie-breaker


def test_ineligible_runs_are_not_ranked() -> None:
    order = rank([c("A", 0.9, eligible=False), c("B", 0.8)], lambda a, b: False)

    assert order == ["B"]


def test_tie_breakers_apply_in_order() -> None:
    candidates = [c("A", 0.8, 0.2, 5.0), c("B", 0.8, 0.2, 1.0), c("C", 0.8, 0.1, 9.0)]

    assert rank(candidates, lambda a, b: True) == ["C", "B", "A"]


def test_no_candidates() -> None:
    assert rank([], lambda a, b: True) == []


def test_config_hash_ignores_tags_only() -> None:
    base = {"name": "x", "task": "toy", "model": {"type": "toy/memory"}}
    a = ExperimentConfig.model_validate(base)
    tagged = ExperimentConfig.model_validate({**base, "tags": {"round": "2"}})
    other = ExperimentConfig.model_validate({**base, "seed": 1})

    assert a.config_hash() == tagged.config_hash() != other.config_hash()


def test_grid_points_are_the_product() -> None:
    config = ExperimentConfig.model_validate(
        {"name": "x", "task": "toy", "model": {"type": "t"}, "grid": {"b": [1, 2], "a": ["x"]}}
    )

    assert config.grid_points() == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]
    assert ExperimentConfig.model_validate(
        {"name": "x", "task": "toy", "model": {"type": "t"}}
    ).grid_points() == [{}]
