"""The decision rule that orders experiment runs, fixed before the runs (FR-3 experiment plan).

1. Only eligible runs are ranked.
2. The leader is the run with the best point estimate of the selection metric.
3. Ties are defined against the leader only, because pairwise ties aren't transitive: the tie set
   is every run statistically tied with the leader, the leader included.
4. The tie set is ordered by tie-breakers (lower is better), then the remaining runs by point
   estimate. Finalists are the first three.
5. Optionally, the first tie-breaker gets its own tie test, judged against its own leader (the
   tie set's lowest value), so a later tie-breaker such as cost can decide between runs tied on
   both (FR-4 §5). Runs tied with that leader come first, ordered by the remaining tie-breakers;
   the rest of the tie set follows by the first tie-breaker.
"""

from collections.abc import Callable
from dataclasses import dataclass

MAX_FINALISTS = 3


@dataclass(frozen=True)
class Candidate:
    run_id: str
    estimate: float  # selection metric on validation; higher is better
    eligible: bool
    tiebreak: tuple[float, ...]  # e.g. calibration error, p95 latency, complexity


def rank(
    candidates: list[Candidate],
    tied_with_leader: Callable[[str, str], bool],
    tiebreak_tied: Callable[[str, str], bool] | None = None,
) -> list[str]:
    """Run IDs in decision-rule order; `tied_with_leader(leader, other)` decides ties, and
    `tiebreak_tied(leader, other)`, if given, ties on the first tie-breaker (rule 5)."""
    by_estimate = sorted(
        (c for c in candidates if c.eligible), key=lambda c: (-c.estimate, c.run_id)
    )
    if not by_estimate:
        return []
    leader = by_estimate[0]
    tie = [c for c in by_estimate if c is leader or tied_with_leader(leader.run_id, c.run_id)]
    rest = [c for c in by_estimate if c not in tie]
    tie.sort(key=lambda c: (c.tiebreak, -c.estimate, c.run_id))
    if tiebreak_tied is not None and len(tie) > 1:
        first = tie[0]  # the lowest first tie-breaker (then the rest, so it's deterministic)
        level = [c for c in tie if c is first or tiebreak_tied(first.run_id, c.run_id)]
        level.sort(key=lambda c: (c.tiebreak[1:], -c.estimate, c.run_id))
        tie = level + [c for c in tie if c not in level]
    return [c.run_id for c in tie + rest]
