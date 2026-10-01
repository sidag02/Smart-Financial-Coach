"""The decision rule that orders experiment runs, fixed before the runs (FR-3 experiment plan).

1. Only eligible runs are ranked.
2. The leader is the run with the best point estimate of the selection metric.
3. Ties are defined against the leader only, because pairwise ties aren't transitive: the tie set
   is every run statistically tied with the leader, the leader included.
4. The tie set is ordered by tie-breakers (lower is better), then the remaining runs by point
   estimate. Finalists are the first three.
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


def rank(candidates: list[Candidate], tied_with_leader: Callable[[str, str], bool]) -> list[str]:
    """Run IDs in decision-rule order; `tied_with_leader(leader, other)` decides ties."""
    by_estimate = sorted(
        (c for c in candidates if c.eligible), key=lambda c: (-c.estimate, c.run_id)
    )
    if not by_estimate:
        return []
    leader = by_estimate[0]
    tie = [c for c in by_estimate if c is leader or tied_with_leader(leader.run_id, c.run_id)]
    rest = [c for c in by_estimate if c not in tie]
    tie.sort(key=lambda c: (c.tiebreak, -c.estimate, c.run_id))
    return [c.run_id for c in tie + rest]
