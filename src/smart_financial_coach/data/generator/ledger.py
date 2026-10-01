"""Per-user accumulator of generated transactions, with stable transaction IDs."""

import hashlib
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import pandas as pd

LEDGER_COLUMNS = (
    "transaction_id",
    "day",
    "minute",
    "amount",
    "merchant_id",
    "category",
    "channel",
    "process",
    "is_recurring",
    "anomaly_kind",
    "copy_of",  # transaction whose rendered text this one reuses (refunds, duplicates)
)


def transaction_id(user_id: str, process: str, counter: int) -> str:
    """Stable per (user, process, counter), so adding events never renumbers other transactions."""
    return "t_" + hashlib.sha1(f"{user_id}|{process}|{counter}".encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._blocks: dict[str, list[pd.DataFrame]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)

    def add(
        self,
        process: str,
        *,
        day: npt.ArrayLike,
        minute: npt.ArrayLike,
        amount: npt.ArrayLike,
        merchant_id: npt.ArrayLike,
        category: npt.ArrayLike,
        channel: npt.ArrayLike,
        is_recurring: bool = False,
        anomaly_kind: str | None = None,
        copy_of: npt.ArrayLike | None = None,
    ) -> None:
        days = np.asarray(day, dtype=np.int64)
        n = len(days)
        if n == 0:
            return
        start = self._counters[process]
        self._counters[process] = start + n
        block = pd.DataFrame(
            {
                "transaction_id": [
                    transaction_id(self.user_id, process, start + i) for i in range(n)
                ],
                "day": days,
                "minute": np.asarray(minute, dtype=np.int64),
                "amount": np.round(np.asarray(amount, dtype=np.float64), 2),
                "merchant_id": np.asarray(merchant_id, dtype=object),
                "category": np.asarray(category, dtype=object),
                "channel": np.asarray(channel, dtype=object),
                "process": process,
                "is_recurring": is_recurring,
                "anomaly_kind": anomaly_kind,
                "copy_of": np.asarray(copy_of, dtype=object) if copy_of is not None else None,
            },
            columns=list(LEDGER_COLUMNS),
        )
        self._blocks[process].append(block)

    def clear(self, process: str) -> None:
        self._blocks.pop(process, None)
        self._counters.pop(process, None)

    def frame(self, processes: tuple[str, ...] | None = None) -> pd.DataFrame:
        keys = processes if processes is not None else tuple(self._blocks)
        blocks = [b for k in keys for b in self._blocks.get(k, [])]
        if not blocks:
            return pd.DataFrame({c: [] for c in LEDGER_COLUMNS})
        return pd.concat(blocks, ignore_index=True)
