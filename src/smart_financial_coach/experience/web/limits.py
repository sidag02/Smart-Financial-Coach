"""In-memory sliding-window rate limits. One replica serves the demo, so memory is enough."""

import threading
import time
from collections import OrderedDict, deque


class RateLimit:
    def __init__(self, limit: int, window_seconds: float, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for `key` and say whether it's within the limit."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.pop(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            allowed = len(hits) < self.limit
            if allowed:
                hits.append(now)
            self._hits[key] = hits
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
            return allowed
