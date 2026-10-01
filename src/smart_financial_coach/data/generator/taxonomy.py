"""Fixed vocabularies shared by the generator: default categories, channels, processes, cities."""

from dataclasses import dataclass
from typing import Final

INCOME: Final = "Income"

DEFAULT_CATEGORIES: Final[tuple[str, ...]] = (
    "Housing",
    "Utilities",
    "Groceries",
    "Dining",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Subscriptions",
    "Health & Fitness",
    "Travel",
    "Childcare & Education",
    "Insurance & Fees",
    INCOME,
)

# Values real bank feeds provide. Deliberately no "recurring": it would leak the label.
CHANNELS: Final[tuple[str, ...]] = ("card_present", "online", "ach", "other")

# Which generating process produced a transaction (recorded in truth_transactions.process).
PROCESSES: Final[tuple[str, ...]] = (
    "income",
    "recurring",
    "discretionary",
    "one_off",
    "refund",
    "unusual_charge",
)

ANOMALY_KINDS: Final[tuple[str, ...]] = ("duplicate", "amount_outlier", "new_merchant_large")

PROCESSORS: Final[tuple[str, ...]] = (
    "none",
    "square",
    "toast",
    "clover",
    "paypal",
    "stripe",
    "zelle",
)


@dataclass(frozen=True)
class City:
    name: str
    state: str
    timezone: str


CITIES: Final[tuple[City, ...]] = (
    City("SAN FRANCISCO", "CA", "America/Los_Angeles"),
    City("LOS ANGELES", "CA", "America/Los_Angeles"),
    City("SAN DIEGO", "CA", "America/Los_Angeles"),
    City("SEATTLE", "WA", "America/Los_Angeles"),
    City("PORTLAND", "OR", "America/Los_Angeles"),
    City("DENVER", "CO", "America/Denver"),
    City("PHOENIX", "AZ", "America/Phoenix"),
    City("AUSTIN", "TX", "America/Chicago"),
    City("DALLAS", "TX", "America/Chicago"),
    City("CHICAGO", "IL", "America/Chicago"),
    City("MINNEAPOLIS", "MN", "America/Chicago"),
    City("ATLANTA", "GA", "America/New_York"),
    City("MIAMI", "FL", "America/New_York"),
    City("BOSTON", "MA", "America/New_York"),
    City("NEW YORK", "NY", "America/New_York"),
    City("PHILADELPHIA", "PA", "America/New_York"),
)
