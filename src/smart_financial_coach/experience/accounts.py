"""Demo accounts and the shared demo password (Web App UI, decision 1).

A few synthetic users get a name and an email in `configs/web/demo_accounts.yaml`; the generated
dataset is unchanged. Everyone signs in with one shared password from `SFC_DEMO_PASSWORD`, which
is kept only as a salted scrypt hash. Real sign-in (v2) replaces this module and the sign-in route.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Account:
    user_id: str
    name: str
    email: str

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    @property
    def initials(self) -> str:
        return "".join(part[0] for part in self.name.split()[:2]).upper()


def load_accounts(path: Path) -> list[Account]:
    entries = yaml.safe_load(path.read_text())["accounts"]
    accounts = [Account(e["user_id"], e["name"], e["email"].lower()) for e in entries]
    for field in ("user_id", "email"):
        values = [getattr(a, field) for a in accounts]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate {field} in {path}")
    return accounts


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)


class SharedPassword:
    """Checks a candidate against the demo password without keeping the password itself."""

    def __init__(self, password: str) -> None:
        if len(password) < 8:
            raise ValueError("the demo password needs at least 8 characters")
        self._salt = os.urandom(16)
        self._hash = _scrypt(password, self._salt)

    def matches(self, candidate: str) -> bool:
        return hmac.compare_digest(_scrypt(candidate, self._salt), self._hash)
