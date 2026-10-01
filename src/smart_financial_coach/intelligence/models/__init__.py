"""Generic model interface, registry, contracts and artifacts, shared by every service."""

from smart_financial_coach.intelligence.models.base import UNVERSIONED, BaseModel, Model
from smart_financial_coach.intelligence.models.contract import Checked, Contract, ContractError
from smart_financial_coach.intelligence.models.registry import build, register, registered

__all__ = [
    "UNVERSIONED",
    "BaseModel",
    "Checked",
    "Contract",
    "ContractError",
    "Model",
    "build",
    "register",
    "registered",
]
