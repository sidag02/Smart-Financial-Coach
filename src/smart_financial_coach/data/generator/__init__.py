"""Synthetic data generator (FR-1): a pure function from a spec to a dataset.

dataset = generate(load_spec("configs/data/small.yaml"))
dataset.to_sqlite("data/synthetic/small.sqlite")
"""

from smart_financial_coach.data.generator.dataset import MODEL_TABLES, TABLES, TRUTH_TABLES, Dataset
from smart_financial_coach.data.generator.pipeline import generate
from smart_financial_coach.data.generator.spec import Spec, load_spec
from smart_financial_coach.data.generator.sqlite_io import read_sqlite

__all__ = [
    "MODEL_TABLES",
    "TABLES",
    "TRUTH_TABLES",
    "Dataset",
    "Spec",
    "generate",
    "load_spec",
    "read_sqlite",
]
