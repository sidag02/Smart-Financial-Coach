"""Evaluation tasks: one per learning problem, each owning its examples, splits and metrics."""

from smart_financial_coach.evaluation.tasks.base import (
    Examples,
    Gate,
    Task,
    get_task,
    latency_ms,
    register_task,
)

__all__ = ["Examples", "Gate", "Task", "get_task", "latency_ms", "register_task"]
