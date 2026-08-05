# =============================================================================
# CRL benchmark framework - public API
# =============================================================================
# run_benchmark is the single call that runs the whole benchmark; the rest is
# the contract agents are written against. See FRAMEWORK.md.

from framework.evaluation import evaluate_policy
from framework.interface import ContinualAgent, TaskSpec, TrainContext
from framework.runner import build_tasks, run_benchmark, run_from_config

__all__ = [
    "ContinualAgent",
    "TaskSpec",
    "TrainContext",
    "evaluate_policy",
    "build_tasks",
    "run_benchmark",
    "run_from_config",
]
