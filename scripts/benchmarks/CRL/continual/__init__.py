# =============================================================================
# Continual-RL method registry
# =============================================================================
# One file per method (MEAL-style); `make_cl_method` is the only entry point
# the orchestrator needs. See continual/base.py for the interface.

from continual.agem import AGEM
from continual.base import CLMethod
from continual.ewc import EWC
from continual.ft import FT
from continual.packnet import PackNet

_REGISTRY = {method.name: method for method in (FT, EWC, AGEM, PackNet)}


def make_cl_method(config: dict, num_tasks: int) -> CLMethod:
    name = str(config.get("CL_METHOD", "ft")).lower()
    assert name in _REGISTRY, f"unknown CL_METHOD {name!r} (supported: {sorted(_REGISTRY)})"
    return _REGISTRY[name](config, num_tasks)
