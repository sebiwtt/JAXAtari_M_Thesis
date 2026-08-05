# =============================================================================
# Continual-RL method registry
# =============================================================================

from agents.ppo.continual.agem import AGEM
from agents.ppo.continual.base import CLMethod
from agents.ppo.continual.ewc import EWC
from agents.ppo.continual.ft import FT
from agents.ppo.continual.packnet import PackNet

_REGISTRY = {method.name: method for method in (FT, EWC, AGEM, PackNet)}


def make_cl_method(config: dict, num_tasks: int) -> CLMethod:
    name = str(config.get("CL_METHOD", "ft")).lower()
    assert name in _REGISTRY, f"unknown CL_METHOD {name!r} (supported: {sorted(_REGISTRY)})"
    return _REGISTRY[name](config, num_tasks)
