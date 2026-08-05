# =============================================================================
# Agent registry: name -> ContinualAgent subclass
# =============================================================================
# Add your agent here (one import + it self-registers via @register_agent), or
# skip the registry entirely and pass your class straight to
# framework.run_benchmark(agent=MyAgent).

from framework.interface import ContinualAgent

AGENT_REGISTRY: "dict[str, type[ContinualAgent]]" = {}


def register_agent(cls: "type[ContinualAgent]") -> "type[ContinualAgent]":
    """Class decorator: makes the agent selectable as AGENT=<cls.name>."""
    assert issubclass(cls, ContinualAgent), f"{cls} must subclass ContinualAgent"
    assert cls.name not in AGENT_REGISTRY, f"duplicate agent name {cls.name!r}"
    AGENT_REGISTRY[cls.name] = cls
    return cls


def make_agent(name: str, config: dict, tasks) -> ContinualAgent:
    name = str(name).lower()
    assert name in AGENT_REGISTRY, f"unknown AGENT {name!r} (registered: {sorted(AGENT_REGISTRY)})"
    return AGENT_REGISTRY[name](config, tasks)


# Built-in agents (importing the module registers the class).
from agents.ppo import PPOCRLAgent  # noqa: E402,F401
from agents.random_policy import RandomPolicyAgent  # noqa: E402,F401
