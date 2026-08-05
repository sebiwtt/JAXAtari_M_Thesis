# =============================================================================
# Minimal complete example agent: uniform-random policy, no learning
# =============================================================================
# The smallest thing that satisfies the ContinualAgent contract, kept as
# executable documentation (and a harness sanity check: its R matrix should sit
# at the R_rand floor, so every Retention cell comes out NaN/undefined).
#
# Run it:
#     python ppo_crl_continual.py AGENT=random sequence=pong_dyn4 TRACK=False
# or:
#     from framework import run_benchmark
#     run_benchmark(agent="random", sequence="pong_dyn4", overrides=["TRACK=False"])
# =============================================================================

import jax
import jax.numpy as jnp

from agents import register_agent
from framework.interface import ContinualAgent, TaskSpec, TrainContext


@register_agent
class RandomPolicyAgent(ContinualAgent):
    name = "random"

    def __init__(self, config: dict, tasks: "list[TaskSpec]"):
        super().__init__(config, tasks)
        # The action space is shared across all tasks of a sequence (same game).
        env = tasks[0].make_eval_env(0)
        self._n_actions = int(env.action_space().n)

    def init_state(self, key):
        # No parameters; an empty dict keeps default (msgpack) checkpointing happy.
        return {}

    def train_task(self, state, task: TaskSpec, ctx: TrainContext):
        print(f"[random] 'training' task {task.index} ({task.label}): nothing to do.")
        return state

    def policy(self, state, eval_task: int, trained_task: int):
        n_actions = self._n_actions

        def act(obs, key):
            key, subkey = jax.random.split(key)
            action = jax.random.randint(subkey, (1,), 0, n_actions)
            return action, key

        return act
