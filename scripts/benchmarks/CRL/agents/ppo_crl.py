# =============================================================================
# Reference agent: single-head PPO + pluggable continual-learning method
# =============================================================================
# The benchmark's built-in submission - a thin ContinualAgent adapter around
# the pre-framework stack, kept behaviorally identical to it:
#   * ppo_trainer.train      - the jitted single-task PPO loop
#   * continual/ (CLMethod)  - ft / ewc / agem / packnet, selected by CL_METHOD
#   * networks.py            - CNN/MLP torso + Actor/Critic heads
# Use it as the template for how a full-featured agent maps onto the contract:
# state = (params, cl_state), policy() = the CL method's eval-params substitution
# (identity for most methods, subnetwork masking for PackNet).
# =============================================================================

from typing import NamedTuple

import flax
import jax
import jax.numpy as jnp
import numpy as np

from agents import register_agent
from continual import make_cl_method
from envs import make_env
from framework.interface import ContinualAgent, TaskSpec, TrainContext
from networks import Actor, AgentParams, Critic, MLP_Network, Network, action_dim_from_params, build_models
from ppo_trainer import train


class PPOState(NamedTuple):
    """params carried across tasks + the CL method's device state (Fisher,
    memory, owner tree, ...); cl_state is None for methods without one."""
    params: AgentParams
    cl_state: object


@register_agent
class PPOCRLAgent(ContinualAgent):
    name = "ppo"

    def __init__(self, config: dict, tasks: "list[TaskSpec]"):
        super().__init__(config, tasks)
        # Validates CL_METHOD and its config keys before any training starts.
        self.method = make_cl_method(config, len(tasks))
        batch_size = int(config["NUM_ENVS"] * config["NUM_STEPS"])
        for t in tasks:
            # train() needs NUM_ITERATIONS >= 1 (RTPT requires max_iterations > 0).
            assert t.budget // batch_size >= 1, (
                f"task {t.index} budget {t.budget:,} is below one batch ({batch_size:,} steps): "
                f"no PPO iteration would run"
            )
        # wandb x-axis offsets in PPO iterations: tasks may have different budgets,
        # so each task starts where the previous one ended.
        iters_per_task = [t.budget // batch_size for t in tasks]
        self._iter_offsets = [int(o) for o in np.cumsum([0] + iters_per_task[:-1])]
        # One shared list collecting every crl_curve point across all tasks (see
        # config.yaml: CRL_CURVE); train() appends in memory only, the runner
        # persists it once with the matrix.
        self._curve_points = [] if (config["TRACK"] and config.get("CRL_CURVE", False)) else None
        self._last_task_config = None

    # ---- ContinualAgent contract ---------------------------------------------

    def init_state(self, key) -> PPOState:
        """Freshly-initialized params; mirrors train()'s fresh-init branch key-split
        for key-split, so init_state(PRNGKey(SEED)) is bit-identical to the params
        train() would build itself."""
        config = self.config
        env = make_env(
            config["ENV_ID"], config["SEED"], 1, [], config["PIXEL_BASED"],
            config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"], config["GRAYSCALE"],
        )()
        network = Network() if config["PIXEL_BASED"] else MLP_Network()
        actor = Actor(action_dim=env.action_space().n)
        critic = Critic()

        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        key, obs_key1, obs_key2, obs_key3 = jax.random.split(key, 4)
        network_params = network.init(network_key, env.observation_space().sample(obs_key1).squeeze()[None, ...])
        params = AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key2).squeeze()]))),
            critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_key3).squeeze()]))),
        )
        # CL state is built from param shapes only, so it's identical no matter
        # which of the two init_state calls (floor / training) constructs it.
        return PPOState(params=params, cl_state=self.method.init_state(params))

    def train_task(self, state: PPOState, task: TaskSpec, ctx: TrainContext) -> PPOState:
        task_config = dict(self.config)
        task_config["TRAIN_MODS"] = tuple(task.mods)
        task_config["TOTAL_TIMESTEPS"] = task.budget  # drives NUM_ITERATIONS + LR anneal inside train()

        extra_kwargs = {}
        if self._curve_points is not None:
            task_config["CRL_CURVE_TASK_IDX"] = task.index  # segment indicator on every curve point
            # Curve evals score the same params the R matrix would for the base task:
            # identity for most methods, PackNet's task-0 subnetwork mask. Bound to the
            # pre-task cl_state, which is correct throughout this task - PackNet's
            # mid-task prune only reassigns FREE weights, never task-0-owned ones.
            extra_kwargs = {
                "crl_curve_sink": self._curve_points,
                "crl_curve_param_fn": lambda p, _s=state.cl_state, _i=task.index: self.method.eval_params(p, _s, 0, _i),
            }

        # Task 0 passes init_params=None so train() runs its own fresh init - the
        # exact pre-framework behavior (init_state's params are identical anyway).
        params, cl_state = self.method.train_task(
            train,
            task_config,
            state.params if task.index > 0 else None,
            state.cl_state,
            task.index,
            run_name=ctx.run_name,
            wandb_step_offset=self._iter_offsets[task.index],
            manage_wandb=False,
            wandb_group=task.label,
            **extra_kwargs,
        )
        self._last_task_config = task_config
        return PPOState(params=params, cl_state=cl_state)

    def policy(self, state: PPOState, eval_task: int, trained_task: int):
        # Methods may substitute task-specific eval params (e.g. PackNet recovers
        # eval_task's subnetwork); identity for ft/ewc/agem and the floor agent.
        params = self.method.eval_params(state.params, state.cl_state, eval_task, trained_task)
        network, actor, _ = build_models(self.config, action_dim_from_params(params))

        def act(obs, key):
            hidden = network.apply(params.network_params, obs)
            logits = actor.apply(params.actor_params, hidden)
            # Gumbel-max trick for categorical sampling (same sampler as training).
            key, subkey = jax.random.split(key)
            u = jax.random.uniform(subkey, shape=logits.shape)
            action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
            return action, key

        return act

    # ---- persistence ---------------------------------------------------------

    def save_checkpoint(self, state: PPOState, run_dir: str, name: str) -> str:
        """cleanrl format ([config, [network, actor, critic]]), readable by
        ppo_eval.evaluate and the video/visualization tools."""
        path = f"{run_dir}/{name}.cleanrl_model"
        saved_config = self._last_task_config if self._last_task_config is not None else self.config
        with open(path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [saved_config, [state.params.network_params, state.params.actor_params, state.params.critic_params]]
                )
            )
        return path

    def save_artifacts(self, state: PPOState, run_dir: str) -> None:
        self.method.save_artifacts(state.cl_state, run_dir)

    def collect_curve_points(self) -> "list[dict]":
        return self._curve_points or []

    def describe(self) -> dict:
        return {"cl_method": self.method.name}
