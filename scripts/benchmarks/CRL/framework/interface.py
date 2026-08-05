# =============================================================================
# The benchmark contract: what YOU implement to plug an agent into the benchmark
# =============================================================================
# The benchmark owns everything that must be identical across submissions for
# results to be comparable: the task sequences (env + mods), the per-task
# training budgets, the evaluation protocol, and the metric definitions
# (R / R_rand / Retention / Drop / Forgetting). You own everything about
# *learning*: the algorithm (PPO, PQN, DQN, world models, ...), the networks,
# the optimizer, and any continual-learning machinery.
#
# To submit an agent, subclass `ContinualAgent`, implement the three required
# methods (`init_state`, `train_task`, `policy`), register it, and run:
#
#     from framework import run_benchmark
#     from agents.my_agent import MyAgent
#
#     result = run_benchmark(agent=MyAgent, sequence="pong_dyn4", modality="oc",
#                            overrides=["SEED=1"])
#
# See agents/random_policy.py for the smallest complete working example and
# agents/ppo/ for the full-featured reference implementation (PPO with
# ft/ewc/agem/packnet). FRAMEWORK.md walks through the whole thing.
#
# Ground rules (enforced socially, not by code -- the harness cannot meter the
# env steps your training loop takes):
#   * `train_task` must not exceed `task.budget` environment steps (frame-
#     skipped steps, i.e. the same unit the reference PPO counts).
#   * Only `task.make_train_env` envs during training; the harness alone
#     produces the official eval numbers via your `policy`.
#   * Everything called inside evaluation (`policy`'s returned act fn) must be
#     jit-traceable JAX.
# =============================================================================

from dataclasses import dataclass
from typing import Any, Callable

import flax.serialization


@dataclass(frozen=True)
class TaskSpec:
    """One task of the continual sequence. Built by the runner from the config;
    handed to the agent -- agents never parse TASK_MODS/budgets themselves."""

    index: int                     # position in the sequence (0 = base task, no mods)
    label: str                     # short human-readable name ("base" or the mod key)
    mods: "tuple[str, ...]"        # jaxatari mod keys applied to the base game (<= 1)
    budget: int                    # env-step training budget for this task (upper bound)
    env_id: str                    # the jaxatari game
    # (seed, num_envs) -> wrapped TRAINING env (episodic life, clipped reward).
    # The returned env exposes reset/step for vmapping, plus
    # observation_space()/action_space(); see framework/envs.py for the wrapper stack.
    make_train_env: Callable[[int, int], Any]
    # (seed,) -> single wrapped EVAL env (true episode boundaries, raw reward).
    # Exposed for the agent's own mid-training probes; official numbers come
    # only from the harness's evaluation of `policy`.
    make_eval_env: Callable[[int], Any]


@dataclass(frozen=True)
class TrainContext:
    """Bookkeeping the runner passes into each train_task call."""

    run_name: str        # unique name for this task's training segment (logging)
    run_dir: str         # the run's output directory (write agent-specific files here)
    env_step_offset: int  # sum of all previous tasks' budgets (x-axis offset for logging)
    track: bool          # whether wandb tracking is on for this run


class ContinualAgent:
    """Base class every benchmark submission implements.

    The runner drives it like this (see framework/runner.py):

        agent       = AgentCls(config, tasks)
        floor_state = agent.init_state(PRNGKey(EVAL_SEED))     # untrained floor
        R_rand[j]  <- evaluate(agent.policy(floor_state, j, trained_task=-1))
        state       = agent.init_state(PRNGKey(SEED))
        for task in tasks:
            state = agent.train_task(state, task, ctx)
            agent.save_checkpoint(state, run_dir, f"task_{task.index}")
            R[i, j] <- evaluate(agent.policy(state, j, trained_task=task.index))
        agent.save_artifacts(state, run_dir)

    `state` is opaque to the runner: any pytree (params, optimizer state, CL
    buffers, replay memory, ...). It is threaded through unchanged, so all
    cross-task memory must live in it (or in `self`, for host-side bookkeeping).
    """

    #: Registry name; override in your subclass (used as AGENT=<name> in config).
    name = "agent"

    def __init__(self, config: dict, tasks: "list[TaskSpec]"):
        """`config` is the flat UPPER_CASE hydra config dict; `tasks` the full
        ordered sequence, known upfront (budgets included) -- agents may size
        buffers or precompute schedules from it."""
        self.config = config
        self.tasks = tasks

    # ---- required ------------------------------------------------------------

    def init_state(self, key) -> Any:
        """Return a freshly-initialized (untrained) agent state from a PRNG key.

        Called twice per run with different keys: once to build the untrained
        random-floor agent for R_rand (EVAL_SEED), once for the state that
        training actually starts from (SEED). Must be deterministic in `key`.
        """
        raise NotImplementedError

    def train_task(self, state, task: TaskSpec, ctx: TrainContext) -> Any:
        """Train on one task (<= task.budget env steps) and return the new state.

        Runs as plain Python (jit whatever you like inside). Carrying knowledge
        forward vs. protecting old tasks is entirely your policy -- the harness
        only measures the consequences.
        """
        raise NotImplementedError

    def policy(self, state, eval_task: int, trained_task: int) -> Callable:
        """Return the jit-safe act function evaluation should use.

        act(obs, key) -> (action, key), where obs has a leading batch axis of
        size 1 (shape (1, *obs_shape)), action has shape (1,) (int), and key is
        the per-episode PRNG key: split it internally if you sample, and return
        the advanced key. The harness vmaps `act` over EVAL_EPISODES episodes.

        `eval_task` is the task being evaluated; `trained_task` the last task
        trained on (-1 for the untrained floor agent). Most agents ignore both;
        methods with per-task sub-policies (e.g. PackNet subnetworks) use them
        to select what to run.
        """
        raise NotImplementedError

    # ---- optional ------------------------------------------------------------

    def save_checkpoint(self, state, run_dir: str, name: str) -> str:
        """Persist `state` after a task; returns the path written. Default:
        flax msgpack serialization of the raw state pytree."""
        path = f"{run_dir}/{name}.ckpt"
        with open(path, "wb") as f:
            f.write(flax.serialization.to_bytes(state))
        return path

    def save_artifacts(self, state, run_dir: str) -> None:
        """Persist anything else needed to reproduce per-task eval policies
        later (e.g. PackNet's owner tree)."""

    def collect_curve_points(self) -> "list[dict]":
        """Mid-training curve points to persist with the matrix (see CRL_CURVE
        in config.yaml); [] if the agent doesn't produce any."""
        return []

    def describe(self) -> dict:
        """Extra key/values merged into matrix.json (e.g. {"cl_method": "ewc"})."""
        return {}
