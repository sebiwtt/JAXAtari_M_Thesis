# =============================================================================
# PackNet - iterative pruning (Mallya & Lazebnik 2018)
# =============================================================================
# Each weight is tagged in an integer "owner" tree shaped like AgentParams:
#   -1 (FREE)   - unassigned capacity, trainable by the current and future tasks
#   t >= 0      - owned by task t: frozen for later tasks, kept when evaluating
#                 any task >= t
#   -2 (SHARED) - always trainable, always kept (the critic: it never drives
#                 action selection at eval, and each task needs a fresh value fn)
#
# Per task t, `train_task` runs train -> prune -> finetune (splitting the
# unchanged per-task budget by PACKNET_FINETUNE_FRAC): training masks grads to
# free/shared weights, pruning gives task t the top 1/(tasks_left+1) of free
# weights per leaf (MEAL's balanced schedule) and zeroes the rest, finetuning
# masks grads to task-t/shared weights so the zeroed free weights stay exactly
# 0 (fresh Adam + zero grads = zero moments = zero updates). Biases of
# torso+actor are owned by task 0 (trained once, then frozen), as in MEAL.
# Unlike MEAL's multi-head setup, the single-head actor kernel is pruned like
# the torso - with a shared head, a later task would otherwise overwrite
# earlier tasks' policy outputs.
#
# The owner tree makes MEAL's per-task boolean mask stack, lax.cond dispatch,
# and layer-name string matching unnecessary: all transitions run in plain
# Python between train() calls; only the (precomputed) boolean grad mask
# enters the jitted update as cl_state.
#
# Config keys: PACKNET_FINETUNE_FRAC, PACKNET_FINETUNE_LR.

import flax
import jax
import jax.numpy as jnp

from continual.base import CLMethod

PACKNET_FREE = -1
PACKNET_SHARED = -2


class PackNet(CLMethod):
    name = "packnet"

    def __init__(self, config: dict, num_tasks: int):
        super().__init__(config, num_tasks)
        self.ft_frac = float(config.get("PACKNET_FINETUNE_FRAC", 0.2))
        assert 0.0 < self.ft_frac < 1.0, "PACKNET_FINETUNE_FRAC must be in (0, 1)"
        self.ft_lr = float(config.get("PACKNET_FINETUNE_LR", config["LEARNING_RATE"]))

    def init_state(self, params):
        """cl_state (orchestrator-level) is the owner tree. Inside train() calls,
        cl_state is instead the current phase's boolean trainable mask - see
        train_task; transform_grads only ever sees the mask."""
        return _init_owner(params)

    def transform_grads(self, grads, params, cl_state, key):
        return jax.tree.map(lambda g, m: jnp.where(m, g, 0.0), grads, cl_state), {}

    def train_task(
        self,
        train_fn,
        task_config: dict,
        init_params,
        owner,
        task_idx: int,
        run_name: str,
        wandb_step_offset: int,
        **train_kwargs,
    ):
        batch_size = int(task_config["NUM_ENVS"] * task_config["NUM_STEPS"])
        num_iterations = int(task_config["TOTAL_TIMESTEPS"] // batch_size)
        ft_iters = max(1, round(num_iterations * self.ft_frac))
        train_iters = num_iterations - ft_iters
        assert train_iters >= 1, "PACKNET_FINETUNE_FRAC leaves no iterations for the train phase"

        # Phase 1/2: train on the free capacity (+ this task's weights + shared critic).
        train_cfg = dict(task_config)
        train_cfg["TOTAL_TIMESTEPS"] = train_iters * batch_size
        params = train_fn(
            train_cfg,
            init_params=init_params,
            run_name=run_name,
            wandb_step_offset=wandb_step_offset,
            cl_method=self,
            cl_state=_train_mask(owner, task_idx),
            **train_kwargs,
        )

        # Prune: task claims its equal share of free weights, the rest are zeroed.
        params, owner = _prune(params, owner, task_idx, self.num_tasks)
        print(f"[PackNet] pruned after task {task_idx}; ownership fractions: {ownership_summary(owner)}")

        # Phase 2/2: finetune only this task's weights to recover from the pruning.
        ft_cfg = dict(task_config)
        ft_cfg["TOTAL_TIMESTEPS"] = ft_iters * batch_size
        ft_cfg["LEARNING_RATE"] = self.ft_lr  # anneal restarts per phase
        params = train_fn(
            ft_cfg,
            init_params=params,
            run_name=f"{run_name}_ft",
            wandb_step_offset=wandb_step_offset + train_iters,
            cl_method=self,
            cl_state=_finetune_mask(owner, task_idx),
            **train_kwargs,
        )
        return params, owner

    def eval_params(self, params, owner, eval_task: int, trained_task: int):
        """Original-PackNet eval: a trained task runs on its recovered subnetwork
        (weights owned by tasks <= eval_task; later tasks' weights and free
        capacity zeroed). Not-yet-trained tasks (forward transfer) keep the full
        current params instead."""
        if eval_task > trained_task:
            return params
        return jax.tree.map(
            lambda p, o: jnp.where(
                ((o >= 0) & (o <= eval_task)) | (o == PACKNET_SHARED), p, jnp.zeros_like(p)
            ),
            params,
            owner,
        )

    def save_artifacts(self, owner, run_dir: str) -> None:
        """The owner tree is needed to recover per-task subnetworks from the (full)
        task_i checkpoints later; without it the masked eval params are lost."""
        owner_path = f"{run_dir}/packnet_owner.msgpack"
        with open(owner_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [owner.network_params, owner.actor_params, owner.critic_params]
                )
            )
        print(f"[PackNet] owner tree saved to {owner_path}")


def _init_owner(params):
    """Kernels free, torso/actor biases owned by task 0, critic shared."""

    def init_field(tree, bias_owner):
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: jnp.full(
                leaf.shape,
                PACKNET_FREE if "kernel" in str(path[-1]) else bias_owner,
                dtype=jnp.int32,
            ),
            tree,
        )

    return type(params)(
        network_params=init_field(params.network_params, 0),
        actor_params=init_field(params.actor_params, 0),
        critic_params=jax.tree.map(
            lambda leaf: jnp.full(leaf.shape, PACKNET_SHARED, dtype=jnp.int32), params.critic_params
        ),
    )


def _train_mask(owner, task: int):
    """Free + shared weights, plus anything already owned by this task (the
    task-0 biases during task 0)."""
    return jax.tree.map(
        lambda o: (o == PACKNET_FREE) | (o == PACKNET_SHARED) | (o == task), owner
    )


def _finetune_mask(owner, task: int):
    """Only this task's weights (+ shared)."""
    return jax.tree.map(lambda o: (o == PACKNET_SHARED) | (o == task), owner)


def _prune(params, owner, task: int, num_tasks: int):
    """Assign task `task` its share of the free weights, zero the rest.

    Per leaf, the top 1/(tasks_left+1) of free weights by magnitude become owned
    by `task`. The last task takes all remaining free weights and nothing is
    zeroed. Leaves without free entries (biases, critic) pass through untouched.
    Runs eagerly between train() calls, so plain Python control flow is fine.

    The strict `> cutoff` means ties at the cutoff are NOT claimed - notably
    weights still exactly 0 from an earlier prune (e.g. input columns of
    constant observation features never receive gradient). Those weights are
    useless to own and remain free for later tasks, so a task may claim less
    than its nominal share without losing anything.
    """
    tasks_left = num_tasks - task - 1

    def prune_leaf(p, o):
        free = o == PACKNET_FREE
        if not bool(free.any()):
            return p, o
        if tasks_left == 0:
            return p, jnp.where(free, task, o)
        prune_frac = tasks_left / (tasks_left + 1)
        cutoff = jnp.nanquantile(jnp.where(free, jnp.abs(p), jnp.nan), prune_frac)
        keep = free & (jnp.abs(p) > cutoff)
        return jnp.where(free & ~keep, 0.0, p), jnp.where(keep, task, o)

    params_leaves, treedef = jax.tree.flatten(params)
    pruned = [prune_leaf(p, o) for p, o in zip(params_leaves, jax.tree.leaves(owner))]
    return treedef.unflatten([p for p, _ in pruned]), treedef.unflatten([o for _, o in pruned])


def ownership_summary(owner) -> dict:
    """Fraction of all params per owner code, e.g. {-2: .0, -1: .6, 0: .2, ...}."""
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree.leaves(owner)])
    vals, counts = jnp.unique(flat, return_counts=True)
    return {int(v): float(c) / flat.size for v, c in zip(vals, counts)}
