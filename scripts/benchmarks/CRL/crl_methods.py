# =============================================================================
# Continual-RL methods for the JAXtari PPO trainer
# =============================================================================
# Everything method-specific (state containers, losses, gradient surgery) lives
# here; `ppo_trainer.train` only calls the thin hooks:
#
#   EWC  (Kirkpatrick et al. 2017)  - `ewc_penalty` inside `ppo_loss`;
#        `make_fisher_fn(...)` estimates the diagonal Fisher post-task;
#        `ewc_update_state` merges it across tasks (last/multi/online).
#   A-GEM (Chaudhry et al. 2019)    - `agem_project` on the raw PPO grads in
#        `update_minibatch`, against a behavioral-cloning gradient on an
#        episodic memory of past-task transitions (`make_agem_grad_fn`);
#        memory blocks are sampled post-task via `agem_sample_block` and
#        concatenated with `agem_extend_memory`.
#   PackNet (Mallya & Lazebnik 2018) - `grad_mask` in `update_minibatch`
#        (built by `packnet_train_mask` / `packnet_finetune_mask` over an
#        integer "owner" tree); `packnet_prune` assigns weights after each
#        task's train phase, `packnet_eval_params` recovers a task's
#        subnetwork at evaluation time.
#
# All are ported from MEAL's IPPO implementation (github.com/TTomilin/MEAL)
# and adapted to this trainer's one-task-per-`train()`-call structure: per-task
# CL state transitions happen in plain Python in the orchestrator, instead of
# MEAL's lax.cond-heavy in-jit updates.
# =============================================================================

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


# =============================================================================
# EWC - Elastic Weight Consolidation
# =============================================================================
# Diagonal-Fisher EWC. F = E_{s, a~pi}[(d/dtheta log pi(a|s))^2] is estimated
# from on-policy samples of the finished task's final policy. The critic head
# gets zero Fisher (log pi does not depend on it), so only torso + actor are
# anchored; the value function stays free to re-adapt on each new task.

@flax.struct.dataclass
class EWCState:
    """Anchor params theta* and diagonal Fisher F accumulated over finished tasks."""
    old_params: "AgentParams"  # noqa: F821 - AgentParams lives in ppo_trainer; pytree-only use here
    fisher: "AgentParams"  # noqa: F821


def ewc_update_state(
    ewc_state: "EWCState | None",
    new_params,
    new_fisher,
    mode: str = "online",
    decay: float = 0.9,
) -> EWCState:
    """Merge a freshly estimated Fisher into the running EWC state after finishing a task.

    Modes (as in MEAL):
      "last"   - keep only the newest task's Fisher
      "multi"  - sum of all tasks' Fishers (standard EWC, with a shared latest anchor)
      "online" - exponential moving average: decay * F_old + (1 - decay) * F_new

    The anchor is always the newest params (older anchors are dropped, as in MEAL).
    After the first task (ewc_state is None) the new Fisher is used as-is in every
    mode, rather than decaying it against an all-zero history.
    """
    assert mode in ("last", "multi", "online"), f"unknown EWC mode {mode!r}"
    if ewc_state is None or mode == "last":
        fisher = new_fisher
    elif mode == "multi":
        fisher = jax.tree.map(jnp.add, ewc_state.fisher, new_fisher)
    else:  # "online"
        fisher = jax.tree.map(
            lambda old, new: decay * old + (1.0 - decay) * new, ewc_state.fisher, new_fisher
        )
    return EWCState(old_params=new_params, fisher=fisher)


def ewc_penalty(params, ewc_state: EWCState, coef: float, n_params: int) -> jnp.ndarray:
    """0.5 * coef * mean_i F_i * (theta_i - theta*_i)^2.

    Dividing by the param count (as MEAL does) makes `coef` roughly comparable
    between the CNN and MLP torsos.
    """
    sq = jax.tree.map(
        lambda p, o, f: (f * (p - o) ** 2).sum(),
        params, ewc_state.old_params, ewc_state.fisher,
    )
    return 0.5 * coef * sum(jax.tree.leaves(sq)) / n_params


def make_fisher_fn(network, actor, config):
    """Build the jitted diagonal-Fisher estimator over a rollout's (s, a) samples.

    F = mean[(grad log pi(a|s))^2] needs per-sample grads (mean of squares !=
    square of mean), so samples are processed as a scan over chunks with a
    vmapped grad inside - peak extra memory is EWC_FISHER_CHUNK x n_params
    instead of samples x n_params, and the whole thing stays jitted/on-device.
    Critic params never enter log pi, so their Fisher is 0.
    """

    @jax.jit
    def compute_fisher(params, storage, key: jax.random.PRNGKey):
        chunk = int(config.get("EWC_FISHER_CHUNK", 128))
        batch = int(storage.actions.shape[0] * storage.actions.shape[1])
        num_samples = min(int(config.get("EWC_FISHER_SAMPLES", 65536)), batch)
        num_samples = (num_samples // chunk) * chunk
        assert num_samples > 0, "EWC_FISHER_SAMPLES and EWC_FISHER_CHUNK yield zero Fisher samples"

        obs_flat = storage.obs.reshape((batch,) + storage.obs.shape[2:])
        actions_flat = storage.actions.reshape(batch)
        # Chunked index gather (instead of materializing obs_flat[idx] up front) keeps
        # the pixel-obs case from allocating a second full observation batch.
        idx = jax.random.permutation(key, batch)[:num_samples].reshape(-1, chunk)

        def logp_single(p, ob, act):
            hidden = network.apply(p.network_params, ob[None, ...])
            logits = actor.apply(p.actor_params, hidden)
            return jax.nn.log_softmax(logits)[0, act]

        grad_single = jax.grad(logp_single)

        def accumulate_chunk(acc, idx_c):
            ob = jnp.take(obs_flat, idx_c, axis=0)
            act = jnp.take(actions_flat, idx_c, axis=0)
            g = jax.vmap(grad_single, in_axes=(None, 0, 0))(params, ob, act)
            return jax.tree.map(lambda a, x: a + jnp.square(x).sum(0), acc, g), None

        fisher0 = jax.tree.map(jnp.zeros_like, params)
        fisher, _ = jax.lax.scan(accumulate_chunk, fisher0, idx)
        fisher = jax.tree.map(lambda x: x / num_samples, fisher)

        if config.get("EWC_NORMALIZE_FISHER", True):
            # Rescale to mean(|F|) = 1 so EWC_COEF keeps the same meaning across
            # tasks/architectures whose raw Fisher magnitudes differ by orders of magnitude.
            leaves = jax.tree.leaves(fisher)
            mean_abs = sum(jnp.abs(x).sum() for x in leaves) / sum(x.size for x in leaves)
            fisher = jax.tree.map(lambda x: x / (mean_abs + 1e-12), fisher)
        return fisher

    return compute_fisher


# =============================================================================
# A-GEM - Averaged Gradient Episodic Memory
# =============================================================================
# Episodic memory of past-task transitions; each PPO minibatch gradient is
# projected so it cannot point against the memory gradient. Because `train()`
# runs one task per call, each finished task contributes exactly one fixed-size
# block sampled from its final-policy rollout - uniform sampling over the
# concatenated memory is therefore automatically task-balanced, and no circular
# buffer / size masking (as in MEAL) is needed.

@flax.struct.dataclass
class AGEMMemory:
    """Flat episodic memory: one row per stored transition, tasks concatenated."""
    obs: jnp.ndarray      # [M, *obs_shape]
    actions: jnp.ndarray  # [M]
    returns: jnp.ndarray  # [M] GAE returns from the source rollout (critic BC targets)


def agem_sample_block(storage, key: jax.random.PRNGKey, num_samples: int) -> AGEMMemory:
    """Uniformly sample one task's memory block from a (GAE-completed) rollout."""
    batch = int(storage.actions.shape[0] * storage.actions.shape[1])
    num_samples = min(num_samples, batch)
    idx = jax.random.permutation(key, batch)[:num_samples]
    obs_flat = storage.obs.reshape((batch,) + storage.obs.shape[2:])
    return AGEMMemory(
        obs=obs_flat[idx],
        actions=storage.actions.reshape(batch)[idx],
        returns=storage.returns.reshape(batch)[idx],
    )


def agem_extend_memory(memory: "AGEMMemory | None", block: AGEMMemory) -> AGEMMemory:
    """Append a finished task's block; first task just becomes the memory."""
    if memory is None:
        return block
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), memory, block)


def make_agem_grad_fn(network, actor, critic, config):
    """Build the reference-gradient fn: BC loss on a random memory batch.

    Behavioral cloning (maximize log pi of the remembered actions, regress the
    critic onto stored returns) instead of the PPO loss on memory, following
    MEAL: importance ratios exp(logpi_new - logpi_old) collapse to ~0 after
    cross-task policy drift, which zeroes every PPO-clipped contribution and
    makes the memory gradient meaningless. BC stays informative regardless of
    drift. Entropy is weighted as in the main loss.
    """
    sample_size = int(config.get("AGEM_SAMPLE_SIZE", 256))
    vf_coef = float(config["VF_COEF"])
    ent_coef = float(config["ENT_COEF"])

    def agem_grads(params, memory: AGEMMemory, key: jax.random.PRNGKey):
        idx = jax.random.randint(key, (sample_size,), 0, memory.actions.shape[0])
        obs = jnp.take(memory.obs, idx, axis=0)
        acts = jnp.take(memory.actions, idx, axis=0)
        rets = jnp.take(memory.returns, idx, axis=0)

        def bc_loss(p):
            hidden = network.apply(p.network_params, obs)
            logits = actor.apply(p.actor_params, hidden)
            logp = jax.nn.log_softmax(logits)
            actor_loss = -logp[jnp.arange(sample_size), acts].mean()
            entropy = -(jax.nn.softmax(logits) * logp).sum(-1).mean()
            value = critic.apply(p.critic_params, hidden).squeeze(-1)
            v_loss = 0.5 * ((value - rets) ** 2).mean()
            return actor_loss + vf_coef * v_loss - ent_coef * entropy

        return jax.grad(bc_loss)(params)

    return agem_grads


def agem_project(grads, mem_grads):
    """A-GEM projection: if g . g_mem < 0, remove g's component along g_mem.

        g <- g - (g . g_mem / ||g_mem||^2) * g_mem

    Returns (projected grads, g . g_mem, projected? as float). Applied to the raw
    PPO grads before the optimizer, as in MEAL: projecting Adam's update instead
    would mix in moment estimates accumulated from unprojected gradients. The
    global-norm clip in the optax chain then only rescales, preserving the
    projected direction. `jnp.where` computes both branches, but the projection
    arithmetic is two cheap vector ops.
    """
    g, unravel = ravel_pytree(grads)
    g_mem, _ = ravel_pytree(mem_grads)
    dot_g = jnp.vdot(g, g_mem)
    projected = jnp.where(dot_g < 0, g - (dot_g / (jnp.vdot(g_mem, g_mem) + 1e-12)) * g_mem, g)
    return unravel(projected), dot_g, (dot_g < 0).astype(jnp.float32)


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
# Per task t: train (grads masked to free/shared weights) -> prune (top
# 1/(tasks_left+1) of free weights per leaf become owned by t, the rest are
# zeroed) -> finetune (grads masked to task-t/shared weights; the zeroed free
# weights get zero grads, so Adam keeps them exactly 0). Biases of torso+actor
# are owned by task 0 (trained once with the first task, then frozen), as in
# MEAL. Unlike MEAL's multi-head setup, the single-head actor kernel is pruned
# like the torso - with a shared head, a later task would otherwise overwrite
# earlier tasks' policy outputs.
#
# The owner tree makes MEAL's per-task boolean mask stack, lax.cond dispatch,
# and layer-name string matching unnecessary: all transitions run in plain
# Python between train() calls; only the (precomputed) boolean grad mask enters
# the jitted update.

PACKNET_FREE = -1
PACKNET_SHARED = -2


def packnet_init_owner(params) -> "AgentParams":  # noqa: F821
    """Build the initial owner tree from (freshly initialized) params."""

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


def packnet_train_mask(owner, task: int):
    """Trainable mask for task `task`'s initial train phase: free + shared weights,
    plus anything already owned by this task (the task-0 biases during task 0)."""
    return jax.tree.map(
        lambda o: (o == PACKNET_FREE) | (o == PACKNET_SHARED) | (o == task), owner
    )


def packnet_finetune_mask(owner, task: int):
    """Trainable mask for the post-prune finetune phase: only this task's weights
    (+ shared). Free weights were just zeroed by the prune and must stay zero."""
    return jax.tree.map(lambda o: (o == PACKNET_SHARED) | (o == task), owner)


def packnet_prune(params, owner, task: int, num_tasks: int):
    """Assign task `task` its share of the free weights, zero the rest.

    Per leaf, the top 1/(tasks_left+1) of free weights by magnitude become owned
    by `task` (equal share of the remaining capacity for every remaining task,
    MEAL's balanced schedule; with 5 tasks each gets ~20% of the network). The
    last task takes all remaining free weights and nothing is zeroed. Leaves
    without free entries (biases, critic) pass through untouched. Runs eagerly
    between train() calls, so plain Python control flow is fine.

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


def packnet_eval_params(params, owner, task: int):
    """Recover task `task`'s subnetwork: keep weights owned by tasks <= task and
    shared ones, zero everything else (weights of later tasks and free capacity)."""
    return jax.tree.map(
        lambda p, o: jnp.where(((o >= 0) & (o <= task)) | (o == PACKNET_SHARED), p, jnp.zeros_like(p)),
        params,
        owner,
    )


def packnet_ownership_summary(owner) -> dict:
    """Fraction of all params per owner code (for logging), e.g. {-2: .0, -1: .6, 0: .2, ...}."""
    flat = jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree.leaves(owner)])
    vals, counts = jnp.unique(flat, return_counts=True)
    return {int(v): float(c) / flat.size for v, c in zip(vals, counts)}
