# =============================================================================
# A-GEM - Averaged Gradient Episodic Memory (Chaudhry et al. 2019)
# =============================================================================
# Episodic memory of past-task transitions; each PPO minibatch gradient is
# projected so it cannot point against the memory gradient. Ported from MEAL
# with two adaptations for this one-task-per-train()-call setup:
#
#  - Memory is filled once per finished task from a final-policy rollout (as in
#    the original paper) instead of MEAL's continuous in-training circular
#    buffer. Each task contributes exactly one fixed-size block, so uniform
#    sampling over the concatenated memory is automatically task-balanced and
#    no ptr/size masking is needed.
#  - The reference gradient is a behavioral-cloning loss on memory (maximize
#    log pi of the remembered actions, regress the critic onto stored returns),
#    following MEAL: importance ratios exp(logpi_new - logpi_old) collapse to
#    ~0 after cross-task policy drift, which zeroes every PPO-clipped
#    contribution and makes a PPO memory gradient meaningless.
#
# Config keys: AGEM_MEMORY_PER_TASK, AGEM_SAMPLE_SIZE.

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from continual.base import CLMethod
from networks import action_dim_from_params, build_models


@flax.struct.dataclass
class AGEMMemory:
    """Flat episodic memory: one row per stored transition, tasks concatenated."""
    obs: jnp.ndarray      # [M, *obs_shape]
    actions: jnp.ndarray  # [M]
    returns: jnp.ndarray  # [M] GAE returns from the source rollout (critic BC targets)


class AGEM(CLMethod):
    name = "agem"
    needs_final_rollout = True

    def __init__(self, config: dict, num_tasks: int):
        super().__init__(config, num_tasks)
        self.memory_per_task = int(config.get("AGEM_MEMORY_PER_TASK", 4096))
        self.sample_size = int(config.get("AGEM_SAMPLE_SIZE", 256))
        self.vf_coef = float(config["VF_COEF"])
        self.ent_coef = float(config["ENT_COEF"])

    def init_state(self, params):
        self._network, self._actor, self._critic = build_models(
            self.config, action_dim_from_params(params)
        )
        return None  # memory is empty before the first task finishes

    def transform_grads(self, grads, params, cl_state, key):
        """Project the raw PPO grads before Adam sees them (as in MEAL: projecting
        Adam's update instead would mix in moment estimates accumulated from
        unprojected gradients). `cl_state is None` (task 0) is trace-time."""
        if cl_state is None:
            return grads, {}
        mem_grads = self._bc_grads(params, cl_state, key)
        grads, dot, projected = _project(grads, mem_grads)
        # agem_projected's per-iteration mean = fraction of minibatches whose
        # gradient conflicted with memory and was projected.
        return grads, {"agem_dot": dot, "agem_projected": projected}

    def update_state(self, cl_state, params, storage, key):
        """Append the finished task's memory block, sampled from its final-policy rollout."""
        batch = int(storage.actions.shape[0] * storage.actions.shape[1])
        num_samples = min(self.memory_per_task, batch)
        idx = jax.random.permutation(key, batch)[:num_samples]
        obs_flat = storage.obs.reshape((batch,) + storage.obs.shape[2:])
        block = AGEMMemory(
            obs=obs_flat[idx],
            actions=storage.actions.reshape(batch)[idx],
            returns=storage.returns.reshape(batch)[idx],
        )
        if cl_state is None:
            return block
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), cl_state, block)

    def _bc_grads(self, params, memory: AGEMMemory, key: jax.random.PRNGKey):
        """Reference gradient: BC loss on a random memory batch (entropy weighted
        as in the main loss)."""
        idx = jax.random.randint(key, (self.sample_size,), 0, memory.actions.shape[0])
        obs = jnp.take(memory.obs, idx, axis=0)
        acts = jnp.take(memory.actions, idx, axis=0)
        rets = jnp.take(memory.returns, idx, axis=0)

        def bc_loss(p):
            hidden = self._network.apply(p.network_params, obs)
            logits = self._actor.apply(p.actor_params, hidden)
            logp = jax.nn.log_softmax(logits)
            actor_loss = -logp[jnp.arange(self.sample_size), acts].mean()
            entropy = -(jax.nn.softmax(logits) * logp).sum(-1).mean()
            value = self._critic.apply(p.critic_params, hidden).squeeze(-1)
            v_loss = 0.5 * ((value - rets) ** 2).mean()
            return actor_loss + self.vf_coef * v_loss - self.ent_coef * entropy

        return jax.grad(bc_loss)(params)


def _project(grads, mem_grads):
    """A-GEM projection: if g . g_mem < 0, remove g's component along g_mem.

        g <- g - (g . g_mem / ||g_mem||^2) * g_mem

    Returns (projected grads, g . g_mem, projected? as float). The global-norm
    clip in the optax chain runs afterwards and only rescales, preserving the
    projected direction. `jnp.where` computes both branches, but the projection
    arithmetic is two cheap vector ops.
    """
    g, unravel = ravel_pytree(grads)
    g_mem, _ = ravel_pytree(mem_grads)
    dot_g = jnp.vdot(g, g_mem)
    projected = jnp.where(dot_g < 0, g - (dot_g / (jnp.vdot(g_mem, g_mem) + 1e-12)) * g_mem, g)
    return unravel(projected), dot_g, (dot_g < 0).astype(jnp.float32)
