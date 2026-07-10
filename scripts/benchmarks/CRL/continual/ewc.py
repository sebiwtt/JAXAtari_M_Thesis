# =============================================================================
# EWC - Elastic Weight Consolidation (Kirkpatrick et al. 2017)
# =============================================================================
# Diagonal-Fisher EWC, ported from MEAL's IPPO implementation to single-agent
# PPO. After each task, F = E_{s, a~pi}[(d/dtheta log pi(a|s))^2] is estimated
# from a final-policy rollout and merged into the running state (EWC_MODE
# last/multi/online); later tasks add the quadratic penalty
# 0.5 * EWC_COEF * mean_i F_i (theta_i - theta*_i)^2 to the PPO loss.
#
# The critic head gets zero Fisher (log pi does not depend on it), so only
# torso + actor are anchored; the value function stays free to re-adapt on
# each new task.
#
# Config keys: EWC_COEF (required > 0), EWC_MODE ("last" | "multi" | "online"),
# EWC_DECAY, EWC_NORMALIZE_FISHER, EWC_FISHER_SAMPLES, EWC_FISHER_CHUNK.

import flax
import jax
import jax.numpy as jnp

from continual.base import CLMethod
from networks import action_dim_from_params, build_models


@flax.struct.dataclass
class EWCState:
    """Anchor params theta* and diagonal Fisher F accumulated over finished tasks."""
    old_params: "AgentParams"  # noqa: F821 - pytree-only use
    fisher: "AgentParams"  # noqa: F821


class EWC(CLMethod):
    name = "ewc"
    needs_final_rollout = True

    def __init__(self, config: dict, num_tasks: int):
        super().__init__(config, num_tasks)
        self.coef = float(config.get("EWC_COEF", 0.0))
        assert self.coef > 0.0, "CL_METHOD=ewc requires EWC_COEF > 0"
        self.mode = str(config.get("EWC_MODE", "online"))
        assert self.mode in ("last", "multi", "online"), f"unknown EWC_MODE {self.mode!r}"
        self.decay = float(config.get("EWC_DECAY", 0.9))

    def init_state(self, params):
        network, actor, _ = build_models(self.config, action_dim_from_params(params))
        self._compute_fisher = _make_fisher_fn(network, actor, self.config)
        # Dividing the penalty by the param count (as MEAL does) makes EWC_COEF
        # roughly comparable between the CNN and MLP torsos.
        self._n_params = sum(x.size for x in jax.tree.leaves(params))
        return None  # no anchor before the first task finishes

    def loss_penalty(self, params, cl_state):
        """(lambda/2) * mean_i F_i * (theta_i - theta*_i)^2. `cl_state is None`
        (task 0) is resolved at trace time - that compilation is plain PPO."""
        if cl_state is None:
            return jnp.array(0.0)
        sq = jax.tree.map(
            lambda p, o, f: (f * (p - o) ** 2).sum(),
            params, cl_state.old_params, cl_state.fisher,
        )
        return 0.5 * self.coef * sum(jax.tree.leaves(sq)) / self._n_params

    def update_state(self, cl_state, params, storage, key):
        """Estimate the finished task's Fisher and merge it (modes as in MEAL):
          "last"   - keep only the newest task's Fisher
          "multi"  - sum of all tasks' Fishers (standard EWC, shared latest anchor)
          "online" - exponential moving average: decay * F_old + (1 - decay) * F_new
        The anchor is always the newest params (older anchors are dropped, as in
        MEAL). After the first task the new Fisher is used as-is in every mode,
        rather than decaying it against an all-zero history.
        """
        new_fisher = self._compute_fisher(params, storage, key)
        if cl_state is None or self.mode == "last":
            fisher = new_fisher
        elif self.mode == "multi":
            fisher = jax.tree.map(jnp.add, cl_state.fisher, new_fisher)
        else:  # "online"
            fisher = jax.tree.map(
                lambda old, new: self.decay * old + (1.0 - self.decay) * new,
                cl_state.fisher, new_fisher,
            )
        return EWCState(old_params=params, fisher=fisher)


def _make_fisher_fn(network, actor, config):
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
