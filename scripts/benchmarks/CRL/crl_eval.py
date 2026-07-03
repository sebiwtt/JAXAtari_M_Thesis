# =============================================================================
# Harness-owned policy evaluation for the CRL benchmark.
#
# Benchmark-owned on purpose: env construction, eval seeding, rollout length,
# reward masking, and episode-completion accounting are identical for every
# algorithm. A plugged-in algorithm only supplies the `act_fn` (via its
# `load_policy`), i.e. it controls which actions get taken - never how they
# are scored.
# =============================================================================

import jax
import jax.numpy as jnp
import numpy as np

from crl_env import make_env

EVAL_SCAN_STEPS = 10_000


def evaluate_policy(act_fn, config: dict, mods: list) -> tuple:
    """Roll out `act_fn` on the eval env built with `mods` and score it.

    act_fn(obs, key) -> (action, key), for one batch-1 observation; it is vmapped
    over EVAL_EPISODES parallel episode streams. Returns (episodic_returns,
    completed), both 1-D numpy arrays of length EVAL_EPISODES. Rewards are only
    counted up to each stream's first episode end; `completed` marks streams
    whose episode actually finished within EVAL_SCAN_STEPS.
    """
    env = make_env(
        config["ENV_ID"], config["EVAL_SEED"], 1, mods,
        config["PIXEL_BASED"], config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"],
        eval=True,
    )()

    @jax.jit
    def wrapped_reset(key):
        # NNs need (B, F, H, W); squeeze + add leading batch dim.
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def step_fn(carry, _):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(act_fn)(next_obs, keys)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, actions)
        return (next_obs, env_state, keys), (done, reward)

    key = jax.random.key(config["EVAL_SEED"])
    reset_keys = jax.random.split(key, config["EVAL_EPISODES"])
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (dones, rewards) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=EVAL_SCAN_STEPS)

    # Zero out rewards after each stream's first episode end.
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)  # shift right by one step
    rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(rewards, axis=0)
    completed = has_finished[-1].astype(bool)

    return np.asarray(jax.device_get(episodic_returns)), np.asarray(jax.device_get(completed))
