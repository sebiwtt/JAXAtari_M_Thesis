# =============================================================================
# Benchmark-owned evaluation: roll out an arbitrary policy on a wrapped env
# =============================================================================
# This is the ONLY place official R / R_rand numbers come from. It takes a
# jit-safe act function instead of a checkpoint + fixed network classes, so any
# ContinualAgent can be evaluated. agents.ppo.eval.evaluate (checkpoint-based,
# used by standalone tools) delegates here.
#
# The PRNG-split sequence before the episode reset keys reproduces the
# pre-framework ppo_eval.evaluate byte-for-byte (it consumed key splits for a
# dummy network init that no longer exists). Do NOT clean this up: it keeps
# every matrix cell bit-identical to results produced before the framework
# refactor, on the same software/hardware stack.
# =============================================================================

from typing import Callable

import jax
import jax.numpy as jnp


def evaluate_policy(
    act: Callable,
    env,
    eval_episodes: int,
    seed: int,
    max_episode_steps: int = 10_000,
):
    """Run `eval_episodes` parallel episodes of `act` on `env`.

    `act(obs, key) -> (action, key)` with obs shaped (1, *obs_shape) and action
    shaped (1,); it is vmapped over episodes and must be jit-traceable.
    `env` is a single (unvmapped) eval-wrapped env from framework.envs.make_env.

    Returns (episodic_returns, env_states_until_done, completed):
      episodic_returns  - (eval_episodes,) first-episode return per stream, from
                          raw (unclipped) rewards
      env_states_until_done - env-state pytree of episode stream 0 up to its
                          first done (for optional video rendering)
      completed         - (eval_episodes,) bool, whether the stream finished an
                          episode within the scan window (if not, its return is
                          a truncated sum and likely off)
    """
    key = jax.random.key(seed)

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

    # Historical key-split sequence (see module header): one split for a reset
    # key and two 4-way splits for a dummy model init, all discarded.
    key, _unused_reset_key = jax.random.split(key)
    key, _u1, _u2, _u3 = jax.random.split(key, 4)
    key, _u4, _u5, _u6 = jax.random.split(key, 4)

    def step_fn(carry, _):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(act)(next_obs, keys)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
        # Use env_reward (raw, unclipped) when present, mirroring LogWrapper's own
        # training-time accounting -- so eval reports the true return regardless
        # of whether clip_reward happens to be on for this env.
        env_reward = infos.get("env_reward", reward)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (next_obs, env_state, keys), (first_states, done, env_reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards, actions) = jax.lax.scan(
        step_fn, (next_obs, env_states, reset_keys), None, length=max_episode_steps
    )

    print("scanned rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))

    first_done = jnp.argmax(dones, axis=0)  # shape: (eval_episodes,)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)  # shift right by one step
    rewards = rewards * (1 - mask_after_first_done)
    print("filtered rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    episodic_returns = jnp.sum(rewards, axis=0)  # shape: (eval_episodes,)

    # Whether each stream hit `done` within the scan window; if not, its return is a
    # truncated-episode sum, not a full episode - inflated/deflated relative to other cells.
    completed = has_finished[-1].astype(bool)  # shape: (eval_episodes,)
    n_completed = int(jnp.sum(completed))
    print(f"episode completion: {n_completed}/{completed.shape[0]} episodes finished within {rewards.shape[0]} steps")
    if n_completed < completed.shape[0]:
        print(f"WARNING: {completed.shape[0] - n_completed} episode(s) did not terminate within the eval scan window; their returns are likely inflated.")

    # Trim to the first completed episode, for the caller to optionally render as a video.
    env_states_until_done = jax.tree.map(lambda x: x[: first_done[0] + 1], first_states.atari_state.atari_state.env_state)

    return episodic_returns, env_states_until_done, completed
