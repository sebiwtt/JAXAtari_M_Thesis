from typing import Callable

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: nn.Module,
    seed=1,
):
    """Checkpoint-based eval for the fixed PPO networks: loads a cleanrl_model
    file, wraps it as an act function, and delegates the rollout to the
    framework's policy-based `evaluate_policy` (which preserves this function's
    historical PRNG stream, so results are unchanged)."""
    from framework.evaluation import evaluate_policy

    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, seed, 1)()
    _Network, _Actor, _Critic = Model
    network = _Network()
    actor = _Actor(action_dim=env.action_space().n)
    critic = _Critic()

    # Dummy init purely as the pytree template for from_bytes; every value is
    # overwritten from the file, so the keys here don't matter.
    key = jax.random.key(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)
    sample_obs = env.observation_space().sample(k3).squeeze()[None, ...]
    network_params = network.init(k1, sample_obs)
    hidden = network.apply(network_params, sample_obs)
    actor_params = actor.init(k2, hidden)
    critic_params = critic.init(k2, hidden)
    # critic_params is unused below but must match the saved checkpoint's pytree structure.
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    def act(obs, key):
        hidden = network.apply(network_params, obs)
        logits = actor.apply(actor_params, hidden)
        # Gumbel-max trick for categorical sampling.
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    return evaluate_policy(act, env, eval_episodes=eval_episodes, seed=seed)


def make_curve_eval_fn(
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: nn.Module,
    seed=1,
):
    """Reusable in-memory eval for the mid-training CRL curve (ppo_trainer).

    Same rollout/first-episode-masking semantics as `evaluate`, but takes params
    directly instead of a checkpoint file and returns one jitted
    `(network_params, actor_params) -> (episodic_returns, completed)` closure,
    so repeated calls within a task compile exactly once. The reset keys are
    fixed by `seed`, so successive curve points differ only through the params.
    """
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, seed, 1)()
    _Network, _Actor, _Critic = Model
    network = _Network()
    actor = _Actor(action_dim=env.action_space().n)

    def wrapped_reset(key):
        # NNs need (B, F, H, W); squeeze + add leading batch dim.
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    def get_action(network_params, actor_params, next_obs, key):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # Gumbel-max trick for categorical sampling.
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    @jax.jit
    def run(network_params, actor_params):
        reset_keys = jax.random.split(jax.random.key(seed), eval_episodes)
        next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)

        def step_fn(carry, input):
            next_obs, env_state, keys = carry
            actions, keys = jax.vmap(get_action, in_axes=(None, None, 0, 0))(network_params, actor_params, next_obs, keys)
            next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
            # Raw, unclipped return - same accounting as `evaluate` and LogWrapper.
            env_reward = infos.get("env_reward", reward)
            return (next_obs, env_state, keys), (done, env_reward)

        _, (dones, rewards) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_000)

        has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
        mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        episodic_returns = jnp.sum(rewards * (1 - mask_after_first_done), axis=0)
        completed = has_finished[-1].astype(bool)
        return episodic_returns, completed

    return run
