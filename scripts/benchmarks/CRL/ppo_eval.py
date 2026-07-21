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
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id, seed, 1)()
    _Network, _Actor, _Critic = Model
    key = jax.random.key(seed)

    @jax.jit
    def wrapped_reset(key):
        # NNs need (B, F, H, W); squeeze + add leading batch dim.
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        next_obs, next_state, reward, terminated, truncated, info =  env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    key, reset_key = jax.random.split(key)
    next_obs, handle = wrapped_reset(reset_key)
    network = _Network()
    actor = _Actor(action_dim=env.action_space().n)
    critic = _Critic()
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, network_key_2, actor_key_2, critic_key_2 = jax.random.split(key, 4)
    network_params = network.init(network_key, env.observation_space().sample(network_key_2).squeeze()[None, ...])
    actor_params = actor.init(actor_key, network.apply(network_params, env.observation_space().sample(actor_key_2).squeeze()[None, ...]))
    critic_params = critic.init(critic_key, network.apply(network_params, env.observation_space().sample(critic_key_2).squeeze()[None, ...]))
    # critic_params is unused below but must match the saved checkpoint's pytree structure.
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: jnp.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # Gumbel-max trick for categorical sampling.
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    def step_fn(carry, input):
        next_obs, env_state, keys = carry
        actions, keys = jax.vmap(get_action_and_value, in_axes=(None, None, 0, 0))(network_params, actor_params, next_obs, keys)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
        # Use env_reward (raw, unclipped) when present, mirroring LogWrapper's own
        # training-time accounting -- so eval reports the true return regardless
        # of whether clip_reward happens to be on for this env.
        env_reward = infos.get("env_reward", reward)
        first_states = jax.tree.map(lambda x: x[0], env_state)
        return (next_obs, env_state, keys), (first_states, done, env_reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)
    _, (first_states, dones, rewards, actions) = jax.lax.scan(step_fn, (next_obs, env_states, reset_keys), None, length=10_000)

    print("scanned rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))

    first_done = jnp.argmax(dones, axis=0)  # shape: (eval_episodes,)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1,0),(0,0)), constant_values=0)  # shift right by one step
    rewards = rewards * (1 - mask_after_first_done)
    print("filtered rewards: ", rewards.shape, jnp.sum(rewards), jnp.mean(rewards))
    episodic_returns = jnp.sum(rewards, axis=0)  # shape: (eval_episodes,)

    # Whether each stream hit `done` within the 10_000-step scan; if not, its return is a
    # truncated-episode sum, not a full episode - inflated/deflated relative to other cells.
    completed = has_finished[-1].astype(bool)  # shape: (eval_episodes,)
    n_completed = int(jnp.sum(completed))
    print(f"episode completion: {n_completed}/{completed.shape[0]} episodes finished within {rewards.shape[0]} steps")
    if n_completed < completed.shape[0]:
        print(f"WARNING: {completed.shape[0] - n_completed} episode(s) did not terminate within the eval scan window; their returns are likely inflated.")

    # Trim to the first completed episode, for the caller to optionally render as a video.
    env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], first_states.atari_state.atari_state.env_state)

    return episodic_returns, env_states_until_done, completed
