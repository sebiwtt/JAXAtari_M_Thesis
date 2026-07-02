"""PPO training entrypoint for the Continual RL fine-tuning suite.

Adapted from CleanRL's ppo_atari_envpool_xla_jax_scan.py for JaxAtari. The whole
rollout + PPO update pipeline is written as JAX-jitted / scanned functions so that a
full training iteration runs as one XLA program on the accelerator.

Usage: `uv run scripts/benchmarks/CRL/ppo_crl_finetune.py +alg=ppo_crl_finetune`
"""
import os
import random
import time
from functools import partial
from typing import Sequence, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jaxatari
import hydra
from omegaconf import OmegaConf
from jaxatari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxatari import spaces
from ppo_crl_eval import evaluate
from video_utils import generate_final_video, log_periodic_eval_video

from rtpt import RTPT


def make_env(env_id, seed, num_envs, mods=[], pixel_based=True, native_downscaling=True, smooth_image=True, eval=False):
    """Build a thunk that constructs one (wrapped) JaxAtari environment.

    `mods` are the CRL task modifications (e.g. altered physics/rules) applied on top
    of the base game. During training we only ever want a single active mod at a time
    (or none), so if more than one is passed while `eval=False` we silently fall back
    to the unmodified base environment - the caller is expected to train sequentially
    on one mod per phase. During evaluation the caller is trusted to pass exactly the
    mod (or list of mods) it wants, since this is also used for per-mod video capture.
    """
    def thunk():
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # jaxatari.make expects either None (no mods) or a non-empty list of mods.
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval,  # episodic-life shaping is a training-only trick
                first_fire=True,
                noop_max=30,
                full_action_space=False,
        )
        if pixel_based:
            # Pixel observations: (frame-stacked, downscaled, grayscale) images for the CNN.
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=True,
                use_native_downscaling=native_downscaling,
                smooth_image=smooth_image,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=True,  # reward clipping is also training-only
            )
        else:
            # Object-centric observations: flattened, normalized feature vectors for the MLP.
            env = FlattenObservationWrapper(
                NormalizeObservationWrapper(
                    ObjectCentricWrapper(
                        env,
                        frame_stack_size=4,
                        frame_skip=4,
                        clip_reward=True,
                    )
                )
            )
        env = LogWrapper(env)  # tracks per-episode return/length for logging
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.is_vector_env = True
        return env
    return thunk


class Network(nn.Module):
    """CNN encoder (Nature-DQN style) used for pixel-based observations."""

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, F, H, W) -> (B, H, W, F) for conv
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class MLP_Network(nn.Module):
    """MLP encoder used for object-centric (flattened feature vector) observations.

    Mirrors the output width (512) of `Network` above so `Actor`/`Critic` can sit on
    top of either encoder unchanged.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            461,  # hidden size chosen to roughly match the CNN's parameter count
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    """Value head: maps encoder features to a scalar state-value estimate."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Policy head: maps encoder features to per-action logits."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class AgentParams(NamedTuple):
    """Bundles the three sets of params so a single `TrainState` can hold/update all of them."""
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    """One rollout's worth of transitions, stacked along a leading time axis by `jax.lax.scan`."""
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


def single_run(config: dict):
    """Run one full PPO training job (one seed, one env/mod configuration) end-to-end."""
    # Hydra gives us the alg sub-config nested under "alg"; flatten it into one dict of
    # UPPER_CASE keys, which is what the rest of this function expects.
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])
    if isinstance(config.get("EVAL_MODS"), list):
        config["EVAL_MODS"] = tuple(config["EVAL_MODS"])

    # Derived sizes: how many env-steps make up one PPO iteration/minibatch, and how
    # many iterations are needed to reach TOTAL_TIMESTEPS.
    config["BATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    config["MINIBATCH_SIZE"] = int(config["BATCH_SIZE"] // config["NUM_MINIBATCHES"])
    config["NUM_ITERATIONS"] = int(config["TOTAL_TIMESTEPS"] // config["BATCH_SIZE"])

    run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'
    if config["TRACK"]:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=run_name,
            save_code=True,
        )

    # Seed every RNG source (Python, numpy, JAX) so the run is reproducible.
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_sample_key1, obs_sample_key2, obs_sample_key3 = jax.random.split(key, 4)

    # Build a single (unvmapped) env instance purely to read out shapes/spaces; the
    # actual rollout vmaps its reset/step functions below to run NUM_ENVS copies in
    # lockstep on the accelerator.
    env = make_env(config["ENV_ID"], config["SEED"], config["NUM_ENVS"], list(config["TRAIN_MODS"]), config["PIXEL_BASED"], config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"])()

    @jax.jit
    def vmap_reset(key):
        # squeeze drops the trailing channel dim JaxAtari observations carry, giving
        # (B, F, H, W) which is what the networks below expect.
        obs, state = jax.vmap(env.reset)(key)
        return obs.squeeze(), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze(), state, reward, next_done, info

    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # Anneal the learning rate to 0 over the course of training. `count` is the
        # optimizer step counter, which increments NUM_MINIBATCHES * UPDATE_EPOCHS
        # times per training iteration.
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_ITERATIONS"]
        return config["LEARNING_RATE"] * frac

    # --- Agent setup: encoder + actor + critic heads, all bundled into one TrainState ---
    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    actor = Actor(action_dim=env.action_space().n)
    critic = Critic()
    # Sample obs shape is (F, H, W); add a leading batch dim of 1 for param init.
    network_params = network.init(network_key, env.observation_space().sample(obs_sample_key1).squeeze()[None, ...])
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key2).squeeze()]))),
            critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key3).squeeze()]))),
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LEARNING_RATE"], eps=1e-5
            ),
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample an action for rollout collection, and record its logprob/value."""
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # Sample from the categorical policy via the Gumbel-max trick, which is easy
        # to vectorize/jit compared to jax.random.categorical in this scan-heavy setup.
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Re-evaluate a batch of stored (obs, action) pairs under the *current* params.

        Used inside the PPO loss to get the new logprob/entropy/value needed for the
        clipped objective - as opposed to `get_action_and_value`, which samples fresh
        actions during rollout collection.
        """
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # Numerically stable entropy via the log-sum-exp normalized logits.
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        """Single backward step of Generalized Advantage Estimation, for use in a scan."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=config["GAMMA"], gae_lambda=config["GAE_LAMBDA"])

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        """Fill in `storage.advantages`/`storage.returns` via GAE, scanning backwards in time."""
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()

        advantages = jnp.zeros((config["NUM_ENVS"],))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        """The clipped PPO surrogate objective, plus value and entropy terms."""
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss: clip the probability ratio so updates can't move too far from
        # the policy that generated the data.
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - config["CLIP_COEF"], 1 + config["CLIP_COEF"])
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss: plain MSE against the GAE-based return target.
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        """Run UPDATE_EPOCHS passes over the rollout, each shuffled into NUM_MINIBATCHES chunks."""
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                # (NUM_STEPS, NUM_ENVS, ...) -> (NUM_STEPS * NUM_ENVS, ...)
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                # Shuffle transitions and split into NUM_MINIBATCHES equal chunks.
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=config["UPDATE_EPOCHS"]
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    def eval_and_vid(iteration):
        """Checkpoint the current params, run a held-out evaluation, and log results/video."""
        model_path = f'runs/{run_name}/{config["EXP_NAME"]}_{iteration}.cleanrl_model'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        config,
                        [
                            agent_state.params.network_params,
                            agent_state.params.actor_params,
                            agent_state.params.critic_params,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

        # `evaluate` (ppo_crl_eval.py) reloads the checkpoint and rolls out the greedy
        # policy on the EVAL_MODS environment, independent of the training env state.
        episodic_returns, env_states = evaluate(
            model_path,
            partial(
                make_env,
                mods=list(config["EVAL_MODS"]),
                pixel_based=config["PIXEL_BASED"],
                native_downscaling=config["NATIVE_DOWNSCALING"],
                smooth_image=config["SMOOTH_IMAGE"],
                eval=True,
            ),
            config["ENV_ID"],
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic) if config["PIXEL_BASED"] else (MLP_Network, Actor, Critic),
            seed=config["SEED"],
        )
        wandb.log({"eval/episodic_return_mod": np.mean(jax.device_get(episodic_returns)), "step": iteration})

        if config["CAPTURE_VIDEO"]:
            log_periodic_eval_video(config["ENV_ID"], env_states, iteration)

    # --- Rollout collection ---
    key, reset_key = jax.random.split(key)
    global_step = 0
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    next_done = jnp.zeros(config["NUM_ENVS"], dtype=jax.numpy.bool_)

    def step_once(carry, step, env_step_fn):
        """One env step for all NUM_ENVS environments in lockstep; scanned NUM_STEPS times."""
        agent_state, obs, done, key, env_state = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        next_obs, env_state, reward, next_done, info = env_step_fn(env_state, action)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return ((agent_state, next_obs, next_done, key, env_state), (storage, info))

    def rollout(agent_state, next_obs, next_done, key, env_state, step_once_fn, max_steps):
        (agent_state, next_obs, next_done, key, env_state), (storage, info) = jax.lax.scan(
            step_once_fn, (agent_state, next_obs, next_done, key, env_state), (), max_steps
        )
        return agent_state, next_obs, next_done, storage, key, env_state, info

    rollout = partial(rollout, step_once_fn=partial(step_once, env_step_fn=vmap_step), max_steps=config["NUM_STEPS"])

    # RTPT reports estimated time-to-completion to the OS process title, handy for
    # keeping track of long CRL fine-tuning jobs on a shared cluster.
    rtpt = RTPT(name_initials=config.get("NAME_INITIALS", "RE"), experiment_name='PPO_CRL_Finetune', max_iterations=config["NUM_ITERATIONS"])
    rtpt.start()
    start_time = time.time()
    compile_time = None

    # --- Main training loop: collect rollout -> compute advantages -> PPO update -> log ---
    for iteration in range(1, config["NUM_ITERATIONS"] + 1):
        rtpt.step()
        if config["EVAL_DURING_TRAIN"] and iteration > 0 and iteration % config["EVAL_EVERY"] == 0:
            eval_and_vid(iteration)

        iteration_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, env_state, info = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        global_step += config["NUM_STEPS"] * config["NUM_ENVS"]
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        if compile_time is None:
            # The first iteration includes JIT compilation time; report it separately
            # so the steady-state throughput numbers below aren't skewed by it.
            compile_time = time.time()
            print(f"Compile + first iteration time: {compile_time - start_time:.2f} seconds.")

        # `loss`/`pg_loss`/etc have shape (UPDATE_EPOCHS, NUM_MINIBATCHES); [-1, -1]
        # takes the last minibatch of the last epoch as a representative sample.
        metrics = {
            "charts/avg_episodic_return": info["returned_episode_returns"].mean(),
            "charts/avg_episodic_length": info["returned_episode_lengths"].mean(),
            "charts/learning_rate": agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            "losses/value_loss": v_loss[-1, -1].item(),
            "losses/policy_loss": pg_loss[-1, -1].item(),
            "losses/entropy": entropy_loss[-1, -1].item(),
            "losses/approx_kl": approx_kl[-1, -1].item(),
            "losses/loss": loss[-1, -1].item(),
            "charts/SPS": int(global_step / (time.time() - start_time)),
            "charts/SPS_update": int(config["NUM_ENVS"] * config["NUM_STEPS"] / (time.time() - iteration_time_start)),
            "charts/time": time.time() - start_time,
            "charts/global_step": global_step,
        }
        wandb.log(metrics, step=iteration)

    end_time = time.time()
    print("Training done.")
    if compile_time is not None:
        print(f"Run time after first iteration: {end_time - compile_time:.2f} seconds.")
    print(f"Total train time: {end_time - start_time:.2f} seconds / {(end_time - start_time)/60:.2f} minutes.")
    generate_final_video(config, network, actor, agent_state, make_env)

    if config["SAVE_MODEL"]:
        eval_and_vid(iteration)

    wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    # Hydra resolves the top-level config.yaml plus whichever `+alg=<name>` group config
    # was selected on the command line (see config/alg/) into one nested dict; merge the
    # alg sub-dict up to the top level so `single_run` sees one flat config.
    config = OmegaConf.to_container(config, resolve=True)
    merged_config = {**config, **config.get("alg", {})}
    print("Config:\n", OmegaConf.to_yaml(OmegaConf.create(config)))
    single_run(merged_config)


if __name__ == "__main__":
    main()
