
# =============================================================================
# PPO trainer for JAXtari (single-agent, on-policy, fully jitted)
# Adapted from CleanRL's ppo_atari_envpool_xla_jax_scan.py
#
# CL baselines live in `crl_methods.py`; the hooks here are intentionally thin:
#   EWC     -> IMPLEMENTED: `ewc_penalty` inside `ppo_loss` anchored to an
#              `EWCState`; Fisher estimated post-task from a fresh on-policy
#              rollout (`return_fisher=True`).
#   A-GEM   -> IMPLEMENTED: `agem_project` on `grads` in `update_minibatch`
#              against a BC gradient on `AGEMMemory`; memory blocks sampled
#              post-task from a fresh on-policy rollout (`return_agem_samples=True`).
#   PackNet -> IMPLEMENTED: boolean `grad_mask` applied to `grads` in
#              `update_minibatch`; all owner-tree bookkeeping (prune, phase
#              masks, eval subnetworks) lives in crl_methods.py / the
#              orchestrator. Single-head Actor is pruned like the torso.
# =============================================================================


import random
import time
from functools import partial
from typing import Callable, Sequence, NamedTuple

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
from jaxatari.wrappers import NormalizeObservationWrapper, ObjectCentricWrapper, PixelObsWrapper, AtariWrapper, LogWrapper, FlattenObservationWrapper
from jaxatari import spaces
from crl_methods import (
    AGEMMemory,
    EWCState,
    agem_project,
    agem_sample_block,
    ewc_penalty,
    make_agem_grad_fn,
    make_fisher_fn,
)
from video_utils import generate_final_video, save_obs_debug_frame

from rtpt import RTPT

# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================
# Returns a thunk (zero-arg closure) building one fully wrapped env, later vmapped
# over NUM_ENVS.

def make_env(env_id, seed, num_envs, mods=[], pixel_based=True, native_downscaling=True, smooth_image=True, grayscale=False, eval=False):
    def thunk():
        active_mods = mods
        if not eval and isinstance(active_mods, (list, tuple)) and len(active_mods) > 1:
            active_mods = []

        # jaxatari.make expects None (no mods) or a non-empty list.
        if isinstance(active_mods, (list, tuple)) and len(active_mods) == 0:
            mods_arg = None
        else:
            mods_arg = active_mods

        env = jaxatari.make(env_id, mods=mods_arg)

        # episodic_life/clip_reward are train-only tricks; eval sees true boundaries/reward.
        env = AtariWrapper(
                env,
                sticky_actions=0.0,
                episodic_life=not eval,
                first_fire=True,
                noop_max=30,
                full_action_space=False,
        )
        if pixel_based:
            env = PixelObsWrapper(
                env,
                do_pixel_resize=True,
                pixel_resize_shape=(84, 84),
                grayscale=grayscale,
                use_native_downscaling=native_downscaling,
                smooth_image=smooth_image,
                frame_stack_size=4,
                frame_skip=4,
                max_pooling=True,
                clip_reward=True,
            )
        else:
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
        env = LogWrapper(env)
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.is_vector_env = True
        return env
    return thunk

# =============================================================================
# NEURAL NETWORKS
# =============================================================================
# Shared TORSO (Network or MLP_Network, output dim 512) + two linear HEADS
# (Actor, Critic); heads are byte-for-byte interchangeable between modalities.

class Network(nn.Module):
    """Pixel torso: Nature-CNN feature extractor."""
    @nn.compact
    def __call__(self, x):
        if x.ndim == 5:
            # (B, F, H, W, C) -> (B, H, W, F*C): each stacked frame's channels become conv input channels.
            b, f, h, w, c = x.shape
            x = jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(b, h, w, f * c)
        else:
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
    """Object-centric torso: 2-layer MLP producing the same 512-d output as Network."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            461,  # roughly matches the CNN's parameter count
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
    """Torso features -> scalar state value V(s)."""

    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

class Actor(nn.Module):
    """Torso features -> action logits."""
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


# =============================================================================
# PARAMETER + ROLLOUT CONTAINERS
# =============================================================================

class AgentParams(NamedTuple):
    """Bundles the three param sets so a single `TrainState` can hold/update all of them."""
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

# =============================================================================
# MAIN ENTRY: one full training run for a single (env, modality, seed) config
# =============================================================================

def train(
    config: dict,
    init_params: "AgentParams | None" = None,
    run_name: str | None = None,
    manage_wandb: bool = True,
    wandb_step_offset: int = 0,
    wandb_group: str | None = None,
    iteration_callback: "Callable[[int, int, AgentParams], bool] | None" = None,
    ewc_state: "EWCState | None" = None,
    return_fisher: bool = False,
    agem_memory: "AGEMMemory | None" = None,
    return_agem_samples: bool = False,
    grad_mask: "AgentParams | None" = None,
) -> "AgentParams | tuple[AgentParams, AgentParams] | tuple[AgentParams, AGEMMemory]":
    """Run one single-task PPO training job and return the final agent params.

    `init_params`, if given, resumes from prior params instead of a fresh init
    (naive finetuning across CRL tasks); the optimizer is always rebuilt fresh.

    CL hooks (see `crl_methods.py`; both are trace-time branches, so with all of
    them off the compiled update is identical to plain PPO):

    `ewc_state`, if given, adds the EWC quadratic penalty
    0.5 * EWC_COEF * mean_i F_i (theta_i - theta*_i)^2 to `ppo_loss`.

    `agem_memory`, if given, projects each minibatch's PPO gradient so it cannot
    conflict with the behavioral-cloning gradient on a batch sampled from the
    past-task memory (A-GEM).

    `grad_mask`, if given, zeroes gradients wherever the boolean mask is False
    (PackNet phase masks). Freezing is exact: the optimizer is rebuilt fresh per
    train() call, so always-zero grads keep Adam's moments (and the weights) at
    exactly their initial values.

    `return_fisher=True` / `return_agem_samples=True` (mutually exclusive)
    collect one extra on-policy rollout with the final params after training and
    return `(params, fisher)` resp. `(params, memory_block)` - the per-task
    ingredients for `ewc_update_state` / `agem_extend_memory`.

    `run_name`/`manage_wandb`/`wandb_step_offset`/`wandb_group` let a caller
    (e.g. the continual orchestrator) run this repeatedly against one shared
    wandb run without checkpoint-path or metric collisions between tasks.

    `iteration_callback(iteration, global_step, params)`, if given, is called
    after every PPO update; returning True stops training early (used by the
    difficulty harness to probe eval performance mid-adaptation and cut off the
    moment a target return is reached). The most recent params are always the
    ones returned, so an early stop still returns the crossing-point agent.
    """
    # Hydra nests the alg sub-config under "alg"; flatten to one UPPER_CASE dict.
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

    assert (ewc_state is not None) + (agem_memory is not None) + (grad_mask is not None) <= 1, (
        "EWC, A-GEM and PackNet (grad_mask) are mutually exclusive"
    )
    assert not (return_fisher and return_agem_samples), "return_fisher and return_agem_samples are mutually exclusive"

    if isinstance(config.get("TRAIN_MODS"), list):
        config["TRAIN_MODS"] = tuple(config["TRAIN_MODS"])

    config["BATCH_SIZE"] = int(config["NUM_ENVS"] * config["NUM_STEPS"])
    config["MINIBATCH_SIZE"] = int(config["BATCH_SIZE"] // config["NUM_MINIBATCHES"])
    config["NUM_ITERATIONS"] = int(config["TOTAL_TIMESTEPS"] // config["BATCH_SIZE"])

    if run_name is None:
        run_name = f'{config["ENV_ID"]}_{config["EXP_NAME"]}_{"oc" if not config["PIXEL_BASED"] else "pixel"}_{config["SEED"]}'
    chart_section = f"charts-{wandb_group}" if wandb_group else "charts"
    loss_section = f"losses-{wandb_group}" if wandb_group else "losses"
    if config["TRACK"] and manage_wandb:
        wandb.init(
            project=config["PROJECT"],
            entity=config["ENTITY"],
            config=config,
            name=run_name,
            save_code=True,
        )

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    key = jax.random.PRNGKey(config["SEED"])
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key, obs_sample_key1, obs_sample_key2, obs_sample_key3 = jax.random.split(key, 4)

    # Unvmapped env instance purely to read out shapes/spaces; the rollout below
    # vmaps reset/step to run NUM_ENVS copies in lockstep.
    env = make_env(config["ENV_ID"], config["SEED"], config["NUM_ENVS"], list(config["TRAIN_MODS"]), config["PIXEL_BASED"], config["NATIVE_DOWNSCALING"], config["SMOOTH_IMAGE"], config["GRAYSCALE"])()

    @jax.jit
    def vmap_reset(key):
        # squeeze drops the trailing channel dim for grayscale, giving (B, F, H, W); RGB keeps (B, F, H, W, C).
        obs, state = jax.vmap(env.reset)(key)
        return obs.squeeze(), state

    @jax.jit
    def vmap_step(state, action):
        next_obs, state, reward, terminated, truncated, info = jax.vmap(env.step)(state, action)
        next_done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze(), state, reward, next_done, info

    assert isinstance(env.action_space(), spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        # count is the optimizer step counter (NUM_MINIBATCHES * UPDATE_EPOCHS per iteration).
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_ITERATIONS"]
        return config["LEARNING_RATE"] * frac

    network = Network() if config["PIXEL_BASED"] else MLP_Network()
    actor = Actor(action_dim=env.action_space().n)
    critic = Critic()

    if init_params is None:
        # Sample obs shape is (F, H, W); add a leading batch dim for param init.
        network_params = network.init(network_key, env.observation_space().sample(obs_sample_key1).squeeze()[None, ...])
        # Heads are initialised on the torso output of a dummy obs, matching input dims.
        params = AgentParams(
            network_params=network_params,
            actor_params=actor.init(actor_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key2).squeeze()]))),
            critic_params=critic.init(critic_key, network.apply(network_params, np.array([env.observation_space().sample(obs_sample_key3).squeeze()]))),
        )
    else:
        # Action space must stay identical across tasks (single-head Actor); check
        # explicitly rather than failing deep inside apply().
        resumed_action_dim = init_params.actor_params["params"]["Dense_0"]["bias"].shape[0]
        assert resumed_action_dim == env.action_space().n, (
            f"action space changed across tasks: init_params has action_dim={resumed_action_dim}, "
            f'but current task ({config.get("TRAIN_MODS")}) has action_dim={env.action_space().n}'
        )
        params = init_params

    # tx.init(params) below always builds fresh optimizer state, so Adam moments
    # never carry across `train()` calls even when `init_params` does.
    agent_state = TrainState.create(
        apply_fn=None,
        params=params,
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
        # Gumbel-max trick: easier to vectorize/jit than jax.random.categorical here.
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
        """Recompute logprob/entropy/value for a given action (new policy scoring old rollout)."""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # Numerically stable entropy via log-sum-exp normalized logits.
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value

    # GAE backward recursion:
    #   delta_t = r_t + gamma * V_{t+1} * (1-done) - V_t
    #   A_t     = delta_t + gamma * lambda * (1-done) * A_{t+1}
    def compute_gae_once(carry, inp, gamma, gae_lambda):
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

    ewc_coef = float(config.get("EWC_COEF", 0.0))
    n_agent_params = sum(x.size for x in jax.tree.leaves(params))
    # A-GEM reference gradient: BC loss on a random batch from past-task memory.
    agem_grad_fn = make_agem_grad_fn(network, actor, critic, config)

    # EWC: (lambda/2) * mean_i F_i * (theta_i - theta*_i)^2, anchoring params to
    # the previous tasks' solution proportionally to their Fisher importance.
    # `ewc_state is None` is resolved at trace time, so the no-EWC compilation is
    # byte-identical to plain PPO.
    def ppo_loss(params, ewc_state, x, a, logp, mb_advantages, mb_returns):
        """Clipped PPO surrogate objective, plus value/entropy terms and optional EWC penalty."""
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if config["NORM_ADV"]:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - config["CLIP_COEF"], 1 + config["CLIP_COEF"])
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - config["ENT_COEF"] * entropy_loss + v_loss * config["VF_COEF"]
        if ewc_state is not None:
            ewc_pen = ewc_penalty(params, ewc_state, ewc_coef, n_agent_params)
            loss = loss + ewc_pen
        else:
            ewc_pen = jnp.array(0.0)
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), ewc_pen)

    # Differentiates w.r.t. argnums=0 (params) only; ewc_state is a constant input.
    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        ewc_state: "EWCState | None",
        agem_memory: "AGEMMemory | None",
        grad_mask: "AgentParams | None",
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey, mem_key = jax.random.split(key, 3)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)
            # Per-minibatch keys for A-GEM memory sampling; dead code when A-GEM is off.
            mb_keys = jax.random.split(mem_key, config["NUM_MINIBATCHES"])

            # A-GEM: project the raw PPO grads before Adam sees them, so the
            # applied update cannot point against the past-task memory gradient.
            # `agem_memory is None` is resolved at trace time (like ewc_state).
            def update_minibatch(agent_state, xs):
                minibatch, mb_key = xs
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    ewc_state,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                if agem_memory is not None:
                    mem_grads = agem_grad_fn(agent_state.params, agem_memory, mb_key)
                    grads, agem_dot, agem_proj = agem_project(grads, mem_grads)
                else:
                    agem_dot, agem_proj = jnp.array(0.0), jnp.array(0.0)
                # PackNet: zero grads of weights owned by other tasks (train phase) or
                # not owned by the current task (finetune phase).
                if grad_mask is not None:
                    grads = jax.tree.map(lambda g, m: jnp.where(m, g, 0.0), grads, grad_mask)
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, grads) = jax.lax.scan(
                update_minibatch, agent_state, (shuffled_storage, mb_keys)
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=config["UPDATE_EPOCHS"]
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, key

    # ========================================================================
    # ROLLOUT + TRAINING LOOP
    # ========================================================================

    key, reset_key = jax.random.split(key)
    global_step = 0
    next_obs, env_state = vmap_reset(jax.random.split(reset_key, config["NUM_ENVS"]))
    next_done = jnp.zeros(config["NUM_ENVS"], dtype=jax.numpy.bool_)

    def step_once(carry, step, env_step_fn):
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

    # Post-task EWC Fisher estimator (see crl_methods.py for the details).
    compute_fisher = make_fisher_fn(network, actor, config)

    # RTPT reports estimated time-to-completion to the OS process title.
    rtpt = RTPT(name_initials=config.get("NAME_INITIALS", "RE"), experiment_name='PPO_CRL_Finetune', max_iterations=config["NUM_ITERATIONS"])
    rtpt.start()
    start_time = time.time()
    compile_time = None

    for iteration in range(1, config["NUM_ITERATIONS"] + 1):
        rtpt.step()

        iteration_time_start = time.time()
        agent_state, next_obs, next_done, storage, key, env_state, info = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        if iteration == 1:
            # Snapshot of the real rollout obs (post-PixelObsWrapper, after NUM_STEPS of
            # actual env stepping - not a blank reset screen) for this task/mod combo.
            # storage.obs is (NUM_STEPS, NUM_ENVS, F, H, W[, C]); take the last timestep.
            save_obs_debug_frame(config, storage.obs[-1], run_name)
        global_step += config["NUM_STEPS"] * config["NUM_ENVS"]
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, ewc_pen, agem_dot, agem_proj, key = update_ppo(
            agent_state,
            storage,
            ewc_state,
            agem_memory,
            grad_mask,
            key,
        )
        if compile_time is None:
            # First iteration includes JIT compile time; report separately.
            compile_time = time.time()
            print(f"Compile + first iteration time: {compile_time - start_time:.2f} seconds.")

        # loss/pg_loss/etc have shape (UPDATE_EPOCHS, NUM_MINIBATCHES); [-1, -1] is
        # the last minibatch of the last epoch.
        #
        # LogWrapper zeroes returned_episode_returns/_lengths on env.reset(), so right
        # after a fresh reset (every task, including resumed ones) they read 0 for any
        # env slot that hasn't finished an episode yet - regardless of policy quality.
        # NaN those still-warming-up slots out instead of dragging the mean toward 0.
        if config.get("NAN_UNTIL_FIRST_EPISODE", False):
            never_completed = info["returned_episode_lengths"] == 0
            avg_episodic_return = jnp.nanmean(jnp.where(never_completed, jnp.nan, info["returned_episode_returns"]))
            avg_episodic_length = jnp.nanmean(jnp.where(never_completed, jnp.nan, info["returned_episode_lengths"]))
        else:
            avg_episodic_return = info["returned_episode_returns"].mean()
            avg_episodic_length = info["returned_episode_lengths"].mean()
        metrics = {
            f"{chart_section}/avg_episodic_return": avg_episodic_return,
            f"{chart_section}/avg_episodic_length": avg_episodic_length,
            f"{chart_section}/learning_rate": agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            f"{loss_section}/value_loss": v_loss[-1, -1].item(),
            f"{loss_section}/policy_loss": pg_loss[-1, -1].item(),
            f"{loss_section}/entropy": entropy_loss[-1, -1].item(),
            f"{loss_section}/approx_kl": approx_kl[-1, -1].item(),
            f"{loss_section}/loss": loss[-1, -1].item(),
            **({f"{loss_section}/ewc_penalty": ewc_pen[-1, -1].item()} if ewc_state is not None else {}),
            # agem_proj is 0/1 per minibatch; the mean is the fraction of minibatches
            # whose gradient conflicted with memory and was projected this iteration.
            **(
                {
                    f"{loss_section}/agem_projected_frac": agem_proj.mean().item(),
                    f"{loss_section}/agem_dot": agem_dot[-1, -1].item(),
                }
                if agem_memory is not None
                else {}
            ),
            f"{chart_section}/SPS": int(global_step / (time.time() - start_time)),
            f"{chart_section}/SPS_update": int(config["NUM_ENVS"] * config["NUM_STEPS"] / (time.time() - iteration_time_start)),
            f"{chart_section}/time": time.time() - start_time,
            f"{chart_section}/global_step": global_step,
        }
        if config["TRACK"]:
            wandb.log(metrics, step=wandb_step_offset + iteration)

        if iteration_callback is not None and iteration_callback(iteration, global_step, agent_state.params):
            print(f"[train] early-stop signalled by callback at iteration {iteration} (global_step={global_step}).")
            break

    end_time = time.time()
    print("Training done.")
    if compile_time is not None:
        print(f"Run time after first iteration: {end_time - compile_time:.2f} seconds.")
    print(f"Total train time: {end_time - start_time:.2f} seconds / {(end_time - start_time)/60:.2f} minutes.")
    if config["TRACK"]:
        generate_final_video(config, network, actor, agent_state, make_env)

    if config["TRACK"] and manage_wandb:
        wandb.finish()

    if return_fisher or return_agem_samples:
        # One extra rollout with the *final* params so the post-task CL state is built
        # from (s, a) ~ pi_theta exactly; the last training rollout was collected by a
        # slightly older policy.
        post_start = time.time()
        _, post_next_obs, post_next_done, post_storage, key, _, _ = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        key, post_key = jax.random.split(key)
        if return_fisher:
            fisher = compute_fisher(agent_state.params, post_storage, post_key)
            fisher = jax.block_until_ready(fisher)
            print(f"[EWC] Fisher estimation (rollout + grads) took {time.time() - post_start:.2f} seconds.")
            return agent_state.params, fisher
        # A-GEM stores GAE returns as critic BC targets, so complete the rollout first.
        post_storage = compute_gae(agent_state, post_next_obs, post_next_done, post_storage)
        block = agem_sample_block(post_storage, post_key, int(config.get("AGEM_MEMORY_PER_TASK", 4096)))
        block = jax.block_until_ready(block)
        print(f"[A-GEM] memory block of {block.actions.shape[0]} transitions sampled in {time.time() - post_start:.2f} seconds.")
        return agent_state.params, block

    return agent_state.params
