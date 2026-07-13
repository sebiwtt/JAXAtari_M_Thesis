
# =============================================================================
# PPO trainer for JAXtari (single-agent, on-policy, fully jitted)
# Adapted from CleanRL's ppo_atari_envpool_xla_jax_scan.py
#
# CL-agnostic: continual-learning methods (see continual/) plug in through
# exactly three generic parameters of train():
#   cl_method            - a continual.base.CLMethod; only its two jit-safe
#                          hooks are called: `loss_penalty` (inside ppo_loss)
#                          and `transform_grads` (raw grads, before Adam)
#   cl_state             - the method's device data (Fisher, memory, mask, ...),
#                          threaded through update_ppo as a jit argument so
#                          buffers are never baked into the executable
#   return_final_rollout - additionally return a GAE-completed rollout of the
#                          final policy; methods build their next cl_state
#                          from it (EWC's Fisher, A-GEM's memory block)
# =============================================================================


import random
import time
from functools import partial
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jaxatari import spaces

from envs import make_env
from networks import Actor, AgentParams, Critic, MLP_Network, Network
from tools.video_utils import generate_final_video, save_obs_debug_frame

from rtpt import RTPT


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
    cl_method=None,
    cl_state=None,
    return_final_rollout: bool = False,
) -> "AgentParams | tuple[AgentParams, tuple[Storage, jax.random.PRNGKey]]":
    """Run one single-task PPO training job and return the final agent params.

    `init_params`, if given, resumes from prior params instead of a fresh init
    (finetuning across CRL tasks); the optimizer is always rebuilt fresh.

    `run_name`/`manage_wandb`/`wandb_step_offset`/`wandb_group` let a caller
    (e.g. the continual orchestrator) run this repeatedly against one shared
    wandb run without checkpoint-path or metric collisions between tasks.

    `iteration_callback(iteration, global_step, params)`, if given, is called
    after every PPO update; returning True stops training early (used by the
    difficulty harness to probe eval performance mid-adaptation and cut off the
    moment a target return is reached). The most recent params are always the
    ones returned, so an early stop still returns the crossing-point agent.

    `cl_method`/`cl_state`: continual-learning hooks, see the module header.
    With `cl_method=None` the compiled update is byte-identical to plain PPO.
    `cl_method.transform_grads` metrics are logged under losses/ (per-iteration
    mean); `cl_method.loss_penalty` is logged as losses/cl_penalty.

    `return_final_rollout=True` collects one extra on-policy rollout with the
    final params after training (GAE completed, so `storage.returns` is usable
    as a target) and returns `(params, (storage, key))`.
    """
    # Hydra nests the alg sub-config under "alg"; flatten to one UPPER_CASE dict.
    config = {k.upper(): v for k, v in config.items() if k != "alg"}

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

    # CL hook: `cl_method is None` (and any trace-time branch inside
    # loss_penalty, e.g. an empty cl_state on the first task) resolves at trace
    # time, so the no-CL compilation is byte-identical to plain PPO.
    def ppo_loss(params, cl_state, x, a, logp, mb_advantages, mb_returns):
        """Clipped PPO surrogate objective, plus value/entropy terms and optional CL penalty."""
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
        if cl_method is not None:
            cl_pen = cl_method.loss_penalty(params, cl_state)
            loss = loss + cl_pen
        else:
            cl_pen = jnp.array(0.0)
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl), cl_pen)

    # Differentiates w.r.t. argnums=0 (params) only; cl_state is a constant input.
    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        cl_state,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey, hook_key = jax.random.split(key, 3)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree.map(flatten, storage)
            shuffled_storage = jax.tree.map(convert_data, flatten_storage)
            # Per-minibatch keys for CL hooks (e.g. A-GEM memory sampling); dead code
            # when no method uses them.
            mb_keys = jax.random.split(hook_key, config["NUM_MINIBATCHES"])

            def update_minibatch(agent_state, xs):
                minibatch, mb_key = xs
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl, cl_pen)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    cl_state,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                if cl_method is not None:
                    grads, cl_metrics = cl_method.transform_grads(grads, agent_state.params, cl_state, mb_key)
                else:
                    cl_metrics = {}
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, grads) = jax.lax.scan(
                update_minibatch, agent_state, (shuffled_storage, mb_keys)
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=config["UPDATE_EPOCHS"]
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, key

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
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, cl_pen, cl_metrics, key = update_ppo(
            agent_state,
            storage,
            cl_state,
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
            **({f"{loss_section}/cl_penalty": cl_pen[-1, -1].item()} if cl_method is not None else {}),
            **{f"{loss_section}/{k}": v.mean().item() for k, v in cl_metrics.items()},
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

    if return_final_rollout:
        # One extra rollout with the *final* params so post-task CL state is built
        # from (s, a) ~ pi_theta exactly; the last training rollout was collected by
        # a slightly older policy. GAE is completed so storage.returns is usable as
        # a regression target.
        post_start = time.time()
        _, post_next_obs, post_next_done, post_storage, key, _, _ = rollout(
            agent_state, next_obs, next_done, key, env_state
        )
        post_storage = compute_gae(agent_state, post_next_obs, post_next_done, post_storage)
        post_storage = jax.block_until_ready(post_storage)
        key, post_key = jax.random.split(key)
        print(f"[train] final-policy rollout for CL state took {time.time() - post_start:.2f} seconds.")
        return agent_state.params, (post_storage, post_key)

    return agent_state.params
