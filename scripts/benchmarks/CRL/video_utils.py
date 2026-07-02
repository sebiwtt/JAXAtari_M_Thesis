"""Video capture and wandb-logging helpers for the PPO CRL fine-tuning run.

Two kinds of clips get produced over the course of a run:

1. Periodic eval videos - built from the env states that ``ppo_crl_eval.evaluate()``
   already produced while checking in on the policy mid-training.
2. Final videos - generated once training ends: one clip on the plain training
   environment, plus one per mod the agent was evaluated on, so a run gives a quick
   visual read on how the agent behaves across the whole CRL task sequence.

These live outside ``ppo_crl_finetune.py`` because they are not part of the actual PPO
algorithm - just observability - and pulling them out keeps the training script focused
on the training loop.
"""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb

import jaxatari


def log_periodic_eval_video(env_id: str, env_states, iteration: int):
    """Render the trajectory from a periodic ``evaluate()`` call and log it to wandb.

    We spin up a fresh renderer directly from ``jaxatari.make`` rather than reusing the
    training env's renderer, because the training env may be pixel-resized/smoothed for
    the agent's benefit - we want full-resolution frames for the video.
    """
    clean_renderer = jaxatari.make(env_id).renderer
    frames = jax.vmap(clean_renderer.render)(env_states)  # (N, H, W, C)
    frames = jnp.transpose(frames, (0, 3, 1, 2))  # wandb expects (N, C, H, W)
    video = wandb.Video(np.array(frames), fps=30, format="mp4")
    wandb.log({"eval/video": video}, step=iteration)
    print(f"Video (eval) logged to wandb with {frames.shape[0]} frames.")


def _generate_single_final_video(
    config: dict,
    network,
    actor,
    agent_state,
    make_env: Callable,
    mods_config,
    video_label: str,
    video_index: int = 0,
):
    """Greedily roll out the trained policy on one env configuration and upload the clip.

    Unlike the periodic eval video, this uses a plain Python loop instead of
    ``jax.lax.scan`` so we can break out as soon as the episode ends instead of having to
    fix the rollout length ahead of time.
    """
    if not config["CAPTURE_VIDEO"]:
        return None

    # A second, unwrapped env instance so the renderer isn't affected by any
    # pixel-resizing/smoothing the training wrappers apply for the agent's benefit.
    renderer_local = jaxatari.make(config["ENV_ID"]).renderer
    env = make_env(
        config["ENV_ID"],
        config["SEED"],
        1,
        mods_config,
        config["PIXEL_BASED"],
        config["NATIVE_DOWNSCALING"],
        config["SMOOTH_IMAGE"],
        eval=True,
    )()

    rng = jax.random.PRNGKey(config["SEED"] + video_index * 10000)
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    obs = obs.squeeze()  # (F, H, W)

    frames = []
    total_reward = 0.0
    max_steps = 5000

    for _ in range(max_steps):
        # The network expects a batch dimension: (B, F, H, W).
        policy_obs = obs[None, ...]
        hidden = network.apply(agent_state.params.network_params, policy_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        action = jnp.argmax(logits, axis=-1)[0]  # greedy action, no exploration for videos

        rng, _ = jax.random.split(rng)
        obs, env_state, reward, terminated, truncated, _ = env.step(env_state, action)
        done = jnp.logical_or(terminated, truncated)
        obs = obs.squeeze()
        total_reward += float(reward)

        # Wrappers nest the underlying Atari state several layers deep; unwrap down to
        # the state the base renderer actually knows how to draw.
        state_for_render = env_state
        while hasattr(state_for_render, "atari_state"):
            state_for_render = state_for_render.atari_state
        if hasattr(state_for_render, "env_state"):
            state_for_render = state_for_render.env_state

        frames.append(np.array(renderer_local.render(state_for_render), dtype=np.uint8))

        if bool(done):
            break

    print(f"Final video ({video_label}): {len(frames)} frames, total reward: {total_reward:.1f}")

    if len(frames) > 0:
        frames = np.stack(frames, axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        video = wandb.Video(frames, fps=30, format="mp4")
        wandb.log(
            {
                f'final_video_seed{config["SEED"]}_{video_label}': video,
                f'final_return_seed{config["SEED"]}_{video_label}': total_reward,
            },
        )
        print(f"Video '{video_label}' logged to wandb.")

    return total_reward


def generate_final_video(config: dict, network, actor, agent_state, make_env: Callable):
    """Generate and upload one video per env configuration once training has finished.

    Always includes the plain training environment first, then one clip per CRL mod the
    agent was evaluated on (falling back to the training mods if no eval mods are set).
    """
    if not config["CAPTURE_VIDEO"]:
        return

    print(f'Generating final videos for seed {config["SEED"]}...')

    video_configs = [([], "train")]

    eval_mods = config["EVAL_MODS"] if len(config["EVAL_MODS"]) > 0 else config["TRAIN_MODS"]
    for mod in list(eval_mods):
        mods_config = [mod] if not isinstance(mod, (list, tuple)) else list(mod)
        mod_label = mod if isinstance(mod, str) else "_".join(str(m) for m in mods_config)
        video_configs.append((mods_config, mod_label))

    for video_index, (mods_config, video_label) in enumerate(video_configs):
        _generate_single_final_video(
            config, network, actor, agent_state, make_env, mods_config, video_label, video_index
        )
