"""Final-video capture and wandb logging, kept out of ppo_trainer.py since it's
observability, not part of the PPO algorithm itself."""
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from PIL import Image

import jaxatari


def save_obs_debug_frame(config: dict, obs, label: str, out_dir: str = "obs_frames"):
    """Dumps env 0's most recent stacked frame (post-PixelObsWrapper - exactly what the
    CNN receives) to a PNG, so a human can eyeball what a task's mod actually looks like
    once it reaches the agent during a real benchmark run.

    No-op unless config["SAVE_OBS_FRAMES"] is truthy; object-centric runs have no image
    to dump.
    """
    if not config.get("SAVE_OBS_FRAMES", False) or not config["PIXEL_BASED"]:
        return

    frame = np.clip(np.array(obs[0, -1]), 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]  # grayscale: drop the size-1 channel dim for PIL

    image = Image.fromarray(frame)
    upscale = 4  # purely for easier viewing; the real observation is tiny (e.g. 84x84)
    image = image.resize((image.width * upscale, image.height * upscale), Image.NEAREST)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"{label}.png"
    image.save(path)
    print(f"[debug] saved obs frame -> {path}")


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
    """Greedily roll out the trained policy on one env config and upload the clip.

    Plain Python loop (not jax.lax.scan) so it can break as soon as the episode ends.
    """
    if not config["CAPTURE_VIDEO"]:
        return None

    # Unwrapped env so the renderer isn't affected by the training wrappers'
    # pixel-resizing/smoothing.
    renderer_local = jaxatari.make(config["ENV_ID"]).renderer
    env = make_env(
        config["ENV_ID"],
        config["SEED"],
        1,
        mods_config,
        config["PIXEL_BASED"],
        config["NATIVE_DOWNSCALING"],
        config["SMOOTH_IMAGE"],
        config["GRAYSCALE"],
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
        policy_obs = obs[None, ...]  # network expects a batch dim: (B, F, H, W)
        hidden = network.apply(agent_state.params.network_params, policy_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        action = jnp.argmax(logits, axis=-1)[0]  # greedy action, no exploration for videos

        rng, _ = jax.random.split(rng)
        obs, env_state, reward, terminated, truncated, _ = env.step(env_state, action)
        done = jnp.logical_or(terminated, truncated)
        obs = obs.squeeze()
        total_reward += float(reward)

        # Unwrap to the state the base renderer knows how to draw.
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
    """Generate and upload one video of the trained policy on its own training environment
    (whatever single mod, if any, TRAIN_MODS set for this task)."""
    if not config["CAPTURE_VIDEO"]:
        return

    print(f'Generating final video for seed {config["SEED"]}...')

    mods_config = list(config["TRAIN_MODS"])
    video_label = "base" if len(mods_config) == 0 else str(mods_config[0])

    _generate_single_final_video(config, network, actor, agent_state, make_env, mods_config, video_label)
