"""Dev utility: dump the observation frame the PPO agent actually receives.

Saves two PNGs per run: the raw full-res render (straight from the game's
renderer, mod applied, before any wrapper touches it) and the "processed"
frame (last frame of the post-PixelObsWrapper stack - resized/grayscaled/
native-downscaled exactly as the network sees it). Compare the two to check
that a visual mod's effect still shows up after the wrapper pipeline.

Usage:
    python save_obs_frame.py --mods change_ball_color
    python save_obs_frame.py --mods change_score_color --steps 30
    python save_obs_frame.py  # base task, no mods
"""
import argparse
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from ppo_trainer import make_env


def _unwrap_to_env_state(state):
    """Walk LogState -> PixelState -> AtariState -> EnvState, same pattern as video_utils.py."""
    while hasattr(state, "atari_state"):
        state = state.atari_state
    if hasattr(state, "env_state"):
        state = state.env_state
    return state


def _save(array: np.ndarray, path: Path, upscale: int = 1):
    array = np.clip(np.array(array), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]  # grayscale: drop the size-1 channel dim for PIL
    image = Image.fromarray(array)
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), Image.NEAREST)
    image.save(path)
    print(f"Saved {array.shape} (x{upscale}) -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="pong")
    parser.add_argument("--mods", nargs="*", default=[])
    parser.add_argument("--steps", type=int, default=20, help="random steps taken before saving, so the frame isn't just the reset screen")
    parser.add_argument("--native-downscaling", dest="native_downscaling", action="store_true", default=True)
    parser.add_argument("--no-native-downscaling", dest="native_downscaling", action="store_false")
    parser.add_argument("--smooth-image", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="obs_frames")
    parser.add_argument("--upscale", type=int, default=4, help="nearest-neighbor scale factor, purely for easier viewing")
    args = parser.parse_args()

    env = make_env(
        args.env_id,
        seed=args.seed,
        num_envs=1,
        mods=list(args.mods),
        pixel_based=True,
        native_downscaling=args.native_downscaling,
        smooth_image=args.smooth_image,
        eval=True,  # make_env only drops mods when not eval and len(mods) > 1
    )()

    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)

    for _ in range(args.steps):
        key, action_key = jax.random.split(key)
        action = jax.random.randint(action_key, (), 0, env.action_space().n)
        obs, state, reward, terminated, truncated, info = env.step(state, action)

    processed_frame = np.array(obs[-1])  # most recent frame in the (F, H, W, C) stack
    raw_frame = np.array(env.render(_unwrap_to_env_state(state)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod_label = "base" if not args.mods else "_".join(args.mods)

    _save(processed_frame, out_dir / f"{args.env_id}_{mod_label}_processed.png", args.upscale)
    _save(raw_frame, out_dir / f"{args.env_id}_{mod_label}_raw.png", args.upscale)


if __name__ == "__main__":
    main()
