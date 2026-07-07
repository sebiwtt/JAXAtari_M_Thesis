import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jaxatari.games.jax_pong import PongState, JaxPong
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
import chex
from jaxatari.environment import JAXAtariAction as Action
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir


def _recolor_sprite(filename: str, original_rgb: tuple, new_rgb: tuple) -> np.ndarray:
    """Load a pong sprite .npy and replace original_rgb with new_rgb (alpha preserved)."""
    sprite_path = os.path.join(get_base_sprite_dir(), "pong", filename)
    sprite = np.load(sprite_path).copy()
    original = np.array([*original_rgb, 255], dtype=np.uint8)
    replacement = np.array([*new_rgb, 255], dtype=np.uint8)
    mask = np.all(sprite == original, axis=-1)
    sprite[mask] = replacement
    return sprite


def _make_recolored_background(new_color: tuple) -> np.ndarray:
    return _recolor_sprite("background.npy", (144, 72, 17), new_color)


def _make_recolored_digits(pattern: str, original_rgb: tuple, new_rgb: tuple) -> np.ndarray:
    """Load the 10 digit sprites, recolor them, and center-pad to uniform dims (mirrors the base digit loader)."""
    digits = [_recolor_sprite(pattern.format(i), original_rgb, new_rgb) for i in range(10)]
    max_h = max(d.shape[0] for d in digits)
    max_w = max(d.shape[1] for d in digits)
    padded = []
    for digit in digits:
        pad_h = max_h - digit.shape[0]
        pad_w = max_w - digit.shape[1]
        padded.append(np.pad(
            digit,
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="constant",
            constant_values=0,
        ))
    return np.stack(padded)

# --- 1. Individual Mod Plugins ---
class LazyEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["random_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        Access the environment via self._env (set by JaxAtariModController).
        """
        should_move = (state.step_counter % 8 != 0) & (state.ball_vel_x < 0)
        direction = jnp.sign(state.ball_y - state.enemy_y)
        new_y = state.enemy_y + (direction * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        final_y = jax.lax.cond(should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        return state.replace(enemy_y=final_y.astype(jnp.int32))

class RandomEnemyMod(JaxAtariInternalModPlugin):
    #conflicts_with = ["lazy_enemy"]

    @partial(jax.jit, static_argnums=(0,))
    def _enemy_step(self, state: PongState) -> PongState:
        """
        Replaces the base _enemy_step logic.
        'self_env' is the bound JaxPong instance.
        'key' is now used for randomness.
        """
        # Split key: use one part for randomness, keep remainder for state
        rng_key, unused_key = jax.random.split(state.key)
        random_dir = jax.random.choice(rng_key, jnp.array([-1, 1]))
        random_cond = state.step_counter % 3 == 0
        new_y = state.enemy_y + (random_dir * self._env.consts.ENEMY_STEP_SIZE).astype(jnp.int32)

        # Clamp to screen bounds
        new_y = jnp.clip(
            new_y,
            self._env.consts.WALL_TOP_Y + self._env.consts.WALL_TOP_HEIGHT - 10,
            self._env.consts.WALL_BOTTOM_Y - 4,
        )

        final_y = jax.lax.cond(random_cond, lambda _: new_y, lambda _: state.enemy_y, operand=None)
        # Return unused_key; step() will replace with new_state_key at the end
        return state.replace(enemy_y=final_y.astype(jnp.int32), key=unused_key)



class AlwaysZeroScoreMod(JaxAtariPostStepModPlugin):    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        return new_state.replace(
            player_score=jnp.array(0, dtype=jnp.int32),
            enemy_score=jnp.array(0, dtype=jnp.int32)
        )
    

class LinearMovementMod(JaxAtariInternalModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def _player_step(self, state: PongState, action: chex.Array) -> PongState:
        up = jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE)
        down = jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE)

        # Direct movement: move 2 pixels per frame when input pressed
        move_amount = jnp.array(2.0, dtype=jnp.float32)

        new_player_y = state.player_y
        new_player_y = jax.lax.cond(
            up,
            lambda y: y - move_amount,
            lambda y: y,
            operand=new_player_y,
        )

        new_player_y = jax.lax.cond(
            down,
            lambda y: y + move_amount,
            lambda y: y,
            operand=new_player_y,
        )

        # Hard boundaries using the analog paddle limits
        new_player_y = jnp.clip(
            new_player_y,
            self._env.consts.PADDLE_MIN_Y,
            self._env.consts.PADDLE_MAX_Y,
        )

        return state.replace(
            player_y=new_player_y,
            player_speed=jnp.array(0.0, dtype=jnp.float32),
        )

class ShiftPlayerMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "PLAYER_X": 136,
    }

class ShiftEnemyMod(JaxAtariInternalModPlugin):
    constants_overrides = {
        "ENEMY_X": 20,
    }


class NoFireMod(JaxAtariInternalModPlugin):
    attribute_overrides = {
        "ACTION_SET": jnp.array([Action.NOOP, Action.RIGHT, Action.LEFT], dtype=jnp.int32),
    }


class ChangeBackgroundColorMod(JaxAtariInternalModPlugin):
    """Changes the playfield background color. Default: navy blue (0, 0, 128)."""
    _NEW_BG_COLOR = (0, 0, 128)

    constants_overrides = {"BACKGROUND_COLOR": _NEW_BG_COLOR}
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_recolored_background(_NEW_BG_COLOR),
        }
    }


class ChangePlayerColorMod(JaxAtariInternalModPlugin):
    """Changes the player paddle color. Default: red (255, 0, 0)."""
    _NEW_PLAYER_COLOR = (255, 0, 0)

    constants_overrides = {"PLAYER_COLOR": _NEW_PLAYER_COLOR}
    asset_overrides = {
        "player": {
            "name": "player",
            "type": "single",
            "data": _recolor_sprite("player.npy", (92, 186, 92), _NEW_PLAYER_COLOR),
        }
    }


class SwapPaddleColorsMod(JaxAtariInternalModPlugin):
    """Swaps the paddle colors: player becomes orange, enemy becomes green."""
    _PLAYER_RGB = (92, 186, 92)
    _ENEMY_RGB = (213, 130, 74)

    constants_overrides = {
        "PLAYER_COLOR": _ENEMY_RGB,
        "ENEMY_COLOR": _PLAYER_RGB,
    }
    asset_overrides = {
        "player": {
            "name": "player",
            "type": "single",
            "data": _recolor_sprite("player.npy", _PLAYER_RGB, _ENEMY_RGB),
        },
        "enemy": {
            "name": "enemy",
            "type": "single",
            "data": _recolor_sprite("enemy.npy", _ENEMY_RGB, _PLAYER_RGB),
        },
    }


class ChangeBallColorMod(JaxAtariInternalModPlugin):
    """Changes the ball color. Default: yellow (255, 255, 0)."""
    _NEW_BALL_COLOR = (255, 255, 0)

    constants_overrides = {"BALL_COLOR": _NEW_BALL_COLOR}
    asset_overrides = {
        "ball": {
            "name": "ball",
            "type": "single",
            "data": _recolor_sprite("ball.npy", (236, 236, 236), _NEW_BALL_COLOR),
        }
    }


class ChangeScoreColorMod(JaxAtariInternalModPlugin):
    """Changes both score displays (green/orange digits) to a single color. Default: white (236, 236, 236)."""
    _NEW_SCORE_COLOR = (236, 236, 236)

    asset_overrides = {
        "player_digits": {
            "name": "player_digits",
            "type": "digits",
            "data": _make_recolored_digits("player_score_{}.npy", (92, 186, 92), _NEW_SCORE_COLOR),
        },
        "enemy_digits": {
            "name": "enemy_digits",
            "type": "digits",
            "data": _make_recolored_digits("enemy_score_{}.npy", (213, 130, 74), _NEW_SCORE_COLOR),
        },
    }


class GrayscaleThemeMod(JaxAtariInternalModPlugin):
    """
    Full monochrome theme: recolors every element to a distinct shade of gray.

    Shades are hand-picked (not a photometric luminance conversion) because the
    original player (92, 186, 92) and enemy (213, 130, 74) have nearly identical
    luminance (~147 vs ~148) and would collapse to the same gray. The chosen
    values keep every element legibly distinct, ordered dark background -> enemy
    -> walls/score -> player -> ball (brightest, easiest to track).
    """
    _ORIG_BACKGROUND = (144, 72, 17)
    _ORIG_PLAYER = (92, 186, 92)
    _ORIG_ENEMY = (213, 130, 74)
    _ORIG_BALL = (236, 236, 236)

    _GRAY_BACKGROUND = (34, 34, 34)
    _GRAY_PLAYER = (200, 200, 200)
    _GRAY_ENEMY = (120, 120, 120)
    _GRAY_BALL = (236, 236, 236)
    _GRAY_WALL = (170, 170, 170)

    constants_overrides = {
        "BACKGROUND_COLOR": _GRAY_BACKGROUND,
        "PLAYER_COLOR": _GRAY_PLAYER,
        "ENEMY_COLOR": _GRAY_ENEMY,
        "BALL_COLOR": _GRAY_BALL,
        # Walls are procedural from SCORE_COLOR; overriding it recolors them.
        "SCORE_COLOR": _GRAY_WALL,
        "WALL_COLOR": _GRAY_WALL,
    }
    asset_overrides = {
        "background": {
            "name": "background",
            "type": "background",
            "data": _make_recolored_background(_GRAY_BACKGROUND),
        },
        "player": {
            "name": "player",
            "type": "single",
            "data": _recolor_sprite("player.npy", _ORIG_PLAYER, _GRAY_PLAYER),
        },
        "enemy": {
            "name": "enemy",
            "type": "single",
            "data": _recolor_sprite("enemy.npy", _ORIG_ENEMY, _GRAY_ENEMY),
        },
        "ball": {
            "name": "ball",
            "type": "single",
            "data": _recolor_sprite("ball.npy", _ORIG_BALL, _GRAY_BALL),
        },
        "player_digits": {
            "name": "player_digits",
            "type": "digits",
            "data": _make_recolored_digits("player_score_{}.npy", _ORIG_PLAYER, _GRAY_PLAYER),
        },
        "enemy_digits": {
            "name": "enemy_digits",
            "type": "digits",
            "data": _make_recolored_digits("enemy_score_{}.npy", _ORIG_ENEMY, _GRAY_ENEMY),
        },
    }


class _BallSpeedMod(JaxAtariInternalModPlugin):
    """
    Base class for ball-speed mods. Reuses the unmodified base ball physics
    (JaxPong._ball_step) instead of reimplementing collision handling.

    - _SUBSTEPS > 1 speeds the ball up by running the base step that many times
      per frame. Each sub-step still moves only 1-4 px and runs full collision
      detection, so the ball never tunnels through a paddle.
    - _PERIOD > 1 slows the ball down by advancing it only once every _PERIOD
      frames (the ball is frozen on the other frames).
    """
    _SUBSTEPS: int = 1
    _PERIOD: int = 1

    @partial(jax.jit, static_argnums=(0,))
    def _ball_step(self, state: PongState, action: chex.Array) -> PongState:
        def advance(st: PongState) -> PongState:
            for _ in range(self._SUBSTEPS):
                st = JaxPong._ball_step(self._env, st, action)
            return st

        if self._PERIOD > 1:
            should_move = (state.step_counter % self._PERIOD) == 0
            return jax.lax.cond(should_move, advance, lambda st: st, state)
        return advance(state)


class FastBallMod(_BallSpeedMod):
    """Ball moves twice as fast (two collision-safe sub-steps per frame)."""
    _SUBSTEPS = 2


class SlowBallMod(_BallSpeedMod):
    """Ball moves at half speed (advances every other frame)."""
    _PERIOD = 2


# --- Magnitude-scaled ball speed (N collision-safe sub-steps per frame) ---
# x3-x5 also scale the enemy paddle speed (ENEMY_STEP_SIZE, base 2) by the same
# factor, so the built-in opponent can still track the faster ball and the game
# stays competitive. x2 leaves the enemy unchanged (it keeps up at that speed).
class BallSpeedX2Mod(_BallSpeedMod):
    """Ball speed x2."""
    _SUBSTEPS = 2
    constants_overrides = {"ENEMY_STEP_SIZE": 4}


class BallSpeedX3Mod(_BallSpeedMod):
    """Ball speed x3, enemy paddle speed scaled x3 to match."""
    _SUBSTEPS = 3
    constants_overrides = {"ENEMY_STEP_SIZE": 6}


class BallSpeedX4Mod(_BallSpeedMod):
    """Ball speed x4, enemy paddle speed scaled x4 to match."""
    _SUBSTEPS = 4
    constants_overrides = {"ENEMY_STEP_SIZE": 8}


class BallSpeedX5Mod(_BallSpeedMod):
    """Ball speed x5, enemy paddle speed scaled x5 to match."""
    _SUBSTEPS = 5
    constants_overrides = {"ENEMY_STEP_SIZE": 10}


class FastPaddleMod(JaxAtariInternalModPlugin):
    """
    Doubles the player paddle's top speed (5.75 -> 11.5).

    PADDLE_MAX_SPEED is the analog target speed the paddle accelerates toward in
    _player_step, and it also sets the threshold for the max-speed ball boost in
    _ball_step, so both scale together and the boost mechanic stays consistent.
    """
    constants_overrides = {"PADDLE_MAX_SPEED": 11.5}


class SlowPaddleMod(JaxAtariInternalModPlugin):
    """Halves the player paddle's top speed (5.75 -> 2.875)."""
    constants_overrides = {"PADDLE_MAX_SPEED": 2.875}


def _random_serve_velocity(key, vx_choices, vy_choices):
    """Pick a random (ball_vel_x, ball_vel_y) serve vector from the given choices."""
    kx, ky = jax.random.split(key)
    vel_x = jax.random.choice(kx, vx_choices).astype(jnp.int32)
    vel_y = jax.random.choice(ky, vy_choices).astype(jnp.int32)
    return vel_x, vel_y


class RandomServeMod(JaxAtariInternalModPlugin):
    """
    Randomizes the direction and angle of every serve.

    Base Pong serves deterministically: horizontal direction always points at
    whoever conceded and the vertical component is always +/-1 (a fixed 45 deg).
    This mod instead draws the serve independently at random:
      - direction: ball_vel_x in {-1, +1}  (served left or right, 50/50)
      - angle:     ball_vel_y in {-2, -1, +1, +2}  (up/down, two steepnesses)
    giving 8 equally likely serve vectors. Both the opening serve (reset) and
    every post-goal serve are covered. Integer velocities are kept small so
    collision detection stays exact (no tunneling).

    Randomness is drawn from state.key, which the base step() advances every
    frame, so each serve differs. The serve vector is chosen once on the goal
    frame and held through the 60-frame respawn pause before release.
    """
    _VEL_X_CHOICES = jnp.array([-1, 1], dtype=jnp.int32)
    _VEL_Y_CHOICES = jnp.array([-2, -1, 1, 2], dtype=jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = jax.random.PRNGKey(42)):
        obs, state = JaxPong.reset(self._env, key)
        vel_x, vel_y = _random_serve_velocity(state.key, self._VEL_X_CHOICES, self._VEL_Y_CHOICES)
        # ball velocity is not part of the observation, so obs is unaffected
        state = state.replace(ball_vel_x=vel_x, ball_vel_y=vel_y)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def _reset_ball_after_goal(self, state_and_goal):
        state, _scored_right = state_and_goal
        vel_x, vel_y = _random_serve_velocity(state.key, self._VEL_X_CHOICES, self._VEL_Y_CHOICES)
        return (
            jnp.array(self._env.consts.BALL_START_X).astype(jnp.int32),
            jnp.array(self._env.consts.BALL_START_Y).astype(jnp.int32),
            vel_x,
            vel_y,
        )


def _nudge_ball(state: PongState, dx: int, dy: int, buffer: int) -> PongState:
    """Shift the ball by (dx, dy) every `buffer` steps; identity otherwise.

    A constant *positional* pull (applied after the step) rather than a change to
    the integer velocity: this bends the ball's path like a steady drift/gravity
    without compounding into runaway speed, so bounce collisions stay exact. A
    1 px overshoot into the (10-16 px thick) walls is corrected by the next
    frame's bounce.
    """
    return jax.lax.cond(
        state.step_counter % buffer == 0,
        lambda s: s.replace(ball_x=s.ball_x + dx, ball_y=s.ball_y + dy),
        lambda s: s,
        operand=state,
    )


class BallGravityMod(JaxAtariPostStepModPlugin):
    """
    Pulls the ball toward the floor by 1 px every _BUFFER steps.

    In Pong y increases downward (top wall y=24, bottom wall y=194), so the pull
    is +y. The downward bias curves the ball's trajectory and, over a rally,
    steadily flattens its bounce angles toward the bottom.
    """
    _BUFFER = 4

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: PongState, new_state: PongState) -> PongState:
        return _nudge_ball(new_state, 0, 1, self._BUFFER)


class BallDriftMod(JaxAtariPostStepModPlugin):
    """
    Steadily drifts the ball sideways by _DIRECTION px every _BUFFER steps
    (default: +1 = toward the player's side on the right).
    """
    _BUFFER = 4
    _DIRECTION = 1

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: PongState, new_state: PongState) -> PongState:
        return _nudge_ball(new_state, self._DIRECTION, 0, self._BUFFER)


class ScaleRewardMod(JaxAtariInternalModPlugin):
    """
    Scales the environment reward by a constant factor (default 2x).

    Base reward is the per-step change in score differential (+1 when the player
    scores, -1 when the enemy scores). Patching _get_reward scales both signs
    symmetrically and leaves the game dynamics untouched. The mod wrapper
    recomputes reward via this same method after each step, so the scaled value
    is what the caller receives.
    """
    _SCALE = 2

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return JaxPong._get_reward(self._env, previous_state, state) * self._SCALE


class RewardPerHitMod(JaxAtariInternalModPlugin):
    """
    Replaces the reward: +_POINTS_PER_HIT each time the player paddle hits the
    ball, and 0 for everything else (scoring and conceding no longer reward).

    A player hit is the only event that reverses the ball's horizontal velocity
    from moving toward the player (>0, rightward) to moving away (<0). Serves and
    enemy-paddle hits never produce that +->- flip, and the extra 'ball in the
    player's half' guard keeps detection correct even alongside random_serve
    (which serves in a random direction). One hit -> one point (the flip happens
    on a single step), so rallies are rewarded independently of winning.
    """
    _POINTS_PER_HIT = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        player_hit = (
            (previous_state.ball_vel_x > 0)
            & (state.ball_vel_x < 0)
            & (state.ball_x > self._env.consts.BALL_START_X)
        )
        return (player_hit.astype(jnp.int32) * self._POINTS_PER_HIT)


class TimePenaltyMod(JaxAtariInternalModPlugin):
    """
    Adds a constant per-step time penalty to the reward to encourage fast
    scoring: reward = base goal reward (+1 score / -1 concede) - _PENALTY.

    The penalty is applied every step (including the respawn pauses, which are
    idle for the agent), so the episode return equals the final score
    differential minus _PENALTY * (number of steps).

    Tuning: for scoring to stay the dominant incentive, _PENALTY must be well
    below 1 / (steps per point). Rallies here run ~500-700 steps per point, so
    the default 0.001 keeps a scored point clearly net-positive (~+0.3 to +0.5)
    while rewarding faster scoring. Raising it toward ~0.01 makes points
    net-negative and perversely encourages losing quickly, so tune to the
    timescale of your agent.
    """
    _PENALTY = 0.001

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return JaxPong._get_reward(self._env, previous_state, state) - self._PENALTY


class AsymmetricRewardMod(JaxAtariInternalModPlugin):
    """
    Asymmetric reward: +1 when the player scores, but no -1 penalty when the
    enemy scores (conceding gives 0). Since conceding is the only source of
    negative reward in base Pong, this is just clipping the reward at 0. The
    game dynamics are untouched; only the reward signal changes.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return jnp.maximum(JaxPong._get_reward(self._env, previous_state, state), 0)


class InvertedRewardMod(JaxAtariInternalModPlugin):
    """
    Inverts the reward: +1 when the enemy scores, -1 when the player scores.
    The objective flips from winning to losing. Game dynamics are untouched;
    only the sign of the reward signal changes.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: PongState, state: PongState):
        return -JaxPong._get_reward(self._env, previous_state, state)


def apply_render_noise(raster: jnp.ndarray, key: chex.PRNGKey, level: float) -> jnp.ndarray:
    """Blend uniform pixel noise into a rendered frame.

    out = (1 - level) * image + level * uniform_noise, per channel, so `level`
    is the noise fraction (0 = clean, 1 = pure static). The noise is keyed on
    the frame's PRNG key (advanced every step), so it animates frame to frame.
    """
    noise = jax.random.uniform(key, raster.shape, minval=0.0, maxval=255.0)
    img = raster.astype(jnp.float32)
    out = (1.0 - level) * img + level * noise
    return jnp.clip(out, 0.0, 255.0).astype(jnp.uint8)


class _RenderNoiseMod(JaxAtariInternalModPlugin):
    """
    Base marker for magnitude-scaled render-noise mods. Adds uniform pixel noise
    to the rendered image (pixel observations), leaving game dynamics and the
    object-centric observation untouched.

    `render` cannot be patched by a plugin (it exists on both the env and the
    renderer, which the mod controller treats as ambiguous), so the noise is
    applied by PongEnvMod.render, which reads _NOISE_LEVEL from the active mod.
    """
    _NOISE_LEVEL: float = 0.0


class RenderNoise10Mod(_RenderNoiseMod):
    """20% render noise (80% image, 20% uniform static)."""
    _NOISE_LEVEL = 0.1


class RenderNoise20Mod(_RenderNoiseMod):
    """40% render noise."""
    _NOISE_LEVEL = 0.2


class RenderNoise30Mod(_RenderNoiseMod):
    """60% render noise."""
    _NOISE_LEVEL = 0.3


class RenderNoise40Mod(_RenderNoiseMod):
    """80% render noise (image barely visible under static)."""
    _NOISE_LEVEL = 0.4