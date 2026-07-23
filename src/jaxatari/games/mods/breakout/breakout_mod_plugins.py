import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_breakout import BreakoutState


class SpeedModeMod(JaxAtariInternalModPlugin):
    """Increase speed to maximum at all time steps."""
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_ball_velocity(self, speed_idx, direction_idx, step_counter):
        """Returns the ball's velocity based on the speed and direction indices."""
        # Override to always return maximum speed
        direction = self._env.consts.BALL_DIRECTIONS[direction_idx]
        speed = 3
        return speed * direction[0], speed * direction[1]


class SmallPaddleMod(JaxAtariInternalModPlugin):
    """Always use a small paddle."""
    
    constants_overrides = {
        "PLAYER_SIZE": (4, 4),
        "PLAYER_SIZE_SMALL": (4, 4),
    }


class BigPaddleMod(JaxAtariInternalModPlugin):
    """Always use a bigger paddle."""
    
    constants_overrides = {
        "PLAYER_SIZE": (40, 4),
        "PLAYER_SIZE_SMALL": (40, 4),
    }


class BallDriftMod(JaxAtariPostStepModPlugin):
    """Consistently drift the ball to the right.

    Drifts +1 px every _drift_buffer steps (here every 2 -> ~0.5 px/frame rightward
    bias). Kept at 2 rather than 1: at every-step the +1 fully cancels the ball's
    leftward vx (1-2 px/frame) and the ball pins against the right wall; every 2
    steps is a strong but survivable lean that the ball can still fight leftward.
    """

    # Drift every 2 steps, direction 1 (right)
    _drift_buffer = 2
    _direction = 1
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BreakoutState, new_state: BreakoutState) -> BreakoutState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        """
        # Drift the ball one step every drift_buffer steps
        # This affects the next step
        return jax.lax.cond(
            new_state.step_counter % self._drift_buffer == 0,
            lambda s: s.replace(ball_x=s.ball_x + self._direction),
            lambda s: s,
            operand=new_state
        )


class BallGravityMod(JaxAtariPostStepModPlugin):
    """
    Pulls the ball down.
    The ball will be pulled down by 1 every gravity_buffer steps. The direction is 1 for down, -1 for up.
    """
    
    # Default: gravity every 4 steps, direction -1 (down, but negative because y increases downward)
    _gravity_buffer = 4
    _direction = -1
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BreakoutState, new_state: BreakoutState) -> BreakoutState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        """
        # Pull the ball down by 1 every gravity_buffer steps
        # This affects the next step
        return jax.lax.cond(
            new_state.step_counter % self._gravity_buffer == 0,
            lambda s: s.replace(ball_y=s.ball_y + self._direction),
            lambda s: s,
            operand=new_state
        )


def recolor_4d_sprite(sprite_array: jnp.ndarray, new_rgb_color: jnp.ndarray) -> jnp.ndarray:
    """
    Recolors the non-transparent pixels of a 4D RGBA sprite array.

    Args:
        sprite_array: The input array with shape (Frame, H, W, 4).
        new_rgb_color: A 3-element array for the new RGB color.

    Returns:
        A new array with the sprite recolored.
    """
    # Create a mask from the alpha channel (the 4th channel)
    is_visible = sprite_array[:, :, :, 3] > 0

    # Use the mask to set the RGB values (:3) of visible pixels
    recolored_array = sprite_array.at[is_visible, :3].set(new_rgb_color)
    
    return recolored_array


class BallColorMod(JaxAtariInternalModPlugin):
    """Changes the balls color to a set color"""
    
    # Default color is yellow, but this can be overridden via constants
    constants_overrides = {
        "BALL_COLOR": (255, 255, 0),  # Yellow by default
    }
    


class BlockColorMod(JaxAtariInternalModPlugin):
    """Changes the blocks color to a set color"""
    
    # Default color is yellow
    constants_overrides = {
        "BLOCK_COLORS": [
            (255, 255, 0),  # All rows yellow
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
            (255, 255, 0),
        ]
    }


class PlayerColorMod(JaxAtariInternalModPlugin):
    """Changes the player color to a set color"""

    # Default color is yellow
    constants_overrides = {
        "PLAYER_COLOR": (255, 255, 0),  # Yellow by default
    }


# --- "change_X_color" visual mods (parallels to Pong/Freeway) ----------------
import os
import numpy as np
from jaxatari.rendering.jax_rendering_utils import JaxRenderingUtils, RendererConfig, get_base_sprite_dir

_jr = JaxRenderingUtils(RendererConfig())
_sprite_dir = os.path.join(get_base_sprite_dir(), "breakout")

# New colors (tweak here). Each mod recolors only its own element.
_NEW_PADDLE_COLOR = (80, 120, 255)     # blue
_NEW_BALL_COLOR = (236, 236, 236)      # white
_NEW_BG_COLOR = (20, 20, 60)           # dark navy (play area only; grey walls unaffected)
_NEW_SCORE_COLOR = (0, 210, 0)         # green


class ChangePaddleColorMod(JaxAtariInternalModPlugin):
    """Changes the paddle color (default: blue). The renderer recolors the paddle
    sprite from PLAYER_COLOR."""
    constants_overrides = {"PLAYER_COLOR": _NEW_PADDLE_COLOR}


class ChangeBallColorMod(JaxAtariInternalModPlugin):
    """Changes the ball color (default: white). The renderer recolors the ball
    sprite from BALL_COLOR."""
    constants_overrides = {"BALL_COLOR": _NEW_BALL_COLOR}


_recolored_bg = _jr.perform_recoloring(
    _jr.loadFrame(os.path.join(_sprite_dir, "background.npy")),
    [{'source': (0, 0, 0), 'target': _NEW_BG_COLOR}],  # black play area -> navy; grey walls kept
)


class ChangeBackgroundColorMod(JaxAtariInternalModPlugin):
    """Changes the background (play-area) color (default: navy). The grey walls
    are left unchanged; empty block cells keep the recolored background."""
    asset_overrides = {
        'background': {'name': 'background', 'type': 'background', 'data': _recolored_bg}
    }


_recolored_score = _jr.perform_recoloring(
    _jr._load_and_pad_digits_from_paths([os.path.join(_sprite_dir, f"score_{i}.npy") for i in range(10)]),
    [{'source': (142, 142, 142), 'target': _NEW_SCORE_COLOR}],
)


class SwapScoreColorMod(JaxAtariInternalModPlugin):
    """Changes the score digit color (default: green). All numeric displays share
    the score_digits sprite, so the score, lives, and player-count all change."""
    asset_overrides = {
        'score_digits': {'name': 'score_digits', 'type': 'digits', 'data': _recolored_score}
    }


_greyscale_bg = _jr.perform_recoloring(
    _jr.loadFrame(os.path.join(_sprite_dir, "background.npy")),
    [
        {'source': (66, 158, 130), 'target': (120, 120, 120)},   # teal logo pixels -> grey
        {'source': (200, 72, 72), 'target': (150, 150, 150)},    # red logo pixels -> grey
    ],
)


class GreyscaleThemeMod(JaxAtariInternalModPlugin):
    """
    Full monochrome theme: recolors the paddle, ball, and the six block rows to
    distinct shades of grey. The score digits and walls are already grey and the
    playfield is black; the background sprite's coloured logo pixels are greyed
    out via an asset override so the whole scene is neutral. Block rows keep six
    distinct shades so they stay legible.
    """
    constants_overrides = {
        "PLAYER_COLOR": (180, 180, 180),
        "BALL_COLOR": (235, 235, 235),
        "BLOCK_COLORS": [
            (205, 205, 205),   # row 0 (top)
            (180, 180, 180),
            (155, 155, 155),
            (130, 130, 130),
            (105, 105, 105),
            (80, 80, 80),      # row 5 (bottom)
        ],
    }
    asset_overrides = {
        'background': {'name': 'background', 'type': 'background', 'data': _greyscale_bg}
    }


def _make_oval_paddle() -> "np.ndarray":
    """Reshape the paddle sprite into an oval/lozenge: full width across the
    middle rows, tapered (inset) on the top and bottom rows. Purely visual -- the
    sprite keeps its footprint and the collision hitbox (PLAYER_SIZE) is unchanged."""
    sprite = _jr.loadFrame(os.path.join(_sprite_dir, "player.npy"))
    sprite = np.asarray(sprite).copy()          # (H=4, W=16, 4)
    h, w = sprite.shape[:2]
    inset = 2
    keep = np.zeros((h, w), dtype=bool)
    keep[1:h - 1, :] = True                     # middle rows: full width
    keep[0, inset:w - inset] = True             # top row: inset
    keep[h - 1, inset:w - inset] = True         # bottom row: inset
    sprite[(~keep) & (sprite[..., 3] > 0)] = 0  # drop the corners -> transparent
    return sprite


class RoundPaddleMod(JaxAtariInternalModPlugin):
    """
    Reshapes the paddle into an oval/lozenge (rounded ends, full-width middle).
    Purely visual: the sprite keeps its 16x4 footprint and the collision hitbox
    (PLAYER_SIZE) is unchanged, so the ball still bounces off the full rectangle.
    """
    asset_overrides = {
        'player': {'name': 'player', 'type': 'single', 'data': _make_oval_paddle()}
    }


# --- Dynamics mods -----------------------------------------------------------
class FasterBallMod(JaxAtariInternalModPlugin):
    """
    Makes the ball much faster by overriding the BALL_VELOCITIES_ABS speed table.

    The lever is the *horizontal* component: vx is pushed up to 4 px/frame (base
    tops out at 2) so the ball zips side-to-side far quicker and is much harder to
    intercept. The vertical component is deliberately held at <= 3 px/frame -- block
    collisions only bounce within 4 px of a block edge and the ball moves its whole
    velocity in one frame (no sub-stepping), so a vy of 4+ would let it tunnel
    through blocks. Side walls *clamp* the ball rather than letting it pass, so the
    high vx is safe. Table shape is [speed_idx, step_counter % 2, (vx, vy)]; the two
    sub-rows let a speed index average a fractional px/frame.
    """
    constants_overrides = {
        "BALL_VELOCITIES_ABS": jnp.array([
            [[3, 2], [3, 2]],   # speed 0: vx 3, vy 2
            [[4, 2], [3, 2]],   # speed 1: vx ~3.5, vy 2
            [[4, 3], [4, 2]],   # speed 2: vx 4, vy ~2.5
            [[4, 3], [4, 3]],   # speed 3: vx 4, vy 3
            [[4, 3], [4, 3]],   # speed 4: vx 4, vy 3 (vy capped at 3 to avoid tunneling)
        ])
    }


class SlowerBallMod(JaxAtariInternalModPlugin):
    """
    Makes the ball slower by flattening the BALL_VELOCITIES_ABS speed table to a
    constant 1 px/frame, so the ball never accelerates (the default speeds up to
    3 px/frame as blocks are cleared and on long rallies).

    1 px/frame is the floor: going sub-pixel would require 0-velocity frames,
    which make the ball stick and oscillate against the side walls (the wall
    bounce re-reverses whenever a velocity component is 0).
    """
    constants_overrides = {
        "BALL_VELOCITIES_ABS": jnp.array([
            [[1, 1], [1, 1]],   # every speed index -> constant 1 px/frame
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ])
    }


class FasterPaddleMod(JaxAtariInternalModPlugin):
    """
    Makes the paddle drastically faster and twitchier -- a large shift in control
    dynamics rather than a subtle tweak. PLAYER_ACCELERATION[0] is the first-frame
    speed after a direction change, so raising it to 14 (base is 3) flings the
    paddle ~14 px on the first input, over a tenth of its 128 px range. The top
    speed is raised to match so the burst isn't immediately clamped. The paddle
    position is clipped to the play area, so the high values stay in bounds.
    """
    constants_overrides = {
        "PLAYER_MAX_SPEED": 24,
        "PLAYER_ACCELERATION": jnp.array([14, 10, 7, 5, 5]),     # first-frame speed 14 (was 3)
        "PLAYER_WALL_ACCELERATION": jnp.array([6, 5, 4, 4, 4]),  # was [1, 2, 1, 1, 1]
    }


class SlowerPaddleMod(JaxAtariInternalModPlugin):
    """
    Makes the paddle noticeably slower from the first frame: the acceleration ramp
    is flattened to 1/frame (PLAYER_ACCELERATION[0] = 1, so it creeps rather than
    jumping to 3) and the top speed is halved.
    """
    constants_overrides = {
        "PLAYER_MAX_SPEED": 3,
        "PLAYER_ACCELERATION": jnp.array([1, 1, 1, 1, 1]),       # base speed 1 (was 3)
        "PLAYER_WALL_ACCELERATION": jnp.array([1, 1, 1, 1, 1]),
    }


class RandomServeMod(JaxAtariPostStepModPlugin):
    """
    Randomizes the ball's serve at each launch: a random horizontal direction
    (left/right) and a random starting angle/speed across the *full* speed table
    (index 0-4, vs the base game's fixed slow serve). The two downward directions
    (0=down-right, 1=down-left) combined with the five speed indices give a wide
    spread of serve angles -- from shallow-slow to steep-fast -- so the launch is
    genuinely unpredictable. The ball still always serves downward toward the
    paddle. Breakout's state carries no PRNG key, so randomness is seeded from
    step_counter (different at every launch).
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: BreakoutState, new_state: BreakoutState) -> BreakoutState:
        # A serve is the frame the round starts (game_started 0 -> 1, i.e. FIRE
        # after a spawn or a lost life).
        serve = jnp.logical_and(
            jnp.logical_not(prev_state.game_started.astype(jnp.bool_)),
            new_state.game_started.astype(jnp.bool_),
        )
        key = jax.random.PRNGKey(new_state.step_counter.astype(jnp.uint32))
        k_dir, k_speed = jax.random.split(key)
        rand_dir = jax.random.randint(k_dir, (), 0, 2).astype(new_state.ball_direction_idx.dtype)      # 0=down-right, 1=down-left
        rand_speed = jax.random.randint(k_speed, (), 0, 5).astype(new_state.ball_speed_idx.dtype)      # full 0-4 angle/speed variety
        vx, vy = self._env._get_ball_velocity(rand_speed, rand_dir, new_state.step_counter)
        vx = vx.astype(new_state.ball_vel_x.dtype)
        vy = vy.astype(new_state.ball_vel_y.dtype)
        return new_state.replace(
            ball_direction_idx=jnp.where(serve, rand_dir, new_state.ball_direction_idx),
            ball_speed_idx=jnp.where(serve, rand_speed, new_state.ball_speed_idx),
            ball_vel_x=jnp.where(serve, vx, new_state.ball_vel_x),
            ball_vel_y=jnp.where(serve, vy, new_state.ball_vel_y),
        )


# --- Reward mods -------------------------------------------------------------
class BallLossPenaltyMod(JaxAtariInternalModPlugin):
    """
    Penalizes losing the ball, on top of the normal brick-breaking reward, to shift
    the optimum from pure brick greed toward keeping the ball alive.

    The penalty is *sustained*, not a single frame: -1 every frame the ball is out
    of play after a life was lost (game_started == 0 and lives < NUM_LIVES), until
    the agent serves again. Under the benchmark's sign-clipped training reward a
    one-frame -1 on the death frame would be a single negative window per death,
    swamped by the ~100 positive brick windows; sustaining it across the re-serve
    wait makes each death cost several negative windows and additionally pressures
    prompt re-serving. The very first serve of the game (lives == NUM_LIVES) is
    excluded so the opening isn't penalized.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, state: BreakoutState):
        base = state.score - previous_state.score
        waiting_after_death = jnp.logical_and(
            state.game_started == 0,
            state.lives < self._env.consts.NUM_LIVES,
        )
        return (base - waiting_after_death.astype(jnp.int32)).astype(jnp.int32)


class FlattenRowValuesMod(JaxAtariInternalModPlugin):
    """
    Every brick pays +1, removing the base game's top-row-worth-more scheme
    (7/4/1 by row). Exactly one brick breaks per step, so a positive score delta
    means one brick fell regardless of its value.

    NOTE: under sign-clipped training reward this is a NO-OP -- the base game's
    7/4/1 all clip to +1 already, so the clipped stream is identical to base. Use
    only with reward clipping disabled. For a clip-surviving row mod see
    TopRowOnlyMod / BottomRowFirstMod, which zero or negate some rows.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, state: BreakoutState):
        broke_brick = state.score > previous_state.score
        return broke_brick.astype(jnp.int32)


class TopRowOnlyMod(JaxAtariInternalModPlugin):
    """
    Only TOP-row bricks score (+1); middle and bottom rows pay nothing (0). The base
    score delta reveals the row (7=top, 4=middle, 1=bottom). This is the opposite
    priority to BottomRowFirstMod, and -- unlike an all-positive "top worth more"
    scheme, which sign-clipping would collapse back to the base "+1 per brick" -- it
    zeroes the lower rows so the clipped signal genuinely differs from base: the
    agent is rewarded only for digging up to and clearing the top rows.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, state: BreakoutState):
        delta = state.score - previous_state.score
        return (delta == 7).astype(jnp.int32)   # +1 only for top-row bricks


class EveryKContactsMod(JaxAtariInternalModPlugin):
    """
    Rewards +1 on every _K-th paddle contact and 0 for breaking bricks -- so the
    objective becomes sustained ball control rather than destruction. A contact is
    a frame where consecutive_paddle_hits increments (the running count of paddle
    bounces, which the base game resets when a ball is lost).
    """
    _K = 3

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, state: BreakoutState):
        contact = state.consecutive_paddle_hits > previous_state.consecutive_paddle_hits
        credited = jnp.logical_and(contact, (state.consecutive_paddle_hits % self._K) == 0)
        return credited.astype(jnp.int32)


class BottomRowFirstMod(JaxAtariInternalModPlugin):
    """
    Sign-inverts the row incentive so it survives the benchmark's reward clipping:
    breaking a TOP-row brick is actively penalized (-1), MIDDLE rows are neutral (0),
    and BOTTOM rows are rewarded (+1). The base score delta reveals the row (7=top,
    4=middle, 1=bottom). Unlike a positive-only "bottom worth more" scheme -- which
    sign-clipping would collapse back to the base "+1 per brick" -- the negative
    top-row reward changes the clipped signal, pushing the agent to clear from the
    bottom up and avoid disturbing the top rows.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: BreakoutState, state: BreakoutState):
        delta = state.score - previous_state.score
        reward = jnp.where(delta == 7, -1,              # top rows -> -1 (penalized)
                  jnp.where(delta == 1, 1, 0))          # bottom -> +1, else (incl. middle=4) 0
        return reward.astype(jnp.int32)

