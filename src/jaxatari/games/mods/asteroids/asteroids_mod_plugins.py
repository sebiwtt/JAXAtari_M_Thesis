import jax.numpy as jnp
import numpy as np
import os
from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_asteroids import AsteroidsState, JaxAsteroids
from jaxatari.environment import JAXAtariAction as Action
import jax.lax
from functools import partial
import jax
from jaxatari.rendering.jax_rendering_utils import get_base_sprite_dir

class DontShootMod(JaxAtariPostStepModPlugin):
    """
    Mod that provides a reward of 20 every 300 frames but penalizes shooting by -5.
    This mod updates the state's score to reflect these changes.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: AsteroidsState, new_state: AsteroidsState):
        periodic_reward = jnp.where(
            (new_state.step_counter % 300 == 0) & (new_state.step_counter > 0),
            20,
            0
        )
        
        missile_lifespan = self._env.consts.MISSILE_LIFESPAN
        shot_fired = jnp.any(
            (new_state.missile_states[:, 5] == missile_lifespan) & 
            (prev_state.missile_states[:, 5] != missile_lifespan)
        )
        penalty = jnp.where(shot_fired, -5, 0)
        
        return new_state.replace(
            score=new_state.score + periodic_reward + penalty
        )

def _recolor_all(sprite: np.ndarray, new_rgb: tuple) -> np.ndarray:
    """Replaces all non-transparent pixels with new_rgb."""
    sprite = sprite.copy()
    if sprite.shape[-1] == 3:
        is_transparent = (sprite[:, :, 0] == 0) & (sprite[:, :, 1] == 0) & (sprite[:, :, 2] == 0)
        alpha = np.where(is_transparent, 0, 255).astype(np.uint8)
        sprite = np.concatenate([sprite, alpha[..., None]], axis=-1)
    mask = sprite[..., 3] > 128
    sprite[mask, :3] = new_rgb
    return sprite

def _load_and_recolor_group(filenames, new_rgb, transpose=False) -> list:
    base_dir = os.path.join(get_base_sprite_dir(), "asteroids")
    sprites = []
    for f in filenames:
        sprite = np.load(os.path.join(base_dir, f))
        if transpose:
            sprite = np.transpose(sprite, (1, 0, 2))
        sprites.append(jnp.array(_recolor_all(sprite, new_rgb)))
    return sprites

def _load_and_recolor_single(filename, new_rgb, transpose=False) -> jnp.ndarray:
    base_dir = os.path.join(get_base_sprite_dir(), "asteroids")
    sprite = np.load(os.path.join(base_dir, filename))
    if transpose:
        sprite = np.transpose(sprite, (1, 0, 2))
    return jnp.array(_recolor_all(sprite, new_rgb))

def _get_player_group_recolored():
    player_files = [f'player_pos{i}.npy' for i in range(16)] + [f'death_player{i}.npy' for i in range(3)]
    return _load_and_recolor_group(player_files, (150, 255, 150))

def _get_asteroid_group_recolored():
    asteroid_files = []
    for size in ['big1', 'big2', 'medium', 'small']:
        for color in ['brown', 'grey', 'lightblue', 'lightyellow', 'pink', 'purple', 'red', 'yellow']:
            asteroid_files.append(f'asteroid_{size}_{color}.npy')
    for size in ['big', 'medium', 'small']:
        for color in ['pink', 'yellow']:
            asteroid_files.append(f'death_{size}_{color}.npy')
    return _load_and_recolor_group(asteroid_files, (0, 200, 0))

def _get_digits_recolored() -> jnp.ndarray:
    sprites = _load_and_recolor_group([f'{i}.npy' for i in range(10)], (0, 255, 0))
    max_height = max(s.shape[0] for s in sprites)
    max_width = max(s.shape[1] for s in sprites)
    padded_digits = []
    for digit in sprites:
        digit = np.array(digit)
        pad_h = max_height - digit.shape[0]
        pad_w = max_width - digit.shape[1]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_digit = np.pad(
            digit,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        padded_digits.append(padded_digit)
    return jnp.stack([jnp.array(p) for p in padded_digits])


# --- "change_X_color" visual mods (parallels to Pong/Freeway/Breakout) -------
# Parametrized versions of the group/digit helpers above, so change_* mods can
# recolor to any target without touching MatrixMod's hardcoded green versions.
_PLAYER_FILES = [f'player_pos{i}.npy' for i in range(16)] + [f'death_player{i}.npy' for i in range(3)]
_ASTEROID_FILES = [
    f'asteroid_{size}_{color}.npy'
    for size in ['big1', 'big2', 'medium', 'small']
    for color in ['brown', 'grey', 'lightblue', 'lightyellow', 'pink', 'purple', 'red', 'yellow']
] + [
    f'death_{size}_{color}.npy'
    for size in ['big', 'medium', 'small']
    for color in ['pink', 'yellow']
]


def _get_digits_recolored_as(new_rgb: tuple) -> jnp.ndarray:
    sprites = _load_and_recolor_group([f'{i}.npy' for i in range(10)], new_rgb)
    max_height = max(s.shape[0] for s in sprites)
    max_width = max(s.shape[1] for s in sprites)
    padded_digits = []
    for digit in sprites:
        digit = np.array(digit)
        pad_h = max_height - digit.shape[0]
        pad_w = max_width - digit.shape[1]
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        padded_digits.append(np.pad(
            digit, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant", constant_values=0,
        ))
    return jnp.stack([jnp.array(p) for p in padded_digits])


def _get_background_recolored(new_rgb: tuple) -> jnp.ndarray:
    base_dir = os.path.join(get_base_sprite_dir(), "asteroids")
    bg = np.load(os.path.join(base_dir, "background.npy"))
    return jnp.array(_recolor_all(bg, new_rgb))


# New colors (tweak here). Each mod recolors only its own element; WALL_COLOR
# (the top/bottom UI bars) is left at its default so the change stays scoped.
_NEW_SHIP_COLOR = (100, 200, 255)      # light blue
_NEW_ASTEROID_COLOR = (255, 140, 0)    # orange (collapses the 8 built-in variants to one)
_NEW_BACKGROUND_COLOR = (20, 20, 45)   # dark navy
_NEW_SCORE_COLOR = (0, 220, 0)         # green


class ChangeShipColorMod(JaxAtariInternalModPlugin):
    """Changes the player ship color (default: light blue). Recolors every
    rotation frame and the death-animation frames."""
    asset_overrides = {
        'player_group': {
            'name': 'player_group', 'type': 'group',
            'data': _load_and_recolor_group(_PLAYER_FILES, _NEW_SHIP_COLOR),
        }
    }


class ChangeAsteroidColorMod(JaxAtariInternalModPlugin):
    """Changes all asteroids to a single color (default: orange), collapsing the
    base game's 8 built-in per-asteroid color variants into one uniform color."""
    asset_overrides = {
        'asteroid_group': {
            'name': 'asteroid_group', 'type': 'group',
            'data': _load_and_recolor_group(_ASTEROID_FILES, _NEW_ASTEROID_COLOR),
        }
    }


class ChangeBackgroundColorMod(JaxAtariInternalModPlugin):
    """Changes the play-field background color (default: dark navy). The
    top/bottom UI bars (WALL_COLOR) are left unchanged."""
    asset_overrides = {
        'background': {
            'name': 'background', 'type': 'background',
            'data': _get_background_recolored(_NEW_BACKGROUND_COLOR),
        }
    }


class ChangeScoreColorMod(JaxAtariInternalModPlugin):
    """Changes the score digit color (default: green)."""
    asset_overrides = {
        'digits': {
            'name': 'digits', 'type': 'digits',
            'data': _get_digits_recolored_as(_NEW_SCORE_COLOR),
        }
    }


class GrayscaleThemeMod(JaxAtariInternalModPlugin):
    """
    Full monochrome theme: recolors the ship, asteroids, missiles, and score to
    distinct shades of grey. The background and UI wall bars are already neutral
    black (0,0,0) by default, so they need no change. Shades are hand-picked
    (not a photometric conversion) so every element stays legibly distinct:
    asteroids (mid grey) < score (light-mid) < ship (light) < missiles
    (white, brightest -- easiest to track).
    """
    _SHIP_GREY = (230, 230, 230)
    _ASTEROID_GREY = (140, 140, 140)
    _MISSILE_GREY = (255, 255, 255)
    _SCORE_GREY = (190, 190, 190)

    asset_overrides = {
        'player_group': {
            'name': 'player_group', 'type': 'group',
            'data': _load_and_recolor_group(_PLAYER_FILES, _SHIP_GREY),
        },
        'asteroid_group': {
            'name': 'asteroid_group', 'type': 'group',
            'data': _load_and_recolor_group(_ASTEROID_FILES, _ASTEROID_GREY),
        },
        'missile1': {
            'name': 'missile1', 'type': 'single',
            'data': _load_and_recolor_single('missile1.npy', _MISSILE_GREY),
        },
        'missile2': {
            'name': 'missile2', 'type': 'single',
            'data': _load_and_recolor_single('missile2.npy', _MISSILE_GREY),
        },
        'digits': {
            'name': 'digits', 'type': 'digits',
            'data': _get_digits_recolored_as(_SCORE_GREY),
        },
    }


class NoFlickerMod(JaxAtariInternalModPlugin):
    """
    Marker mod: disables the base game's Atari-hardware sprite-flicker
    emulation, so the ship/missiles and the asteroids render every frame instead
    of alternating. This mod patches no methods and has no overrides -- `render`
    exists on both the env and the renderer, which the mod controller treats as
    ambiguous, so it cannot be patched by a plugin. The actual logic lives in
    AsteroidsEnvMod.render, which detects this marker and renders twice (once at
    each step_counter parity) to recover both groups, then merges them.
    """
    pass


# --- Dynamics mods -----------------------------------------------------------
_BASE_ACCEL_PER_ROTATION = jnp.array([
    (0, -64), (-25, -59), (-45, -45), (-59, -25), (-64, 0), (-59, 25), (-45, 45), (-25, 59),
    (0, 64), (25, 59), (45, 45), (59, 25), (64, 0), (59, -25), (45, -45), (25, -59),
])
_BASE_MAX_PLAYER_SPEED = 60 * 256 - 1


class FasterAsteroidsMod(JaxAtariInternalModPlugin):
    """
    Makes asteroids much faster by scaling ASTEROID_SPEED (default (2, 1): 2 px
    horizontal on periodic side-step frames, 1 px vertical every frame) by
    _SPEED_MULTIPLIER, giving (6, 3) -- asteroids rush down at 3 px/frame.

    Collision is a same-frame bounding-box check with no swept collision, but this
    stays safe: the smallest asteroid is 8 px tall / 4 px wide and the ship is
    10 px tall / 5 px wide, so even the 3 px vertical step (asteroid+ship spans
    18 px) and the 6 px side-step (spans 9 px) still overlap for multiple frames
    when passing -- nothing tunnels. Verified empirically: destroy rate stays
    healthy (missiles still connect) at this multiplier.
    """
    _SPEED_MULTIPLIER = 3

    constants_overrides = {
        "ASTEROID_SPEED": (2 * _SPEED_MULTIPLIER, 1 * _SPEED_MULTIPLIER),
    }


# --- Magnitude-scaled asteroid speed (asteroid_speed_x2 .. x5) ---------------
# Scales ASTEROID_SPEED (base (2, 1): 2 px horizontal on periodic side-step frames,
# 1 px vertical every frame) uniformly by N -> (2N, N). Vertical is the every-frame
# threat; N up to 5 stays tunnel-safe against the ship (10 px tall) and asteroids
# (>= 8 px tall), whose combined >= 18 px span dwarfs a 5 px step. The horizontal
# 2N px only applies on the infrequent side-step frames, so even 10 px at x5 needs
# both axes to overlap to hit -- verified empirically that destroy rate stays healthy
# and the ship keeps dying (no tunneling) across all levels.
class AsteroidSpeedX2Mod(JaxAtariInternalModPlugin):
    """Asteroid speed x2 -> ASTEROID_SPEED (4, 2)."""
    constants_overrides = {"ASTEROID_SPEED": (4, 2)}


class AsteroidSpeedX3Mod(JaxAtariInternalModPlugin):
    """Asteroid speed x3 -> ASTEROID_SPEED (6, 3) (same as faster_asteroids)."""
    constants_overrides = {"ASTEROID_SPEED": (6, 3)}


class AsteroidSpeedX4Mod(JaxAtariInternalModPlugin):
    """Asteroid speed x4 -> ASTEROID_SPEED (8, 4)."""
    constants_overrides = {"ASTEROID_SPEED": (8, 4)}


class AsteroidSpeedX5Mod(JaxAtariInternalModPlugin):
    """Asteroid speed x5 -> ASTEROID_SPEED (10, 5)."""
    constants_overrides = {"ASTEROID_SPEED": (10, 5)}


class SlowerAsteroidsMod(JaxAtariInternalModPlugin):
    """
    Makes asteroids slower. ASTEROID_SPEED's vertical component is already at
    its integer floor (1 px, applied every frame with no gating), so a
    constant-scaling approach can't go below default speed. Instead this patches
    asteroids_step (the whole per-frame movement update) to run only on every
    2nd frame, halving effective speed on both axes while keeping the same
    ASTEROID_SPEED per-update magnitude (and therefore the same collision-safety
    margins as the unmodded game).
    """
    @partial(jax.jit, static_argnums=(0,))
    def asteroids_step(self, asteroids_state: AsteroidsState):
        should_move = asteroids_state.step_counter % 2 == 0

        def _move(_):
            return JaxAsteroids.asteroids_step(self._env, asteroids_state)

        def _no_move(_):
            return asteroids_state.asteroid_states, asteroids_state.side_step_counter, asteroids_state.rng_key

        return jax.lax.cond(should_move, _move, _no_move, operand=None)


class FasterShipMod(JaxAtariInternalModPlugin):
    """
    Makes the ship faster: scales both thrust power (ACCEL_PER_ROTATION,
    applied every frame while thrusting) and the top speed cap
    (MAX_PLAYER_SPEED) by _SPEED_MULTIPLIER. Unlike a paddle with a slow
    acceleration ramp, thrust here is a constant per-frame acceleration, so
    scaling it is immediately noticeable (faster speed buildup, not just a
    higher ceiling that's rarely reached). At 3x the ship is very zippy and
    overshoots easily -- a large shift in control feel, not a subtle nudge.
    """
    _SPEED_MULTIPLIER = 3.0

    constants_overrides = {
        "ACCEL_PER_ROTATION": (_BASE_ACCEL_PER_ROTATION * _SPEED_MULTIPLIER).astype(jnp.int32),
        "MAX_PLAYER_SPEED": int(_BASE_MAX_PLAYER_SPEED * _SPEED_MULTIPLIER),
    }


class SlowerShipMod(JaxAtariInternalModPlugin):
    """
    Makes the ship slower: scales both thrust power (ACCEL_PER_ROTATION) and
    the top speed cap (MAX_PLAYER_SPEED) down by _SPEED_MULTIPLIER.
    """
    _SPEED_MULTIPLIER = 0.5

    constants_overrides = {
        "ACCEL_PER_ROTATION": (_BASE_ACCEL_PER_ROTATION * _SPEED_MULTIPLIER).astype(jnp.int32),
        "MAX_PLAYER_SPEED": int(_BASE_MAX_PLAYER_SPEED * _SPEED_MULTIPLIER),
    }


class RandomizeAsteroidSpawnMod(JaxAtariPostStepModPlugin):
    """
    Randomizes where the 4 starting asteroids appear and which of the 4
    directions they travel. The base game's initial asteroid layout
    (INITIAL_ASTEROID_STATES) is a fixed constant -- every episode starts with
    the exact same 4 asteroids at the exact same positions/headings. This mod
    redraws their (x, y) position and direction at every reset, keeping their
    size and color (and therefore difficulty) unchanged, so only the starting
    conditions vary. Implemented via after_reset (the wrapper hook made for
    modifying initial state) rather than patching reset() directly.
    """
    @partial(jax.jit, static_argnums=(0,))
    def after_reset(self, obs, state: AsteroidsState):
        c = self._env.consts
        key, kx, ky, kdir = jax.random.split(state.rng_key, 4)
        n = c.MAX_NUMBER_OF_ASTEROIDS

        was_active = state.asteroid_states[:, 3] != c.INACTIVE
        new_x = jax.random.randint(kx, (n,), c.MIN_ENTITY_X, c.MAX_ENTITY_X + 1)
        new_y = jax.random.randint(ky, (n,), c.MIN_ENTITY_Y, c.MAX_ENTITY_Y + 1)
        new_dir = jax.random.randint(kdir, (n,), 0, 4)

        new_asteroid_states = state.asteroid_states.at[:, 0].set(
            jnp.where(was_active, new_x, state.asteroid_states[:, 0])
        ).at[:, 1].set(
            jnp.where(was_active, new_y, state.asteroid_states[:, 1])
        ).at[:, 2].set(
            jnp.where(was_active, new_dir, state.asteroid_states[:, 2])
        )

        return obs, state.replace(asteroid_states=new_asteroid_states, rng_key=key)


# A dense 10-asteroid starting layout (base has 4). All large (size 1/2), spread
# around the periphery so none overlaps the player's spawn safe zone: the ship
# starts dead-centre at screen (80, 100) with no spawn invulnerability, so every
# asteroid is kept clear of the box x[60,100] x y[66,134]. Rows are [x, y, rot(0-3),
# size(1|2), color(0-7)]; the remaining 7 of the 17 slots stay INACTIVE, leaving
# room for split fragments.
_DENSE_INITIAL_ASTEROIDS = jnp.array([
    [ 20,  30, 0, 1, 0],
    [ 80,  25, 1, 2, 1],
    [140,  30, 2, 1, 2],
    [ 15, 100, 3, 2, 3],
    [145, 100, 0, 1, 4],
    [ 20, 170, 1, 2, 5],
    [ 80, 180, 2, 1, 6],
    [145, 175, 3, 2, 7],
    [ 40,  55, 0, 1, 0],
    [120, 145, 2, 2, 3],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
    [  0,   0, 0, 0, 0],
], dtype=jnp.int32)


class MoreAsteroidsMod(JaxAtariInternalModPlugin):
    """
    Floods the field with far more asteroids than the base game, for a much
    denser, harder-to-survive board -- a large change in the tactical situation.

    Two constant overrides: the initial board starts with 10 large asteroids
    instead of 4 (INITIAL_ASTEROID_STATES), and each cleared wave respawns 12
    instead of 6 (NEW_ASTEROIDS_COUNT). Both stay within MAX_NUMBER_OF_ASTEROIDS
    (17) -- which is left untouched because it defines the observation size and
    must stay constant across the CRL task sequence -- leaving a few slots for
    split fragments. (Replaces the old randomize_asteroid_spawn, which only
    shuffled the same 4 starting asteroids' positions; "a lot more" asteroids is a
    bigger, more distinct dynamics shift than reshuffling four of them.)
    """
    constants_overrides = {
        "INITIAL_ASTEROID_STATES": _DENSE_INITIAL_ASTEROIDS,
        "NEW_ASTEROIDS_COUNT": 12,
    }


class ShipInertiaMod(JaxAtariInternalModPlugin):
    """
    Removes the base game's built-in velocity dampening: the ship keeps its
    momentum indefinitely once thrust is released (frictionless Newtonian
    flight) instead of gradually decelerating back toward a stop. The ship must
    be actively counter-thrust to slow down or change direction, which is more
    physically realistic but noticeably harder to control precisely.
    """
    @partial(jax.jit, static_argnums=(0,))
    def decel_func(self, speed):
        return jnp.zeros_like(speed)


# --- Reward mods -------------------------------------------------------------
def _destroy_masks_by_size(consts, prev_asteroid_states, new_asteroid_states):
    """Per-size destroy-transition masks, matching the base game's own scoring
    detection (JaxAsteroids.get_transition_score)."""
    prev_sizes = prev_asteroid_states[:, 3]
    new_sizes = new_asteroid_states[:, 3]
    is_large_destroy = ((prev_sizes == consts.LARGE_1) | (prev_sizes == consts.LARGE_2)) & (new_sizes == consts.MEDIUM)
    is_medium_destroy = (prev_sizes == consts.MEDIUM) & (new_sizes == consts.SMALL)
    is_small_destroy = (prev_sizes == consts.SMALL) & (new_sizes == consts.INACTIVE)
    return is_large_destroy, is_medium_destroy, is_small_destroy


class LifeLossPenaltyMod(JaxAtariInternalModPlugin):
    """
    Penalizes losing a life, on top of the normal score-delta reward, to shift the
    optimum from pure destruction toward survival.

    The penalty is *sustained* over the first _SUSTAIN_FRAMES of the post-death
    respawn (respawn_timer counts down from RESPAWN_DELAY=136 on a hit), giving a
    -1 across several frame-skip windows per death instead of a single frame.
    Under the benchmark's sign-clipped training reward a one-frame -1 would be a
    lone negative window per death, nearly lost among the destroy rewards; a few
    windows makes losing a life genuinely costly. It is capped (not the full
    ~136-frame respawn) so a death costs a handful of windows, not ~34 -- which
    would swamp the destroy reward and collapse this into a pure-survival objective
    (that is what survival_reward is). Uses respawn_timer rather than a lives delta
    so it is unaffected by same-frame extra-life awards (POINTS_PER_LIFE); the
    upper bound `<= RESPAWN_DELAY` excludes the hyperspace timer (set above
    RESPAWN_DELAY), which is not a death.
    """
    _SUSTAIN_FRAMES = 8   # a few frame-skip windows per death: bigger than a lone -1, non-dominant

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        base = state.score - previous_state.score
        c = self._env.consts
        in_death_respawn = jnp.logical_and(
            state.respawn_timer > c.RESPAWN_DELAY - self._SUSTAIN_FRAMES,
            state.respawn_timer <= c.RESPAWN_DELAY,
        )
        return (base - in_death_respawn.astype(jnp.int32)).astype(jnp.int32)


class FlattenAsteroidValuesMod(JaxAtariInternalModPlugin):
    """
    Every asteroid destroyed pays +1, removing the base game's size-based
    scoring scheme (large=20, medium=50, small=100 points). Detected via the
    same size-transition logic the base game uses for real scoring.

    NOTE: under sign-clipped training reward this is a NO-OP -- the base game's
    20/50/100 all clip to +1 already, so the clipped stream is identical to base.
    Use only with reward clipping disabled. For a clip-surviving size mod see
    LargeAsteroidOnlyMod / SmallAsteroidOnlyMod, which zero the other sizes.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        is_l, is_m, is_s = _destroy_masks_by_size(self._env.consts, previous_state.asteroid_states, state.asteroid_states)
        total_destroys = jnp.sum(is_l.astype(jnp.int32)) + jnp.sum(is_m.astype(jnp.int32)) + jnp.sum(is_s.astype(jnp.int32))
        return total_destroys.astype(jnp.int32)


class LargeAsteroidOnlyMod(JaxAtariInternalModPlugin):
    """
    Rewards +1 only for destroying LARGE asteroids (the first, easiest hit that
    splits a big rock); medium and small destroys give 0. This is the opposite
    priority to SmallAsteroidOnlyMod, and -- unlike an all-positive size scheme
    (which sign-clipping collapses to the base "+1 per destroy") -- it zeroes the
    smaller sizes so the clipped signal genuinely differs from base: the agent is
    rewarded for cracking large rocks and ignoring the fragments, rather than
    finishing every asteroid off.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        is_l, _, _ = _destroy_masks_by_size(self._env.consts, previous_state.asteroid_states, state.asteroid_states)
        return jnp.sum(is_l.astype(jnp.int32)).astype(jnp.int32)


class SmallAsteroidOnlyMod(JaxAtariInternalModPlugin):
    """
    Rewards +1 only for destroying SMALL asteroids -- the hardest to hit, and
    the highest-value target in the base game's own scoring (100 pts). Large
    and medium destroys give 0.
    """
    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        _, _, is_s = _destroy_masks_by_size(self._env.consts, previous_state.asteroid_states, state.asteroid_states)
        return jnp.sum(is_s.astype(jnp.int32)).astype(jnp.int32)


class WaveClearBonusMod(JaxAtariInternalModPlugin):
    """
    Adds a +_BONUS reward when a wave (stage) is cleared, on top of the normal
    score-delta reward. Reads the base game's wave_count (incremented at the
    exact real stage-clear trigger inside step()). An earlier version tried to
    infer this from the active-asteroid count jumping to NEW_ASTEROIDS_COUNT,
    but that is NOT a reliable signature: a normal large-asteroid split spawns
    an extra 'ghost' asteroid and can coincidentally also raise the count to
    NEW_ASTEROIDS_COUNT, causing false positives (confirmed empirically -- it
    fired on ~1 in 2000 random steps with zero genuine wave clears).
    """
    _BONUS = 50

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        base = state.score - previous_state.score
        wave_cleared = state.wave_count > previous_state.wave_count
        return (base + wave_cleared.astype(jnp.int32) * self._BONUS).astype(jnp.int32)


class EveryKKillsMod(JaxAtariInternalModPlugin):
    """
    Rewards +1 on every _K-th asteroid kill and 0 for everything else (including
    life loss and wave clears), replacing the score-based reward entirely. Uses
    the base game's kill_count (a plain cumulative kill counter -- immune to the
    ambiguity of deriving kill count from size-weighted score deltas, since a
    single step's point delta can't always be uniquely decoded back into a kill
    count) and credits once per Kth-boundary crossed, which correctly handles
    steps with multiple simultaneous kills.
    """
    _K = 5

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        crossed = state.kill_count // self._K - previous_state.kill_count // self._K
        return crossed.astype(jnp.int32)


class SurvivalRewardMod(JaxAtariInternalModPlugin):
    """
    Rewards +_PER_STEP for every step taken, regardless of destroying asteroids,
    wave clears, or anything else -- replaces the score-based reward entirely.
    The episode ends exactly when lives <= 0 (_get_done), so a flat per-step
    reward already is "reward for time alive": the training loop stops calling
    step() once done, so no extra survival/lives check is needed here.
    """
    _PER_STEP = 1

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: AsteroidsState, state: AsteroidsState):
        return jnp.array(self._PER_STEP, dtype=jnp.int32)


class MatrixMod(JaxAtariInternalModPlugin):
    """A Matrix-themed mod for Asteroids: black background, green elements."""
    name = "matrix_theme"

    constants_overrides = {
        'WALL_COLOR': (0, 100, 0),
    }
    
    asset_overrides = {
        'player_group': {
            'name': 'player_group',
            'type': 'group',
            'data': _get_player_group_recolored()
        },
        'asteroid_group': {
            'name': 'asteroid_group',
            'type': 'group',
            'data': _get_asteroid_group_recolored()
        },
        'missile1': {
            'name': 'missile1',
            'type': 'single',
            'data': _load_and_recolor_single('missile1.npy', (0, 255, 0))
        },
        'missile2': {
            'name': 'missile2',
            'type': 'single',
            'data': _load_and_recolor_single('missile2.npy', (0, 255, 0))
        },
        'digits': {
            'name': 'digits',
            'type': 'digits',
            'data': _get_digits_recolored()
        },
        'minus_sign': {
            'name': 'minus_sign',
            'type': 'procedural',
            'data': jnp.zeros((10, 12, 4), dtype=jnp.uint8).at[4:6, 2:10, :].set(jnp.array([0, 255, 0, 255], dtype=jnp.uint8))
        },
        'wall_color': {
            'name': 'wall_color',
            'type': 'procedural',
            'data': jnp.array([0, 100, 0, 255], dtype=jnp.uint8).reshape(1, 1, 4)
        }
    }

class InstantTurnMod(JaxAtariInternalModPlugin):
    """Directly places the ship in the direction given by the action and applies thrust."""
    name = "instant_turn"

    attribute_overrides = {
        "ACTION_SET": jnp.array(
            [
                Action.NOOP,
                Action.FIRE,
                Action.UP,
                Action.RIGHT,
                Action.LEFT,
                Action.DOWN,
                Action.UPRIGHT,
                Action.UPLEFT,
                Action.DOWNRIGHT,
                Action.DOWNLEFT,
                Action.UPFIRE,
                Action.RIGHTFIRE,
                Action.LEFTFIRE,
                Action.DOWNFIRE,
                Action.UPRIGHTFIRE,
                Action.UPLEFTFIRE,
                Action.DOWNRIGHTFIRE,
                Action.DOWNLEFTFIRE,
            ],
            dtype=jnp.int32,
        )
    }

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        state_player_x,
        state_player_y,
        state_player_speed_x,
        state_player_speed_y,
        state_player_rotation,
        action,
        state_respawn_timer,
        rng_key
    ):
        # 1. Parse actions into logical directions
        left = jnp.logical_or(jnp.logical_or(action == Action.LEFT, action == Action.LEFTFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE)))
        right = jnp.logical_or(jnp.logical_or(action == Action.RIGHT, action == Action.RIGHTFIRE),
                               jnp.logical_or(jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE),
                                              jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))
        up = jnp.logical_or(jnp.logical_or(action == Action.UP, action == Action.UPFIRE),
                            jnp.logical_or(jnp.logical_or(action == Action.UPLEFT, action == Action.UPLEFTFIRE),
                                           jnp.logical_or(action == Action.UPRIGHT, action == Action.UPRIGHTFIRE)))
        down = jnp.logical_or(jnp.logical_or(action == Action.DOWN, action == Action.DOWNFIRE),
                              jnp.logical_or(jnp.logical_or(action == Action.DOWNLEFT, action == Action.DOWNLEFTFIRE),
                                             jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNRIGHTFIRE)))

        any_direction = jnp.logical_or(jnp.logical_or(up, down), jnp.logical_or(right, left))

        # 2. Determine new rotation (instant)
        # UP=0, UPLEFT=2, LEFT=4, DOWNLEFT=6, DOWN=8, DOWNRIGHT=10, RIGHT=12, UPRIGHT=14
        new_rotation = jax.lax.cond(
            up,
            lambda: jax.lax.cond(left, lambda: 2, lambda: jax.lax.cond(right, lambda: 14, lambda: 0)),
            lambda: jax.lax.cond(
                down,
                lambda: jax.lax.cond(left, lambda: 6, lambda: jax.lax.cond(right, lambda: 10, lambda: 8)),
                lambda: jax.lax.cond(
                    left, lambda: 4, lambda: jax.lax.cond(right, lambda: 12, lambda: state_player_rotation)
                )
            )
        )

        player_rotation = jax.lax.cond(
            any_direction,
            lambda: new_rotation,
            lambda: state_player_rotation
        )

        # 3. Apply physics based on the new rotation
        player_x = state_player_x
        player_y = state_player_y
        player_speed_x = state_player_speed_x
        player_speed_y = state_player_speed_y

        decel_x = self._env.decel_func(player_speed_x)
        decel_y = self._env.decel_func(player_speed_y)

        accel_x = self._env.consts.ACCEL_PER_ROTATION[player_rotation][0]
        accel_y = self._env.consts.ACCEL_PER_ROTATION[player_rotation][1]

        # In instant turn mod, pressing any direction triggers thrust
        is_thrusting = any_direction

        adj_speed_x = jnp.logical_and(
            jnp.logical_and(is_thrusting, jnp.abs(player_speed_x + accel_x) < self._env.consts.MAX_PLAYER_SPEED),
            jnp.logical_not(player_rotation%8 == 0))
        adj_speed_y = jnp.logical_and(
            jnp.logical_and(is_thrusting, jnp.abs(player_speed_y + accel_y) < self._env.consts.MAX_PLAYER_SPEED),
            jnp.logical_not((player_rotation-4)%8 == 0))

        # calculate new player speed
        player_speed_x = jax.lax.cond(
            adj_speed_x,
            lambda: player_speed_x + accel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) > jnp.abs(decel_x)),
            lambda: player_speed_x + decel_x,
            lambda: player_speed_x
        )
        player_speed_x = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_x), jnp.abs(player_speed_x) <= jnp.abs(decel_x)),
            lambda: 0,
            lambda: player_speed_x
        )

        player_speed_y = jax.lax.cond(
            adj_speed_y,
            lambda: player_speed_y + accel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) > jnp.abs(decel_y)),
            lambda: player_speed_y + decel_y,
            lambda: player_speed_y
        )
        player_speed_y = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(adj_speed_y), jnp.abs(player_speed_y) <= jnp.abs(decel_y)),
            lambda: 0,
            lambda: player_speed_y
        )

        displace_x = self._env.speed_func(player_speed_x)
        displace_y = self._env.speed_func(player_speed_y)

        player_x = jnp.int32(self._env.final_pos(self._env.consts.MIN_PLAYER_X, self._env.consts.MAX_PLAYER_X, player_x + displace_x))
        player_y = jnp.int32(self._env.final_pos(self._env.consts.MIN_PLAYER_Y, self._env.consts.MAX_PLAYER_Y, player_y + displace_y))

        # We remove hyperspace (down) entirely so you can fly down without teleporting
        
        return jax.lax.cond(
            state_respawn_timer <= 0,
            lambda: (player_x, player_y, player_speed_x, player_speed_y,
                     player_rotation, state_respawn_timer, rng_key),
            lambda: (state_player_x, state_player_y, state_player_speed_x, state_player_speed_y,
                     state_player_rotation, state_respawn_timer, rng_key)
        )
