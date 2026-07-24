import jax
import jax.numpy as jnp
import chex
from functools import partial
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestState, SpawnState


class DisableEnemiesMod(JaxAtariPostStepModPlugin):
    """Disable enemies in the environment."""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Zero out all enemy positions
        return new_state.replace(
            shark_positions=jnp.zeros_like(new_state.shark_positions),
            sub_positions=jnp.zeros_like(new_state.sub_positions),
            enemy_missile_positions=jnp.zeros_like(new_state.enemy_missile_positions),
            surface_sub_position=jnp.zeros_like(new_state.surface_sub_position)
        )


class NoDiversMod(JaxAtariInternalModPlugin):
    """
    Internal mod to remove Divers from the game.
    It suppresses the logic that updates/spawns divers and disables their rendering.
    """

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step_diver_movement(self,
            diver_positions: chex.Array,
            shark_positions: chex.Array,
            state_player_x: chex.Array,
            state_player_y: chex.Array,
            state_divers_collected: chex.Array,
            spawn_state: SpawnState,
            step_counter: chex.Array,
            rng: chex.PRNGKey
        ):
        """
        Override for _diver_step (or equivalent logic function).
        We return off-screen positions and inactive flags.
        """
        
        # We assume the diver step returns: 
        # (new_positions, new_actives, new_timers, score_addition)
        
        return (
            jnp.full_like(diver_positions, -1), 
            state_divers_collected,  
            spawn_state,
            rng
        )

    @partial(jax.jit, static_argnums=(0,))
    def _draw_divers(self, raster: jnp.ndarray, state: SeaquestState):
        """
        Override for the renderer to skip drawing divers.
        """
        # Simply return the raster without drawing the sprite
        return raster


class EnemyMinesMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

    asset_overrides = {
        "shark_base": {
            'name': 'shark_base',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        },
        "enemy_sub": {
            'name': 'enemy_sub',
            'type': 'group',
            'files': ['mods/mine.npy', 'mods/mine.npy']
        }
    }

    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([[128, 128, 128]] * 5),
    }


class FireBallsMod(JaxAtariInternalModPlugin):
    """
    Replaces both Sharks and Enemy Submarines with Mine sprites.
    
    This is a visual-only mod. Hitboxes and movement logic remain identical 
    to the original enemies. The 'Sharks' (now Mines) will not change color 
    based on difficulty level due to the game's rendering logic.
    """

class UnlimitedOxygenMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        return new_state.replace(oxygen=jnp.array(64, dtype=jnp.int32))

class GravityMod(JaxAtariPostStepModPlugin):
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        new_player_y = jnp.where(
            new_state.step_counter % 4 == 0,
            jnp.minimum(new_state.player_y + 1, self._env.consts.PLAYER_BOUNDS[1, 1]),
            new_state.player_y
        )
        return new_state.replace(player_y=new_player_y)

class RandomColorEnemiesMod(JaxAtariInternalModPlugin):
    pass




# ============================================================================ #
# Visual mods: single-element recolours + a grayscale theme
# (parallels the kangaroo/freeway/pong change_*_color mods). Seaquest's renderer
# is palette-based: most entities are single-colour baked sprites, EXCEPT sharks,
# which are recoloured at render time from the SHARK_DIFFICULTY_COLORS constant
# (so enemy recolouring is part constant / part sprite).
# ============================================================================ #
import os
import numpy as np
from jaxatari.rendering.jax_rendering_utils import (
    JaxRenderingUtils, RendererConfig, get_base_sprite_dir,
)

_jr = JaxRenderingUtils(RendererConfig())
_SPRITE_DIR = os.path.join(get_base_sprite_dir(), "seaquest")


def _load(fname):
    return _jr.loadFrame(os.path.join(_SPRITE_DIR, fname))


# Source colours baked into the base sprites (verified directly from the .npy).
_SUB_SRC       = (187, 187, 53)    # player sub + player torpedo (yellow)
_ENEMY_SUB_SRC = (170, 170, 170)   # enemy sub (grey)
_SCORE_SRC     = (210, 210, 64)    # score digits + life indicator (yellow)
# bg water blues, main -> lighter (kept: black, surface grey 142, seaweed green)
_WATER_SRCS    = [(0, 28, 136), (24, 59, 157), (45, 50, 184)]

# New colours (tweak here). Each recolour mod touches only its own element.
_NEW_SUB_COLOR   = (235, 120, 40)   # orange
_NEW_ENEMY_COLOR = (200, 60, 200)   # magenta
_NEW_WATER_COLOR = (18, 78, 58)     # dark teal
_NEW_SCORE_COLOR = (0, 220, 220)    # cyan

# Base SHARK_DIFFICULTY_COLORS (the 5 per-difficulty shark tints), needed to
# grayscale the sharks (which are coloured via this constant, not their sprite).
_SHARK_DIFFICULTY_COLORS = [(92, 186, 92), (213, 130, 74), (170, 92, 170), (213, 92, 130), (186, 92, 92)]

_PLAYER_SUB_FILES = ['player_sub/1.npy', 'player_sub/2.npy', 'player_sub/3.npy']
_ENEMY_SUB_FILES  = ['enemy_sub/1.npy', 'enemy_sub/2.npy', 'enemy_sub/3.npy']
_DIVER_FILES      = ['diver/1.npy', 'diver/2.npy']
_DIGIT_PATHS      = [os.path.join(_SPRITE_DIR, f'digits/{i}.npy') for i in range(10)]


def _recolor_group(files, src, tgt):
    rule = [{'source': src, 'target': tgt}]
    return [_jr.perform_recoloring(_load(f), rule) for f in files]


def _lighter(rgb, f=1.4):
    return tuple(min(255, int(round(c * f))) for c in rgb)


class ChangeSubColorMod(JaxAtariInternalModPlugin):
    """Recolours the player submarine (and its torpedo, which shares the colour). Default: orange."""
    asset_overrides = {
        "player_sub": {
            'name': 'player_sub', 'type': 'group',
            'data': _recolor_group(_PLAYER_SUB_FILES, _SUB_SRC, _NEW_SUB_COLOR),
        },
        "player_torp": {
            'name': 'player_torp', 'type': 'single',
            'data': _jr.perform_recoloring(_load('player_torp/1.npy'), [{'source': _SUB_SRC, 'target': _NEW_SUB_COLOR}]),
        },
    }


class ChangeEnemyColorMod(JaxAtariInternalModPlugin):
    """
    Recolours the enemies: sharks (via the SHARK_DIFFICULTY_COLORS constant, which
    the renderer tints them with -- flattened to one colour across all difficulties)
    and enemy subs (sprite recolour). Default: magenta.
    """
    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([list(_NEW_ENEMY_COLOR)] * 5),
    }
    asset_overrides = {
        "enemy_sub": {
            'name': 'enemy_sub', 'type': 'group',
            'data': _recolor_group(_ENEMY_SUB_FILES, _ENEMY_SUB_SRC, _NEW_ENEMY_COLOR),
        },
    }


class ChangeWaterColorMod(JaxAtariInternalModPlugin):
    """
    Recolours the water: the three blue tones of the background sprite are mapped
    to a new hue (main -> lighter), leaving the black, the surface line and the
    seaweed green untouched. Default: dark teal.
    """
    asset_overrides = {
        "background": {
            'name': 'background', 'type': 'background',
            'data': _jr.perform_recoloring(
                _load('bg/1.npy'),
                [
                    {'source': _WATER_SRCS[0], 'target': _NEW_WATER_COLOR},
                    {'source': _WATER_SRCS[1], 'target': _lighter(_NEW_WATER_COLOR, 1.3)},
                    {'source': _WATER_SRCS[2], 'target': _lighter(_NEW_WATER_COLOR, 1.6)},
                ],
            ),
        },
    }


class ChangeScoreColorMod(JaxAtariInternalModPlugin):
    """
    Recolours the yellow UI -- the score digits and the life indicator, which share
    the same baked colour (210,210,64). Default: cyan. (The oxygen bar is white and
    the diver indicator blue; those are separate elements and untouched.)
    """
    _rule = [{'source': _SCORE_SRC, 'target': _NEW_SCORE_COLOR}]
    asset_overrides = {
        "digits": {
            'name': 'digits', 'type': 'digits',
            'data': _jr.perform_recoloring(_jr._load_and_pad_digits_from_paths(_DIGIT_PATHS), _rule),
        },
        "life_indicator": {
            'name': 'life_indicator', 'type': 'single',
            'data': _jr.perform_recoloring(_load('life_indicator/1.npy'), _rule),
        },
    }


def _to_gray(arr):
    """Luminance-grayscale an RGBA sprite array (alpha preserved)."""
    a = np.array(arr, dtype=np.uint8)
    r, g, b = a[..., 0].astype(np.float32), a[..., 1].astype(np.float32), a[..., 2].astype(np.float32)
    lum = np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    a[..., 0] = a[..., 1] = a[..., 2] = lum
    return jnp.asarray(a)


def _gray_group(files):
    return [_to_gray(_load(f)) for f in files]


def _gray_rgb(rgb):
    l = int(round(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]))
    return [l, l, l]


class GrayscaleThemeMod(JaxAtariInternalModPlugin):
    """
    Desaturates the whole scene to luminance grayscale. Sprites are grayscaled via
    asset_overrides; the render-time colour constants (SHARK_DIFFICULTY_COLORS for
    the sharks, and the two oxygen-bar colours, which are looked up in the palette)
    are grayscaled via constants_overrides so those lookups still resolve.
    """
    constants_overrides = {
        "SHARK_DIFFICULTY_COLORS": jnp.array([_gray_rgb(c) for c in _SHARK_DIFFICULTY_COLORS]),
        "OXYGEN_BAR_COLOR": jnp.array(_gray_rgb((214, 214, 214)) + [255]),
        "OXYGEN_BAR_BG_COLOR": jnp.array(_gray_rgb((163, 57, 21)) + [255]),
    }
    asset_overrides = {
        "background":      {'name': 'background', 'type': 'background', 'data': _to_gray(_load('bg/1.npy'))},
        "player_sub":      {'name': 'player_sub', 'type': 'group', 'data': _gray_group(_PLAYER_SUB_FILES)},
        "enemy_sub":       {'name': 'enemy_sub', 'type': 'group', 'data': _gray_group(_ENEMY_SUB_FILES)},
        "diver":           {'name': 'diver', 'type': 'group', 'data': _gray_group(_DIVER_FILES)},
        "player_torp":     {'name': 'player_torp', 'type': 'single', 'data': _to_gray(_load('player_torp/1.npy'))},
        "enemy_torp":      {'name': 'enemy_torp', 'type': 'single', 'data': _to_gray(_load('enemy_torp/1.npy'))},
        "digits":          {'name': 'digits', 'type': 'digits', 'data': _to_gray(_jr._load_and_pad_digits_from_paths(_DIGIT_PATHS))},
        "life_indicator":  {'name': 'life_indicator', 'type': 'single', 'data': _to_gray(_load('life_indicator/1.npy'))},
        "diver_indicator": {'name': 'diver_indicator', 'type': 'single', 'data': _to_gray(_load('diver_indicator/1.npy'))},
    }


# ============================================================================ #
# Dynamic mods: enemy/sub speed, oxygen drain, spawn behaviour.
# The speed/oxygen mods are post-step (there is no single speed/drain constant to
# override); the spawn mods patch small, self-contained env methods.
# ============================================================================ #

# --- Enemy speed -------------------------------------------------------------
class FasterEnemiesMod(JaxAtariPostStepModPlugin):
    """
    Makes sharks and enemy subs move faster horizontally (default 2x). Enemy speed
    has no constant (it's derived from difficulty in calculate_movement_speed), so
    this post-step mod amplifies the *pixel step* the base game took this frame --
    reading each enemy's dx and adding (_MULT-1) more px in its travel direction --
    then re-applies the base game's off-screen despawn (x <= -8 or x >= 168).

    Only genuine per-frame moves are amplified (both frames active, |dx| <= 4), so
    spawns/despawns (large position jumps) are left alone. Tunnel note: the base
    checks player-torpedo/enemy collisions before this runs, so at 2x the enemy
    lands 2 px further -- still well within the ~8 px sprite overlap, so kills
    still register.
    """
    _MULT = 2

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        def amp(p, n):
            do = (p[:, 2] != 0) & (n[:, 2] != 0) & (jnp.abs(n[:, 0] - p[:, 0]) <= 4)
            x = jnp.where(do, p[:, 0] + (n[:, 0] - p[:, 0]) * self._MULT, n[:, 0])
            oob = (x <= -8) | (x >= 168)
            res = n.at[:, 0].set(jnp.where(oob, 0, x).astype(n.dtype))
            return jnp.where(oob[:, None], jnp.zeros_like(res), res)

        return new_state.replace(
            shark_positions=amp(prev_state.shark_positions, new_state.shark_positions),
            sub_positions=amp(prev_state.sub_positions, new_state.sub_positions),
        )


class SlowerEnemiesMod(JaxAtariPostStepModPlugin):
    """
    Halves enemy horizontal speed: on every other frame the enemies' x is held at
    its previous value (so they advance only half as often). y (the shark bob) and
    spawn/despawn bookkeeping are left to the base game, so only the horizontal
    advance is slowed.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        skip = (new_state.step_counter % 2) == 0

        def slow(p, n):
            return n.at[:, 0].set(jnp.where(skip, p[:, 0], n[:, 0]))

        return new_state.replace(
            shark_positions=slow(prev_state.shark_positions, new_state.shark_positions),
            sub_positions=slow(prev_state.sub_positions, new_state.sub_positions),
        )


# --- Player-sub speed --------------------------------------------------------
class FasterSubMod(JaxAtariPostStepModPlugin):
    """
    Makes the player submarine move faster (default 2x). The base game moves the
    sub +/-1 px/frame (hardcoded), so this post-step mod amplifies the voluntary
    move this frame by (_MULT-1) extra px, clamped to PLAYER_BOUNDS. Blocked frames
    (surfacing/refuel, where the base holds the sub still) have dx=0, so they're
    unaffected.
    """
    _MULT = 2

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        b = self._env.consts.PLAYER_BOUNDS
        nx = jnp.clip(prev_state.player_x + (new_state.player_x - prev_state.player_x) * self._MULT, b[0, 0], b[0, 1])
        ny = jnp.clip(prev_state.player_y + (new_state.player_y - prev_state.player_y) * self._MULT, b[1, 0], b[1, 1])
        return new_state.replace(
            player_x=nx.astype(new_state.player_x.dtype),
            player_y=ny.astype(new_state.player_y.dtype),
        )


class SlowerSubMod(JaxAtariPostStepModPlugin):
    """
    Halves the player submarine's speed: on every other frame its position is held
    at the previous value, so it advances only half as often.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        skip = (new_state.step_counter % 2) == 0
        return new_state.replace(
            player_x=jnp.where(skip, prev_state.player_x, new_state.player_x),
            player_y=jnp.where(skip, prev_state.player_y, new_state.player_y),
        )


# --- Oxygen ------------------------------------------------------------------
class FasterOxygenDrainMod(JaxAtariPostStepModPlugin):
    """
    Makes oxygen deplete faster (default 2x), so the player must surface more often.
    The base game drains 1 unit every 32 frames underwater; on each of those drain
    frames this mod removes an extra _EXTRA units (total drop 1 + _EXTRA), so a full
    64-unit tank empties in ~1/(1+_EXTRA) of the time. Only frames where oxygen
    actually dropped are touched, so the surface refill is unaffected.

    This is the natural base for a magnitude ladder (oxygen_drain_xN); see the
    seaquest mag4 discussion.
    """
    _EXTRA = 1

    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: SeaquestState, new_state: SeaquestState) -> SeaquestState:
        dropped = (new_state.oxygen < prev_state.oxygen) & (new_state.oxygen > 0)
        new_ox = jnp.maximum(new_state.oxygen - jnp.where(dropped, self._EXTRA, 0), 0)
        return new_state.replace(oxygen=new_ox.astype(new_state.oxygen.dtype))


# --- Spawns ------------------------------------------------------------------
class DenseSpawnMod(JaxAtariInternalModPlugin):
    """
    Spawns the maximum-density enemy formation every wave. The base game picks a
    formation from difficulty (1, 2, 2, or 3 enemies per lane); this patches
    get_pattern_for_difficulty to always return the full three-in-a-row pattern, so
    every lane spawns 3 enemies -> a much denser field.
    """
    @partial(jax.jit, static_argnums=(0,))
    def get_pattern_for_difficulty(self, current_pattern, moving_left):
        return jnp.array([1, 1, 1])


class RandomizeSpawnMod(JaxAtariInternalModPlugin):
    """
    Randomizes the enemy formation per wave instead of tying it to the difficulty
    level. Patches get_pattern_for_difficulty to pick one of the four formations
    via a PRNG keyed on the wave's pattern index and (randomly chosen) direction,
    so which formation appears no longer follows the difficulty ramp predictably.

    NOTE: entropy is limited (the method only receives the pattern index + the
    random moving_left flag), so formations vary per wave but aren't fully i.i.d.;
    a per-frame-random version would need overriding the whole spawn scan.
    """
    _PATTERNS = jnp.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    @partial(jax.jit, static_argnums=(0,))
    def get_pattern_for_difficulty(self, current_pattern, moving_left):
        seed = current_pattern.astype(jnp.uint32) * jnp.uint32(2) + moving_left.astype(jnp.uint32)
        idx = jax.random.randint(jax.random.PRNGKey(seed), (), 0, 4)
        return self._PATTERNS[idx]


class DiverSpawnRateMod(JaxAtariInternalModPlugin):
    """
    Increases how many divers appear: the base game only opens diver spawning in 2
    of the 4 lanes (diver_array starts [1,1,0,0]); this marks all four lanes as
    eligible before delegating to the base spawn_divers, so divers can spawn in
    every lane -> more rescues available (and more surfacing traffic to dodge).
    """
    @partial(jax.jit, static_argnums=(0,))
    def spawn_divers(self, spawn_state, diver_positions, shark_positions, sub_positions, step_counter):
        boosted = spawn_state.replace(
            diver_array=jnp.where(spawn_state.diver_array == 0, 1, spawn_state.diver_array)
        )
        return JaxSeaquest.spawn_divers(self._env, boosted, diver_positions, shark_positions, sub_positions, step_counter)
